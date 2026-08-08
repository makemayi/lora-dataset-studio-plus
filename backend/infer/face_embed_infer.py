"""Bank face pass — InsightFace antelopev2 embeddings + person clustering, run
in the DEDICATED ML interpreter (insightface/numpy are not in the Flask venv).
Device is chosen by the parent: 'cpu' (default — never touches the GPU/ComfyUI)
or 'cuda' (only when onnxruntime-gpu is present; the parent then runs this pass
inside its GPU-exclusive window). CUDA requested but unavailable → CPU fallback.

Protocol (same family as face_score_infer.py):
  stdin  : {"images": [abs paths], "models_root": path|null,
            "cache": abs path to a .npz|null, "threshold": 0.45,
            "device": "cpu"|"cuda", "require_yaw": bool,
            "groups": [{"name": str, "images": [abs paths]}]  # OPTIONAL}
  stdout : ONE JSON line {"ok": bool,
            "results": {path: {state, det, bbox_frac, yaw|null}},
            "clusters": {path: int}, "used_gpu": bool, "error"?: str,
            "group_clusters": {name: {path: int}}  # only when groups was given}
  stderr : "[embed] i/N <state>" progress lines (the parent streams these to
            drive the UI progress bar).

Embeddings are CACHED in the .npz (parallel arrays paths/embs/states/dets/
bfracs/yaws/sigs/hashes) and written incrementally every CACHE_EVERY images — killing the
pass mid-way loses at most that slice, and re-clustering at another threshold is
then near-instant.

``yaws`` is the newest array and every cache written before it exists lacks it.
Those entries load with yaw = NaN and are reported as ``yaw: null`` — "not
measured", never 0.0, which would read as a perfectly frontal face. ``require_yaw``
asks for those entries to be RE-DETECTED (the angle is not recoverable from the
stored embedding); it is what the app's opt-in ⤢ backfill sets, and it is off by
default so an ordinary resume never turns into hours of re-detection.

Clustering = union-find over cosine ≥ threshold on the
L2-normed embeddings of the SCORABLE faces (biggest face per image — a group
photo clusters by its dominant face); cluster ids are 1-based, ordered by
cluster size descending, singletons included (a person seen once is still a
cluster of one)."""
from __future__ import annotations
import hashlib
import json
import os
import sys

CACHE_EVERY = 50
DET_MIN, BBOX_MIN, YAW_MAX = 0.50, 0.06, 40.0   # same gates as face_score_infer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank_image_guard import read_validated_bank_image  # noqa: E402
from face_score_infer import _repair_nested_antelopev2  # noqa: E402
import npz_atomic  # noqa: E402


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _cancel_requested(cancel_file):
    """The parent drops this sentinel file to ask for a clean stop, so the pass
    flushes its cache and exits between images instead of being SIGKILLed
    mid-compute (which would lose up to CACHE_EVERY images)."""
    return bool(cancel_file) and os.path.exists(cancel_file)


def _write_count(cache_path, n):
    """Plain-text sidecar (``<cache>.count``) with how many images are cached so
    far. The Flask parent has no numpy to read the .npz, so this is how a stopped
    pass can still report an honest "N cached (M remaining)" — even in the rare
    case it had to be hard-killed before it could print its own cancel line."""
    if not cache_path:
        return
    try:
        with open(cache_path + '.count', 'w', encoding='utf-8') as f:
            f.write(str(int(n)))
    except OSError:
        pass


def _file_sig(path):
    """Cheap runtime invalidation signature, aligned with the Score cache."""
    try:
        st = os.stat(path)
        return f'{st.st_size}:{st.st_mtime_ns}'
    except OSError:
        return ''


def _file_hash(path):
    try:
        before = _file_sig(path)
        if not before:
            return b''
        digest = hashlib.sha256()
        with open(path, 'rb') as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.digest() if _file_sig(path) == before else b''
    except OSError:
        return b''


def _load_cache(path):
    """{path: (state, det, bbox_frac, emb, yaw, sig)}. A cache written before the yaw
    array existed loads with yaw = NaN — additive, so no user ever has to
    re-embed a bank just because a new number was added to the tuple."""
    import numpy as np
    out = {}
    if not path or not os.path.isfile(path):
        return out
    try:
        with np.load(path, allow_pickle=False) as z:
            paths, states = z['paths'], z['states']
            embs, dets, bfracs = z['embs'], z['dets'], z['bfracs']
            yaws = z['yaws'] if 'yaws' in z.files else None
            sigs = z['sigs'] if 'sigs' in z.files else None
            hashes = z['hashes'] if 'hashes' in z.files else None
            if (hashes is not None
                    and (hashes.shape != (len(paths), 32)
                         or hashes.dtype != np.dtype('uint8'))):
                raise ValueError('invalid cache hash shape')
        for i, p in enumerate(paths):
            digest = hashes[i].tobytes() \
                if hashes is not None else b''
            if digest == b'\0' * 32:
                digest = b''
            out[str(p)] = (states[i], float(dets[i]), float(bfracs[i]), embs[i],
                           float(yaws[i]) if yaws is not None else float('nan'),
                           str(sigs[i]) if sigs is not None else '', digest)
    except Exception as e:  # noqa: BLE001 — a corrupt cache = recompute, never fatal
        _log(f'[embed] cache unreadable, recomputing: {e}')
        return {}
    return out


def _save_cache(path, cache):
    import numpy as np
    if not path:
        return
    if not cache:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    paths = list(cache)
    npz_atomic.save_npz_atomic(path, dict(
        paths=np.array(paths),
        states=np.array([cache[p][0] for p in paths]),
        dets=np.array([cache[p][1] for p in paths], dtype='float32'),
        bfracs=np.array([cache[p][2] for p in paths], dtype='float32'),
        yaws=np.array([cache[p][4] for p in paths], dtype='float32'),
        embs=np.stack([cache[p][3] for p in paths]).astype('float32'),
        sigs=np.array([_cache_sig(cache[p]) for p in paths]),
        hashes=np.frombuffer(b''.join(
            _cache_hash(cache[p]) or (b'\0' * 32) for p in paths),
            dtype='uint8').reshape(len(paths), 32)))


def _flush_cache(path, cache):
    """Save, and never let a HELD cache file kill the pass — the twin of the same
    helper in bank_score_infer. The finished archive stays on disk under its
    temporary name, and the next run salvages it."""
    try:
        _save_cache(path, cache)
    except npz_atomic.NpzReplaceLocked as error:
        _log(f'[embed] {error}')


def _salvage_cache(path):
    """Promote a temporary left by an interrupted run, if it reads back clean.

    ``_load_cache`` is the validator on purpose: it applies exactly the parsing a
    real read applies and touches every array, so a file truncated by a kill
    mid-write fails it instead of being promoted."""
    if not path:
        return
    npz_atomic.salvage_orphan_tmp(path, lambda p: len(_load_cache(p)), _log)


def _cache_sig(entry):
    return entry[5] if len(entry) > 5 else ''


def _cache_hash(entry):
    value = entry[6] if len(entry) > 6 else b''
    return value if isinstance(value, bytes) and len(value) == 32 else b''


def _is_stale(path, entry):
    stored = _cache_sig(entry)
    digest = _cache_hash(entry)
    if not stored or not digest:
        return True
    current = _file_sig(path)
    return (not current or current != stored
            or _file_hash(path) != digest)


def _cluster(order, cache, threshold):
    """{path: 1-based cluster id} for the scorable faces of ``order``."""
    import numpy as np
    scorable = [p for p in order
                if p in cache and cache[p][0] == 'scorable' and cache[p][3] is not None]
    if not scorable:
        return {}
    E = np.stack([cache[p][3] for p in scorable]).astype('float32')
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)   # normed already; belt & braces
    n = len(scorable)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    chunk = 512
    for i0 in range(0, n, chunk):
        sims = E[i0:i0 + chunk] @ E.T
        for a, b in np.argwhere(sims >= threshold):
            a += i0
            if a < b:
                union(int(a), int(b))
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    ordered = sorted(groups.values(), key=lambda m: (-len(m), m[0]))
    out = {}
    for cid, members in enumerate(ordered, start=1):
        for i in members:
            out[scorable[i]] = cid
    return out


def main() -> int:
    raw = sys.stdin.read()
    try:
        req = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        print(json.dumps({'ok': False, 'results': {}, 'clusters': {},
                          'error': f'bad json: {e}'}))
        return 1
    images = [str(p) for p in (req.get('images') or [])]
    if not images:
        print(json.dumps({'ok': False, 'results': {}, 'clusters': {},
                          'error': 'no images'}))
        return 1
    models_root = req.get('models_root') or None
    cache_path = req.get('cache') or None
    cancel_file = req.get('cancel_file') or None
    threshold = float(req.get('threshold') or 0.45)
    device = str(req.get('device') or 'cpu').lower()   # 'cpu' | 'cuda'
    # Opt-in only: re-detect the entries a pre-yaw build cached, because the
    # angle cannot be recovered from a stored embedding. See the module docstring.
    require_yaw = bool(req.get('require_yaw'))

    used_gpu = False   # set when the model actually loads on CUDA below
    # Claim any embedding work a previous run finished but could not rename into
    # place (a held destination) or never got to rename at all (a hard kill).
    _salvage_cache(cache_path)
    cache = _load_cache(cache_path)

    def _needs_work(p):
        if p not in cache:
            return True
        if _is_stale(p, cache[p]):
            return True
        return require_yaw and not (cache[p][4] == cache[p][4])   # NaN test

    todo = [p for p in images if _needs_work(p)]
    _write_count(cache_path, len(images) - len(todo))
    _log(f'[embed] {len(images)} image(s), {len(images) - len(todo)} cached')

    if todo:
        import cv2
        import numpy as np  # noqa: F401 — insightface needs it importable
        from insightface.app import FaceAnalysis
        _repair_nested_antelopev2(models_root)
        # Provider selection is EXPLICIT per requested device — a bare
        # ['CUDAExecutionProvider', ...] would silently grab the GPU the moment
        # onnxruntime-gpu is present, outside the parent's GPU-exclusive window.
        # cpu → CPU only (ctx_id=-1); cuda → try CUDA, fall back to CPU (logged).
        import onnxruntime as ort
        avail = ort.get_available_providers()
        used_gpu = device == 'cuda' and 'CUDAExecutionProvider' in avail
        if device == 'cuda' and not used_gpu:
            _log('[embed] CUDA requested but CUDAExecutionProvider unavailable '
                 '(install onnxruntime-gpu) — falling back to CPU')
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if used_gpu else ['CPUExecutionProvider'])
        _log(f'[embed] providers={avail} device={device} used_gpu={used_gpu}')
        try:
            kwargs = {'name': 'antelopev2', 'providers': providers}
            if models_root:
                kwargs['root'] = models_root
            app = FaceAnalysis(**kwargs)
            app.prepare(ctx_id=0 if used_gpu else -1, det_size=(640, 640))
        except Exception as e:  # noqa: BLE001 — must exit as clean JSON, not a mute traceback
            print(json.dumps({'ok': False, 'results': {}, 'clusters': {},
                              'error': f'model load failed: {type(e).__name__}: {e}'}))
            return 1

        def biggest(faces):
            return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) \
                if faces else None

        import numpy as _np
        zero = _np.zeros(512, dtype='float32')
        done_since_save = 0
        uncached_changed = 0
        if _cancel_requested(cancel_file):   # cancelled during the model load
            cached = len(images) - len(todo)
            _write_count(cache_path, cached)
            print(json.dumps({'ok': True, 'cancelled': True,
                              'cached': cached, 'remaining': len(todo)}))
            return 0
        for i, p in enumerate(todo, 1):
            changed_while_reading = False
            signature = ''
            payload_hash = b''
            result = None
            try:
                # Read a bounded, validated snapshot rather than opening the
                # live Bank path with cv2.  The parent can only preflight a
                # path; it cannot stop that path being replaced before this
                # dedicated interpreter starts.
                signature = _file_sig(p)
                payload = read_validated_bank_image(p)
                payload_hash = hashlib.sha256(payload).digest()
                if not signature or _file_sig(p) != signature:
                    changed_while_reading = True
                    raise RuntimeError('image changed while it was read')
                img = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    result = ('unreadable', 0.0, 0.0, zero, float('nan'),
                              signature, payload_hash)
                else:
                    h, w = img.shape[:2]
                    f = biggest(app.get(img))
                    if f is None:   # padding rescue: SCRFD misses full-frame closeups
                        pad = int(0.25 * max(h, w))
                        f = biggest(app.get(cv2.copyMakeBorder(
                            img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))))
                        scale = (w + 2 * pad) * (h + 2 * pad) / (w * h) if f is not None else 1.0
                    else:
                        scale = 1.0
                    if f is None:
                        result = ('no_face', 0.0, 0.0, zero, float('nan'),
                                  signature, payload_hash)
                    else:
                        area = (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                        bbox_frac = float(area / (w * h) / scale)
                        det = float(f.det_score)
                        # The pose estimate: kept BOTH as the 'extreme_pose'
                        # gate (its original and only use) and, now, as a stored
                        # measurement — it is what the app's head-angle facet is
                        # made of. No pose at all stays NaN rather than becoming
                        # 0.0, which would claim a perfectly frontal face.
                        pose = getattr(f, 'pose', None)
                        yaw = float(pose[1]) if pose is not None else float('nan')
                        gate = 0.0 if yaw != yaw else yaw
                        state = 'scorable'
                        if det < DET_MIN:
                            state = 'low_det'
                        elif bbox_frac < BBOX_MIN:
                            state = 'too_small'
                        elif abs(gate) > YAW_MAX:
                            state = 'extreme_pose'
                        result = (state, round(det, 3), round(bbox_frac, 4),
                                  f.normed_embedding.astype('float32'),
                                  yaw if yaw != yaw else round(yaw, 2), signature,
                                  payload_hash)
                # Inference may be much slower than the validated read.  Commit
                # its local result only while the live path still identifies
                # the bytes captured above.
                if not signature or _file_sig(p) != signature:
                    changed_while_reading = True
                    raise RuntimeError('image changed while it was analysed')
                cache[p] = result
            except Exception as e:  # noqa: BLE001 — one broken file must not sink the pass
                # The validated read can itself fail because a live Bank file
                # was replaced or removed.  Compare the identity captured
                # before the read even on that exception path; otherwise an
                # ERROR could be signed for replacement bytes no model saw.
                if not signature or _file_sig(p) != signature:
                    changed_while_reading = True
                if changed_while_reading:
                    # Never bless the replacement with an ERROR computed from
                    # different bytes. Missing entry means the next run retries.
                    cache.pop(p, None)
                    uncached_changed += 1
                else:
                    cache[p] = ('error', 0.0, 0.0, zero, float('nan'), signature,
                                payload_hash)
                _log(f'[embed] {i}/{len(todo)} ERROR {e}')
                continue
            finally:
                done_since_save += 1
                if cache_path and done_since_save >= CACHE_EVERY:
                    _flush_cache(cache_path, cache)
                    _write_count(
                        cache_path,
                        len(images) - len(todo) + i - uncached_changed)
                    done_since_save = 0
            _log(f'[embed] {i}/{len(todo)} '
                 f'{cache[p][0] if p in cache else "changed; retry next run"}')
            if _cancel_requested(cancel_file):   # clean stop between images
                if cache_path:
                    _flush_cache(cache_path, cache)
                cached = sum(path in cache for path in images)
                _write_count(cache_path, cached)
                print(json.dumps({'ok': True, 'cancelled': True,
                                  'cached': cached, 'remaining': len(todo) - i}))
                return 0
        if cache_path:
            _flush_cache(cache_path, cache)
            _write_count(cache_path, sum(path in cache for path in images))

    results = {}
    for p in images:
        entry = cache.get(p) or ('error', 0.0, 0.0, None, float('nan'), '', b'')
        state, det, bfrac, _emb, yaw = entry[:5]
        digest = _cache_hash(entry)
        results[p] = {'state': str(state), 'det': float(det), 'bbox_frac': float(bfrac),
                      'fingerprint': digest.hex() if digest else None,
                      # null, never 0.0 — "not measured" is its own answer.
                      'yaw': None if yaw != yaw else float(yaw)}
    clusters = _cluster(images, cache, threshold)
    out = {'ok': True, 'results': results, 'clusters': clusters,
           'used_gpu': used_gpu}
    # Optional per-GROUP clustering, for the caller asking "is EACH of these
    # folders one person?" — a question the flat clustering above cannot answer:
    # it would happily merge two folders into a single cluster and let that read
    # as "consistent". Same threshold, same union-find, run once per group, and
    # riding on the SAME child call so a scan over forty folders costs one
    # subprocess and one model load — none at all when every image is cached.
    groups = req.get('groups') or []
    if groups:
        out['group_clusters'] = {
            str(g.get('name')): _cluster(
                [str(p) for p in (g.get('images') or [])], cache, threshold)
            for g in groups}
    print(json.dumps(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
