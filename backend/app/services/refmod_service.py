"""MiniMax-H3 RefMod generation for a dataset (the 🧩 button).

Turns a dataset's kept images into one `minimaxh3_<name>_v1_refmod.safetensors`
reference file for the ComfyUI-MiniMaxH3Mod node pack. The heavy lifting runs in
the COMFYUI interpreter (it owns torch) through refmod_extract_worker.py — this
module only resolves paths, picks the images, and drives the subprocess.

Every path derives from the already-configured ComfyUI base_dir, so the feature
has no settings block of its own: python from the portable bundle layout,
the node pack and the VAE from the ComfyUI tree. A missing piece is a ValueError
with the remedy in the text — the toast shows it verbatim.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import urllib.request

logger = logging.getLogger(__name__)
from pathlib import Path

from .. import config as cfg
from ..models import FaceDatasetImage
from ..services import face_mask as face_mask_service
from ..services.dataset_storage import dataset_path

# Per-framing picks, identity-first: the face signal lives in close-ups and
# half/full frames are what carry CLOTHING into the latent (the #1 complaint).
# Ladder face → half → full tops up only when a dataset lacks close-ups.
_MAX_FACE, _MAX_HALF, _MAX_FULL = 12, 4, 4
_TOKEN_BUDGET = 20480   # 20 frames x <=1024 tokens at a 1024px-long-side encode
_BACKGROUND_RETENTION = 0.0   # outside the face mask collapses to a blurred copy
_MASKS_DIR_NAME = 'refmod'
_SUBPROCESS_TIMEOUT_S = 1800
_CROP_KIND = 'refmod_face_crop'   # dataset rows created by the face harvester
_UPSCALE_BELOW = 1024   # the encode pool caps at a 1024px short edge; softer crops go to Topaz
_CROP_MIN_DET = 0.65     # InsightFace det_score floor, on source AND on the upscaled crop
_CROP_MIN_FACE_PX = 240  # source face box max-dim; below this even Topaz is a 4x+ stretch
_MAX_CROP_SIDE = 2048    # Topaz overshoots (6x → 7000px); the encode caps at 1024 anyway
_CROP_MIN_SIDE = 96      # sanity floor in source px; below this even Topaz is hopeless


def _comfy_root() -> Path:
    raw = (cfg.get('comfyui.base_dir') or '').strip()
    if not raw:
        raise ValueError('ComfyUI base_dir is not configured — set it in Settings ▸ ComfyUI.')
    root = Path(raw)
    if not root.is_dir():
        raise ValueError(f'ComfyUI base_dir does not exist: {root}')
    return root


def _python_exe(root: Path) -> Path:
    """The interpreter that owns torch: the portable bundle's python first
    (aki builds ship `<bundle>/python`, stock portables `python_embeded`),
    then a venv checkout."""
    candidates = [root.parent / 'python' / 'python.exe',
                  root.parent / 'python_embeded' / 'python.exe',
                  root / 'venv' / 'Scripts' / 'python.exe']
    for cand in candidates:
        if cand.is_file():
            return cand
    raise ValueError('No ComfyUI python found (looked for python/python.exe, '
                     'python_embeded/python.exe, venv/Scripts/python.exe next to '
                     'the ComfyUI base dir).')


def _node_dir(root: Path) -> Path:
    node = root / 'custom_nodes' / 'ComfyUI-MiniMaxH3Mod'
    if not node.is_dir():
        raise ValueError('ComfyUI-MiniMaxH3Mod is not installed — '
                         'clone it into custom_nodes/ and restart ComfyUI.')
    return node


def _vae_path(root: Path) -> Path:
    """The H3 VIDEO vae. Named loosely on purpose (fp16/fp32/int8 variants come
    and go); anything with 'h3' that is not the audio codec qualifies, and a
    name carrying 'video' wins the tie."""
    vae_dir = root / 'models' / 'vae'
    if not vae_dir.is_dir():
        raise ValueError('No models/vae directory under the ComfyUI base dir.')
    candidates = [f for f in vae_dir.glob('*.safetensors')
                  if 'h3' in f.name.lower() and 'audio' not in f.name.lower()]
    if not candidates:
        raise ValueError('No H3 video VAE found in models/vae — '
                         'download minimax_h3_video_vae first.')
    return sorted(candidates, key=lambda f: ('video' not in f.name.lower(), f.name))[0]


_BAD_FACE_STATES = ('no_face', 'too_small', 'unreadable', 'error')
_WEAK_FACE_STATES = ('low_det', 'extreme_pose')


def _face_order(rows, limit):
    """正脸优先，侧脸为辅，避开遮挡: usable faces only (no_face/too_small/
    unreadable/error are not face references), ordered frontal-first (|yaw|),
    profiles and unmeasured rows after, and low_det (occluded/unclear) or
    extreme_pose demoted to the tail — still eligible when a dataset has
    nothing better."""
    usable = [r for r in rows
              if (r.face_state or 'scorable') not in _BAD_FACE_STATES]
    return sorted(usable, key=lambda r: (
        (r.face_state or '') in _WEAK_FACE_STATES,
        abs(r.face_yaw) if r.face_yaw is not None else 90.0,
        -(r.face_score or 0), r.id))[:limit]


def pick_images(images) -> list:
    """Identity-first pick, ≤20 rows: up to 12 face close-ups — 正脸优先，
    侧脸为辅，避开遮挡 (frontal |yaw| first, profiles after, low_det/
    extreme_pose demoted, dead face states excluded) — then 4 half shots and
    4 full frames: the figure and clothing ride to the mod too, so fulls are
    ALWAYS taken when the dataset has them (they are not a thin-set fallback
    any more; a thin set is whatever it is — every kept row is already in).
    House framing mapping (Bank build's): face→face, bust→half, body→full; a
    back view is not an identity reference and is never picked. Inside a
    bucket, scored rows lead (face_score DESC)."""
    by: dict[str, list] = {'face': [], 'half': [], 'full': []}
    for row in images:
        if getattr(row, 'status', None) != 'keep' or not getattr(row, 'filename', None):
            continue
        fr = (getattr(row, 'framing', None) or '').lower()
        if fr == 'back':
            continue
        by[{'face': 'face', 'bust': 'half'}.get(fr, 'full')].append(row)
    for bucket in by.values():
        bucket.sort(key=lambda r: (r.face_score is None, -(r.face_score or 0), r.id))
    picked = (_face_order(by['face'], _MAX_FACE)
              + by['half'][:_MAX_HALF] + by['full'][:_MAX_FULL])
    if not (by['face'] or by['half']):
        # A dataset with only full frames has no closer option — clothing is
        # unavoidable there, so at least keep 12 angles for coverage.
        picked = by['full'][:12]
    return picked[:_MAX_FACE + _MAX_HALF + _MAX_FULL]



def _kept_rows(ds):
    """The dataset's kept rows with files. FaceDataset has no images
    relationship — the store is queried by dataset_id, like everywhere else
    (measured: an ``ds.images`` guess 500s the whole route)."""
    return (FaceDatasetImage.query
            .filter_by(dataset_id=ds.id, status='keep')
            .filter(FaceDatasetImage.filename.isnot(None))
            .all())


def _draw_identity_mask(image_path, boxes, out_dir, expand=2.0):
    """Write a face mask for a 'too_large' frame: same dilate_box + ellipse +
    feather math as face_mask_infer (LDS convention, face=BLACK), but WITHOUT
    the 0.5-coverage refusal — a face crop where the face fills the frame is
    exactly the identity-dense image this feature must mask. Returns the PNG
    path or None."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    with Image.open(image_path) as img:
        w, h = img.size
    shift_up, feather_frac = 0.10, 0.03
    mask = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(mask)
    for b in boxes or []:
        x1, y1, x2, y2 = b
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0 - (y2 - y1) * shift_up
        hw, hh = (x2 - x1) * expand / 2.0, (y2 - y1) * expand / 2.0
        draw.ellipse([(cx - hw) * w, (cy - hh) * h, (cx + hw) * w, (cy + hh) * h], fill=0)
    r = max(1, int(min(w, h) * feather_frac))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=r))
    name = os.path.splitext(os.path.basename(image_path))[0] + '.png'
    out = os.path.join(out_dir, 'identity_' + name)
    mask.save(out, 'PNG')
    return out


def _face_masks_for(image_paths, ds_id):
    """Face-mask PNG per picked image (None where no confident face), via the
    same InsightFace pass masked training uses. LDS mask convention is face=BLACK
    / background=WHITE; the worker inverts to the RefMod keep=1 convention.
    Unavailable detection degrades to no masks — an unmasked RefMod is still a
    valid RefMod, it just carries what the frame carries."""
    if not image_paths:
        return [None] * len(image_paths)
    out_dir = os.path.join(os.path.dirname(dataset_path(ds_id)), os.pardir,
                           'masks', _MASKS_DIR_NAME, str(ds_id))
    out_dir = os.path.abspath(out_dir)
    try:
        result = face_mask_service.generate_face_masks(image_paths, out_dir)
    except Exception:  # noqa: BLE001 - masks are an enhancement, never a gate
        return [None] * len(image_paths)
    results = (result or {}).get('results') or {}
    masks = []
    for p in image_paths:
        entry = results.get(p) or {}
        state = entry.get('state')
        stem = os.path.splitext(os.path.basename(p))[0]
        mask_path = os.path.join(out_dir, stem + '.png')
        if state == 'masked' and os.path.isfile(mask_path):
            masks.append(mask_path)
        elif state == 'too_large' and entry.get('boxes'):
            # The training-oriented 0.5-coverage cap refuses exactly the
            # face-filling close-ups identity needs; redraw without the cap.
            try:
                masks.append(_draw_identity_mask(p, entry['boxes'], out_dir))
            except OSError:
                masks.append(None)
        else:
            masks.append(None)
    return masks


def _harvest_face_crops(ds, sources, deficit, note):
    """Thin face bucket: crop faces out of the dataset's own half/full pictures
    and persist them as real face rows (parent linked, derivation_kind tagged)
    so training and future regenerations reuse them. Crops softer than
    _UPSCALE_BELOW go through one Topaz batch (bank-build style); Topaz being
    unavailable degrades to the raw crops — the mod is never blocked."""
    if deficit <= 0 or not sources:
        return [], note
    face_rows = []   # 整图入选的（脸部≥50%）+ 新裁片，都作为 face 帧返回
    already = {r.parent_image_id for r in FaceDatasetImage.query.filter_by(
        dataset_id=ds.id, derivation_kind=_CROP_KIND).all() if r.parent_image_id}
    sources = [r for r in sources if r.id not in already]
    if not sources:
        return [], note
    paths = [str(Path(dataset_path(ds.id)) / r.filename) for r in sources]
    set_refmod_stage(ds.id, 'detecting faces')
    det = face_mask_service.detect_faces(paths)
    if not det.get('ok'):
        return [], (note + f'; face detection unavailable ({det.get("error", "?")}) — no crops')
    results = det.get('results') or {}
    storage = Path(dataset_path(ds.id))
    import uuid
    from PIL import Image
    from ..extensions import db
    from . import topaz_helper
    tmp_in = None
    staged = []   # (src_row, tmp_path, yaw) crops awaiting Topaz/persist
    crop_face_px = {}   # tmp_path -> largest face's SHORT side in crop pixels
    for idx, row in enumerate(sources):
        rec = results.get(paths[idx]) or {}
        if rec.get('state') not in ('masked', 'too_large', 'ok') or not rec.get('faces'):
            continue
        meta = max((f for f in rec.get('faces', [])
                    if f.get('det_score', 0) >= _CROP_MIN_DET),
                   key=lambda f: f.get('area', 0), default=None)
        if meta is None:
            continue
        if meta.get('area', 0) >= 0.5:
            # 脸部已占画面一半以上：这张图本身就是脸图，整图直接入选，不裁。
            face_rows.append(row)
            if len(face_rows) >= deficit:
                break
            continue
        src = storage / row.filename
        try:
            with Image.open(src) as im:
                w, h = im.size
                bi = rec['faces'].index(meta)
                x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in rec['boxes'][bi])
                bw, bh = (x2 - x1) * w, (y2 - y1) * h
                if max(bw, bh) < _CROP_MIN_FACE_PX:
                    continue
                # 方向外扩：头发在上、下巴在下都不能切。脸占裁片 ~25%，
                # 「≥50%」规则只用于判定原图是否直接入选，裁片以完整头部为先。
                cx = (x1 + x2) * w / 2
                mx, top, bot = 0.45 * bw, 0.75 * bh, 0.35 * bh
                l = max(0, int(cx - bw / 2 - mx))
                r_ = min(w, int(cx + bw / 2 + mx))
                t = max(0, int(y1 * h - top))
                b_ = min(h, int(y2 * h + bot))
                if r_ - l < _CROP_MIN_SIDE or b_ - t < _CROP_MIN_SIDE:
                    continue
                crop = im.convert('RGB').crop((l, t, r_, b_))
            set_refmod_stage(ds.id, 'cropping faces')
            if tmp_in is None:
                import tempfile
                tmp_in = tempfile.mkdtemp(prefix='lds-refmod-crops-')
            name = f'crop_{row.id}.png'
            path = os.path.join(tmp_in, name)
            crop.save(path, 'PNG')
            # Face pixels are invariant under cropping — the box was just
            # measured on the source, so the crop's face size comes free.
            crop_face_px[path] = min(bw, bh)
            staged.append((row, path, meta.get('yaw')))
            if len(face_rows) + len(staged) >= deficit:   # 剪图凑够：缺几张补几张
                break
        except Exception:  # noqa: BLE001 — one bad source must not stop the harvest
            continue
    if not staged and not face_rows:
        return [], note
    # Upscale EVERY crop first — the quality judge runs on the UPSCALED image,
    # not on the tiny source: a 170px face is exactly what Topaz exists for.
    # Smart face recovery: face pixel size is ALREADY known per crop (measured
    # at crop time) — tier crops and run one tpai pass per value, so no crop
    # ever meets Autopilot's 0.8-on-every-face wax default. Failure leaves the
    # tier's crops raw, exactly like the old single-pass failure path.
    upscaled_dir = None
    try:
        topaz_helper.preflight()
        import tempfile
        upscaled_dir = tempfile.mkdtemp(prefix='lds-refmod-topaz-')
        by_val = {}
        for path, px in crop_face_px.items():
            by_val.setdefault(topaz_helper.face_recovery_value(px), []).append(path)
        for value, crop_paths in sorted(by_val.items(), key=lambda kv: str(kv[0])):
            set_refmod_stage(ds.id, f'upscaling crops (Topaz, {len(crop_paths)} left)')
            tier_dir = os.path.join(tmp_in, f'v{value}')
            os.makedirs(tier_dir, exist_ok=True)
            for p in crop_paths:
                shutil.move(p, os.path.join(tier_dir, os.path.basename(p)))
            status, message = topaz_helper.run_tpai(
                topaz_helper.resolve_exe(), tier_dir, upscaled_dir,
                face_recovery=value, timeout=120 + 90 * len(crop_paths))
            if status not in ('ok', 'partial', None):   # run_tpai 返回状态字符串，不是退出码
                note += f'; Topaz failed ({message}) — crops stay raw'
    except Exception as e:  # noqa: BLE001 — Topaz is an enhancement, never a gate
        note += f'; Topaz unavailable ({e}) — crops stay raw'
    # Final file per crop: the upscaled one where Topaz produced it, raw otherwise.
    finals = []
    for row, tmp_path, yaw in staged:
        stem = os.path.splitext(os.path.basename(tmp_path))[0]
        up = os.path.join(upscaled_dir, stem + '.png') if upscaled_dir else None
        finals.append((row, up if up and os.path.isfile(up) else tmp_path, yaw))
    # THE judge: a crop is a face only if the detector still finds one in the
    # final image. Garbage in, garbage upscaled — still gets discarded here.
    det2 = face_mask_service.detect_faces([p for _, p, _ in finals]) if finals else {'results': {}}
    res2 = det2.get('results') or {}
    out_rows = []
    for row, p, yaw in finals:
        rec = res2.get(p) or {}
        if not any(f.get('det_score', 0) >= _CROP_MIN_DET for f in rec.get('faces', [])):
            continue
        final = storage / f'{_safe_name(ds.name, ds.id)}_facecrop_{row.id}_{uuid.uuid4().hex[:8]}.png'
        shutil.copy2(p, final)
        img = FaceDatasetImage(
            dataset_id=ds.id, source=row.source or 'import', status='keep',
            filename=final.name, parent_image_id=row.id, derivation_kind=_CROP_KIND,
            framing='face', face_yaw=yaw,
            variation_label=f'face crop from #{row.id}')
        db.session.add(img)
        out_rows.append(img)
    if len(out_rows) < len(staged):
        note += f'; {len(out_rows)}/{len(staged)} crops passed the post-upscale face check'
    db.session.commit()
    shutil.rmtree(tmp_in, ignore_errors=True)
    if upscaled_dir:
        shutil.rmtree(upscaled_dir, ignore_errors=True)
    return face_rows + out_rows, note


# -- live stage progress ------------------------------------------------------
# The generation route is SYNCHRONOUS (the VAE load dominates, ~1-3 min), so
# there is no job row to poll. The frontend instead polls the tiny progress
# endpoint below while its request is in flight; the current stage lives in
# this process-local dict (single-process Flask, one generation per GPU window
# thanks to the exclusive vision window — no cross-process registry needed).
_STAGES: dict[int, str] = {}


def set_refmod_stage(ds_id, stage):
    """Publish the current generation stage (None clears it)."""
    if stage:
        _STAGES[int(ds_id)] = stage
    else:
        _STAGES.pop(int(ds_id), None)


def current_stage(ds_id):
    return _STAGES.get(int(ds_id))


def _sharpness(path):
    """Cheap focus proxy: stddev of the FIND_EDGES pass (in-process PIL, no
    subprocess) — sharper picture, higher value."""
    try:
        from PIL import Image, ImageFilter, ImageStat
        with Image.open(path) as im:
            return ImageStat.Stat(
                im.convert('L').filter(ImageFilter.FIND_EDGES)).stddev[0]
    except Exception:                                            # noqa: BLE001
        return 0.0


_FILL_FRAMINGS = {'face': ('face',), 'half': ('bust',), 'full': ('body', 'full')}


def _fill_from_pending(ds, by, note):
    """Kept counts under quota → promote the best PENDING rows of the SAME
    framing to keep, ranked by similarity (face_score) then sharpness:
    缺脸找脸，缺全身找全身 — the triage pile is the first place the mod looks
    before it starts cropping. 'reject' is a decision and is never overridden;
    a pending row without a file on disk is skipped. Returns (count, note).
    ponytail: sharpness is an edge-energy proxy, computed for the candidate
    pool only — swap in a real focus model if the ranking measurably misfires."""
    quotas = {'face': _MAX_FACE, 'half': _MAX_HALF, 'full': _MAX_FULL}
    storage = Path(dataset_path(ds.id))
    total = 0
    for bucket, quota in quotas.items():
        deficit = quota - len(by[bucket])
        if deficit <= 0:
            continue
        from sqlalchemy import or_
        candidates = (FaceDatasetImage.query
                      .filter_by(dataset_id=ds.id, status='pending')
                      .filter(FaceDatasetImage.framing.in_(_FILL_FRAMINGS[bucket]))
                      .filter(FaceDatasetImage.filename.isnot(None))
                      .filter(or_(FaceDatasetImage.face_state.is_(None),
                                  FaceDatasetImage.face_state.notin_(_BAD_FACE_STATES)))
                      .all())
        ranked = sorted(candidates, key=lambda r: (
            (r.face_state or '') in _WEAK_FACE_STATES,
            -(r.face_score or 0),
            -_sharpness(str(storage / r.filename)), r.id))
        fills = ranked[:deficit]
        if not fills:
            continue
        for r in fills:
            r.status = 'keep'
        by[bucket].extend(fills)
        total += len(fills)
        note += (f'; {len(fills)} {bucket} frame(s) promoted from the triage '
                 f'pile (similarity+sharpness)')
    if total:
        from ..extensions import db
        db.session.commit()
    return total, note


_HINT_PROMPT = (
    '只根据这张人脸照片，用中文客观描述这个人的稳定身份特征，供绘画提示词使用：'
    '脸部特征（脸型、眼、鼻、唇）、发型与发色、肤色、明显的脸部特征点（如痣、雀斑、疤痕）。'
    '不要描述表情、情绪、背景、服装和摄影风格；不要评价美丑；用逗号分隔的短语，40字以内。'
    '/no_think')


_LLAMA_BASE = 'http://127.0.0.1:8080'   # the local llama.cpp llama-server


def _llama_describe(image_path, prompt, timeout=300):
    """Describe an image through the local llama-server (OpenAI-compatible;
    vision via a base64 image_url part — NOT Ollama, per the user's stack).
    Model auto-picked from /v1/models (the multi-model server rejects an
    unnamed request); thinking models spend small budgets on
    reasoning_content, so that field is the fallback. Returns '' on any
    failure — the hint degrades to its prefix."""
    import base64
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    try:
        with urllib.request.urlopen(f'{_LLAMA_BASE}/v1/models', timeout=10) as r:
            models = json.load(r).get('data') or []
        model = next((m['id'] for m in models if m.get('id')), '')
        if not model:
            return ''
        body = json.dumps({
            'model': model, 'max_tokens': 300, 'temperature': 0.3,
            'stream': False,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/png;base64,{b64}'}}]}]},
            ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(
            f'{_LLAMA_BASE}/v1/chat/completions', data=body,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            msg = (json.load(r).get('choices') or [{}])[0].get('message') or {}
        return (msg.get('content') or msg.get('reasoning_content') or '').strip()
    except Exception as e:                                       # noqa: BLE001
        logger.warning('refmod: llama describe failed: %s', e)
        return ''


def _identity_hint(ds, image_paths, picked, note):
    """Build the prompt_hint: a one-line identity description (facial features,
    hairstyle, skin tone, marks like moles) prefixed with the subject tag, e.g.
    ``<Subject 1>是一名名叫Joyce 的中国女性，…``. Best kept FACE frame drives the
    vision call (Ollama, best-effort); a silent/unavailable model degrades to
    the bare prefix — a hint that only half-exists still composes a prompt.
    Returns (hint, note)."""
    prefix = f'<Subject 1>是一名名叫{ds.name} 的中国女性'
    faces = [p for p, r in zip(image_paths, picked)
             if (r.framing or '') == 'face']
    target = (faces or image_paths)[:1]
    if not target:
        return prefix, note
    try:
        desc = _llama_describe(target[0], _HINT_PROMPT)
    except Exception as e:                                       # noqa: BLE001
        logger.warning('refmod: identity hint unavailable: %s', e)
        desc = ''
    if not desc:
        note += '; identity hint unavailable (vision model silent) — prefix only'
        return prefix, note
    return f'{prefix}，{desc.strip().rstrip("。")}。', note


def generate_for_dataset(ds, masked=False) -> dict:
    """Encode ds's kept images into one RefMod. Synchronous (~1-3 min: the VAE
    load dominates); the caller holds the GPU vision window.

    ``masked=False`` is the default (user-measured 2026-09-12: the face-mask
    suppression costs identity information instead of adding it); the plain
    name carries no suffix. ``masked=True`` applies face-mask suppression and
    names the output with a ``_mask`` suffix, kept for A/B comparison."""
    set_refmod_stage(ds.id, 'picking images')
    try:
        return _generate_for_dataset(ds, masked)
    finally:
        set_refmod_stage(ds.id, None)


def _generate_for_dataset(ds, masked=False) -> dict:
    root = _comfy_root()
    python_exe = _python_exe(root)
    node_dir = _node_dir(root)
    vae_path = _vae_path(root)

    rows = _kept_rows(ds)
    by: dict[str, list] = {'face': [], 'half': [], 'full': []}
    for row in rows:
        fr = (getattr(row, 'framing', None) or '').lower()
        if fr != 'back':
            by[{'face': 'face', 'bust': 'half'}.get(fr, 'full')].append(row)
    note = ''
    promoted, note = _fill_from_pending(ds, by, note)
    if promoted:
        note = f'{promoted} frame(s) promoted from the triage pile' + note
    deficit = _MAX_FACE - len(by['face'])
    if deficit > 0:
        pickups, note = _harvest_face_crops(ds, by['half'] + by['full'], deficit, note)
        if pickups:
            by['face'].extend(pickups)
            ids = {r.id for r in pickups}
            for key in ('half', 'full'):   # 直接入选的原图不再重复计票
                by[key] = [r for r in by[key] if r.id not in ids]
            note = f'{len(pickups)} face frame(s) harvested (whole-image + crops)' + note
    picked = pick_images([r for bucket in by.values() for r in bucket])
    if not picked:
        raise ValueError('No kept images to encode.')
    storage = Path(dataset_path(ds.id))
    image_paths = [str(storage / row.filename) for row in picked]
    set_refmod_stage(ds.id, 'generating face masks')
    masks = _face_masks_for(image_paths, ds.id) if masked else [None] * len(image_paths)
    masked_n = sum(1 for m in masks if m)
    suffix = '_mask' if masked else ''

    worker = Path(__file__).with_name('refmod_extract_worker.py')
    manifest = {
        'node_dir': str(node_dir),
        'images': image_paths,
        'vae': str(vae_path),
        'output_dir': str(root / 'models' / 'refmods'),
        'description': f'identity baseline from LDS dataset {ds.id} ({ds.name})',
        'max_tokens': _TOKEN_BUDGET,
        'masks': masks,
        'background_retention': _BACKGROUND_RETENTION,
    }
    hint, note = _identity_hint(ds, image_paths, picked, note)
    manifest['description'] = (f'identity baseline from LDS dataset '
                              f'{ds.id} ({ds.name}) | {hint}')
    manifest['name'] = f"minimaxh3_{_safe_name(ds.name, ds.id)}_v1{suffix}_refmod"
    os.makedirs(manifest['output_dir'], exist_ok=True)

    set_refmod_stage(ds.id, 'encoding (VAE load takes a minute)')
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [str(python_exe), str(worker)],
        input=json.dumps(manifest).encode('utf-8'),
        capture_output=True, timeout=_SUBPROCESS_TIMEOUT_S,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'})
    lines = [ln for ln in proc.stdout.decode('utf-8', 'replace').splitlines()
             if ln.strip().startswith('{')]
    try:
        result = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        result = {}
    if proc.returncode != 0 or not result.get('ok'):
        detail = result.get('error') or proc.stderr.decode('utf-8', 'replace')[-400:]
        raise RuntimeError(f'RefMod extraction failed: {detail}')
    try:
        with open(os.path.join(manifest['output_dir'],
                               manifest['name'] + '_prompt_hint.txt'),
                  'w', encoding='utf-8') as f:
            f.write(hint)
    except OSError:
        pass
    return {'name': manifest['name'], 'prompt_hint': hint,
            'tokens': result.get('tokens'),
            'frames': result.get('frames'), 'path': result.get('path'),
            'mb': result.get('mb'), 'masked': result.get('masked', 0),
            'note': note.strip('; ')}


def _safe_name(name: str, ds_id: int) -> str:
    out = ''.join(c if (c.isalnum() or c == '_') else '_' for c in name.strip())
    return out.strip('_').lower() or f'dataset{ds_id}'
