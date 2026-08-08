"""Stopping the face-mask preview must KEEP what it already found.

Asked while watching "Looking for faces… analyzing image 4 of 153": there was no
way out of the pass at all. A Stop button on its own would have been the wrong
answer, and this file is mostly about why.

The detector runs in a subprocess that loads antelopev2 before image 1 — a fixed
price (measured ~4-5 s warm on the reference machine, tens of seconds cold, plus
a ~350 MB download on the very first run) paid again on every start, because the
process exits with the pass. So a Stop that discarded the run would make the
retry cost the whole pass over: a button that looks like it saves time and spends
it. What is asserted here is the bargain that makes Stop worth offering:

  1. the child WINDS UP rather than being killed — it holds its boxes in memory
     until its final JSON line, so a kill is what loses them;
  2. those boxes are banked and the next start hands the child ONLY the images
     that are left;
  3. the bank is dropped when the kept set moves, under the SAME fingerprint that
     already flags a stored preview as stale;
  4. the bar keeps counting the whole kept set across the seam;
  5. the pass never held the GPU-exclusive window, so stopping cannot leak it.

No real detection: InsightFace is not installed on the dev machine, so stand-in
scripts speak the same stdin/stdout/stderr protocol as infer/face_mask_infer.py.
That protocol is the contract under test.
"""
import json
import os
import sys

from PIL import Image

from app.extensions import db
from app.models import FaceDatasetImage
from app.services import face_dataset_service as svc
from app.services import face_mask, face_mask_preview as fmp
from app.config import LOCAL_USER, save_config

PREVIEW = '/api/dataset/{}/train/face-mask-preview'
STOP = PREVIEW + '/stop'


def _dataset(app, tmp_path, n=5):
    """A concept dataset with `n` kept images. Same context discipline as
    test_face_mask_preview_progress: the `app` fixture does not push one."""
    save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'FMS', 'fms_act', kind='concept',
                                concept_desc='balancing a spoon')
        img_dir = svc._dataset_dir(ds.id)
        for i in range(n):
            fn = f'k{i}.png'
            Image.new('RGB', (64, 64)).save(os.path.join(img_dir, fn))
            db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn))
        db.session.commit()
        dataset_id = ds.id
    fmp.reset()
    return dataset_id


def _use(monkeypatch, tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding='utf-8')
    monkeypatch.setattr(face_mask, '_SCRIPT', str(p))
    monkeypatch.setattr(face_mask, '_face_python', lambda: sys.executable)
    monkeypatch.setattr(face_mask, 'is_available', lambda: True)


# A child that behaves EXACTLY like face_mask_infer under a stop: it checks the
# sentinel before each image and, when it appears, prints what it has with
# `cancelled: true` instead of dying. It also records the images it was handed,
# which is how a resume is proven to be a resume.
#
# `FMS_PAUSE_AT` makes the stop DETERMINISTIC: at that image the child waits for
# the sentinel instead of racing the parent's poller. Without it the stub runs to
# completion in milliseconds and the test asserts on whichever won — the classic
# green-when-broken timing test.
_STUB = (
    "import json, os, sys, time\n"
    "req = json.loads(sys.stdin.read())\n"
    "imgs = req['images']\n"
    "cancel = req.get('cancel_file')\n"
    "pause_at = int(os.environ.get('FMS_PAUSE_AT', '-1'))\n"
    "open(os.environ['FMS_LOG'], 'a', encoding='utf-8').write(json.dumps(imgs) + '\\n')\n"
    "print('[facemask] phase=detecting', file=sys.stderr, flush=True)\n"
    "out, cancelled = {}, False\n"
    "for i, p in enumerate(imgs, 1):\n"
    "    if i - 1 == pause_at and cancel:\n"
    "        deadline = time.time() + 15\n"
    "        while not os.path.exists(cancel) and time.time() < deadline:\n"
    "            time.sleep(0.02)\n"
    "    if cancel and os.path.exists(cancel):\n"
    "        cancelled = True\n"
    "        break\n"
    "    out[p] = {'state': 'masked', 'boxes': [[.4, .3, .6, .5]]}\n"
    "    print('[facemask] %d/%d masked faces=1' % (i, len(imgs)), file=sys.stderr, flush=True)\n"
    "print(json.dumps({'ok': True, 'written': 0, 'cancelled': cancelled, 'results': out}))\n"
)


def _log_path(tmp_path):
    return str(tmp_path / 'handed.log')


def _handed(tmp_path):
    """The image lists the child was handed, one per run."""
    p = _log_path(tmp_path)
    if not os.path.exists(p):
        return []
    return [json.loads(ln) for ln in open(p, encoding='utf-8').read().splitlines() if ln.strip()]


def _arm(monkeypatch, tmp_path, after):
    """Run the preview with a stop pressed while the child sits at image `after`.

    Driving `stop_requested` directly is the honest way to test this: under
    TESTING the job runs INLINE, so the POST does not return until the pass is
    over and there is no second thread to press the button from. What is under
    test is the plumbing from that flag to a banked partial, not the click. The
    child then waits for the sentinel at `after`, so the split is exact rather
    than whatever the two threads happened to do."""
    monkeypatch.setenv('FMS_LOG', _log_path(tmp_path))
    monkeypatch.setenv('FMS_PAUSE_AT', str(after))
    monkeypatch.setattr(fmp, 'stop_requested', lambda job: True)


# --- 1. a stop banks what was found, and does not publish a partial preview ---

def test_a_stopped_pass_banks_its_detections_instead_of_losing_them(
        app, client, tmp_path, monkeypatch):
    """RED before this wave: there was no stop at all, and the only way out of a
    pass — leaving the page — kept nothing."""
    _use(monkeypatch, tmp_path, 'stub.py', _STUB)
    ds_id = _dataset(app, tmp_path, n=5)
    _arm(monkeypatch, tmp_path, after=2)

    r = client.post(PREVIEW.format(ds_id), json={'limit': 6})
    assert r.status_code == 202

    body = client.get(PREVIEW.format(ds_id)).get_json()
    # It ended because it was ASKED to, not because it failed. Those are different
    # states and the panel says different things about them.
    assert body['job']['stopped'] is True
    assert body['job']['error'] is None
    # No preview is published: `coverage` is a safety figure over the WHOLE kept
    # set, so "masked on 2 of 2" for a 5-image set would be a WRONG number rather
    # than a partial one.
    assert body['result'] is None
    # But the work survived, and the panel is told what it is worth.
    assert body['resume']['total'] == 5
    assert 0 < body['resume']['done'] < 5


def test_the_stop_endpoint_reports_whether_there_was_a_pass_to_stop(
        app, client, tmp_path, monkeypatch):
    _use(monkeypatch, tmp_path, 'stub.py', _STUB)
    ds_id = _dataset(app, tmp_path, n=3)
    monkeypatch.setenv('FMS_LOG', _log_path(tmp_path))
    # Nothing running: an honest false, not a 404 and not a cheerful true.
    r = client.post(STOP.format(ds_id), json={})
    assert r.status_code == 200 and r.get_json()['stopping'] is False


# --- 2. the next start resumes ------------------------------------------------

def test_starting_again_hands_the_child_only_the_images_that_are_left(
        app, client, tmp_path, monkeypatch):
    """The point of the whole mechanism. Before it, a second start re-detected
    every image and re-paid the detector load for nothing."""
    _use(monkeypatch, tmp_path, 'stub.py', _STUB)
    ds_id = _dataset(app, tmp_path, n=5)
    _arm(monkeypatch, tmp_path, after=2)
    client.post(PREVIEW.format(ds_id), json={'limit': 6})

    first = _handed(tmp_path)[0]
    banked = client.get(PREVIEW.format(ds_id)).get_json()['resume']['done']
    assert len(first) == 5          # the stopped pass was handed everything
    # Pinned, not read-and-trusted: every assertion below is relative to `banked`,
    # so a stop that banked NOTHING would satisfy them all trivially — the test
    # would go green over the exact regression it exists to catch.
    assert banked == 2

    # Second start: no stop this time.
    monkeypatch.setattr(fmp, 'stop_requested', lambda job: False)
    client.post(PREVIEW.format(ds_id), json={'limit': 6})

    second = _handed(tmp_path)[1]
    assert len(second) == 5 - banked
    assert not set(second) & set(first[:banked])

    body = client.get(PREVIEW.format(ds_id)).get_json()
    # The published preview covers the WHOLE kept set — the banked half and the
    # freshly detected half, folded into one result.
    assert body['result']['coverage']['total'] == 5
    assert body['result']['coverage']['masked'] == 5
    assert body['job']['stopped'] is False
    # The bank is spent, not left behind to shadow a later pass.
    assert body['resume'] is None


def test_a_resume_with_nothing_left_publishes_without_loading_the_detector(
        app, client, tmp_path, monkeypatch):
    """Stopped on the very last image: everything is banked. Re-running the child
    would pay an InsightFace load to re-derive what is already in hand."""
    _use(monkeypatch, tmp_path, 'stub.py', _STUB)
    ds_id = _dataset(app, tmp_path, n=3)
    monkeypatch.setenv('FMS_LOG', _log_path(tmp_path))
    with app.app_context():
        from app.routes import training as tr
        by_path, fp = tr._face_preview_kept(ds_id)
    fmp.remember_partial(ds_id, {p: {'state': 'masked', 'boxes': [[.4, .3, .6, .5]]}
                                 for p in by_path}, fp)

    client.post(PREVIEW.format(ds_id), json={'limit': 6})

    assert _handed(tmp_path) == []          # the child was never started
    body = client.get(PREVIEW.format(ds_id)).get_json()
    assert body['result']['coverage']['masked'] == 3
    assert body['resume'] is None


# --- 3. the SAME fingerprint guards the bank ----------------------------------

def test_changing_the_kept_set_drops_the_bank_instead_of_resuming_onto_it(
        app, client, tmp_path, monkeypatch):
    """The one failure mode worse than starting over: boxes detected on photos
    that have since left the run, drawn as if they described it."""
    _use(monkeypatch, tmp_path, 'stub.py', _STUB)
    ds_id = _dataset(app, tmp_path, n=5)
    _arm(monkeypatch, tmp_path, after=2)
    client.post(PREVIEW.format(ds_id), json={'limit': 6})
    assert client.get(PREVIEW.format(ds_id)).get_json()['resume']['done'] > 0

    # The kept set moves: one more image is kept.
    with app.app_context():
        fn = 'k9.png'
        Image.new('RGB', (64, 64)).save(os.path.join(svc._dataset_dir(ds_id), fn))
        db.session.add(FaceDatasetImage(dataset_id=ds_id, status='keep', filename=fn))
        db.session.commit()

    # The credit disappears from the snapshot the moment the set no longer matches.
    assert client.get(PREVIEW.format(ds_id)).get_json()['resume'] is None

    monkeypatch.setattr(fmp, 'stop_requested', lambda job: False)
    client.post(PREVIEW.format(ds_id), json={'limit': 6})
    # A FULL pass: nothing was carried across the change.
    assert len(_handed(tmp_path)[1]) == 6


def test_the_bank_is_keyed_by_the_same_fingerprint_the_result_uses(app, tmp_path):
    """Not a second staleness mechanism — the same one. A mismatch drops the bank
    whole rather than salvaging rows: the fingerprint covers the set at once, so
    a mismatch says nothing in it can be trusted to describe the run."""
    ds_id = _dataset(app, tmp_path, n=2)
    fp = fmp.fingerprint([(1, 'a.png', 10, 5)])
    other = fmp.fingerprint([(1, 'a.png', 10, 6)])
    assert fp != other
    fmp.remember_partial(ds_id, {'/x/a.png': {'state': 'masked', 'boxes': []}}, fp)
    assert fmp.partial(ds_id, fp) == {'/x/a.png': {'state': 'masked', 'boxes': []}}
    assert fmp.partial(ds_id, other) == {}
    # And it is GONE, not merely hidden — a later call with the old fingerprint
    # must not resurrect detections already judged untrustworthy.
    assert fmp.partial(ds_id, fp) == {}


# --- 4. the bar counts the whole kept set across the seam ---------------------

def test_a_resumed_pass_keeps_counting_the_whole_kept_set(
        app, client, tmp_path, monkeypatch):
    """The child counts its OWN images, which on a resume are only the remaining
    ones. Left alone, a resumed pass restarts its bar at "image 1 of 3" and reads
    exactly like the stop having thrown everything away."""
    _use(monkeypatch, tmp_path, 'stub.py', _STUB)
    ds_id = _dataset(app, tmp_path, n=5)
    with app.app_context():
        from app.routes import training as tr
        by_path, fp = tr._face_preview_kept(ds_id)
    monkeypatch.setenv('FMS_LOG', _log_path(tmp_path))
    two = list(by_path)[:2]
    fmp.remember_partial(ds_id, {p: {'state': 'masked', 'boxes': []} for p in two}, fp)

    seen = []
    real_progress = fmp.progress

    def _spy(job, rec):
        real_progress(job, rec)
        if rec.get('done') is not None:
            seen.append((rec['done'], rec.get('total')))
    monkeypatch.setattr(fmp, 'progress', _spy)

    client.post(PREVIEW.format(ds_id), json={'limit': 6})

    # Three images left, but the count runs 3,4,5 of 5 — never 1,2,3 of 3.
    assert [d for d, _ in seen][-3:] == [3, 4, 5]
    assert {t for _, t in seen} == {5}


# --- 5. the GPU-exclusive window is not held, so it cannot be leaked ----------

def test_the_preview_pass_never_holds_the_gpu_exclusive_window(
        app, client, tmp_path, monkeypatch):
    """A stop that freed the screen without freeing the card would leave the next
    pass waiting on a lock nobody holds.

    It cannot happen here, and the reason is worth pinning rather than trusting:
    this detector is CPU-only by construction (face_mask_infer pins
    CPUExecutionProvider and ctx_id=-1), so the preview never takes the window in
    the first place. Bank scoring takes it ONLY when its child really runs on the
    card — same rule, opposite answer. If someone ever wires a window in here,
    this fails and they have to think about the stop path with it."""
    from app import gpu_window
    _use(monkeypatch, tmp_path, 'stub.py', _STUB)
    ds_id = _dataset(app, tmp_path, n=3)
    monkeypatch.setenv('FMS_LOG', _log_path(tmp_path))

    owned = []
    real = face_mask.detect_faces

    def _spy(*a, **kw):
        owned.append((gpu_window.vision_window_is_owned(),
                      gpu_window.vision_gpu_window_blocks_gpu()))
        return real(*a, **kw)
    monkeypatch.setattr(face_mask, 'detect_faces', _spy)

    client.post(PREVIEW.format(ds_id), json={'limit': 6})

    assert owned == [(False, False)]
    # And afterwards nothing is left owned — the state a leak would show up in.
    assert gpu_window.vision_window_is_owned() is False
    assert gpu_window.vision_gpu_window_blocks_gpu() is False


def test_the_detector_is_pinned_to_the_cpu(app):
    """The premise of the test above, asserted at its source rather than assumed.
    The day this script asks for CUDA, the preview starts contending for the card
    and the window question becomes real."""
    from app import config as cfg
    src = (cfg.BACKEND_DIR / 'infer' / 'face_mask_infer.py').read_text(encoding='utf-8')
    assert "'CPUExecutionProvider'" in src
    assert 'ctx_id=-1' in src


# --- 6. the child's own half of the contract ---------------------------------

def test_the_child_winds_up_on_the_sentinel_and_prints_what_it_found(tmp_path):
    """The real infer script, exercised without InsightFace: `cancel_requested`
    and the graceful exit are what make a stop cheap. A SIGKILL here would be the
    bug — the results exist nowhere but in that process until the last line."""
    sys.path.insert(0, str((__import__('app.config', fromlist=['x']).BACKEND_DIR / 'infer')))
    import face_mask_infer

    assert face_mask_infer.cancel_requested(None) is False
    sentinel = tmp_path / 'stop'
    assert face_mask_infer.cancel_requested(str(sentinel)) is False
    sentinel.write_text('stop', encoding='utf-8')
    assert face_mask_infer.cancel_requested(str(sentinel)) is True


def test_a_stop_is_not_a_failure(app, client, tmp_path, monkeypatch):
    """`ok` stays true on a cancelled pass. Reporting it as an error would send
    the parent down the failure path and discard the very detections the wind-up
    exists to hand back — and would tell the user their pass crashed when they
    stopped it."""
    _use(monkeypatch, tmp_path, 'stub.py', _STUB)
    ds_id = _dataset(app, tmp_path, n=4)
    monkeypatch.setenv('FMS_LOG', _log_path(tmp_path))
    monkeypatch.setenv('FMS_PAUSE_AT', '1')
    paths = [os.path.join(svc._dataset_dir(ds_id), f'k{i}.png') for i in range(4)]

    data = face_mask.detect_faces(paths, timeout=60, should_stop=lambda: True)
    assert data['ok'] is True
    assert data['cancelled'] is True


def test_the_sentinel_directory_does_not_outlive_the_pass(app, tmp_path, monkeypatch):
    """A leftover sentinel would silently cancel the NEXT pass — a nastier bug
    than the one this fixes."""
    _use(monkeypatch, tmp_path, 'stub.py', _STUB)
    ds_id = _dataset(app, tmp_path, n=2)
    monkeypatch.setenv('FMS_LOG', _log_path(tmp_path))
    paths = [os.path.join(svc._dataset_dir(ds_id), f'k{i}.png') for i in range(2)]

    seen = {}
    real = face_mask._stop_plumbing

    def _spy(should_stop):
        path, on_stop, cleanup = real(should_stop)
        seen['path'] = path
        return path, on_stop, cleanup
    monkeypatch.setattr(face_mask, '_stop_plumbing', _spy)

    face_mask.detect_faces(paths, timeout=60, should_stop=lambda: True)
    assert seen['path'] and not os.path.exists(os.path.dirname(seen['path']))
