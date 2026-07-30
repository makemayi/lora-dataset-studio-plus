"""Reference edit as a SERVER BACKGROUND JOB (survives a backgrounded mobile tab).

The engine is ALWAYS stubbed here — no network, no dollars. Invariants pinned:
  - the edit runs server-side; the candidate is rediscovered via the payload;
  - set_ready() happens BEFORE the activity ends, so the poll's final refresh sees
    a ready candidate (or the modal would hang on the spinner);
  - an abandoned candidate is purged by the TTL (the leak-proofing that replaces
    the old client-held Blob);
  - Keep reuses the atomic commit; a failed commit leaves the old reference intact.
"""
import contextlib
import io
import os
import time

import pytest
from PIL import Image

from app.services import face_dataset_service as svc
from app.services import reference_edit_jobs as rej
from app.services import dataset_activity


def _webp(color, size=(300, 300)):
    b = io.BytesIO()
    Image.new('RGB', size, color).save(b, 'WEBP')
    return b.getvalue()


def _png():
    b = io.BytesIO()
    Image.new('RGB', (256, 256), (120, 40, 40)).save(b, 'PNG')
    return b.getvalue()


def _oriented_jpeg(marker, size=(40, 80)):
    """A camera-like JPEG whose marker must never leave in an API data URL."""
    b = io.BytesIO()
    image = Image.new('RGB', size, (20, 130, 220))
    exif = image.getexif()
    exif[274] = 6
    exif[271] = marker
    image.save(b, 'JPEG', quality=95, subsampling=0, exif=exif)
    return b.getvalue()


def _seed_ref(ds, *, original, cropped):
    d = svc._dataset_dir(ds.id)
    orig_fn, ref_fn = 'local_datasetreforig_e.webp', 'local_datasetref_e.webp'
    if original is not None:
        with open(os.path.join(d, orig_fn), 'wb') as f:
            f.write(original)
        ds.ref_original_filename = orig_fn
    with open(os.path.join(d, ref_fn), 'wb') as f:
        f.write(cropped)
    ds.ref_filename = ref_fn
    svc.db.session.commit()
    return d, orig_fn, ref_fn


@pytest.fixture(autouse=True)
def _clean_registry():
    rej.reset()
    dataset_activity.reset()
    yield
    rej.reset()
    dataset_activity.reset()


# --- registry (reference_edit_jobs) -----------------------------------------

def test_start_supersedes_and_deletes_previous_candidate(app, tmp_path):
    d = str(tmp_path)
    t1 = rej.start(1, d, 'chatgpt', 'p1')
    cand = f'local{rej.CANDIDATE_MARKER}old.webp'
    open(os.path.join(d, cand), 'wb').close()
    assert rej.set_ready(1, t1, cand) is True
    # A new edit supersedes: previous candidate file deleted, old token now stale.
    t2 = rej.start(1, d, 'nanobanana', 'p2')
    assert t2 != t1
    assert not os.path.exists(os.path.join(d, cand))
    assert rej.set_ready(1, t1, cand) is False       # stale worker ignored
    assert rej.get(1)['status'] == 'running'


def test_get_lazy_ttl_deletes_candidate_and_entry(app, tmp_path, monkeypatch):
    d = str(tmp_path)
    t = rej.start(1, d, 'chatgpt', 'p')
    cand = f'local{rej.CANDIDATE_MARKER}x.webp'
    open(os.path.join(d, cand), 'wb').close()
    rej.set_ready(1, t, cand)
    # Age the entry past the TTL -> get() purges it AND deletes the file.
    monkeypatch.setattr(rej, '_TTL_SECONDS', -1)
    assert rej.get(1) is None
    assert not os.path.exists(os.path.join(d, cand))


def test_sweep_deletes_old_orphan_candidate(app, tmp_path, monkeypatch):
    d = str(tmp_path)
    orphan = os.path.join(d, f'local{rej.CANDIDATE_MARKER}crash.webp')
    open(orphan, 'wb').close()
    monkeypatch.setattr(rej, '_TTL_SECONDS', -1)     # everything counts as old
    rej.sweep(d)
    assert not os.path.exists(orphan)


# --- worker (_run_reference_edit) -------------------------------------------

def _run_sync(app, monkeypatch, out_bytes, *, engine='chatgpt', prompt='add glasses'):
    """Seed a dataset, start a job, run the worker synchronously with a stubbed
    engine returning `out_bytes` (or None). Returns the dataset id."""
    with app.app_context():
        ds = svc.create_dataset('local', 'W', f'zchar_{time.time_ns()}')
        _seed_ref(ds, original=None, cropped=_webp((5, 5, 5)))
        monkeypatch.setattr(svc, '_edit_engine_call', lambda e, refs, p: out_bytes)
        token = rej.start(ds.id, svc._dataset_dir(ds.id), engine, prompt)
        act = dataset_activity.begin(ds.id, 'edit_reference', total=1, engine=engine)
        svc._run_reference_edit(app, 'local', ds.id, token, act, engine,
                                [_webp((5, 5, 5))], prompt)
        return ds.id


def test_worker_success_sets_ready_before_activity_ends(app, monkeypatch):
    """KEY TEST (1): job finished -> the payload shows reference_edit READY *and*
    activity None together, so the poll's final refresh flips the modal to ready
    (set_ready ran before end())."""
    dsid = _run_sync(app, monkeypatch, _webp((0, 0, 255)))
    with app.app_context():
        payload = svc.dataset_payload('local', dsid)
        assert payload['activity'] is None
        assert payload['reference_edit']['status'] == 'ready'
        cand = payload['reference_edit']['candidate_filename']
        assert cand and rej.CANDIDATE_MARKER in cand
        assert os.path.exists(os.path.join(svc._dataset_dir(dsid), cand))


def test_worker_none_marks_failed(app, monkeypatch):
    dsid = _run_sync(app, monkeypatch, None)
    with app.app_context():
        re = svc.dataset_payload('local', dsid)['reference_edit']
        assert re['status'] == 'failed' and 'empty response' in re['error']


def test_worker_superseded_discards_its_candidate(app, monkeypatch):
    with app.app_context():
        ds = svc.create_dataset('local', 'S', 'zchar_sup')
        _seed_ref(ds, original=None, cropped=_webp((5, 5, 5)))
        dsdir = svc._dataset_dir(ds.id)
        stale = rej.start(ds.id, dsdir, 'chatgpt', 'p')
        rej.start(ds.id, dsdir, 'chatgpt', 'p2')      # supersedes -> stale token
        monkeypatch.setattr(svc, '_edit_engine_call', lambda e, refs, p: _webp((1, 2, 3)))
        act = dataset_activity.begin(ds.id, 'edit_reference')
        svc._run_reference_edit(app, 'local', ds.id, stale, act, 'chatgpt',
                                [_webp((5, 5, 5))], 'p')
        # The stale worker must not leak a candidate nor clobber the newer job.
        assert rej.get(ds.id)['status'] == 'running'
        cands = [n for n in os.listdir(dsdir) if rej.CANDIDATE_MARKER in n]
        assert cands == []


def test_worker_quota_exceeded_verbatim(app, monkeypatch):
    with app.app_context():
        ds = svc.create_dataset('local', 'Q', 'zchar_q')
        _seed_ref(ds, original=None, cropped=_webp((5, 5, 5)))
        def boom(e, refs, p):
            raise svc.SubscriptionQuotaExceeded('quota reached — rerun in API-key mode')
        monkeypatch.setattr(svc, '_edit_engine_call', boom)
        token = rej.start(ds.id, svc._dataset_dir(ds.id), 'chatgpt', 'p')
        act = dataset_activity.begin(ds.id, 'edit_reference')
        svc._run_reference_edit(app, 'local', ds.id, token, act, 'chatgpt',
                                [_webp((5, 5, 5))], 'p')
        re = svc.dataset_payload('local', ds.id)['reference_edit']
        assert re['status'] == 'failed' and 'quota reached' in re['error']


# --- start validation -------------------------------------------------------

def test_start_reference_edit_rejects_bad_engine_and_empty_prompt(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'V', 'zchar_v')
        _seed_ref(ds, original=None, cropped=_webp((5, 5, 5)))
        with pytest.raises(ValueError):
            svc.start_reference_edit(app, 'local', ds.id, 'midjourney', 'p')
        with pytest.raises(ValueError):
            svc.start_reference_edit(app, 'local', ds.id, 'chatgpt', '   ')


# --- Keep / Discard / Invalidate --------------------------------------------

def test_keep_promotes_and_removes_candidate(app, monkeypatch):
    dsid = _run_sync(app, monkeypatch, _webp((0, 0, 255)))
    with app.app_context():
        old_ref = svc.get_dataset('local', dsid).ref_filename
        new = svc.keep_reference_edit('local', dsid)
        assert new and new != old_ref
        assert svc.get_dataset('local', dsid).ref_filename == new
        assert svc.dataset_payload('local', dsid)['reference_edit'] is None
        cands = [n for n in os.listdir(svc._dataset_dir(dsid)) if rej.CANDIDATE_MARKER in n]
        assert cands == []


def test_keep_without_ready_returns_none(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'K', 'zchar_k')
        _seed_ref(ds, original=None, cropped=_webp((5, 5, 5)))
        assert svc.keep_reference_edit('local', ds.id) is None


def test_discard_clears_and_deletes(app, monkeypatch):
    dsid = _run_sync(app, monkeypatch, _webp((0, 0, 255)))
    with app.app_context():
        svc.discard_reference_edit(dsid)
        assert svc.dataset_payload('local', dsid)['reference_edit'] is None
        cands = [n for n in os.listdir(svc._dataset_dir(dsid)) if rej.CANDIDATE_MARKER in n]
        assert cands == []


def test_crop_reference_invalidates_pending_candidate(app, monkeypatch):
    dsid = _run_sync(app, monkeypatch, _webp((0, 0, 255)))
    with app.app_context():
        assert svc.dataset_payload('local', dsid)['reference_edit'] is not None
        assert svc.crop_reference('local', dsid, 0, 0, 64, 64) is True
        assert svc.dataset_payload('local', dsid)['reference_edit'] is None


# --- commit fail-safe (unchanged invariant) ---------------------------------

def test_commit_failure_preserves_old_reference(app):
    with app.app_context():
        ds = svc.create_dataset('local', 'Fa', 'zchar_fa')
        d, old_orig, old_ref = _seed_ref(
            ds, original=_webp((255, 0, 0)), cropped=_webp((0, 255, 0), (64, 64)))
        with pytest.raises(Exception):
            svc.commit_edited_reference('local', ds.id, b'not an image at all')
        assert ds.ref_filename == old_ref
        assert ds.ref_original_filename == old_orig
        assert os.path.exists(os.path.join(d, old_ref))
        assert os.path.exists(os.path.join(d, old_orig))


# --- routes -----------------------------------------------------------------

class _SyncThread:
    """Run the worker synchronously so route tests are deterministic."""
    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._t, self._a, self._k = target, args, kwargs or {}

    def start(self):
        self._t(*self._a, **self._k)


def _create_with_ref(client, monkeypatch, name, trig):
    import app.routes.datasets as dr
    monkeypatch.setattr(dr, 'gpu_exclusive_vision_window', lambda: contextlib.nullcontext())
    monkeypatch.setattr(dr.svc, 'face_crop_to_square_webp', lambda raw, **k: (_webp((1, 2, 3)), True))
    did = client.post('/api/dataset/create',
                      json={'name': name, 'trigger_word': trig}).get_json()['id']
    client.post(f'/api/dataset/{did}/ref',
                data={'file': (io.BytesIO(_png()), 'r.png')},
                content_type='multipart/form-data')
    return did


def test_route_edit_returns_202_and_payload_exposes_ready(client, monkeypatch):
    did = _create_with_ref(client, monkeypatch, 'Zoe', 'zchar_zoe')
    monkeypatch.setattr(svc, '_edit_engine_call', lambda e, refs, p: _webp((0, 0, 255)))
    monkeypatch.setattr(svc.threading, 'Thread', _SyncThread)   # run worker inline
    resp = client.post(f'/api/dataset/{did}/ref/edit',
                       data={'prompt': 'add glasses', 'engine': 'chatgpt'},
                       content_type='multipart/form-data')
    assert resp.status_code == 202
    re = client.get(f'/api/dataset/{did}').get_json()['reference_edit']
    assert re['status'] == 'ready' and re['candidate_filename']


def test_route_keep_swaps_reference(client, monkeypatch):
    did = _create_with_ref(client, monkeypatch, 'Ivy', 'zchar_ivy')
    before = client.get(f'/api/dataset/{did}').get_json()['ref_filename']
    monkeypatch.setattr(svc, '_edit_engine_call', lambda e, refs, p: _webp((0, 0, 255)))
    monkeypatch.setattr(svc.threading, 'Thread', _SyncThread)
    client.post(f'/api/dataset/{did}/ref/edit',
                data={'prompt': 'x', 'engine': 'nanobanana'},
                content_type='multipart/form-data')
    resp = client.post(f'/api/dataset/{did}/ref/edit/keep')
    assert resp.status_code == 200
    j = resp.get_json()
    assert j['ok'] is True and j['ref_filename'] != before
    assert client.get(f'/api/dataset/{did}').get_json()['ref_filename'] == j['ref_filename']


def test_route_keep_without_ready_409(client, monkeypatch):
    did = _create_with_ref(client, monkeypatch, 'No', 'zchar_no')
    assert client.post(f'/api/dataset/{did}/ref/edit/keep').status_code == 409


def test_route_discard_clears(client, monkeypatch):
    did = _create_with_ref(client, monkeypatch, 'Di', 'zchar_di')
    monkeypatch.setattr(svc, '_edit_engine_call', lambda e, refs, p: _webp((0, 0, 255)))
    monkeypatch.setattr(svc.threading, 'Thread', _SyncThread)
    client.post(f'/api/dataset/{did}/ref/edit',
                data={'prompt': 'x', 'engine': 'chatgpt'},
                content_type='multipart/form-data')
    assert client.post(f'/api/dataset/{did}/ref/edit/discard').status_code == 200
    assert client.get(f'/api/dataset/{did}').get_json()['reference_edit'] is None


def test_route_edit_bad_engine_400(client, monkeypatch):
    """An engine that does not exist is refused, and the refusal NAMES the ones
    that do — derived from editable_engines() rather than a sentence that rots
    when an engine lands. (Klein used to be the fixture here, on the grounds that
    a local GPU engine could not edit; it can, through the ComfyUI queue, so it is
    now in the list this message is built from — see
    test_ref_edit_local_engines.py.)"""
    did = _create_with_ref(client, monkeypatch, 'Jay', 'zchar_jay')
    resp = client.post(f'/api/dataset/{did}/ref/edit',
                       data={'prompt': 'x', 'engine': 'midjourney'},
                       content_type='multipart/form-data')
    assert resp.status_code == 400
    msg = resp.get_json()['error']
    assert msg == svc.edit_engine_choice_message()
    for engine in svc.editable_engines():
        assert svc.engine_labels()[engine] in msg
    assert 'midjourney' not in msg


def test_route_edit_openrouter_sends_every_reference_with_the_configured_model(
        app, client, monkeypatch):
    """OpenRouter is a first-class edit engine (it was left out of ✦ Edit when the
    engine shipped for generation only). Two things are pinned here, both of them
    money-shaped: the edit uses the model configured in Settings — never a slug
    hardcoded on the edit path — and EVERY reference travels (the dataset one plus
    the anchors added in the modal), because dropping some silently would return a
    face the user didn't ask for. The transport is stubbed: no call, no dollar."""
    from app import config as cfg
    from app.services import openrouter

    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-v1-testkeyvalue0123456789')
    with app.app_context():
        cfg.save_config({'engines': {'openrouter_model': 'acme/pixel-forge-1'}})

    did = _create_with_ref(client, monkeypatch, 'Ora', 'zchar_ora')
    sent = {}

    class _Resp:
        status_code = 200
        text = ''

        @staticmethod
        def json():
            import base64
            return {'data': [{'b64_json': base64.b64encode(_webp((9, 9, 9))).decode(),
                              'media_type': 'image/webp'}]}

    def _fake_post(url, **kwargs):
        sent['url'] = url
        sent['json'] = kwargs.get('json')
        return _Resp()

    monkeypatch.setattr(openrouter.requests, 'post', _fake_post)
    monkeypatch.setattr(svc.threading, 'Thread', _SyncThread)      # worker inline

    resp = client.post(
        f'/api/dataset/{did}/ref/edit',
        data={'prompt': 'add glasses', 'engine': 'openrouter',
              'ref': (io.BytesIO(_png()), 'anchor.png')},
        content_type='multipart/form-data')

    assert resp.status_code == 202
    payload = client.get(f'/api/dataset/{did}').get_json()['reference_edit']
    assert payload['status'] == 'ready' and payload['candidate_filename']
    assert sent['url'] == 'https://openrouter.ai/api/v1/images'
    assert sent['json']['model'] == 'acme/pixel-forge-1'
    # dataset reference + the modal anchor: both, in input_references.
    assert len(sent['json']['input_references']) == 2
    assert all(r['image_url']['url'].startswith('data:image/')
               for r in sent['json']['input_references'])


def test_external_edit_sanitizes_persistent_and_transient_camera_refs(
        app, client, monkeypatch):
    """Every API ref is upright pixels only; neither camera EXIF marker leaves."""
    import base64

    from app import config as cfg
    from app.config import LOCAL_USER
    from app.services import openrouter

    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-v1-testkeyvalue0123456789')
    with app.app_context():
        cfg.save_config({'engines': {'openrouter_model': 'acme/pixel-forge-1'}})
    did = _create_with_ref(client, monkeypatch, 'Sanitized', 'zchar_sanitized')
    primary_marker, extra_marker = 'PRIMARY-EXIF-MARKER', 'EXTRA-EXIF-MARKER'
    primary_raw = _oriented_jpeg(primary_marker)
    extra_raw = _oriented_jpeg(extra_marker, size=(60, 30))
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, did)
        ds.ref_filename = 'legacy-camera.jpg'
        primary_path = os.path.join(svc._dataset_dir(did), ds.ref_filename)
        with open(primary_path, 'wb') as fh:
            fh.write(primary_raw)
        svc.db.session.commit()

    sent = {}

    class _Resp:
        status_code = 200
        text = ''

        @staticmethod
        def json():
            return {'data': [{'b64_json': base64.b64encode(_webp((9, 9, 9))).decode(),
                              'media_type': 'image/webp'}]}

    def _fake_post(_url, **kwargs):
        sent['json'] = kwargs['json']
        return _Resp()

    monkeypatch.setattr(openrouter.requests, 'post', _fake_post)
    monkeypatch.setattr(svc.threading, 'Thread', _SyncThread)

    response = client.post(
        f'/api/dataset/{did}/ref/edit',
        data={'prompt': 'add glasses', 'engine': 'openrouter',
              'ref': (io.BytesIO(extra_raw), 'camera.jpg')},
        content_type='multipart/form-data')

    assert response.status_code == 202
    refs = sent['json']['input_references']
    assert len(refs) == 2
    remote = [base64.b64decode(ref['image_url']['url'].split(',', 1)[1]) for ref in refs]
    assert primary_marker.encode() not in remote[0]
    assert extra_marker.encode() not in remote[1]
    assert open(primary_path, 'rb').read() == primary_raw  # egress never mutates master
    with Image.open(io.BytesIO(remote[0])) as primary:
        assert primary.format == 'WEBP' and primary.size == (80, 40)
        assert not primary.getexif() and 'exif' not in primary.info
    with Image.open(io.BytesIO(remote[1])) as extra:
        assert extra.format == 'WEBP' and extra.size == (30, 60)
        assert not extra.getexif() and 'exif' not in extra.info


def test_external_edit_rejects_invalid_transient_ref_before_provider(client, monkeypatch):
    """Malformed modal bytes must not start a paid provider worker."""
    did = _create_with_ref(client, monkeypatch, 'Bad ref', 'zchar_bad_ref')
    calls = []
    monkeypatch.setattr(svc, '_edit_engine_call',
                        lambda *args: calls.append(args) or _webp((9, 9, 9)))

    response = client.post(
        f'/api/dataset/{did}/ref/edit',
        data={'prompt': 'add glasses', 'engine': 'chatgpt',
              'ref': (io.BytesIO(b'not an image'), 'broken.jpg')},
        content_type='multipart/form-data')

    assert response.status_code == 400
    assert 'extra edit reference 1' in response.get_json()['error']
    assert calls == []


def test_external_edit_caps_transient_route_read_before_service(client, monkeypatch):
    """The request snapshot uses max+1, so an oversized upload reaches no worker."""
    import app.routes.datasets as routes

    did = _create_with_ref(client, monkeypatch, 'Big ref', 'zchar_big_ref')
    called = []
    monkeypatch.setattr(svc, 'EXTERNAL_REFERENCE_MAX_BYTES', 8)
    monkeypatch.setattr(routes.svc, 'start_reference_edit',
                        lambda *args, **kwargs: called.append((args, kwargs)))

    response = client.post(
        f'/api/dataset/{did}/ref/edit',
        data={'prompt': 'add glasses', 'engine': 'chatgpt',
              'ref': (io.BytesIO(b'012345678'), 'large.jpg')},
        content_type='multipart/form-data')

    assert response.status_code == 400
    assert 'too large' in response.get_json()['error']
    assert called == []


def test_route_edit_openrouter_names_a_model_that_cannot_take_references(
        app, client, monkeypatch):
    """Not every OpenRouter model accepts reference images, and the model is free
    text in Settings. The failure has to be LOUD and NAMED — the engine's own words
    about the reference count — not a mute 'empty response' the user reads as a
    content refusal on their prompt."""
    from app.services import openrouter

    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-v1-testkeyvalue0123456789')
    did = _create_with_ref(client, monkeypatch, 'Nope', 'zchar_nope')

    class _Refused:
        status_code = 400
        text = ''

        @staticmethod
        def json():
            return {'error': {'code': 400, 'message': 'this model takes no image input'}}

    # Two POSTs happen (the aspect_ratio retry); both refuse.
    monkeypatch.setattr(openrouter.requests, 'post', lambda url, **k: _Refused())
    monkeypatch.setattr(svc.threading, 'Thread', _SyncThread)

    client.post(f'/api/dataset/{did}/ref/edit',
                data={'prompt': 'x', 'engine': 'openrouter',
                      'ref': (io.BytesIO(_png()), 'anchor.png')},
                content_type='multipart/form-data')

    err = client.get(f'/api/dataset/{did}').get_json()['reference_edit']['error']
    assert err.startswith('openrouter: ')
    assert 'this model takes no image input' in err
    assert 'may accept fewer' in err          # the reference-count hint, not silence


def test_route_edit_missing_dataset_404(client):
    resp = client.post('/api/dataset/999999/ref/edit',
                       data={'prompt': 'x', 'engine': 'chatgpt'},
                       content_type='multipart/form-data')
    assert resp.status_code == 404
