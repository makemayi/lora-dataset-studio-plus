"""✦ Edit reference on a LOCAL engine (Klein / Krea 2 Edit).

WHY THIS FILE EXISTS
--------------------
Editing the reference is the most exploratory gesture in the app — you re-run it
until the photo looks right — and it was the one gesture that could ONLY be done
on a paid API. The two engines that render on the user's own ComfyUI were absent
from the modal for a reason that had expired: the edit used to be a blocking
provider call, and a local engine has no blocking call to make. It has a QUEUE,
the same one every local generation already waits on.

So what is pinned here is the seam, not the render:
  - the local engines are OFFERED (the red half: Krea was not in the list at all);
  - starting one enqueues a ComfyUI job and registers it, instead of blocking;
  - the queue's completion callback fills the SAME candidate the modal polls, so
    the client contract (running -> ready|failed) is identical on both lanes;
  - an engine this install can't run is EXPLAINED (an actionable 409 on the click)
    and leaves no phantom "running" entry behind;
  - references a local graph can't take are REFUSED, never silently dropped;
  - the three API engines are untouched.

No GPU, no ComfyUI, no network, no dollars: every enqueue is stubbed.
"""
import contextlib
import io
import os

import pytest
from PIL import Image

from app.services import face_dataset_service as svc
from app.services import reference_edit_jobs as rej
from app.services import dataset_activity


def _png():
    b = io.BytesIO()
    Image.new('RGB', (256, 256), (120, 40, 40)).save(b, 'PNG')
    return b.getvalue()


def _webp(color, size=(300, 300)):
    b = io.BytesIO()
    Image.new('RGB', size, color).save(b, 'WEBP')
    return b.getvalue()


class _SyncThread:
    def __init__(self, target=None, args=(), kwargs=None, **_unused):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


@pytest.fixture(autouse=True)
def _clean_registry():
    rej.reset()
    dataset_activity.reset()
    yield
    rej.reset()
    dataset_activity.reset()


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


def _stub_krea(monkeypatch, calls, job_id='krea-job-1'):
    """A Krea install that is ready and whose enqueue records its arguments."""
    from app.services import krea_edit_helper as keh
    monkeypatch.setattr(keh, 'preflight', lambda: None)

    def _enqueue(**kw):
        calls.append(kw)
        return job_id
    monkeypatch.setattr(keh, 'enqueue_krea_edit', _enqueue)


def _stub_klein(monkeypatch, calls, job_id='klein-job-1'):
    from app.services import klein_edit_helper as keh

    def _enqueue(**kw):
        calls.append(kw)
        return job_id
    monkeypatch.setattr(keh, 'enqueue_klein_edit', _enqueue)


# --- the list ---------------------------------------------------------------

def test_both_local_engines_can_edit_the_reference():
    """THE red assertion. Krea 2 Edit (and Klein) were simply not in the set the
    route accepts, so the modal could not offer them however it was written."""
    editable = svc.editable_engines()
    assert 'krea' in editable
    assert 'klein' in editable


def test_the_api_engines_stay_editable_and_come_after_the_free_ones():
    """Non-regression on the paid lane, plus the reading order: a gesture billed
    per press lists its free options first."""
    editable = svc.editable_engines()
    for engine in svc.API_ENGINES:
        assert engine in editable
    assert editable[:len(svc.LOCAL_ENGINES)] == tuple(svc.LOCAL_ENGINES)


def test_the_refusal_names_every_editable_engine_including_the_local_ones():
    msg = svc.edit_engine_choice_message()
    labels = svc.engine_labels()
    for engine in svc.editable_engines():
        assert labels[engine] in msg, engine


# --- starting a local edit --------------------------------------------------

def test_krea_edit_enqueues_a_comfy_job_instead_of_blocking(client, monkeypatch):
    """The green half: the route answers 202 at once, a queue job is enqueued with
    the reference and the raw prompt, and the modal's registry entry is 'running'
    and knows which job it waits on."""
    did = _create_with_ref(client, monkeypatch, 'Kay', 'zchar_kay')
    calls = []
    _stub_krea(monkeypatch, calls)

    resp = client.post(f'/api/dataset/{did}/ref/edit',
                       data={'prompt': 'plain studio-grey background', 'engine': 'krea'},
                       content_type='multipart/form-data')

    assert resp.status_code == 202
    assert len(calls) == 1
    assert calls[0]['edit_prompt'] == 'plain studio-grey background'
    assert calls[0]['extra_metadata']['is_reference_edit'] is True
    entry = client.get(f'/api/dataset/{did}').get_json()['reference_edit']
    assert entry['status'] == 'running' and entry['engine'] == 'krea'
    # The callback must be able to find its way back to this edit.
    assert rej.find_by_job('krea-job-1')['dataset_id'] == did


def test_klein_edit_forwards_the_datasets_extra_references(client, monkeypatch):
    """Klein chains extra refs as native ReferenceLatent nodes, so the dataset's
    own anchors DO travel — the identity lock the API lane gets as bytes."""
    did = _create_with_ref(client, monkeypatch, 'Lex', 'zchar_lex')
    client.post(f'/api/dataset/{did}/ref/extra',
                data={'file': (io.BytesIO(_png()), 'x.png')},
                content_type='multipart/form-data')
    calls = []
    _stub_klein(monkeypatch, calls)

    resp = client.post(f'/api/dataset/{did}/ref/edit',
                       data={'prompt': 'warmer lighting', 'engine': 'klein'},
                       content_type='multipart/form-data')

    assert resp.status_code == 202
    assert len(calls) == 1
    assert len(calls[0]['extra_ref_paths']) == 1
    assert os.path.basename(calls[0]['extra_ref_paths'][0]).endswith('.webp')


def test_a_local_engine_refuses_the_modals_transient_references(client, monkeypatch):
    """Both local graphs take file PATHS; the modal's uploads are request-scoped
    bytes. Refused with the engine named — a silent drop would return an edit that
    ignored half of what the user handed it."""
    did = _create_with_ref(client, monkeypatch, 'Mo', 'zchar_mo')
    calls = []
    _stub_krea(monkeypatch, calls)

    resp = client.post(f'/api/dataset/{did}/ref/edit',
                       data={'prompt': 'add glasses', 'engine': 'krea',
                             'ref': (io.BytesIO(_png()), 'anchor.png')},
                       content_type='multipart/form-data')

    assert resp.status_code == 400
    assert 'Krea 2 Edit' in resp.get_json()['error']
    assert not calls                                   # nothing queued
    assert client.get(f'/api/dataset/{did}').get_json()['reference_edit'] is None


def test_an_unavailable_local_engine_is_explained_and_leaves_no_phantom_job(
        client, monkeypatch):
    """ComfyUI is there but Krea's weights/nodes are not: the click comes back with
    the SAME actionable 409 the generate path returns, immediately — not a spinner
    that ends in a raw ComfyUI error. And nothing is left 'running'."""
    did = _create_with_ref(client, monkeypatch, 'Nia', 'zchar_nia')
    from app.services import krea_edit_helper as keh

    def _boom():
        raise keh.KreaModelsMissing(['krea_model'])
    monkeypatch.setattr(keh, 'preflight', _boom)

    resp = client.post(f'/api/dataset/{did}/ref/edit',
                       data={'prompt': 'add glasses', 'engine': 'krea'},
                       content_type='multipart/form-data')

    assert resp.status_code == 409
    assert resp.get_json()['error']                    # actionable, not empty
    assert client.get(f'/api/dataset/{did}').get_json()['reference_edit'] is None
    assert dataset_activity.get(did) is None           # no ✦ badge left lit


# --- landing ----------------------------------------------------------------

def test_the_queue_callback_fills_the_candidate_the_modal_polls(client, monkeypatch):
    """The whole point of reusing the queue: the completion callback produces the
    same 'ready' + candidate_filename shape the API worker produces, so the client
    has ONE contract for both lanes."""
    did = _create_with_ref(client, monkeypatch, 'Ora', 'zchar_ora')
    calls = []
    _stub_krea(monkeypatch, calls)
    client.post(f'/api/dataset/{did}/ref/edit',
                data={'prompt': 'x', 'engine': 'krea'},
                content_type='multipart/form-data')
    # The render "landed": bytes come back through the /view fallback, so the test
    # needs no ComfyUI output folder at all.
    monkeypatch.setattr(svc, '_read_comfy_output', lambda fn: _webp((7, 7, 7)))
    monkeypatch.setattr(svc, '_drop_comfy_output', lambda fn: None)

    svc.link_completed_reference_edit('krea-job-1', 'out_00001_.png', failed=False)

    entry = client.get(f'/api/dataset/{did}').get_json()['reference_edit']
    assert entry['status'] == 'ready' and entry['candidate_filename']
    assert os.path.exists(os.path.join(svc._dataset_dir(did), entry['candidate_filename']))
    # Activity closed AFTER the candidate is ready — the poll's final refresh must
    # already see it, or the modal stays on the spinner forever.
    assert dataset_activity.get(did) is None


def test_a_failed_local_render_says_which_engine_failed_and_why(client, monkeypatch):
    did = _create_with_ref(client, monkeypatch, 'Pia', 'zchar_pia')
    calls = []
    _stub_krea(monkeypatch, calls)
    client.post(f'/api/dataset/{did}/ref/edit',
                data={'prompt': 'x', 'engine': 'krea'},
                content_type='multipart/form-data')

    svc.link_completed_reference_edit('krea-job-1', None, failed=True,
                                      reason='KSampler: out of memory')

    entry = client.get(f'/api/dataset/{did}').get_json()['reference_edit']
    assert entry['status'] == 'failed'
    assert 'krea' in entry['error'] and 'out of memory' in entry['error']
    assert dataset_activity.get(did) is None


def test_discarding_a_running_local_edit_cancels_the_render(client, monkeypatch):
    """A local edit CAN be cancelled (an API call already sent cannot), and leaving
    it running would hold the GPU for a result nobody will ever see."""
    did = _create_with_ref(client, monkeypatch, 'Rae', 'zchar_rae')
    calls = []
    _stub_krea(monkeypatch, calls)
    client.post(f'/api/dataset/{did}/ref/edit',
                data={'prompt': 'x', 'engine': 'krea'},
                content_type='multipart/form-data')
    cancelled = []
    from app import job_queue
    monkeypatch.setattr(job_queue.queue_manager, 'cancel_job',
                        lambda jid, **kw: cancelled.append(jid) or True)

    assert client.post(f'/api/dataset/{did}/ref/edit/discard').status_code == 200

    assert cancelled == ['krea-job-1']
    assert client.get(f'/api/dataset/{did}').get_json()['reference_edit'] is None
    assert dataset_activity.get(did) is None


def test_switching_to_an_api_engine_cancels_the_local_render_it_supersedes(
        client, monkeypatch):
    """Cross-lane supersede. Starting a ChatGPT edit over a running Krea one used
    to drop the registry entry and leave the GPU rendering a result whose callback
    would find nothing — with the ✦ badge lit until the TTL."""
    did = _create_with_ref(client, monkeypatch, 'Sam', 'zchar_sam')
    calls = []
    _stub_krea(monkeypatch, calls)
    client.post(f'/api/dataset/{did}/ref/edit',
                data={'prompt': 'x', 'engine': 'krea'},
                content_type='multipart/form-data')
    cancelled = []
    from app import job_queue
    monkeypatch.setattr(job_queue.queue_manager, 'cancel_job',
                        lambda jid, **kw: cancelled.append(jid) or True)

    class _SyncThread:
        def __init__(self, target=None, args=(), **kw):
            self._t, self._a = target, args

        def start(self):
            self._t(*self._a)
    monkeypatch.setattr(svc, '_edit_engine_call', lambda e, refs, p: _webp((0, 0, 255)))
    monkeypatch.setattr(svc.threading, 'Thread', _SyncThread)
    client.post(f'/api/dataset/{did}/ref/edit',
                data={'prompt': 'y', 'engine': 'chatgpt'},
                content_type='multipart/form-data')

    assert cancelled == ['krea-job-1']
    entry = client.get(f'/api/dataset/{did}').get_json()['reference_edit']
    assert entry['engine'] == 'chatgpt' and entry['status'] == 'ready'


def test_a_landing_nobody_awaits_deletes_its_output(client, monkeypatch):
    """Discarded/superseded while ComfyUI rendered: the finished file is removed
    rather than left in the output folder for the user to wonder about."""
    dropped = []
    monkeypatch.setattr(svc, '_drop_comfy_output', lambda fn: dropped.append(fn))
    svc.link_completed_reference_edit('nobody-waits', 'out_00002_.png', failed=False)
    assert dropped == ['out_00002_.png']


def test_mixed_api_local_batch_shares_snapshot_and_activity_until_local_lands(
        app, client, monkeypatch):
    from werkzeug.datastructures import MultiDict

    did = _create_with_ref(client, monkeypatch, 'Mixed', 'zchar_mixed')
    payload = client.get(f'/api/dataset/{did}').get_json()
    master_path = os.path.join(svc._dataset_dir(did), payload['ref_filename'])
    local_primary = []
    krea_calls = []
    from app.services import krea_edit_helper as keh
    monkeypatch.setattr(keh, 'preflight', lambda: None)

    def _enqueue_krea(**kwargs):
        krea_calls.append(kwargs)
        with open(kwargs['source_path'], 'rb') as fh:
            local_primary.append(fh.read())
        # Mutate the master after local admission. The API worker starts later
        # but must still receive the exact launch snapshot, not these new bytes.
        with open(master_path, 'wb') as fh:
            fh.write(_webp((250, 250, 0)))
        return 'mixed-krea-job'

    monkeypatch.setattr(keh, 'enqueue_krea_edit', _enqueue_krea)
    api_calls = []

    def _api(engine, refs, prompt):
        api_calls.append((engine, refs, prompt))
        return _webp((0, 100, 250))

    monkeypatch.setattr(svc, '_edit_engine_call', _api)
    monkeypatch.setattr(svc.threading, 'Thread', _SyncThread)
    real_end = dataset_activity.end
    ended = []

    def _end(token):
        ended.append(token)
        real_end(token)

    monkeypatch.setattr(dataset_activity, 'end', _end)
    response = client.post(
        f'/api/dataset/{did}/ref/edit',
        data=MultiDict([
            ('prompt', 'add glasses'),
            ('engines', 'krea'),
            ('engines', 'chatgpt'),
            ('ref', (io.BytesIO(_png()), 'modal-anchor.png')),
        ]),
        content_type='multipart/form-data')

    assert response.status_code == 202
    assert len(krea_calls) == 1
    assert 'extra_ref_paths' not in krea_calls[0]
    assert api_calls[0][0] == 'chatgpt'
    assert len(api_calls[0][1]) == 2       # primary + modal upload, API only
    assert api_calls[0][1][0] == local_primary[0]
    current = client.get(f'/api/dataset/{did}').get_json()['reference_edit']
    assert current['candidates']['chatgpt']['status'] == 'ready'
    assert current['candidates']['krea']['status'] == 'running'
    activity = dataset_activity.get(did)
    assert activity['done'] == 1 and activity['total'] == 2
    assert ended == []

    monkeypatch.setattr(svc, '_read_comfy_output', lambda filename: _webp((7, 7, 7)))
    monkeypatch.setattr(svc, '_drop_comfy_output', lambda filename: None)
    svc.link_completed_reference_edit(
        'mixed-krea-job', 'mixed-out.png', failed=False)

    final = client.get(f'/api/dataset/{did}').get_json()['reference_edit']
    assert all(candidate['status'] == 'ready'
               for candidate in final['candidates'].values())
    assert dataset_activity.get(did) is None
    assert len(ended) == 1


def test_discard_mixed_batch_cancels_all_locals_and_deletes_ready_api_candidate(
        client, monkeypatch):
    from werkzeug.datastructures import MultiDict

    did = _create_with_ref(client, monkeypatch, 'Discard all', 'zchar_discard_all')
    _stub_krea(monkeypatch, [], job_id='discard-krea')
    _stub_klein(monkeypatch, [], job_id='discard-klein')
    monkeypatch.setattr(
        svc, '_edit_engine_call',
        lambda engine, refs, prompt: _webp((10, 20, 30)))
    monkeypatch.setattr(svc.threading, 'Thread', _SyncThread)
    response = client.post(
        f'/api/dataset/{did}/ref/edit',
        data=MultiDict([
            ('prompt', 'x'),
            ('engines', 'krea'),
            ('engines', 'klein'),
            ('engines', 'chatgpt'),
        ]),
        content_type='multipart/form-data')
    assert response.status_code == 202
    current = client.get(f'/api/dataset/{did}').get_json()['reference_edit']
    candidate_path = os.path.join(
        svc._dataset_dir(did),
        current['candidates']['chatgpt']['candidate_filename'])
    assert os.path.exists(candidate_path)

    cancelled = []
    from app import job_queue
    monkeypatch.setattr(
        job_queue.queue_manager, 'cancel_job',
        lambda job_id, **kwargs: cancelled.append(job_id) or True)
    assert client.post(
        f'/api/dataset/{did}/ref/edit/discard').status_code == 200
    assert sorted(cancelled) == ['discard-klein', 'discard-krea']
    assert not os.path.exists(candidate_path)
    assert client.get(
        f'/api/dataset/{did}').get_json()['reference_edit'] is None
    assert dataset_activity.get(did) is None


# --- untrusted Comfy completion filenames ---------------------------------

def test_comfy_output_rejects_malicious_names_before_read_delete_or_view(
        tmp_path, monkeypatch):
    output = tmp_path / 'output'
    output.mkdir()
    outside = tmp_path / 'outside.webp'
    outside.write_bytes(b'do-not-touch')
    monkeypatch.setattr(svc, '_comfy_output_dir', lambda: str(output))
    from app.utils import comfyui
    view_calls = []
    monkeypatch.setattr(
        comfyui, 'fetch_output_image_bytes',
        lambda name: view_calls.append(name) or b'should-not-be-returned')

    malicious_names = (
        None,
        '',
        '\x00evil.webp',
        '.',
        '..',
        '../outside.webp',
        r'..\outside.webp',
        str(outside),
        r'C:\Windows\win.ini',
        r'\\server\share\outside.webp',
        '/etc/passwd',
    )
    for filename in malicious_names:
        assert svc._read_comfy_output(filename) is None
        svc._drop_comfy_output(filename)
        # Stale completions take the cleanup path even though no registry entry
        # remains; they must be just as constrained as a live completion.
        svc.link_completed_reference_edit(
            'stale-malicious-output', filename, failed=False)

    assert outside.read_bytes() == b'do-not-touch'
    assert view_calls == []


def test_comfy_output_allows_regular_contained_file(tmp_path, monkeypatch):
    output = tmp_path / 'output'
    output.mkdir()
    candidate = output / 'safe-output.webp'
    candidate.write_bytes(b'safe-bytes')
    monkeypatch.setattr(svc, '_comfy_output_dir', lambda: str(output))

    assert svc._read_comfy_output(candidate.name) == b'safe-bytes'
    svc._drop_comfy_output(candidate.name)
    assert not candidate.exists()


def test_comfy_output_refuses_symlink_file_for_read_and_delete(
        tmp_path, monkeypatch):
    output = tmp_path / 'output'
    output.mkdir()
    outside = tmp_path / 'outside.webp'
    outside.write_bytes(b'outside-secret')
    linked = output / 'linked.webp'
    try:
        os.symlink(str(outside), str(linked))
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f'symlinks unavailable on this platform: {exc}')
    monkeypatch.setattr(svc, '_comfy_output_dir', lambda: str(output))
    from app.utils import comfyui
    view_calls = []
    monkeypatch.setattr(
        comfyui, 'fetch_output_image_bytes',
        lambda name: view_calls.append(name) or b'unsafe-fallback')

    assert svc._read_comfy_output(linked.name) is None
    svc._drop_comfy_output(linked.name)
    svc.link_completed_reference_edit(
        'stale-symlink-output', linked.name, failed=False)

    assert os.path.lexists(linked)
    assert outside.read_bytes() == b'outside-secret'
    assert view_calls == []


def test_comfy_output_refuses_symlink_or_reparse_ancestor(
        tmp_path, monkeypatch):
    real_output = tmp_path / 'real-output'
    real_output.mkdir()
    candidate = real_output / 'candidate.webp'
    candidate.write_bytes(b'ancestor-secret')
    linked_output = tmp_path / 'linked-output'
    try:
        os.symlink(str(real_output), str(linked_output), target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f'directory symlinks unavailable on this platform: {exc}')
    monkeypatch.setattr(svc, '_comfy_output_dir', lambda: str(linked_output))
    from app.utils import comfyui
    view_calls = []
    monkeypatch.setattr(
        comfyui, 'fetch_output_image_bytes',
        lambda name: view_calls.append(name) or b'unsafe-fallback')

    assert svc._read_comfy_output(candidate.name) is None
    svc._drop_comfy_output(candidate.name)

    assert candidate.read_bytes() == b'ancestor-secret'
    assert view_calls == []


def test_comfy_output_deterministically_refuses_reparse_ancestor(
        tmp_path, monkeypatch):
    """Windows CI may lack symlink privilege; pin reparse propagation anyway."""
    output = tmp_path / 'output'
    output.mkdir()
    candidate = output / 'candidate.webp'
    candidate.write_bytes(b'reparse-secret')
    monkeypatch.setattr(svc, '_comfy_output_dir', lambda: str(output))
    real_lstat = svc.os.lstat
    output_key = os.path.normcase(os.path.abspath(str(output)))
    output_mode = real_lstat(output).st_mode

    class _ReparseDirectoryStat:
        st_mode = output_mode
        st_file_attributes = 0x400

    def _lstat_with_reparse_ancestor(path):
        key = os.path.normcase(os.path.abspath(os.fspath(path)))
        if key == output_key:
            return _ReparseDirectoryStat()
        return real_lstat(path)

    monkeypatch.setattr(svc.os, 'lstat', _lstat_with_reparse_ancestor)
    from app.utils import comfyui
    view_calls = []
    monkeypatch.setattr(
        comfyui, 'fetch_output_image_bytes',
        lambda name: view_calls.append(name) or b'unsafe-fallback')

    assert svc._read_comfy_output(candidate.name) is None
    svc._drop_comfy_output(candidate.name)

    assert candidate.read_bytes() == b'reparse-secret'
    assert view_calls == []
