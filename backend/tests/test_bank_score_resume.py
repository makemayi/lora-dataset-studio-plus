"""✨ Score — resuming, stopping, and the explicit re-score lane.

These tests measure the PROPERTY, not the plumbing: every child-side test counts
how many images CLIP was actually asked to embed, and every parent-side test
reads the rows back out of the database. "A filter is applied" would prove
nothing about whether a relaunch is cheap or whether stopping keeps its work.

The child is exercised in-process against stub torch/open_clip modules — that is
what makes "how many images were embedded" observable at all, and it keeps the
suite free of a 1 GB model download.
"""
import importlib.util
import io
import json
import os
import pathlib
import sys
import types

import pytest
from PIL import Image

np = pytest.importorskip('numpy')

INFER = (pathlib.Path(__file__).resolve().parents[1] / 'infer'
         / 'bank_score_infer.py')


def _load_child():
    spec = importlib.util.spec_from_file_location('bank_score_infer_test', INFER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- stub ML stack (the only way to count real embedding work) ---------------
class _Val:
    def __init__(self, v):
        self.v = float(v)

    def item(self):
        return self.v


class _Tensor:
    """Just enough of a torch tensor for the child's four operations."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype='float32')

    def norm(self, dim=None, keepdim=False):
        return _Tensor(np.linalg.norm(self.arr, axis=-1, keepdims=True))

    def __truediv__(self, other):
        return _Tensor(self.arr / other.arr)

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _Token:
    def __init__(self, value):
        self.value = value

    def unsqueeze(self, _dim):
        return self

    def to(self, _device):
        return self


class _Clip:
    """Counts every image it is asked to embed — the number under test."""

    def __init__(self, calls):
        self.calls = calls

    def to(self, _device):
        return self

    def eval(self):
        return self

    def encode_image(self, token):
        self.calls.append(token.value)
        # Two directions, chosen by brightness: dark images share a style, bright
        # ones share another. Deterministic, so clusters are assertable.
        vec = np.zeros(768, dtype='float32')
        vec[0 if token.value < 128 else 1] = 1.0
        return _Tensor(vec[None, :])


def _install_stubs(monkeypatch, mod, calls, aes_ok=True, nsfw_ok=True):
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    torch.no_grad = _NoGrad
    torch.softmax = lambda t, dim=None: t
    open_clip = types.ModuleType('open_clip')

    def _preprocess(im):
        return _Token(float(np.asarray(im.convert('L')).mean()))

    clip = _Clip(calls)
    open_clip.create_model_and_transforms = \
        lambda *a, **k: (clip, None, _preprocess)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'open_clip', open_clip)

    head = (lambda emb: [[_Val(7.25)]]) if aes_ok else None
    monkeypatch.setattr(mod, '_load_aesthetic_head',
                        lambda *a, **k: (head, aes_ok,
                                         None if aes_ok else 'URLError: unreachable'))

    class _Model:
        def __call__(self, **kw):
            return types.SimpleNamespace(logits=[[_Val(0.1), _Val(0.9)]])

    class _Proc:
        def __call__(self, images=None, return_tensors=None):
            return self

        def to(self, _device):
            return {}

    bundle = (_Model(), _Proc(), 1) if nsfw_ok else None
    monkeypatch.setattr(mod, '_load_nsfw',
                        lambda *a, **k: (bundle, nsfw_ok,
                                         None if nsfw_ok else 'OSError: unreachable'))


def _run_child(mod, monkeypatch, request):
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(request)))
    out = io.StringIO()
    monkeypatch.setattr('sys.stdout', out)
    rc = mod.main()
    lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
    return rc, json.loads(lines[-1])


def _images(tmp_path, spec):
    """{name: grey level} -> [abs paths], written as real decodable JPEGs."""
    paths = []
    for name, level in spec.items():
        p = tmp_path / name
        Image.new('RGB', (64, 64), (level, level, level)).save(str(p), 'JPEG')
        paths.append(str(p))
    return paths


# --- 1. a relaunch computes only what is missing ------------------------------
def test_relaunch_embeds_only_the_images_that_have_no_score(tmp_path, monkeypatch):
    mod = _load_child()
    paths = _images(tmp_path, {f'i{i}.jpg': 20 + i for i in range(5)})
    cache = str(tmp_path / 'score_cache.npz')
    req = {'images': paths, 'cache': cache, 'style_threshold': 0.6}

    first_calls = []
    _install_stubs(monkeypatch, mod, first_calls)
    rc, data = _run_child(mod, monkeypatch, {**req, 'images': paths[:3]})
    assert rc == 0 and data['ok']
    assert len(first_calls) == 3            # nothing cached yet

    # Relaunch over the WHOLE bank: the three already done must not be re-embedded.
    second_calls = []
    mod2 = _load_child()
    _install_stubs(monkeypatch, mod2, second_calls)
    rc, data = _run_child(mod2, monkeypatch, req)
    assert rc == 0 and data['ok']
    assert len(second_calls) == 2, 'only the two missing images may be embedded'
    assert data['computed'] == 2 and data['reused'] == 3
    assert set(data['results']) == set(paths)
    assert all(r['state'] == 'ok' for r in data['results'].values())

    # And a third run over a complete cache embeds nothing at all.
    third_calls = []
    mod3 = _load_child()
    _install_stubs(monkeypatch, mod3, third_calls)
    rc, data = _run_child(mod3, monkeypatch, req)
    assert third_calls == []
    assert data['computed'] == 0 and data['reused'] == 5
    assert len(data['clusters']) == 5       # the partition still covers everything


def test_style_cluster_ids_stay_one_partition_across_a_resume(tmp_path, monkeypatch):
    """The whole reason the pool is never filtered: ids are renumbered globally,
    so a resumed pass must still hand back ONE numbering covering every image."""
    mod = _load_child()
    dark = _images(tmp_path, {'d1.jpg': 20, 'd2.jpg': 22, 'd3.jpg': 24})
    bright = _images(tmp_path, {'b1.jpg': 200, 'b2.jpg': 210})
    cache = str(tmp_path / 'score_cache.npz')

    _install_stubs(monkeypatch, mod, [])
    _run_child(mod, monkeypatch,
               {'images': dark, 'cache': cache, 'style_threshold': 0.6})

    mod2 = _load_child()
    calls = []
    _install_stubs(monkeypatch, mod2, calls)
    _rc, data = _run_child(mod2, monkeypatch,
                           {'images': dark + bright, 'cache': cache,
                            'style_threshold': 0.6})
    assert len(calls) == 2
    clusters = data['clusters']
    assert set(clusters) == set(dark + bright)
    assert len({clusters[p] for p in dark}) == 1
    assert len({clusters[p] for p in bright}) == 1
    assert clusters[dark[0]] != clusters[bright[0]]
    # Biggest first, contiguous from 1 — a partition, not two numberings.
    assert sorted(set(clusters.values())) == [1, 2]
    assert clusters[dark[0]] == 1


# --- 2. holes left by a head that was down are filled on the next run ---------
def test_a_head_that_was_down_is_retried_and_never_churns(tmp_path, monkeypatch):
    mod = _load_child()
    paths = _images(tmp_path, {'a.jpg': 30, 'b.jpg': 40})
    cache = str(tmp_path / 'score_cache.npz')
    req = {'images': paths, 'cache': cache, 'style_threshold': 0.6}

    _install_stubs(monkeypatch, mod, [], nsfw_ok=False)
    _rc, data = _run_child(mod, monkeypatch, req)
    assert all('nsfw' not in r for r in data['results'].values())
    assert all('aesthetic' in r for r in data['results'].values())

    # Still down: nothing is recomputed. A permanently missing model must not
    # turn every relaunch into a full re-score.
    calls = []
    mod2 = _load_child()
    _install_stubs(monkeypatch, mod2, calls, nsfw_ok=False)
    _run_child(mod2, monkeypatch, req)
    assert calls == []

    # Back up: exactly the holed images are re-scored, and the aesthetic value
    # they already had survives.
    calls = []
    mod3 = _load_child()
    _install_stubs(monkeypatch, mod3, calls, nsfw_ok=True)
    _rc, data = _run_child(mod3, monkeypatch, req)
    assert len(calls) == 2
    assert all(r['nsfw'] == pytest.approx(0.9) for r in data['results'].values())
    assert all(r['aesthetic'] == pytest.approx(7.25)
               for r in data['results'].values())


# --- 3. the explicit re-score lane -------------------------------------------
def test_rescore_recomputes_everything(tmp_path, monkeypatch):
    mod = _load_child()
    paths = _images(tmp_path, {'a.jpg': 30, 'b.jpg': 40, 'c.jpg': 50})
    cache = str(tmp_path / 'score_cache.npz')
    req = {'images': paths, 'cache': cache, 'style_threshold': 0.6}
    _install_stubs(monkeypatch, mod, [])
    _run_child(mod, monkeypatch, req)

    calls = []
    mod2 = _load_child()
    _install_stubs(monkeypatch, mod2, calls)
    _rc, data = _run_child(mod2, monkeypatch, {**req, 'rescore': True})
    assert len(calls) == 3, 'rescore must ignore the cache entirely'
    assert data['computed'] == 3 and data['reused'] == 0
    assert len(data['clusters']) == 3


# --- 4. stopping mid-compute hands back the paid work, minus the partition ----
def test_cancelling_returns_the_computed_scores_and_no_half_partition(
        tmp_path, monkeypatch):
    mod = _load_child()
    paths = _images(tmp_path, {f'i{i}.jpg': 30 + i for i in range(4)})
    cache = str(tmp_path / 'score_cache.npz')
    sentinel = str(tmp_path / 'stop.flag')

    # Drop the sentinel after the first image, exactly like the parent's Stop.
    real_sig = mod._file_sig
    seen = {'n': 0}

    def _sig(path):
        seen['n'] += 1
        if seen['n'] == 1:
            with open(sentinel, 'w', encoding='utf-8') as f:
                f.write('1')
        return real_sig(path)

    monkeypatch.setattr(mod, '_file_sig', _sig)
    _install_stubs(monkeypatch, mod, [])
    _rc, data = _run_child(mod, monkeypatch,
                           {'images': paths, 'cache': cache,
                            'cancel_file': sentinel, 'style_threshold': 0.6})
    assert data['cancelled'] is True
    assert data['clusters'] is None, 'a half partition is worse than none'
    assert len(data['results']) == 1, 'only what was actually computed'
    assert data['results'][paths[0]]['aesthetic'] == pytest.approx(7.25)
    assert data['cached'] == 1 and data['remaining'] == 3


# ---------------------------------------------------------------------------
# Parent side: what reaches the database.
# ---------------------------------------------------------------------------
def _flat(level=128):
    return Image.new('RGB', (128, 128), (level, level, level))


def _mkbank(client, tmp_path, files, name='B'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        os.makedirs(os.path.dirname(str(src / rel)), exist_ok=True)
        im.save(str(src / rel), 'JPEG', quality=92)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id']


def _fake_child(tmp_path, body, name='fake_score.py'):
    """A stand-in scoring child: reads the payload, writes ``body(req)``'s JSON.

    ``body`` is literal source pasted into the generated script (``req`` and
    ``images`` are in scope there), so a test can answer with results derived
    from the paths the parent really handed over. No eval anywhere.
    """
    script = tmp_path / name
    script.write_text(
        'import hashlib, json, sys\n'
        'req = json.loads(sys.stdin.read())\n'
        'images = req["images"]\n'
        'with open(req["cache"] + ".seen", "w", encoding="utf-8") as f:\n'
        '    json.dump(req, f)\n'
        f'{body}\n'
        '# Mirror the real child contract: every result is bound to the exact\n'
        '# bytes it measured, including synthetic results used by these tests.\n'
        'for path, result in (out.get("results") or {}).items():\n'
        '    if isinstance(result, dict) and "fingerprint" not in result:\n'
        '        try:\n'
        '            with open(path, "rb") as source:\n'
        '                result["fingerprint"] = hashlib.sha256(source.read()).hexdigest()\n'
        '        except OSError:\n'
        '            result["fingerprint"] = None\n'
        'print(json.dumps(out))\n', encoding='utf-8')
    return str(script)


def _rows(app, bank_id):
    with app.app_context():
        from app.models import BankImage
        return {os.path.basename(r.relpath): r
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}


@pytest.fixture
def scoring_available(monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_bank_scoring',
                        lambda: {'ok': True, 'detail': ''})
    monkeypatch.setattr(capabilities, 'bank_scoring_gpu_available', lambda: False)


def test_stopping_writes_the_computed_scores_and_leaves_style_alone(
        client, tmp_path, app, monkeypatch, scoring_available):
    """The defect this pass had: an interrupted run reached the disk cache and
    NOT a single row. The style ids must stay exactly as they were."""
    bank_id = _mkbank(client, tmp_path, {'a.jpg': _flat(30), 'b.jpg': _flat(60)})
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage, ImageBank
        from app.services import bank_transfer_metadata as transfer
        bank = db.session.get(ImageBank, bank_id)
        for r in BankImage.query.filter_by(bank_id=bank_id).all():
            r.style_cluster = 7
            # This is trusted current analysis. Without the exact-byte binding,
            # the first hardened pass must (correctly) treat the legacy style id
            # as stale rather than promise to preserve it.
            r.analysis_fingerprint = transfer.content_fingerprint_path(
                os.path.join(bank.source_path, r.relpath))
        db.session.commit()

    from app.services import image_bank_service as banks
    script = _fake_child(tmp_path,
                         'out = {"ok": True, "cancelled": True,\n'
                         '       "cached": 1, "remaining": 1,\n'
                         '       "computed": 1, "reused": 0,\n'
                         '       "results": {images[0]: {"state": "ok",\n'
                         '                               "aesthetic": 6.5,\n'
                         '                               "nsfw": 0.25}},\n'
                         '       "clusters": None}')
    monkeypatch.setattr(banks, '_SCORE_SCRIPT', script)
    assert client.post(f'/api/bank/{bank_id}/score', json={}).status_code == 202

    rows = _rows(app, bank_id)
    first = min(rows, key=lambda n: rows[n].id)
    assert rows[first].aesthetic_score == pytest.approx(6.5)
    assert rows[first].nsfw_score == pytest.approx(0.25)
    assert all(r.style_cluster == 7 for r in rows.values()), \
        'a stopped pass must not rewrite the style partition'
    other = next(n for n in rows if n != first)
    assert rows[other].aesthetic_score is None
    detail = client.get(f'/api/bank/{bank_id}').get_json()['activity']['detail']
    assert 'Stopped' in detail and '1 score(s) saved' in detail
    assert 'style groups need a full pass' in detail


def test_a_missing_head_never_blanks_the_scores_already_in_the_bank(
        client, tmp_path, app, monkeypatch, scoring_available):
    """A run where the aesthetic weights fail to download used to wipe every
    aesthetic score in the bank, because the write-back assigned res.get()."""
    bank_id = _mkbank(client, tmp_path, {'a.jpg': _flat(30)})
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage, ImageBank
        from app.services import bank_transfer_metadata as transfer
        bank = db.session.get(ImageBank, bank_id)
        row = BankImage.query.filter_by(bank_id=bank_id).one()
        row.aesthetic_score = 8.0
        row.nsfw_score = 0.1
        row.analysis_fingerprint = transfer.content_fingerprint_path(
            os.path.join(bank.source_path, row.relpath))
        db.session.commit()

    from app.services import image_bank_service as banks
    script = _fake_child(
        tmp_path,
        'out = {"ok": True, "computed": 1, "reused": 0,\n'
        '       "results": {p: {"state": "ok", "nsfw": 0.4} for p in images},\n'
        '       "clusters": {p: 1 for p in images}}')
    monkeypatch.setattr(banks, '_SCORE_SCRIPT', script)
    assert client.post(f'/api/bank/{bank_id}/score', json={}).status_code == 202

    row = _rows(app, bank_id)['a.jpg']
    assert row.aesthetic_score == pytest.approx(8.0), 'kept, not blanked'
    assert row.nsfw_score == pytest.approx(0.4)
    detail = client.get(f'/api/bank/{bank_id}').get_json()['activity']['detail']
    assert 'aesthetic head unavailable' in detail


def test_the_pool_is_the_whole_bank_minus_rejects_and_rescore_is_explicit(
        client, tmp_path, app, monkeypatch, scoring_available):
    bank_id = _mkbank(client, tmp_path, {'keep.jpg': _flat(30),
                                         'bad.jpg': _flat(60)})
    rows = _rows(app, bank_id)
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [rows['bad.jpg'].id], 'status': 'reject'})

    from app.services import image_bank_service as banks
    script = _fake_child(
        tmp_path,
        'out = {"ok": True, "computed": len(images), "reused": 0,\n'
        '       "results": {p: {"state": "ok", "aesthetic": 5.0, "nsfw": 0.2}\n'
        '                   for p in images},\n'
        '       "clusters": {p: 1 for p in images}}')
    monkeypatch.setattr(banks, '_SCORE_SCRIPT', script)
    cache_seen = None

    assert client.post(f'/api/bank/{bank_id}/score', json={}).status_code == 202
    with app.app_context():
        cache_seen = str(banks._score_cache_path(bank_id)) + '.seen'
    seen = json.loads(pathlib.Path(cache_seen).read_text(encoding='utf-8'))
    assert [os.path.basename(p) for p in seen['images']] == ['keep.jpg']
    assert seen['rescore'] is False
    rows = _rows(app, bank_id)
    assert rows['bad.jpg'].aesthetic_score is None      # a reject is never scored
    assert rows['keep.jpg'].aesthetic_score == pytest.approx(5.0)

    # The explicit lane reaches the child; the plain button never does.
    assert client.post(f'/api/bank/{bank_id}/score',
                       json={'rescore': True}).status_code == 202
    seen = json.loads(pathlib.Path(cache_seen).read_text(encoding='utf-8'))
    assert seen['rescore'] is True


def test_the_closing_line_separates_recomputed_from_reused(
        client, tmp_path, app, monkeypatch, scoring_available):
    """The auto-reject mistake in miniature: a total that looks complete while
    the run did almost nothing. Both numbers must be said."""
    bank_id = _mkbank(client, tmp_path, {'a.jpg': _flat(30), 'b.jpg': _flat(60),
                                         'c.jpg': _flat(90)})
    from app.services import image_bank_service as banks
    script = _fake_child(
        tmp_path,
        'out = {"ok": True, "computed": 1, "reused": 2,\n'
        '       "results": {p: {"state": "ok", "aesthetic": 5.0, "nsfw": 0.2}\n'
        '                   for p in images},\n'
        '       "clusters": {p: 1 for p in images}}')
    monkeypatch.setattr(banks, '_SCORE_SCRIPT', script)
    client.post(f'/api/bank/{bank_id}/score', json={})
    detail = client.get(f'/api/bank/{bank_id}').get_json()['activity']['detail']
    assert 'scored 3 image(s)' in detail
    assert '1 newly computed, 2 reused from cache' in detail


def test_a_pass_with_nothing_to_do_says_so_instead_of_borrowing_a_total(
        client, tmp_path, app, monkeypatch, scoring_available):
    """The one-click tunnel reads its count from the database when a step says
    nothing — so a step that did nothing must not stay silent."""
    bank_id = _mkbank(client, tmp_path, {'a.jpg': _flat(30)})
    rows = _rows(app, bank_id)
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [rows['a.jpg'].id], 'status': 'reject'})
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_SCORE_SCRIPT',
                        _fake_child(tmp_path, 'out = {"ok": True}', 'never.py'))
    assert client.post(f'/api/bank/{bank_id}/score', json={}).status_code == 202
    detail = client.get(f'/api/bank/{bank_id}').get_json()['activity']['detail']
    assert 'nothing to score' in detail
    assert 'scored' not in detail.replace('nothing to score', '')


def test_stopping_during_the_write_back_keeps_what_landed(
        client, tmp_path, app, monkeypatch, scoring_available):
    """Stop pressed after the child finished: the rows already written stay
    written, the partition is not touched, and the relaunch is cache-cheap."""
    bank_id = _mkbank(client, tmp_path, {f'i{i}.jpg': _flat(30 + i)
                                         for i in range(4)})
    from app.services import bank_jobs
    from app.services import image_bank_service as banks
    script = _fake_child(
        tmp_path,
        'out = {"ok": True, "computed": len(images), "reused": 0,\n'
        '       "results": {p: {"state": "ok", "aesthetic": 5.0, "nsfw": 0.2}\n'
        '                   for p in images},\n'
        '       "clusters": {p: 1 for p in images}}')
    monkeypatch.setattr(banks, '_SCORE_SCRIPT', script)
    monkeypatch.setattr(banks, '_SCORE_COMMIT_EVERY', 1)

    real_live = banks._live_image
    state = {'n': 0, 'job': None}

    def _live(image_id):
        state['n'] += 1
        if state['n'] == 2 and state['job'] is not None:
            bank_jobs.cancel(bank_id)
        return real_live(image_id)

    monkeypatch.setattr(banks, '_live_image', _live)
    real_start = bank_jobs.start

    def _start(app_, bid, kind, fn, total=0):
        def wrapped(job):
            state['job'] = job
            return fn(job)
        return real_start(app_, bid, kind, wrapped, total=total)

    monkeypatch.setattr(bank_jobs, 'start', _start)
    client.post(f'/api/bank/{bank_id}/score', json={})

    rows = _rows(app, bank_id)
    written = [r for r in rows.values() if r.aesthetic_score is not None]
    assert 0 < len(written) < 4, 'some rows saved, the rest left for the relaunch'
    assert all(r.style_cluster is None for r in rows.values()), \
        'an interrupted write must not publish a partial partition'
    detail = client.get(f'/api/bank/{bank_id}').get_json()['activity']['detail']
    assert 'Stopped while saving' in detail
    assert 'relaunch finishes from the cache' in detail


def test_a_head_that_failed_says_WHY_not_just_that_it_is_unavailable(
        client, tmp_path, app, monkeypatch, scoring_available):
    """A degraded pass named the head and stopped there: "(aesthetic + NSFW head
    unavailable)". Both heads fetch their weights over the network on first use —
    the LAION MLP from GitHub, the NSFW classifier from Hugging Face — so on an
    install whose container has no egress BOTH go down at once, the pass reports
    "done", every score is empty, and sorting by aesthetic stays greyed out with
    nothing to explain it. The child already knows the exact exception and logs
    it; it was simply never carried back. It now travels in the child's JSON and
    lands in the sentence the user actually reads.

    Reported by @_nofaceman (Discord): "it tells me it completed but the sort
    option remains greyed out"."""
    bank_id = _mkbank(client, tmp_path, {'a.jpg': _flat(30)})
    from app.services import image_bank_service as banks
    script = _fake_child(
        tmp_path,
        'out = {"ok": True, "computed": 1, "reused": 0,\n'
        '       "results": {p: {"state": "ok"} for p in images},\n'
        '       "clusters": {p: 1 for p in images},\n'
        '       "head_errors": {"aesthetic": "URLError: <urlopen error '
        '[Errno -3] Temporary failure in name resolution>",\n'
        '                       "nsfw": "OSError: We couldn\'t connect to '
        'huggingface.co"}}')
    monkeypatch.setattr(banks, '_SCORE_SCRIPT', script)
    assert client.post(f'/api/bank/{bank_id}/score', json={}).status_code == 202

    detail = client.get(f'/api/bank/{bank_id}').get_json()['activity']['detail']
    assert 'aesthetic + NSFW head unavailable' in detail, 'the old sentence stays'
    assert 'name resolution' in detail or 'huggingface.co' in detail, \
        'the cause the child already knew must reach the user'


def test_a_head_failure_with_no_reason_reported_still_reads_cleanly(
        client, tmp_path, app, monkeypatch, scoring_available):
    """An older child, or a head that failed without a message, must not produce
    a dangling "unavailable ()" — the sentence degrades to what it said before."""
    bank_id = _mkbank(client, tmp_path, {'a.jpg': _flat(30)})
    from app.services import image_bank_service as banks
    script = _fake_child(
        tmp_path,
        'out = {"ok": True, "computed": 1, "reused": 0,\n'
        '       "results": {p: {"state": "ok", "nsfw": 0.3} for p in images},\n'
        '       "clusters": {p: 1 for p in images}}')
    monkeypatch.setattr(banks, '_SCORE_SCRIPT', script)
    assert client.post(f'/api/bank/{bank_id}/score', json={}).status_code == 202

    detail = client.get(f'/api/bank/{bank_id}').get_json()['activity']['detail']
    assert 'aesthetic head unavailable' in detail
    assert '()' not in detail and 'unavailable —' not in detail
