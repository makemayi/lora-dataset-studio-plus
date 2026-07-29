"""🗃️ Curation selectors — what a click COSTS, and the promise that it still
returns the same images.

A 🎨 Pick diverse on a real 9 500-image pool took 32 seconds while the docstring
promised "~sub-second". Profiled on that bank, the split was:

    typicality guard (all-pairs E @ E.T)   12.6 s   89 %
    pool path resolution (realpath/row)     1.1 s    8 %
    score-cache parse (.npz + stat/row)     0.35 s   2 %
    the farthest-point loop itself          0.11 s   1 %

so all three fixes below are about the 99 % that was NOT the sampling. Every one
of them is an optimisation, which makes the interesting assertion in this file
not "is it faster" (a timing test is a flaky test) but **"does it still return
exactly the same selection"** — same ids, same order, deterministic. That is what
these tests pin.
"""
import os

import pytest
from PIL import Image

np = pytest.importorskip('numpy')


def _flat(size=64, value=128):
    return Image.new('RGB', (size, size), (value, value, value))


def _mkbank(client, tmp_path, names, name='B'):
    src = tmp_path / 'src'
    for rel in names:
        os.makedirs(os.path.dirname(str(src / rel)), exist_ok=True)
        _flat(value=(hash(rel) % 200) + 20).save(str(src / rel))
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _uid():
    from app.config import LOCAL_USER
    return LOCAL_USER


def _write_score_cache(app, bank_id, embs_by_name):
    with app.app_context():
        from app.models import BankImage
        from app.services import image_bank_service as banks
        bank = banks.get_bank(_uid(), bank_id)
        rows = {os.path.basename(r.relpath): r
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        paths, states, arr, sigs = [], [], [], []
        for nm, e in embs_by_name.items():
            r = rows[nm]
            p = banks.abs_image_path(bank, r)
            paths.append(p)
            states.append('ok')
            arr.append(np.asarray(e, dtype='float32'))
            st = os.stat(p)
            sigs.append(f'{st.st_size}:{st.st_mtime_ns}')
        cache_path = banks._score_cache_path(bank_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            paths=np.array(paths), states=np.array(states),
            aes=np.array([float('nan')] * len(paths), dtype='float32'),
            nsfw=np.array([float('nan')] * len(paths), dtype='float32'),
            embs=np.stack(arr).astype('float32'), sigs=np.array(sigs))


def _spread_bank(client, tmp_path, app, m=48, d=16, seed=7):
    """A pool big enough for the typicality guard to actually engage (its floor is
    _TYPICALITY_MIN_POOL = 32 rows), with a reproducible spread of embeddings."""
    rng = np.random.default_rng(seed)
    names = [f'i{i:03d}.jpg' for i in range(m)]
    embs = {}
    for i, nm in enumerate(names):
        v = np.zeros(768, dtype='float32')
        v[:d] = rng.standard_normal(d)
        if i % 11 == 0:                       # a few deliberate loners
            v[:d] *= 0.1
            v[d + (i % 5)] = 3.0
        v /= (np.linalg.norm(v) + 1e-8)
        embs[nm] = v
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, embs)
    return bank_id


# --- the guard rail: an optimisation may not move a single id ----------------
@pytest.mark.parametrize('n', [5, 20])
@pytest.mark.parametrize('typ', [0.25, 0.5, 1.0])
def test_optimised_blas_returns_the_same_selection_as_plain_numpy(
        client, tmp_path, app, n, typ):
    """THE test of this pass. The similarity block now goes through scipy's BLAS
    when one is reachable; a different BLAS sums the same products in a different
    order, so this asserts the thing that actually matters — that the selection is
    identical, id for id and in the same order, either way."""
    bank_id = _spread_bank(client, tmp_path, app)
    with app.app_context():
        from app.services import image_bank_service as banks
        fast = banks.select_diverse(_uid(), bank_id, n, typicality=typ)
        banks.reset_score_memo()
        saved = banks._BLAS_GEMM
        try:
            banks._BLAS_GEMM = None                  # force the numpy fallback
            plain = banks.select_diverse(_uid(), bank_id, n, typicality=typ)
        finally:
            banks._BLAS_GEMM = saved
    assert fast['image_ids'] == plain['image_ids']
    assert fast['pool'] == plain['pool']


def test_balanced_selection_is_backend_independent_too(client, tmp_path, app):
    """The balanced lane slices the SAME pool-wide penalty, so it inherits the
    guarantee — pinned rather than assumed."""
    bank_id = _spread_bank(client, tmp_path, app)
    with app.app_context():
        from app.models import BankImage
        from app.services import image_bank_service as banks
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(
            BankImage.id.asc()).all()
        for i, r in enumerate(rows):
            r.framing = ('face', 'bust', 'body', 'back')[i % 4]
        from app.extensions import db
        db.session.commit()
        fast = banks.select_balanced(_uid(), bank_id, 12)
        banks.reset_score_memo()
        saved = banks._BLAS_GEMM
        try:
            banks._BLAS_GEMM = None
            plain = banks.select_balanced(_uid(), bank_id, 12)
        finally:
            banks._BLAS_GEMM = saved
    assert fast['image_ids'] == plain['image_ids']
    assert [b['selected'] for b in fast['buckets']] == \
           [b['selected'] for b in plain['buckets']]


def test_selection_stays_deterministic_across_repeated_calls(client, tmp_path, app):
    """Three identical calls, including one served from the score memo, return the
    identical list — the memo must not reorder or drop anything."""
    bank_id = _spread_bank(client, tmp_path, app)
    with app.app_context():
        from app.services import image_bank_service as banks
        first = banks.select_diverse(_uid(), bank_id, 15)['image_ids']
        second = banks.select_diverse(_uid(), bank_id, 15)['image_ids']
        banks.reset_score_memo()
        third = banks.select_diverse(_uid(), bank_id, 15)['image_ids']
    assert first == second == third
    assert first == sorted(first)


# --- the three costs, each pinned by the work it must no longer redo ---------
def test_sim_block_uses_the_optimised_blas_when_one_is_reachable(app):
    """The similarity product must actually reach scipy's BLAS — the whole 90×
    lives there. Skipped (not failed) where scipy is genuinely absent."""
    from app.services import image_bank_service as banks
    banks._BLAS_GEMM = ...                              # force a fresh probe
    gemm = banks._fast_gemm_nt()
    if gemm is None:
        pytest.skip('no scipy BLAS in this environment')
    calls = []
    banks._BLAS_GEMM = lambda *a, **k: (calls.append(1), gemm(*a, **k))[1]
    try:
        E = np.eye(4, 8, dtype='float32')
        out = banks._sim_block(E[:2], E)
    finally:
        banks._BLAS_GEMM = gemm
    assert calls, '_sim_block bypassed the optimised BLAS'
    assert out.shape == (2, 4)
    assert out.flags['C_CONTIGUOUS'], 'partition(axis=1) needs C-order rows'
    assert np.allclose(out, E[:2] @ E.T, atol=1e-5)


def test_sim_block_falls_back_to_numpy_without_scipy(app):
    """No scipy (or a wheel that refuses the call) must degrade, never raise."""
    from app.services import image_bank_service as banks
    saved = banks._BLAS_GEMM
    try:
        banks._BLAS_GEMM = None
        E = np.eye(4, 8, dtype='float32')
        assert np.array_equal(banks._sim_block(E[:2], E), E[:2] @ E.T)
        banks._BLAS_GEMM = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('nope'))
        assert np.array_equal(banks._sim_block(E[:2], E), E[:2] @ E.T)
    finally:
        banks._BLAS_GEMM = saved


def test_pool_resolution_does_not_realpath_the_bank_folder_per_row(
        client, tmp_path, app, monkeypatch):
    """``os.path.realpath`` is a filesystem call and the bank folder does not
    change between two rows of the same pool. Resolving it per row cost 424 ms of
    a 6 353-row pool; this pins "once for the base, once per image", not twice per
    image."""
    bank_id = _spread_bank(client, tmp_path, app, m=40)
    with app.app_context():
        from app.services import image_bank_service as banks
        bank = banks.get_bank(_uid(), bank_id)
        emb = banks._load_score_embeddings(bank)
        real = os.path.realpath
        calls = []
        monkeypatch.setattr(os.path, 'realpath',
                            lambda p: (calls.append(p), real(p))[1])
        ids, _E = banks._pool_embeddings(bank, emb, {})
    assert len(ids) == 40
    assert len(calls) <= len(ids) + 1, (
        f'{len(calls)} realpath calls for {len(ids)} rows — the bank folder is '
        f'being re-resolved per row')


def test_score_cache_is_parsed_once_for_two_consecutive_selections(
        client, tmp_path, app, monkeypatch):
    """Two clicks on an unchanged bank must not parse the same 40 MB .npz twice."""
    bank_id = _spread_bank(client, tmp_path, app)
    with app.app_context():
        from app.services import image_bank_service as banks
        banks.reset_score_memo()
        loads = []
        real_load = np.load
        monkeypatch.setattr(np, 'load',
                            lambda *a, **k: (loads.append(a[0]), real_load(*a, **k))[1])
        banks.select_diverse(_uid(), bank_id, 10)
        banks.select_diverse(_uid(), bank_id, 10)
    assert len(loads) == 1, f'score cache parsed {len(loads)}× for two selections'


def test_score_memo_is_dropped_when_the_scoring_pass_rewrites_the_cache(
        client, tmp_path, app):
    """The memo is keyed on the .npz's own size+mtime, so a finished ✨ Score pass
    invalidates it without anyone remembering to — a stale embedding set would
    silently curate on images that are no longer there."""
    bank_id = _spread_bank(client, tmp_path, app, m=40, seed=3)
    with app.app_context():
        from app.models import BankImage
        from app.services import image_bank_service as banks
        bank = banks.get_bank(_uid(), bank_id)
        before = len(banks._load_score_embeddings(bank))
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(
            BankImage.id.asc()).all()
        keep = {os.path.basename(r.relpath) for r in rows[:10]}
    _write_score_cache(app, bank_id, {
        nm: np.eye(1, 768, dtype='float32')[0] for nm in sorted(keep)})
    with app.app_context():
        from app.services import image_bank_service as banks
        bank = banks.get_bank(_uid(), bank_id)
        after = len(banks._load_score_embeddings(bank))
    assert before == 40
    assert after == 10, 'the rewritten score cache was served from a stale memo'
