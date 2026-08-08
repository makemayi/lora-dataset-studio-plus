"""🗃️ Image bank — CURATION selectors (turn a big dump into a good LoRA subset).

Two pure-CPU selectors that reuse the CLIP embeddings the ✨ Score pass cached
(no GPU, no re-scan — same contract as the semantic-dedup stage):

  • select-diverse  — farthest-point sampling: the N images that best COVER the
                      visual space (the antidote to 4 000 near-identical shots);
  • select-similar  — rank by cosine similarity to a REFERENCE bank image
                      ("keep what looks like THIS").

Both only ever return a SELECTION (a set of image ids the UI checks) — nothing
is mutated or deleted. We seed a synthetic score_cache.npz (as the scoring
subprocess would) so grouping runs WITHOUT torch. Background jobs run inline
under TESTING; these selectors are synchronous.
"""
import hashlib
import os

import pytest
from PIL import Image

np = pytest.importorskip('numpy')


# --- factories (mirror the semantic-dedup suite) -----------------------------
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


def _emb(*coords):
    """A 768-dim L2-normed vector with the given leading coordinates."""
    v = np.zeros(768, dtype='float32')
    v[:len(coords)] = coords
    v /= (np.linalg.norm(v) + 1e-8)
    return v


def _write_score_cache(app, bank_id, embs_by_name, state='ok'):
    with app.app_context():
        from app.models import BankImage
        from app.services import image_bank_service as banks
        bank = banks.get_bank(_uid(), bank_id)
        rows = {os.path.basename(r.relpath): r
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        paths, states, arr, sigs, hashes = [], [], [], [], []
        for nm, e in embs_by_name.items():
            r = rows[nm]
            p = banks.abs_image_path(bank, r)
            paths.append(p)
            states.append(state)
            arr.append(np.asarray(e, dtype='float32'))
            st = os.stat(p)
            sigs.append(f'{st.st_size}:{st.st_mtime_ns}')
            with open(p, 'rb') as fh:
                digest = hashlib.sha256(fh.read()).digest()
            hashes.append(np.frombuffer(digest, dtype='uint8'))
            r.analysis_fingerprint = digest.hex()
        cache_path = banks._score_cache_path(bank_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            paths=np.array(paths), states=np.array(states),
            aes=np.array([float('nan')] * len(paths), dtype='float32'),
            nsfw=np.array([float('nan')] * len(paths), dtype='float32'),
            embs=np.stack(arr).astype('float32'), sigs=np.array(sigs),
            hashes=np.stack(hashes).astype('uint8'))
        banks.db.session.commit()
        banks.reset_score_memo()


def _id_of(app, bank_id, name):
    with app.app_context():
        from app.models import BankImage
        rows = {os.path.basename(r.relpath): r.id
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        return rows[name]


def _names_of(app, bank_id, ids):
    with app.app_context():
        from app.models import BankImage
        by_id = {r.id: os.path.basename(r.relpath)
                 for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        return {by_id[i] for i in ids}


# --- diversity (farthest-point sampling) -------------------------------------
def test_diverse_picks_one_per_cluster(client, tmp_path, app):
    """4 tight clusters of near-identical embeddings; asking for 4 diverse images
    picks ONE from each cluster — never 4 near-duplicates of one cluster."""
    names = []
    embs = {}
    # cluster c along axis c, 3 members each with a tiny wobble.
    for c in range(4):
        for k in range(3):
            nm = f'c{c}_{k}.jpg'
            names.append(nm)
            base = [0.0, 0.0, 0.0, 0.0]
            base[c] = 1.0
            base[(c + 1) % 4] = 0.02 * k        # small intra-cluster wobble
            embs[nm] = _emb(*base)
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, embs)
    r = client.post(f'/api/bank/{bank_id}/select-diverse', json={'n': 4})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body['pool'] == 12 and len(body['image_ids']) == 4
    picked = _names_of(app, bank_id, body['image_ids'])
    clusters = {nm.split('_')[0] for nm in picked}
    assert clusters == {'c0', 'c1', 'c2', 'c3'}     # one per cluster, full coverage


def test_diverse_is_deterministic(client, tmp_path, app):
    """Same pool + same n ⇒ byte-identical selection (lowest-id seed, id tie-break)."""
    names = [f'i{k}.jpg' for k in range(8)]
    embs = {nm: _emb(*np.random.RandomState(k).randn(5)) for k, nm in enumerate(names)}
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, embs)
    a = client.post(f'/api/bank/{bank_id}/select-diverse', json={'n': 5}).get_json()
    b = client.post(f'/api/bank/{bank_id}/select-diverse', json={'n': 5}).get_json()
    assert a['image_ids'] == b['image_ids']
    assert a['image_ids'] == sorted(a['image_ids'])   # returned sorted


def test_diverse_whole_pool_when_n_exceeds(client, tmp_path, app):
    names = ['a.jpg', 'b.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0), 'b.jpg': _emb(0.0, 1.0)})
    body = client.post(f'/api/bank/{bank_id}/select-diverse',
                       json={'n': 50}).get_json()
    assert body['pool'] == 2 and len(body['image_ids']) == 2


def test_diverse_hint_when_no_embeddings(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    r = client.post(f'/api/bank/{bank_id}/select-diverse', json={'n': 4})
    assert r.status_code == 400
    assert 'Score first' in r.get_json()['error']


def test_diverse_composes_with_filter_and_excludes_rejects(client, tmp_path, app):
    """The pool honours the grid filter (here a style_cluster) AND, with no status
    filter, drops rejected rows — you curate from what you might keep."""
    names = [f'i{k}.jpg' for k in range(6)]
    embs = {nm: _emb(*np.random.RandomState(k + 100).randn(5)) for k, nm in enumerate(names)}
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, embs)
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        rows = sorted(BankImage.query.filter_by(bank_id=bank_id).all(), key=lambda r: r.id)
        rows[0].style_cluster = 1                 # only i0..i2 in style 1
        rows[1].style_cluster = 1
        rows[2].style_cluster = 1
        rows[2].status = 'reject'                 # i2 rejected → out of the pool
        db.session.commit()
    body = client.post(f'/api/bank/{bank_id}/select-diverse',
                       json={'n': 10, 'style': 1}).get_json()
    assert body['pool'] == 2                       # i0, i1 (i2 rejected)
    picked = _names_of(app, bank_id, body['image_ids'])
    assert picked == {'i0.jpg', 'i1.jpg'}


# --- diversity: the typicality guard -----------------------------------------
# Pure farthest-point sampling maximises the distance to what is already chosen,
# which is EXACTLY the criterion that prefers isolated points: on a collected
# bank its first picks are the aberrations (a meme, someone else's photo, a
# botched frame), not the variety of the subject. `typicality` discounts that
# isolation; 0 must restore the historical selection pick for pick.

def _cloud_with_outliers():
    """A realistically shaped pool: ONE subject (a shared component every shot has)
    seen in 4 different settings — 32 images, ~0.75 cosine across settings, ~1.0
    within one — plus 3 aberrations on their own axes, isolated from the cloud AND
    from each other (a meme, a screenshot, someone else's photo)."""
    embs = {}
    for c in range(4):
        for j in range(8):
            v = [0.0] * 9
            v[0] = 0.87                        # "the subject", in every shot
            v[1 + c] = 0.5                     # the setting of this cluster
            v[7] = 0.04 * j                    # tiny intra-cluster wobble
            v[8] = 0.02 * ((j * 7) % 5)
            embs[f'c{c}_{j}.jpg'] = _emb(*v)
    for x in range(3):
        v = [0.0] * (10 + x)
        v[9 + x] = 1.0                         # orthogonal to everything else
        embs[f'x{x}.jpg'] = _emb(*v)
    return embs


def _outlier_bank(client, tmp_path, app):
    embs = _cloud_with_outliers()
    bank_id, _ = _mkbank(client, tmp_path, list(embs))
    _write_score_cache(app, bank_id, embs)
    return bank_id


def test_diverse_without_guard_picks_the_aberrations_first(client, tmp_path, app):
    """RED, kept as documentation of the flaw: with the guard OFF (typicality=0,
    i.e. the historical behaviour) asking for 4 "most diverse" spends THREE QUARTERS of the
    budget on the 3 planted aberrations — they win purely on isolation."""
    bank_id = _outlier_bank(client, tmp_path, app)
    body = client.post(f'/api/bank/{bank_id}/select-diverse',
                       json={'n': 4, 'typicality': 0}).get_json()
    picked = _names_of(app, bank_id, body['image_ids'])
    assert {'x0.jpg', 'x1.jpg', 'x2.jpg'} <= picked      # ALL THREE, picks 2-4
    assert body['typicality'] == 0.0


def test_diverse_guard_keeps_the_aberrations_out(client, tmp_path, app):
    """GREEN: the default guard drops every aberration and spends the whole budget
    on the subject — while still COVERING it (one image per cluster at least)."""
    bank_id = _outlier_bank(client, tmp_path, app)
    body = client.post(f'/api/bank/{bank_id}/select-diverse',
                       json={'n': 4}).get_json()
    picked = _names_of(app, bank_id, body['image_ids'])
    assert not (picked & {'x0.jpg', 'x1.jpg', 'x2.jpg'})
    assert {nm.split('_')[0] for nm in picked} == {'c0', 'c1', 'c2', 'c3'}
    assert body['typicality'] == pytest.approx(0.5)      # the shipped default


def test_diverse_guard_does_not_collapse_into_look_alikes(client, tmp_path, app):
    """The opposite trap: a guard cranked to the maximum must NOT return N images
    from the middle of the cloud. Rows at or above the median density are never
    penalised, so coverage of the subject survives even at typicality=1."""
    bank_id = _outlier_bank(client, tmp_path, app)
    body = client.post(f'/api/bank/{bank_id}/select-diverse',
                       json={'n': 4, 'typicality': 1}).get_json()
    picked = _names_of(app, bank_id, body['image_ids'])
    assert {nm.split('_')[0] for nm in picked} == {'c0', 'c1', 'c2', 'c3'}


def _legacy_fps(E, n, np_):
    """The pre-guard implementation, verbatim, as an independent oracle for the
    compatibility contract below (an assertion against the new code comparing it
    to itself would prove nothing)."""
    chosen = [0]
    min_dist = 1.0 - E @ E[0]
    min_dist[0] = -1.0
    for _ in range(n - 1):
        nxt = int(np_.argmax(min_dist))
        if min_dist[nxt] <= -1.0:
            break
        chosen.append(nxt)
        min_dist = np_.minimum(min_dist, 1.0 - E @ E[nxt])
        min_dist[nxt] = -1.0
    return chosen


def test_diverse_typicality_zero_is_bit_for_bit_the_old_selection(client, tmp_path, app):
    """THE GOLDEN TEST. typicality=0 must reproduce the historical farthest-point
    selection EXACTLY — same rows, same order of picking — on a pool with clusters,
    outliers and near-ties. This is the anti-regression guard of the whole change."""
    embs = dict(_cloud_with_outliers())
    rs = np.random.RandomState(7)
    for k in range(20):                        # plus unstructured noise rows
        embs[f'r{k:02d}.jpg'] = _emb(*rs.randn(9))
    bank_id, _ = _mkbank(client, tmp_path, list(embs))
    _write_score_cache(app, bank_id, embs)
    with app.app_context():
        from app.services import image_bank_service as banks
        bank = banks.get_bank(_uid(), bank_id)
        ids, E = banks._pool_embeddings(bank, banks._load_score_embeddings(bank), {})
        for n in (2, 5, 12, 30):
            expect = [ids[i] for i in _legacy_fps(E, n, np)]
            got = banks.select_diverse(_uid(), bank_id, n=n, typicality=0)
            assert got['image_ids'] == sorted(expect), f'n={n}'
            # …and the guard, when on, is a DIFFERENT answer (else nothing changed)
            assert banks.select_diverse(_uid(), bank_id, n=6)['image_ids'] \
                != sorted([ids[i] for i in _legacy_fps(E, 6, np)])


def test_diverse_stays_deterministic_with_the_guard(client, tmp_path, app):
    """Same pool + same n + same typicality ⇒ the same selection, every time —
    otherwise two datasets curated from one bank stop being comparable."""
    bank_id = _outlier_bank(client, tmp_path, app)
    runs = [client.post(f'/api/bank/{bank_id}/select-diverse',
                        json={'n': 7, 'typicality': 0.45}).get_json()['image_ids']
            for _ in range(3)]
    assert runs[0] == runs[1] == runs[2] == sorted(runs[0])


def test_isolation_penalty_never_materialises_a_full_matrix(app):
    """Cost guard: a 24 000-image bank would need ~2.3 GB for a full E @ E.T
    (24000² × 4 bytes). The density pass must stay blocked — here every matmul it
    performs is watched, and none may be pool-sized."""
    from app.services import image_bank_service as banks
    seen = []
    # Every similarity this pass computes goes through the ONE chokepoint
    # `_sim_block` (numpy or an optimised BLAS underneath — the memory shape is
    # the same guarantee either way), so watching it watches everything.
    real_sim = banks._sim_block

    def watched(A, E):
        out = real_sim(A, E)
        seen.append(out.shape)
        return out

    m = banks._TYPICALITY_BLOCK * 3 + 7        # several blocks + a short tail
    rs = np.random.RandomState(3)
    E = rs.randn(m, 16).astype('float32')
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    banks._sim_block = watched
    try:
        pen = banks._isolation_penalty(E)
    finally:
        banks._sim_block = real_sim
    assert pen.shape == (m,) and float(pen.min()) == 0.0
    assert seen, 'the density pass did compute similarities'
    assert all(s[0] <= banks._TYPICALITY_BLOCK for s in seen), seen
    assert max(s[0] * s[1] for s in seen) <= banks._TYPICALITY_BLOCK * m
    # the same penalties, block size aside — blocking is an implementation detail
    ref = banks._isolation_penalty(E, block=m)
    assert np.allclose(pen, ref, atol=1e-6)


# --- reference similarity ----------------------------------------------------
def test_similar_ranks_by_cosine_to_reference(client, tmp_path, app):
    """Top-N most similar to the reference are its near-neighbours; the reference
    itself is always included (cosine 1.0), the far image is excluded at N=2."""
    names = ['ref.jpg', 'near.jpg', 'mid.jpg', 'far.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {
        'ref.jpg':  _emb(1.0, 0.0),
        'near.jpg': _emb(0.98, np.sqrt(1 - 0.98 ** 2)),
        'mid.jpg':  _emb(0.80, np.sqrt(1 - 0.80 ** 2)),
        'far.jpg':  _emb(0.10, np.sqrt(1 - 0.10 ** 2))})
    ref_id = _id_of(app, bank_id, 'ref.jpg')
    body = client.post(f'/api/bank/{bank_id}/select-similar',
                       json={'ref_id': ref_id, 'n': 2}).get_json()
    assert _names_of(app, bank_id, body['image_ids']) == {'ref.jpg', 'near.jpg'}
    # results are score-ranked, reference first at cosine 1.0.
    assert body['results'][0]['id'] == ref_id
    assert body['results'][0]['score'] == pytest.approx(1.0, abs=1e-3)
    assert body['results'][1]['score'] > body['results'][-1]['score'] \
        if len(body['results']) > 2 else True


def test_similar_threshold_mode(client, tmp_path, app):
    """min_score keeps everything at/above the cosine cut, whatever the count."""
    names = ['ref.jpg', 'near.jpg', 'far.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {
        'ref.jpg':  _emb(1.0, 0.0),
        'near.jpg': _emb(0.95, np.sqrt(1 - 0.95 ** 2)),
        'far.jpg':  _emb(0.50, np.sqrt(1 - 0.50 ** 2))})
    ref_id = _id_of(app, bank_id, 'ref.jpg')
    body = client.post(f'/api/bank/{bank_id}/select-similar',
                       json={'ref_id': ref_id, 'min_score': 0.9}).get_json()
    assert _names_of(app, bank_id, body['image_ids']) == {'ref.jpg', 'near.jpg'}


def test_similar_requires_ref_id(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0)})
    r = client.post(f'/api/bank/{bank_id}/select-similar', json={'n': 4})
    assert r.status_code == 400
    assert 'ref_id' in r.get_json()['error']


def test_similar_hint_when_no_embeddings(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
    ref_id = _id_of(app, bank_id, 'a.jpg')
    r = client.post(f'/api/bank/{bank_id}/select-similar',
                    json={'ref_id': ref_id, 'n': 2})
    assert r.status_code == 400
    assert 'Score first' in r.get_json()['error']


def test_similar_ref_without_embedding_is_400(client, tmp_path, app):
    """A reference that has no cached embedding (e.g. never scored) gets a clear
    error, not a silent empty selection."""
    names = ['ref.jpg', 'a.jpg', 'b.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    # Cache covers a/b but NOT ref.
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0, 0.0),
                                      'b.jpg': _emb(0.0, 1.0)})
    ref_id = _id_of(app, bank_id, 'ref.jpg')
    r = client.post(f'/api/bank/{bank_id}/select-similar',
                    json={'ref_id': ref_id, 'n': 2})
    assert r.status_code == 400
    assert 'embedding' in r.get_json()['error']


# --- "show selected" grid view (?ids=…) --------------------------------------
# The curation selectors return a SELECTION scattered across the bank; the grid
# must be able to render exactly those ids, in the order given, so the result is
# actually visible (and a similarity ranking reads closest→farthest).
def test_ids_view_preserves_given_order(client, tmp_path, app):
    names = [f'i{k}.jpg' for k in range(5)]
    bank_id, _ = _mkbank(client, tmp_path, names)
    a, b, c = (_id_of(app, bank_id, n) for n in ('i0.jpg', 'i2.jpg', 'i4.jpg'))
    r = client.get(f'/api/bank/{bank_id}/images?ids={c},{a},{b}')
    assert r.status_code == 200
    body = r.get_json()
    assert body['total'] == 3
    assert [im['id'] for im in body['images']] == [c, a, b]     # exact order kept


def test_ids_view_overrides_facets_and_drops_unknown(client, tmp_path, app):
    """The id list is the scope: it ignores status/flag facets, and ids that
    aren't in this bank (or don't exist) are silently dropped."""
    names = [f'i{k}.jpg' for k in range(4)]
    bank_id, _ = _mkbank(client, tmp_path, names)
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        rows = sorted(BankImage.query.filter_by(bank_id=bank_id).all(), key=lambda r: r.id)
        rows[0].status = 'reject'                  # i0 rejected — must still show in the view
        db.session.commit()
    i0, i1 = (_id_of(app, bank_id, n) for n in ('i0.jpg', 'i1.jpg'))
    # status=keep would normally hide i0/i1; the id view ignores it. 999999 is unknown.
    r = client.get(f'/api/bank/{bank_id}/images?status=keep&ids={i1},999999,{i0}')
    body = r.get_json()
    assert [im['id'] for im in body['images']] == [i1, i0]      # facet ignored, unknown dropped
    assert body['total'] == 2


def test_ids_view_paginates_in_order(client, tmp_path, app):
    names = [f'i{k}.jpg' for k in range(6)]
    bank_id, _ = _mkbank(client, tmp_path, names)
    order = [_id_of(app, bank_id, f'i{k}.jpg') for k in (5, 4, 3, 2, 1, 0)]
    csv = ','.join(str(i) for i in order)
    p1 = client.get(f'/api/bank/{bank_id}/images?ids={csv}&offset=0&limit=2').get_json()
    p2 = client.get(f'/api/bank/{bank_id}/images?ids={csv}&offset=2&limit=2').get_json()
    assert p1['total'] == 6 and p2['total'] == 6
    assert [im['id'] for im in p1['images']] == order[:2]
    assert [im['id'] for im in p2['images']] == order[2:4]


def test_similar_result_feeds_ids_view_end_to_end(client, tmp_path, app):
    """The whole fix: select-similar returns ranked ids, and feeding them back as
    the ?ids= view renders the grid reference-first, closest→farthest."""
    names = ['ref.jpg', 'near.jpg', 'mid.jpg', 'far.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {
        'ref.jpg':  _emb(1.0, 0.0),
        'near.jpg': _emb(0.98, np.sqrt(1 - 0.98 ** 2)),
        'mid.jpg':  _emb(0.80, np.sqrt(1 - 0.80 ** 2)),
        'far.jpg':  _emb(0.10, np.sqrt(1 - 0.10 ** 2))})
    ref_id = _id_of(app, bank_id, 'ref.jpg')
    sel = client.post(f'/api/bank/{bank_id}/select-similar',
                      json={'ref_id': ref_id, 'n': 4}).get_json()
    ranked = sel['image_ids']
    assert ranked[0] == ref_id                     # reference first (cosine 1.0)
    csv = ','.join(str(i) for i in ranked)
    grid = client.get(f'/api/bank/{bank_id}/images?ids={csv}').get_json()
    assert [im['id'] for im in grid['images']] == ranked   # grid mirrors the ranking


# --- the two selectors are DISTINCT, and similarity is REFERENCE-sensitive ---
# Regression guard for the reported symptom "🎨 Pick diverse and 🎯 Similar to
# selected show EXACTLY the same thing, whatever the reference". Every other test
# here exercises one selector in isolation; none pins that the two DISAGREE on a
# shared pool, nor that select-similar actually follows its ref_id. A pool whose
# embeddings collapsed to one vector (or a selector that ignored the cache / the
# reference) would silently pass the isolated tests yet return identical,
# ref-insensitive selections — exactly the failure this locks out. Two disjoint
# neighbour clusters (around refA and refB) plus two off-axis outliers.
def _two_cluster_bank(client, tmp_path, app):
    embs = {
        'a0.jpg': _emb(1.0, 0.0),                       # refA cluster …
        'a1.jpg': _emb(0.96, np.sqrt(1 - 0.96 ** 2)),
        'a2.jpg': _emb(0.92, np.sqrt(1 - 0.92 ** 2)),
        'b0.jpg': _emb(0.0, 1.0),                       # refB cluster (orthogonal) …
        'b1.jpg': _emb(np.sqrt(1 - 0.96 ** 2), 0.96),
        'b2.jpg': _emb(np.sqrt(1 - 0.92 ** 2), 0.92),
        'x0.jpg': _emb(0.0, 0.0, 1.0),                  # far outliers on their own axes
        'x1.jpg': _emb(0.0, 0.0, 0.0, 1.0),
    }
    bank_id, _ = _mkbank(client, tmp_path, list(embs))
    _write_score_cache(app, bank_id, embs)
    return bank_id


def test_similar_is_reference_sensitive(client, tmp_path, app):
    """Two DIFFERENT references over the same pool ⇒ two DIFFERENT selections,
    each led by its own reference — the core of "regardless of the reference"."""
    bank_id = _two_cluster_bank(client, tmp_path, app)
    a0, b0 = (_id_of(app, bank_id, n) for n in ('a0.jpg', 'b0.jpg'))
    sa = client.post(f'/api/bank/{bank_id}/select-similar',
                     json={'ref_id': a0, 'n': 3}).get_json()
    sb = client.post(f'/api/bank/{bank_id}/select-similar',
                     json={'ref_id': b0, 'n': 3}).get_json()
    assert sa['results'][0]['id'] == a0 and sb['results'][0]['id'] == b0   # ref first
    assert sa['image_ids'] != sb['image_ids']                             # NOT identical
    # Each ref pulls its own orthogonal cluster — the two selections are disjoint.
    assert set(sa['image_ids']).isdisjoint(sb['image_ids'])
    assert _names_of(app, bank_id, sa['image_ids']) == {'a0.jpg', 'a1.jpg', 'a2.jpg'}
    assert _names_of(app, bank_id, sb['image_ids']) == {'b0.jpg', 'b1.jpg', 'b2.jpg'}


def test_diverse_differs_from_similar_on_same_pool(client, tmp_path, app):
    """Diversity coverage ≠ reference similarity on the same real-shaped pool: FPS
    reaches the far outliers a near-neighbour ranking never would, so the two
    selectors return genuinely different sets (they must never coincide)."""
    bank_id = _two_cluster_bank(client, tmp_path, app)
    a0 = _id_of(app, bank_id, 'a0.jpg')
    div = client.post(f'/api/bank/{bank_id}/select-diverse',
                      json={'n': 3}).get_json()
    sim = client.post(f'/api/bank/{bank_id}/select-similar',
                      json={'ref_id': a0, 'n': 3}).get_json()
    assert sorted(div['image_ids']) != sorted(sim['image_ids'])           # NOT identical
    # Diversity reaches the outliers; similarity-to-a0 stays inside a0's cluster.
    div_names = _names_of(app, bank_id, div['image_ids'])
    assert div_names & {'x0.jpg', 'x1.jpg'}                               # coverage grabs a far point
    assert _names_of(app, bank_id, sim['image_ids']) == {'a0.jpg', 'a1.jpg', 'a2.jpg'}


# --- balance: coverage of the LABELS, not of the embedding space --------------
# `select_diverse` answers "is my set varied?"; `select_balanced` answers "does my
# set COVER what I want to generate?". Different questions, and on a lopsided bank
# the first one returns the bank's own proportions — 8 body shots for 1 back view
# — because framing is only a faint direction in CLIP space: two shots of the same
# scene sit closer together than two body shots of different scenes. That is the
# shape these fixtures reproduce.

_MIX = ('body',) * 8 + ('bust',) * 6 + ('face',) * 2 + ('back',)   # per scene
_FRAMING_AX = {'face': 0, 'bust': 1, 'body': 2, 'back': 3}


def _set_framings(app, bank_id, by_name):
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        for r in BankImage.query.filter_by(bank_id=bank_id).all():
            fr = by_name.get(os.path.basename(r.relpath))
            if fr is not None:
                r.framing = fr
        db.session.commit()


def _framing_hist(app, bank_id, ids):
    with app.app_context():
        from app.models import BankImage
        by_id = {r.id: r.framing
                 for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        h = {k: 0 for k in ('face', 'bust', 'body', 'back')}
        for i in ids:
            if by_id.get(i) in h:
                h[by_id[i]] += 1
        return h


def _lopsided_bank(client, tmp_path, app, mix=_MIX, scenes=5, outlier=None):
    """One subject, ``scenes`` settings, framings distributed inside each scene in
    a realistically lopsided mix. The SCENE dominates the embedding (0.4) and the
    framing barely shows (0.05) — the measured shape of a mono-subject bank, where
    everything is 0.6–0.9 similar and the label carries information the pixels
    hardly separate. ``outlier`` adds one image orthogonal to the whole bank
    (a meme, a screenshot) carrying that framing label."""
    embs, framings = {}, {}
    for s in range(scenes):
        for j, fr in enumerate(mix):
            v = [0.0] * 12
            v[0] = 0.9                     # the subject, in every shot
            v[1 + s] = 0.4                 # the scene — the dominant variation
            v[6 + _FRAMING_AX[fr]] = 0.05  # the framing — faint, as measured
            v[10] = 0.01 * j               # intra-scene wobble
            v[11] = 0.004 * ((j * 7) % 5)
            nm = f's{s}_{j:02d}_{fr}.jpg'
            embs[nm] = _emb(*v)
            framings[nm] = fr
    if outlier:
        v = [0.0] * 16
        v[15] = 1.0                        # orthogonal to everything above
        embs['zz_outlier.jpg'] = _emb(*v)
        framings['zz_outlier.jpg'] = outlier
    names = sorted(embs)
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {k: embs[k] for k in names})
    _set_framings(app, bank_id, framings)
    return bank_id


def test_diverse_alone_returns_the_banks_own_imbalance(client, tmp_path, app):
    """RED, kept as the reason this feature exists: on the lopsided bank, asking
    the DIVERSE selector for 20 images returns the bank's proportions — the rare
    framing stays rare. Nothing is wrong with that answer; it just isn't the
    answer to "cover my framings", which is why a second selector exists."""
    bank_id = _lopsided_bank(client, tmp_path, app)
    body = client.post(f'/api/bank/{bank_id}/select-diverse',
                       json={'n': 20}).get_json()
    h = _framing_hist(app, bank_id, body['image_ids'])
    assert sum(h.values()) == 20
    assert max(h.values()) - min(h.values()) >= 4     # visibly lopsided
    assert h['back'] < 5 or h['face'] < 5             # the thin framings stay thin


def test_balanced_spreads_over_the_framings(client, tmp_path, app):
    """THE CENTRAL ASSERTION. Same bank, same 20 images asked for: an even split
    over the four framings, 5 / 5 / 5 / 5 — while `select_diverse` above returns
    the bank's own lopsided mix."""
    bank_id = _lopsided_bank(client, tmp_path, app)
    r = client.post(f'/api/bank/{bank_id}/select-balanced', json={'n': 20})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body['axis'] == 'framing' and body['selected'] == 20
    assert _framing_hist(app, bank_id, body['image_ids']) == \
        {'face': 5, 'bust': 5, 'body': 5, 'back': 5}
    # …and it SAYS so, per bucket, in plain numbers (what the panel reads out).
    got = {b['framing']: (b['selected'], b['fair_share'], b['short'])
           for b in body['buckets']}
    assert got == {'face': (5, 5, False), 'bust': (5, 5, False),
                   'body': (5, 5, False), 'back': (5, 5, False)}
    assert body['unlabelled'] == 0 and body['shortfall'] == 0


def test_balanced_reports_an_axis_it_cannot_satisfy(client, tmp_path, app):
    """The impossible case: only 2 back views exist but an even split wants 5.
    The bucket is filled to the brim, the deficit is REPORTED (selected < fair
    share, short=True) and the freed picks go to the framings that have room —
    never silently, and never as a smaller set than was asked for."""
    mix = ('body',) * 9 + ('bust',) * 6 + ('face',) * 2     # no back in the mix
    bank_id = _lopsided_bank(client, tmp_path, app, mix=mix)
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        rows = sorted(BankImage.query.filter_by(bank_id=bank_id).all(),
                      key=lambda r: r.id)
        for r in rows[:2]:
            r.framing = 'back'                              # exactly 2 back views
        db.session.commit()
    body = client.post(f'/api/bank/{bank_id}/select-balanced',
                       json={'n': 20}).get_json()
    back = next(b for b in body['buckets'] if b['framing'] == 'back')
    assert (back['available'], back['selected'], back['fair_share'],
            back['short']) == (2, 2, 5, True)
    assert body['selected'] == 20                # topped up, not shrunk…
    assert sum(b['selected'] for b in body['buckets']) == 20
    assert [b['framing'] for b in body['buckets'] if b['short']] == ['back']  # …and named


def test_balanced_is_deterministic(client, tmp_path, app):
    bank_id = _lopsided_bank(client, tmp_path, app)
    a = client.post(f'/api/bank/{bank_id}/select-balanced', json={'n': 13}).get_json()
    b = client.post(f'/api/bank/{bank_id}/select-balanced', json={'n': 13}).get_json()
    assert a['image_ids'] == b['image_ids'] == sorted(a['image_ids'])
    assert a['buckets'] == b['buckets']


def test_balanced_keeps_the_typicality_guard(client, tmp_path, app):
    """ANTI-REGRESSION on the guard `select_diverse` was just given, now inside a
    bucket: an image isolated from the WHOLE bank must stop winning on isolation
    alone. Guard off picks the aberration first (max-min distance IS the criterion
    that prefers isolated points); guard on leaves it out and takes real back
    views instead. The bucket has room to choose — a bucket so thin that every
    member is needed keeps the aberration, and rightly says so as a shortfall."""
    mix = ('body',) * 8 + ('bust',) * 6 + ('face',) * 2 + ('back',) * 3
    bank_id = _lopsided_bank(client, tmp_path, app, mix=mix, outlier='back')
    odd_id = _id_of(app, bank_id, 'zz_outlier.jpg')
    off = client.post(f'/api/bank/{bank_id}/select-balanced',
                      json={'n': 20, 'typicality': 0}).get_json()
    on = client.post(f'/api/bank/{bank_id}/select-balanced',
                     json={'n': 20, 'typicality': 1}).get_json()
    assert off['typicality'] == 0.0 and on['typicality'] == 1.0
    assert len(off['image_ids']) == len(on['image_ids']) == 20
    assert odd_id in off['image_ids']          # pure max-min loves the aberration
    assert odd_id not in on['image_ids']       # …the guard does not


def test_balanced_says_which_pass_is_missing_on_an_unlabelled_bank(client, tmp_path, app):
    """A bank nobody has classified is the DEFAULT state, not an error: no empty
    selection, no misleading one — the exact missing pass, with the numbers."""
    names = [f'i{k}.jpg' for k in range(6)]
    embs = {nm: _emb(*np.random.RandomState(k + 7).randn(5)) for k, nm in enumerate(names)}
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, embs)
    r = client.post(f'/api/bank/{bank_id}/select-balanced', json={'n': 4})
    assert r.status_code == 400
    err = r.get_json()['error']
    assert 'Framing pass' in err and '6' in err
    assert 'Pick diverse' in err              # the selector that still works


def test_balanced_hint_when_no_embeddings(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    r = client.post(f'/api/bank/{bank_id}/select-balanced', json={'n': 4})
    assert r.status_code == 400 and 'Score first' in r.get_json()['error']


def test_balanced_person_axis_is_opt_in(client, tmp_path, app):
    """The person axis crosses framing with face_cluster. Measured on a real bank
    it is sparse (4.7% of rows, 561 clusters) — hence opt-in, never the default."""
    bank_id = _lopsided_bank(client, tmp_path, app, scenes=3)
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        rows = sorted(BankImage.query.filter_by(bank_id=bank_id).all(),
                      key=lambda r: r.id)
        for k, r in enumerate(rows):
            r.face_cluster = 1 if k % 2 == 0 else 2
        db.session.commit()
    body = client.post(f'/api/bank/{bank_id}/select-balanced',
                       json={'n': 16, 'axis': 'framing+person'}).get_json()
    assert body['axis'] == 'framing+person'
    assert {b['cluster'] for b in body['buckets']} == {1, 2}
    assert len(body['buckets']) == 8                     # 4 framings × 2 people
