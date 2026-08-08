"""🗃️ Bank coverage — the two axes the LABELS cannot see.

The framing/cluster/resolution numbers are labels. They cannot tell two hundred
near-identical shots from two hundred different ones, and they say nothing about
outfits, lighting or camera angle. Two sources already on disk can:

  • the CAPTIONS from the 🏷️ pass, read by the SHARED `caption_coverage` lexicon
    (the same module the dataset panel uses — imported, never copied);
  • the CLIP embeddings from ✨ Score, for whether the pool actually LOOKS varied.

Neither runs a model here. Embeddings are seeded as a real score_cache.npz
exactly as the scoring subprocess would write one, so the loader's staleness and
state checks are exercised for real rather than mocked away.
"""
import hashlib
import os

import pytest
from PIL import Image

np = pytest.importorskip('numpy')


def _flat(size=64, value=128):
    return Image.new('RGB', (size, size), (value, value, value))


def _uid():
    from app.config import LOCAL_USER
    return LOCAL_USER


def _mkbank(client, tmp_path, names, name='B'):
    src = tmp_path / 'src'
    for rel in names:
        os.makedirs(os.path.dirname(str(src / rel)), exist_ok=True)
        _flat(value=(hash(rel) % 200) + 20).save(str(src / rel))
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


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


def _set_captions(app, bank_id, caption_for):
    with app.app_context():
        from app.models import BankImage
        from app.services import image_bank_service as banks
        for r in BankImage.query.filter_by(bank_id=bank_id).all():
            r.caption = caption_for(os.path.basename(r.relpath))
            r.status = 'keep'
        banks.db.session.commit()


def _vec(rng, spread):
    """A 768-dim L2-normed vector: a shared direction plus `spread` × noise.

    The resulting mean pairwise cosine is ≈ 1/(1 + 768·spread²) — the dimension
    is in there, which is why "a small number" is not automatically a tight pool:
    spread 0.02 already lands at 0.77, comfortably OUTSIDE the redundant band.
    Small spread ⇒ everything alike; large ⇒ independent directions.
    """
    base = np.zeros(768, dtype='float32')
    base[0] = 1.0
    noise = rng.normal(size=768).astype('float32')
    v = base + spread * noise
    return v / (np.linalg.norm(v) + 1e-8)


NAMES = [f'i{i:02d}.png' for i in range(14)]


# --- the caption lexicon, wired to the bank ---------------------------------

def test_bank_coverage_reports_caption_variety_from_the_shared_lexicon(app, client, tmp_path):
    bank_id, _ = _mkbank(client, tmp_path, NAMES)
    _set_captions(app, bank_id, lambda nm:
                  'a woman facing the camera, studio lighting, plain background, '
                  'wearing a white t-shirt, smiling')
    p = client.get(f'/api/bank/{bank_id}/coverage').get_json()

    assert p['variety']['captioned'] == len(NAMES)
    # Nested on purpose: `analyse` has its own `total`, and flattening would
    # overwrite the bank pool size.
    assert p['total'] == len(NAMES)
    assert p['variety']['total'] == len(NAMES)
    text = ' | '.join(a['text'] for a in p['advice'])
    assert 'profile' in text and 'three-quarter' in text
    assert 'outfit' in text.lower()


def test_bank_coverage_says_when_there_are_no_captions_to_read(app, client, tmp_path):
    bank_id, _ = _mkbank(client, tmp_path, NAMES)
    p = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    assert p['variety']['captioned'] == 0
    text = ' | '.join(a['text'] for a in p['advice'])
    assert 'No captions yet' in text
    # Surface-neutral wording: this sentence is shared with the dataset panel,
    # so it must not name the dataset's composition bar.
    assert 'composition' not in text


def test_bank_is_judged_as_a_character_source(app, client, tmp_path):
    """The bank has no kind. Every other axis of this panel already assumes a
    character (12/6/6/1 framing, "one consistent subject"), so the caption axes
    match it rather than introducing a stored per-bank preference."""
    from app.services import image_bank_service as banks
    from app.services import caption_coverage as cc
    bank_id, _ = _mkbank(client, tmp_path, NAMES)
    _set_captions(app, bank_id, lambda nm: 'a woman facing the camera, smiling')
    p = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    assert banks._COVERAGE_KIND == 'character'
    assert p['variety']['kind'] == 'character'
    assert [a['id'] for a in p['variety']['axes']] == cc.axes_for_kind('character')


# --- visual spread, from the cached embeddings ------------------------------

def test_a_repetitive_pool_is_named_as_such(app, client, tmp_path):
    bank_id, _ = _mkbank(client, tmp_path, NAMES)
    _set_captions(app, bank_id, lambda nm: 'a woman facing the camera')
    rng = np.random.default_rng(7)
    _write_score_cache(app, bank_id, {nm: _vec(rng, 0.008) for nm in NAMES})

    p = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    v = p['visual']
    assert v['scored'] == len(NAMES)
    assert v['similarity'] > 0.9 and v['band'] == 'redundant'
    warns = [a['text'] for a in p['advice'] if a['tone'] == 'warn']
    assert any('look very alike' in t for t in warns)


def test_a_varied_pool_says_nothing_rather_than_padding_the_panel(app, client, tmp_path):
    bank_id, _ = _mkbank(client, tmp_path, NAMES)
    _set_captions(app, bank_id, lambda nm: 'a woman facing the camera')
    rng = np.random.default_rng(3)
    _write_score_cache(app, bank_id, {nm: _vec(rng, 60.0) for nm in NAMES})

    p = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    assert p['visual']['band'] == 'varied'
    assert not any('alike' in a['text'] for a in p['advice'])


def test_without_the_score_pass_it_says_unmeasured_not_varied(app, client, tmp_path):
    """The failure this guards: reporting a set as varied because nothing looked."""
    bank_id, _ = _mkbank(client, tmp_path, NAMES)
    _set_captions(app, bank_id, lambda nm: 'a woman facing the camera')
    p = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    v = p['visual']
    assert v['scored'] == 0 and v['similarity'] is None and v['band'] == 'unknown'
    text = ' | '.join(a['text'] for a in p['advice'])
    assert 'Run ✨ Score' in text
    assert not any('alike' in a['text'] for a in p['advice'])


def test_a_tiny_pool_declines_to_judge_its_spread(app, client, tmp_path):
    few = NAMES[:4]
    bank_id, _ = _mkbank(client, tmp_path, few)
    rng = np.random.default_rng(1)
    _write_score_cache(app, bank_id, {nm: _vec(rng, 0.008) for nm in few})
    with app.app_context():
        from app.models import BankImage
        from app.services import image_bank_service as banks
        for r in BankImage.query.filter_by(bank_id=bank_id).all():
            r.status = 'keep'
        banks.db.session.commit()

    p = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    assert p['visual']['similarity'] is None      # measured nothing, claims nothing
    assert not any('alike' in a['text'] for a in p['advice'])


def test_a_corrupt_embedding_cache_degrades_instead_of_breaking_the_panel(app, client, tmp_path, monkeypatch):
    bank_id, _ = _mkbank(client, tmp_path, NAMES)
    _set_captions(app, bank_id, lambda nm: 'a woman facing the camera')

    from app.services import image_bank_service as banks

    def boom(_bank):
        raise ValueError('cache went bad mid-read')
    monkeypatch.setattr(banks, '_load_score_embeddings', boom)

    r = client.get(f'/api/bank/{bank_id}/coverage')
    assert r.status_code == 200
    assert r.get_json()['visual']['band'] == 'unknown'


def test_the_mean_similarity_is_the_exact_pairwise_mean(app, client, tmp_path):
    """The closed form (||sum||^2 - n)/(n^2-n) must equal the brute-force mean —
    it is the whole reason a 47k-image bank costs one pass instead of a matrix."""
    from app.services import image_bank_service as banks
    bank_id, _ = _mkbank(client, tmp_path, NAMES)
    rng = np.random.default_rng(11)
    embs = {nm: _vec(rng, 1.5) for nm in NAMES}
    _write_score_cache(app, bank_id, embs)
    with app.app_context():
        from app.models import BankImage
        for r in BankImage.query.filter_by(bank_id=bank_id).all():
            r.status = 'keep'
        banks.db.session.commit()
        bank = banks.get_bank(_uid(), bank_id)
        from app.models import BankImage as BI
        got = banks._visual_spread(bank, BI.status == 'keep')

    E = np.stack([embs[nm] for nm in NAMES]).astype('float32')
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    sims = E @ E.T
    n = len(E)
    brute = (sims.sum() - n) / (n * n - n)
    assert got['similarity'] == pytest.approx(float(brute), abs=1e-4)


def test_the_stats_dict_stays_json_serialisable(app, client, tmp_path):
    """The pool criterion is shared between the label counts and the two new
    reads. Passing it through the stats dict would have worked — right up until
    someone jsonify()'d it without stripping it, which is a 500 in production and
    green in every unit test that only reads keys it knows about."""
    import json
    from app.services import image_bank_service as banks
    bank_id, _ = _mkbank(client, tmp_path, NAMES)
    with app.app_context():
        stats = banks._coverage_stats(bank_id)
    json.dumps(stats)                      # raises TypeError if a criterion leaks
    assert not [k for k in stats if k.startswith('_')]


def test_rejected_images_are_out_of_both_new_reads(app, client, tmp_path):
    """Same pool as the rest of the panel — a rejected image must not drag the
    similarity or contribute a caption."""
    from app.services import image_bank_service as banks
    bank_id, _ = _mkbank(client, tmp_path, NAMES)
    rng = np.random.default_rng(5)
    _write_score_cache(app, bank_id, {nm: _vec(rng, 1.0) for nm in NAMES})
    with app.app_context():
        from app.models import BankImage
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(BankImage.id).all()
        for r in rows:
            r.status, r.caption = 'keep', 'a woman facing the camera'
        for r in rows[:4]:
            r.status, r.caption = 'reject', 'a woman in profile outdoors at night'
        banks.db.session.commit()

    p = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    assert p['total'] == len(NAMES) - 4
    assert p['variety']['total'] == len(NAMES) - 4
    assert p['visual']['scored'] == len(NAMES) - 4
    views = {b['id']: b['count'] for b in p['variety']['axes'][0]['buckets']}
    assert views['profile'] == 0        # the rejected captions were not read
