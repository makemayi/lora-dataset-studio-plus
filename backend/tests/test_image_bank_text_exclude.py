"""🔤 Text search — pushing an unwanted trait DOWN the ranking.

CLIP has no negation. Measured on this exact model: on a photo of an astronaut
WEARING a helmet, "without a helmet" scored 0.217 against 0.212 for "with a
helmet" — the negation ranked HIGHER. And measured over 7,316 real captioned
bank images, "a photo of a woman without a bikini" returned 60% bikinis against
a 10.1% base rate. So "without" is not imprecise, it is INVERTED, and no amount
of prompt wording fixes it.

What the embedding space does support is arithmetic. So the excluded phrase is
encoded like the positive one and subtracted:

    score = sim(positive) − weight · sim(excluded)

which pushes matching images down instead of removing them. That distinction is
the whole contract and these tests exist to keep the code honest about it: an
excluded image STAYS in the pool and can still come back if it is otherwise the
best answer. Calibration of the default weight lives in
``image_bank_service.PUSH_DOWN_WEIGHT_DEFAULT``.

No real model is ever loaded: every phrase maps to a hand-built vector.
"""
import hashlib
import os

import pytest
from PIL import Image

np = pytest.importorskip('numpy')


def _mkbank(client, tmp_path, names, name='B'):
    src = tmp_path / 'src'
    for rel in names:
        os.makedirs(os.path.dirname(str(src / rel)), exist_ok=True)
        Image.new('RGB', (64, 64), (128, 128, 128)).save(str(src / rel))
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _emb(*coords):
    v = np.zeros(768, dtype='float32')
    v[:len(coords)] = coords
    v /= (np.linalg.norm(v) + 1e-8)
    return v


def _write_score_cache(app, bank_id, embs_by_name):
    with app.app_context():
        from app.models import BankImage
        from app.services import image_bank_service as banks
        bank = banks.get_bank(_uid(), bank_id)
        rows = {os.path.basename(r.relpath): r
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        paths, arr, sigs, hashes = [], [], [], []
        for nm, e in embs_by_name.items():
            r = rows[nm]
            p = banks.abs_image_path(bank, r)
            paths.append(p)
            arr.append(np.asarray(e, dtype='float32'))
            st = os.stat(p)
            sigs.append(f'{st.st_size}:{st.st_mtime_ns}')
            with open(p, 'rb') as fh:
                digest = hashlib.sha256(fh.read()).digest()
            hashes.append(np.frombuffer(digest, dtype='uint8'))
            r.analysis_fingerprint = digest.hex()
        path = banks._score_cache_path(bank_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path), paths=np.array(paths),
            states=np.array(['ok'] * len(paths)),
            aes=np.array([float('nan')] * len(paths), dtype='float32'),
            nsfw=np.array([float('nan')] * len(paths), dtype='float32'),
            embs=np.stack(arr).astype('float32'), sigs=np.array(sigs),
            hashes=np.stack(hashes).astype('uint8'))
        banks.db.session.commit()
        banks.reset_score_memo()


def _uid():
    from app.config import LOCAL_USER
    return LOCAL_USER


def _names_of(app, bank_id, ids):
    with app.app_context():
        from app.models import BankImage
        by_id = {r.id: os.path.basename(r.relpath)
                 for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        return [by_id[i] for i in ids]


def _encoder(monkeypatch, by_text, seen=None):
    """Map each phrase to its own vector — the point of every test here is that
    the POSITIVE and the EXCLUDED phrase pull in different directions."""
    from app.services import clip_text_encoder

    def _run(texts, **kwargs):
        if seen is not None:
            seen.extend(texts)
        return [np.asarray(by_text[t], dtype='float32') for t in texts]

    monkeypatch.setattr(clip_text_encoder, '_encode_uncached', _run)
    monkeypatch.setattr(clip_text_encoder, 'unavailable_reason', lambda: None)


# Axis 0 is "what you asked for", axis 1 is "what you asked to push down", and
# axis 2 is everything else in the picture. THREE axes, not two: on a unit
# sphere with only the first two, carrying less of the unwanted trait would
# force carrying more of the wanted one, and every ranking would be decided
# twice over by the same number. The third axis absorbs the norm so the two
# similarities can be set independently — which is the situation the feature
# actually faces.
WANT = _emb(1.0, 0.0, 0.0)
AVOID = _emb(0.0, 1.0, 0.0)
IMAGES = {                                 # (wanted, unwanted, other)
    'want_only.jpg':  _emb(0.90, 0.00, 0.4359),
    'want_avoid.jpg': _emb(0.85, 0.50, 0.1658),
    'mid.jpg':        _emb(0.80, 0.15, 0.5809),
    'avoid_only.jpg': _emb(0.10, 0.95, 0.2958),
}


def _bank(client, tmp_path, app, monkeypatch, seen=None):
    bank_id, _ = _mkbank(client, tmp_path, list(IMAGES))
    _write_score_cache(app, bank_id, IMAGES)
    _encoder(monkeypatch, {'a woman': WANT, 'a hat': AVOID,
                           'a woman -hat': WANT, 'hat': AVOID,
                           'a close-up of a woman': WANT}, seen)
    return bank_id


def _search(client, bank_id, **payload):
    payload.setdefault('n', 4)
    r = client.post(f'/api/bank/{bank_id}/search-text', json=payload)
    assert r.status_code == 200, r.get_json()
    return r.get_json()


# --- the push-down itself ------------------------------------------------------
def test_excluding_a_trait_reorders_the_ranking(client, tmp_path, app, monkeypatch):
    """Without the exclusion, the image that matches BOTH outranks the milder
    one; with it, the balance flips. Ordering only — nothing is removed."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    plain = _search(client, bank_id, query='a woman')
    assert _names_of(app, bank_id, plain['image_ids'])[:2] == \
        ['want_only.jpg', 'want_avoid.jpg']
    excluded = _search(client, bank_id, query='a woman', push_down='a hat')
    order = _names_of(app, bank_id, excluded['image_ids'])
    assert order[0] == 'want_only.jpg'
    assert order.index('mid.jpg') < order.index('want_avoid.jpg')


def test_an_excluded_image_is_pushed_down_NOT_removed(client, tmp_path, app,
                                                      monkeypatch):
    """The load-bearing promise. The UI says "pushes down, cannot guarantee
    absence" — so the pool must stay whole and every image must still come back
    when the caller asks for them all."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    body = _search(client, bank_id, query='a woman', push_down='a hat', n=4)
    assert body['pool'] == 4
    assert len(body['image_ids']) == 4
    assert 'avoid_only.jpg' in _names_of(app, bank_id, body['image_ids'])


def test_weight_zero_is_exactly_no_exclusion(client, tmp_path, app, monkeypatch):
    """The bottom of the scale has to be a true no-op, or "Gentle" would be a
    lie about a subtraction that still happened."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    plain = _search(client, bank_id, query='a woman')
    zero = _search(client, bank_id, query='a woman', push_down='a hat',
                   push_down_weight=0)
    assert plain['image_ids'] == zero['image_ids']


def test_a_stronger_weight_pushes_further(client, tmp_path, app, monkeypatch):
    """Monotone by construction: the more you pay, the lower the trait sinks."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    ranks = []
    for w in (0.0, 0.6, 2.0):
        body = _search(client, bank_id, query='a woman', push_down='a hat',
                       push_down_weight=w)
        order = _names_of(app, bank_id, body['image_ids'])
        ranks.append(order.index('want_avoid.jpg'))
    assert ranks == sorted(ranks), f'expected non-decreasing ranks, got {ranks}'
    assert ranks[-1] > ranks[0]


# --- the `-term` shorthand -----------------------------------------------------
def test_minus_term_in_the_query_means_the_same_as_the_field(client, tmp_path,
                                                             app, monkeypatch):
    """One grammar, two entry points — `-hat` must not be a second, subtly
    different feature."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    field = _search(client, bank_id, query='a woman', push_down='hat')
    dashed = _search(client, bank_id, query='a woman -hat')
    assert dashed['query'] == 'a woman'
    assert dashed['push_down'] == 'hat'
    assert dashed['image_ids'] == field['image_ids']


def test_an_inner_hyphen_is_not_an_exclusion(client, tmp_path, app, monkeypatch):
    """"close-up", "thigh-high", "2026-07" — a dash INSIDE a word is part of the
    word. Only a dash that opens a token is grammar, and getting this wrong
    would silently amputate ordinary photographic vocabulary."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    body = _search(client, bank_id, query='a close-up of a woman')
    assert body['query'] == 'a close-up of a woman'
    assert body['push_down'] is None


def test_the_field_and_the_dash_combine(client, tmp_path, app, monkeypatch):
    """Both filled = both excluded, comma-joined into the one phrase CLIP sees."""
    from app.services.clip_text_encoder import split_query
    assert split_query('a woman -hat -sunglasses') == ('a woman', 'hat, sunglasses')
    assert split_query('a woman') == ('a woman', '')
    assert split_query('  -hat  ') == ('', 'hat')
    # A doubled dash is a typo, not a request for the literal string "-hat".
    assert split_query('a woman --hat') == ('a woman', 'hat')
    # A dash that is its own token excludes nothing — there is no term after it.
    assert split_query('a woman - hat') == ('a woman - hat', '')


def test_the_word_filter_and_the_push_down_are_DIFFERENT_wires(client, tmp_path,
                                                               app, monkeypatch):
    """Two features in this workspace mean opposite things by "exclude": the
    grid's 🚫 Exclude box HIDES images whose caption carries a word (a hard
    filter, guaranteed absence), and this one re-ranks by meaning (guaranteed
    nothing). They ride in the SAME request body, because a text search composes
    with the current filter — so they cannot share a key.

    They did, briefly, and nothing failed: each feature's own tests passed while
    the grid filter silently overwrote whatever the user typed under Push down.
    ``exclude`` stays the filter's; the push-down is ``push_down``."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    plain = _search(client, bank_id, query='a woman', push_down='a hat')
    # The filter key present as well: it must scope the pool, never become the
    # phrase being pushed down.
    both = _search(client, bank_id, query='a woman', push_down='a hat',
                   exclude='some-word-no-caption-has')
    assert both['push_down'] == 'a hat', 'the filter must not hijack the phrase'
    assert both['image_ids'] == plain['image_ids']


def test_the_hard_filter_WINS_over_the_push_down_on_the_same_word(
        client, tmp_path, app, monkeypatch):
    """The seam between the two features, with the same word typed in both.

    🚫 Exclude promises an ABSENCE and Push down promises only an order, so when
    they disagree the strong promise has to win: an image the word filter hides
    must not come back merely because the ranking judged it a close match. That
    holds because the push-down ranks the pool the filter already produced —
    ``search_by_text`` builds its candidates from ``_pool_embeddings``, which
    goes through ``_pool_query(**filters)``. This test asserts the ABSENCE of
    the id, not its rank: 'last' would still be a broken promise."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    with app.app_context():
        from app.models import BankImage, db
        row = next(r for r in BankImage.query.filter_by(bank_id=bank_id).all()
                   if os.path.basename(r.relpath) == 'want_avoid.jpg')
        row.caption = 'a woman wearing a hat'
        db.session.commit()
        tagged_id = row.id

    # Ranked alone, it is a strong match and comes back.
    ranked = _search(client, bank_id, query='a woman', push_down='a hat', n=60)
    assert tagged_id in ranked['image_ids']

    both = _search(client, bank_id, query='a woman', push_down='a hat',
                   exclude='hat', n=60)
    assert tagged_id not in both['image_ids'], \
        'the hard filter promises absence; a re-ranking must not hand it back'
    assert both['pool'] == ranked['pool'] - 1, \
        'the filter must shrink the POOL, not just the returned page'


def test_an_exclusion_alone_is_refused(client, tmp_path, app, monkeypatch):
    """Ranking by "least like a hat" returns whatever is least like anything —
    noise in the costume of an answer. 400, with words."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    r = client.post(f'/api/bank/{bank_id}/search-text', json={'query': '-hat'})
    assert r.status_code == 400
    assert 'excluded term alone' in r.get_json()['error']


# --- saying what it did --------------------------------------------------------
def test_it_reports_how_many_results_the_exclusion_moved(client, tmp_path, app,
                                                         monkeypatch):
    """An exclusion that changed nothing looks identical to one that worked. The
    count is what lets the UI tell the two apart."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    body = _search(client, bank_id, query='a woman', push_down='a hat', n=2)
    assert body['push_down_moved'] == 1        # mid.jpg displaced want_avoid.jpg
    same = _search(client, bank_id, query='a woman', push_down='a hat',
                   push_down_weight=0, n=2)
    assert same['push_down_moved'] == 0


def test_a_reordered_whole_pool_is_NOT_reported_as_unchanged(client, tmp_path,
                                                             app, monkeypatch):
    """The count is over PLACES, not membership — and this is the case that
    proves why. The default n is 60 while a real bank being triaged is often
    smaller, so the request routinely covers the entire pool: the same images
    come back, reordered. Counting newcomers returns 0 there and the UI
    announces "this changed nothing" over a grid the user can SEE rearranged.
    Caught on the live proof instance, not in review."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    plain = _search(client, bank_id, query='a woman', n=60)
    body = _search(client, bank_id, query='a woman', push_down='a hat', n=60)
    # Every image comes back both times: membership CANNOT change here.
    assert sorted(plain['image_ids']) == sorted(body['image_ids'])
    assert plain['image_ids'] != body['image_ids'], 'the order must have changed'
    assert body['push_down_moved'] > 0, 'a reordering is not "nothing happened"'
    # And the median comparison is withheld rather than reported as a tautology:
    # over the whole pool, median(results) IS median(pool), and dressing that up
    # as "level with a typical image — too tangled to separate" is a verdict read
    # off an identity. Same bug, one statistic further along.
    assert body['push_down_median'] is None


def test_it_reports_the_excluded_match_against_the_pool_baseline(
        client, tmp_path, app, monkeypatch):
    """Measured, never assumed: how much of the excluded trait a TYPICAL image
    here carries, against how much the returned set carries. Below = it worked
    on this bank; level = it did not. No universal constant is involved."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    body = _search(client, bank_id, query='a woman', push_down='a hat', n=2)
    assert body['push_down_median']['results'] < body['push_down_median']['pool']
    assert body['push_down_weight'] == pytest.approx(0.6)


def test_no_exclusion_leaves_every_exclusion_field_empty(client, tmp_path, app,
                                                         monkeypatch):
    """A plain search must not start reporting a push-down it did not do."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    body = _search(client, bank_id, query='a woman')
    assert body['push_down'] is None
    assert body['push_down_weight'] is None
    assert body['push_down_moved'] is None
    assert body['push_down_median'] is None


def test_results_carry_both_halves_of_the_score(client, tmp_path, app, monkeypatch):
    """``score`` is what ordered the list (so the order never looks arbitrary),
    and both ingredients are shown next to it rather than hidden."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    top = _search(client, bank_id, query='a woman', push_down='a hat')['results'][0]
    assert top['score'] == pytest.approx(top['match'] - 0.6 * top['excluded_match'],
                                         abs=1e-3)


def test_the_range_stays_in_match_units(client, tmp_path, app, monkeypatch):
    """``score_range`` is read against ``pool_median`` to judge whether a ranking
    discriminates. A composite score is not on that scale — it can even go
    negative — so the range keeps reporting how well the returned set matches
    the WORDS, which is what the sentence above the grid claims."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    body = _search(client, bank_id, query='a woman', push_down='a hat', n=4)
    assert body['score_range']['top'] == pytest.approx(0.90, abs=1e-3)
    assert body['score_range']['bottom'] >= 0.0
    assert body['score_range']['bottom'] <= body['score_range']['top']


# --- cost ---------------------------------------------------------------------
def test_the_excluded_phrase_is_cached_like_any_other(client, tmp_path, app,
                                                      monkeypatch):
    """Both halves go through the same persistent cache, so a repeated exclusion
    is free. ``cached`` only claims "instant" when BOTH were already known."""
    seen = []
    bank_id = _bank(client, tmp_path, app, monkeypatch, seen)
    first = _search(client, bank_id, query='a woman', push_down='a hat')
    assert first['cached'] is False
    assert sorted(seen) == ['a hat', 'a woman']
    seen.clear()
    again = _search(client, bank_id, query='a woman', push_down='a hat')
    assert again['cached'] is True
    assert seen == [], 'a repeated exclusion must not wake the encoder'


def test_a_new_exclusion_on_a_cached_query_is_not_announced_as_instant(
        client, tmp_path, app, monkeypatch):
    """Half a cache hit is not a cache hit — promising instant and then loading
    CLIP for eight seconds is exactly how this reads as a freeze."""
    seen = []
    bank_id = _bank(client, tmp_path, app, monkeypatch, seen)
    assert _search(client, bank_id, query='a woman')['cached'] is False
    seen.clear()
    body = _search(client, bank_id, query='a woman', push_down='a hat')
    assert body['cached'] is False
    assert seen == ['a hat'], 'only the unknown half is encoded'


# --- the knob that is NOT offered ---------------------------------------------
def test_there_is_still_no_similarity_threshold(client, tmp_path, app, monkeypatch):
    """A weight scales a subtraction inside one ranking; a threshold would claim
    a relevance boundary the measurements say does not exist (correct hits
    0.177-0.233 overlapping unrelated pairs up to 0.197). Adding a weight must
    not have smuggled the other one in."""
    bank_id = _bank(client, tmp_path, app, monkeypatch)
    plain = _search(client, bank_id, query='a woman', push_down='a hat')
    cut = _search(client, bank_id, query='a woman', push_down='a hat',
                  min_score=0.9)
    assert plain['image_ids'] == cut['image_ids']
    assert len(cut['image_ids']) == 4


def test_a_nonsense_weight_falls_back_to_the_calibrated_default(
        client, tmp_path, app, monkeypatch):
    """Never to 0: an exclusion silently ignored looks exactly like an exclusion
    that found nothing, and the user would trust the wrong screen."""
    from app.services.image_bank_service import _push_down_weight, \
        PUSH_DOWN_WEIGHT_DEFAULT
    assert _push_down_weight('nonsense') == PUSH_DOWN_WEIGHT_DEFAULT
    assert _push_down_weight(None) == PUSH_DOWN_WEIGHT_DEFAULT
    assert _push_down_weight(float('nan')) == PUSH_DOWN_WEIGHT_DEFAULT
    assert _push_down_weight(-3) == 0.0
    assert _push_down_weight(99) == 2.0
    assert _push_down_weight('1.0') == 1.0
