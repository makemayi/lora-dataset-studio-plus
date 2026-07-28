"""🗃️ Image bank — TEXT search ("brunette outdoors, wide shot" → ranked images).

Same engine as the two curation selectors: the CLIP embeddings the ✨ Score pass
already cached. The ONE new ingredient is a query VECTOR, produced by running
CLIP's text tower once in the ML interpreter (torch/open_clip are not in the
Flask venv). Ranking itself is pure numpy in-process — identical to
``select_similar``, only the reference vector comes from words instead of a
picture.

What these tests pin:
  * ranking is by cosine, descending, DETERMINISTIC (stable id tie-break);
  * an unscored bank says "run ✨ Score first" instead of returning nothing;
  * images WITHOUT an embedding are counted and reported, never silently
    dropped — otherwise the user concludes their image is gone;
  * no ML interpreter ⇒ an announced 503 "unavailable", not a raw traceback;
  * an identical query is served from the persistent cache — the encoder is
    invoked ONCE.

No real model is ever loaded: the encoder is monkeypatched everywhere.
"""
import json
import os

import pytest
from PIL import Image

np = pytest.importorskip('numpy')


# --- factories (mirror the curation suite) -----------------------------------
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
        paths, states, arr, sigs = [], [], [], []
        for nm, e in embs_by_name.items():
            r = rows[nm]
            p = banks.abs_image_path(bank, r)
            paths.append(p)
            states.append(state)
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


def _names_of(app, bank_id, ids):
    with app.app_context():
        from app.models import BankImage
        by_id = {r.id: os.path.basename(r.relpath)
                 for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        return {by_id[i] for i in ids}


def _fake_encoder(monkeypatch, vector, counter=None):
    """Stand in for the CLIP text tower: every query maps to ``vector``. Counts
    the calls so the cache can be proven to actually spare the subprocess."""
    from app.services import clip_text_encoder

    def _run(texts, **kwargs):
        if counter is not None:
            counter.append(list(texts))
        return [np.asarray(vector, dtype='float32') for _ in texts]

    monkeypatch.setattr(clip_text_encoder, '_encode_uncached', _run)
    monkeypatch.setattr(clip_text_encoder, 'unavailable_reason', lambda: None)


# --- ranking ------------------------------------------------------------------
def test_text_search_ranks_by_cosine_to_the_query(client, tmp_path, app, monkeypatch):
    """The query vector points at axis 0; the images rank nearest-first and the
    far one falls out at n=2."""
    names = ['hit.jpg', 'near.jpg', 'mid.jpg', 'far.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {
        'hit.jpg':  _emb(1.0, 0.0),
        'near.jpg': _emb(0.98, np.sqrt(1 - 0.98 ** 2)),
        'mid.jpg':  _emb(0.80, np.sqrt(1 - 0.80 ** 2)),
        'far.jpg':  _emb(0.10, np.sqrt(1 - 0.10 ** 2))})
    _fake_encoder(monkeypatch, _emb(1.0, 0.0))
    r = client.post(f'/api/bank/{bank_id}/search-text',
                    json={'query': 'brunette outdoors, wide shot', 'n': 2})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert _names_of(app, bank_id, body['image_ids']) == {'hit.jpg', 'near.jpg'}
    # Scores are exposed: this is a RANKING, and the UI has to be able to say so.
    assert body['results'][0]['score'] == pytest.approx(1.0, abs=1e-3)
    assert body['results'][0]['score'] > body['results'][1]['score']
    assert body['pool'] == 4
    assert body['query'] == 'brunette outdoors, wide shot'


def test_text_search_is_deterministic(client, tmp_path, app, monkeypatch):
    """Same bank + same query ⇒ byte-identical ranking, run after run."""
    names = [f'i{k}.jpg' for k in range(8)]
    embs = {nm: _emb(*np.random.RandomState(k).randn(5)) for k, nm in enumerate(names)}
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, embs)
    _fake_encoder(monkeypatch, _emb(0.3, 0.7, -0.2))
    a = client.post(f'/api/bank/{bank_id}/search-text',
                    json={'query': 'a woman', 'n': 5}).get_json()
    b = client.post(f'/api/bank/{bank_id}/search-text',
                    json={'query': 'a woman', 'n': 5}).get_json()
    assert a['image_ids'] == b['image_ids']
    assert [x['score'] for x in a['results']] == [x['score'] for x in b['results']]
    # Descending, so the grid reads best-first.
    scores = [x['score'] for x in a['results']]
    assert scores == sorted(scores, reverse=True)


def test_text_search_ties_break_on_id(client, tmp_path, app, monkeypatch):
    """Identical embeddings ⇒ identical cosines; the order must still be stable
    and id-ascending rather than whatever argsort felt like."""
    names = [f'i{k}.jpg' for k in range(5)]
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {nm: _emb(1.0, 0.0) for nm in names})
    _fake_encoder(monkeypatch, _emb(1.0, 0.0))
    body = client.post(f'/api/bank/{bank_id}/search-text',
                       json={'query': 'x', 'n': 5}).get_json()
    assert body['image_ids'] == sorted(body['image_ids'])


def test_text_search_offers_NO_similarity_threshold(client, tmp_path, app, monkeypatch):
    """Deliberate absence, locked by a test so it is not "helpfully" added back.

    Measured on a real bank with this exact model: verified-correct top-1 hits
    scored 0.177-0.233 while guaranteed-unrelated pairs reached 0.197 (median
    0.112). The distributions OVERLAP — the unrelated ceiling beats two correct
    answers — so no cut separates relevant from irrelevant. A min_score control
    would be a knob over a boundary that does not exist, and would quietly drop
    true matches. Passing one must therefore change NOTHING."""
    names = ['a.jpg', 'b.jpg', 'c.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {
        'a.jpg': _emb(1.0, 0.0),
        'b.jpg': _emb(0.95, np.sqrt(1 - 0.95 ** 2)),
        'c.jpg': _emb(0.50, np.sqrt(1 - 0.50 ** 2))})
    _fake_encoder(monkeypatch, _emb(1.0, 0.0))
    plain = client.post(f'/api/bank/{bank_id}/search-text',
                        json={'query': 'x', 'n': 3}).get_json()
    with_cut = client.post(f'/api/bank/{bank_id}/search-text',
                           json={'query': 'x', 'n': 3, 'min_score': 0.9}).get_json()
    assert plain['image_ids'] == with_cut['image_ids'] == sorted(plain['image_ids'])
    assert len(plain['image_ids']) == 3, 'every ranked image comes back; nothing is cut'


def test_text_search_reports_the_pool_median(client, tmp_path, app, monkeypatch):
    """The baseline that makes "does this ranking discriminate?" answerable
    WITHOUT any hard-coded band: what a typical image of this bank scores for
    this query. Measured per bank, per query — because a single-subject bank
    compresses the discriminating gap by 30-70%, and that is the normal case."""
    names = ['hit.jpg', 'mid.jpg', 'far.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {
        'hit.jpg': _emb(1.0, 0.0),
        'mid.jpg': _emb(0.60, np.sqrt(1 - 0.60 ** 2)),
        'far.jpg': _emb(0.20, np.sqrt(1 - 0.20 ** 2))})
    _fake_encoder(monkeypatch, _emb(1.0, 0.0))
    body = client.post(f'/api/bank/{bank_id}/search-text',
                       json={'query': 'x', 'n': 3}).get_json()
    # Median of {1.0, 0.60, 0.20} = 0.60, over the WHOLE pool, not the top-N.
    assert body['pool_median'] == pytest.approx(0.60, abs=1e-3)
    assert body['score_range']['top'] == pytest.approx(1.0, abs=1e-3)


def test_text_search_composes_with_the_current_filter(client, tmp_path, app, monkeypatch):
    """Search runs INSIDE the current facets (and drops rejects) — it refines the
    grid instead of replacing it."""
    names = [f'i{k}.jpg' for k in range(6)]
    embs = {nm: _emb(*np.random.RandomState(k + 100).randn(5)) for k, nm in enumerate(names)}
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, embs)
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        rows = sorted(BankImage.query.filter_by(bank_id=bank_id).all(), key=lambda r: r.id)
        rows[0].style_cluster = 1
        rows[1].style_cluster = 1
        rows[2].style_cluster = 1
        rows[2].status = 'reject'          # rejected → out of the pool
        db.session.commit()
    _fake_encoder(monkeypatch, _emb(1.0))
    body = client.post(f'/api/bank/{bank_id}/search-text',
                       json={'query': 'x', 'n': 10, 'style': 1}).get_json()
    assert body['pool'] == 2
    assert _names_of(app, bank_id, body['image_ids']) == {'i0.jpg', 'i1.jpg'}


# --- honesty: what CANNOT be found -------------------------------------------
def test_text_search_hint_when_nothing_is_scored(client, tmp_path, app, monkeypatch):
    """A bank that never ran ✨ Score has no embeddings at all — say so, loudly.
    An empty result list here would read as "your images don't match"."""
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
    _fake_encoder(monkeypatch, _emb(1.0))
    r = client.post(f'/api/bank/{bank_id}/search-text', json={'query': 'a woman'})
    assert r.status_code == 400
    assert 'Score first' in r.get_json()['error']


def test_text_search_reports_images_without_an_embedding(client, tmp_path, app, monkeypatch):
    """Half the bank is scored, half isn't. The unscored half CANNOT be found by
    text — the response must count it so the UI can say "3 of 5 images in this
    filter have no ✨ Score embedding and were not searched"."""
    names = ['s0.jpg', 's1.jpg', 'u0.jpg', 'u1.jpg', 'u2.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {'s0.jpg': _emb(1.0, 0.0),
                                      's1.jpg': _emb(0.9, 0.1)})
    _fake_encoder(monkeypatch, _emb(1.0, 0.0))
    body = client.post(f'/api/bank/{bank_id}/search-text',
                       json={'query': 'x', 'n': 10}).get_json()
    assert body['pool'] == 2            # only the scored two were searchable
    assert body['unscored'] == 3        # …and the other three are REPORTED
    assert body['filtered'] == 5        # out of this many in the current filter


def test_text_search_rejects_an_empty_query(client, tmp_path, app, monkeypatch):
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0)})
    _fake_encoder(monkeypatch, _emb(1.0))
    r = client.post(f'/api/bank/{bank_id}/search-text', json={'query': '   '})
    assert r.status_code == 400
    assert 'query' in r.get_json()['error'].lower()


# --- degradation: no ML interpreter ------------------------------------------
def test_text_search_without_ml_python_is_announced_not_a_traceback(
        client, tmp_path, app, monkeypatch):
    """The Flask venv has no torch. On an install where NO interpreter can run
    CLIP either, text search is simply UNAVAILABLE — and must say so in words,
    with a 503 and a pointer to the fix. Never a 500, never a raw exception."""
    from app.services import clip_text_encoder
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0)})
    monkeypatch.setattr(clip_text_encoder, 'unavailable_reason',
                        lambda: 'open_clip is not installed in the ✨ Score interpreter')

    def _boom(texts, **kwargs):
        raise AssertionError('the encoder must not be invoked when unavailable')
    monkeypatch.setattr(clip_text_encoder, '_encode_uncached', _boom)

    r = client.post(f'/api/bank/{bank_id}/search-text', json={'query': 'a woman'})
    assert r.status_code == 503
    body = r.get_json()
    assert body['reason'] == 'encoder_unavailable'
    assert 'open_clip' in body['error']
    assert 'Traceback' not in body['error']


def test_text_search_surfaces_an_encoder_failure_in_words(
        client, tmp_path, app, monkeypatch):
    """The interpreter looked fine but the run failed (weights download blocked,
    OOM…). Still a 503 with the child's own message, not a 500."""
    from app.services import clip_text_encoder
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0)})
    monkeypatch.setattr(clip_text_encoder, 'unavailable_reason', lambda: None)

    def _fail(texts, **kwargs):
        raise clip_text_encoder.TextEncodeError('CLIP load failed: no route to host')
    monkeypatch.setattr(clip_text_encoder, '_encode_uncached', _fail)

    r = client.post(f'/api/bank/{bank_id}/search-text', json={'query': 'a woman'})
    assert r.status_code == 503
    assert 'no route to host' in r.get_json()['error']


# --- the latency arbitration: a repeated query costs NOTHING -------------------
def test_repeated_query_is_served_from_the_persistent_cache(
        client, tmp_path, app, monkeypatch):
    """Encoding a sentence is trivial; LOADING CLIP is not. So a query vector is
    cached on disk, keyed by the normalised text: the same phrase never pays the
    model load twice — not even after a restart."""
    calls = []
    names = ['a.jpg', 'b.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0, 0.0),
                                      'b.jpg': _emb(0.0, 1.0)})
    _fake_encoder(monkeypatch, _emb(1.0, 0.0), counter=calls)
    first = client.post(f'/api/bank/{bank_id}/search-text',
                        json={'query': 'Brunette  Outdoors'}).get_json()
    second = client.post(f'/api/bank/{bank_id}/search-text',
                         json={'query': 'brunette outdoors'}).get_json()
    assert len(calls) == 1, 'the second (equivalent) query must not re-encode'
    assert first['image_ids'] == second['image_ids']
    assert first['cached'] is False and second['cached'] is True


def test_query_cache_survives_a_process_restart(client, tmp_path, app, monkeypatch):
    """The cache is a FILE, not a dict: dropping the in-memory layer (as a
    restart would) still answers without the encoder."""
    from app.services import clip_text_encoder
    calls = []
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0, 0.0)})
    _fake_encoder(monkeypatch, _emb(1.0, 0.0), counter=calls)
    client.post(f'/api/bank/{bank_id}/search-text', json={'query': 'a woman'})
    assert len(calls) == 1
    clip_text_encoder.forget_memory_cache()          # simulate a fresh process
    client.post(f'/api/bank/{bank_id}/search-text', json={'query': 'a woman'})
    assert len(calls) == 1, 'the on-disk cache must survive a restart'


# --- the warm worker: kept for a session, reaped afterwards -------------------
class _FakeWorker:
    """Stands in for the CLIP child: answers every query with a fixed vector and
    records whether it was shut down. No model, no subprocess."""

    def __init__(self, vector):
        self._vector = [float(x) for x in vector]
        self.closed = False
        self.killed = False
        self._pending = []

        class _In:
            closed = False

            def write(_self, line):
                self._pending.append(line)

            def flush(_self):
                pass

            def close(_self):
                _self.closed = True
                self.closed = True

        class _Out:
            def readline(_self):
                return json.dumps({'ok': True, 'vector': self._vector}) + '\n'

        self.stdin, self.stdout = _In(), _Out()

    def poll(self):
        return 0 if self.closed else None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


def _fake_worker_lifecycle(monkeypatch, vector):
    """Patch worker STARTUP only, so the real warm/reap logic is exercised."""
    from app.services import clip_text_encoder
    starts = []

    def _start():
        w = _FakeWorker(vector)
        starts.append(w)
        clip_text_encoder._proc = w
        return w

    monkeypatch.setattr(clip_text_encoder, '_start_worker_locked', _start)
    monkeypatch.setattr(clip_text_encoder, 'unavailable_reason', lambda: None)
    return starts


def test_the_worker_is_started_once_and_reused_across_queries(
        client, tmp_path, app, monkeypatch):
    """The whole point of staying warm: loading CLIP costs ~8.5 s, encoding a
    phrase ~20 ms. Two DIFFERENT queries in one session must pay the load once."""
    from app.services import clip_text_encoder
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0, 0.0)})
    starts = _fake_worker_lifecycle(monkeypatch, _emb(1.0, 0.0))
    monkeypatch.setattr(clip_text_encoder, 'idle_minutes', lambda: 10)
    client.post(f'/api/bank/{bank_id}/search-text', json={'query': 'first phrase'})
    client.post(f'/api/bank/{bank_id}/search-text', json={'query': 'second phrase'})
    assert len(starts) == 1, 'the warm worker must be reused, not restarted'
    assert clip_text_encoder.status()['warm'] is True


def test_idle_zero_reaps_the_worker_after_every_query(
        client, tmp_path, app, monkeypatch):
    """The escape hatch for a memory-tight machine: 0 minutes means the ~2.4 GB
    is handed back immediately, at the price of paying the load every time."""
    from app.services import clip_text_encoder
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0, 0.0)})
    starts = _fake_worker_lifecycle(monkeypatch, _emb(1.0, 0.0))
    monkeypatch.setattr(clip_text_encoder, 'idle_minutes', lambda: 0)
    client.post(f'/api/bank/{bank_id}/search-text', json={'query': 'first phrase'})
    assert starts[0].closed is True, 'idle_minutes=0 must reap immediately'
    assert clip_text_encoder.status()['warm'] is False
    client.post(f'/api/bank/{bank_id}/search-text', json={'query': 'second phrase'})
    assert len(starts) == 2, 'each query pays its own load when never warm'


def test_release_endpoint_reaps_the_worker(client, tmp_path, app, monkeypatch):
    """Closing the search panel gives the memory back at once — the idle timer
    is only the backstop for a tab that vanished."""
    from app.services import clip_text_encoder
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0, 0.0)})
    starts = _fake_worker_lifecycle(monkeypatch, _emb(1.0, 0.0))
    monkeypatch.setattr(clip_text_encoder, 'idle_minutes', lambda: 10)
    client.post(f'/api/bank/{bank_id}/search-text', json={'query': 'a phrase'})
    r = client.post('/api/bank/text-search/release')
    assert r.get_json()['released'] is True
    assert starts[0].closed is True
    # Idempotent: releasing again is a no-op, not an error.
    assert client.post('/api/bank/text-search/release').get_json()['released'] is False


def test_a_cache_hit_never_wakes_the_worker(client, tmp_path, app, monkeypatch):
    """The cheapest layer really is cheapest: a phrase already on disk must not
    start a 2.4 GB process to answer."""
    from app.services import clip_text_encoder
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0, 0.0)})
    starts = _fake_worker_lifecycle(monkeypatch, _emb(1.0, 0.0))
    monkeypatch.setattr(clip_text_encoder, 'idle_minutes', lambda: 10)
    client.post(f'/api/bank/{bank_id}/search-text', json={'query': 'a phrase'})
    clip_text_encoder.release()
    body = client.post(f'/api/bank/{bank_id}/search-text',
                       json={'query': 'A  PHRASE'}).get_json()
    assert body['cached'] is True
    assert len(starts) == 1, 'a cached phrase must not start the encoder again'


def test_status_reports_unavailability_without_raising(client, app, monkeypatch):
    """The status probe is what the UI asks BEFORE showing the field — it must
    answer on an install that cannot search at all, never explode."""
    from app.services import clip_text_encoder
    monkeypatch.setattr(clip_text_encoder, 'unavailable_reason',
                        lambda: 'torch is not installed')
    body = client.get('/api/bank/text-search/status').get_json()
    assert body['available'] is False
    assert 'torch' in body['reason']
    assert body['warm'] is False


def test_idle_minutes_is_clamped_and_never_a_trap(app, monkeypatch):
    """A hand-edited config must not be able to pin 2.4 GB forever, nor crash."""
    from app.config import save_config
    from app.services import clip_text_encoder
    with app.app_context():
        save_config({'bank_scoring': {'text_search_idle_minutes': 99999}})
        assert clip_text_encoder.idle_minutes() == 120.0      # capped
        save_config({'bank_scoring': {'text_search_idle_minutes': -5}})
        assert clip_text_encoder.idle_minutes() == 10         # default, not negative
        save_config({'bank_scoring': {'text_search_idle_minutes': 'nonsense'}})
        assert clip_text_encoder.idle_minutes() == 10


def test_query_normalisation_is_stable():
    """The cache key: case- and whitespace-insensitive, nothing else. Kept in one
    place so a future tweak cannot silently invalidate everyone's cache."""
    from app.services.clip_text_encoder import normalize_query
    assert normalize_query('  Brunette   Outdoors,  Wide Shot ') == \
        'brunette outdoors, wide shot'
    assert normalize_query('a woman') == normalize_query('A  WOMAN')
