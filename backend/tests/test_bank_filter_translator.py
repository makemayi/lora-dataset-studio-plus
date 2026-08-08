"""The translator turns a sentence into the bank's own filter — and the interesting
tests are the ones about what it REFUSES to pass through. A filter the grid ignores,
announced in a confident summary, is worse than no feature: the user reads that the
request was understood while looking at an unchanged grid.

No Ollama here. Every function under test is pure: a reply string in, a validated
filter out. The model's reliability is not what these pin down — the validation is.
"""
import pytest
from PIL import Image

from app.services import bank_filter_translator as tr


def _seed_bank(app, tmp_path, n=3):
    """A real bank with real decodable files — the route test asserts that NOTHING
    about these rows changes, so they have to exist first."""
    from app.services import image_bank_service as banks
    src = tmp_path / 'src'
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new('RGB', (900, 900), (40, 90, 160)).save(str(src / f'{i}.jpg'))
    with app.app_context():
        bank, _added = banks.create_bank('local', 'Dump', str(src))
        return bank.id


def _stats():
    """Counts shaped like axis_stats(), with one deliberately empty bucket."""
    return {
        'medium': {'photo': 900, 'anime': 40, 'render3d': 0, 'illustration': 5, 'unsure': 12},
        'framing': {'face': 3, 'bust': 0, 'body': 1, 'back': 0, 'unknown': 950},
        'angle': {'frontal': 10, 'three_quarter': 4, 'profile': 2, 'behind': 0, 'unknown': 900},
        'origin': {'ai': 30, 'camera': 12, 'unknown': 900},
        'res_bucket': {'res_lt_025': 5, 'res_025_1': 200, 'res_1_2': 400,
                       'res_2_4': 300, 'res_gt_4': 40},
    }


def _reply(**payload):
    import json
    return 'Sure, here you go:\n```json\n' + json.dumps(payload) + '\n```'


def test_a_plain_request_becomes_a_filter_the_grid_accepts():
    out = tr.parse_reply(_reply(filter={'medium': 'photo', 'flag': 'low_aesthetic'},
                                sort='aesthetic_asc',
                                understood=['photographic', 'least polished first']),
                         _stats())
    assert out['filter'] == {'medium': 'photo', 'flag': 'low_aesthetic'}
    assert out['sort'] == 'aesthetic_asc'
    assert out['refused'] is False
    assert out['dropped'] == []


def test_a_negated_search_is_refused_and_says_why():
    """MEASURED on this app's encoder: a negated phrase ranks the named thing
    HIGHER ("a woman without a bikini" -> 60% bikinis against a 10.1% baseline).
    Passing it through would return exactly what the user asked to avoid."""
    out = tr.parse_reply(_reply(filter={'search': 'a portrait without a watermark'}),
                         _stats())
    assert 'search' not in out['filter']
    assert out['refused'] is True
    assert any('exclusion' in d and 'HIGHER' in d for d in out['dropped']), out['dropped']


def test_an_invented_field_is_dropped_and_named():
    """Never silently: a field the grid does not have must be reported, or the
    summary claims a narrowing that never happened."""
    out = tr.parse_reply(_reply(filter={'medium': 'photo', 'mood': 'moody',
                                        'aesthetic_max': 4.2}), _stats())
    assert out['filter'] == {'medium': 'photo'}
    assert any("'mood'" in d for d in out['dropped'])
    assert any("'aesthetic_max'" in d for d in out['dropped'])


def test_a_value_outside_the_banks_vocabulary_is_dropped():
    out = tr.parse_reply(_reply(filter={'medium': 'photograph'}), _stats())
    # 'photograph' is the English word; the stored key is 'photo'. A near-miss is
    # exactly what a model produces, and exactly what must not reach the grid.
    assert out['filter'] == {}
    assert out['refused'] is True
    assert any('photograph' in d for d in out['dropped'])


def test_a_bucket_with_zero_images_is_kept_but_flagged():
    """Kept, because the model DID understand the request and the grid's own
    counter is the honest place to learn the bucket is empty. Flagged, because a
    silent zero reads as a broken filter."""
    out = tr.parse_reply(_reply(filter={'medium': 'render3d'}), _stats())
    assert out['filter'] == {'medium': 'render3d'}
    assert any('0 images' in d for d in out['dropped'])


def test_status_alone_is_not_a_translation():
    """"Only the kept ones" is one click. Presenting it as a translation would make
    the feature look like it read a request it never parsed."""
    out = tr.parse_reply(_reply(filter={'status': 'keep'}), _stats())
    assert out['refused'] is True


def test_an_unexpressible_request_reports_it_instead_of_inventing():
    out = tr.parse_reply(_reply(filter={}, unsupported=['outdoors — needs captions']),
                         _stats())
    assert out['filter'] == {}
    assert out['refused'] is True
    assert out['unsupported'] == ['outdoors — needs captions']


def test_a_reply_that_is_not_json_refuses_rather_than_guessing():
    out = tr.parse_reply('I think you probably want the photographic ones.', _stats())
    assert out['refused'] is True
    assert out['filter'] == {}
    assert any('JSON' in d for d in out['dropped'])


def test_an_unknown_sort_is_dropped_not_forwarded():
    out = tr.parse_reply(_reply(filter={'medium': 'photo'}, sort='prettiness_desc'),
                         _stats())
    assert out['sort'] is None
    assert any('prettiness_desc' in d for d in out['dropped'])


def test_the_vocabulary_comes_from_the_service_not_a_copy():
    """The guard against the one failure this design cannot survive: a retyped list
    that drifts from the grid's. If the service renames a medium, this test fails
    here rather than in a user's silent no-op filter."""
    from app.services import image_bank_service as svc
    vocab = tr._vocab()
    assert vocab['medium'] == tuple(svc.MEDIUM_KEYS)
    assert vocab['flag'] == tuple(svc._QUALITY_FLAGS) + tuple(svc._SCORE_FLAGS)
    assert 'low_aesthetic' in vocab['flag']


@pytest.mark.parametrize('phrase', [
    'a woman without a bikini', 'no watermark', 'not a cartoon', 'excluding anime',
])
def test_every_shape_of_exclusion_is_caught(phrase):
    cleaned, why = tr._clean_text(phrase)
    assert cleaned is None
    assert 'exclusion' in why


def test_a_positive_phrase_survives_untouched():
    """The guard must not eat the normal case — proof the probe discriminates."""
    cleaned, why = tr._clean_text('  a candid  snapshot at a party ')
    assert cleaned == 'a candid snapshot at a party'
    assert why is None


def test_the_prompt_shows_counts_so_an_empty_bucket_is_visible():
    """The model reasons over THIS bank or it invents absolutes. Counts in the
    prompt are what make "do not pick a value worth 0" checkable rather than a
    pious instruction."""
    text = tr.build_prompt('I want an amateur dataset', _stats(),
                           ['captions: only 453 of 50397 images have it'])
    assert 'photo (900 images)' in text
    assert 'render3d (0 images)' in text
    assert 'captions: only 453' in text
    assert 'NEVER phrase `search` as an exclusion' in text


# --- The route: it must APPLY NOTHING, and it must not prompt for a missing bank.

def test_the_route_returns_a_filter_and_changes_no_row(client, app, tmp_path, monkeypatch):
    """The guarantee the whole design rests on: the model moves controls, never
    images. If this ever writes, the feature stops being cheap to be wrong about."""
    from app.services import bank_filter_translator as t
    from app.models import BankImage
    bank_id = _seed_bank(app, tmp_path)
    with app.app_context():
        before = {r.id: (r.status, r.caption) for r in
                  BankImage.query.filter_by(bank_id=bank_id).all()}
    monkeypatch.setattr(t, 'translate', lambda bid, s, **k: {
        'filter': {'medium': 'photo'}, 'sort': None, 'understood': ['photographic'],
        'unsupported': [], 'dropped': [], 'refused': False, 'request': s,
        'coverage': []})
    r = client.post(f'/api/bank/{bank_id}/describe-filter',
                    json={'request': 'an amateur dataset'})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()['filter'] == {'medium': 'photo'}
    with app.app_context():
        after = {r_.id: (r_.status, r_.caption) for r_ in
                 BankImage.query.filter_by(bank_id=bank_id).all()}
    assert after == before


def test_a_missing_bank_is_404_before_any_model_call(client, monkeypatch):
    """A round-trip to the model for a bank that is not there costs seconds and
    answers nobody's question."""
    from app.services import bank_filter_translator as t
    called = []
    monkeypatch.setattr(t, 'translate', lambda *a, **k: called.append(1) or {})
    r = client.post('/api/bank/999999/describe-filter', json={'request': 'anything'})
    assert r.status_code == 404
    assert called == []


def test_an_empty_request_is_400_and_never_reaches_the_model(client, app, tmp_path, monkeypatch):
    from app.services import vision_ollama
    bank_id = _seed_bank(app, tmp_path)
    called = []
    monkeypatch.setattr(vision_ollama, 'generate_text_ollama',
                        lambda *a, **k: called.append(1) or '{}')
    r = client.post(f'/api/bank/{bank_id}/describe-filter', json={'request': '   '})
    assert r.status_code == 400
    assert called == []
