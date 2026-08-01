"""quick_generate.custom_components: same shape/spirit as custom_shots —
optional, config-stored, sanitized on read, additive-only (never shadows a
built-in id)."""
from app.services import face_variations as fv


def test_default_config_key_is_empty():
    from app import config as cfg
    assert cfg.DEFAULTS.get('quick_generate', {}).get('custom_components') == {}


def test_sanitize_drops_malformed_entries():
    raw = {
        'human': {
            'face': {'expression': [
                {'id': 'wink', 'phrase': 'a playful wink'},   # valid
                {'id': '', 'phrase': 'x'},                    # empty id, dropped
                {'id': 'no_phrase'},                          # missing phrase, dropped
                'not-a-dict',                                 # dropped
            ]},
        },
    }
    out = fv.sanitize_quick_gen_custom_components(raw)
    assert out == {'human': {'face': {'expression': [
        {'id': 'wink', 'phrase': 'a playful wink', 'compatible_angles': None},
    ]}}}


def test_sanitize_never_shadows_a_built_in_id():
    raw = {'human': {'face': {'expression': [
        {'id': 'neutral', 'phrase': 'a suspiciously different neutral'},
    ]}}}
    out = fv.sanitize_quick_gen_custom_components(raw)
    assert out == {}


def test_sanitize_rejects_an_unknown_framing_or_axis():
    raw = {'human': {'not_a_framing': {'angle': [{'id': 'x', 'phrase': 'y'}]},
                     'face': {'not_an_axis': [{'id': 'x', 'phrase': 'y'}]}}}
    assert fv.sanitize_quick_gen_custom_components(raw) == {}


def test_sanitize_never_raises_on_garbage_input():
    assert fv.sanitize_quick_gen_custom_components(None) == {}
    assert fv.sanitize_quick_gen_custom_components('garbage') == {}
    assert fv.sanitize_quick_gen_custom_components(42) == {}


def test_quick_gen_pools_for_merges_custom_into_the_shipped_pool(app, monkeypatch):
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'quick_generate': {'custom_components': {
            'human': {'face': {'expression': [
                {'id': 'wink', 'phrase': 'a playful wink'},
            ]}}}}})
        pools = fv.quick_gen_pools_for('human')
        ids = {e['id'] for e in pools['face']['expression']}
        assert 'wink' in ids
        assert 'neutral' in ids  # shipped defaults still present
