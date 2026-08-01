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


def test_sanitize_caps_phrase_length():
    raw = {'human': {'face': {'expression': [
        {'id': 'too_long', 'phrase': 'x' * (fv._MAX_QUICK_GEN_PHRASE + 1)},
    ]}}}
    out = fv.sanitize_quick_gen_custom_components(raw)
    # The only entry is dropped for exceeding the phrase cap, so the axis,
    # the framing and the subject never survive to the output at all.
    assert out == {}


def test_sanitize_caps_entries_per_axis():
    entries = [{'id': f'id_{i}', 'phrase': f'phrase for id_{i}'} for i in range(101)]
    raw = {'human': {'face': {'expression': entries}}}
    out = fv.sanitize_quick_gen_custom_components(raw)
    ids = {e['id'] for e in out['human']['face']['expression']}
    assert len(ids) == fv._MAX_QUICK_GEN_CUSTOM_PER_AXIS
    for i in range(100):
        assert f'id_{i}' in ids
    assert 'id_100' not in ids


def test_sanitize_drops_duplicate_ids_within_the_same_batch():
    raw = {'human': {'face': {'expression': [
        {'id': 'dup', 'phrase': 'the first phrase wins'},
        {'id': 'dup', 'phrase': 'the second phrase must be dropped'},
    ]}}}
    out = fv.sanitize_quick_gen_custom_components(raw)
    kept = out['human']['face']['expression']
    assert len(kept) == 1
    assert kept[0] == {'id': 'dup', 'phrase': 'the first phrase wins', 'compatible_angles': None}


def test_sanitize_compatible_angles_pass_through_and_rejection():
    raw = {'human': {'face': {'expression': [
        {'id': 'with_angles', 'phrase': 'valid angles',
         'compatible_angles': ['front', 'three_quarter_left']},
        {'id': 'bad_angles', 'phrase': 'invalid angles', 'compatible_angles': 'front'},
    ]}}}
    out = fv.sanitize_quick_gen_custom_components(raw)
    kept = {e['id']: e for e in out['human']['face']['expression']}
    # A well-formed list of strings passes through verbatim.
    assert kept['with_angles']['compatible_angles'] == ['front', 'three_quarter_left']
    # A malformed value (here: a string, not a list) is coerced to None —
    # the entry itself is kept, not dropped.
    assert kept['bad_angles']['compatible_angles'] is None
