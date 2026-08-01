"""quick_generate.custom_nsfw: same storage spirit as quick_generate.custom_components,
but flatter shape (no axis nesting, no compatible_angles) since each NSFW entry is a
whole self-contained state+pose+décor prompt, never decomposed into independent axes —
see _QUICK_GEN_NSFW_POOL's own docstring/comment for why."""
from app.services import face_variations as fv


def test_default_config_key_is_empty():
    from app import config as cfg
    assert cfg.DEFAULTS.get('quick_generate', {}).get('custom_nsfw') == {}


def test_sanitize_drops_malformed_entries():
    raw = {
        'human': {
            'bust': [
                {'id': 'my_bust_pose', 'label': 'Bust, custom pose', 'prompt': 'a valid prompt'},  # valid
                {'id': '', 'label': 'x', 'prompt': 'y'},                    # empty id, dropped
                {'id': 'no_prompt', 'label': 'x'},                          # missing prompt, dropped
                {'id': 'no_label', 'prompt': 'y'},                          # missing label, dropped
                'not-a-dict',                                              # dropped
            ],
        },
    }
    out = fv.sanitize_quick_gen_custom_nsfw(raw)
    assert out == {'human': {'bust': [
        {'id': 'my_bust_pose', 'label': 'Bust, custom pose', 'prompt': 'a valid prompt'},
    ]}}


def test_sanitize_never_shadows_a_built_in_id():
    # nsfw_bust_lingerie is a real built-in id in NSFW_VARIATION_CATALOG
    raw = {'human': {'bust': [
        {'id': 'nsfw_bust_lingerie', 'label': 'trying to shadow', 'prompt': 'x'},
    ]}}
    out = fv.sanitize_quick_gen_custom_nsfw(raw)
    assert out == {}


def test_sanitize_rejects_an_unknown_framing():
    raw = {'human': {'not_a_framing': [{'id': 'x', 'label': 'y', 'prompt': 'z'}],
                     'back': [{'id': 'x', 'label': 'y', 'prompt': 'z'}]}}  # 'back' unsupported here too
    assert fv.sanitize_quick_gen_custom_nsfw(raw) == {}


def test_sanitize_never_raises_on_garbage_input():
    assert fv.sanitize_quick_gen_custom_nsfw(None) == {}
    assert fv.sanitize_quick_gen_custom_nsfw('garbage') == {}
    assert fv.sanitize_quick_gen_custom_nsfw(42) == {}


def test_sanitize_only_accepts_human_subject_type():
    """NSFW content stays human-only — same rule compose_quick_generate_variations
    already enforces (see the st == 'human' guard there)."""
    raw = {'animal': {'body': [{'id': 'x', 'label': 'y', 'prompt': 'z'}]}}
    assert fv.sanitize_quick_gen_custom_nsfw(raw) == {}


def test_custom_nsfw_entries_are_drawn_by_the_composer(app, monkeypatch):
    from app import config as cfg
    import random
    with app.app_context():
        cfg.save_config({'quick_generate': {'custom_nsfw': {
            'human': {'face': [
                {'id': 'my_custom_nsfw_face', 'label': 'My Custom NSFW Face', 'prompt': 'a custom nsfw face prompt'},
            ]}}}})
        out = fv.compose_quick_generate_variations(
            total=30, framing_ratios={'face': 100, 'bust': 0, 'body': 0},
            angle_ratios={'face': {'front': 100}}, nsfw_ratio=100,
            rng=random.Random(21))
        labels = {v['label'] for v in out}
        assert 'My Custom NSFW Face' in labels
        # shipped defaults must still be reachable too, not replaced:
        assert len(labels) > 1
