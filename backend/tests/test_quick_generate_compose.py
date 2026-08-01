"""compose_quick_generate_variations: exact-count allocation (largest
remainder), angle-compatibility is never violated, self-describing poses
never get a redundant second angle clause, deterministic with an injected
rng."""
import random
import re

import pytest

from app.services import face_variations as fv


def _rng(seed):
    return random.Random(seed)


def test_total_count_is_exact_even_on_ugly_percentages():
    out = fv.compose_quick_generate_variations(
        total=7, framing_ratios={'face': 34, 'bust': 33, 'body': 33},
        angle_ratios={'face': {'front': 100}, 'bust': {'front': 100},
                     'body': {'front': 100}},
        rng=_rng(1))
    assert len(out) == 7


def test_framing_allocation_matches_ratio_via_largest_remainder():
    out = fv.compose_quick_generate_variations(
        total=10, framing_ratios={'face': 50, 'bust': 30, 'body': 20},
        angle_ratios={'face': {'front': 100}, 'bust': {'front': 100},
                     'body': {'front': 100}},
        rng=_rng(2))
    counts = {}
    for v in out:
        counts[v['framing']] = counts.get(v['framing'], 0) + 1
    assert counts == {'face': 5, 'bust': 3, 'body': 2}


def test_a_zero_ratio_framing_produces_no_slots():
    out = fv.compose_quick_generate_variations(
        total=5, framing_ratios={'face': 100, 'bust': 0, 'body': 0},
        angle_ratios={'face': {'front': 100}}, rng=_rng(3))
    assert all(v['framing'] == 'face' for v in out)
    assert len(out) == 5


@pytest.mark.parametrize('seed', range(20))
def test_pose_never_violates_its_own_angle_compatibility(seed):
    """Property test: over many seeds, every drawn (angle, pose) pair for
    bust/body is one the pose actually allows."""
    out = fv.compose_quick_generate_variations(
        total=40, framing_ratios={'face': 0, 'bust': 50, 'body': 50},
        angle_ratios={
            'bust': {'front': 25, 'three_quarter_left': 25, 'profile_left': 25, 'low_angle': 25},
            'body': {'front': 25, 'three_quarter_left': 25, 'low_angle_close': 25, 'lean_close': 25},
        },
        rng=_rng(seed))
    pools = fv.quick_gen_pools_for('human')
    for v in out:
        framing = v['framing']
        pose_pool = {e['id']: e for e in pools[framing]['pose']}
        drawn_pose_id = v['_debug_pose_id']  # test-only debug field, see Step 3
        pose = pose_pool[drawn_pose_id]
        if pose['compatible_angles'] is None:
            continue  # compatible with everything, nothing to violate
        if pose['compatible_angles'] == []:
            # Self-describing: the composer must NOT have drawn a separate
            # angle line for this slot.
            assert v['_debug_angle_id'] is None
            continue
        assert v['_debug_angle_id'] in pose['compatible_angles']


def test_self_describing_pose_prompt_has_no_redundant_angle_clause():
    """over_shoulder's own phrase already says the angle — the assembled
    prompt must not ALSO contain a second, independently-drawn angle phrase
    (e.g. 'front view') stacked in front of it."""
    out = fv.compose_quick_generate_variations(
        total=30, framing_ratios={'face': 0, 'bust': 100, 'body': 0},
        angle_ratios={'bust': {'front': 100}},  # forces angle draw attempts
        rng=_rng(4))
    over_shoulder_hits = [v for v in out if 'over the shoulder' in v['prompt']]
    assert over_shoulder_hits, 'over_shoulder pose never got drawn across 30 slots — widen the seed/pool if this flakes'
    for v in over_shoulder_hits:
        assert 'front view' not in v['prompt']


def test_expression_pool_never_yields_a_filtered_out_entry():
    out = fv.compose_quick_generate_variations(
        total=50, framing_ratios={'face': 100, 'bust': 0, 'body': 0},
        angle_ratios={'face': {'front': 100}}, rng=_rng(5))
    for v in out:
        assert 'laugh' not in v['prompt'].lower()
        assert 'surprised' not in v['prompt'].lower()


def test_face_prompt_has_no_pose_clause_but_carries_outfit_and_background_fallbacks():
    """Face has no pose axis (never will — a close-up has no body pose to
    vary), but it also has no outfit/background axis of its own, so without
    an explicit fallback here the model tends to default back to the
    reference photo's clothing/setting despite IDENTITY_GUARD's generic
    'do NOT copy' line. Every face prompt must therefore always carry
    OUTFIT_VARY and BACKGROUND_VARY verbatim."""
    out = fv.compose_quick_generate_variations(
        total=5, framing_ratios={'face': 100, 'bust': 0, 'body': 0},
        angle_ratios={'face': {'front': 100}}, rng=_rng(6))
    for v in out:
        assert v['prompt'].startswith('close-up portrait,')
        assert fv.OUTFIT_VARY in v['prompt']
        assert fv.BACKGROUND_VARY in v['prompt']


def test_face_prompt_draws_a_lighting_phrase_for_visual_variety():
    """Angle + expression alone reads repetitive at a high face ratio (e.g.
    15 of a 30-image batch) even when every combination is textually
    distinct — lighting is a third, independent axis that must show up in
    the composed prompt."""
    lighting_phrases = {e['phrase'] for e in fv.QUICK_GEN_COMPONENTS['human']['face']['lighting']}
    out = fv.compose_quick_generate_variations(
        total=20, framing_ratios={'face': 100, 'bust': 0, 'body': 0},
        angle_ratios={'face': {'front': 100}}, rng=_rng(9))
    used = set()
    for v in out:
        matches = [p for p in lighting_phrases if p in v['prompt']]
        assert len(matches) == 1, f"expected exactly one lighting phrase in {v['prompt']!r}"
        used.add(matches[0])
    # 20 draws from 7 lighting options should hit more than just one or two.
    assert len(used) >= 4


def test_bust_and_body_prompts_use_their_own_opening_clause():
    out = fv.compose_quick_generate_variations(
        total=4, framing_ratios={'face': 0, 'bust': 50, 'body': 50},
        angle_ratios={'bust': {'front': 100}, 'body': {'front': 100}}, rng=_rng(7))
    for v in out:
        if v['framing'] == 'bust':
            assert v['prompt'].startswith('upper body portrait,')
        else:
            assert v['prompt'].startswith('full body shot,')


def test_is_deterministic_with_the_same_injected_rng_seed():
    kw = dict(total=12, framing_ratios={'face': 50, 'bust': 30, 'body': 20},
             angle_ratios={'face': {'front': 60, 'three_quarter_left': 40},
                          'bust': {'front': 100}, 'body': {'front': 100}})
    a = fv.compose_quick_generate_variations(**kw, rng=_rng(42))
    b = fv.compose_quick_generate_variations(**kw, rng=_rng(42))
    assert [v['prompt'] for v in a] == [v['prompt'] for v in b]


def test_total_over_the_hard_cap_raises():
    with pytest.raises(ValueError, match='200'):
        fv.compose_quick_generate_variations(
            total=201, framing_ratios={'face': 100}, angle_ratios={'face': {'front': 100}})


def test_framing_ratios_must_sum_to_100():
    with pytest.raises(ValueError, match='sum to 100'):
        fv.compose_quick_generate_variations(
            total=5, framing_ratios={'face': 60, 'bust': 30}, angle_ratios={'face': {'front': 100}})


def test_a_framings_angle_ratios_must_sum_to_100_when_that_framing_is_used():
    with pytest.raises(ValueError, match='angle_ratios\\[.face.\\]'):
        fv.compose_quick_generate_variations(
            total=5, framing_ratios={'face': 100, 'bust': 0, 'body': 0},
            angle_ratios={'face': {'front': 50}})


def test_unknown_angle_id_in_angle_ratios_raises_a_clean_value_error():
    with pytest.raises(ValueError, match='unknown angle id'):
        fv.compose_quick_generate_variations(
            total=5, framing_ratios={'face': 100, 'bust': 0, 'body': 0},
            angle_ratios={'face': {'not_a_real_angle': 100}})


def test_total_must_be_an_int_not_a_float():
    with pytest.raises(ValueError, match='total must be an integer'):
        fv.compose_quick_generate_variations(
            total=7.5, framing_ratios={'face': 100, 'bust': 0, 'body': 0},
            angle_ratios={'face': {'front': 100}})


def test_every_framing_pose_pool_has_at_least_one_universally_compatible_pose():
    """If a future catalog edit leaves a framing's pose axis with no
    angle-independent entry, `_compose_one_slot` can draw an empty
    pose_candidates list and crash with IndexError. Guard this invariant so
    a regression fails loudly here instead of during a real compose call."""
    for framing in ('bust', 'body'):
        axes = fv.QUICK_GEN_COMPONENTS['human'][framing]
        assert any(p['compatible_angles'] is None for p in axes['pose']), (
            f"{framing}'s pose pool has no angle-independent (compatible_angles=None) entry")


def test_nsfw_ratio_zero_never_draws_nsfw_content():
    out = fv.compose_quick_generate_variations(
        total=20, framing_ratios={'face': 34, 'bust': 33, 'body': 33},
        angle_ratios={
            'face': {'front': 100},
            'bust': {'front': 100},
            'body': {'front': 100},
        },
        nsfw_ratio=0, rng=_rng(11))
    assert not any(v.get('nsfw') for v in out)


def test_nsfw_ratio_hundred_always_draws_nsfw_content_for_every_slot():
    out = fv.compose_quick_generate_variations(
        total=20, framing_ratios={'face': 34, 'bust': 33, 'body': 33},
        angle_ratios={
            'face': {'front': 100},
            'bust': {'front': 100},
            'body': {'front': 100},
        },
        nsfw_ratio=100, rng=_rng(12))
    assert len(out) == 20
    for v in out:
        assert v.get('nsfw') is True


def test_nsfw_slots_reuse_the_curated_catalog_label_and_prompt_verbatim():
    """An NSFW quick-gen slot must be a byte-for-byte NSFW_VARIATION_CATALOG
    entry's label+prompt, not a newly-composed one — this is what makes
    is_nsfw_label() and aspect_for_label() resolve it correctly downstream
    with zero extra plumbing."""
    catalog_by_framing = {}
    for e in fv.NSFW_VARIATION_CATALOG:
        catalog_by_framing.setdefault(e['framing'], []).append((e['label'], e['prompt']))
    out = fv.compose_quick_generate_variations(
        total=15, framing_ratios={'face': 34, 'bust': 33, 'body': 33},
        angle_ratios={
            'face': {'front': 100},
            'bust': {'front': 100},
            'body': {'front': 100},
        },
        nsfw_ratio=100, rng=_rng(13))
    for v in out:
        assert (v['label'], v['prompt']) in catalog_by_framing[v['framing']]
        assert fv.is_nsfw_label(v['label'])


def test_nsfw_ratio_partial_gives_a_mix_and_still_sums_to_total():
    out = fv.compose_quick_generate_variations(
        total=40, framing_ratios={'face': 34, 'bust': 33, 'body': 33},
        angle_ratios={
            'face': {'front': 100},
            'bust': {'front': 100},
            'body': {'front': 100},
        },
        nsfw_ratio=50, rng=_rng(14))
    assert len(out) == 40
    nsfw_count = sum(1 for v in out if v.get('nsfw'))
    sfw_count = len(out) - nsfw_count
    assert nsfw_count > 0 and sfw_count > 0


def test_sfw_slots_never_carry_an_nsfw_flag():
    out = fv.compose_quick_generate_variations(
        total=20, framing_ratios={'face': 100, 'bust': 0, 'body': 0},
        angle_ratios={'face': {'front': 100}},
        nsfw_ratio=30, rng=_rng(15))
    for v in out:
        if not v.get('nsfw'):
            assert v['prompt'].startswith('close-up portrait,')


def test_nsfw_ratio_out_of_range_raises_value_error():
    with pytest.raises(ValueError):
        fv.compose_quick_generate_variations(
            total=5, framing_ratios={'face': 100, 'bust': 0, 'body': 0},
            angle_ratios={'face': {'front': 100}}, nsfw_ratio=101, rng=_rng(16))
    with pytest.raises(ValueError):
        fv.compose_quick_generate_variations(
            total=5, framing_ratios={'face': 100, 'bust': 0, 'body': 0},
            angle_ratios={'face': {'front': 100}}, nsfw_ratio=-1, rng=_rng(17))


def test_default_nsfw_ratio_is_zero_for_backward_compatibility():
    """Existing callers that never pass nsfw_ratio must see IDENTICAL
    behavior to before this change — no NSFW content, ever."""
    out = fv.compose_quick_generate_variations(
        total=10, framing_ratios={'face': 100, 'bust': 0, 'body': 0},
        angle_ratios={'face': {'front': 100}}, rng=_rng(18))
    assert not any(v.get('nsfw') for v in out)
