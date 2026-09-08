"""The one piece of genuinely new judgement in the Bank->dataset build.

A picture gives ONE framing while the pool covers the request (the operator's
rule, and CLAUDE.md's). Only when the request exceeds the pool may a picture give
a second and a third — never the same framing twice. Everything else here is
arithmetic that must be provable without a database.
"""
from app.services.promotion_quota import allocate


def _pics(n, with_face=True, start=1):
    return [{'id': i, 'has_face_box': with_face} for i in range(start, start + n)]


def test_a_pool_larger_than_the_request_never_reuses_a_picture():
    out = allocate(_pics(120), {'face': 30, 'half': 30, 'full': 30})
    assert out['counts'] == {'face': 30, 'half': 30, 'full': 30}
    assert out['reused'] == 0
    assert out['shortfall'] == {}
    assert len(out['assignments']) == 90
    assert len({pid for pid, _ in out['assignments']}) == 90


def test_a_pool_of_exactly_the_request_uses_each_picture_once():
    out = allocate(_pics(90), {'face': 30, 'half': 30, 'full': 30})
    assert out['counts'] == {'face': 30, 'half': 30, 'full': 30}
    assert out['reused'] == 0


def test_a_short_pool_reuses_the_best_pictures_in_rounds():
    # 40 pictures, 90 asked for: pass 1 gives 40, then rounds of one extra
    # framing each until the quotas fill.
    out = allocate(_pics(40), {'face': 30, 'half': 30, 'full': 30})
    assert out['counts'] == {'face': 30, 'half': 30, 'full': 30}
    assert out['shortfall'] == {}
    assert out['reused'] == 40, 'every picture gave more than one framing'
    per_picture = {}
    for pid, framing in out['assignments']:
        per_picture.setdefault(pid, []).append(framing)
    assert all(len(v) == len(set(v)) for v in per_picture.values()), \
        'the same picture must never give the same framing twice'
    assert max(len(v) for v in per_picture.values()) <= 3
    # Reuse starts at the TOP of the ranking, not the start of the list.
    assert len(per_picture[1]) == 3


def test_a_pool_that_cannot_cover_the_request_reports_the_shortfall():
    out = allocate(_pics(10), {'face': 30, 'half': 30, 'full': 30})
    assert sum(out['counts'].values()) == 30, 'ten pictures give at most 30 images'
    assert out['shortfall'] == {'face': 20, 'half': 20, 'full': 20}


def test_a_picture_without_a_face_box_can_only_give_the_full_frame():
    out = allocate(_pics(5, with_face=False), {'face': 5, 'half': 5, 'full': 5})
    assert out['counts'] == {'full': 5}
    assert all(f == 'full' for _pid, f in out['assignments'])
    assert out['shortfall'] == {'face': 5, 'half': 5}


def test_the_split_tracks_the_requested_ratio_while_the_pool_lasts():
    out = allocate(_pics(30), {'face': 20, 'half': 10, 'full': 0})
    assert out['counts'] == {'face': 20, 'half': 10}
    assert 'full' not in out['counts']


def test_a_zero_quota_is_not_a_framing_and_an_empty_pool_is_not_a_crash():
    assert allocate([], {'face': 1})['counts'] == {}
    assert allocate(_pics(3), {})['assignments'] == []


def test_the_walk_round_robins_the_angle_buckets_when_caps_bite():
    # Four pictures, two per angle, quota for two full frames: the interleave
    # spends the caps on one picture PER ANGLE instead of both on whoever
    # ranked first. Unmeasured pictures keep their order, last.
    pics = [{'id': 1, 'has_face_box': True, 'angle': 'frontal'},
            {'id': 2, 'has_face_box': True, 'angle': 'frontal'},
            {'id': 3, 'has_face_box': True, 'angle': 'profile'},
            {'id': 4, 'has_face_box': True, 'angle': 'profile'}]
    out = allocate(pics, {'full': 2})
    chosen = [pid for pid, f in out['assignments']]
    assert chosen == [1, 3], 'one frontal and one profile, not two frontals'


def test_pictures_without_an_angle_keep_their_original_order():
    # The pre-angle contract: no 'angle' keys means the interleave is invisible.
    pics = _pics(6)
    out = allocate(pics, {'full': 3})
    assert [pid for pid, _f in out['assignments']] == [1, 2, 3]
