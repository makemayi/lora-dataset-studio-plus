"""Frame selection is where "sharpest N" quietly becomes "the same picture N
times", so most of these tests are about what the ranking is NOT allowed to do.
"""

import pytest

from app.services import video_frame_select as vfs


def frame(t, sharp, luma=0.5, **extra):
    out = {'t': t, 'sharp': sharp, 'luma': luma}
    out.update(extra)
    return out


def good_face(**over):
    face = {'ok': True, 'det': 0.9, 'bbox_frac': 0.30, 'yaw': 5.0}
    face.update(over)
    return face


# ── the point of the module ──────────────────────────────────────────────────

def test_the_sharpest_run_of_neighbours_does_not_win_every_slot():
    """Sharpness is autocorrelated: without a time gap the top 3 are one moment."""
    frames = [frame(0.00, 100), frame(0.04, 99), frame(0.08, 98),
              frame(5.00, 40), frame(9.00, 30)]
    out = vfs.select_frames(frames, limit=3, require_face=False)
    times = [f['t'] for f in out['picked']]
    assert times == [0.00, 5.00, 9.00]
    assert out['rejected']['too_close'] == 2


def test_a_thin_shot_returns_fewer_than_asked_rather_than_padding():
    frames = [frame(0.0, 100), frame(0.1, 99)]
    out = vfs.select_frames(frames, limit=5, min_gap_s=0.75, require_face=False)
    assert len(out['picked']) == 1, 'the second frame is 100 ms away — same picture'


def test_everything_blurred_still_returns_the_best_of_it():
    """Blur is not a hard filter: 'the sharpest available' is a real answer, and
    a caller that wants a floor applies one to the returned `sharp` values."""
    frames = [frame(0.0, 1.0), frame(3.0, 2.0)]
    out = vfs.select_frames(frames, limit=2, require_face=False)
    assert [f['sharp'] for f in out['picked']] == [1.0, 2.0]


def test_output_is_chronological_not_sharpness_ordered():
    frames = [frame(0.0, 10), frame(4.0, 90), frame(8.0, 50)]
    out = vfs.select_frames(frames, limit=3, require_face=False)
    assert [f['t'] for f in out['picked']] == [0.0, 4.0, 8.0]


def test_ties_break_deterministically():
    a = vfs.select_frames([frame(0.0, 50), frame(9.0, 50)], limit=1,
                          require_face=False)
    b = vfs.select_frames([frame(9.0, 50), frame(0.0, 50)], limit=1,
                          require_face=False)
    assert [f['t'] for f in a['picked']] == [f['t'] for f in b['picked']]


# ── hard filters ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('luma', [0.0, 0.02, 0.99, 1.0])
def test_exposure_extremes_are_refused_however_sharp(luma):
    """A dissolve-to-black edge carries huge local contrast and is useless."""
    frames = [frame(0.0, 10_000, luma=luma), frame(5.0, 5, luma=0.5)]
    out = vfs.select_frames(frames, limit=2, require_face=False)
    assert [f['t'] for f in out['picked']] == [5.0]
    assert out['rejected']['exposure'] == 1


def test_a_frame_with_no_sharpness_reading_is_not_treated_as_zero():
    out = vfs.select_frames([frame(0.0, None), frame(4.0, 10)], limit=2,
                            require_face=False)
    assert [f['t'] for f in out['picked']] == [4.0]
    assert out['rejected']['unmeasured'] == 1


def test_a_tiny_face_is_refused_even_when_it_is_the_sharpest_frame():
    frames = [frame(0.0, 900, face=good_face(bbox_frac=0.005)),
              frame(5.0, 10, face=good_face())]
    out = vfs.select_frames(frames, limit=2)
    assert [f['t'] for f in out['picked']] == [5.0]
    assert out['rejected']['face_too_small'] == 1


def test_extreme_profile_is_refused():
    frames = [frame(0.0, 900, face=good_face(yaw=-75.0)),
              frame(5.0, 10, face=good_face(yaw=12.0))]
    out = vfs.select_frames(frames, limit=2)
    assert [f['t'] for f in out['picked']] == [5.0]
    assert out['rejected']['extreme_profile'] == 1


def test_low_detection_confidence_is_refused():
    out = vfs.select_frames([frame(0.0, 900, face=good_face(det=0.1))], limit=1)
    assert out['picked'] == []
    assert out['rejected']['low_det'] == 1


def test_a_full_body_shot_is_not_thrown_away_for_having_a_small_face():
    """The first draft floored this at 0.12 on the theory that a small face
    carries no identity. Measured on dataset 4 (2026-08-16), faces at 2-5 % of
    the photo score 0.48-0.89 — the same distribution as rows already called
    scorable — and those are the full-body shots a set needs for variety."""
    out = vfs.select_frames([frame(0.0, 50, face=good_face(bbox_frac=0.03))],
                            limit=1)
    assert len(out['picked']) == 1


def test_a_missing_face_reading_is_not_a_rejection():
    """Absent evidence is not evidence of absence: a decoder that could not run
    the face pass must not make every frame look faceless."""
    out = vfs.select_frames([frame(0.0, 10)], limit=1, require_face=True)
    assert len(out['picked']) == 1


def test_an_explicit_no_face_IS_a_rejection():
    out = vfs.select_frames([frame(0.0, 10, face={'ok': False})], limit=1)
    assert out['picked'] == []
    assert out['rejected']['no_face'] == 1


def test_require_face_off_admits_a_frame_the_gates_would_refuse():
    f = frame(0.0, 10, face=good_face(bbox_frac=0.01, yaw=88.0))
    assert len(vfs.select_frames([f], limit=1, require_face=False)['picked']) == 1
    assert vfs.select_frames([f], limit=1, require_face=True)['picked'] == []


# ── dedup ────────────────────────────────────────────────────────────────────

def test_a_locked_off_shot_is_deduplicated_across_the_time_gap():
    """The gap cannot catch this one: the subject is still for ten seconds."""
    frames = [frame(0.0, 90, embedding=[1.0, 0.0, 0.0]),
              frame(10.0, 80, embedding=[0.999, 0.01, 0.0]),
              frame(20.0, 70, embedding=[0.0, 1.0, 0.0])]
    out = vfs.select_frames(frames, limit=3, require_face=False)
    assert [f['t'] for f in out['picked']] == [0.0, 20.0]
    assert out['rejected']['duplicate'] == 1


def test_frames_without_embeddings_skip_the_dedup_stage():
    frames = [frame(0.0, 90), frame(10.0, 80)]
    out = vfs.select_frames(frames, limit=2, require_face=False)
    assert len(out['picked']) == 2


# ── degenerate inputs ────────────────────────────────────────────────────────

@pytest.mark.parametrize('frames', [None, []])
def test_no_frames_is_an_empty_answer_not_a_crash(frames):
    assert vfs.select_frames(frames, limit=3)['picked'] == []


@pytest.mark.parametrize('limit', [0, -1, None])
def test_a_zero_budget_picks_nothing(limit):
    out = vfs.select_frames([frame(0.0, 10)], limit=limit, require_face=False)
    assert out['picked'] == []


# ── quota across sources ─────────────────────────────────────────────────────

def test_one_long_video_cannot_eat_the_whole_budget():
    quota = vfs.spread_quota({'long': 500, 'short': 4}, 20)
    assert quota['short'] == 4, 'the short source gives everything it has'
    assert quota['long'] == 16
    assert sum(quota.values()) == 20


def test_a_budget_larger_than_the_supply_is_not_invented():
    quota = vfs.spread_quota({'a': 2, 'b': 3}, 100)
    assert quota == {'a': 2, 'b': 3}


def test_a_budget_smaller_than_the_source_count_is_handed_out_one_at_a_time():
    quota = vfs.spread_quota({'a': 10, 'b': 10, 'c': 10}, 2)
    assert sum(quota.values()) == 2
    assert all(v <= 1 for v in quota.values())


def test_quota_is_deterministic():
    counts = {'a': 10, 'b': 10, 'c': 10}
    assert vfs.spread_quota(counts, 5) == vfs.spread_quota(counts, 5)


@pytest.mark.parametrize('total', [0, -3, None])
def test_no_budget_means_no_quota(total):
    assert vfs.spread_quota({'a': 5}, total) == {'a': 0}


def test_no_sources_is_an_empty_quota():
    assert vfs.spread_quota({}, 10) == {}


# ── the two tables that must not drift ───────────────────────────────────────

def test_the_face_gates_stay_equal_to_the_scorers_own(tmp_path):
    """`face_score_infer.py` runs in a DIFFERENT interpreter, so its constants
    cannot be imported — they are read here instead. If that file's gates move
    and these do not, this module admits frames the rest of the app then refuses
    to score, and nothing else would say so."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / 'infer'
           / 'face_score_infer.py').read_text(encoding='utf-8')
    m = re.search(r'DET_MIN,\s*BBOX_MIN,\s*YAW_MAX\s*=\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)',
                  src)
    assert m, 'the scorer no longer declares its gates on one line — re-read it'
    det, bbox, yaw = (float(x) for x in m.groups())
    assert vfs.DET_MIN == det
    assert vfs.YAW_MAX == yaw
    assert vfs.FACE_BBOX_MIN == bbox


# ── the character gates ──────────────────────────────────────────────────────
# Off unless a CHARACTER dataset asks for them: that is the one kind where a
# technically fine frame showing the wrong face, or a face too small to hold
# detail, is worse than no frame at all.

def test_face_pixels_needs_the_frames_own_size_and_refuses_to_guess():
    """A fabricated pixel count is worse than an ungated frame — it looks like
    a measurement."""
    assert vfs.face_pixels({'face': {'bbox_frac': 0.04}}) is None
    assert vfs.face_pixels({'face': {'bbox_frac': 0.04}, 'w': 1920}) is None
    assert vfs.face_pixels({'w': 1920, 'h': 1080}) is None


def test_the_same_fraction_is_a_usable_face_at_4k_and_a_smudge_at_720p():
    """The whole reason the gate counts pixels and not the fraction."""
    big = {'face': {'bbox_frac': 0.02}, 'w': 3840, 'h': 2160}
    small = {'face': {'bbox_frac': 0.02}, 'w': 640, 'h': 360}
    assert vfs.face_pixels(big) > vfs.MIN_FACE_PX
    assert vfs.face_pixels(small) < vfs.MIN_FACE_PX


def test_a_character_set_refuses_a_face_too_small_in_pixels():
    # The SAME 2 % of the picture: a ~67 px face at 640x360, ~407 px at 4K.
    f = frame(0.0, 900, face=good_face(bbox_frac=0.02), w=640, h=360)
    assert vfs.select_frames([f], limit=1)['picked'], 'ungated, it is admissible'
    out = vfs.select_frames([f], limit=1, min_face_px=vfs.MIN_FACE_PX)
    assert out['picked'] == []
    assert out['rejected']['face_too_few_pixels'] == 1


def test_the_same_face_at_4k_passes_the_pixel_gate():
    f = frame(0.0, 900, face=good_face(bbox_frac=0.02), w=3840, h=2160)
    out = vfs.select_frames([f], limit=1, min_face_px=vfs.MIN_FACE_PX)
    assert len(out['picked']) == 1


def test_an_unmeasurable_frame_does_not_silently_disable_the_pixel_gate():
    """A decoder that forgot to report its own size must not become the one
    input for which a character set's floor does not apply."""
    f = frame(0.0, 900, face=good_face(bbox_frac=0.02))     # no w/h
    out = vfs.select_frames([f], limit=1, min_face_px=vfs.MIN_FACE_PX)
    assert out['picked'] == []
    assert out['rejected']['face_px_unknown'] == 1


def test_a_sharp_frame_of_the_wrong_person_is_refused():
    frames = [frame(0.0, 900, face=good_face(sim=0.10)),
              frame(5.0, 10, face=good_face(sim=0.80))]
    out = vfs.select_frames(frames, limit=2, min_sim=vfs.MIN_SIM)
    assert [f['t'] for f in out['picked']] == [5.0]
    assert out['rejected']['wrong_person'] == 1


def test_no_similarity_reading_is_not_read_as_a_mismatch():
    """`sim` absent means the scorer had no reference, which is not the same
    claim as "does not resemble"."""
    out = vfs.select_frames([frame(0.0, 10, face=good_face())], limit=1,
                            min_sim=vfs.MIN_SIM)
    assert len(out['picked']) == 1


def test_the_character_gates_are_off_unless_asked_for():
    f = frame(0.0, 10, face=good_face(bbox_frac=0.02, sim=0.01), w=640, h=360)
    assert len(vfs.select_frames([f], limit=1)['picked']) == 1
