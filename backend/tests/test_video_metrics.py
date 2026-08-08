"""🎬 The one decode that measures everything — and how the per-frame numbers
become a per-clip verdict.

TWO THINGS ARE BEING DESIGNED HERE, AND THE SECOND IS THE SUBTLE ONE.

The first is cost. Decoding is ~85 % of this lane's time (measured on a real
corpus, once shot detection moved to the GPU), so every metric has to come out of
ONE pass. Four independent passes would cost four times the only expensive part.

The second is HOW per-frame numbers collapse into one clip score, and it is not
uniform. The model learns EVERY frame, so what matters is rarely the average:

  exposure  → MIN. A half-second fade to black in the middle ruins the sample,
              and an average hides it completely.
  sharpness → a high percentile. "Is there real sharpness anywhere in this clip?"
              Legitimate motion blur drags a mean down and would reject the best
              clips; an absolute threshold on a mean is the classic mistake here.
  motion    → mean AND a high percentile. Two different questions — "does anything
              move at all?" and "is it thrashing?" — that cannot share a number.

Everything here is pure arithmetic over already-decoded numbers. The decode itself
sits behind one seam, so these stay green on an install with no video extra.
"""
import pytest

from app.services import video_metrics as vm


# --- the aggregation semantics ------------------------------------------------

def test_exposure_is_the_worst_frame_not_the_average():
    """A clip that is well lit for 3 seconds and black for half of one is not a
    well-lit clip: the model trains on the black frames too. The average says 0.87
    and hides it; the minimum says 0.02 and does not."""
    frames = [_frame(luma=0.9)] * 30 + [_frame(luma=0.02)] * 15 + [_frame(luma=0.9)] * 30

    summary = vm.summarise(frames, fps=25)

    assert summary['luma_min'] == pytest.approx(0.02)
    assert summary['luma_mean'] > 0.5          # the average really does hide it


def test_sharpness_is_a_high_percentile_not_the_average():
    """Motion blur is legitimate and drags a mean down. The question worth asking
    is "does real sharpness exist in this clip?" — a mean would reject exactly the
    clips with the most interesting movement.

    p90 and not p75, and the arithmetic is what decides it: a clip sharp for a
    fifth of its length is perfectly usable, and p75 lands inside the blurred four
    fifths and calls it soft."""
    frames = [_frame(sharp=10.0)] * 80 + [_frame(sharp=900.0)] * 20

    summary = vm.summarise(frames, fps=25)

    assert summary['sharpness_p90'] > 100      # the sharp fifth is visible
    assert summary['sharpness_p90'] > sum(f['sharp'] for f in frames) / len(frames)


def test_one_fluke_sharp_frame_does_not_make_a_clip_sharp():
    """The reason it is not simply the maximum: a single lucky frame in a hundred
    is noise, and a max would let it vouch for the whole clip."""
    frames = [_frame(sharp=10.0)] * 99 + [_frame(sharp=900.0)]

    assert vm.summarise(frames, fps=25)['sharpness_p90'] < 100


def test_motion_answers_two_questions_with_two_numbers():
    """"Does anything move?" and "is it thrashing?" cannot share a figure. A clip
    that is still for most of its length and then whips across answers yes to both
    — its mean is low and its p95 is high, and both facts matter."""
    frames = [_frame(motion=0.0001)] * 90 + [_frame(motion=0.05)] * 10

    summary = vm.summarise(frames, fps=25)

    assert summary['motion_mean'] < 0.01
    assert summary['motion_p95'] > 0.01


def test_a_freeze_in_the_middle_is_found_even_when_the_clip_moves():
    """A frozen second inside an otherwise lively shot is a common artefact of
    scraped material, and the clip's overall motion stays perfectly healthy — so
    the mean can never reveal it. The share of near-still frames can."""
    frames = [_frame(motion=0.006)] * 50 + [_frame(motion=0.0)] * 25 + [_frame(motion=0.006)] * 25

    summary = vm.summarise(frames, fps=25)

    assert summary['motion_mean'] > 0.003      # the clip looks fine on average
    assert summary['freeze_ratio'] == pytest.approx(0.25, abs=0.02)


def test_the_sharpest_frame_is_reported_with_its_timestamp():
    """The thumbnail currently comes from the middle of the shot, which is a
    reasonable guess. Once every frame has been measured anyway, the sharpest one
    is a better answer and costs nothing extra."""
    frames = [_frame(sharp=5.0)] * 10 + [_frame(sharp=500.0)] + [_frame(sharp=5.0)] * 10

    summary = vm.summarise(frames, fps=10)

    assert summary['sharpest_frame_s'] == pytest.approx(1.0)


# --- degenerate input ---------------------------------------------------------

def test_a_clip_with_no_decodable_frame_measures_nothing_rather_than_zero():
    """Zero is a measurement; "we could not measure" is not. Collapsing the two
    would make an unreadable clip look like a perfectly black, perfectly still
    one — and it would be filtered out for the wrong reason."""
    summary = vm.summarise([], fps=25)

    assert summary['metrics_state'] == 'unreadable'
    assert summary['motion_mean'] is None
    assert summary['luma_min'] is None


def test_a_single_frame_still_produces_a_summary():
    summary = vm.summarise([_frame(luma=0.4, sharp=30.0, motion=0.002)], fps=25)

    assert summary['metrics_state'] == 'ok'
    assert summary['luma_min'] == pytest.approx(0.4)


def test_a_percentile_of_one_value_is_that_value():
    assert vm.percentile([7.0], 0.95) == pytest.approx(7.0)


def test_a_percentile_of_nothing_is_none_not_zero():
    assert vm.percentile([], 0.95) is None


# --- verdicts are computed at READ time ---------------------------------------

def test_a_still_clip_is_flagged_against_the_bank_s_own_distribution():
    """A published threshold from another corpus does not transfer: on this
    machine's 4.5-hour test bank, the value that a public pipeline uses as its
    floor lands at the 7th percentile. So the cut is expressed as a percentile of
    the bank being worked on, and the raw scores stay in the database untouched."""
    scores = {'motion_mean': 0.0002, 'luma_min': 0.4, 'sharpness_p90': 200.0,
              'freeze_ratio': 0.0}
    floor = {'motion_floor': 0.001}

    assert 'still' in vm.verdicts(scores, floor)


def test_a_lively_clip_carries_no_flag():
    scores = {'motion_mean': 0.004, 'luma_min': 0.4, 'sharpness_p90': 200.0,
              'freeze_ratio': 0.0}

    assert vm.verdicts(scores, {'motion_floor': 0.001}) == set()


def test_a_clip_with_a_black_moment_is_flagged_however_bright_the_rest():
    scores = {'motion_mean': 0.004, 'luma_min': 0.01, 'sharpness_p90': 200.0,
              'freeze_ratio': 0.0}

    assert 'black' in vm.verdicts(scores, {'luma_floor': 0.05})


def test_a_frozen_stretch_is_its_own_flag_not_a_stillness_one():
    """They are different defects with different fixes: a still clip is useless,
    a clip with a freeze can be re-cut around it."""
    scores = {'motion_mean': 0.004, 'luma_min': 0.4, 'sharpness_p90': 200.0,
              'freeze_ratio': 0.3}

    flags = vm.verdicts(scores, {'freeze_max': 0.1})
    assert 'freeze' in flags and 'still' not in flags


def test_an_unmeasured_clip_is_never_flagged():
    """Absence of measurement must never read as a defect — that is how a scan
    that failed silently becomes a bank that "filtered" half its clips."""
    scores = {'motion_mean': None, 'luma_min': None, 'sharpness_p90': None,
              'freeze_ratio': None}

    assert vm.verdicts(scores, {'motion_floor': 0.001, 'luma_floor': 0.05}) == set()


# --- the dry run --------------------------------------------------------------

def test_a_dry_run_says_how_many_clips_each_cut_would_remove():
    """Never a silent filter. A public dataset pipeline once kept 47 clips out of
    1493 with a mis-set threshold and only found out afterwards; the answer is to
    show the count BEFORE committing, per rule, not a total."""
    bank = [{'motion_mean': 0.0001, 'luma_min': 0.5, 'sharpness_p90': 100.0, 'freeze_ratio': 0.0},
            {'motion_mean': 0.004, 'luma_min': 0.01, 'sharpness_p90': 100.0, 'freeze_ratio': 0.0},
            {'motion_mean': 0.004, 'luma_min': 0.5, 'sharpness_p90': 100.0, 'freeze_ratio': 0.0}]

    preview = vm.dry_run(bank, {'motion_floor': 0.001, 'luma_floor': 0.05})

    assert preview['still'] == 1
    assert preview['black'] == 1
    assert preview['total_flagged'] == 2       # not 2 rules x 1 clip counted twice


def test_a_dry_run_on_an_empty_bank_reports_zero_not_an_error():
    assert vm.dry_run([], {'motion_floor': 0.001})['total_flagged'] == 0


# --- helpers ------------------------------------------------------------------

def _frame(luma=0.5, sharp=100.0, motion=0.003):
    return {'luma': luma, 'sharp': sharp, 'motion': motion}


# --- the ambassador frame is chosen under constraints --------------------------

def test_a_flash_frame_cannot_become_the_ambassador():
    """The sharpness SCORE uses p90 so one lucky frame cannot vouch for a clip —
    but the frame CHOICE used a raw argmax, and a single overexposed flash frame
    (huge local contrast, useless as a thumbnail or an embedding) would win it.
    The sharpest frame is now picked among frames with sane exposure."""
    frames = [_frame(luma=0.5, sharp=100.0)] * 20
    frames[7] = {'luma': 0.99, 'sharp': 900.0, 'motion': 0.003}   # flash
    frames[12] = {'luma': 0.5, 'sharp': 400.0, 'motion': 0.003}   # real winner

    assert vm.summarise(frames, fps=10)['sharpest_frame_s'] == pytest.approx(1.2)


def test_a_near_black_frame_cannot_become_the_ambassador():
    """Dissolves to black have violent local contrast on their edges; the frame is
    still useless to look at. Same constraint, dark side."""
    frames = [_frame(luma=0.5, sharp=100.0)] * 20
    frames[3] = {'luma': 0.02, 'sharp': 900.0, 'motion': 0.003}
    frames[15] = {'luma': 0.5, 'sharp': 300.0, 'motion': 0.003}

    assert vm.summarise(frames, fps=10)['sharpest_frame_s'] == pytest.approx(1.5)


def test_an_all_bad_clip_still_gets_an_ambassador():
    """When every frame violates the constraints (a clip that is all flash, or all
    black), the plain argmax returns rather than nothing: a bad thumbnail beats no
    thumbnail, and the quality flags already tell the story."""
    frames = [{'luma': 0.99, 'sharp': float(i), 'motion': 0.003} for i in range(10)]

    assert vm.summarise(frames, fps=10)['sharpest_frame_s'] == pytest.approx(0.9)


# --- the first frame is measured on its own ------------------------------------

def test_the_first_frame_sharpness_is_stored_separately():
    """For image-to-video targets the FIRST frame is the conditioning image, and
    nobody chooses it — it is whatever the cut starts on. A gorgeous clip whose
    first frame is blurred is a BAD i2v clip: the model would learn to animate
    from a degraded image, the opposite of real use. The measure is free — the
    Laplacian of frame 0 was already computed."""
    frames = [_frame(sharp=12.5)] + [_frame(sharp=500.0)] * 20

    assert vm.summarise(frames, fps=25)['first_frame_sharpness'] == pytest.approx(12.5)


def test_a_soft_start_is_flagged_only_when_a_floor_is_set():
    scores = {'motion_mean': 0.004, 'luma_min': 0.4, 'sharpness_p90': 500.0,
              'freeze_ratio': 0.0, 'first_frame_sharpness': 10.0}

    assert 'soft_start' in vm.verdicts(scores, {'first_frame_floor': 50.0})
    assert vm.verdicts(scores, {}) == set()
