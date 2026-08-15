"""The two-pass extraction, tested without a video file.

Decode and the face pass are injected, so what is under test is the ORDER and
the degradation: that the face gate runs on pass B's real pictures, that a
missing face pass widens rather than empties the result, and that an unreadable
clip costs one clip instead of the batch.
"""

import pytest

from app.services import video_frame_extract as vfe


def readings(n=40, *, sharp=None, luma=0.5, step=0.2):
    """A clip of `n` frames, 200 ms apart, sharpness rising with time."""
    return [{'t': round(i * step, 3), 'luma': luma,
             'sharp': (sharp(i) if sharp else float(i)), 'motion': 0.1}
            for i in range(n)]


def fake_decode(_path, times):
    return [{'t': t, 'bytes': f'IMG@{t}'.encode()} for t in times]


def run(**over):
    kw = dict(path='x.mp4', start_s=0.0, end_s=8.0, fps=25, limit=3,
              read_frames=lambda *a: readings(),
              decode=fake_decode, clip_id=7, source_id=3)
    kw.update(over)
    return vfe.extract_from_clip(**kw)


# ── the ordering the module exists for ───────────────────────────────────────

def test_the_face_pass_sees_pass_b_timestamps_not_the_whole_clip():
    seen = {}

    def faces(frames):
        # The hook must receive the decoded pictures, not bare timestamps: the
        # scorer reads files and these frames exist only as bytes here.
        assert all('bytes' in f for f in frames)
        seen['times'] = [f['t'] for f in frames]
        return [{'ok': True, 'det': 0.9, 'bbox_frac': 0.3, 'yaw': 3.0}
                for _ in frames]

    out = run(face_scores=faces)
    assert out, 'a clip of usable frames must yield frames'
    assert len(seen['times']) < 40, 'the face pass must not score every frame'
    assert len(seen['times']) >= len(out)


def test_the_shortlist_is_wider_than_the_budget_so_the_gate_can_reject():
    seen = {}

    def faces(frames):
        seen['n'] = len(frames)
        return [{'ok': True, 'det': 0.9, 'bbox_frac': 0.3, 'yaw': 3.0}
                for _ in frames]

    run(limit=2, face_scores=faces)
    assert seen['n'] > 2


def test_a_tiny_face_on_the_sharpest_frame_hands_the_slot_to_the_next():
    def faces(frames):
        # The LAST timestamp is the sharpest frame in this fixture.
        worst = max(f['t'] for f in frames)
        return [{'ok': True, 'det': 0.9, 'yaw': 2.0,
                 'bbox_frac': 0.02 if f['t'] == worst else 0.3} for f in frames]

    out = run(limit=1, face_scores=faces)
    assert len(out) == 1
    assert out[0]['provenance']['timestamp_s'] != max(
        f['t'] for f in readings())


# ── degradation ──────────────────────────────────────────────────────────────

def test_no_face_pass_still_returns_frames():
    """A missing interpreter must not look like a video with nobody in it."""
    out = run(face_scores=None)
    assert len(out) == 3
    assert all(p['provenance']['face'] is None for p in out)


def test_a_face_pass_that_refuses_everything_returns_nothing_and_says_so():
    out = run(face_scores=lambda frames: [{'ok': False} for _ in frames])
    assert out == []


def test_an_unreadable_clip_costs_one_clip_not_the_batch():
    def boom(*_a):
        raise RuntimeError('moov atom not found')

    assert run(read_frames=boom) == []


def test_a_clip_with_nothing_admissible_yields_nothing():
    dark = [{'t': i * 0.5, 'luma': 0.0, 'sharp': 100.0} for i in range(10)]
    assert run(read_frames=lambda *a: dark) == []


def test_a_decode_that_returns_nothing_is_not_a_crash():
    assert run(decode=lambda *_a: []) == []


# ── what leaves the module ───────────────────────────────────────────────────

def test_each_frame_carries_bytes_and_traceable_provenance():
    out = run(face_scores=None)
    for item in out:
        assert isinstance(item['bytes'], (bytes, bytearray))
        p = item['provenance']
        assert p['source'] == 'video_frame'
        assert p['clip_id'] == 7 and p['source_id'] == 3
        assert isinstance(p['timestamp_s'], float)
        assert p['sharpness'] is not None


def test_the_timestamp_is_the_frame_that_was_decoded_not_the_one_requested():
    """Pass B lands on the nearest real frame; provenance must say WHICH."""
    def drifting(_path, times):
        return [{'t': t + 0.017, 'bytes': b'x'} for t in times]

    out = run(decode=drifting, face_scores=None)
    assert all(abs(p['provenance']['timestamp_s'] % 0.2 - 0.017) < 1e-6
               for p in out)


def test_results_are_chronological():
    out = run(face_scores=None)
    times = [p['provenance']['timestamp_s'] for p in out]
    assert times == sorted(times)


@pytest.mark.parametrize('limit', [1, 2, 5])
def test_the_budget_is_never_exceeded(limit):
    assert len(run(limit=limit, face_scores=None)) <= limit


# ── the face-reading adapter ─────────────────────────────────────────────────

def test_a_scorable_result_becomes_an_ok_reading():
    r = vfe.face_reading({'state': 'scorable', 'det': 0.8, 'bbox_frac': 0.2,
                          'yaw': 10.0, 'sim': 0.6})
    assert r['ok'] and r['det'] == 0.8 and r['sim'] == 0.6


@pytest.mark.parametrize('state', ['low_det', 'too_small', 'no_face'])
def test_a_non_scorable_result_is_not_ok(state):
    assert vfe.face_reading({'state': state})['ok'] is False


def test_an_empty_result_is_not_ok():
    assert vfe.face_reading(None)['ok'] is False
