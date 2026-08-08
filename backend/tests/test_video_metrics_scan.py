"""🎬 The metrics scan: one decode per clip feeds every measurement.

What is under test here is NOT the arithmetic (test_video_metrics.py owns that)
but the plumbing decisions around the decode:

  * one clip = ONE pass over its frames, motion vectors riding along;
  * the motion-vector option lands on the STREAM's codec context — the container
    form `av.open(path, options=...)` runs without error and yields zero vectors,
    which is precisely how the cheap motion metric would silently become "no
    motion anywhere" (this exact wrong form was written first, and measured);
  * frames are measured at a reduced size, because the Laplacian at full 1080p
    costs more than the decode itself;
  * a clip whose segment cannot be decoded is 'unreadable', never zeros;
  * one broken clip costs that clip, not the scan.

The decode sits behind `_read_clip_frames`; these tests monkeypatch it and stay
green with no PyAV installed. The single integration test that exercises the real
seam skips itself when the extra is absent, and says so.
"""
import pytest

from app.services import video_metrics_scan as scan


def _fake_frames(n=48, luma=0.5, sharp=100.0, motion=0.003):
    return [{'luma': luma, 'sharp': sharp, 'motion': motion} for _ in range(n)]


# --- the scan over a bank -----------------------------------------------------

def test_every_readable_clip_gets_its_summary_stored(app, monkeypatch):
    bank_id = _bank_with_clips(app, 3)
    monkeypatch.setattr(scan, '_read_clip_frames',
                        lambda path, start, end, fps: _fake_frames())

    with app.app_context():
        result = scan.run_metrics(bank_id)

    assert result['measured'] == 3
    stored = _summaries(app, bank_id)
    assert all(s['metrics_state'] == 'ok' for s in stored)
    assert all(s['motion_mean'] is not None for s in stored)


def test_a_clip_that_cannot_be_decoded_costs_that_clip_not_the_scan(app, monkeypatch):
    """A bank is scanned in bulk; one corrupt segment among hundreds must leave
    the other summaries standing."""
    bank_id = _bank_with_clips(app, 3)
    calls = {'n': 0}

    def flaky(path, start, end, fps):
        calls['n'] += 1
        if calls['n'] == 2:
            raise OSError('bitstream error')
        return _fake_frames()
    monkeypatch.setattr(scan, '_read_clip_frames', flaky)

    with app.app_context():
        result = scan.run_metrics(bank_id)

    assert result['measured'] == 2
    assert result['unreadable'] == 1
    states = [s['metrics_state'] for s in _summaries(app, bank_id)]
    assert states.count('ok') == 2 and states.count('unreadable') == 1


def test_a_rerun_skips_what_is_already_measured(app, monkeypatch):
    """Same resume contract as every other pass: stopping loses nothing, and the
    second run only pays for what the first one had not reached."""
    bank_id = _bank_with_clips(app, 3)
    seen = []
    monkeypatch.setattr(scan, '_read_clip_frames',
                        lambda path, start, end, fps: (seen.append(start), _fake_frames())[1])

    with app.app_context():
        scan.run_metrics(bank_id)
        first = len(seen)
        scan.run_metrics(bank_id)

    assert first == 3
    assert len(seen) == 3          # nothing re-measured


def test_a_remeasure_flag_remeasures_everything(app, monkeypatch):
    bank_id = _bank_with_clips(app, 2)
    seen = []
    monkeypatch.setattr(scan, '_read_clip_frames',
                        lambda path, start, end, fps: (seen.append(start), _fake_frames())[1])

    with app.app_context():
        scan.run_metrics(bank_id)
        scan.run_metrics(bank_id, remeasure=True)

    assert len(seen) == 4


def test_the_sharpest_frame_updates_the_thumbnail_timestamp(app, monkeypatch):
    """The middle-of-shot thumbnail was a guess made before any frame had been
    measured. Once the scan knows the sharpest frame, the guess has no reason to
    survive — but only for clips the scan actually measured."""
    bank_id = _bank_with_clips(app, 1)
    frames = _fake_frames(20, sharp=5.0)
    frames[7] = {'luma': 0.5, 'sharp': 400.0, 'motion': 0.003}
    monkeypatch.setattr(scan, '_read_clip_frames',
                        lambda path, start, end, fps: frames)

    with app.app_context():
        scan.run_metrics(bank_id)

    stored = _summaries(app, bank_id)[0]
    assert stored['sharpest_frame_s'] is not None
    assert stored['sharpest_frame_s'] > 0


# --- the motion-vector seam ---------------------------------------------------

def test_the_export_mvs_option_is_set_on_the_stream_not_the_container():
    """The single most expensive quiet mistake available in this lane: the
    container form runs, decodes, raises nothing and yields ZERO vectors — so
    every clip would measure motion_mean == 0 and the still-clip filter would
    flag the entire bank. The stream-level form yields two million vectors on the
    same file. This pins the correct form as DATA, so a refactor that moves the
    option cannot pass review by looking plausible."""
    assert scan.MV_OPTIONS == {'flags2': '+export_mvs'}
    assert scan.MV_OPTIONS_TARGET == 'stream_codec_context'


def test_analysis_frames_are_reduced_before_measuring():
    """A Laplacian over full 1080p costs more than the decode; at the analysis
    size the whole measurement rides inside the decode's own budget. The exact
    size is a knob — what is pinned is that it exists and is sane."""
    assert 64 <= scan.ANALYSIS_WIDTH <= 512


# --- the real seam, when the extra is present ----------------------------------

def test_real_decode_yields_frames_with_all_three_readings(tmp_path):
    """Integration: the one test that runs the true PyAV path, on a tiny
    synthesised clip. Skips — loudly — when the video extra is absent."""
    av = pytest.importorskip('av', reason='video extra not installed')
    import numpy as np

    path = str(tmp_path / 'clip.mp4')
    with av.open(path, 'w') as container:
        vs = container.add_stream('h264', rate=8)
        vs.width, vs.height, vs.pix_fmt = 64, 64, 'yuv420p'
        for i in range(16):
            img = np.full((64, 64, 3), (i * 16) % 255, dtype=np.uint8)
            for packet in vs.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
                container.mux(packet)
        for packet in vs.encode():
            container.mux(packet)

    frames = scan._read_clip_frames(path, 0.0, 2.0, fps=8)

    assert len(frames) >= 8
    for f in frames[:3]:
        assert set(f) >= {'luma', 'sharp', 'motion'}
        assert 0.0 <= f['luma'] <= 1.0


# --- helpers ------------------------------------------------------------------

def _bank_with_clips(app, n):
    """Returns the bank ID, never the ORM object: the app context this opens is
    closed on return, and a detached instance blows up on first attribute read —
    which is a fact about the test, not about the scan."""
    from app.extensions import db
    from app.models import VideoBank, VideoClip, VideoSource
    with app.app_context():
        bank = VideoBank(name='b', source_path='/srv/rushes')
        db.session.add(bank)
        db.session.flush()
        src = VideoSource(bank_id=bank.id, relpath='a.mp4', duration_s=60.0,
                          fps_native=25.0, probe_state='ok')
        db.session.add(src)
        db.session.flush()
        for i in range(n):
            db.session.add(VideoClip(bank_id=bank.id, source_id=src.id,
                                     start_s=float(i * 10), end_s=float(i * 10 + 5)))
        db.session.commit()
        return bank.id


def _summaries(app, bank_id):
    import json
    from app.models import VideoClip
    with app.app_context():
        rows = VideoClip.query.filter_by(bank_id=bank_id).order_by(VideoClip.id).all()
        return [json.loads(r.metrics_json) if r.metrics_json else {} for r in rows]
