"""🎬 Reading what a source video IS, and pulling one frame to look at.

Both go through PyAV in-process rather than an ffprobe subprocess, for a reason
worth stating once: imageio-ffmpeg — the package that makes the video extra
self-sufficient — bundles ffmpeg WITHOUT ffprobe. An ffprobe-based probe would
work on a developer machine with a full install on PATH and fail on exactly the
machine the extra exists for.

No real video is decoded here. The decode is behind one seam so that the
judgements around it — what "unreadable" means, where the frame rate comes from,
which frame gets picked — stay testable on an install with no video extra at all.
"""
import pytest

from app.services import video_probe as vp


class _FakeStream:
    def __init__(self, **kw):
        self.type = kw.pop('type', 'video')
        self.average_rate = kw.pop('average_rate', 30)
        self.duration = kw.pop('duration', None)
        self.time_base = kw.pop('time_base', None)
        self.frames = kw.pop('frames', 0)
        self.width = kw.pop('width', 1920)
        self.height = kw.pop('height', 1080)
        self.codec_context = type('C', (), {'name': kw.pop('codec', 'h264')})()


class _FakeContainer:
    def __init__(self, streams, duration=None):
        self.streams = type('S', (), {'video': [s for s in streams
                                                if s.type == 'video']})()
        self.duration = duration      # microseconds, like PyAV

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patched(monkeypatch, container):
    monkeypatch.setattr(vp, '_open', lambda path: container)


# --- what a probe reports -----------------------------------------------------

def test_a_readable_file_reports_its_geometry(monkeypatch):
    _patched(monkeypatch, _FakeContainer([_FakeStream(width=1280, height=720)],
                                         duration=142_500_000))
    result = vp.probe('/src/a.mp4')

    assert result['probe_state'] == 'ok'
    assert (result['width'], result['height']) == (1280, 720)
    assert result['codec'] == 'h264'
    assert result['duration_s'] == pytest.approx(142.5)


def test_a_fractional_frame_rate_is_not_rounded(monkeypatch):
    """29.97 fps is 30000/1001, and PyAV hands it over as a Fraction. Rounding it
    to 30 makes every timestamp drift — about one second per thousand."""
    from fractions import Fraction
    _patched(monkeypatch, _FakeContainer(
        [_FakeStream(average_rate=Fraction(30000, 1001))], duration=1_000_000))

    assert vp.probe('/src/a.mp4')['fps_native'] == pytest.approx(29.97, abs=0.01)


def test_a_file_that_cannot_be_opened_is_unreadable_not_an_exception(monkeypatch):
    """A bank is scanned in bulk. One corrupt file among four hundred must cost
    that file, not the scan."""
    def boom(path):
        raise OSError('moov atom not found')
    monkeypatch.setattr(vp, '_open', boom)

    assert vp.probe('/src/broken.mp4')['probe_state'] == 'unreadable'


def test_a_file_with_no_video_stream_is_unreadable(monkeypatch):
    """An .mp4 holding only audio opens fine and has nothing to cut."""
    _patched(monkeypatch, _FakeContainer([_FakeStream(type='audio')]))

    assert vp.probe('/src/podcast.mp4')['probe_state'] == 'unreadable'


def test_a_missing_duration_is_none_rather_than_zero(monkeypatch):
    """Some containers carry no duration. Zero would read as 'a file with no
    content' and make every later length check pass or fail for the wrong
    reason."""
    _patched(monkeypatch, _FakeContainer([_FakeStream()], duration=None))

    assert vp.probe('/src/stream.mp4')['duration_s'] is None


def test_the_stream_duration_is_used_when_the_container_has_none(monkeypatch):
    """Concatenated or remuxed files routinely carry the duration on the stream
    only."""
    from fractions import Fraction
    _patched(monkeypatch, _FakeContainer(
        [_FakeStream(duration=3000, time_base=Fraction(1, 1000))], duration=None))

    assert vp.probe('/src/remuxed.mp4')['duration_s'] == pytest.approx(3.0)


# --- which frame becomes the thumbnail ----------------------------------------

def test_the_thumbnail_is_taken_from_the_middle_of_the_shot():
    """Not the first frame. A shot boundary is where a cut just happened, so the
    opening frames routinely carry the tail of a dissolve or a black frame — the
    two things that make a grid of thumbnails useless."""
    assert vp.thumbnail_timestamp(41.2, 46.2) == pytest.approx(43.7)


def test_a_degenerate_shot_still_yields_a_timestamp():
    """Zero-length bounds should not produce a NaN that propagates into a seek."""
    assert vp.thumbnail_timestamp(10.0, 10.0) == pytest.approx(10.0)
