"""Reading what a source video IS, and pulling one frame to look at.

In-process through PyAV rather than an ffprobe subprocess. That is not a
preference: `imageio-ffmpeg`, the package that makes the video extra
self-sufficient, bundles ffmpeg WITHOUT ffprobe. An ffprobe-based probe would
work on a developer machine — which has a full install on PATH — and fail on
precisely the machine the extra was written for.

Everything here is per-FILE. Per-clip facts live on VideoClip; the split exists
because a probe that fails must fail once per file rather than once per shot.

`fps_native` is the SOURCE's frame rate and is never what a clip is encoded at.
The target profile decides that. Reading this value at encode time is how a 16 fps
target ends up with accelerated motion, so it is stored to be DISPLAYED and to
convert the detector's frame indices — never to encode from.
"""
import os

# How far into a shot the thumbnail frame is taken, as a fraction of its length.
# The middle, not the start: a shot boundary is where a cut just happened, so the
# opening frames routinely carry the tail of a dissolve or a black frame — the two
# things that make a grid of thumbnails useless.
_THUMB_POSITION = 0.5


def _open(path):
    """The single decode seam. Imported lazily so this module stays importable on
    an install with no video extra — the capability probe is what tells the user,
    not an ImportError at startup."""
    import av
    return av.open(path)


def _duration_seconds(container, stream):
    """Seconds, or None. NEVER zero as a stand-in for "unknown": zero reads as a
    file with no content and makes every later length check pass or fail for the
    wrong reason."""
    if container.duration:
        # PyAV reports the container duration in microseconds.
        return container.duration / 1_000_000
    if stream.duration and stream.time_base:
        return float(stream.duration * stream.time_base)
    return None


def probe(path):
    """What this file is. Always returns a dict; never raises.

    A bank is scanned in bulk, so one corrupt file among four hundred must cost
    that file and not the scan. `probe_state` carries the verdict:
      'ok'         — geometry below is usable
      'unreadable' — could not be opened, or holds no video stream
    """
    result = {'duration_s': None, 'fps_native': None, 'width': None,
              'height': None, 'codec': None, 'probe_state': 'unreadable',
              'file_size': None}
    try:
        result['file_size'] = os.path.getsize(path)
    except OSError:
        pass
    try:
        container = _open(path)
    except Exception:                       # noqa: BLE001 — any decode error
        return result
    try:
        streams = container.streams.video
        if not streams:
            # An .mp4 that holds only audio opens perfectly and has nothing to cut.
            return result
        stream = streams[0]
        result.update(
            width=stream.width,
            height=stream.height,
            codec=getattr(stream.codec_context, 'name', None),
            # float() of a Fraction, not round(): 29.97 is 30000/1001, and calling
            # it 30 drifts about a second every thousand.
            fps_native=float(stream.average_rate) if stream.average_rate else None,
            duration_s=_duration_seconds(container, stream),
            probe_state='ok',
        )
        return result
    except Exception:                       # noqa: BLE001
        result['probe_state'] = 'unreadable'
        return result
    finally:
        try:
            container.close()
        except Exception:                   # noqa: BLE001
            pass


def thumbnail_timestamp(start_s, end_s):
    """The instant a shot's thumbnail is grabbed from — its middle.

    Choosing the shot's FIRST frame is the tempting shortcut and the wrong one: a
    boundary is where a cut just happened, so that frame is disproportionately a
    dissolve or a black frame.
    """
    return start_s + (end_s - start_s) * _THUMB_POSITION
