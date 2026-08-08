"""Where the ffmpeg binary comes from.

The video lane needs ffmpeg for exactly ONE step — encoding a kept clip into the
dataset — and nothing else. Probing a file, reading its timestamps and pulling a
thumbnail frame all happen in-process through PyAV, so a missing binary disables
the export and leaves scanning, detection and triage working. That split is only
possible if "do we have ffmpeg?" is a question with its own answer, which is what
this module is.

Two sources, in order:

1. `imageio-ffmpeg`, a pip package that ships a static binary. This is what makes
   the extra self-sufficient — a Windows user who has never heard of ffmpeg
   installs the extra and the export works. It is why the video extra exists as a
   pip install at all rather than a "please download ffmpeg" note.
2. Whatever is on PATH. A user who already runs ffmpeg should not be made to
   download a second copy; the scraper already resolves it this way
   (scrape/netfetch.py).

NOT ffprobe. imageio-ffmpeg bundles ffmpeg alone, so an ffprobe-based inspection
would work on the developer's machine — where a full ffmpeg install is on PATH —
and fail on the install the extra was written for. PyAV answers the same
questions in-process, and the video lane needs it for decoding anyway.

RESOLVING a path and being ABLE TO ENCODE are two different questions, and only
the second one is what the Setup row and the installer are allowed to answer
with. `os.path.isfile` says yes about an interrupted 40-byte download, about a
binary an antivirus has emptied into a stub, and about a file with no execute
bit in a container. Each of those turns into an ffmpeg crash from the middle of
an export — the very failure the split above exists to prevent — so
`ffmpeg_ready()` runs the binary once and believes the exit code, not the
directory entry.
"""
import os
import shutil
import subprocess
import time

from ..utils.redact import redact_user_paths


def ffmpeg_path():
    """Absolute path to a usable ffmpeg binary, or None.

    Never raises: callers decide what a missing encoder means for them, and
    resolution itself must not be the thing that fails.
    """
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        # imageio-ffmpeg answers with a path whether or not its download ever
        # finished. Believing it turns a half-installed extra into "ffmpeg not
        # found" raised from inside an encode, instead of the missing-extra
        # message the user can act on.
        if bundled and os.path.isfile(bundled):
            return bundled
    except Exception:
        pass                      # not installed, or broken — fall through to PATH
    return shutil.which('ffmpeg')


# `ffmpeg -version` prints a banner and exits in ~50 ms. The budget is generous
# only so a cold first run behind an on-access antivirus scan is not called a
# failure.
_PROBE_TIMEOUT = 20
# The verdict is cached because probe_video() runs on EVERY /api/capabilities
# poll and must not spawn a process each time. Same shape and lifetime as the
# import-probe cache in capabilities.py, and cleared by the same
# clear_import_cache() so a fresh install flips the row without a restart.
_READY_TTL = 600.0
_ready_cache = None      # (ts, verdict-dict)


def clear_cache() -> None:
    """Forget the cached encoder verdict (called right after an install)."""
    global _ready_cache
    _ready_cache = None


def ffmpeg_ready(force: bool = False) -> dict:
    """Can this process actually encode? -> {'ok', 'path', 'reason'}.

    This is the ONE definition of "the encoder works", shared by the Setup probe
    and by the installer's post-install check — when those two drift, an install
    reports success while the row it was supposed to turn green stays ✗ and the
    user reinstalls the wrong half (issue #24's shape, applied to video).

    A binary that does not ANSWER (timeout) counts as present: an unproven
    absence must never turn a working install red, exactly like the cold-import
    timeouts elsewhere. Never raises.
    """
    global _ready_cache
    if not force and _ready_cache is not None:
        ts, verdict = _ready_cache
        if time.time() - ts < _READY_TTL:
            return dict(verdict)
    verdict = _measure_ready()
    _ready_cache = (time.time(), verdict)
    return dict(verdict)


def _measure_ready() -> dict:
    # Paths reach the Setup panel and pasted diagnostics, so the account name is
    # stripped out of every SENTENCE. `path` itself stays real — callers encode
    # with it.
    path = ffmpeg_path()
    shown = redact_user_paths(path or '')
    if not path:
        return {'ok': False, 'path': None,
                'reason': 'no ffmpeg binary found — imageio-ffmpeg is not installed '
                          'in this Python, and there is none on PATH'}
    try:
        proc = subprocess.run([path, '-version'], capture_output=True, text=True,
                              encoding='utf-8', errors='replace',
                              timeout=_PROBE_TIMEOUT,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        return {'ok': True, 'path': path,
                'reason': 'ffmpeg was slow to answer — treated as usable'}
    except Exception as e:            # noqa: BLE001 — cannot launch it at all
        return {'ok': False, 'path': path,
                'reason': f'the ffmpeg at {shown} could not be launched: {e}'}
    if proc.returncode == 0:
        return {'ok': True, 'path': path, 'reason': 'ffmpeg runs'}
    tail = ((proc.stderr or proc.stdout or '').strip().splitlines() or [''])[-1]
    return {'ok': False, 'path': path,
            'reason': f'the ffmpeg at {shown} exists but does not run '
                      f'(exit {proc.returncode}{": " + tail if tail else ""}) — a '
                      'truncated download or a quarantined binary looks exactly like this'}


def has_ffmpeg():
    """True when an encode is possible at all."""
    return ffmpeg_ready()['ok']
