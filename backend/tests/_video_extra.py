"""Make the video extra READY for a test that is not about the video extra.

Every `/api/video-bank/...` route that decodes, encodes or detects opens with the
same gate: `capabilities.probe_video()`, and a 503 carrying `detail` when a piece
is missing. That gate reads the MACHINE — PyAV importable, ffmpeg on PATH,
transnetv2 in the scoring interpreter — so a route test's verdict depends on what
the box running it happens to have installed.

On the maintainer's machine everything is present and the tests pass. CI installs
`requirements-dev.txt` and nothing else, so the gate fires first and five tests
that never mention the video extra fail on it: a 202 arrives as 503, a 404 as 503,
and the one test that asserts its OWN 503 gets the wrong sentence. That surfaced
on a release tag, because CI path-gates the heavy job and a small commit never
runs it.

`pytest.importorskip` would be the wrong tool: it would delete the coverage in the
one environment where it is cheapest to run. These tests do not need PyAV, ffmpeg
or a GPU — every service call underneath is already stubbed. They only need the
gate to stop answering for the machine.

Tests that are ABOUT a missing piece override this with their own `monkeypatch`
(see test_video_advisory_routes.py, which sets `decode: False` to assert the very
sentence this fixture suppresses). Last patch wins, so nothing here weakens them.
"""
import pytest


@pytest.fixture(autouse=True)
def video_extra_ready(monkeypatch):
    """Report every video piece as installed, for this module only."""
    monkeypatch.setattr('app.capabilities.probe_video', lambda: {
        'ok': True, 'decode': True, 'detect': True, 'encode': True,
        'detail': 'video extra ready',
    })
