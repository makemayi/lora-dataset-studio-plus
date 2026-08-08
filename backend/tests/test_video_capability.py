"""🎬 The `video` extra — where ffmpeg comes from, and what the app says when a
piece of the video lane is missing.

The whole point of this file is the second half. Today a .mp4 dropped into a bank
is skipped in SILENCE: the extension filter does not match, nothing is logged,
and the user is left thinking their file was scanned. The video lane must never
reproduce that, so the probe is required to NAME the piece that is missing.

Nothing here imports av, torch or ffmpeg — these tests describe the app's
behaviour when those are absent, which is the case they exist for.
"""
import sys

import pytest

from app import capabilities
from app.services import ffmpeg_tools


class _FakeImageioFfmpeg:
    def __init__(self, path):
        self._path = path

    def get_ffmpeg_exe(self):
        return self._path


# --- where the binary comes from ----------------------------------------------

def test_ffmpeg_prefers_the_binary_bundled_with_the_extra(monkeypatch, tmp_path):
    """imageio-ffmpeg ships a static binary, which is what makes the extra
    self-sufficient: the user installs nothing by hand."""
    bundled = tmp_path / 'ffmpeg.exe'
    bundled.write_bytes(b'')
    monkeypatch.setitem(sys.modules, 'imageio_ffmpeg',
                        _FakeImageioFfmpeg(str(bundled)))
    monkeypatch.setattr(ffmpeg_tools.shutil, 'which', lambda name: '/usr/bin/ffmpeg')

    assert ffmpeg_tools.ffmpeg_path() == str(bundled)


def test_ffmpeg_falls_back_to_the_one_already_on_the_system(monkeypatch):
    """A user who already has ffmpeg installed should not be made to download a
    second copy — the scraper already resolves it this way."""
    monkeypatch.setitem(sys.modules, 'imageio_ffmpeg', None)
    monkeypatch.setattr(ffmpeg_tools.shutil, 'which', lambda name: '/usr/bin/ffmpeg')

    assert ffmpeg_tools.ffmpeg_path() == '/usr/bin/ffmpeg'


def test_a_bundled_path_that_is_not_on_disk_is_not_believed(monkeypatch):
    """imageio-ffmpeg answers with a path whether or not the download completed.
    Trusting it blindly turns a half-finished install into 'ffmpeg not found'
    errors from deep inside an encode, instead of a missing-extra message."""
    monkeypatch.setitem(sys.modules, 'imageio_ffmpeg',
                        _FakeImageioFfmpeg('/nowhere/ffmpeg'))
    monkeypatch.setattr(ffmpeg_tools.shutil, 'which', lambda name: '/usr/bin/ffmpeg')

    assert ffmpeg_tools.ffmpeg_path() == '/usr/bin/ffmpeg'


def test_no_ffmpeg_anywhere_answers_none_instead_of_raising(monkeypatch):
    """Callers decide what to do about it; resolving must not be the thing that
    explodes."""
    monkeypatch.setitem(sys.modules, 'imageio_ffmpeg', None)
    monkeypatch.setattr(ffmpeg_tools.shutil, 'which', lambda name: None)

    assert ffmpeg_tools.ffmpeg_path() is None


# --- resolving a path is NOT the same as being able to encode ------------------

@pytest.fixture(autouse=True)
def _fresh_encoder_verdict():
    """The encoder verdict is cached (it costs a subprocess); no test may read
    another test's answer."""
    ffmpeg_tools.clear_cache()
    yield
    ffmpeg_tools.clear_cache()


def _fake_ffmpeg_run(monkeypatch, *, returncode=0, stderr='', boom=None):
    def fake_run(cmd, **kw):
        if boom is not None:
            raise boom
        class _P:
            pass
        p = _P()
        p.returncode = returncode
        p.stderr = stderr
        p.stdout = ''
        return p
    monkeypatch.setattr(ffmpeg_tools.subprocess, 'run', fake_run)


def test_a_binary_that_is_there_but_cannot_run_is_not_an_encoder(monkeypatch, tmp_path):
    """The half-installed extra, exactly: imageio-ffmpeg is installed, a file sits
    at the resolved path — and it is a truncated download, an emptied antivirus
    stub, or a file with no execute bit. os.path.isfile says yes to all three and
    the export then dies mid-encode."""
    stub = tmp_path / 'ffmpeg.exe'
    stub.write_bytes(b'')
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: str(stub))
    _fake_ffmpeg_run(monkeypatch, returncode=1, stderr='not a valid executable')

    verdict = ffmpeg_tools.ffmpeg_ready()

    assert verdict['ok'] is False
    assert 'does not run' in verdict['reason']


def test_a_binary_that_cannot_even_be_launched_is_not_an_encoder(monkeypatch, tmp_path):
    stub = tmp_path / 'ffmpeg.exe'
    stub.write_bytes(b'')
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: str(stub))
    _fake_ffmpeg_run(monkeypatch, boom=OSError('Exec format error'))

    verdict = ffmpeg_tools.ffmpeg_ready()

    assert verdict['ok'] is False
    assert 'could not be launched' in verdict['reason']


def test_an_ffmpeg_that_answers_is_ready(monkeypatch, tmp_path):
    stub = tmp_path / 'ffmpeg.exe'
    stub.write_bytes(b'')
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: str(stub))
    _fake_ffmpeg_run(monkeypatch, returncode=0)

    assert ffmpeg_tools.ffmpeg_ready()['ok'] is True
    assert ffmpeg_tools.has_ffmpeg() is True


def test_a_slow_ffmpeg_is_not_called_broken(monkeypatch, tmp_path):
    """Same rule as the cold-import probes: an unproven absence must never turn a
    working install red (an on-access antivirus scan of a 70 MB binary is the
    ordinary case here)."""
    import subprocess as _sp
    stub = tmp_path / 'ffmpeg.exe'
    stub.write_bytes(b'')
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: str(stub))
    _fake_ffmpeg_run(monkeypatch, boom=_sp.TimeoutExpired('ffmpeg', 1))

    assert ffmpeg_tools.ffmpeg_ready()['ok'] is True


def test_no_binary_at_all_says_where_it_looked(monkeypatch):
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: None)

    verdict = ffmpeg_tools.ffmpeg_ready()

    assert verdict['ok'] is False
    assert 'imageio-ffmpeg' in verdict['reason'] and 'PATH' in verdict['reason']


def test_the_encoder_is_not_re_run_on_every_capability_poll(monkeypatch, tmp_path):
    """probe_video() runs on every /api/capabilities call; spawning ffmpeg each
    time would be a process per poll."""
    stub = tmp_path / 'ffmpeg.exe'
    stub.write_bytes(b'')
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: str(stub))
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        class _P:
            returncode = 0
            stderr = ''
            stdout = ''
        return _P()

    monkeypatch.setattr(ffmpeg_tools.subprocess, 'run', fake_run)
    ffmpeg_tools.ffmpeg_ready()
    ffmpeg_tools.ffmpeg_ready()
    assert len(calls) == 1

    # …but an install that just fixed it must not wait out the TTL to show green.
    capabilities.clear_import_cache()
    ffmpeg_tools.ffmpeg_ready()
    assert len(calls) == 2


def test_the_probe_and_the_installer_share_one_definition_of_a_working_encoder():
    """If they drift, an install can report success about the exact row it left
    ✗ — the #24 shape applied to video."""
    from app import setup_installer
    assert setup_installer._CAPABILITY_EXTRA_CHECKS['video'] is \
        setup_installer._verify_video_encoder


# --- what the app says is missing ---------------------------------------------

def _probe(monkeypatch, *, decode=True, detect=True, encode=True, seen=None,
           reason='no ffmpeg binary found'):
    # Route on the probe KEY, not on the import expression: the detect expression
    # legitimately contains 'av' too, and matching on it made both pieces answer
    # with the same value.
    def fake_import(key, python, expr):
        if seen is not None:
            seen[key] = python
        return decode if key.endswith('decode') else detect

    monkeypatch.setattr(capabilities, '_cached_import', fake_import)
    monkeypatch.setattr(capabilities.ffmpeg_tools, 'ffmpeg_ready',
                        lambda: {'ok': encode,
                                 'path': '/usr/bin/ffmpeg' if encode else None,
                                 'reason': 'ffmpeg runs' if encode else reason})
    return capabilities.probe_video()


def test_the_probe_is_ok_only_when_every_piece_is_present(monkeypatch):
    assert _probe(monkeypatch)['ok'] is True


@pytest.mark.parametrize('missing, word', [
    ('decode', 'av'),
    ('detect', 'shot detection'),
    ('encode', 'ffmpeg'),
])
def test_the_probe_names_the_piece_that_is_missing(monkeypatch, missing, word):
    """Three different absences, three different reasons. Collapsing them into one
    'video unavailable' is how a user ends up reinstalling the wrong thing."""
    result = _probe(monkeypatch, **{missing: False})

    assert result['ok'] is False
    assert word in result['detail'].lower()


def test_shot_detection_rides_the_environment_that_already_has_torch(monkeypatch):
    """TransNetV2 needs torch, and the app already manages an environment holding
    it for bank scoring. Installing a second copy would cost the user ~2.5 GB for
    nothing — the watermark detector settled this exact question the same way, and
    the probe has to resolve where the installer puts things or the two drift.

    Decoding is the opposite case: PyAV is imported IN-PROCESS by Flask, so it has
    to be the app's own interpreter, not the scoring one.
    """
    monkeypatch.setattr(capabilities.cfg, 'get',
                        lambda key, *a, **k: '/envs/scoring/python'
                        if key == 'bank_scoring.python' else None)
    seen = {}
    _probe(monkeypatch, seen=seen)

    assert seen['video_detect'] == '/envs/scoring/python'
    assert seen['video_decode'] == sys.executable


def test_the_probe_says_WHY_encoding_is_unavailable(monkeypatch):
    """'ffmpeg missing' and 'ffmpeg present but broken' are fixed differently —
    one is an install, the other a re-download or an antivirus exclusion."""
    result = _probe(monkeypatch, encode=False,
                    reason='the ffmpeg at ~/x/ffmpeg.exe exists but does not run (exit 1)')

    assert 'does not run' in result['detail']


def test_the_probe_reports_each_piece_separately(monkeypatch):
    """The UI needs the parts, not just the sentence: decoding a bank and encoding
    a dataset are different actions and can be offered independently."""
    result = _probe(monkeypatch, encode=False)

    assert result['decode'] is True
    assert result['detect'] is True
    assert result['encode'] is False
