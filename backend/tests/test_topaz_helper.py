"""Topaz Photo AI helper — exe discovery, CLI command building, return codes."""
import pytest

from app.services import topaz_helper as th


def test_standard_paths_include_common_installs():
    """The probe list must cover the user's install layout and the usual ones."""
    joined = ' / '.join(th.STANDARD_TPAI_PATHS)
    assert 'Topaz Photo AI' in joined and 'tpai.exe' in joined


def test_resolve_exe_prefers_config_override(monkeypatch):
    monkeypatch.setattr(th.cfg, 'get', lambda key, *a: 'C:/custom/tpai.exe')
    assert th.resolve_exe() == 'C:/custom/tpai.exe'


def test_resolve_exe_returns_none_when_nothing_exists(monkeypatch):
    monkeypatch.setattr(th.cfg, 'get', lambda key, *a: '')
    monkeypatch.setattr(th.os.path, 'isfile', lambda p: False)
    assert th.resolve_exe() is None


def test_build_command_upscale_only():
    cmd = th.build_command('tpai.exe', 'in.png', 'out', format='png')
    assert cmd[0] == 'tpai.exe'
    assert 'in.png' in cmd
    assert cmd[cmd.index('-o') + 1] == 'out'
    assert '--format' in cmd and 'png' in cmd
    assert '--upscale' in cmd and cmd[cmd.index('--upscale') + 1] == 'enabled=true'


def test_run_tpai_maps_return_codes(monkeypatch):
    import subprocess

    state = {'code': 0}

    def fake_run(cmd, **kw):
        return type('P', (), {'returncode': state['code'],
                              'stdout': '', 'stderr': ''})

    monkeypatch.setattr(subprocess, 'run', fake_run)

    for code, expected in ((0, 'ok'), (1, 'partial'), (-1, 'no_valid_files'),
                           (-2, 'license'), (-3, 'bad_args'), (99, 'unknown')):
        state['code'] = code
        status, _ = th.run_tpai('tpai.exe', 'in.png', 'out')
        assert status == expected, f'rc={code} -> {status}'


def test_run_tpai_license_message_is_actionable(monkeypatch):
    import subprocess

    class _P:
        returncode = -2
        stdout = ''
        stderr = ''

    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: _P())
    status, message = th.run_tpai('tpai.exe', 'in.png', 'out')
    assert status == 'license'
    assert 'open Topaz' in message
