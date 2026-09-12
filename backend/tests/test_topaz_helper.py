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
    """The input is a POSITIONAL path (there is no -i flag; it exits 127), and
    by default only the upscale toggle is sent — Autopilot decides the rest."""
    cmd = th.build_command('tpai.exe', 'in.png', 'out', format='png')
    assert cmd[0] == 'tpai.exe'
    assert cmd[1] == 'in.png'            # positional input, NOT '-i'
    assert '-i' not in cmd
    assert cmd[cmd.index('-o') + 1] == 'out'
    assert '--format' in cmd and 'png' in cmd
    assert '--upscale' in cmd and cmd[cmd.index('--upscale') + 1] == 'enabled=true'
    # Autopilot decides denoise/sharpen/lighting/color unless forced.
    assert '--noise' not in cmd and '--sharpen' not in cmd


def test_build_command_can_force_toggles():
    cmd = th.build_command('tpai.exe', 'in.png', 'out',
                           denoise=False, sharpen=True)
    assert '--noise' in cmd and cmd[cmd.index('--noise') + 1] == 'enabled=false'
    assert cmd[cmd.index('--sharpen') + 1] == 'enabled=true'


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


def test_build_command_face_recovery_off_and_strength():
    """The verified --faceRecovery channel: False -> enabled=false; a float
    -> enabled=true + strength (both verified on REAL runs — param1= also
    writes through --showSettings but breaks the engine: 'Error running
    model', exit 0, no output)."""
    cmd = th.build_command('tpai.exe', 'in.png', 'out', face_recovery=False)
    assert cmd[cmd.index('--faceRecovery') + 1] == 'enabled=false'
    cmd = th.build_command('tpai.exe', 'in.png', 'out', face_recovery=0.3)
    i = cmd.index('--faceRecovery')
    assert cmd[i + 1:i + 3] == ['enabled=true', 'strength=0.3']
    # None (Autopilot) sends nothing at all.
    assert '--faceRecovery' not in th.build_command('tpai.exe', 'in.png', 'out')


def test_face_recovery_tiers_scale_down_as_faces_grow():
    """Topaz's own docs: faces >=~512px get over-processed into plastic; only
    smaller faces benefit, and smaller still -> a bit more strength."""
    assert th.face_recovery_value(640) is False
    assert th.face_recovery_value(512) is False
    assert th.face_recovery_value(256) == 0.30
    assert th.face_recovery_value(128) == 0.50
    assert th.face_recovery_value(64) == 0.70


def test_plan_tiers_by_largest_face(monkeypatch):
    """Each image is tiered by its LARGEST face's short side; no-face images
    are decided OFF, never handed to Autopilot's 0.8-everything default."""
    from app.services import face_mask
    paths = ['big.png', 'small.png', 'noface.png']
    boxes = {'big.png': [[0, 0, 800, 600]],      # short side 600 -> off
             'small.png': [[0, 0, 300, 200]],    # short side 200 -> 0.50
             'noface.png': []}
    monkeypatch.setattr(th, '_decodable', lambda p: True)
    monkeypatch.setattr(face_mask, 'is_available', lambda: True)
    monkeypatch.setattr(face_mask, 'detect_faces', lambda imgs, timeout=900:
                        {'ok': True, 'results': {p: {'state': 'x', 'boxes': boxes[p]}
                                                 for p in paths}})
    assert th.plan_face_recovery(paths) == {'big.png': False, 'small.png': 0.50,
                                            'noface.png': False}


def test_plan_face_recovery_fails_off_when_detector_unavailable(monkeypatch):
    """Detector missing or detection failed -> everything OFF. A broken
    detector must never silently upgrade itself into the wax default."""
    from app.services import face_mask
    monkeypatch.setattr(th, '_decodable', lambda p: True)
    monkeypatch.setattr(face_mask, 'is_available', lambda: False)
    assert th.plan_face_recovery(['a.png']) == {'a.png': False}
    monkeypatch.setattr(face_mask, 'is_available', lambda: True)
    monkeypatch.setattr(face_mask, 'detect_faces', lambda imgs, timeout=900:
                        {'ok': False, 'error': 'boom'})
    assert th.plan_face_recovery(['a.png']) == {'a.png': False}


def test_plan_skips_detector_for_undecodable_paths():
    """Non-files / fake bytes are decided OFF in-process (no subprocess)."""
    assert th.plan_face_recovery(['C:/nope/x.png']) == {'C:/nope/x.png': False}


def test_resolve_face_recovery_modes():
    assert th.resolve_face_recovery(True, ['p']) is None      # Autopilot
    assert th.resolve_face_recovery(None, ['p']) is None      # Autopilot
    assert th.resolve_face_recovery(False, ['p']) is False    # forced off
    plan = th.resolve_face_recovery('smart', ['p'])           # default: smart
    assert plan['p'] is False                                 # undecodable -> off
