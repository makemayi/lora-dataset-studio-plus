"""An ML capability install that does not actually work must not report success.

Reported by 1Tomber (GitHub #24) on Linux/Python 3.12: installing "Person masks"
from Setup ▸ Install/repair individually reported success — every requirement
"already satisfied" — while the capability stayed ✗ Not installed and masked
training silently degraded to unmasked. Three defects stacked:

  1. the masks install set omitted onnxruntime, which rembg ≥2.0.50 imports at
     load but no longer declares;
  2. the install reported success without ever re-checking the capability, so the
     app said one thing while knowing another — and the import traceback, which
     names the missing module, was thrown away;
  3. the resulting run trained unmasked with no pre-launch warning.

These tests pin all three, plus the constraint that the fix must not reinstall
onnxruntime on the very many machines that already have one — least of all
replace a user's onnxruntime-gpu with the CPU build.

No test here touches the real network or a real pip: every subprocess seam is
stubbed.
"""
import subprocess

import pytest


# --- 1. the missing dependency ------------------------------------------------

def test_masks_install_set_carries_onnxruntime():
    """rembg imports onnxruntime at module load and stopped declaring it, so the
    scoped masks install has to name it or the capability cannot come up."""
    from app import setup_installer
    canon = {setup_installer._canon(p)
             for p in setup_installer._CAPABILITY_PACKAGES['masks']}
    assert 'onnxruntime' in canon


def test_every_capability_install_set_covers_its_own_probe_imports():
    """The family guard. A capability whose probe is a FULL import of module X
    while X is absent from its install set is exactly the #24 shape, and it would
    fail the same way. Checked for the top-level modules the probes import that
    the ML requirements file knows as packages."""
    from app import setup_installer, capabilities
    # module name -> the distribution that provides it, for the ML packages whose
    # import name differs from (or matches) the distribution name.
    provider = {'rembg': 'rembg', 'onnxruntime': 'onnxruntime',
                'insightface': 'insightface', 'cv2': 'opencv-python-headless',
                'numpy': 'numpy', 'simple_lama_inpainting': 'simple-lama-inpainting'}
    for action, pkgs in setup_installer._CAPABILITY_PACKAGES.items():
        expr = capabilities.CAPABILITY_IMPORTS.get(action)
        if not expr:
            continue
        owned = {setup_installer._canon(p) for p in pkgs}
        for mod in expr.replace('import', '').replace(',', ' ').split():
            dist = provider.get(mod)
            if dist is None:
                continue
            assert setup_installer._canon(dist) in owned, (
                f'{action}: its probe imports {mod} but its install set does not '
                f'carry {dist} — an install would report success and the '
                f'capability would stay off (the #24 defect)')


# --- 2. an install that does not work must not say it worked -------------------

def _stub_pip(monkeypatch, setup_installer, rc=0):
    seen = {}

    def fake_run_pip(action, cmd):
        seen['cmd'] = cmd
        return rc

    monkeypatch.setattr(setup_installer, '_run_pip', fake_run_pip)
    return seen


def _stub_import(monkeypatch, setup_installer, returncode, stderr=''):
    """Stub the verification subprocess only — pip is stubbed separately."""
    class _Proc:
        pass

    def fake_run(cmd, **kw):
        p = _Proc()
        p.returncode = returncode
        p.stderr = stderr
        p.stdout = ''
        return p

    monkeypatch.setattr(setup_installer.subprocess, 'run', fake_run)
    monkeypatch.setattr(setup_installer.os.path, 'isfile', lambda p: True)


REMBG_TRACEBACK = (
    'Traceback (most recent call last):\n'
    '  File "<string>", line 1, in <module>\n'
    '  File "/opt/env/lib/python3.12/site-packages/rembg/__init__.py", line 3, in <module>\n'
    '    from .bg import remove\n'
    '  File "/opt/env/lib/python3.12/site-packages/rembg/bg.py", line 7, in <module>\n'
    '    import onnxruntime as ort\n'
    "ModuleNotFoundError: No module named 'onnxruntime'\n"
)


def test_install_that_leaves_the_capability_broken_fails_and_says_why(app, monkeypatch):
    """THE reported scenario, replayed: pip reports success (everything "already
    satisfied") while `import rembg` dies on onnxruntime. Before the fix this
    returned 0 — "installed successfully" next to "✗ Not installed", with the one
    useful line (the module name) discarded."""
    from app import setup_installer, config
    lines = []
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: lines.append(l))
    _stub_pip(monkeypatch, setup_installer, rc=0)
    _stub_import(monkeypatch, setup_installer, returncode=1, stderr=REMBG_TRACEBACK)
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        rc = setup_installer._run_ml_capability('masks')
    assert rc == 1, 'a capability that does not import must not report success'
    log = '\n'.join(lines)
    assert 'missing module: onnxruntime' in log, (
        'the import traceback names the missing module — it is the most useful '
        'fact in the whole chain and it must reach the screen')
    assert 'person masks' in log.lower()


def test_install_whose_import_works_reports_success(app, monkeypatch):
    from app import setup_installer, config
    lines = []
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: lines.append(l))
    _stub_pip(monkeypatch, setup_installer, rc=0)
    _stub_import(monkeypatch, setup_installer, returncode=0)
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        rc = setup_installer._run_ml_capability('masks')
    assert rc == 0
    assert any('import OK' in l for l in lines)


def test_a_slow_cold_import_is_warming_not_a_failure(app, monkeypatch):
    """The first `import rembg` after an install compiles caches while the AV
    scans fresh DLLs (~20 s measured). Timing out must never fail a good install
    — that is the mirror-image lie."""
    from app import setup_installer, config
    lines = []
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: lines.append(l))
    _stub_pip(monkeypatch, setup_installer, rc=0)

    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(setup_installer.subprocess, 'run', boom)
    monkeypatch.setattr(setup_installer.os.path, 'isfile', lambda p: True)
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        rc = setup_installer._run_ml_capability('masks')
    assert rc == 0
    assert any('warming' in l for l in lines)


def test_the_verification_runs_the_exact_import_the_probe_runs():
    """If the two ever drift, "install OK" and "capability ✗" can disagree again.
    One dict, used by both."""
    from app import setup_installer, capabilities
    for action in setup_installer._CAPABILITY_ML_ACTIONS:
        assert capabilities.CAPABILITY_IMPORTS.get(action)
    assert capabilities.CAPABILITY_IMPORTS['masks'] == 'import rembg'


# --- 3. don't break the installs that already work -----------------------------

def test_onnxruntime_is_not_reinstalled_when_the_env_already_imports_one(app, monkeypatch):
    """Idempotence AND a performance guard in one rule. onnxruntime, -gpu,
    -directml and -silicon are different distributions providing the same module;
    pip does not know they conflict, so installing the CPU one over a user's CUDA
    build "succeeds" and silently costs them their accelerator."""
    from app import setup_installer, config
    lines = []
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: lines.append(l))
    seen = _stub_pip(monkeypatch, setup_installer, rc=0)
    monkeypatch.setattr(setup_installer, '_onnxruntime_provided', lambda p: True)
    monkeypatch.setattr(setup_installer, '_verify_capability_import', lambda a, p: True)
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        setup_installer._run_ml_capability('masks')
        rembg = setup_installer._requirement_spec('rembg')
    cmd = seen['cmd']
    assert rembg in cmd
    assert not any('onnxruntime' in str(c) for c in cmd), (
        'a working onnxruntime (possibly the GPU build) must be left alone'
    )
    assert any('already imports' in l for l in lines)


def test_onnxruntime_is_installed_when_the_env_cannot_import_it(app, monkeypatch):
    from app import setup_installer, config
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: None)
    seen = _stub_pip(monkeypatch, setup_installer, rc=0)
    monkeypatch.setattr(setup_installer, '_onnxruntime_provided', lambda p: False)
    monkeypatch.setattr(setup_installer, '_verify_capability_import', lambda a, p: True)
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        setup_installer._run_ml_capability('masks')
        onnx = setup_installer._requirement_spec('onnxruntime')
    assert onnx in seen['cmd']


def test_an_onnxruntime_probe_that_times_out_counts_as_present(monkeypatch):
    """A missing module raises instantly; a SLOW import means a real (large)
    runtime is loading. Reading a timeout as "absent" would install the CPU build
    on top of exactly the heavy GPU one we must not touch."""
    from app import setup_installer

    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(setup_installer.os.path, 'isfile', lambda p: True)
    monkeypatch.setattr(setup_installer.subprocess, 'run', boom)
    assert setup_installer._onnxruntime_provided('/some/py') is True


def test_an_unlaunchable_interpreter_does_not_claim_onnxruntime_is_present(monkeypatch):
    from app import setup_installer
    monkeypatch.setattr(setup_installer.os.path, 'isfile', lambda p: True)

    def boom(cmd, **kw):
        raise OSError('no such interpreter')

    monkeypatch.setattr(setup_installer.subprocess, 'run', boom)
    assert setup_installer._onnxruntime_provided('/nope/py') is False
