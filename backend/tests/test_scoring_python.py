"""✨ Score on a GPU Python you already own.

The trap this feature exists to avoid: an interpreter can have a perfect CUDA
torch and STILL be unable to run the pass, because bank_score_infer.py also
imports open_clip and transformers/timm. Accepting it on `torch.cuda.is_available()`
alone would swap an hour of slow-but-working CPU scoring for an import error an
hour in. So the probe reports every dependency, the refusal names the missing
one, and nothing is ever installed into an environment the app did not build.

No real subprocess runs here: `scoring_python._run_probe` is the single seam.
"""
import json
import subprocess
from unittest.mock import patch

import pytest


def _facts(cuda=True, missing=(), device='NVIDIA GeForce RTX 4090'):
    """Raw probe output for an interpreter, minus `missing` modules."""
    from app.services import scoring_python as sp
    mods = {d['module']: d['module'] not in missing for d in sp.SCORING_DEPS}
    return {'python': '3.11.9', 'modules': mods, 'cuda': cuda,
            'device_name': device if cuda else None, 'torch_version': '2.5.1+cu124'}


@pytest.fixture()
def sp(app):
    """The service with a clean probe cache (it is process-global)."""
    from app.services import scoring_python
    scoring_python.clear_cache()
    yield scoring_python
    scoring_python.clear_cache()


# ── The case this whole feature is about ─────────────────────────────────────

def test_cuda_torch_without_open_clip_is_refused_and_names_the_dependency(sp, app, tmp_path):
    """The most likely real machine: a daily-driver training venv with CUDA
    torch, no OpenCLIP. It must be refused, and the message must say WHICH
    package — 'no' with no noun is what makes a user give up."""
    fake = tmp_path / 'aitoolkit-python'
    fake.write_text('')
    with app.app_context(), \
         patch.object(sp, '_run_probe', lambda p: _facts(cuda=True, missing=('open_clip',))):
        verdict = sp.describe(str(fake), sp.probe(str(fake)))
        assert verdict['status'] == 'incomplete'
        assert verdict['usable'] is False
        assert verdict['cuda'] is True, 'CUDA is real here — the refusal is about the deps'
        assert verdict['missing'] == ['open_clip_torch']
        assert 'OpenCLIP' in verdict['detail'] and 'CUDA' in verdict['detail']
        # The exact command to fix it, pip name (open_clip_torch) not module name.
        assert 'pip install open_clip_torch' in verdict['install_command']
        assert str(fake) in verdict['install_command']

        # …and selecting it changes nothing: we stay on the working CPU setup.
        from app import config as cfg
        with pytest.raises(sp.SelectionError) as err:
            sp.select(str(fake))
        assert 'OpenCLIP' in str(err.value)
        assert err.value.verdict['missing'] == ['open_clip_torch']
        assert (cfg.get('bank_scoring.python') or '') == ''


def test_a_complete_cuda_interpreter_is_accepted_and_lights_the_gpu_capability(sp, app, tmp_path):
    from app import capabilities, config as cfg
    good = tmp_path / 'good-python'
    good.write_text('')
    with app.app_context(), patch.object(sp, '_run_probe', lambda p: _facts(cuda=True)):
        verdict = sp.describe(str(good), sp.probe(str(good)))
        assert verdict['status'] == 'gpu_ready'
        assert verdict['usable'] and verdict['gpu']
        assert verdict['missing'] == []
        assert 'RTX 4090' in verdict['detail']

        result = sp.select(str(good))
        assert result['selected'] == str(good)
        assert cfg.get('bank_scoring.python') == str(good)

        # The pass reads bank_scoring_gpu_available(), so the selection has to
        # reach THAT probe — including dropping its 10-minute cache.
        def selected_python(py, expr, timeout=60):
            executable = py[0] if isinstance(py, (tuple, list)) else py
            return executable == str(good)

        with patch.object(capabilities, '_import_ok', selected_python):
            assert capabilities.bank_scoring_gpu_available() is True


def test_a_complete_but_cpu_only_interpreter_is_accepted_and_says_so(sp, app, tmp_path):
    """Selectable — the user may have a reason — but never sold as a speed-up."""
    cpu = tmp_path / 'cpu-python'
    cpu.write_text('')
    with app.app_context(), patch.object(sp, '_run_probe', lambda p: _facts(cuda=False)):
        verdict = sp.describe(str(cpu), sp.probe(str(cpu)))
        assert verdict['status'] == 'cpu_only'
        assert verdict['usable'] is True and verdict['gpu'] is False
        assert 'CPU' in verdict['detail']
        assert sp.select(str(cpu))['selected'] == str(cpu)


# ── Failing safe ─────────────────────────────────────────────────────────────

def test_a_path_that_is_not_an_interpreter_degrades_instead_of_exploding(sp, app):
    from app import config as cfg
    with app.app_context(), patch.object(sp, '_run_probe', lambda p: None):
        # _run_probe returns None for anything that doesn't answer (missing file,
        # broken venv, cold-import timeout) — never a raise.
        verdict = sp.describe('Z:/nope/python.exe', sp.probe('Z:/nope/python.exe'))
        assert verdict['status'] == 'unreachable'
        assert verdict['usable'] is False
        assert verdict['missing'] == [d['pip'] for d in sp.SCORING_DEPS]
        with pytest.raises(sp.SelectionError):
            sp.select('Z:/nope/python.exe')
        assert (cfg.get('bank_scoring.python') or '') == ''


def test_detection_lists_a_broken_candidate_as_a_row_not_an_error(sp, app, client, tmp_path):
    """A candidate that explodes mid-probe must not take the page down."""
    from app import config as cfg
    good = tmp_path / 'good-python'
    good.write_text('')
    with app.app_context():
        cfg.save_config({'bank_scoring': {'python': str(good)}})

    def boom(path):
        raise OSError('the venv is on an unplugged drive')

    with patch.object(sp, '_run_probe', boom):
        res = client.get('/api/scoring-python')
    assert res.status_code == 200
    rows = res.get_json()['interpreters']
    assert rows, 'the configured interpreter is still listed'
    assert all(r['status'] == 'unreachable' for r in rows)


def test_a_detection_crash_is_told_apart_from_an_empty_machine(sp, app, client):
    """The endpoint degrades instead of 500ing — but it used to degrade into
    EXACTLY the payload that means 'nothing to borrow here', so a user could not
    tell a broken search from an honest empty one and had no reason to retry."""
    import sys
    with patch.object(sp, 'detect', side_effect=RuntimeError('detection exploded')):
        res = client.get('/api/scoring-python')
    assert res.status_code == 200
    body = res.get_json()
    assert body['detection_failed'] is True
    assert 'exploded' in body['detection_error']
    assert body['interpreters'] == []
    # the one thing we know no matter what still comes back, so the panel can
    # keep saying what the pass runs in today
    assert body['default_python'] == sys.executable


def test_an_empty_machine_is_not_flagged_as_a_failure(sp, app, client):
    """The other half of the same contract: a genuine 'nothing here' must NOT
    carry the failure flag, or the warning becomes noise everyone learns to
    ignore."""
    with patch.object(sp, 'candidates', lambda profile=None: []):
        res = client.get('/api/scoring-python')
    body = res.get_json()
    assert body.get('detection_failed') in (None, False)
    assert body['interpreters'] == []


def test_a_forced_rescan_also_drops_the_capability_caches(sp, app, client, tmp_path):
    """A user who just pip-installed a package clicks ↻. If only our own cache is
    dropped, the capability probes keep saying 'not installed' for ten more
    minutes and the fix looks broken."""
    from app import capabilities
    dropped = {'n': 0}
    with patch.object(capabilities, 'clear_import_cache',
                      lambda: dropped.__setitem__('n', dropped['n'] + 1)), \
         patch.object(sp, '_run_probe', lambda p: _facts(cuda=True)):
        client.get('/api/scoring-python')
        assert dropped['n'] == 0, 'a plain read must not invalidate anything'
        client.get('/api/scoring-python?force=1')
        assert dropped['n'] == 1


def test_reverting_to_the_app_default_clears_the_override(sp, app, tmp_path):
    from app import config as cfg
    good = tmp_path / 'good-python'
    good.write_text('')
    with app.app_context(), patch.object(sp, '_run_probe', lambda p: _facts(cuda=True)):
        sp.select(str(good))
        assert cfg.get('bank_scoring.python') == str(good)
        assert sp.select('')['reverted'] is True
        assert (cfg.get('bank_scoring.python') or '') == ''


# ── Machines that are NOT this one ───────────────────────────────────────────
# The developer's box is a test case, not the target. These pin the states a
# stranger's install lands in — the majority state (nothing configured) most of
# all: it must be the best-handled one, not the least.

def test_nothing_configured_at_all_is_a_usable_state_not_a_dead_end(sp, app, client):
    """The default install: no ai-toolkit, no ComfyUI, nothing selected. The
    panel must still answer, say what to do, and leave the pass on the CPU."""
    from app import config as cfg
    with app.app_context():
        assert (cfg.get('bank_scoring.python') or '') == ''
        assert (cfg.get('aitoolkit.dir') or '') == ''
        assert (cfg.get('comfyui.base_dir') or '') == ''
    with patch.object(sp, '_run_probe', lambda p: _facts(cuda=False, missing=('torch', 'open_clip'))):
        res = client.get('/api/scoring-python')
    assert res.status_code == 200
    body = res.get_json()
    rows = body['interpreters']
    # Exactly one row — the app's own Python — and it is named, not blamed.
    assert [r['source'] for r in rows] == ['app']
    assert rows[0]['status'] == 'incomplete'
    assert rows[0]['install_command'], 'the way forward is spelled out'
    assert body['selected'] == '', 'nothing was silently selected'
    assert body['default_python']


def test_a_machine_with_no_nvidia_card_is_told_so_and_offered_no_cuda_fix(sp, app, client):
    """No card, an AMD/Intel card, or no driver: three STATES, never errors.
    gpu_vram_gb() answers None for all three and the payload must carry that so
    the UI can drop every word about CUDA."""
    from app import capabilities
    with patch.object(capabilities, 'gpu_vram_gb', lambda: None), \
         patch.object(sp, '_run_probe', lambda p: _facts(cuda=False)):
        assert sp.nvidia_present() is False
        res = client.get('/api/scoring-python')
    assert res.status_code == 200
    assert res.get_json()['nvidia_present'] is False
    # …and the probe still works: a card-less machine can borrow the packages.
    assert all(r['status'] == 'cpu_only' for r in res.get_json()['interpreters'])


def test_a_probe_that_cannot_tell_whether_there_is_a_card_never_raises(sp, app):
    from app import capabilities
    with app.app_context(), patch.object(
            capabilities, 'gpu_vram_gb', side_effect=OSError('nvidia-smi is gone')):
        assert sp.nvidia_present() is False


@pytest.mark.parametrize('name', [
    'Program Files with spaces',       # the classic Windows install location
    'énvironnement-accentué',          # non-ASCII, and it must survive round-trip
    "it's mine",                       # an apostrophe in a folder name
])
def test_an_exotic_path_is_accepted_or_refused_for_an_exact_reason(sp, app, tmp_path, name):
    """Spaces, accents, quotes, another drive: never a crash, always a verdict
    naming the interpreter EXACTLY as the user gave it."""
    env = tmp_path / name
    env.mkdir()
    exe = env / 'python.exe'
    exe.write_text('')
    with app.app_context(), \
         patch.object(sp, '_run_probe', lambda p: _facts(cuda=True, missing=('open_clip',))):
        verdict = sp.describe(str(exe), sp.probe(str(exe)))
        assert verdict['status'] == 'incomplete'
        assert verdict['path'] == str(exe), 'the path is echoed byte-for-byte'
        # A path with a space is quoted in the copyable command, so pasting it works.
        assert str(exe) in verdict['install_command'].replace('"', '')
        if ' ' in str(exe):
            assert f'"{exe}"' in verdict['install_command']


def test_a_path_pasted_with_quotes_or_stray_whitespace_still_resolves(sp, tmp_path):
    """"Copy as path" on Windows wraps the path in quotes; a terminal copy drags
    a trailing space along. Neither is a different interpreter."""
    exe = tmp_path / 'my env' / 'python.exe'
    exe.parent.mkdir()
    exe.write_text('')
    for raw in (f'"{exe}"', f'  {exe}  ', f"'{exe}'", str(exe)):
        assert sp.resolve_entered_path(raw) == [str(exe)]


def test_an_environment_FOLDER_is_accepted_as_readily_as_an_interpreter(sp, tmp_path):
    """People have "my conda env", not "my conda env's python.exe". Both must
    lead to the same place, and the layout is TRIED, never assumed: a conda env
    keeps python.exe at its root, a venv hides it under Scripts/ or bin/."""
    venv = tmp_path / 'venv'
    (venv / 'Scripts').mkdir(parents=True)
    (venv / 'Scripts' / 'python.exe').write_text('')
    assert sp.resolve_entered_path(str(venv)) == [str(venv / 'Scripts' / 'python.exe')]

    posix = tmp_path / 'posixenv'
    (posix / 'bin').mkdir(parents=True)
    (posix / 'bin' / 'python').write_text('')
    assert sp.resolve_entered_path(str(posix)) == [str(posix / 'bin' / 'python')]

    conda = tmp_path / 'envs' / 'ml'
    conda.mkdir(parents=True)
    (conda / 'python.exe').write_text('')
    assert sp.resolve_entered_path(str(conda)) == [str(conda / 'python.exe')]

    # A folder holding no interpreter at all: no candidates, no crash.
    assert sp.resolve_entered_path(str(tmp_path / 'empty')) == [str(tmp_path / 'empty')]
    (tmp_path / 'empty').mkdir()
    assert sp.resolve_entered_path(str(tmp_path / 'empty')) == []
    assert sp.resolve_entered_path('') == []
    assert sp.resolve_entered_path(None) == []


def test_a_typed_path_always_produces_an_answer_even_when_already_listed(sp, app, tmp_path):
    """Found live: entering a path that resolves onto an interpreter the list
    ALREADY holds produced no new row — the screen looked unchanged and the
    button looked broken. The row is now marked, so the UI can say which one it
    is. Silence is the worst reply for the route most installs depend on."""
    import sys
    with app.app_context(), patch.object(sp, '_run_probe', lambda p: _facts(cuda=True)):
        # The app's own interpreter is always listed; typing it must be ACKNOWLEDGED.
        res = sp.detect(extra_path=sys.executable)
        assert res['entered_status'] == 'resolved'
        hit = [r for r in res['interpreters'] if r.get('entered')]
        assert len(hit) == 1 and hit[0]['source'] == 'app', \
            'the typed path is marked on the row it landed on, not duplicated'
        assert len([r for r in res['interpreters'] if r['source'] == 'manual']) == 0

        # A folder that exists but holds no interpreter: named as such, not silence.
        empty = tmp_path / 'not-an-env'
        empty.mkdir()
        res = sp.detect(extra_path=str(empty))
        assert res['entered_status'] == 'no_interpreter'
        assert not any(r.get('entered') for r in res['interpreters'])

        # Nothing typed at all stays quiet.
        assert sp.detect()['entered_status'] == ''
        assert sp.detect(extra_path='   ')['entered_status'] == ''


def test_an_interpreter_is_never_rejected_on_its_FILENAME(sp, tmp_path):
    """python3.11, a shim, a wrapper script — the name proves nothing. Only the
    probe's answer decides."""
    for name in ('python3.11', 'python', 'py', 'python3.11.exe', 'ml-python'):
        exe = tmp_path / name
        exe.write_text('')
        assert sp.resolve_entered_path(str(exe)) == [str(exe)]


def test_no_torch_or_cuda_VERSION_is_ever_required(sp, app, tmp_path):
    """Demanding a torch/CUDA version would rule out perfectly working rigs — an
    older card on cu118, a 50-series that only works on cu128, a nightly. The
    contract is exactly what the script needs: the modules import and
    torch.cuda.is_available() is true."""
    exe = tmp_path / 'python'
    exe.write_text('')
    for version, cuda_build in (('1.13.1+cu117', '11.7'), ('2.9.1+cu128', '12.8'),
                                ('2.10.0.dev20260101+cu130', '13.0'), (None, None)):
        facts = _facts(cuda=True)
        facts['torch_version'] = version
        with app.app_context(), patch.object(sp, '_run_probe', lambda p, f=facts: f):
            verdict = sp.describe(str(exe), sp.probe(str(exe), force=True))
        assert verdict['status'] == 'gpu_ready', f'{version} must be usable'
    # And the probe program asks nothing about versions beyond reporting them.
    assert '__version__' in sp._PROBE_CODE
    assert 'version_info' in sp._PROBE_CODE
    for forbidden in ('>=', 'parse_version', 'LooseVersion', 'packaging'):
        assert forbidden not in sp._PROBE_CODE, f'no version gate: {forbidden}'


def test_an_install_that_works_today_keeps_working_untouched(sp, app):
    """The update must change nothing for someone who never opens the picker:
    the pass reads bank_scoring.python, it is still empty, and the fallback is
    still the app's own interpreter. Detection is an OFFER, never a prerequisite."""
    import sys
    from app import capabilities, config as cfg
    with app.app_context():
        assert (cfg.get('bank_scoring.python') or '') == ''
        seen = {}
        # the CUDA question goes through the three-valued probe (unknown has to
        # stay distinguishable from 'no CUDA'), the readiness one through the
        # boolean gate — both must resolve the SAME interpreter.
        with patch.object(capabilities, '_cached_import',
                          lambda key, python, expr: seen.setdefault(key, python) and False), \
                patch.object(capabilities, '_cached_import_state',
                             lambda key, python, expr: seen.setdefault(key, python) and False):
            capabilities.probe_bank_scoring()
            capabilities.bank_scoring_gpu_available()
        assert seen['bank_scoring'] == sys.executable
        assert seen['bank_scoring_gpu'] == sys.executable


# ── Candidates & caching ─────────────────────────────────────────────────────

def test_candidates_are_known_interpreters_only_deduplicated_and_existing(sp, app, tmp_path):
    """Known Pythons, not a disk sweep — and never the same one twice."""
    from app import config as cfg
    shared = tmp_path / 'shared-python'
    shared.write_text('')
    with app.app_context():
        cfg.save_config({'bank_scoring': {'python': str(shared)},
                         'aitoolkit': {'dir': str(tmp_path), 'python': str(shared)}})
        rows = sp.candidates()
        paths = [c['path'] for c in rows]
        assert paths.count(str(shared)) == 1, 'one interpreter, one row'
        # The selected interpreter IS the ai-toolkit one here: it must keep the
        # label that says so. "Currently used" is carried by `selected`, and
        # letting it win would hide where the interpreter actually comes from.
        row = next(c for c in rows if c['path'] == str(shared))
        assert row['source'] == 'aitoolkit' and 'ai-toolkit' in row['label']
        sources = {c['source'] for c in rows}
        assert 'app' in sources, "the app's own Python is the way back"
        assert str(tmp_path / 'ghost.exe') not in paths


def test_a_configured_interpreter_the_app_does_not_recognise_is_still_listed(sp, app, tmp_path):
    from app import config as cfg
    stranger = tmp_path / 'conda' / 'python.exe'
    stranger.parent.mkdir()
    stranger.write_text('')
    with app.app_context():
        cfg.save_config({'bank_scoring': {'python': str(stranger)}})
        rows = sp.candidates()
        row = next(c for c in rows if c['path'] == str(stranger))
        assert row['source'] == 'configured'


def test_a_typed_path_is_probed_even_when_it_is_not_a_known_candidate(sp, app, tmp_path):
    typed = tmp_path / 'conda' / 'python.exe'
    typed.parent.mkdir()
    typed.write_text('')
    with app.app_context(), patch.object(sp, '_run_probe', lambda p: _facts(cuda=True)):
        rows = sp.detect(extra_path=str(typed))['interpreters']
        manual = [r for r in rows if r['source'] == 'manual']
        assert len(manual) == 1 and manual[0]['status'] == 'gpu_ready'


def test_a_rescan_sees_a_dependency_installed_since_the_last_probe(sp, app, tmp_path):
    """Without this the user installs open_clip and the app keeps saying it is
    missing for ten minutes — the exact way a good feature loses trust."""
    py = tmp_path / 'python'
    py.write_text('')
    calls = {'n': 0}

    def evolving(path):
        calls['n'] += 1
        return _facts(cuda=True, missing=('open_clip',) if calls['n'] == 1 else ())

    with app.app_context(), patch.object(sp, '_run_probe', evolving):
        assert sp.describe(str(py), sp.probe(str(py)))['status'] == 'incomplete'
        assert sp.probe(str(py))['modules']['open_clip'] is False   # cached, no new call
        assert calls['n'] == 1
        sp.clear_cache()
        assert sp.describe(str(py), sp.probe(str(py)))['status'] == 'gpu_ready'


def test_an_unreachable_probe_is_never_cached_as_a_fact(sp, app, tmp_path):
    """A cold-import timeout must not freeze a working venv into 'unreachable'."""
    py = tmp_path / 'python'
    py.write_text('')
    calls = {'n': 0}

    def flaky(path):
        calls['n'] += 1
        return None if calls['n'] == 1 else _facts(cuda=True)

    with app.app_context(), patch.object(sp, '_run_probe', flaky):
        assert sp.probe(str(py)) is None
        assert sp.describe(str(py), sp.probe(str(py)))['status'] == 'gpu_ready'


# ── The probe program itself ─────────────────────────────────────────────────

def test_the_probe_program_reports_every_scoring_dependency(sp):
    """Guards the pairing: the code that runs in the child must ask about the
    same module list the verdict renders."""
    for dep in sp.SCORING_DEPS:
        assert f"'{dep['module']}'" in sp._PROBE_CODE or f'"{dep["module"]}"' in sp._PROBE_CODE
    # open_clip is the one that matters and the one a CUDA-only check misses.
    assert 'open_clip' in sp._PROBE_CODE
    assert 'cuda.is_available' in sp._PROBE_CODE
    assert 'find_spec' not in sp._PROBE_CODE, \
        'a discoverable package is not necessarily importable'


def test_the_probe_program_really_runs_and_emits_the_expected_shape(sp):
    """The generated program is EXECUTED here — a syntax error in it would
    otherwise read as 'every Python on your machine is unreachable' — but in
    THIS process, with every real dependency import forced to fail, so it takes
    its no-torch branch.

    No subprocess and no `import torch` anywhere in the suite: importing torch
    for real would load the CUDA runtime (hundreds of MB per process, and this
    machine is somebody's desktop)."""
    import contextlib
    import importlib
    import io as _io

    buf = _io.StringIO()
    attempted = []

    def missing(name):
        attempted.append(name)
        raise ImportError(name)

    with patch.object(importlib, 'import_module', missing), \
         contextlib.redirect_stdout(buf):
        exec(compile(sp._PROBE_CODE, '<probe>', 'exec'), {'__name__': '__probe__'})
    info = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert set(info['modules']) == {d['module'] for d in sp.SCORING_DEPS}
    assert info['modules'] == {d['module']: False for d in sp.SCORING_DEPS}
    assert attempted == [d['module'] for d in sp.SCORING_DEPS]
    assert info['cuda'] is False and info['torch_version'] is None
    assert info['python'].count('.') == 2


def test_probe_disables_user_site_for_a_borrowed_python(sp):
    """The picker must test the same isolated environment Score will run in.
    Otherwise a package in the Windows user-site can make find/import results
    depend on who launched LDS instead of what ComfyUI actually contains."""
    from types import SimpleNamespace
    facts = _facts(cuda=True)
    proc = SimpleNamespace(returncode=0, stdout=json.dumps(facts), stderr='')
    with patch.object(sp.subprocess, 'run', return_value=proc) as run:
        assert sp._run_probe(r'C:\ComfyUI\python_embeded\python.exe') == facts
    command = run.call_args.args[0]
    assert command[:3] == [r'C:\ComfyUI\python_embeded\python.exe', '-s', '-c']


def test_the_probe_never_raises_when_the_interpreter_cannot_even_start(sp):
    """OSError (path is not executable, missing DLL, permission) and a timeout
    are both UNKNOWN, not a crash — and neither reaches the request."""
    for boom in (OSError('not executable'),
                 subprocess.TimeoutExpired(cmd='python', timeout=1),
                 ValueError('embedded null byte')):
        with patch.object(sp.subprocess, 'run', side_effect=boom):
            assert sp._run_probe('whatever') is None


# ── Route contract ───────────────────────────────────────────────────────────

def test_the_endpoint_refuses_an_incomplete_interpreter_with_the_reason(sp, app, client, tmp_path):
    py = tmp_path / 'python'
    py.write_text('')
    with patch.object(sp, '_run_probe', lambda p: _facts(cuda=True, missing=('open_clip', 'timm'))):
        res = client.post('/api/scoring-python', json={'python': str(py)})
    assert res.status_code == 400
    body = res.get_json()
    assert 'OpenCLIP' in body['error'] and 'timm' in body['error']
    assert body['verdict']['missing'] == ['open_clip_torch', 'timm']
    with app.app_context():
        from app import config as cfg
        assert (cfg.get('bank_scoring.python') or '') == ''
