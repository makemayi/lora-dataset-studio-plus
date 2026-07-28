"""One file answers "what do I need to run the tests", and CI installs THAT file.

The bug this pins is not a crash, it is a divergence. pytest-flask was installed on
a dev machine and absent from CI, so nine tests that forgot `with app.app_context():`
passed locally and failed on the release tag — the suite was green against an
environment nobody had written down. backend/pytest.ini neutralises that particular
plugin; this keeps the environment itself from drifting again, by refusing to let
the workflows install anything other than requirements-dev.txt.

Pure text checks on files in the repo — no network, no pip.
"""
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DEV = _ROOT / 'backend' / 'requirements-dev.txt'
_RUNTIME = _ROOT / 'backend' / 'requirements.txt'
_CI = _ROOT / '.github' / 'workflows' / 'ci.yml'
_RELEASE = _ROOT / '.github' / 'workflows' / 'release.yml'


def _lines(path):
    return [l.strip() for l in path.read_text(encoding='utf-8').splitlines()
            if l.strip() and not l.strip().startswith('#')]


def test_dev_requirements_exists_and_pulls_the_runtime_file():
    assert _DEV.is_file(), 'backend/requirements-dev.txt is the one test environment'
    assert '-r requirements.txt' in _lines(_DEV)


def test_pytest_is_pinned_exactly_and_only_in_the_dev_file():
    """`pytest>=8.0` floating in the RUNTIME file was both a dependency end users
    never run and a version CI and a contributor could disagree on."""
    dev = _lines(_DEV)
    assert any(re.fullmatch(r'pytest==\d+(\.\d+)*', l) for l in dev), \
        'pin pytest exactly — a suite green on another collector is not evidence about CI'
    assert not any(l.lower().startswith('pytest') for l in _lines(_RUNTIME)), \
        'test dependencies belong in requirements-dev.txt, not in the end-user install'


def test_the_test_only_ml_extras_are_declared_not_inlined_in_ci():
    """numpy/OpenCV make test_watermarks.py exercise real pixel maths. They were
    two literal pip arguments inside each workflow — i.e. a second, invisible
    requirements file that only CI had."""
    dev = ' '.join(_lines(_DEV))
    assert 'numpy>=1.26,<2' in dev
    assert 'opencv-python-headless' in dev


@pytest.mark.parametrize('workflow', [_CI, _RELEASE], ids=['ci', 'release'])
def test_workflows_install_the_dev_requirements_and_nothing_else(workflow):
    text = workflow.read_text(encoding='utf-8')
    assert 'requirements-dev.txt' in text, f'{workflow.name} must install the declared test env'
    # No loose `pip install "<package>"` for test extras alongside it: that is
    # exactly the shape the divergence took.
    stray = re.findall(r'pip install[^\n]*(?:numpy|opencv|pytest-)[^\n]*', text)
    assert not stray, f'{workflow.name} installs test deps inline: {stray}'


def test_pytest_ini_still_blocks_the_plugin_that_caused_the_red_release():
    ini = (_ROOT / 'backend' / 'pytest.ini').read_text(encoding='utf-8')
    assert '-p no:flask' in ini, \
        'without this, a machine carrying pytest-flask runs a different suite than CI'
