"""Behavioral tests for the secret-safe Ollama deployment-mode helper."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / 'scripts' / 'docker-ollama-mode.ps1'
POWERSHELL = shutil.which('powershell.exe')


def run_helper(config, *extra):
    if POWERSHELL is None:
        pytest.skip('Windows PowerShell 5.1 is unavailable')
    result = subprocess.run(
        [
            POWERSHELL,
            '-NoLogo',
            '-NoProfile',
            '-NonInteractive',
            '-ExecutionPolicy',
            'Bypass',
            '-File',
            str(HELPER),
            '-Action',
            'Read',
            '-ConfigPath',
            str(config),
            *extra,
        ],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=15,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    return dict(
        line.split('=', 1)
        for line in result.stdout.splitlines()
        if '=' in line
    ), result


@pytest.mark.parametrize(
    ('payload', 'mode'),
    [
        ({}, 'unset'),
        ({'ollama': {}}, 'unset'),
        ({'ollama': {'deployment_mode': None}}, 'unset'),
        ({'ollama': {'deployment_mode': 'unconfigured'}}, 'unset'),
        ({'ollama': {'deployment_mode': 'none'}}, 'none'),
        ({'ollama': {'deployment_mode': 'host'}}, 'host'),
        ({'ollama': {'deployment_mode': 'docker'}}, 'docker'),
    ],
)
def test_reads_only_the_allowed_persistent_mode(tmp_path, payload, mode):
    config = tmp_path / 'config.json'
    payload.setdefault('secret_token', 'must-never-be-printed')
    config.write_text(json.dumps(payload), encoding='utf-8')

    values, result = run_helper(config)

    assert values == {'STATE': 'VALID', 'MODE': mode}
    assert 'must-never-be-printed' not in result.stdout
    assert 'must-never-be-printed' not in result.stderr


def test_missing_config_is_unconfigured_and_not_created(tmp_path):
    config = tmp_path / 'missing.json'

    values, _ = run_helper(config)

    assert values == {'STATE': 'VALID', 'MODE': 'unset'}
    assert not config.exists()


@pytest.mark.parametrize(
    'contents',
    [
        '{bad json',
        '[]',
        json.dumps({'ollama': 'not-an-object'}),
        json.dumps({'ollama': {'deployment_mode': 'surprise'}}),
        json.dumps({'ollama': {'deployment_mode': 7}}),
    ],
)
def test_invalid_data_fails_closed_without_echoing_it(tmp_path, contents):
    config = tmp_path / 'config.json'
    config.write_text(contents, encoding='utf-8')

    values, result = run_helper(config)

    assert values == {'STATE': 'INVALID', 'MODE': 'unset'}
    assert contents not in result.stdout


def test_wait_is_bounded(tmp_path):
    if POWERSHELL is None:
        pytest.skip('Windows PowerShell 5.1 is unavailable')
    config = tmp_path / 'missing.json'

    result = subprocess.run(
        [
            POWERSHELL,
            '-NoLogo',
            '-NoProfile',
            '-NonInteractive',
            '-ExecutionPolicy',
            'Bypass',
            '-File',
            str(HELPER),
            '-Action',
            'Wait',
            '-ConfigPath',
            str(config),
            '-TimeoutSeconds',
            '1',
            '-PollSeconds',
            '1',
        ],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert 'STATE=TIMEOUT' in result.stdout
    assert 'MODE=unset' in result.stdout
