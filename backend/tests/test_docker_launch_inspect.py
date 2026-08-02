"""Behavioral tests for the Windows Docker launcher inspection helper.

The tests use a fake Docker CLI and never contact the real Docker daemon.
"""

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts" / "docker-launch-inspect.ps1"
POWERSHELL = shutil.which("powershell.exe")


@pytest.fixture
def fake_docker(tmp_path):
    cli = tmp_path / "fake docker.cmd"
    cli.write_text(
        "@echo off\n"
        'if /i "%FAKE_DOCKER_MODE%"=="absent" (\n'
        "  >&2 echo Error: No such object: lora-dataset-studio-gpu\n"
        "  exit /b 1\n"
        ")\n"
        'type "%FAKE_DOCKER_JSON%"\n'
        "exit /b 0\n",
        encoding="ascii",
    )
    return cli


@pytest.fixture
def checkout(tmp_path):
    path = tmp_path / "test neuf é"
    path.mkdir()
    return path


def _container(checkout, *, status="running", app_ports=("5071", "5071"),
               comfy_ports=("8210", "8210"),
               host_ips=("127.0.0.1", "::1")):
    def bindings(values):
        return [
            {"HostIp": host_ips[index % len(host_ips)], "HostPort": value}
            for index, value in enumerate(values)
        ]

    return {
        "Config": {
            "Labels": {
                "com.docker.compose.project": "lora-dataset-studio-gpu",
                "com.docker.compose.service": "studio",
                "com.docker.compose.project.working_dir": str(checkout),
            }
        },
        "State": {"Status": status},
        "NetworkSettings": {
            "Ports": {
                "5050/tcp": bindings(app_ports),
                "8188/tcp": bindings(comfy_ports),
            }
        },
    }


def _run_helper(
    fake_docker,
    checkout,
    tmp_path,
    payload=None,
    mode="json",
    expected_service="studio",
    extra_args=(),
):
    if POWERSHELL is None:
        pytest.skip("Windows PowerShell 5.1 is not available")

    env = os.environ.copy()
    env["FAKE_DOCKER_MODE"] = mode
    if payload is not None:
        json_path = tmp_path / "inspect.json"
        json_path.write_text(
            json.dumps([payload], ensure_ascii=False),
            encoding="utf-8",
        )
        env["FAKE_DOCKER_JSON"] = str(json_path)

    result = subprocess.run(
        [
            POWERSHELL,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(HELPER),
            "-DockerExe",
            str(fake_docker),
            "-ContainerName",
            "lora-dataset-studio-gpu",
            "-ExpectedProject",
            "lora-dataset-studio-gpu",
            "-ExpectedService",
            expected_service,
            "-ExpectedWorkingDir",
            str(checkout),
            *extra_args,
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def test_helper_requires_the_explicit_lds_role_before_sidecar_actions(
    fake_docker, checkout, tmp_path
):
    payload = _container(checkout)
    payload["Config"]["Labels"]["com.docker.compose.service"] = "ollama"
    payload["Config"]["Labels"]["io.lora-dataset-studio.role"] = "ollama"

    result = _run_helper(
        fake_docker,
        checkout,
        tmp_path,
        payload,
        expected_service="ollama",
        extra_args=(
            "-ExpectedRole",
            "ollama",
            "-NoPublishedPorts",
        ),
    )

    assert result["STATE"] == "RUNNING"
    assert result["APP_PORT"] == ""
    assert result["COMFY_PORT"] == ""


@pytest.mark.parametrize("actual_role", [None, "foreign"])
def test_helper_rejects_sidecar_without_matching_ownership_role(
    fake_docker, checkout, tmp_path, actual_role
):
    payload = _container(checkout)
    payload["Config"]["Labels"]["com.docker.compose.service"] = "ollama"
    if actual_role is not None:
        payload["Config"]["Labels"]["io.lora-dataset-studio.role"] = actual_role

    result = _run_helper(
        fake_docker,
        checkout,
        tmp_path,
        payload,
        expected_service="ollama",
        extra_args=(
            "-ExpectedRole",
            "ollama",
            "-NoPublishedPorts",
        ),
    )

    assert result["STATE"] == "COLLISION"


def test_helper_reports_absent_without_real_docker(fake_docker, checkout, tmp_path):
    result = _run_helper(
        fake_docker, checkout, tmp_path, payload=None, mode="absent"
    )

    assert result["STATE"] == "ABSENT"
    assert result["APP_PORT"] == ""
    assert result["COMFY_PORT"] == ""


def test_helper_reuses_deduplicated_ipv4_ipv6_mappings(
        fake_docker, checkout, tmp_path):
    result = _run_helper(
        fake_docker, checkout, tmp_path, _container(checkout)
    )

    assert result == {
        "STATE": "RUNNING",
        "APP_PORT": "5071",
        "COMFY_PORT": "8210",
        "MESSAGE": "",
    }


def test_helper_marks_owned_wildcard_bindings_for_safe_recreate(
        fake_docker, checkout, tmp_path):
    result = _run_helper(
        fake_docker,
        checkout,
        tmp_path,
        _container(checkout, host_ips=("0.0.0.0", "::")),
    )

    assert result == {
        "STATE": "SWITCH",
        "APP_PORT": "5071",
        "COMFY_PORT": "8210",
        "MESSAGE": "",
    }


def test_helper_accepts_wildcard_bindings_when_lan_is_explicit(
        fake_docker, checkout, tmp_path):
    result = _run_helper(
        fake_docker,
        checkout,
        tmp_path,
        _container(checkout, host_ips=("0.0.0.0", "::")),
        extra_args=("-ExpectedHostIp", "0.0.0.0"),
    )

    assert result["STATE"] == "RUNNING"
    assert result["APP_PORT"] == "5071"
    assert result["COMFY_PORT"] == "8210"


def test_helper_accepts_owned_stopped_container_without_ports(
        fake_docker, checkout, tmp_path):
    payload = _container(checkout, status="exited")
    payload["NetworkSettings"]["Ports"] = {}

    result = _run_helper(fake_docker, checkout, tmp_path, payload)

    assert result["STATE"] == "STOPPED"
    assert result["APP_PORT"] == ""
    assert result["COMFY_PORT"] == ""


@pytest.mark.parametrize("collision", ["project", "service", "working_dir"])
def test_helper_rejects_container_owned_elsewhere(
        fake_docker, checkout, tmp_path, collision):
    payload = _container(checkout)
    labels = payload["Config"]["Labels"]
    if collision == "project":
        labels["com.docker.compose.project"] = "another-project"
    elif collision == "service":
        labels["com.docker.compose.service"] = "another-service"
    else:
        other = tmp_path / "another checkout"
        other.mkdir()
        labels["com.docker.compose.project.working_dir"] = str(other)

    result = _run_helper(fake_docker, checkout, tmp_path, payload)

    assert result["STATE"] == "COLLISION"
    assert not result["APP_PORT"]
    assert not result["COMFY_PORT"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["NetworkSettings"]["Ports"].pop("8188/tcp"),
        lambda payload: payload["NetworkSettings"]["Ports"].__setitem__(
            "5050/tcp",
            [
                {"HostIp": "127.0.0.1", "HostPort": "5071"},
                {"HostIp": "::1", "HostPort": "5072"},
            ],
        ),
        lambda payload: payload["NetworkSettings"]["Ports"].__setitem__(
            "5050/tcp", [{"HostIp": "127.0.0.1", "HostPort": "not-a-port"}]
        ),
    ],
    ids=["missing", "ambiguous", "non-numeric"],
)
def test_helper_rejects_invalid_running_mapping(
        fake_docker, checkout, tmp_path, mutate):
    payload = copy.deepcopy(_container(checkout))
    mutate(payload)

    result = _run_helper(fake_docker, checkout, tmp_path, payload)

    assert result["STATE"] == "INVALID"
    assert not result["APP_PORT"]
    assert not result["COMFY_PORT"]
