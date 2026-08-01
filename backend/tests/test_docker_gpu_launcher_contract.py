"""Static safety contract for the novice Windows GPU Docker launcher."""
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "start-docker-gpu.bat"


def _script():
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_is_space_safe_and_has_no_machine_specific_path():
    script = _script()

    assert 'pushd "%~dp0"' in script
    assert "popd" in script
    assert '"%DOCKER_EXE%" compose' in script
    assert '"%DOCKER_DESKTOP%"' in script
    assert not re.search(r"[A-Za-z]:\\Users\\[^%\\]+", script, re.I)


def test_launcher_finds_docker_and_validates_compose():
    script = _script()

    user_install = script.index(
        "%LOCALAPPDATA%\\Programs\\DockerDesktop\\resources\\bin\\docker.exe")
    machine_install = script.index(
        "%ProgramFiles%\\Docker\\Docker\\resources\\bin\\docker.exe")
    path_fallback = script.index("where.exe docker.exe")
    helper_path = script.index(
        'for %%I in ("%DOCKER_EXE%") do set "PATH=%%~dpI;%PATH%"')
    compose_check = script.index('"%DOCKER_EXE%" compose version')

    assert user_install < machine_install < path_fallback
    assert "do if not defined DOCKER_EXE" in script[path_fallback:]
    assert path_fallback < helper_path < compose_check


def test_launcher_starts_docker_desktop_and_waits_a_bounded_time():
    script = _script()

    assert '"%DOCKER_EXE%" info' in script
    assert "%LOCALAPPDATA%\\Programs\\DockerDesktop\\Docker Desktop.exe" in script
    assert "%ProgramFiles%\\Docker\\Docker\\Docker Desktop.exe" in script
    assert 'start "" "%DOCKER_DESKTOP%"' in script
    loops = [int(value) for value in re.findall(
        r"for /l %%I in \(1,1,(\d+)\)", script, re.I)]
    assert loops == [90, 300]
    assert "timeout /t 2" in script
    assert "timeout /t 5" in script
    assert "within twenty-five minutes" in script


def test_launcher_creates_env_once_without_overwriting_it():
    script = _script()
    guard = script.index('if not exist ".env" (')
    copy = script.index('copy ".env.example" ".env"')
    guard_end = script.index("\n)", copy)

    assert guard < copy < guard_end
    assert script.count('copy ".env.example" ".env"') == 1
    assert 'copy /y ".env.example" ".env"' not in script.lower()


def test_launcher_uses_only_isolated_repo_local_persistent_mounts():
    script = _script()

    for folder in ("run", "basedir", "data-docker-gpu", "bank-images"):
        assert f'"{folder}"' in script
    assert 'set "LDS_COMFY_RUN=./run"' in script
    assert 'set "LDS_COMFY_BASEDIR=./basedir"' in script
    assert 'set "LDS_DATA=./data-docker-gpu"' in script
    assert 'set "LDS_BANK_SOURCES=./bank-images"' in script
    uid_override = script.index('set "LDS_UID=0"')
    gid_override = script.index('set "LDS_GID=0"')
    compose_config = script.index('compose -f "%COMPOSE_FILE%" config --quiet')
    compose_up = script.index('compose -f "%COMPOSE_FILE%" up -d --build')
    assert uid_override < compose_config < compose_up
    assert gid_override < compose_config < compose_up
    assert not re.search(r"(?:where|dir|forfiles).*comfyui", script, re.I)


def test_launcher_validates_starts_and_monitors_the_gpu_stack():
    script = _script()

    assert 'compose -f "%COMPOSE_FILE%" config --quiet' in script
    assert 'compose -f "%COMPOSE_FILE%" up -d --build' in script
    assert 'set "CONTAINER_NAME=lora-dataset-studio-gpu"' in script
    assert 'findstr /x /c:"unhealthy"' in script
    assert '/c:"exited"' in script
    assert 'logs --tail 120' in script
    assert "pause" in script


def test_launcher_opens_the_exact_local_url_and_never_cleans_up_data():
    script = _script()

    assert 'set "APP_URL=http://127.0.0.1:5050/"' in script
    assert 'start "" "%APP_URL%"' in script
    lowered = script.lower()
    for destructive in (" compose down", "docker rm", "docker system prune", "rmdir ", "del /q"):
        assert destructive not in lowered
