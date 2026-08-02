"""Behavioral tests for the external-ComfyUI override generator."""

import base64
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts" / "configure-external-comfy.ps1"
POWERSHELL = shutil.which("powershell.exe")


def run_helper(override: Path, *extra: str, env=None):
    if POWERSHELL is None:
        pytest.skip("Windows PowerShell 5.1 is unavailable")
    result = subprocess.run(
        [
            POWERSHELL,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(HELPER),
            "-OverridePath",
            str(override),
            *extra,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
        env=env or os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip(), result


def make_comfy(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "main.py").write_text("# test\n", encoding="utf-8")
    (root / "models").mkdir()
    return root.resolve()


@pytest.mark.parametrize("choose_parent", [False, True])
def test_accepts_exact_root_or_portable_parent_and_generates_rw_bind(
    tmp_path, choose_parent
):
    portable = tmp_path / "Portable Comfy UI é ! % & '$ ${X}"
    comfy = make_comfy(portable / "ComfyUI")
    selected = portable if choose_parent else comfy
    override = tmp_path / "generated override.yml"

    stdout, result = run_helper(
        override, "-Configure", "-CandidatePath", str(selected)
    )

    assert stdout == "STATE=SAVED"
    assert str(comfy) not in result.stdout
    raw = override.read_text(encoding="utf-8")
    encoded = base64.b64encode(str(comfy).encode("utf-8")).decode("ascii")
    assert raw.splitlines()[:2] == [
        "# lds-external-comfy-schema: 1",
        f"# lds-external-comfy-path-base64: {encoded}",
    ]
    assert "target: /external-comfyui" in raw
    assert "read_only" not in raw
    assert "$$" in raw
    assert "''" in raw

    stdout, result = run_helper(override)
    assert stdout == "STATE=VALID"
    assert str(comfy) not in result.stdout


def test_rejects_incomplete_or_relative_candidate_without_writing(tmp_path):
    incomplete = tmp_path / "not comfy"
    incomplete.mkdir()
    override = tmp_path / "override.yml"

    stdout, _ = run_helper(
        override, "-Configure", "-CandidatePath", str(incomplete)
    )

    assert stdout == "STATE=INVALID"
    assert not override.exists()


def test_detects_tampering_and_never_discloses_saved_path(tmp_path):
    secret_named_comfy = make_comfy(tmp_path / "private-secret-folder" / "ComfyUI")
    override = tmp_path / "override.yml"
    run_helper(
        override, "-Configure", "-CandidatePath", str(secret_named_comfy)
    )
    override.write_text(
        override.read_text(encoding="utf-8") + "# modified\n", encoding="utf-8"
    )

    stdout, result = run_helper(override)

    assert stdout == "STATE=INVALID"
    assert "private-secret-folder" not in result.stdout
    assert "private-secret-folder" not in result.stderr


def test_cancelled_picker_does_not_touch_existing_override(tmp_path):
    override = tmp_path / "override.yml"
    override.write_text("keep me\n", encoding="utf-8")
    env = os.environ.copy()
    env["LDS_TEST_CANCEL_FOLDER_PICKER"] = "1"

    stdout, _ = run_helper(override, "-Configure", env=env)

    assert stdout == "STATE=CANCELLED"
    assert override.read_text(encoding="utf-8") == "keep me\n"
