"""Activation and read-only inspection for the ai-toolkit state bridge.

The preferred API is :func:`subprocess_environment`: it returns an isolated
environment mapping and never mutates ``os.environ``.  ``temporary_activation``
exists for callers that genuinely need process-wide activation and guarantees
rollback even when the body raises.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterator, Mapping, MutableMapping

from app.training_bridge.lds_aitk_bridge_contract import (
    BRIDGE_PROTOCOL,
    ENV_AITK_ROOT,
    ENV_ENABLE,
    ENV_IDENTITY_FILE,
    ENV_KEEP,
    ENV_MAX_BUNDLE_BYTES,
    ENV_MAX_STORE_BYTES,
    ENV_PROTOCOL,
    ENV_RESERVE_BYTES,
    ENV_RESTORE_DIR,
    ENV_STATUS_FILE,
    ENV_STRICT,
    read_json_nofollow,
    probe_aitoolkit_source,
)


BRIDGE_DIR = Path(__file__).resolve().parent.parent / "training_bridge"
SERVICES_DIR = Path(__file__).resolve().parent
_BRIDGE_ENV_KEYS = (
    ENV_ENABLE,
    ENV_PROTOCOL,
    ENV_AITK_ROOT,
    ENV_STATUS_FILE,
    ENV_RESTORE_DIR,
    ENV_KEEP,
    ENV_IDENTITY_FILE,
    ENV_STRICT,
    ENV_MAX_BUNDLE_BYTES,
    ENV_MAX_STORE_BYTES,
    ENV_RESERVE_BYTES,
)
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


def _lexical_bridge_target(
    path: os.PathLike[str] | str,
    *,
    label: str,
    directory: bool = False,
) -> Path:
    """Return an absolute lexical target after rejecting linked components.

    Resolving a status/identity destination follows a pre-planted final symlink
    and exports the victim path to the child, defeating the no-follow writer.
    Keep the intended lexical name and validate each component with ``lstat``;
    the child-side readers/writers repeat the check at use time.
    """
    expanded = os.path.expanduser(os.fspath(path))
    if "\x00" in expanded or (os.name == "nt" and expanded.startswith("\\\\")):
        raise ValueError(f"{label} is unsafe")
    target = Path(os.path.abspath(expanded))
    parts = target.parts
    if not parts:
        raise ValueError(f"{label} is empty")
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"{label} could not be validated") from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or (getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)
        ):
            raise ValueError(f"{label} contains a link or reparse point")
        final = index == len(parts) - 1
        if not final and not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{label} has a non-directory parent")
        if final:
            expected = stat.S_ISDIR if directory else stat.S_ISREG
            if not expected(info.st_mode):
                kind = "directory" if directory else "regular file"
                raise ValueError(f"{label} is not a {kind}")
    if directory:
        try:
            info = os.lstat(target)
        except OSError as exc:
            raise ValueError(f"{label} is not an existing directory") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{label} is not an existing directory")
    return target


def probe(aitoolkit_dir: os.PathLike[str] | str) -> dict[str, Any]:
    """Probe source compatibility without importing torch or ai-toolkit."""
    return probe_aitoolkit_source(aitoolkit_dir)


def _pythonpath_with_bridge(current: str | None) -> str:
    injected = (str(BRIDGE_DIR), str(SERVICES_DIR))
    injected_norm = {
        os.path.normcase(os.path.abspath(path)) for path in injected
    }
    existing = [
        part
        for part in (current or "").split(os.pathsep)
        if part and os.path.normcase(os.path.abspath(part)) not in injected_norm
    ]
    return os.pathsep.join([*injected, *existing])


def environment_overlay(
    *,
    aitoolkit_dir: os.PathLike[str] | str,
    status_file: os.PathLike[str] | str,
    identity_file: os.PathLike[str] | str,
    restore_dir: os.PathLike[str] | str | None = None,
    keep: int = 2,
    max_bundle_bytes: int | None = None,
    max_store_bytes: int | None = None,
    reserve_bytes: int | None = None,
    strict: bool = True,
    current_pythonpath: str | None = None,
) -> dict[str, str]:
    """Return only the variables needed for an opt-in bridge launch."""
    if not 1 <= int(keep) <= 20:
        raise ValueError("keep must be between 1 and 20")
    limits = {
        ENV_MAX_BUNDLE_BYTES: max_bundle_bytes,
        ENV_MAX_STORE_BYTES: max_store_bytes,
        ENV_RESERVE_BYTES: reserve_bytes,
    }
    for name, value in limits.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer byte count")
    root = Path(aitoolkit_dir).expanduser().resolve()
    identity = _lexical_bridge_target(
        identity_file, label="identity_file")
    status = _lexical_bridge_target(
        status_file, label="status_file")
    overlay = {
        "PYTHONPATH": _pythonpath_with_bridge(current_pythonpath),
        ENV_ENABLE: "1",
        ENV_PROTOCOL: str(BRIDGE_PROTOCOL),
        ENV_AITK_ROOT: str(root),
        ENV_STATUS_FILE: str(status),
        ENV_KEEP: str(int(keep)),
        ENV_IDENTITY_FILE: str(identity),
        ENV_STRICT: "1" if strict else "0",
    }
    if restore_dir is not None:
        overlay[ENV_RESTORE_DIR] = str(_lexical_bridge_target(
            restore_dir, label="restore_dir", directory=True))
    for name, value in limits.items():
        if value is not None:
            overlay[name] = str(value)
    return overlay


def subprocess_environment(
    *,
    aitoolkit_dir: os.PathLike[str] | str,
    status_file: os.PathLike[str] | str,
    identity_file: os.PathLike[str] | str,
    restore_dir: os.PathLike[str] | str | None = None,
    keep: int = 2,
    max_bundle_bytes: int | None = None,
    max_store_bytes: int | None = None,
    reserve_bytes: int | None = None,
    strict: bool = True,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a subprocess environment without mutating the caller."""
    result = dict(os.environ if base is None else base)
    result.update(
        environment_overlay(
            aitoolkit_dir=aitoolkit_dir,
            status_file=status_file,
            identity_file=identity_file,
            restore_dir=restore_dir,
            keep=keep,
            max_bundle_bytes=max_bundle_bytes,
            max_store_bytes=max_store_bytes,
            reserve_bytes=reserve_bytes,
            strict=strict,
            current_pythonpath=result.get("PYTHONPATH"),
        )
    )
    return result


def deactivate_environment(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy with bridge variables and its PYTHONPATH entry removed."""
    result = dict(env)
    for key in _BRIDGE_ENV_KEYS:
        result.pop(key, None)
    current = result.get("PYTHONPATH", "")
    injected_norm = {
        os.path.normcase(os.path.abspath(BRIDGE_DIR)),
        os.path.normcase(os.path.abspath(SERVICES_DIR)),
    }
    parts = [
        part
        for part in current.split(os.pathsep)
        if part and os.path.normcase(os.path.abspath(part)) not in injected_norm
    ]
    if parts:
        result["PYTHONPATH"] = os.pathsep.join(parts)
    else:
        result.pop("PYTHONPATH", None)
    return result


@contextlib.contextmanager
def temporary_activation(**kwargs: Any) -> Iterator[MutableMapping[str, str]]:
    """Temporarily mutate ``os.environ`` and restore it byte-for-byte."""
    before = dict(os.environ)
    activated = subprocess_environment(base=before, **kwargs)
    os.environ.clear()
    os.environ.update(activated)
    try:
        yield os.environ
    finally:
        os.environ.clear()
        os.environ.update(before)


def read_status(path: os.PathLike[str] | str) -> dict[str, Any] | None:
    """Read a runtime handshake; incomplete/invalid writes are never trusted."""
    try:
        value = read_json_nofollow(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None
