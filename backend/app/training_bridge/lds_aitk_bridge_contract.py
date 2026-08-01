"""Pure-stdlib contract shared by LDS and the injected ai-toolkit runtime.

This module must stay importable without torch.  The Flask process uses it to
probe an ai-toolkit checkout and to build a subprocess environment; the
ai-toolkit interpreter imports the same file as a top-level module through the
temporary ``PYTHONPATH`` overlay.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping


BRIDGE_NAME = "lds-aitoolkit-state-bridge"
BRIDGE_VERSION = "1.1.0"
BRIDGE_PROTOCOL = 2
STATE_SCHEMA = "lds.training-state/v1"
IDENTITY_SCHEMA = "lds.training-state-context/v1"
SHAPE_REVISION = "aitk-base-sd-train/v2"

STATE_DIRNAME = ".lds-state"
STATUS_FILENAME = "bridge-status.json"
LATEST_FILENAME = "latest.json"
COMPLETE_FILENAME = "COMPLETE"
MANIFEST_FILENAME = "manifest.json"
ARTIFACTS_DIRNAME = "artifacts"

ENV_ENABLE = "LDS_AITK_BRIDGE_ENABLE"
ENV_PROTOCOL = "LDS_AITK_BRIDGE_PROTOCOL"
ENV_AITK_ROOT = "LDS_AITK_ROOT"
ENV_STATUS_FILE = "LDS_AITK_BRIDGE_STATUS_FILE"
ENV_RESTORE_DIR = "LDS_AITK_STATE_RESTORE_DIR"
ENV_KEEP = "LDS_AITK_STATE_KEEP"
ENV_IDENTITY_FILE = "LDS_AITK_IDENTITY_FILE"
ENV_STRICT = "LDS_AITK_BRIDGE_STRICT"
ENV_MAX_BUNDLE_BYTES = "LDS_AITK_STATE_MAX_BUNDLE_BYTES"
ENV_MAX_STORE_BYTES = "LDS_AITK_STATE_MAX_STORE_BYTES"
ENV_RESERVE_BYTES = "LDS_AITK_STATE_RESERVE_BYTES"

REQUIRED_IDENTITY_FIELDS = (
    "config_hash",
    "dataset_hash",
    "base_hash",
    "network_hash",
    "toolkit_revision",
    "runtime",
)

ARTIFACT_FILENAMES = {
    "raw_weights": "raw_weights.pt",
    "optimizer": "optimizer.pt",
    "scheduler": "scheduler.pt",
    "scaler": "scaler.pt",
    "ema": "ema.pt",
    "rng_json": "rng.json",
    "rng_torch": "rng.pt",
    "dataloader": "dataloader.json",
    "latent_cache": "latent_cache.tar",
    "trainer": "trainer.json",
    "public_checkpoint": "checkpoint.safetensors",
}

_BASE_REL = Path("jobs") / "process" / "BaseSDTrainProcess.py"
_SD_REL = Path("extensions_built_in") / "sd_trainer" / "SDTrainer.py"
_DATA_REL = Path("toolkit") / "data_loader.py"
_MIXIN_REL = Path("toolkit") / "dataloader_mixins.py"
_VERSION_REL = Path("version.py")
_RUN_REL = Path("run.py")
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)

_BASE_MARKERS = (
    "def save(self, step=None):",
    "self.ema.eval()",
    "self.ema.train()",
    "def hook_before_train_loop(self):",
    "self.prepare_accelerator()",
    "self.lr_scheduler.step(self.step_num)",
    "dataloader_iterator = iter(dataloader)",
    "self.save(self.step_num)",
    "self.step_num = step + 1",
)
_SD_MARKERS = (
    "def hook_before_train_loop(self):",
    "super().hook_before_train_loop()",
    "def hook_train_loop(",
    "self.optimizer.step()",
    "self.ema.update()",
    "self.lr_scheduler.step()",
)
_DATA_MARKERS = (
    "class AiToolkitDataset(",
    "def setup_epoch(self):",
    "self.setup_buckets()",
    "self.cache_latents_all_latents()",
    "self.epoch_num += 1",
)
_MIXIN_MARKERS = (
    "class BucketsMixin:",
    "def setup_buckets(self",
    "self.buckets = {}",
    "self.shuffle_buckets()",
    "self.build_batch_indices()",
    "def cache_latents_all_latents(self",
    "file_item.get_latent_path(recalculate=True)",
    "if os.path.exists(latent_path):",
    "def cache_text_embeddings(self",
    "file_item.get_text_embedding_path(recalculate=True)",
    "if not os.path.exists(text_embedding_path):",
)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return bool(
        stat.S_ISLNK(info.st_mode)
        or (getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)
    )


def _absolute_local_path(path: os.PathLike[str] | str) -> Path:
    value = Path(os.path.abspath(os.fspath(path)))
    text = os.fspath(value)
    if "\x00" in text or (os.name == "nt" and text.startswith("\\\\")):
        raise OSError("unsafe local path")
    return value


def _assert_no_link_components(path: Path) -> None:
    absolute = _absolute_local_path(path)
    parts = absolute.parts
    if not parts:
        raise OSError("unsafe empty path")
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        if (
            stat.S_ISLNK(info.st_mode)
            or (getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)
        ):
            raise OSError("path contains a link or reparse point")


def _ensure_safe_parent(parent: Path) -> None:
    parent = _absolute_local_path(parent)
    missing = []
    current = parent
    while True:
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if current.parent == current:
                raise OSError("no existing path anchor")
            missing.append(current)
            current = current.parent
            continue
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or (getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)
        ):
            raise OSError("status parent is not a real directory")
        break
    _assert_no_link_components(current)
    for directory in reversed(missing):
        os.mkdir(directory)
        info = os.lstat(directory)
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or (getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)
        ):
            raise OSError("created status parent is unsafe")
    _assert_no_link_components(parent)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_json_nofollow(
    path: os.PathLike[str] | str,
    value: Mapping[str, Any],
) -> None:
    """Atomically replace one local JSON file without following links/junctions."""
    target = _absolute_local_path(path)
    _ensure_safe_parent(target.parent)
    try:
        target_info = os.lstat(target)
    except FileNotFoundError:
        target_info = None
    if target_info is not None and (
        not stat.S_ISREG(target_info.st_mode)
        or stat.S_ISLNK(target_info.st_mode)
        or (getattr(target_info, "st_file_attributes", 0) & _REPARSE_POINT)
    ):
        raise OSError("status target is unsafe")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _assert_no_link_components(target.parent)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json_nofollow(
    path: os.PathLike[str] | str,
    *,
    max_bytes: int = 1 << 20,
) -> Any:
    """Read a bounded regular JSON file while refusing links and reparses."""
    target = _absolute_local_path(path)
    _assert_no_link_components(target.parent)
    if _is_link_or_reparse(target):
        raise OSError("JSON target is a link or reparse point")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags)
    try:
        info = os.fstat(descriptor)
        current = os.lstat(target)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size > max_bytes
            or not stat.S_ISREG(current.st_mode)
            or _is_link_or_reparse(target)
            or (current.st_dev, current.st_ino, current.st_size)
            != (info.st_dev, info.st_ino, info.st_size)
        ):
            raise OSError("JSON target is not a bounded regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            raw = stream.read(max_bytes + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > max_bytes:
        raise OSError("JSON target is too large")
    return json.loads(raw.decode("utf-8"))


def _path_is_within(root: Path, candidate: Path) -> bool:
    try:
        common = os.path.commonpath((os.fspath(root), os.fspath(candidate)))
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(os.fspath(root))


def _file_stability_token(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000))),
        int(getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000))),
    )


def _hash_model_file(
    path: Path,
    *,
    trusted_link_root: Path | None = None,
) -> tuple[str, int]:
    """Hash one stable regular file, allowing only HF snapshot->blob links."""
    lexical = _absolute_local_path(path)
    link_token = None
    if _is_link_or_reparse(lexical):
        if trusted_link_root is None:
            raise OSError("model artifact is a link or reparse point")
        _assert_no_link_components(lexical.parent)
        link_before = os.lstat(lexical)
        if not stat.S_ISLNK(link_before.st_mode):
            raise OSError("model artifact reparse point is unsupported")
        link_token = _file_stability_token(link_before)
        opened_path = Path(os.path.realpath(lexical))
        trusted_link_root = _absolute_local_path(trusted_link_root)
        if not _path_is_within(trusted_link_root, opened_path):
            raise OSError("pinned model link escapes its immutable blob store")
        _assert_no_link_components(opened_path)
    else:
        _assert_no_link_components(lexical)
        opened_path = lexical

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(opened_path, flags)
    try:
        before = os.fstat(descriptor)
        current = os.lstat(opened_path)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or _is_link_or_reparse(opened_path)
            or (int(before.st_dev), int(before.st_ino))
            != (int(current.st_dev), int(current.st_ino))
        ):
            raise OSError("model artifact is not a stable regular file")
        before_token = _file_stability_token(before)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 8 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after_token = _file_stability_token(os.fstat(descriptor))
        current_after = os.lstat(opened_path)
        if (
            before_token != after_token
            or (int(before.st_dev), int(before.st_ino))
            != (int(current_after.st_dev), int(current_after.st_ino))
        ):
            raise OSError("model artifact changed while being verified")
    finally:
        os.close(descriptor)
    if link_token is not None:
        link_after = os.lstat(lexical)
        if (
            not stat.S_ISLNK(link_after.st_mode)
            or _file_stability_token(link_after) != link_token
            or Path(os.path.realpath(lexical)) != opened_path
        ):
            raise OSError("pinned model link changed while being verified")
    return digest.hexdigest(), int(before.st_size)


def _pinned_hf_roots(
    path: Path,
    repo: str,
    commit: str,
) -> tuple[Path, Path]:
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*",
        repo,
    ):
        raise ValueError("model artifact repository is invalid")
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
        raise ValueError("model artifact commit is invalid")
    parts = path.parts
    snapshot_index = None
    for index, part in enumerate(parts[:-1]):
        if (
            part == "snapshots"
            and index + 1 < len(parts)
            and parts[index + 1].lower() == commit.lower()
        ):
            snapshot_index = index
    if snapshot_index is None:
        raise OSError("pinned model path is outside its commit snapshot")
    snapshot_root = Path(*parts[: snapshot_index + 2])
    model_root = snapshot_root.parent.parent
    if model_root.name != "models--" + repo.replace("/", "--"):
        raise OSError("pinned model path does not match its repository")
    if not _path_is_within(snapshot_root, path):
        raise OSError("pinned model path escapes its commit snapshot")
    _assert_no_link_components(snapshot_root)
    blobs = model_root / "blobs"
    _assert_no_link_components(blobs)
    info = os.lstat(blobs)
    if not stat.S_ISDIR(info.st_mode):
        raise OSError("pinned model blob store is unavailable")
    return snapshot_root, blobs


def _hash_model_directory(
    root: Path,
    *,
    trusted_link_root: Path | None = None,
) -> tuple[str, int, int]:
    _assert_no_link_components(root)
    root_info = os.lstat(root)
    if not stat.S_ISDIR(root_info.st_mode) or _is_link_or_reparse(root):
        raise OSError("model artifact directory is unsafe")
    entries: list[dict[str, Any]] = []
    total = 0

    def walk_error(exc: OSError) -> None:
        raise exc

    for current, directories, filenames in os.walk(
        root, followlinks=False, onerror=walk_error
    ):
        current_path = Path(current)
        _assert_no_link_components(current_path)
        directories.sort()
        for directory in directories:
            child = current_path / directory
            info = os.lstat(child)
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or (getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)
            ):
                raise OSError("model artifact directory contains an unsafe link")
        for filename in sorted(filenames):
            child = current_path / filename
            digest, size = _hash_model_file(
                child, trusted_link_root=trusted_link_root)
            entries.append({
                "path": child.relative_to(root).as_posix(),
                "size": size,
                "sha256": digest,
            })
            total += size
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), total, len(entries)


def verify_identity_model_artifacts(identity: Mapping[str, Any]) -> None:
    """Re-hash parent-pinned model bytes in the child before model imports."""
    records = identity.get("model_artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("identity has no child-verifiable model artifacts")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("identity model artifact contract is invalid")
        field = record.get("field")
        raw_path = record.get("lexical_path")
        kind = record.get("kind")
        expected_hash = record.get("sha256")
        expected_size = record.get("size")
        if (
            not isinstance(field, str)
            or not field
            or field in seen
            or not isinstance(raw_path, str)
            or not os.path.isabs(raw_path)
            or kind not in ("file", "directory")
            or not isinstance(expected_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ValueError("identity model artifact contract is invalid")
        seen.add(field)
        path = _absolute_local_path(raw_path)
        repo = record.get("repo")
        commit = record.get("commit")
        trusted_link_root = None
        if repo is not None or commit is not None:
            if not isinstance(repo, str) or not isinstance(commit, str):
                raise ValueError("identity hosted model pin is invalid")
            _snapshot_root, trusted_link_root = _pinned_hf_roots(
                path, repo, commit)
        if kind == "file":
            actual_hash, actual_size = _hash_model_file(
                path, trusted_link_root=trusted_link_root)
            actual_files = None
        else:
            actual_hash, actual_size, actual_files = _hash_model_directory(
                path, trusted_link_root=trusted_link_root)
            expected_files = record.get("files")
            if (
                isinstance(expected_files, bool)
                or not isinstance(expected_files, int)
                or expected_files < 0
                or actual_files != expected_files
            ):
                raise ValueError(f"{field} model artifact file set changed")
        if actual_size != expected_size or actual_hash != expected_hash:
            raise ValueError(f"{field} model artifact bytes changed before load")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_git_revision(root: Path) -> str | None:
    """Resolve HEAD without spawning git (safe in the Flask process)."""
    git = root / ".git"
    if git.is_file():
        match = re.search(r"gitdir:\s*(.+)", _read_text(git), re.IGNORECASE)
        if not match:
            return None
        git = (git.parent / match.group(1).strip()).resolve()
    head = git / "HEAD"
    if not head.is_file():
        return None
    value = _read_text(head).strip()
    if not value.startswith("ref:"):
        return value if re.fullmatch(r"[0-9a-fA-F]{40,64}", value) else None
    ref = value[4:].strip()
    loose = git / Path(ref)
    if loose.is_file():
        candidate = _read_text(loose).strip()
        return candidate if re.fullmatch(r"[0-9a-fA-F]{40,64}", candidate) else None
    packed = git / "packed-refs"
    if packed.is_file():
        for line in _read_text(packed).splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            candidate, _, packed_ref = line.partition(" ")
            if packed_ref.strip() == ref and re.fullmatch(
                r"[0-9a-fA-F]{40,64}", candidate
            ):
                return candidate
    return None


def _tracked_worktree_clean(root: Path) -> bool | None:
    """Whether tracked ai-toolkit sources exactly match HEAD.

    A commit hash is not an immutable runtime identity when tracked files have
    local edits. Generated model/cache files are intentionally ignored here;
    they do not replace a tracked Python module.
    """
    try:
        process = subprocess.run(
            ["git", "-C", str(root), "diff-index", "--quiet", "HEAD", "--"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if process.returncode == 0:
        return True
    if process.returncode == 1:
        return False
    return None


def _version_from_source(root: Path) -> str | None:
    path = root / _VERSION_REL
    if not path.is_file():
        return None
    match = re.search(
        r"^\s*VERSION\s*=\s*['\"]([^'\"]+)['\"]",
        _read_text(path),
        re.MULTILINE,
    )
    return match.group(1) if match else None


def probe_aitoolkit_source(root: os.PathLike[str] | str) -> dict[str, Any]:
    """Recognise the inspected BaseSDTrainProcess/SDTrainer lifecycle.

    A version string alone is not sufficient: ai-toolkit has changed its train
    loop without consistently changing every public API.  The bridge therefore
    checks the exact lifecycle seams it relies on and reports ``supported=False``
    if any of them disappear or move into an unsafe order.
    """
    root = Path(root).expanduser().resolve()
    base_path = root / _BASE_REL
    sd_path = root / _SD_REL
    data_path = root / _DATA_REL
    mixin_path = root / _MIXIN_REL
    run_path = root / _RUN_REL
    missing_files = [
        str(path.relative_to(root))
        for path in (
            base_path,
            sd_path,
            data_path,
            mixin_path,
            run_path,
            root / _VERSION_REL,
        )
        if not path.is_file()
    ]
    revision = _resolve_git_revision(root)
    tracked_clean = _tracked_worktree_clean(root) if revision else None
    result: dict[str, Any] = {
        "bridge": BRIDGE_NAME,
        "bridge_version": BRIDGE_VERSION,
        "protocol": BRIDGE_PROTOCOL,
        "shape_revision": SHAPE_REVISION,
        "aitoolkit_root": str(root),
        "aitoolkit_version": _version_from_source(root),
        "aitoolkit_revision": revision,
        "tracked_worktree_clean": tracked_clean,
        "supported": False,
        "reasons": [],
    }
    if missing_files:
        result["reasons"].append(
            "missing ai-toolkit files: " + ", ".join(missing_files)
        )
        return result
    if revision and tracked_clean is not True:
        result["reasons"].append(
            "tracked ai-toolkit sources are locally modified"
            if tracked_clean is False
            else "cannot verify that tracked ai-toolkit sources match HEAD"
        )

    try:
        base_source = _read_text(base_path)
        sd_source = _read_text(sd_path)
        data_source = _read_text(data_path)
        mixin_source = _read_text(mixin_path)
        run_source = _read_text(run_path)
    except OSError as exc:
        result["reasons"].append(f"cannot read ai-toolkit sources: {exc}")
        return result

    missing_base = [marker for marker in _BASE_MARKERS if marker not in base_source]
    missing_sd = [marker for marker in _SD_MARKERS if marker not in sd_source]
    missing_data = [marker for marker in _DATA_MARKERS if marker not in data_source]
    missing_mixin = [
        marker for marker in _MIXIN_MARKERS if marker not in mixin_source
    ]
    if missing_base:
        result["reasons"].append(
            "unrecognised BaseSDTrainProcess lifecycle: "
            + ", ".join(missing_base)
        )
    if missing_sd:
        result["reasons"].append(
            "unrecognised SDTrainer optimizer lifecycle: " + ", ".join(missing_sd)
        )
    if missing_data:
        result["reasons"].append(
            "unrecognised AiToolkitDataset setup lifecycle: "
            + ", ".join(missing_data)
        )
    if missing_mixin:
        result["reasons"].append(
            "unrecognised bucket/latent-cache lifecycle: "
            + ", ".join(missing_mixin)
        )

    run_order = (
        run_source.find("load_dotenv()"),
        run_source.find("DISABLE_TELEMETRY"),
        run_source.find("import torch"),
        run_source.find("from toolkit.accelerator import get_accelerator"),
        run_source.find("accelerator = get_accelerator()"),
    )
    if min(run_order) < 0 or run_order != tuple(sorted(run_order)):
        result["reasons"].append(
            "run.py environment/torch/accelerator bootstrap order is unrecognised"
        )

    run_at = base_source.find("    def run(self):")
    run_source = base_source[run_at:] if run_at >= 0 else ""
    run_order = (
        run_source.find("self.hook_train_loop("),
        run_source.find("self.save(self.step_num)"),
        run_source.find("self.step_num = step + 1"),
    )
    if min(run_order) < 0 or run_order != tuple(sorted(run_order)):
        result["reasons"].append(
            "save is not recognised between the completed train step and next-step update"
        )

    hook_at = sd_source.find("    def hook_train_loop(")
    hook_source = sd_source[hook_at:] if hook_at >= 0 else ""
    optimizer_order = (
        hook_source.find("self.optimizer.step()"),
        hook_source.find("self.ema.update()"),
        hook_source.find("self.lr_scheduler.step()"),
    )
    # EMA can be conditional, but in the inspected lifecycle it still occurs
    # after optimizer.step and before scheduler.step.
    if min(optimizer_order) < 0 or optimizer_order != tuple(sorted(optimizer_order)):
        result["reasons"].append(
            "optimizer/EMA/scheduler order does not match the supported lifecycle"
        )

    setup_at = data_source.find("    def setup_epoch(self):")
    setup_source = data_source[setup_at:] if setup_at >= 0 else ""
    dataset_setup_order = (
        setup_source.find("self.setup_buckets()"),
        setup_source.find("self.cache_latents_all_latents()"),
        setup_source.find("self.epoch_num += 1"),
    )
    if (
        min(dataset_setup_order) < 0
        or dataset_setup_order != tuple(sorted(dataset_setup_order))
    ):
        result["reasons"].append(
            "bucket setup/cache/epoch order does not match the supported lifecycle"
        )

    bucket_at = mixin_source.find("    def setup_buckets(self")
    bucket_source = mixin_source[bucket_at:] if bucket_at >= 0 else ""
    bucket_order = (
        bucket_source.find("self.buckets = {}"),
        bucket_source.find("self.shuffle_buckets()"),
        bucket_source.find("self.build_batch_indices()"),
    )
    if min(bucket_order) < 0 or bucket_order != tuple(sorted(bucket_order)):
        result["reasons"].append(
            "bucket creation/shuffle/batch order does not match the supported lifecycle"
        )

    cache_at = mixin_source.find("    def cache_latents_all_latents(self")
    cache_source = mixin_source[cache_at:] if cache_at >= 0 else ""
    cache_order = (
        cache_source.find("file_item.get_latent_path(recalculate=True)"),
        cache_source.find("if os.path.exists(latent_path):"),
    )
    if min(cache_order) < 0 or cache_order != tuple(sorted(cache_order)):
        result["reasons"].append(
            "latent cache lookup order does not match the supported lifecycle"
        )

    text_cache_at = mixin_source.find("    def cache_text_embeddings(self")
    text_cache_source = (
        mixin_source[text_cache_at:] if text_cache_at >= 0 else ""
    )
    text_cache_order = (
        text_cache_source.find(
            "file_item.get_text_embedding_path(recalculate=True)"
        ),
        text_cache_source.find("if not os.path.exists(text_embedding_path):"),
    )
    if (
        min(text_cache_order) < 0
        or text_cache_order != tuple(sorted(text_cache_order))
    ):
        result["reasons"].append(
            "text cache lookup order does not match the supported lifecycle"
        )

    result["source_hashes"] = {
        str(_BASE_REL).replace("\\", "/"): _sha256_bytes(
            base_source.encode("utf-8")
        ),
        str(_SD_REL).replace("\\", "/"): _sha256_bytes(sd_source.encode("utf-8")),
        str(_DATA_REL).replace("\\", "/"): _sha256_bytes(
            data_source.encode("utf-8")
        ),
        str(_MIXIN_REL).replace("\\", "/"): _sha256_bytes(
            mixin_source.encode("utf-8")
        ),
        str(_RUN_REL).replace("\\", "/"): _sha256_bytes(
            run_source.encode("utf-8")
        ),
    }
    result["supported"] = not result["reasons"]
    return result


def load_identity(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Load and validate the immutable LDS context required for exact state."""
    identity_path = Path(path)
    try:
        value = read_json_nofollow(identity_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid identity file: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("identity file must contain a JSON object")
    if value.get("schema") != IDENTITY_SCHEMA:
        raise ValueError(
            f"identity schema must be {IDENTITY_SCHEMA!r}, got {value.get('schema')!r}"
        )
    missing = [field for field in REQUIRED_IDENTITY_FIELDS if not value.get(field)]
    if missing:
        raise ValueError("identity file is missing: " + ", ".join(missing))
    # Hashes are opaque canonical fingerprints supplied by LDS.  Requiring
    # strings prevents accidental null/structured values without assuming that
    # every fingerprint must use the same digest algorithm forever.
    for field in REQUIRED_IDENTITY_FIELDS:
        if field == "runtime":
            if not isinstance(value[field], Mapping):
                raise ValueError("identity runtime must be a JSON object")
        elif not isinstance(value[field], str):
            raise ValueError(f"identity {field} must be a string")
    return value
