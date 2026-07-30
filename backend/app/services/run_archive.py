"""Keep the PIXELS of every image a run trained on, so a deleted image is still
visible in a comparison months later.

The gap this closes: "what changed between these two trainings?" is answered by
the manifest for everything except the one change that matters most — *"an image
was removed"*. Once it is gone from the dataset there is nothing left to look at,
and the diff can only recite an id.

The design that made this affordable rather than a per-run gigabyte:

* **Content-addressed.** A file is stored once under its own content hash
  (`run_images/ab/ab12…cd.jpg`), the same hash the run snapshot already computes.
  Ten runs on an unchanged dataset store nothing the second time; a run that
  edited three crops stores three files. Two datasets that share an image share
  its blob.
* **Off the launch path.** Copying happens in a daemon thread AFTER the record is
  committed. A launch never waits for it and never fails because of it.
* **Bounded.** The store stops growing at `provenance.archive_max_gb` (default 5)
  instead of quietly eating the disk. Past the ceiling nothing is added and the
  diff honestly reports the image as unavailable rather than showing a wrong one.
* **Copies, not hardlinks.** A hardlink would cost zero bytes, but the app
  rewrites images IN PLACE (re-crop, "Reset to auto", watermark cleaning) and an
  in-place rewrite reuses the inode — the "archived" copy would silently change
  into the new version, which is the exact lie this exists to prevent.

Measured on a real 20-dataset library: 1471 kept images, 444 MB in total, ~295 KB
mean. A full deduplicated archive of the entire training history is well under a
gigabyte, and an incremental launch copies only what actually changed.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading

from .. import config as cfg

logger = logging.getLogger(__name__)

_DEFAULT_MAX_GB = 5.0
_ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

_lock = threading.Lock()
_size_cache = {'bytes': None}


def archive_root():
    root = cfg._data_dir() / 'run_images'
    root.mkdir(parents=True, exist_ok=True)
    return root


def _root_only():
    """The archive path WITHOUT creating it. `_blob_path` is called several times
    per image while probing extensions, and a compare payload probes a hundred
    images — an `mkdir` per probe is hundreds of pointless syscalls on a read
    path. Creation belongs to the write path (`store`), which calls
    `archive_root` itself."""
    return cfg._data_dir() / 'run_images'


def _blob_path(sig, ext):
    """`<root>/<first two hex chars>/<sig><ext>` — the two-level fan-out keeps a
    directory listing sane once the store holds thousands of blobs."""
    sig = str(sig)
    return _root_only() / sig[:2] / f'{sig}{ext}'


def _normalise_ext(filename):
    ext = os.path.splitext(str(filename or ''))[1].lower()
    return ext if ext in _ALLOWED_EXT else '.png'


def max_bytes() -> int:
    try:
        gb = float(cfg.get('provenance.archive_max_gb', _DEFAULT_MAX_GB))
    except (TypeError, ValueError):
        gb = _DEFAULT_MAX_GB
    return int(max(0.0, gb) * (1 << 30))


def enabled() -> bool:
    """False disables archiving entirely (`provenance.archive_max_gb: 0`). The
    diff then reports removed images as unavailable — honest, never invented."""
    return max_bytes() > 0


def sweep_partials() -> int:
    """Delete leftover `.part` files and return how many went.

    `store` copies to `<blob>.part` then renames, so a crash (or a kill) mid-copy
    leaves an orphan that nothing will ever finish or address — `path_for` only
    ever looks for the final names. They still counted towards the ceiling, so a
    few interrupted runs could quietly shrink the usable archive until the user
    hit "Clear archive" and wiped everything, including the good blobs. Run at
    the start of a store pass, where a `.part` can only be stale: the copy that
    would own one has not started yet, and the pass holds `_lock`.

    Never raises — a failed cleanup must not stop the archiving it precedes."""
    removed = 0
    try:
        for dirpath, _dirs, files in os.walk(_root_only()):
            for fn in files:
                if not fn.endswith('.part'):
                    continue
                try:
                    os.remove(os.path.join(dirpath, fn))
                    removed += 1
                except OSError:
                    logger.debug('could not remove the stale %s', fn, exc_info=True)
    except OSError:
        logger.debug('could not sweep stale .part files', exc_info=True)
    if removed:
        logger.info('run image archive: removed %d interrupted copy/copies', removed)
        _size_cache['bytes'] = None
    return removed


def size_bytes(refresh=False) -> int:
    """Bytes currently held by the archive. Cached: the Settings card asks for it
    on every open and walking thousands of small files each time is wasteful.

    `.part` files are EXCLUDED: an interrupted copy is not archive content — it
    is addressable by nothing and about to be swept — and counting it towards the
    ceiling would slowly starve the archive on a machine that gets killed a lot."""
    if not refresh and _size_cache['bytes'] is not None:
        return _size_cache['bytes']
    total = 0
    try:
        for dirpath, _dirs, files in os.walk(archive_root()):
            for fn in files:
                if fn.endswith('.part'):
                    continue
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    continue
    except OSError:
        total = 0
    _size_cache['bytes'] = total
    return total


def clear() -> dict:
    """Delete every archived blob. Destroys the ability to SHOW images that have
    since left their dataset; the runs, their settings and their captions are in
    the database and are untouched."""
    freed = size_bytes(refresh=True)
    try:
        shutil.rmtree(archive_root(), ignore_errors=True)
    finally:
        _size_cache['bytes'] = None
    archive_root()
    return {'freed_bytes': freed}


def path_for(sig, ext=None):
    """Absolute path of an archived blob, or None when it isn't stored (never
    archived, archived before the ceiling was hit, or cleared). `ext` narrows the
    lookup; without it every known extension is probed."""
    if not sig or not str(sig).isalnum():
        return None
    for candidate in ((ext,) if ext else _ALLOWED_EXT):
        p = _blob_path(sig, candidate)
        if p.is_file():
            return str(p)
    return None


def release(sigs) -> dict:
    """Delete the blobs of `sigs` — the LAST step of removing a run, once the
    caller has PROVEN no other run references them.

    The store is content-addressed and therefore SHARED: ten runs on an unchanged
    dataset all point at the same blob, which is exactly why a whole training
    history fits in well under a gigabyte. Deleting "the images of this run"
    naively would blank the comparison of every run that shares them, so this
    function refuses to do the accounting itself — it deletes precisely what it
    is handed, and the reference count is the caller's job (see
    `cloud_training._releasable_blob_sigs`, which subtracts every other run's
    snapshot signatures before calling here).

    Returns `{'deleted', 'freed_bytes'}`. Never raises: a blob we cannot remove
    (locked, already gone) is simply left in place — wasted space, never a failed
    deletion."""
    deleted = 0
    freed = 0
    with _lock:
        for sig in (sigs or ()):
            path = path_for(sig)
            if not path:
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            try:
                os.remove(path)
            except OSError:
                logger.debug('could not release archived blob %s', sig, exc_info=True)
                continue
            deleted += 1
            freed += size
        if deleted and _size_cache['bytes'] is not None:
            _size_cache['bytes'] = max(0, _size_cache['bytes'] - freed)
    return {'deleted': deleted, 'freed_bytes': freed}


def stored_count(sigs) -> int:
    """How many of `sigs` currently have a blob on disk — what the confirmation
    dialog announces as "N archived source images released"."""
    return sum(1 for sig in (sigs or ()) if path_for(sig))


def store(plan) -> dict:
    """Copy every `(source_path, sig, filename)` of `plan` that isn't stored yet.
    Synchronous; returns `{'added', 'skipped', 'full'}`. Never raises."""
    added = skipped = 0
    full = False
    if not enabled():
        return {'added': 0, 'skipped': len(plan or ()), 'full': True}
    ceiling = max_bytes()
    archive_root()                      # the one place the folder is created
    with _lock:
        # Under the lock, so any `.part` present is necessarily an orphan from a
        # past interrupted pass — this one has not written its own yet.
        sweep_partials()
        current = size_bytes()
        for src, sig, filename in (plan or ()):
            if not src or not sig:
                continue
            ext = _normalise_ext(filename)
            dst = _blob_path(sig, ext)
            try:
                if dst.is_file() or path_for(sig):
                    skipped += 1
                    continue
                size = os.path.getsize(src)
                if current + size > ceiling:
                    full = True
                    break
                dst.parent.mkdir(parents=True, exist_ok=True)
                # Copy to a temp name then rename: a crash mid-copy must never
                # leave a TRUNCATED blob sitting at the address of a valid hash.
                tmp = dst.with_suffix(dst.suffix + '.part')
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)
                current += size
                added += 1
            except OSError:
                logger.debug('could not archive %s', sig, exc_info=True)
                continue
        _size_cache['bytes'] = current
    if full:
        logger.info('run image archive is at its %.1f GB ceiling — not adding more',
                    ceiling / (1 << 30))
    return {'added': added, 'skipped': skipped, 'full': full}


def archive_async(plan) -> None:
    """Fire-and-forget `store(plan)` on a daemon thread. The launch has already
    committed by the time this is called; nothing downstream depends on it, so a
    machine that is killed mid-copy simply has fewer archived images."""
    if not plan or not enabled():
        return
    try:
        threading.Thread(target=_store_quietly, args=(list(plan),),
                         name='run-archive', daemon=True).start()
    except RuntimeError:
        logger.debug('could not start the archive thread', exc_info=True)


def _store_quietly(plan):
    try:
        store(plan)
    except Exception:
        logger.debug('run image archiving failed', exc_info=True)
