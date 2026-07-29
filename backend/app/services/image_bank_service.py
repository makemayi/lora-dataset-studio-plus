"""🗃️ Image bank service — triage a big unsorted folder into dataset-ready
selections.

The founding use case: "I exported 9 000 images from Telegram — now what?".
A bank references that folder IN PLACE (no copy, the source files are never
written to) and layers a funnel on top:

  1. inventory      — instant: walk the folder, register every image file;
  2. quality pass   — background thread, CPU/pure-PIL: sharpness, noise,
                      uniformity, dimensions, dHash + duplicate groups;
                      raw scores persist, FLAGS are computed at read time
                      against the config thresholds ('bank' section) so
                      recalibrating never needs a rescan;
  3. duplicates     — near-duplicate groups (same 64-bit dHash family as the
                      dataset import dedup) with keep-best / keep-first /
                      manual resolution — losers are REJECTED (a status),
                      never deleted from disk;
  4. subject pass   — optional, needs the face-scoring extra: InsightFace
                      embeddings (cached in an .npz next to the thumbs) +
                      person clustering, to sort a mixed dump by WHO is in
                      the frame without any reference photo;
  5. promotion      — the kept selection is COPIED into a dataset through the
                      normal import path (normalize to webp + perceptual
                      dedup vs the dataset), inheriting every downstream tool
                      (captions, watermarks, face scoring, training).

Long passes run through bank_jobs (one background thread per bank, polled via
the bank payload) — a 9 000-image folder must scan in minutes without ever
holding an HTTP request open or freezing the UI.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from sqlalchemy import and_, case, func, or_

from .. import config as cfg
from ..extensions import db
from ..models import BankImage, FaceDataset, FaceDatasetImage, ImageBank
from . import bank_jobs, bank_undo, path_guard, trash
from .face_dataset_service import (SCRAPE_IMPORT_MAX, _dhash, _download_scrape_item,
                                   _hamming, _SCRAPE_DL_WORKERS, _watermark_regions_payload,
                                   import_images, normalize_watermark_regions)
from .image_quality import ANALYSIS_MAX_SIDE, quality_metrics
from .image_provenance import ORIGINS, provenance_metrics

logger = logging.getLogger(__name__)

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
# Sanity cap — a bank is a triage layer, not a filesystem indexer. Way above
# the founding 9 000-image case, low enough to catch "I pointed it at C:\".
BANK_MAX_FILES = 50000
THUMB_MAX_SIDE = 320
_COMMIT_EVERY = 25          # scan DB flush cadence
_PROMOTE_CHUNK = 20         # files per import_images call (bounded memory)
_SQL_IN_CHUNK = 500         # SQLite bound-variable ceiling is 999
# A quality pass that keeps finding NOTHING on disk is not looking at a bank of
# broken images — it is looking at the wrong folder. Bail out after this many
# absent files (when they are at least half of what has been walked) rather than
# grind through 30 000 of them; a handful of genuinely deleted files stays a
# non-event and only shows up in the folder-sync note.
_MISSING_ABORT_AT = 20
MOVED_FOLDER_MSG = (
    "this folder no longer holds the bank's images — it may have been moved, "
    'renamed, or sit on a drive that is disconnected. Nothing was changed: '
    'point the bank at its new folder, then run the pass again.')


# --- thresholds -------------------------------------------------------------
def thresholds() -> dict:
    """The 'bank' config section, sanitized (a corrupt config.json value falls
    back to the default instead of poisoning every flag computation)."""
    out = {}
    for key, default in cfg.DEFAULTS['bank'].items():
        try:
            out[key] = float(cfg.get(f'bank.{key}', default))
        except (TypeError, ValueError):
            out[key] = float(default)
    out['dup_distance'] = int(out['dup_distance'])
    out['min_side'] = int(out['min_side'])
    return out


# --- storage helpers --------------------------------------------------------
def _bank_dir(bank_id) -> Path:
    return cfg.banks_root() / str(bank_id)


def _thumbs_dir(bank_id) -> Path:
    return _bank_dir(bank_id) / 'thumbs'


def _face_cache_path(bank_id) -> Path:
    return _bank_dir(bank_id) / 'face_cache.npz'


def _score_cache_path(bank_id) -> Path:
    return _bank_dir(bank_id) / 'score_cache.npz'


def _abs_under(base: str, relpath: str) -> str | None:
    """The containment-checked realpath of ``relpath`` under an ALREADY resolved
    ``base``. Split out of ``abs_image_path`` so a loop over thousands of rows can
    resolve the bank folder ONCE instead of per row: ``os.path.realpath`` is a
    filesystem call, and re-resolving the same unchanging bank folder for every
    image cost 424 ms of the 6 353-row curation pool alone (measured). Same
    strings out, one syscall in."""
    full = os.path.realpath(os.path.join(base, relpath))
    if os.path.normcase(full).startswith(os.path.normcase(base + os.sep)):
        return full
    return None


def abs_image_path(bank: ImageBank, row: BankImage) -> str | None:
    """Absolute SOURCE path of a bank image, or None when it escapes the
    bank's folder (belt & braces — relpaths only ever come from our own walk).

    ⚠️ This is the user's own file. It is READ-ONLY for us, and it is NOT what
    the app should display or copy once a watermark has been cleaned — every
    reader must go through resolved_image_path() instead (see its docstring)."""
    return _abs_under(os.path.realpath(bank.source_path), row.relpath)


def _clean_dir(bank_id) -> Path:
    """Where watermark-cleaned versions live — the bank's OWN working directory,
    next to thumbs/. Never inside the user's folder."""
    return _bank_dir(bank_id) / 'clean'


def clean_image_path(bank_id, image_id) -> Path:
    """The cleaned blob of one image (may not exist)."""
    return _clean_dir(bank_id) / f'{image_id}.webp'


def _rotated_dir(bank_id) -> Path:
    """Where manually TURNED copies live — the bank's own working directory,
    next to clean/. Never inside the user's folder."""
    return _bank_dir(bank_id) / 'rotated'


def rotated_image_path(bank_id, image_id, rotation, source: str) -> Path:
    """Where the turned copy of one image lives. Keyed on the angle AND on the
    source's extension so a rotated PNG stays a PNG; keyed on the CLEAN state too
    (via the source name) is unnecessary because every clean change drops the
    whole derived set (see drop_derived)."""
    ext = os.path.splitext(source)[1].lower() or '.png'
    return _rotated_dir(bank_id) / f'{image_id}.r{int(rotation)}{ext}'


def _ensure_rotated(bank_id, row: BankImage, source: str) -> str:
    """Materialise (once) the turned copy of ``source`` and return its path.

    ALWAYS built from the pristine source — never from a previously rotated copy
    — so the angle is applied exactly once no matter how many times the user
    clicked. Fails OPEN: an encoder problem serves the un-turned image rather
    than a 404, because a bank must stay browsable."""
    from .face_dataset_service import (normalize_rotation,
                                       transformed_image_bytes, rotate_transform)
    try:
        turn = normalize_rotation(row.rotation)
    except ValueError:
        return source
    if not turn:
        return source
    dst = rotated_image_path(bank_id, row.id, turn, source)
    if dst.is_file():
        return str(dst)
    try:
        payload = transformed_image_bytes(source, rotate_transform(turn))
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Two simultaneous GETs of the same turned image (grid + lightbox) each
        # build it. A SHARED temp name would make one of them fail on Windows
        # (the other still holds it) and degrade to serving the un-turned source;
        # a unique one lets both win and the last atomic replace decide.
        tmp = dst.with_name(f'{dst.name}.{os.getpid()}-{threading.get_ident()}.tmp')
        try:
            tmp.write_bytes(payload)
            os.replace(tmp, dst)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return str(dst)
    except (ValueError, OSError) as e:
        logger.warning('bank image %s could not be rotated: %s', row.id, e)
        return source


def resolved_image_path(bank: ImageBank, row: BankImage) -> str | None:
    """THE path every reader must use: the watermark-cleaned version when one
    exists, the untouched source otherwise — turned by the row's manual rotation
    when it carries one.

    A bank is a read-only view over a folder we must never write to, so neither
    cleaning a watermark nor turning an image can rewrite the source: both write
    a separate blob under the bank's working directory. That only pays off if the
    READERS prefer it, so there is exactly ONE resolver and all three known
    readers call it: promotion (the blob handed to import_images), the grid
    thumbnail, and the /bank/<id>/file route. A new reader calling
    abs_image_path() directly would silently serve the watermarked original —
    test_bank_watermark_clean.py asserts these three go through here."""
    path = None
    if row.watermark_clean_method:
        cleaned = clean_image_path(bank.id, row.id)
        if cleaned.is_file():
            path = str(cleaned)
    if path is None:
        path = abs_image_path(bank, row)
    if path is None or not getattr(row, 'rotation', None):
        return path
    return _ensure_rotated(bank.id, row, path)


# --- CRUD -------------------------------------------------------------------
def get_bank(user_id, bank_id) -> ImageBank | None:
    return ImageBank.query.filter_by(id=bank_id, user_id=user_id).first()


def create_bank(user_id, name, folder):
    """Register a folder as a bank: walk it recursively and create one row per
    image file. Instant (no decode) — scoring is the separate scan pass.
    Returns (bank, added). ValueError on a missing folder / too many files."""
    name = (name or '').strip()
    # Windows «Copier en tant que chemin» pastes the path quoted — unquote so
    # the direct paste works first try (same nicety as the dataset folder import).
    folder = (folder or '').strip().strip('"\'')
    if not name:
        raise ValueError('name is required')
    if not folder or not os.path.isdir(folder):
        raise ValueError(f'folder not found or not readable: {folder or "(empty)"}')
    # A bank and a dataset only ever TRANSIT images (by copy) — they never share
    # them. This folder is the one place that door was open: a dataset's storage
    # folder pasted here made a bank over the dataset's LIVE files, and this
    # bank's 🗑 Delete rejected then deleted images out of the dataset.
    conflict = path_guard.dataset_folder_conflict(folder)
    if conflict:
        raise ValueError(conflict['message'])
    folder = os.path.realpath(folder)
    rels = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                rels.append(os.path.relpath(os.path.join(root, f), folder))
                if len(rels) > BANK_MAX_FILES:
                    raise ValueError(
                        f'too many images in the folder (max {BANK_MAX_FILES})')
    bank = ImageBank(user_id=user_id, name=name, source_path=folder)
    db.session.add(bank)
    db.session.flush()          # need bank.id for the child rows
    for i, rel in enumerate(rels, 1):
        try:
            size = os.path.getsize(os.path.join(folder, rel))
        except OSError:
            size = None
        db.session.add(BankImage(bank_id=bank.id, relpath=rel, file_size=size))
        if i % 500 == 0:
            db.session.flush()
    db.session.commit()
    return bank, len(rels)


# --- folder sync (incremental re-inventory) ---------------------------------
# A bank points at a LIVE folder: the user keeps scraping/exporting into it long
# after the bank was created. Re-walking it is cheap (~5 ms for 3 000 files), so
# the app does it for them instead of making them rebuild a bank they have
# already triaged. The cooldown is not about CPU — it keeps the workspace's 2 s
# poll from hitting the disk (possibly a spun-down external drive) constantly.
FOLDER_SYNC_COOLDOWN = 60.0
_folder_sync = {}       # bank_id -> {'at': monotonic, 'result': {...}}
_EMPTY_SYNC = {'added': 0, 'missing': 0, 'unavailable': False, 'error': None}


def reset_folder_sync():
    """Drop the per-bank walk cooldowns (tests: bank ids restart at 1 with an
    in-memory DB, so a stale entry would silently skip the next test's walk)."""
    _folder_sync.clear()


def _sync_cached(bank_id) -> dict:
    """The last known folder state, with ``added`` zeroed — nothing was added by
    the call that is being answered from the cache."""
    last = _folder_sync.get(bank_id)
    return {**(last['result'] if last else _EMPTY_SYNC), 'added': 0}


def refresh_bank(user_id, bank_id, force=False) -> dict | None:
    """Re-inventory a bank's source folder: register the images that appeared in
    it since the last walk.

    STRICTLY ADDITIVE — the only write is an INSERT of relpaths we don't know
    yet. No row is ever deleted and no decision is ever reset (status, scores,
    quality_state, duplicate/semantic groups, captions, face verdicts), so a
    bank triaged over hours survives any number of refreshes. New rows land
    exactly like freshly inventoried ones (pending, unscanned), which is all the
    downstream passes need: the quality scan already only picks up rows with no
    quality_state, and it rebuilds the duplicate groups when it lands.

    Files that VANISHED from the folder are counted, never removed: an unplugged
    drive or a renamed folder would otherwise wipe a whole triage in one silent
    pass. The count is surfaced so the user can decide.

    Returns {'added', 'missing', 'unavailable', 'error'}, or None when the bank
    is unknown. ``force`` bypasses the cooldown (bank opened by hand)."""
    bank = get_bank(user_id, bank_id)
    if bank is None:
        return None
    now = time.monotonic()
    last = _folder_sync.get(bank_id)
    if not force and last and (now - last['at']) < FOLDER_SYNC_COOLDOWN:
        return _sync_cached(bank_id)
    # A live pass owns this bank's rows (the scan job works off a snapshot of
    # them and reports progress against a fixed total). Adding rows underneath
    # it is harmless for the data but would silently fall outside that total —
    # the next refresh, a second later, picks them up.
    if bank_jobs.running(bank_id):
        return _sync_cached(bank_id)

    folder = bank.source_path
    if not folder or not os.path.isdir(folder):
        return _remember_sync(bank_id, now, {**_EMPTY_SYNC, 'unavailable': True})

    known = {os.path.normcase(rel) for (rel,) in
             db.session.query(BankImage.relpath).filter_by(bank_id=bank_id)}
    seen, new_rels = set(), []
    try:
        for root, _dirs, files in os.walk(folder, onerror=lambda _e: None):
            for f in files:
                if not f.lower().endswith(IMG_EXTS):
                    continue
                rel = os.path.relpath(os.path.join(root, f), folder)
                key = os.path.normcase(rel)
                seen.add(key)
                if key not in known:
                    new_rels.append(rel)
    except OSError:
        # The folder went away mid-walk (drive unplugged) — report it and keep
        # every row: a partial walk must never be read as "these files are gone".
        return _remember_sync(bank_id, now, {**_EMPTY_SYNC, 'unavailable': True})

    error = None
    if new_rels and len(known) + len(new_rels) > BANK_MAX_FILES:
        # Same sanity cap as create_bank, applied to the TOTAL after the add.
        # Nothing is inserted: a half-imported folder is worse than an honest no.
        new_rels, error = [], (f'the folder now holds more than {BANK_MAX_FILES} '
                               'images — the new files were not added')
    new_rels.sort()
    for i, rel in enumerate(new_rels, 1):
        try:
            size = os.path.getsize(os.path.join(folder, rel))
        except OSError:
            size = None
        db.session.add(BankImage(bank_id=bank_id, relpath=rel, file_size=size))
        if i % 500 == 0:
            db.session.flush()
    if new_rels:
        db.session.commit()
    return _remember_sync(bank_id, now, {
        'added': len(new_rels),
        'missing': sum(1 for k in known if k not in seen),
        'unavailable': False, 'error': error})


def _remember_sync(bank_id, at, result) -> dict:
    _folder_sync[bank_id] = {'at': at, 'result': result}
    return dict(result)


def refresh_banks(user_id, force=False) -> dict:
    """refresh_bank() over every bank of the user — {bank_id: result}. Used by
    the bank list, which is loaded when the user NAVIGATES to the page (never
    polled), so it forces the walk: opening the tab right after dropping files
    in a folder must show them, and the cooldown would swallow that. Measured on
    a real library of 6 banks / 22 000 images: ~175 ms in total, the bulk of it
    one 15 800-image bank. A bank whose folder is unavailable simply reports it;
    it never fails the list."""
    out = {}
    ids = [row.id for row in ImageBank.query.with_entities(ImageBank.id)
           .filter_by(user_id=user_id).all()]
    for bank_id in ids:
        res = refresh_bank(user_id, bank_id, force=force)
        if res is not None:
            out[bank_id] = res
    return out


# --- relocate ---------------------------------------------------------------
_MISSING_SAMPLE = 8         # relpaths quoted back so the user can recognise them


class BankRelocateMismatch(ValueError):
    """The candidate folder holds none of the bank's files. Carries the counts
    so the route can report them instead of a bare sentence."""
    def __init__(self, message, preview):
        super().__init__(message)
        self.preview = preview


def _relocate_target(folder) -> str:
    """Normalise a pasted folder into an absolute path we can walk. Same
    unquoting nicety as create_bank (Windows «Copy as path» pastes quoted)."""
    folder = (folder or '').strip().strip('"\'')
    if not folder:
        raise ValueError('a folder is required')
    if not os.path.isdir(folder):
        raise ValueError(f'folder not found or not readable: {folder}')
    # Relocation is the SAME door as creation, just later: repointing a bank at a
    # dataset's storage folder shares the files exactly as creating it there
    # would. Refused in the dry run too, so the dialog never offers the move.
    conflict = path_guard.dataset_folder_conflict(folder)
    if conflict:
        raise ValueError(conflict['message'])
    return os.path.realpath(folder)


def relocate_preview(user_id, bank_id, folder) -> dict:
    """Dry-run a relocation: how much of THIS bank is in THAT folder?

    Nothing is written. Every known relpath is looked up under the candidate
    folder (one os.walk, matched case-insensitively through normcase — same
    keying as refresh_bank, so a drive letter or a folder re-created in another
    case still counts as the same tree). Returns {folder, total, found, missing,
    missing_sample, extra, same_folder}; ValueError on an unknown bank/folder.

    The point of the two numbers is that the user decides: a bank moved whole
    reads 29 759 found / 0 missing, and a mistyped folder reads 0 / 29 759 — a
    distinction the caller must never make silently on their behalf."""
    bank = get_bank(user_id, bank_id)
    if bank is None:
        raise ValueError('bank not found')
    target = _relocate_target(folder)
    known = {}
    for (rel,) in db.session.query(BankImage.relpath).filter_by(bank_id=bank_id):
        known.setdefault(os.path.normcase(rel), rel)
    seen = set()
    for root, _dirs, files in os.walk(target, onerror=lambda _e: None):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                seen.add(os.path.normcase(
                    os.path.relpath(os.path.join(root, f), target)))
    hit = seen & set(known)
    gone = sorted(known[k] for k in set(known) - hit)
    return {
        'folder': target,
        'total': len(known),
        'found': len(hit),
        'missing': len(gone),
        'missing_sample': gone[:_MISSING_SAMPLE],
        'extra': len(seen - hit),
        'same_folder': os.path.normcase(target) == os.path.normcase(
            os.path.realpath(bank.source_path or '')),
    }


def relocate_bank(user_id, bank_id, folder, confirm=False) -> dict:
    """Point a bank at a NEW folder, keeping every row and every analysis.

    Moving a bank costs nothing by construction: BankImage.relpath is relative
    to source_path, and scores / dhash / duplicate groups / face verdicts /
    captions / keep-reject decisions all hang off the row id. So this only
    rewrites ONE string — the danger is never the write, it is aiming it wrong.

    Hence: no call applies anything without ``confirm``; a folder that holds
    NONE of the bank's files is refused outright (that is a different folder,
    not a moved one); and a partial match goes through but deletes nothing —
    rows whose file did not come along keep their analysis and simply read as
    missing in the folder-sync note. Returns the preview dict plus
    {'applied', 'needs_confirm', 'overlaps'}."""
    if bank_jobs.running(bank_id):
        raise bank_jobs.BankJobBusy(bank_jobs.get(bank_id)['kind'])
    out = relocate_preview(user_id, bank_id, folder)
    out['needs_confirm'] = out['missing'] > 0
    out['applied'] = False
    if out['total'] and not out['found']:
        raise BankRelocateMismatch(
            'none of this bank\'s '
            f"{out['total']} image(s) are in that folder — it does not look "
            'like this bank. Pick the folder that CONTAINS the images '
            '(the one you moved), not its parent.', out)
    if not confirm:
        return out
    bank = get_bank(user_id, bank_id)
    bank.source_path = out['folder']
    db.session.commit()
    _folder_sync.pop(bank_id, None)     # next walk must see the new folder
    out['applied'] = True
    out['overlaps'] = overlapping_banks(user_id, bank_id)
    return out


def _is_imported_source(path) -> bool:
    """True when the bank's folder is one WE made ("Import to bank"), i.e. it sits
    under bank_sources_root — as opposed to a folder of the user's own that a bank
    merely points at, which we must never touch."""
    try:
        root = os.path.realpath(cfg.bank_sources_root())
        p = os.path.realpath(str(path or ''))
        # commonpath RAISES on Windows when the two paths sit on different drives
        # ("Paths don't have the same drive") — and a bank pointing at another disk
        # is precisely the common case. Raising here turned every such delete into
        # a 500. Different drive == certainly not under our root, so: False.
        return bool(p) and os.path.commonpath([root, p]) == root and p != root
    except (OSError, ValueError):
        return False


def delete_bank(user_id, bank_id) -> bool:
    """Drop the bank's ROWS and working data (thumbs + face cache). A folder of the
    user's OWN and its images are never touched.

    The one exception is a bank built by "Import to bank": its folder is a copy WE
    made under bank_sources_root, so deleting the bank must take it too — otherwise
    a full duplicate of the dataset stays on disk forever with nothing in the UI
    pointing at it. It goes to Trash, not unlink, so it stays recoverable."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return False
    if bank_jobs.running(bank_id):
        bank_jobs.cancel(bank_id)
    imported_source = bank.source_path if _is_imported_source(bank.source_path) else None
    BankImage.query.filter_by(bank_id=bank_id).delete(synchronize_session=False)
    db.session.delete(bank)
    db.session.commit()
    bank_undo.clear(bank_id)     # its rows are gone; a stale offer would outlive them
    reset_score_memo()           # ~45 MB of embeddings for a bank that no longer is
    shutil.rmtree(_bank_dir(bank_id), ignore_errors=True)
    if imported_source and os.path.isdir(imported_source):
        try:
            trash.send_to_trash(imported_source, context=f'bank-{bank_id}')
        except OSError:
            logger.warning('delete_bank: could not trash the imported copy %s',
                           imported_source, exc_info=True)
    return True


# --- flags & payloads -------------------------------------------------------
def image_flags(row: BankImage, th: dict) -> list:
    """Threshold verdicts for one image, recomputed from the raw scores."""
    if row.quality_state == 'unreadable':
        return ['unreadable']
    flags = []
    if row.quality_state == 'ok':
        if row.blur_score is not None and row.blur_score < th['sharpness_min']:
            flags.append('blur')
        if row.noise_score is not None and row.noise_score > th['noise_max']:
            flags.append('noise')
        if row.uniformity_score is not None and row.uniformity_score < th['uniformity_min']:
            flags.append('uniform')
        if row.width and row.height and min(row.width, row.height) < th['min_side']:
            flags.append('small')
        # Effective resolution — the picture stops before the pixels do. Same
        # read-time-verdict philosophy as blur: raw score in, threshold applied
        # here, so retuning detail_min re-sorts the bank with no rescan.
        if row.detail_ratio is not None and row.detail_ratio < th['detail_min']:
            flags.append('soft_detail')
        if row.bars_ratio is not None and row.bars_ratio > th['bars_max']:
            flags.append('bars')
    # V2 scoring flags — derived from the persisted scores against the live
    # thresholds too, but NOT gated on the quality state (a watermarked or NSFW
    # image can be perfectly sharp). Only present once the relevant pass has run.
    if row.aesthetic_score is not None and row.aesthetic_score < th['aesthetic_min']:
        flags.append('low_aesthetic')
    if row.nsfw_score is not None and row.nsfw_score > th['nsfw_max']:
        flags.append('nsfw')
    if row.watermark_state == 'detected':
        flags.append('watermark')
    return flags


def _promoted_dataset_by_image(image_ids) -> dict:
    """{bank_image_id: dataset_id} for the images a dataset REALLY holds right
    now, read off the back-links. Only the ids of the page being rendered, so a
    30 000-image bank still costs one small query. An image promoted into
    several datasets reports the lowest id — the ⬆ badge only says THAT it
    landed somewhere, and a stable pick keeps the grid from flickering."""
    out: dict = {}
    ids = [int(i) for i in image_ids]
    for i0 in range(0, len(ids), _SQL_IN_CHUNK):
        rows = (db.session.query(FaceDatasetImage.bank_image_id,
                                 func.min(FaceDatasetImage.dataset_id))
                .filter(FaceDatasetImage.bank_image_id.in_(ids[i0:i0 + _SQL_IN_CHUNK]))
                .group_by(FaceDatasetImage.bank_image_id).all())
        out.update({bid: ds for bid, ds in rows})
    return out


def _page_images(rows, th: dict) -> list:
    """One page of grid payloads, with the ⬆ promoted state resolved in a single
    extra query for the whole page (never one per row)."""
    promoted_by = _promoted_dataset_by_image([r.id for r in rows])
    return [_image_dict(r, th, promoted_by) for r in rows]


def _image_dict(row: BankImage, th: dict, promoted_by: dict | None = None) -> dict:
    # ⬆ promoted = the dataset that holds this image TODAY (back-link), falling
    # back to the legacy one-way flag for promotions that predate it. Deriving it
    # means the badge disappears when the user deletes the image in the dataset,
    # instead of advertising a copy that is gone.
    promoted = (promoted_by or {}).get(row.id, row.promoted_dataset_id)
    # A quarter turn transposes what the user SEES. The columns keep the source's
    # own numbers (a re-scan rewrites them from the file, which never changed), so
    # the swap happens here, at read time — the payload can never drift out of
    # sync with the stored angle.
    rotation = int(row.rotation or 0) % 360
    width, height = ((row.height, row.width) if rotation in (90, 270)
                     else (row.width, row.height))
    # The mask editor's seed, carried ONLY on the rows that can open it: a bank
    # page is thousands of images and the other 99% would pay for three null keys.
    mask = {}
    if row.watermark_state == 'detected' or row.watermark_regions is not None:
        import json as _json
        try:
            bbox = _json.loads(row.watermark_bbox or '')
        except (ValueError, TypeError):
            bbox = None
        mask = {'watermark_bbox': bbox if isinstance(bbox, list) and len(bbox) == 4
                else None,
                **_watermark_regions_payload(row)}
    return {
        **mask,
        'id': row.id,
        'name': os.path.basename(row.relpath),
        'relpath': row.relpath,
        'rotation': rotation,
        'width': width, 'height': height, 'file_size': row.file_size,
        'quality_state': row.quality_state,
        'blur_score': row.blur_score, 'noise_score': row.noise_score,
        'uniformity_score': row.uniformity_score,
        'aesthetic_score': row.aesthetic_score, 'nsfw_score': row.nsfw_score,
        'style_cluster': row.style_cluster, 'watermark_state': row.watermark_state,
        'watermark_clean_method': row.watermark_clean_method,
        'detail_ratio': row.detail_ratio, 'bars_ratio': row.bars_ratio,
        'jpeg_quality': row.jpeg_quality,
        'origin': row.origin, 'origin_evidence': row.origin_evidence,
        'subfolder': _subfolder_of(row.relpath),
        'flags': image_flags(row, th),
        'dup_group': row.dup_group,
        'semantic_dup_group': row.semantic_dup_group,
        'face_state': row.face_state, 'face_cluster': row.face_cluster,
        'framing': row.framing,
        'status': row.status, 'reject_reason': row.reject_reason,
        'promoted_dataset_id': promoted,
        # The OTHER destination. Kept as its own key rather than overloading the
        # dataset one, which is stored in user databases and read as a dataset id.
        'promoted_bank_id': row.promoted_bank_id,
        'caption': row.caption,
    }


def _flag_filter(flag: str, th: dict):
    """SQLAlchemy criterion for one flag name (mirrors image_flags)."""
    if flag == 'unreadable':
        return BankImage.quality_state == 'unreadable'
    # V2 scoring flags — not gated on quality_state (see image_flags) and only
    # true where the score actually exists (a NULL score is "not scored", never
    # "below threshold"). watermark is a discrete state, no threshold column.
    if flag == 'low_aesthetic':
        return and_(BankImage.aesthetic_score.isnot(None),
                    BankImage.aesthetic_score < th['aesthetic_min'])
    if flag == 'nsfw':
        return and_(BankImage.nsfw_score.isnot(None),
                    BankImage.nsfw_score > th['nsfw_max'])
    if flag == 'watermark':
        return BankImage.watermark_state == 'detected'
    ok = BankImage.quality_state == 'ok'
    crit = {
        'blur': BankImage.blur_score < th['sharpness_min'],
        'noise': BankImage.noise_score > th['noise_max'],
        'uniform': BankImage.uniformity_score < th['uniformity_min'],
        'small': or_(BankImage.width < th['min_side'],
                     BankImage.height < th['min_side']),
        # NULL-safe: a row scanned before the provenance pass existed carries no
        # score, and "not measured" must never read as "below threshold".
        'soft_detail': and_(BankImage.detail_ratio.isnot(None),
                            BankImage.detail_ratio < th['detail_min']),
        'bars': and_(BankImage.bars_ratio.isnot(None),
                     BankImage.bars_ratio > th['bars_max']),
    }.get(flag)
    return (ok & crit) if crit is not None else None


_QUALITY_FLAGS = ('blur', 'noise', 'uniform', 'small', 'soft_detail', 'bars',
                  'unreadable')
# V2 score-derived flags. Kept separate from _QUALITY_FLAGS so the "flagged" /
# "clean" quality aggregate stays about the CPU quality pass, while these count
# and filter independently (each only meaningful once its pass has run).
_SCORE_FLAGS = ('low_aesthetic', 'nsfw', 'watermark')

# Resolution tiers for the Bank grid — bucketed on MEGAPIXELS (width×height, the
# same rank as the resolution sort) so a mixed dump can be skimmed and mass-acted
# one tier at a time. Each entry is (stable_id, lo, hi) in raw pixels, a HALF-OPEN
# [lo, hi) range (lower-inclusive, upper-exclusive); hi=None means "no ceiling".
# So a 1000×1000 (1.00 MP) and a 1024×1024 (1.05 MP) both land in 'res_1_2', and a
# 2000×2000 (4.00 MP) lands in 'res_gt_4'. The 0.25 MP floor (not 0.30) is chosen
# so a 512×512 (0.26 MP) — a legit small training crop — sits in '0.25–1 MP', while
# only true junk (Telegram thumbnails ~0.1–0.2 MP, ≤448²) falls in '< 0.25 MP'.
# Ids are user-facing filter keys — never rename without an alias.
_RES_BUCKETS = (
    ('res_lt_025', 0, 250_000),
    ('res_025_1', 250_000, 1_000_000),
    ('res_1_2', 1_000_000, 2_000_000),
    ('res_2_4', 2_000_000, 4_000_000),
    ('res_gt_4', 4_000_000, None),
)
_RES_BOUNDS = {bid: (lo, hi) for bid, lo, hi in _RES_BUCKETS}

# Framing buckets — the SAME four shot types the datasets classify (face close-up,
# bust, full body, back view). 'unknown' is a parseable-but-not-one-of-four answer;
# it counts and filters but is never a training target. Ids are user-facing filter
# keys — never rename without an alias. The built-in character composition aims for
# a 12/6/6/1 face/bust/body/back mix; the coverage advice phrases against that
# proportion, never as a hard rule.
_FRAMINGS = ('face', 'bust', 'body', 'back')
_FRAMING_KEYS = _FRAMINGS + ('unknown',)
_FRAMING_TARGET = {'face': 12, 'bust': 6, 'body': 6, 'back': 1}


def _framing_counts(bank_id, extra_crit=None) -> dict:
    """Per-bucket image counts for the 📐 Framing chips in ONE GROUP BY. Rows with
    a NULL framing (not classified) are excluded, so every key is present with a
    real count. ``extra_crit`` narrows the pool (e.g. status='keep' for coverage)."""
    q = (db.session.query(BankImage.framing, func.count(BankImage.id))
         .filter(BankImage.bank_id == bank_id, BankImage.framing.isnot(None)))
    if extra_crit is not None:
        q = q.filter(extra_crit)
    got = {k: n for k, n in q.group_by(BankImage.framing).all()}
    return {k: int(got.get(k, 0)) for k in _FRAMING_KEYS}


def _origin_counts(bank_id) -> dict:
    """Per-state image counts for the 🔎 Origin chips in ONE GROUP BY.

    Every state of ORIGINS is always present with a real count, INCLUDING
    'unknown' — which is the honest majority answer on any scraped or chat-sourced
    bank (measured: 3000/3000 on a real Telegram export) and has to be visible as
    such. Rows with a NULL origin (scanned before this pass existed, or
    unreadable) are excluded: they are "not measured", a different thing again."""
    q = (db.session.query(BankImage.origin, func.count(BankImage.id))
         .filter(BankImage.bank_id == bank_id, BankImage.origin.isnot(None)))
    got = {k: n for k, n in q.group_by(BankImage.origin).all()}
    return {k: int(got.get(k, 0)) for k in ORIGINS}


def _subfolder_of(relpath: str) -> str:
    """Top-level subfolder of a bank-relative path ('' for a root-level file) —
    the natural scoping axis for a Telegram export (one folder per chat/date)."""
    parts = (relpath or '').replace('\\', '/').split('/', 1)
    return parts[0] if len(parts) > 1 else ''


def _unresolved_dup_groups_q(bank_id, col=BankImage.dup_group):
    """Groups still holding ≥2 NON-rejected members — i.e. still to resolve. ``col``
    selects the stage: dup_group (exact/resized) or semantic_dup_group (crops)."""
    return (db.session.query(col)
            .filter(BankImage.bank_id == bank_id,
                    col.isnot(None),
                    BankImage.status != 'reject')
            .group_by(col)
            .having(func.count(BankImage.id) >= 2))


def _res_bucket_case():
    """A single SQL CASE mapping each scanned row to its resolution-tier id, used
    both to COUNT per tier (one GROUP BY) and — via _RES_BOUNDS — to FILTER a page
    to one tier. Rows with a NULL dimension never reach this (callers pre-filter
    width/height NOT NULL), so no NULL-misfile into the top tier."""
    area = BankImage.width * BankImage.height
    whens = [(area < hi, bid) for bid, _lo, hi in _RES_BUCKETS if hi is not None]
    return case(*whens, else_=_RES_BUCKETS[-1][0])


def _res_bucket_counts(bank_id) -> dict:
    """Per-tier image counts for the resolution chips (bank-wide, like the flag
    totals) in ONE GROUP BY. Unscanned rows (width/height NULL) are excluded, so a
    tier that no image falls into simply reports 0. Every tier id is present."""
    bucket = _res_bucket_case()
    rows = (db.session.query(bucket, func.count(BankImage.id))
            .filter(BankImage.bank_id == bank_id,
                    BankImage.width.isnot(None),
                    BankImage.height.isnot(None))
            .group_by(bucket).all())
    got = {bid: n for bid, n in rows}
    return {bid: int(got.get(bid, 0)) for bid, _lo, _hi in _RES_BUCKETS}


# --- explicit grid sorts -----------------------------------------------------
# The grid can ORDER on what the passes already MEASURED, instead of only
# filtering on it (asked for by nofaceman on Discord). One entry per sortable
# quantity; the UI offers each in both directions as '<key>_desc' / '<key>_asc'.
# Ids are user-facing query values — treat them like catalog labels and never
# rename one without an alias.
#   res       megapixels (width×height) — the original sort, kept as-is.
#   aesthetic the ✨ Score pass's 1–10 rating: ↓ surfaces the keepers, ↑ the duds.
#   sharp     the 🔎 Scan pass's Laplacian variance: ↑ surfaces the blurry misses.
# Deliberately NOT here: noise / uniformity / bars / detail_ratio / NSFW. Each
# already has a chip that filters AND orders worst-first, so a sort entry would
# duplicate an existing gesture — and a fifteen-line menu slows the review down
# more than the missing order costs.
_SORT_KEYS = {
    'aesthetic': lambda: BankImage.aesthetic_score,
    'sharp': lambda: BankImage.blur_score,
    'res': lambda: BankImage.width * BankImage.height,
}
GRID_SORTS = tuple(f'{k}_{d}' for k in ('res', 'aesthetic', 'sharp')
                   for d in ('desc', 'asc'))


def _sort_order(sort):
    """The ORDER BY tuple for an explicit grid sort, or None for 'default' and
    for anything unknown (an unrecognised value must degrade to the server's own
    order, never 500). Rows the relevant pass never reached carry NULL and sink
    to the END in BOTH directions — ordering by "is NULL" first (0 before 1)
    — because a sort that opens on the un-measured pile is worse than no sort.
    Tie-break on id so a page boundary is stable."""
    key, _, direction = (sort or '').rpartition('_')
    if direction not in ('asc', 'desc') or key not in _SORT_KEYS:
        return None
    col = _SORT_KEYS[key]()
    ranked = col.desc() if direction == 'desc' else col.asc()
    return (col.is_(None).asc(), ranked, BankImage.id.asc())


def bank_payload(user_id, bank_id) -> dict | None:
    """Everything the bank workspace needs on one poll: counts, flag totals,
    duplicate/cluster summaries, live job, thresholds."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    base = BankImage.query.filter_by(bank_id=bank_id)
    total = base.count()
    counts = {
        'total': total,
        'scanned': base.filter(BankImage.quality_state.isnot(None)).count(),
        'pending': base.filter_by(status='pending').count(),
        'keep': base.filter_by(status='keep').count(),
        'reject': base.filter_by(status='reject').count(),
        # Images a dataset REALLY holds today (back-link), plus the ones promoted
        # before that link existed (legacy flag). Counting the flag alone kept
        # advertising copies the user had since deleted.
        'promoted': base.filter(or_(
            BankImage.promoted_dataset_id.isnot(None),
            # ...or into another BANK, the second destination. Counted here so
            # the "promoted" stat and the ⬆ badge on the tiles never disagree.
            BankImage.promoted_bank_id.isnot(None),
            BankImage.id.in_(db.session.query(FaceDatasetImage.bank_image_id)
                             .filter(FaceDatasetImage.bank_image_id.isnot(None))),
        )).count(),
        # V2 pass progress — how many images the scoring / watermark passes reached
        # (so the UI can show "scored 0/9000" and enable the threshold facets).
        'scored': base.filter(or_(BankImage.aesthetic_score.isnot(None),
                                  BankImage.nsfw_score.isnot(None))).count(),
        'watermark_scanned': base.filter(BankImage.watermark_state.isnot(None)).count(),
        'framing_classified': base.filter(BankImage.framing.isnot(None)).count(),
    }
    framing = _framing_counts(bank_id)
    flags = {}
    for flag in _QUALITY_FLAGS + _SCORE_FLAGS:
        crit = _flag_filter(flag, th)
        flags[flag] = base.filter(crit).count() if crit is not None else 0
    res_buckets = _res_bucket_counts(bank_id)
    origins = _origin_counts(bank_id)
    dup_rows = (db.session.query(BankImage.dup_group, func.count(BankImage.id))
                .filter(BankImage.bank_id == bank_id,
                        BankImage.dup_group.isnot(None))
                .group_by(BankImage.dup_group).all())
    dup = {'groups': len(dup_rows),
           'images': sum(n for _g, n in dup_rows),
           'unresolved': _unresolved_dup_groups_q(bank_id).count()}
    # Stage-2 semantic near-duplicate groups (crops/variants), same summary shape.
    sem_rows = (db.session.query(BankImage.semantic_dup_group, func.count(BankImage.id))
                .filter(BankImage.bank_id == bank_id,
                        BankImage.semantic_dup_group.isnot(None))
                .group_by(BankImage.semantic_dup_group).all())
    semantic_dup = {
        'groups': len(sem_rows),
        'images': sum(n for _g, n in sem_rows),
        'unresolved': _unresolved_dup_groups_q(
            bank_id, BankImage.semantic_dup_group).count()}
    # Person clusters, biggest first; cover = the member with the surest face.
    cl_rows = (db.session.query(BankImage.face_cluster, func.count(BankImage.id))
               .filter(BankImage.bank_id == bank_id,
                       BankImage.face_cluster.isnot(None))
               .group_by(BankImage.face_cluster)
               .order_by(func.count(BankImage.id).desc(),
                         BankImage.face_cluster.asc())
               .limit(40).all())
    clusters = []
    for cid, size in cl_rows:
        cover = (BankImage.query
                 .filter_by(bank_id=bank_id, face_cluster=cid)
                 .order_by(BankImage.face_det.desc().nullslast(),
                           BankImage.id.asc())
                 .first())
        clusters.append({'id': cid, 'size': size,
                         'cover_image_id': cover.id if cover else None})
    faces_scanned = base.filter(BankImage.face_state.isnot(None)).count()
    # Style clusters (group by visual style), biggest first — the "group by
    # style" counterpart to the person clusters above. Cover = the lowest id of
    # the cluster (stable, no per-image quality signal to rank by here).
    st_rows = (db.session.query(BankImage.style_cluster, func.count(BankImage.id))
               .filter(BankImage.bank_id == bank_id,
                       BankImage.style_cluster.isnot(None))
               .group_by(BankImage.style_cluster)
               .order_by(func.count(BankImage.id).desc(),
                         BankImage.style_cluster.asc())
               .limit(40).all())
    style_clusters = []
    for cid, size in st_rows:
        cover = (BankImage.query
                 .filter_by(bank_id=bank_id, style_cluster=cid)
                 .order_by(BankImage.id.asc()).first())
        style_clusters.append({'id': cid, 'size': size,
                               'cover_image_id': cover.id if cover else None})
    return {
        'id': bank.id, 'name': bank.name, 'source_path': bank.source_path,
        'created_at': bank.created_at.isoformat() if bank.created_at else None,
        'counts': counts, 'flags': flags, 'res_buckets': res_buckets,
        'framing': framing, 'origins': origins, 'dup': dup,
        'semantic_dup': semantic_dup,
        'clusters': clusters, 'faces_scanned': faces_scanned,
        'style_clusters': style_clusters,
        'activity': bank_jobs.get(bank_id),
        # ↩ the one-step-back offer, so the bar survives a reload (the decision
        # it takes back is in the database, not in a tab).
        'undo': bank_undo.peek(bank_id),
        'pipeline_report': _load_pipeline_report(bank),
        'score_device': score_device_info(bank_id),
        # Non-null only on a bank created before the create-time guard, whose
        # folder IS a dataset's storage. The workspace turns it into a standing
        # banner and disables 🗑 Delete rejected, which the server refuses anyway.
        'dataset_conflict': bank_dataset_conflict(user_id, bank_id),
        'thresholds': th,
    }


def flag_preview(user_id, bank_id, overrides=None) -> dict | None:
    """Per-flag image counts for a CANDIDATE threshold set — what the bank WOULD
    look like at those numbers, without saving anything.

    This is what turns tuning a threshold from a guess into a decision: the
    quality scan persists RAW scores and every verdict is recomputed at read
    time (see BankImage's docstring), so answering "how many images would a
    sharpness floor of 140 flag?" is the same COUNT the payload already runs,
    with a different dict. No decode, no pass, no write.

    Only the read-time thresholds are meaningful here. The four grouping ones
    (dup_distance, face/style/semantic similarity) are baked into stored group
    ids by their pass, so a count against a candidate value would be a number
    about the OLD grouping — the UI says "applies at the next pass" instead.

    Unknown keys and junk values are ignored rather than 400: this is a live
    preview firing on every keystroke, and a half-typed "0." must degrade to
    "no change yet", never to an error toast."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    for key, val in (overrides or {}).items():
        if key not in cfg.DEFAULTS['bank']:
            continue
        try:
            th[key] = float(val)
        except (TypeError, ValueError):
            continue
    th['dup_distance'] = int(th['dup_distance'])
    th['min_side'] = int(th['min_side'])
    base = BankImage.query.filter_by(bank_id=bank_id)
    flags = {}
    for flag in _QUALITY_FLAGS + _SCORE_FLAGS:
        crit = _flag_filter(flag, th)
        flags[flag] = base.filter(crit).count() if crit is not None else 0
    return {'flags': flags, 'thresholds': th, 'total': base.count()}


def _load_pipeline_report(bank: ImageBank):
    """The persisted 'Launch all' summary (parsed), or None. A corrupt blob is
    swallowed — a broken report must never 500 the whole bank payload."""
    import json as _json
    raw = getattr(bank, 'pipeline_report', None)
    if not raw:
        return None
    try:
        return _json.loads(raw)
    except (ValueError, TypeError):
        return None


def list_banks(user_id, dataset_id=None) -> list:
    """Every bank of the user, newest first, with its triage counters, the ids
    of its card preview images — and, when ``dataset_id`` is given, how many
    kept images each bank would promote into THAT dataset (``promotable``).

    The promotable counts ride along on purpose: the dataset-side bank chooser
    used to ask /bank/<id>/promotable once per bank, so a library of 12 banks
    cost 13 requests to open one panel. One grouped query answers them all."""
    promotable = _promotable_counts(user_id, dataset_id) if dataset_id is not None else None
    out = []
    for bank in (ImageBank.query.filter_by(user_id=user_id)
                 .order_by(ImageBank.created_at.desc()).all()):
        base = BankImage.query.filter_by(bank_id=bank.id)
        row = {
            'id': bank.id, 'name': bank.name, 'source_path': bank.source_path,
            'created_at': bank.created_at.isoformat() if bank.created_at else None,
            'total': base.count(),
            'keep': base.filter_by(status='keep').count(),
            'reject': base.filter_by(status='reject').count(),
            'scanned': base.filter(BankImage.quality_state.isnot(None)).count(),
            'preview_ids': _preview_ids(bank.id),
            'activity': bank_jobs.get(bank.id),
        }
        if promotable is not None:
            row['promotable'] = promotable.get(bank.id, 0)
        out.append(row)
    return out


def _promotable_counts(user_id, dataset_id) -> dict | None:
    """{bank_id: promotable count} for EVERY bank at once — the batched form of
    promotable_count(), same eligibility rule (see _promotable_query). None when
    the dataset is gone or the id is junk, so the caller omits the field rather
    than publishing zeros it can't stand behind. Banks with nothing eligible are
    absent from the mapping; list_banks reads them as 0."""
    try:
        dataset_id = int(dataset_id)
    except (TypeError, ValueError):
        return None
    if not FaceDataset.query.filter_by(id=dataset_id, user_id=user_id).first():
        return None
    rows = (db.session.query(BankImage.bank_id, func.count(BankImage.id))
            .join(ImageBank, ImageBank.id == BankImage.bank_id)
            .filter(ImageBank.user_id == user_id,
                    BankImage.status == 'keep',
                    _not_already_on(dataset_id))
            .group_by(BankImage.bank_id).all())
    return {bank_id: n for bank_id, n in rows}


PREVIEW_COUNT = 5


def _preview_ids(bank_id, limit=PREVIEW_COUNT) -> list:
    """The first few image ids of a bank, for the card's thumbnail strip.
    Ordered by id (= inventory order), so the strip is STABLE across reloads —
    and rejected shots are skipped so a triaged bank doesn't advertise its
    discards. Kept images are deliberately NOT promoted to the front: most banks
    sit at zero keeps for their whole life, and re-ordering the strip as the user
    triages would make the card flicker under them. One query per bank, so the
    whole page still costs a single HTTP request."""
    rows = (BankImage.query.with_entities(BankImage.id)
            .filter(BankImage.bank_id == bank_id, BankImage.status != 'reject')
            .order_by(BankImage.id.asc()).limit(limit).all())
    return [r[0] for r in rows]


def list_images(user_id, bank_id, status=None, flag=None, cluster=None,
                group=None, style=None, subfolder=None, search=None,
                semantic_group=None, sort=None, res_bucket=None, framing=None,
                origin=None, ids=None, offset=0, limit=200) -> dict | None:
    """One PAGE of the bank grid (a 9 000-image bank must never ship whole).
    Filters compose: status ∩ flag ∩ cluster ∩ dup-group ∩ style ∩ subfolder ∩ search.
    ``search`` is a plain full-text term matched (case-insensitive LIKE) against the
    caption AND the relpath — so captions double as searchable tags for a big dump
    ("red dress"), combinable with every other filter. Flag filters sort by the
    relevant score (worst first) so the review reads top-down.
    ``sort`` (a GRID_SORTS id — resolution / aesthetic / sharpness, each way)
    overrides that order: resolution ranks on megapixels (width×height, so
    900×900 outranks 1200×300), aesthetic on the ✨ Score rating, sharpness on
    the 🔎 Scan Laplacian variance. Rows the matching pass never reached (NULL)
    always sink to the end, in BOTH directions. It composes with every filter,
    and — since "Select all in filter" / ▶ Review page this SAME endpoint — the
    selection walks the order the user is looking at.
    ``res_bucket`` (a _RES_BUCKETS id) narrows to one resolution tier — a
    half-open [lo, hi) megapixel band — and composes with every filter AND the
    sort (the tier + Resolution↑/↓ combo is the mixed-dump cleanup flow).
    ``ids`` is the "show selected" VIEW: an explicit ordered list of image ids
    that OVERRIDES every facet/sort (the selection IS the scope) and renders the
    page in the SAME order the caller passed — so a similarity ranking from
    ``select_similar`` shows reference-first, closest-to-farthest, instead of the
    default id order. Unknown/foreign ids are dropped silently."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    if ids is not None:
        # Explicit id view — the selection is the scope, order is preserved.
        # Dedupe keeping first occurrence so the requested order is authoritative.
        seen = set()
        ordered = [i for i in ids if not (i in seen or seen.add(i))]
        by_id = {}
        # Chunk the IN() so a big selection can't blow past SQLite's bound-
        # variable limit (default 999); the curation cap is 2 000 ids.
        for start in range(0, len(ordered), 500):
            chunk = ordered[start:start + 500]
            for r in (BankImage.query
                      .filter(BankImage.bank_id == bank_id,
                              BankImage.id.in_(chunk)).all()):
                by_id[r.id] = r
        ordered_rows = [by_id[i] for i in ordered if i in by_id]
        total = len(ordered_rows)
        off = max(0, int(offset))
        page = ordered_rows[off:off + max(1, min(500, int(limit)))]
        return {'images': _page_images(page, th), 'total': total, 'offset': off}
    q = BankImage.query.filter_by(bank_id=bank_id)
    if status in ('pending', 'keep', 'reject'):
        q = q.filter(BankImage.status == status)
    order = BankImage.id.asc()
    if flag == 'flagged':
        crits = [c for c in (_flag_filter(f, th) for f in _QUALITY_FLAGS)
                 if c is not None]
        q = q.filter(or_(*crits))
    elif flag == 'clean':
        q = q.filter(BankImage.quality_state == 'ok')
        # Every quality flag except 'unreadable' (that one IS the quality_state
        # already pinned to 'ok' above). Each criterion is NULL-safe, so a row
        # from a build that predates one of these scores still counts as clean
        # for it instead of dropping out of the chip entirely.
        for f in ('blur', 'noise', 'uniform', 'small', 'soft_detail', 'bars'):
            q = q.filter(~_flag_filter(f, th))
    elif flag == 'dups':
        q = q.filter(BankImage.dup_group.isnot(None))
        order = (BankImage.dup_group.asc(), BankImage.id.asc())
    elif flag == 'semantic_dups':
        q = q.filter(BankImage.semantic_dup_group.isnot(None))
        order = (BankImage.semantic_dup_group.asc(), BankImage.id.asc())
    elif flag == 'no_face':
        # Literally "no face was found" — ONLY face_state == 'no_face'. The other
        # non-scorable states (low_det / too_small / extreme_pose) DID detect a
        # face; lumping them in here surfaced photos with visible faces under a
        # "No face" chip. 'unreadable'/'error' are read failures, not "no face".
        q = q.filter(BankImage.face_state == 'no_face')
    elif flag in _QUALITY_FLAGS:
        crit = _flag_filter(flag, th)
        if crit is not None:
            q = q.filter(crit)
        order = {'blur': BankImage.blur_score.asc(),
                 'noise': BankImage.noise_score.desc(),
                 'uniform': BankImage.uniformity_score.asc(),
                 'small': BankImage.width.asc(),
                 'soft_detail': BankImage.detail_ratio.asc(),
                 'bars': BankImage.bars_ratio.desc(),
                 'unreadable': BankImage.id.asc()}[flag]
    elif flag in _SCORE_FLAGS:
        crit = _flag_filter(flag, th)
        if crit is not None:
            q = q.filter(crit)
        # Worst first: least aesthetic / most-confident NSFW at the top.
        order = {'low_aesthetic': BankImage.aesthetic_score.asc(),
                 'nsfw': BankImage.nsfw_score.desc(),
                 'watermark': BankImage.id.asc()}[flag]
    if cluster is not None:
        q = q.filter(BankImage.face_cluster == int(cluster))
    if group is not None:
        q = q.filter(BankImage.dup_group == int(group))
    if semantic_group is not None:
        q = q.filter(BankImage.semantic_dup_group == int(semantic_group))
    if style is not None:
        q = q.filter(BankImage.style_cluster == int(style))
    if framing in _FRAMING_KEYS:
        # One framing bucket (face/bust/body/back/unknown) — composes with every
        # other facet. An unknown/absent value simply doesn't filter.
        q = q.filter(BankImage.framing == framing)
    if origin in ORIGINS:
        # One provenance state. 'unknown' is a real, selectable answer — it is
        # what a stripped file honestly is, and the user must be able to see that
        # pile rather than have it silently merged into "not AI".
        q = q.filter(BankImage.origin == origin)
    if subfolder is not None:
        # '' scopes to root-level files; any other value to that top-level folder
        # and everything nested under it. startswith() escapes LIKE metachars.
        if subfolder == '':
            q = q.filter(~BankImage.relpath.contains(os.sep))
        else:
            q = q.filter(BankImage.relpath.startswith(subfolder + os.sep))
    term = (search or '').strip()
    if term:
        # Full-text over caption + relpath. Escape LIKE metacharacters so a literal
        # '%'/'_' in the query matches itself, then wrap in wildcards.
        esc = term.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        like = f'%{esc}%'
        q = q.filter(or_(BankImage.caption.ilike(like, escape='\\'),
                         BankImage.relpath.ilike(like, escape='\\')))
    if res_bucket in _RES_BOUNDS:
        # One resolution tier: [lo, hi) on megapixels (width×height). The NOT-NULL
        # guards drop unscanned rows (a NULL product would satisfy neither bound
        # cleanly), so a tier never leaks unscanned images. Composes with the sort.
        lo, hi = _RES_BOUNDS[res_bucket]
        area = BankImage.width * BankImage.height
        q = q.filter(BankImage.width.isnot(None), BankImage.height.isnot(None))
        if lo:
            q = q.filter(area >= lo)
        if hi is not None:
            q = q.filter(area < hi)
    explicit = _sort_order(sort)
    if explicit is not None:
        # An explicit sort (resolution / aesthetic / sharpness) wins over the flag
        # worst-first order; see _sort_order for the NULL-sinks-last contract.
        order = explicit
    total = q.count()
    order_by = order if isinstance(order, tuple) else (order,)
    rows = q.order_by(*order_by).offset(max(0, int(offset))) \
            .limit(max(1, min(500, int(limit)))).all()
    return {'images': _page_images(rows, th), 'total': total,
            'offset': max(0, int(offset))}


# --- thumbnails -------------------------------------------------------------
def _thumb_path(bank_id, row: BankImage) -> Path:
    """Where this image's thumbnail lives. A watermark-cleaned or manually
    TURNED image gets its OWN thumbnail file, named after the cleaning method and
    the angle: the cached source thumbnail is never overwritten (so an undo — or
    a fourth quarter turn — instantly shows the original again) and the grid can
    never serve a stale pre-clean crop or a sideways tile. Deleting a cached
    thumbnail in place would be the fragile version of this — on Windows the file
    may still be held open by the response that just served it."""
    suffix = f'.{row.watermark_clean_method}' if row.watermark_clean_method else ''
    if getattr(row, 'rotation', None):
        suffix += f'.r{int(row.rotation)}'
    return _thumbs_dir(bank_id) / f'{row.id}{suffix}.webp'


def drop_derived(bank_id, image_id) -> None:
    """Best-effort removal of every DERIVED blob of one image — the cleaned and
    turned thumbnails plus the turned full-size copies (an undo, a re-scan or a
    new clean just invalidated them). The pristine `<id>.webp` thumbnail of the
    untouched source is deliberately kept. A leftover is harmless — nothing
    points at it once the row's state moved on — so a locked file is not an
    error."""
    for pattern, folder in ((f'{image_id}.*.webp', _thumbs_dir(bank_id)),
                            (f'{image_id}.r*', _rotated_dir(bank_id))):
        try:
            for stale in folder.glob(pattern):
                try:
                    stale.unlink()
                except OSError:
                    pass
        except OSError:
            pass


#: Historical alias — external callers/tests may still name the narrow version.
drop_clean_thumbs = drop_derived


def ensure_thumb(bank: ImageBank, row: BankImage) -> Path | None:
    """The image's grid thumbnail, generated lazily when the scan hasn't made
    it yet (so the grid is browsable straight after inventory). Built from the
    RESOLVED path, so a cleaned image shows clean in the grid."""
    tpath = _thumb_path(bank.id, row)
    if tpath.is_file():
        return tpath
    src = resolved_image_path(bank, row)
    if not src or not os.path.isfile(src):
        return None
    try:
        tpath.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im.draft(None, (THUMB_MAX_SIDE * 2, THUMB_MAX_SIDE * 2))
            im = im.convert('RGB')
            im.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.LANCZOS)
            im.save(tpath, 'WEBP', quality=72)
        return tpath
    except (OSError, ValueError):
        return None


# --- quality scan (background) ----------------------------------------------
def _scan_one(src_root: str, thumbs: Path, item: tuple) -> dict:
    """Worker: decode ONE file, compute metrics + dHash + thumbnail. Pure
    filesystem/PIL — no DB access (the job thread owns the session)."""
    image_id, relpath = item
    path = os.path.join(src_root, relpath)
    out = {'id': image_id, 'quality_state': 'unreadable', 'width': None,
           'height': None, 'file_size': None, 'dhash': None, 'metrics': None,
           'provenance': None}
    try:
        out['file_size'] = os.path.getsize(path)
    except OSError:
        # ABSENT ≠ CORRUPT. A file that is simply not there says nothing about
        # the image — the folder moved, or its drive is unplugged. 'missing' is
        # an in-memory signal for the job loop only (it is never written to
        # quality_state, so the row stays unscanned and a later pass retries it).
        if not os.path.exists(path):
            out['quality_state'] = 'missing'
            return out
    try:
        with Image.open(path) as im:
            out['width'], out['height'] = im.size
            # JPEG fast path: decode at reduced scale — the metrics run on a
            # ≤1024 working copy anyway, and dHash (9×8) is resize-invariant.
            im.draft(None, (ANALYSIS_MAX_SIDE * 2, ANALYSIS_MAX_SIDE * 2))
            im.load()
            out['metrics'] = quality_metrics(im)
            # Provenance rides along on the SAME decode — re-opening the file for
            # it would double the I/O of a 36 000-image pass for nothing. It reads
            # the drafted image (native pixels up to ANALYSIS_MAX_SIDE*2), which is
            # what the effective-resolution measure needs: it crops, never resizes.
            out['provenance'] = provenance_metrics(im)
            out['dhash'] = f'{_dhash(im):016x}'
            tpath = thumbs / f'{image_id}.webp'
            if not tpath.is_file():
                t = im.convert('RGB')
                t.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.LANCZOS)
                t.save(tpath, 'WEBP', quality=72)
        out['quality_state'] = 'ok'
    except (OSError, ValueError, SyntaxError):
        pass  # stays 'unreadable' — surfaced as a flag, never fatal
    return out


def start_scan(app, user_id, bank_id, rescan=False):
    """Launch the quality pass. Raises BankJobBusy when a job is already live,
    ValueError when the bank is unknown."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    total = _scan_pool(bank_id, rescan).count()
    return bank_jobs.start(app, bank_id, 'scan',
                           _scan_job(bank_id, rescan), total=total)


def _scan_pool(bank_id, rescan):
    """What the quality pass has to look at. Rejected images are OUT, like every
    other pass: on a 30 000-image bank two thirds of a rescan went to shots the
    user had already thrown away.

    Skipping them cannot swallow a FIRST scan: an image is only rejected by hand
    (after it was scanned) or by this very pass when it turns out unreadable, so
    a never-scanned image is still pending here. Un-reject one and it comes back
    into the pool on its own — that is why the filter is `!= reject` rather than
    an explicit pending/keep list."""
    q = (BankImage.query.filter_by(bank_id=bank_id)
         .filter(BankImage.status != 'reject'))
    if not rescan:
        # Never-scanned rows, PLUS rows a previous build scanned before the
        # provenance signals existed. Retrofitting the bank the user already has
        # is the point: telling them "only images scanned from now on get an
        # effective resolution" would leave a 36 000-image bank permanently half
        # measured, with no way to fix it short of a full rescan of everything.
        # `origin` is the sentinel because it is the one signal that always lands
        # on a readable file (one of ai/camera/unknown, never NULL) — keying off
        # detail_ratio would re-pick flat images forever, since those legitimately
        # measure nothing.
        q = q.filter(or_(BankImage.quality_state.is_(None),
                         and_(BankImage.quality_state == 'ok',
                              BankImage.origin.is_(None))))
    return q


def _scan_job(bank_id, rescan):
    def run(job):
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        items = [(r.id, r.relpath) for r in
                 _scan_pool(bank_id, rescan).order_by(BankImage.id.asc()).all()]
        bank_jobs.progress(job, done=0, total=len(items), detail='quality scan')
        thumbs = _thumbs_dir(bank_id)
        thumbs.mkdir(parents=True, exist_ok=True)
        src_root = bank.source_path
        if items and not os.path.isdir(src_root or ''):
            bank_jobs.fail(job, MOVED_FOLDER_MSG)
            return
        workers = min(8, os.cpu_count() or 4)
        done = 0
        missing = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            it = iter(items)
            futures = deque()

            def submit_next():
                nxt = next(it, None)
                if nxt is not None:
                    futures.append(ex.submit(_scan_one, src_root, thumbs, nxt))

            for _ in range(workers * 2):
                submit_next()
            while futures:
                res = futures.popleft().result()
                if res['quality_state'] == 'missing':
                    # Leave the row EXACTLY as it was (unscanned, undecided) and
                    # count it. Grading an absent file would auto-reject it, and
                    # a folder that moved makes every file absent at once — that
                    # path silently rejects a whole bank, so it must not exist.
                    missing += 1
                    done += 1
                    bank_jobs.bump(job)
                    if missing >= _MISSING_ABORT_AT and missing * 2 >= done:
                        db.session.commit()
                        bank_jobs.fail(job, MOVED_FOLDER_MSG)
                        return
                    if not bank_jobs.cancelled(job):
                        submit_next()
                    continue
                row = db.session.get(BankImage, res['id'])
                if row is not None:
                    row.quality_state = res['quality_state']
                    row.width, row.height = res['width'], res['height']
                    if res['file_size'] is not None:
                        row.file_size = res['file_size']
                    row.dhash = res['dhash']
                    if res['metrics']:
                        row.blur_score = res['metrics']['blur_score']
                        row.noise_score = res['metrics']['noise_score']
                        row.uniformity_score = res['metrics']['uniformity_score']
                    if res['provenance']:
                        p = res['provenance']
                        row.detail_ratio = p['detail_ratio']
                        row.bars_ratio = p['bars_ratio']
                        row.jpeg_quality = p['jpeg_quality']
                        row.origin = p['origin']
                        row.origin_evidence = p['origin_evidence']
                    # An unreadable file can never be promoted — auto-reject it
                    # (only over 'pending': a manual decision is never flipped).
                    if res['quality_state'] == 'unreadable' and row.status == 'pending':
                        row.status, row.reject_reason = 'reject', 'unreadable'
                done += 1
                if done % _COMMIT_EVERY == 0:
                    db.session.commit()
                bank_jobs.bump(job)
                if not bank_jobs.cancelled(job):
                    submit_next()
        db.session.commit()
        if not bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail='grouping duplicates')
            groups = rebuild_dup_groups(bank_id)
            tail = (f' — {missing} file(s) were not on disk and were left '
                    'untouched') if missing else ''
            bank_jobs.progress(
                job, detail=f'done — {groups} duplicate group(s){tail}')
    return run


# --- duplicate groups -------------------------------------------------------
def rebuild_dup_groups(bank_id, max_distance=None) -> int:
    """Recompute near-duplicate groups over every hashed image of the bank.
    Banded prefilter (pigeonhole: two hashes within Hamming d share at least
    one of d+1 equal bands) keeps this out of the full O(n²) — then candidate
    pairs are verified exactly and grouped by union-find. Groups of ≥2 get a
    1-based id ordered by size (biggest first). Returns the group count."""
    th = thresholds()
    d = int(th['dup_distance'] if max_distance is None else max_distance)
    rows = (db.session.query(BankImage.id, BankImage.dhash)
            .filter(BankImage.bank_id == bank_id, BankImage.dhash.isnot(None))
            .order_by(BankImage.id.asc()).all())
    ids = [r[0] for r in rows]
    hashes = [int(r[1], 16) for r in rows]
    n = len(ids)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    bands = max(1, min(16, d + 1))
    band_bits = 64 // bands
    buckets: dict = {}
    for i, h in enumerate(hashes):
        for b in range(bands):
            key = (b, (h >> (b * band_bits)) & ((1 << band_bits) - 1))
            buckets.setdefault(key, []).append(i)
    seen_pairs = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                if find(a) == find(b) or (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                if _hamming(hashes[a], hashes[b]) <= d:
                    union(a, b)
    comps: dict = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    groups = sorted((m for m in comps.values() if len(m) >= 2),
                    key=lambda m: (-len(m), m[0]))
    BankImage.query.filter_by(bank_id=bank_id).update(
        {'dup_group': None}, synchronize_session=False)
    for gid, members in enumerate(groups, start=1):
        member_ids = [ids[i] for i in members]
        for i0 in range(0, len(member_ids), _SQL_IN_CHUNK):
            BankImage.query.filter(
                BankImage.id.in_(member_ids[i0:i0 + _SQL_IN_CHUNK])).update(
                {'dup_group': gid}, synchronize_session=False)
    db.session.commit()
    return len(groups)


# --- semantic near-duplicate groups (stage 2 — crops / re-compressed variants) --
# One-entry memo for the parsed score cache. Reading it is 350 ms on a 14 700-row
# bank (40 MB .npz + a stat per row), and a user tuning a curation slider clicks
# three or four times on the SAME unchanged cache — that was 350 ms paid over and
# over for a file nobody touched. Bounded on purpose:
#   • ONE bank at a time (~45 MB of float32 at 14 700 × 768 — switching banks frees
#     the previous one rather than accumulating);
#   • keyed on the .npz's own (size, mtime_ns), so a finished ✨ Score pass — which
#     rewrites the file — invalidates it without anyone having to remember to;
#   • and it expires anyway after _SCORE_MEMO_TTL, because the per-row staleness
#     stats it skips are how a since-edited IMAGE gets dropped. A short window
#     covers the double-click; a session-long one would hide a real edit.
_SCORE_MEMO_TTL = 60.0
_score_memo = None            # (key, at, {path: emb}) — see reset_score_memo()


def reset_score_memo() -> None:
    """Drop the parsed-score-cache memo (tests; bank deletion)."""
    global _score_memo
    _score_memo = None


def _load_score_embeddings(bank: ImageBank) -> dict:
    """{abs_path: emb (np.float32, L2-normed)} from the ✨ Score pass cache, for the
    scored 'ok' images whose file still matches what was scored. Empty when the pass
    never ran (no cache) — the caller then surfaces the "run Score first" hint. A
    STALE entry (a same-path edit since scoring, detected via the cached size+mtime
    signature) is dropped, so a semantic group is never built on an outdated
    embedding. Reads the .npz directly (numpy is in the Flask venv); torch/open_clip
    are NOT needed here — stage 2 costs no new GPU work, it reuses Score's output."""
    global _score_memo
    import numpy as np
    path = _score_cache_path(bank.id)
    if not path.is_file():
        return {}
    try:
        st = path.stat()
        key = (bank.id, str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        key = None
    if key is not None and _score_memo is not None:
        mkey, at, cached = _score_memo
        if mkey == key and (time.time() - at) < _SCORE_MEMO_TTL:
            return cached
    try:
        with np.load(str(path), allow_pickle=False) as z:
            paths = [str(p) for p in z['paths']]
            states = [str(s) for s in z['states']]
            embs = z['embs']
            sigs = ([str(s) for s in z['sigs']] if 'sigs' in z.files
                    else [''] * len(paths))
    except Exception as e:  # noqa: BLE001 — a corrupt cache = "no embeddings", never fatal
        logger.warning('bank %s score cache unreadable: %s', bank.id, e)
        return {}
    out = {}
    for i, p in enumerate(paths):
        if states[i] != 'ok':
            continue
        emb = embs[i]
        if float(np.abs(emb).sum()) <= 0:       # zero sentinel = errored image
            continue
        sig = sigs[i]
        if sig:                                 # drop a since-edited file
            try:
                st = os.stat(p)
                if f'{st.st_size}:{st.st_mtime_ns}' != sig:
                    continue
            except OSError:
                continue
        out[p] = np.asarray(emb, dtype='float32')
    if key is not None:
        _score_memo = (key, time.time(), out)
    return out


def rebuild_semantic_dup_groups(bank_id, threshold=None) -> int | None:
    """Stage-2 near-duplicate grouping over the CLIP embeddings the ✨ Score pass
    cached — catches crops and re-compressed variants of the SAME shot that the
    dHash (stage 1) misses. Returns the group count (groups of ≥2), or None when NO
    embeddings are available (Score hasn't run) so the caller shows the "run Score
    first" hint instead of a silent empty result.

    Cost: a semantic near-dup (cosine ≥ threshold) is necessarily inside one style
    union-find component (that clustering uses style_threshold ≤ threshold), so we
    BLOCK by style_cluster and only compare within a block — Σ block² dot-products,
    not the full n². A config with style_threshold > threshold would break that
    guarantee, so we fall back to a single global block then. Re-running at another
    threshold is CPU-only and near-instant: it re-reads the cached embeddings — no
    GPU, no re-scan."""
    import numpy as np
    bank = db.session.get(ImageBank, bank_id)
    if not bank:
        return None
    emb_by_path = _load_score_embeddings(bank)
    if not emb_by_path:
        return None
    th = thresholds()
    t = float(threshold if threshold is not None else th['semantic_dup_threshold'])
    block_by_style = th['style_threshold'] <= t
    rows = (BankImage.query.filter_by(bank_id=bank_id)
            .order_by(BankImage.id.asc()).all())
    items = []      # (image_id, block_key, emb)
    for r in rows:
        p = abs_image_path(bank, r)
        emb = emb_by_path.get(p) if p else None
        if emb is None:
            continue
        block = (r.style_cluster if r.style_cluster is not None else -1) \
            if block_by_style else 0
        items.append((r.id, block, emb))
    # A re-run fully recomputes — clear every semantic group first.
    BankImage.query.filter_by(bank_id=bank_id).update(
        {'semantic_dup_group': None}, synchronize_session=False)
    if not items:
        db.session.commit()
        return 0
    blocks: dict = {}
    for idx, (_id, block, _emb) in enumerate(items):
        blocks.setdefault(block, []).append(idx)
    parent = list(range(len(items)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    chunk = 512
    for members in blocks.values():
        if len(members) < 2:
            continue
        E = np.stack([items[i][2] for i in members]).astype('float32')
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        m = len(members)
        for i0 in range(0, m, chunk):
            sims = E[i0:i0 + chunk] @ E.T
            for a, b in np.argwhere(sims >= t):
                a += i0
                if a < b:                       # skip the diagonal + mirror pairs
                    union(members[int(a)], members[int(b)])
    comps: dict = {}
    for i in range(len(items)):
        comps.setdefault(find(i), []).append(i)
    groups = sorted((m for m in comps.values() if len(m) >= 2),
                    key=lambda m: (-len(m), items[m[0]][0]))
    for gid, members in enumerate(groups, start=1):
        member_ids = [items[i][0] for i in members]
        for i0 in range(0, len(member_ids), _SQL_IN_CHUNK):
            BankImage.query.filter(
                BankImage.id.in_(member_ids[i0:i0 + _SQL_IN_CHUNK])).update(
                {'semantic_dup_group': gid}, synchronize_session=False)
    db.session.commit()
    return len(groups)


def start_semantic_dedup(app, user_id, bank_id, threshold=None):
    """Launch the stage-2 semantic near-duplicate pass (CPU, reuses the ✨ Score
    embeddings — no GPU). ValueError (→400) when Score hasn't produced any usable
    embedding yet, so the UI shows the clear "run Score first" hint rather than a
    job that quietly does nothing."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if not _load_score_embeddings(bank):
        raise ValueError('run ✨ Score first — semantic near-duplicates reuse its '
                         'embeddings')
    return bank_jobs.start(app, bank_id, 'semantic_dedup',
                           _semantic_dedup_job(bank_id, threshold), total=0)


def _semantic_dedup_job(bank_id, threshold):
    def run(job):
        bank_jobs.progress(job, done=0, total=0, detail='finding crops & variants')
        n = rebuild_semantic_dup_groups(bank_id, threshold)
        if n is None:
            bank_jobs.progress(job, detail='no embeddings — run ✨ Score first')
            return
        bank_jobs.progress(job, detail=f'done — {n} semantic near-duplicate group(s)')
    return run


def dup_groups_payload(user_id, bank_id, offset=0, limit=50,
                       col=BankImage.dup_group) -> dict | None:
    """Unresolved groups (≥2 non-rejected members) with their full membership,
    for the resolution panel. ``col`` picks the stage: dup_group (exact/resized) or
    semantic_dup_group (crops/variants)."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    gids = [g for (g,) in _unresolved_dup_groups_q(bank_id, col)
            .order_by(col.asc()).all()]
    total = len(gids)
    page = gids[max(0, int(offset)):max(0, int(offset)) + max(1, min(200, int(limit)))]
    groups = []
    for gid in page:
        rows = (BankImage.query.filter(BankImage.bank_id == bank_id, col == gid)
                .order_by(BankImage.id.asc()).all())
        groups.append({'group': gid,
                       'best_id': _best_of(rows).id if rows else None,
                       'images': _page_images(rows, th)})
    return {'groups': groups, 'total': total, 'offset': max(0, int(offset))}


def semantic_dup_groups_payload(user_id, bank_id, offset=0, limit=50) -> dict | None:
    """dup_groups_payload for stage 2 (semantic_dup_group)."""
    return dup_groups_payload(user_id, bank_id, offset=offset, limit=limit,
                              col=BankImage.semantic_dup_group)


def _best_of(rows):
    """'Keep best' heuristic for a duplicate group. When the aesthetic pass has
    run it leads (the ask: keep the NICE copy, not merely the biggest); a
    scored image always outranks an unscored one (sentinel -1 < the ~1..10 range).
    Then most pixels, sharpest, heaviest file — a Telegram dump's duplicates are
    mostly re-compressed or downscaled copies, so surface area is the honest
    fallback key."""
    def key(r):
        return (r.aesthetic_score if r.aesthetic_score is not None else -1.0,
                (r.width or 0) * (r.height or 0), r.blur_score or 0.0,
                r.file_size or 0, -r.id)
    return max(rows, key=key)


def resolve_dups(user_id, bank_id, strategy='best', group=None, keep_ids=None,
                 col=BankImage.dup_group, attr='dup_group', reason='duplicate',
                 respect_existing_keep=True, snapshot=None):
    """Resolve duplicate groups: keep one member, REJECT the others (a status,
    never a file deletion, so it's reversible). strategy 'best'|'first' applies to
    one group or, when ``group`` is None, to every unresolved group at once;
    explicit ``keep_ids`` (manual pick) applies to their own groups. Only
    non-rejected members are touched. ``col``/``attr``/``reason`` pick the stage:
    dup_group (exact/'duplicate') or semantic_dup_group (crops/'semantic_dup').

    ``respect_existing_keep`` (default True) protects members the user already
    KEPT from a bulk resolve — right for the AUTOMATIC pipeline auto-reject, so a
    mass resolve never un-keeps a manual pick. An EXPLICIT resolve the user fired
    from a dup/same-shot group passes False: the whole point is to collapse the
    group to ONE, and the members of a same-shot group are typically ALL 'keep',
    so respecting keep would reject nobody. The elected keeper is always safe
    (``r.id in keep``); with False every OTHER member falls to reject, keep
    included. Returns {'resolved': groups, 'rejected': images}.

    ``snapshot``: pass a live :class:`bank_undo.Snapshot` to fold this resolve
    into a WIDER undo step (the pipeline's auto-reject is a flag pass plus this
    one); omit it and the call publishes its own one-step undo offer."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    own_snapshot = snapshot is None
    if own_snapshot:
        snapshot = bank_undo.Snapshot(
            'Resolve same-shot groups' if reason == 'semantic_dup'
            else 'Resolve duplicate groups')
    keep_by_group = {}
    if keep_ids:
        rows = BankImage.query.filter(BankImage.bank_id == bank_id,
                                      BankImage.id.in_(list(keep_ids)[:_SQL_IN_CHUNK])).all()
        for r in rows:
            g = getattr(r, attr)
            if g:
                keep_by_group.setdefault(g, set()).add(r.id)
        gids = list(keep_by_group)
    elif group is not None:
        gids = [int(group)]
    else:
        gids = [g for (g,) in _unresolved_dup_groups_q(bank_id, col).all()]
    resolved = rejected = 0
    for gid in gids:
        rows = (BankImage.query.filter(BankImage.bank_id == bank_id, col == gid)
                .filter(BankImage.status != 'reject')
                .order_by(BankImage.id.asc()).all())
        if len(rows) < 2 and gid not in keep_by_group:
            continue
        if gid in keep_by_group:
            keep = keep_by_group[gid]
        elif strategy == 'first':
            keep = {rows[0].id}
        else:
            keep = {_best_of(rows).id}
        changed = False
        for r in rows:
            if r.id in keep or (respect_existing_keep and r.status == 'keep'):
                continue
            snapshot.note(r, 'reject', reason)
            r.status, r.reject_reason = 'reject', reason
            rejected += 1
            changed = True
        if changed or len(rows) >= 2:
            resolved += 1
    db.session.commit()
    if own_snapshot:
        snapshot.commit(bank_id)
    return {'resolved': resolved, 'rejected': rejected}


def resolve_semantic_dups(user_id, bank_id, strategy='best', group=None,
                          keep_ids=None, respect_existing_keep=True,
                          snapshot=None):
    """resolve_dups for stage 2 (semantic_dup_group, reject reason
    'semantic_dup')."""
    return resolve_dups(user_id, bank_id, strategy=strategy, group=group,
                        keep_ids=keep_ids, col=BankImage.semantic_dup_group,
                        attr='semantic_dup_group', reason='semantic_dup',
                        respect_existing_keep=respect_existing_keep,
                        snapshot=snapshot)


# --- statuses & flag application --------------------------------------------
_STATUS_UNDO_LABEL = {'keep': 'Keep images', 'reject': 'Reject images',
                      'pending': 'Set images back to undecided'}


def set_status(user_id, bank_id, ids, status) -> int:
    """Manual keep/reject/pending on a selection. Returns rows changed.

    Snapshots the prior (status, reason) of every row it actually flips, so the
    workspace can offer ONE step back — this is the gesture that puts hundreds of
    decisions in flight at once (select the whole filter, then ✕)."""
    if status not in ('pending', 'keep', 'reject'):
        raise ValueError('bad status')
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    ids = [int(i) for i in (ids or [])]
    reason = 'manual' if status == 'reject' else None
    snapshot = bank_undo.Snapshot(_STATUS_UNDO_LABEL[status])
    n = 0
    for i0 in range(0, len(ids), _SQL_IN_CHUNK):
        rows = BankImage.query.filter(
            BankImage.bank_id == bank_id,
            BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all()
        for r in rows:
            snapshot.note(r, status, reason)
            r.status = status
            r.reject_reason = reason
            n += 1
    db.session.commit()
    snapshot.commit(bank_id)
    return n


def rotate_images(user_id, bank_id, ids, delta) -> dict:
    """Turn a selection by ``delta`` degrees CLOCKWISE (idea by 1Tomber, #17).

    The user's files are NEVER written to: the new angle is stored on the row and
    the derived blobs (turned copy + thumbnail) are dropped so the ONE resolver
    rebuilds them from the pristine source on the next read. That is what makes
    the turn free of loss where it counts — a fourth quarter turn puts the row
    back at 0 and every reader is served the original bytes again, byte for byte.

    ``delta`` may be negative (-90 = turn left). Returns {'rotated': n,
    'rotations': {image_id: angle}} so the grid can re-render without a refetch.
    """
    from .face_dataset_service import normalize_rotation
    step = normalize_rotation(delta)
    if step == 0:
        raise ValueError('rotation must be 90, 180 or 270 degrees')
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    ids = [int(i) for i in (ids or [])]
    if not ids:
        raise ValueError('select at least one image')
    rotations = {}
    for i0 in range(0, len(ids), _SQL_IN_CHUNK):
        rows = BankImage.query.filter(
            BankImage.bank_id == bank_id,
            BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all()
        for r in rows:
            r.rotation = (int(r.rotation or 0) + step) % 360 or None
            drop_derived(bank_id, r.id)
            rotations[r.id] = int(r.rotation or 0)
    db.session.commit()
    return {'rotated': len(rotations), 'rotations': rotations}


def apply_flags(user_id, bank_id, flags, snapshot=None) -> dict:
    """Bulk-reject the PENDING images carrying the given flags. Manual ✓/✕
    decisions are never flipped (only status='pending' is touched) — same
    contract as the dataset auto-triage. Returns per-flag reject counts.

    This is the mis-set-threshold accident in one click, so it snapshots what it
    flipped. ``snapshot``: pass a live :class:`bank_undo.Snapshot` to fold the
    pass into a wider undo step (the pipeline does); omit it and the call
    publishes its own offer."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    own_snapshot = snapshot is None
    if own_snapshot:
        snapshot = bank_undo.Snapshot('Auto-reject by flag')
    th = thresholds()
    out = {}
    for flag in flags or []:
        if flag not in _QUALITY_FLAGS + _SCORE_FLAGS:
            continue
        crit = _flag_filter(flag, th)
        if crit is None:
            continue
        rows = (BankImage.query.filter_by(bank_id=bank_id, status='pending')
                .filter(crit).all())
        for r in rows:
            snapshot.note(r, 'reject', flag)
            r.status, r.reject_reason = 'reject', flag
        out[flag] = len(rows)
    db.session.commit()
    if own_snapshot:
        snapshot.commit(bank_id)
    return out


# --- ↩ undo the last bulk decision ------------------------------------------
_UNDO_NAME_SAMPLE = 8        # conflicting files quoted back so the user can find them


def undo_offer(user_id, bank_id) -> dict | None:
    """{label, count, at} for the workspace's ↩ bar, or None. Rides in the bank
    payload the workspace already polls, which is what makes the offer survive a
    reload — the decision it takes back lives in the database, not in a tab."""
    if not get_bank(user_id, bank_id):
        return None
    return bank_undo.peek(bank_id)


def undo_last(user_id, bank_id) -> dict:
    """Put every row the last bulk decision changed back to what it was.

    Three outcomes per row, all counted, because a restore that quietly missed
    half of its rows would be worse than no undo at all:

    * **restored** — the row is still there and still carries what the action
      set, so it goes back to its recorded prior value (status AND reason: the
      flag counters read the reason column);
    * **missing** — the row left the bank since (a re-scan dropped a file that
      disappeared from the folder);
    * **conflict** — someone changed it since (another tab, ▶ Review, a later
      pass). It is LEFT ALONE and named: overwriting a newer decision with an
      older one is not "undo", it is a second accident.

    Rows the action never touched are untouched here by construction — only the
    snapshot's ids are read. Synchronous even on a big lot: the work is one
    indexed SELECT plus in-place writes over the ids we already know, so a
    5 000-image restore lands in well under a second — a progress bar would take
    longer to render than the job it reports on.
    """
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if bank_jobs.running(bank_id):
        raise RuntimeError('a pass is running on this bank — stop it first')
    snap = bank_undo.take(bank_id)
    if not snap or not snap['rows']:
        raise ValueError('nothing to undo')

    entries = snap['rows']
    ids = list(entries)
    seen = set()
    restored = conflicts = 0
    conflict_names = []
    for i0 in range(0, len(ids), _SQL_IN_CHUNK):
        rows = BankImage.query.filter(
            BankImage.bank_id == bank_id,
            BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all()
        for r in rows:
            seen.add(r.id)
            entry = entries[r.id]
            if (r.status, r.reject_reason) != tuple(entry['after']):
                conflicts += 1
                if len(conflict_names) < _UNDO_NAME_SAMPLE:
                    conflict_names.append(os.path.basename(r.relpath))
                continue
            r.status, r.reject_reason = entry['before']
            restored += 1
    db.session.commit()
    return {'label': snap['label'], 'total': len(ids), 'restored': restored,
            'missing': len(ids) - len(seen), 'conflicts': conflicts,
            'conflict_names': conflict_names}


# --- curation selectors (diversity · reference similarity) ------------------
# Both reuse the CLIP embeddings the ✨ Score pass already cached — no GPU, no
# re-scan (same contract as the semantic-dedup stage). They only ever build a
# SELECTION (a set of image ids the UI checks); the user reviews it before any
# Keep / Reject / Promote — nothing is mutated or deleted here.
_CURATION_MAX_N = 2000       # a curated LoRA set is 20–200 images; this is generous


def _pool_query(bank_id, th, *, status=None, flag=None, cluster=None,
                style=None, subfolder=None, search=None):
    """The candidate-pool query for the curation selectors — the SAME filter
    composition as list_images (status ∩ flag ∩ cluster ∩ style ∩ subfolder ∩
    search), minus the ordering/pagination, so "give me 60 diverse images" is
    composable with whatever the grid is currently showing.

    Kept as its own function (a small, deliberate mirror of the list_images WHERE
    clauses) rather than a shared refactor: three curation-related branches touch
    this file in parallel, so an additive helper rebases clean where an edit to
    the list_images hot path would collide. When NO status is chosen the reject
    pile is excluded — you curate from what you might keep, never from the bin."""
    q = BankImage.query.filter_by(bank_id=bank_id)
    if status in ('pending', 'keep', 'reject'):
        q = q.filter(BankImage.status == status)
    else:
        q = q.filter(BankImage.status != 'reject')
    if flag == 'flagged':
        crits = [c for c in (_flag_filter(f, th) for f in _QUALITY_FLAGS)
                 if c is not None]
        q = q.filter(or_(*crits))
    elif flag == 'clean':
        q = q.filter(BankImage.quality_state == 'ok')
        # Every quality flag except 'unreadable' (that one IS the quality_state
        # already pinned to 'ok' above). Each criterion is NULL-safe, so a row
        # from a build that predates one of these scores still counts as clean
        # for it instead of dropping out of the chip entirely.
        for f in ('blur', 'noise', 'uniform', 'small', 'soft_detail', 'bars'):
            q = q.filter(~_flag_filter(f, th))
    elif flag == 'dups':
        q = q.filter(BankImage.dup_group.isnot(None))
    elif flag == 'semantic_dups':
        q = q.filter(BankImage.semantic_dup_group.isnot(None))
    elif flag == 'no_face':
        q = q.filter(BankImage.face_state == 'no_face')
    elif flag in _QUALITY_FLAGS + _SCORE_FLAGS:
        crit = _flag_filter(flag, th)
        if crit is not None:
            q = q.filter(crit)
    if cluster is not None:
        q = q.filter(BankImage.face_cluster == int(cluster))
    if style is not None:
        q = q.filter(BankImage.style_cluster == int(style))
    if subfolder is not None:
        if subfolder == '':
            q = q.filter(~BankImage.relpath.contains(os.sep))
        else:
            q = q.filter(BankImage.relpath.startswith(subfolder + os.sep))
    term = (search or '').strip()
    if term:
        esc = term.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        like = f'%{esc}%'
        q = q.filter(or_(BankImage.caption.ilike(like, escape='\\'),
                         BankImage.relpath.ilike(like, escape='\\')))
    return q


def _pool_embeddings(bank, emb_by_path, filters):
    """(ids, E) for the filtered pool rows that HAVE a cached embedding: ids is a
    list ordered by image id (so every tie-break below is deterministic), E is the
    matching (m×d) float32 matrix, L2-normalised. Empty ids ⇒ E is None."""
    import numpy as np
    rows = (_pool_query(bank.id, thresholds(), **filters)
            .order_by(BankImage.id.asc()).all())
    base = os.path.realpath(bank.source_path)   # once, not once per row
    prefix = os.path.normcase(base + os.sep)
    ids, vecs = [], []
    for r in rows:
        # Fast path: the keys of emb_by_path were THEMSELVES produced by
        # _abs_under (the ✨ Score pass walks the same rows), so a lexical
        # normpath that HITS the dict is provably the very string realpath would
        # have returned — no filesystem call needed to know that. A miss is the
        # only case that can be a symlink/junction, and it falls through to the
        # real resolution, so the result set is identical either way. That matters
        # because realpath is a syscall and this loop runs once per pool image:
        # 756 ms of a 6 353-row pool, against 6 ms for the lexical form.
        p = os.path.normpath(os.path.join(base, r.relpath))
        emb = emb_by_path.get(p) if os.path.normcase(p).startswith(prefix) else None
        if emb is None:
            p = _abs_under(base, r.relpath)
            emb = emb_by_path.get(p) if p else None
        if emb is not None:
            ids.append(r.id)
            vecs.append(emb)
    if not ids:
        return ids, None
    E = np.stack(vecs).astype('float32')
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return ids, E


_TYPICALITY_DEFAULT = 0.5    # see select_diverse — 0 restores pure farthest-point
_TYPICALITY_K = 10           # neighbours whose mean similarity IS the local density
_TYPICALITY_BLOCK = 512      # rows per similarity block — NEVER a full (m×m) matrix
_TYPICALITY_Z = 3.0          # robust deviations below the median density = full penalty
_TYPICALITY_DECADES = 3.0    # novelty discount at full guard + full penalty: 10⁻³
_TYPICALITY_MIN_POOL = 32    # under this a median/MAD "tail" is noise — guard off

_BLAS_GEMM = ...             # unprobed sentinel; None once probed and unavailable


def _fast_gemm_nt():
    """scipy's single-precision GEMM (``A @ B.T``), or None when unreachable.

    WHY this exists, measured rather than assumed: the numpy wheels this app runs
    on ship WITHOUT an optimised BLAS (``threadpoolctl.threadpool_info()`` returns
    an empty list), so ``E @ E.T`` runs single-threaded at ~5 GFLOP/s. On a real
    6 353-image pool that is 12.6 s for ONE similarity pass — 89 % of a curation
    click. scipy's wheels bundle OpenBLAS (24 threads here): the identical product
    takes 0.14 s, ~90× less, for the same 62 GFLOP.

    scipy is not in ``requirements.txt``, and it does not need to be: this lane
    only runs when numpy is present, numpy only arrives with
    ``requirements-ml.txt``, and insightface (the first line of that file) depends
    on scipy. So every install that CAN reach this code has the fast path, and the
    numpy fallback below is the belt-and-braces branch, not the normal one.

    Probed once per process and remembered — importing scipy.linalg is ~200 ms."""
    global _BLAS_GEMM
    if _BLAS_GEMM is ...:
        try:
            from scipy.linalg.blas import sgemm
            _BLAS_GEMM = sgemm
        except Exception as e:  # noqa: BLE001 — no scipy / broken wheel = slow path
            logger.info('no scipy BLAS for curation sampling (%s); '
                        'falling back to numpy matmul', e)
            _BLAS_GEMM = None
    return _BLAS_GEMM


def _sim_block(A, E):
    """``A @ E.T`` for L2-normed float32 rows — the similarity block of the
    typicality pass, routed through an optimised BLAS when one is reachable.

    Returns a C-contiguous (len(A) × len(E)) float32 array: scipy hands back a
    Fortran-ordered result, and the ``np.partition(..., axis=1)`` that follows
    walks rows, so the copy pays for itself several times over.

    Float caveat, stated because it is the one thing this change can affect: a
    different BLAS sums the same products in a different order, so a similarity
    can differ from numpy's by ~1e-6 (measured max absolute deviation over the
    real bank). That is far below any threshold this module compares against, and
    the selections were verified id-for-id identical on the production bank across
    n ∈ {20, 60, 200} and typicality ∈ {0.25, 0.5, 1.0} — but it is an equality
    of results, not of bits. ``typicality=0`` never reaches here at all, so the
    golden "historical behaviour" path is untouched by construction."""
    import numpy as np
    gemm = _fast_gemm_nt()
    if gemm is not None and A.dtype == np.float32 and E.dtype == np.float32:
        try:
            return np.ascontiguousarray(gemm(1.0, A, E, trans_b=True))
        except Exception as e:  # noqa: BLE001 — never fail a click over an optimisation
            logger.warning('scipy BLAS gemm refused the pool (%s); using numpy', e)
    return A @ E.T


def _isolation_penalty(E, *, k=_TYPICALITY_K, block=_TYPICALITY_BLOCK):
    """How ALONE each row is, as (m,) floats in [0, 1] — 0 for anything at or above
    the pool's median local density, 1 for the genuinely isolated tail.

    Local density = mean cosine similarity to the ``k`` nearest OTHER rows (the
    same cached embeddings the whole curation lane uses). Turning that density
    into a penalty is done on a ROBUST scale — median and MAD, not min/max — for
    two reasons that matter here:

      • a single meme in 24 000 photos would own the whole min/max range and
        squash every real difference to nothing;
      • everything at or above the median density gets penalty exactly 0, so the
        normal population of the bank is left strictly untouched. This selector
        DISCOUNTS the isolated tail, it never REWARDS the centre — which is what
        keeps a typicality guard from quietly turning "the 60 most varied" into
        60 look-alikes from the middle of the cloud.

    The ramp is quadratic, so being *slightly* below the median density barely
    costs anything and only the real tail is hit hard: at 1 robust deviation the
    penalty is 0.11, at 2 it is 0.44, at 3 and beyond it saturates at 1.

    Memory: the similarity pass runs in row blocks, so peak allocation is
    (block × m) float32 (~49 MB at m=24 000, measured 148 MB peak including
    temporaries) instead of the ~2.3 GB a full ``E @ E.T`` would need.
    Deterministic — median/MAD/partition, no sampling, no RNG.

    Time: this is an exact all-pairs pass, Θ(m²·d) — the same shape of work the
    semantic-dedup stage already does, and the reason the guard is computed ONLY
    when it is on. It is also, by a wide margin, the most expensive thing a
    curation click does; ``_sim_block`` explains why the product goes through
    scipy's BLAS rather than numpy's (12.6 s → 0.14 s on a 6 353-image pool,
    measured — the numpy this app ships on has no optimised BLAS, so the "~50×
    slower" case that paragraph used to describe as a hazard WAS the normal one).
    An approximation (subsampling the reference set) was rejected on purpose: it
    would make a small but legitimate group — eight shots of one rare outfit —
    look isolated and get penalised, which is exactly the variety this selector
    exists to preserve."""
    import numpy as np
    m = int(E.shape[0])
    k = min(int(k), m - 1)
    if k < 1 or m < _TYPICALITY_MIN_POOL:
        # Too few rows for a median and a MAD to mean anything: a 12-image pool
        # has no "isolated tail", only 12 images. Staying out is the honest
        # answer AND keeps small banks on the historical behaviour.
        return np.zeros(m, dtype='float32')
    dens = np.empty(m, dtype='float32')
    for a in range(0, m, block):
        S = _sim_block(E[a:a + block], E)    # (b, m) block — never (m, m)
        rows = np.arange(S.shape[0])
        S[rows, rows + a] = -np.inf          # a row is not its own neighbour
        top = np.partition(S, m - k, axis=1)[:, m - k:]
        dens[a:a + block] = top.mean(axis=1)
    med = float(np.median(dens))
    mad = float(np.median(np.abs(dens - med)))
    scale = 1.4826 * mad                     # MAD → σ-comparable, robust
    if not (scale > 1e-6):                   # degenerate pool (all alike) ⇒ no tail
        return np.zeros(m, dtype='float32')
    z = (med - dens) / scale                 # >0 only BELOW the median density
    ramp = np.clip(z / _TYPICALITY_Z, 0.0, 1.0)
    return (ramp * ramp).astype('float32')


def _farthest_point(E, factor, n):
    """Greedy farthest-point sampling over the L2-normed rows of ``E`` (m×d),
    returning ``n`` ROW POSITIONS in pick order. Seeded on position 0 — callers
    pass rows in ascending id order, so the seed and every tie-break resolve to
    the lowest id and the result is deterministic.

    ``factor`` is the per-row novelty multiplier in (0, 1] (the typicality guard,
    see ``_isolation_penalty``) or None to sample on pure max-min distance. Shared
    by ``select_diverse`` (whole pool) and ``select_balanced`` (one call per
    bucket, on a slice of the same E and the same pool-wide factor) so the guard
    behaves identically in both — there is exactly one copy of this loop."""
    import numpy as np
    m = int(E.shape[0])
    if n <= 0 or m == 0:
        return []
    if n >= m:
        return list(range(m))
    # min_dist[i] = cosine distance from row i to the NEAREST chosen row so far.
    chosen = [0]                                 # seed = lowest id (E[0])
    min_dist = 1.0 - E @ E[0]
    min_dist[0] = -np.inf                        # never re-pick a chosen row
    for _ in range(n - 1):
        score = min_dist if factor is None else min_dist * factor
        nxt = int(np.argmax(score))              # ties → lowest index = lowest id
        if not np.isfinite(score[nxt]):          # pool exhausted (all chosen)
            break
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, 1.0 - E @ E[nxt])
        min_dist[nxt] = -np.inf
    return chosen


def select_diverse(user_id, bank_id, n=60, *, typicality=_TYPICALITY_DEFAULT,
                   filters=None):
    """Farthest-point sampling over the ✨ Score CLIP embeddings, tempered by a
    TYPICALITY guard: the ``n`` images of the (filtered) pool that best COVER the
    visual space — the antidote to a dump of 4 000 near-identical shots. Greedy
    FPS: seed with the lowest-id row (deterministic), then repeatedly add the
    point whose nearest already-chosen neighbour is FARTHEST (max-min cosine
    distance). The SAMPLING itself is O(n·m·d) — one (m×d)·(d,) product per pick,
    110 ms at m=6 353 / n=60 (measured).

    That figure used to be quoted as the cost of the WHOLE call ("~sub-second
    even at m=24 000"), and on real data it was wrong by a factor of thirty: the
    click took 32 s, of which the loop below was 0.1 s. The rest was the guard's
    all-pairs pass (see ``_isolation_penalty`` / ``_sim_block``). Whatever else
    changes here, keep this docstring measured — a comment promising a second
    where the user waits half a minute is a debt, not documentation.

    ``typicality`` (0–1) exists because pure max-min distance is, mathematically,
    the criterion that prefers ISOLATED points: on a collected bank the first
    picks are therefore structurally biased towards the aberrations (a meme, a
    photo of someone else, a botched frame) rather than towards variety of the
    subject. Each candidate's novelty is multiplied by
    ``10 ** (-3 × typicality × isolation)`` (see ``_isolation_penalty``), so an
    image still gets picked for the variety it adds, but being alone stops being
    a quality in itself. The discount is GEOMETRIC on purpose: an aberration's
    distance advantage over a normal image is a ratio (2–4× in practice, and it
    grows as the easy variety gets used up), so a merely linear penalty would need
    a near-maximal setting to ever bite.

      • 0    → EXACTLY the historical behaviour, pick for pick (the guard is not
               even computed);
      • 0.5  → the default: a saturated outlier keeps ~3% of its novelty (÷32) —
               decisively beaten by any genuinely varied shot — while a merely
               below-average image (penalty 0.11) keeps 68%;
      • 1    → the isolated tail is all but excluded (÷1000).

    The guard is bounded on BOTH sides: rows at or above the median density are
    never penalised at all (factor exactly 1.0), so it cannot collapse the
    selection into look-alikes from the middle of the cloud.

    Returns {'image_ids': [...] (sorted), 'pool': m, 'requested': n,
    'typicality': w}. Raises ValueError (→400, "run ✨ Score first") when no
    embedding exists yet, so the UI shows the clear hint instead of an empty,
    unexplained selection."""
    import numpy as np
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    emb_by_path = _load_score_embeddings(bank)
    if not emb_by_path:
        raise ValueError('run ✨ Score first — diversity sampling reuses its '
                         'embeddings')
    n = max(1, min(int(n), _CURATION_MAX_N))
    try:
        w = 0.0 if typicality is None else float(typicality)
    except (TypeError, ValueError):
        w = _TYPICALITY_DEFAULT
    w = max(0.0, min(1.0, w))
    ids, E = _pool_embeddings(bank, emb_by_path, filters or {})
    m = len(ids)
    if m <= n:                                   # whole pool already fits
        return {'image_ids': sorted(ids), 'pool': m, 'requested': n,
                'typicality': w}
    # Novelty multiplier, in (0, 1] — never 0, so the -inf "already chosen"
    # sentinel below stays -inf (0 × -inf would be a NaN) and so even a fully
    # penalised row stays a last resort rather than an unpickable one.
    factor = None
    if w > 0.0:
        factor = 10.0 ** (-_TYPICALITY_DECADES * w * _isolation_penalty(E))
    chosen = _farthest_point(E, factor, n)
    return {'image_ids': sorted(ids[i] for i in chosen),
            'pool': m, 'requested': n, 'typicality': w}


# --- balanced selection (coverage of the LABELS, not of the embedding space) --
# `select_diverse` answers "is my set VARIED?"; this one answers a different
# question no per-image score can ask: "does my set COVER what I want to be able
# to generate?". Asking for the 60 best/most varied of a bank that is 49% full
# body and 3.6% face shots returns those proportions — the LoRA then renders one
# framing well and the rest badly, with nothing having said so.
#
# WHICH AXIS. Measured on a real 43 000-image bank rather than assumed: `framing`
# had 13 000 rows classified across all four buckets (body 49%, bust 37%, back
# 11%, face 3.6%) — a discrete label with a real, actionable imbalance. Over the
# same bank `face_cluster` covered 4.7% of the rows and shattered them into 561
# clusters whose biggest held 34% — on a mono-subject bank a semantic/identity
# split is sparse and arbitrary, so balancing on it would spread a selection over
# noise. Hence: framing is the DEFAULT axis, and person is an explicit opt-in for
# the genuinely multi-subject dump.
_BALANCE_AXES = ('framing', 'framing+person')
_BALANCE_DEFAULT_AXIS = 'framing'   # stored in localStorage — never rename


def _balanced_quotas(sizes: dict, n: int) -> dict:
    """Split ``n`` picks as evenly as possible over the buckets, capped by what
    each one actually HAS (largest-remainder water-filling).

    A bucket that cannot serve its equal share is filled to the brim and its
    unused share is redistributed over the buckets that still have room — the
    ARBITRATION being: asking for 60 when a perfect split only yields 42 returns
    60, not 42, because throwing 18 usable images away buys a purity nobody asked
    for. What is forbidden is doing it SILENTLY: every caller gets ``fair_share``
    next to ``selected`` per bucket, so the top-up is visible as the deficit it
    is. Deterministic: buckets are walked in sorted key order, and a leftover
    single pick goes to the bucket with the most room left (key ascending on a
    tie)."""
    keys = sorted(sizes)
    quota = {k: 0 for k in keys}
    open_keys = [k for k in keys if sizes[k] > 0]
    remaining = min(int(n), sum(sizes.values()))
    while open_keys and remaining > 0:
        base = remaining // len(open_keys)
        if base == 0:                       # fewer picks left than buckets
            order = sorted(open_keys, key=lambda k: (-(sizes[k] - quota[k]), k))
            for k in order[:remaining]:
                quota[k] += 1
            break
        capped = [k for k in open_keys if sizes[k] - quota[k] <= base]
        if capped:
            for k in capped:
                take = sizes[k] - quota[k]
                quota[k] += take
                remaining -= take
            capped_set = set(capped)
            open_keys = [k for k in open_keys if k not in capped_set]
        else:
            for k in open_keys:
                quota[k] += base
            remaining -= base * len(open_keys)
    return quota


def _pool_labels(bank, filters) -> dict:
    """{image_id: (framing, face_cluster)} for the same pool ``_pool_embeddings``
    walks — one extra column read, no GPU."""
    rows = _pool_query(bank.id, thresholds(), **(filters or {})).all()
    return {r.id: (r.framing, r.face_cluster) for r in rows}


def _balance_axis_hint(axis, m, unlabelled, unknown) -> str:
    """The honest message for a bank that simply has not been labelled yet — the
    DEFAULT state of a fresh bank, not an error. Names the pass that is missing
    and the numbers, instead of returning an empty or misleading selection."""
    what = ('the shot type of each image' if axis == 'framing'
            else 'the shot type AND the person of each image')
    passes = ('run the 📐 Framing pass first' if axis == 'framing'
              else 'run the 📐 Framing and 👥 Group by person passes first')
    tail = ''
    if unknown and not unlabelled:
        tail = (f' — {unknown} of {m} came back as "unknown" framing, which is a '
                f'classification the balance cannot use')
    else:
        tail = f' — {unlabelled} of {m} images here have no label yet'
    return (f'{passes}: balanced selection needs {what}{tail}. '
            f'🎨 Pick diverse works without it.')


def select_balanced(user_id, bank_id, n=60, *, axis=_BALANCE_DEFAULT_AXIS,
                    typicality=_TYPICALITY_DEFAULT, filters=None):
    """Select ``n`` images SPREAD OVER the labels of ``axis`` instead of taking
    the top of one ranking: an even split across framings (and optionally across
    people), each bucket filled with the same farthest-point + typicality
    sampling ``select_diverse`` uses. So it is "the most varied 15 face shots,
    the most varied 15 busts, …" rather than "the most varied 60", which on a
    lopsided bank is 30 bodies and 2 faces.

    It ACCOMPANIES ``select_diverse``, it does not replace it: variety inside a
    space and coverage of a label are different questions, and the diverse
    selector still works on a bank with no labels at all (which is most banks
    until the 📐 Framing pass has run).

    Composition with the typicality guard: the isolation penalty is computed ONCE
    over the WHOLE filtered pool and then sliced per bucket — deliberately, since
    "alone in the bank" is a property of the bank. Computing it per bucket would
    make every member of a small bucket look isolated and penalise exactly the
    images the balance exists to bring in.

    Returns {'image_ids', 'pool', 'requested', 'selected', 'typicality', 'axis',
    'buckets': [{key, framing, cluster, available, fair_share, selected, short}],
    'unlabelled', 'unknown', 'shortfall'}. Raises ValueError (→400) when Score
    has not run, or when nothing in the filter carries the axis label."""
    import numpy as np
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if axis not in _BALANCE_AXES:
        axis = _BALANCE_DEFAULT_AXIS
    emb_by_path = _load_score_embeddings(bank)
    if not emb_by_path:
        raise ValueError('run ✨ Score first — balanced selection reuses its '
                         'embeddings')
    n = max(1, min(int(n), _CURATION_MAX_N))
    try:
        w = 0.0 if typicality is None else float(typicality)
    except (TypeError, ValueError):
        w = _TYPICALITY_DEFAULT
    w = max(0.0, min(1.0, w))
    ids, E = _pool_embeddings(bank, emb_by_path, filters or {})
    m = len(ids)
    if not m:
        raise ValueError('nothing to select from in the current filter')

    labels = _pool_labels(bank, filters or {})
    buckets, meta = {}, {}
    unlabelled = unknown = 0
    for pos, iid in enumerate(ids):
        fr, cl = labels.get(iid, (None, None))
        if fr is None:
            unlabelled += 1
            continue
        if fr not in _FRAMINGS:              # 'unknown' — a real classification,
            unknown += 1                     # but not one you can balance on
            continue
        if axis == 'framing+person':
            if cl is None:
                unlabelled += 1
                continue
            key = f'{fr}#{int(cl)}'
            meta[key] = (fr, int(cl))
        else:
            key = fr
            meta[key] = (fr, None)
        buckets.setdefault(key, []).append(pos)
    if not buckets:
        raise ValueError(_balance_axis_hint(axis, m, unlabelled, unknown))

    sizes = {k: len(v) for k, v in buckets.items()}
    quota = _balanced_quotas(sizes, n)
    # What a split with no ceiling WOULD have given each bucket — the yardstick
    # a shortfall is reported against ("back: 3 of an even 15").
    fair = _balanced_quotas({k: n for k in sizes}, n)

    factor = None
    if w > 0.0:
        factor = 10.0 ** (-_TYPICALITY_DECADES * w * _isolation_penalty(E))
    chosen, report = [], []
    for key in sorted(sizes):
        pos = buckets[key]                   # ascending id order (pool order)
        take = quota[key]
        if take >= len(pos):
            picked = list(pos)
        elif take <= 0:
            picked = []
        else:
            idx = np.asarray(pos)
            sub = np.ascontiguousarray(E[idx])
            subf = None if factor is None else np.ascontiguousarray(factor[idx])
            picked = [pos[i] for i in _farthest_point(sub, subf, take)]
        chosen.extend(picked)
        fr, cl = meta[key]
        report.append({'key': key, 'framing': fr, 'cluster': cl,
                       'available': len(pos), 'fair_share': fair[key],
                       'selected': len(picked),
                       'short': len(pos) < fair[key]})
    order = {k: i for i, k in enumerate(_FRAMINGS)}
    report.sort(key=lambda b: (order.get(b['framing'], 99),
                               b['cluster'] if b['cluster'] is not None else -1))
    return {'image_ids': sorted(ids[i] for i in chosen),
            'pool': m, 'requested': n, 'selected': len(chosen),
            'typicality': w, 'axis': axis, 'buckets': report,
            'unlabelled': unlabelled, 'unknown': unknown,
            'shortfall': max(0, n - len(chosen))}


def select_similar(user_id, bank_id, ref_id, n=60, min_score=None, *, filters=None):
    """Rank the (filtered) pool by CLIP cosine similarity to a REFERENCE bank image
    (its own cached ✨ Score embedding) — "keep what looks like THIS", to pull one
    person / look out of a mixed dump. Returns the top-``n`` most similar ids, OR
    everything with cosine ≥ ``min_score`` when that is given; the reference itself
    (cosine 1.0) is always included. Reuses the cached embeddings — no GPU.

    Returns {'results': [{id, score}], 'image_ids': [...], 'pool': m, 'ref_id'}.
    Raises ValueError (→400) when Score hasn't run or the reference has no cached
    embedding (e.g. it was rejected before Score, or edited since)."""
    import numpy as np
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    emb_by_path = _load_score_embeddings(bank)
    if not emb_by_path:
        raise ValueError('run ✨ Score first — reference similarity reuses its '
                         'embeddings')
    ref = db.session.get(BankImage, int(ref_id))
    if ref is None or ref.bank_id != bank_id:
        raise ValueError('reference image not found in this bank')
    ref_path = abs_image_path(bank, ref)
    ref_emb = emb_by_path.get(ref_path) if ref_path else None
    if ref_emb is None:
        raise ValueError('the reference image has no ✨ Score embedding — score '
                         'it first (it may have been rejected before Score ran)')
    ids, E = _pool_embeddings(bank, emb_by_path, filters or {})
    if not ids:
        return {'results': [], 'image_ids': [], 'pool': 0, 'ref_id': int(ref_id)}
    rv = np.asarray(ref_emb, dtype='float32')
    rv /= (np.linalg.norm(rv) + 1e-8)
    sims = E @ rv                                 # cosine similarity, (m,)
    order = np.argsort(-sims, kind='stable')     # desc; stable ⇒ id tie-break
    if min_score is not None:
        keep = [int(k) for k in order if sims[k] >= float(min_score)]
    else:
        n = max(1, min(int(n), _CURATION_MAX_N))
        keep = [int(k) for k in order[:n]]
    results = [{'id': ids[k], 'score': round(float(sims[k]), 4)} for k in keep]
    return {'results': results, 'image_ids': [ids[k] for k in keep],
            'pool': len(ids), 'ref_id': int(ref_id)}


def search_by_text(user_id, bank_id, query, n=60, *, filters=None):
    """Rank the (filtered) pool by CLIP similarity to a written QUERY — "brunette
    outdoors, wide shot" instead of a reference picture.

    Mechanically this is ``select_similar`` with the reference vector produced by
    CLIP's TEXT tower rather than read from the image cache: same embeddings, same
    cosine, same deterministic stable-argsort ordering. Ranking costs no GPU and no
    re-scan; only encoding the phrase leaves the Flask process (see
    clip_text_encoder, which caches every phrase on disk).

    It REFINES the current filter rather than replacing it — the candidate pool is
    exactly what the grid is showing, so "wide shot, inside this subfolder, among
    the undecided" composes without a second search grammar.

    TOP-N ONLY — there is deliberately no similarity threshold, and adding one
    would be a mistake rather than a feature. Measured on a real bank (48 images,
    8 unrelated datasets, this exact model): verified-correct top-1 hits scored
    0.177–0.233, while guaranteed-unrelated pairs reached up to 0.197 (median
    0.112). The two distributions OVERLAP — the unrelated ceiling outranks two
    genuinely correct answers. No cut exists that separates "relevant" from
    "unrelated": below ~0.20 it admits false positives, above ~0.18 it discards
    true ones. A threshold control would therefore offer the illusion of a
    boundary that does not exist, so the ranking is the whole product.

    Returns {'results': [{id, score}], 'image_ids', 'pool', 'filtered', 'unscored',
    'query', 'cached', 'score_range', 'pool_median'}.
      * ``unscored`` — an image with no ✨ Score embedding CANNOT be found by text;
        saying "0 results" without saying that would let the user conclude the
        image is gone.
      * ``pool_median`` — the median cosine over the WHOLE candidate pool for this
        query. It is the empirical "what a typical image here scores" baseline,
        measured per bank and per query, and it is what lets the UI judge whether
        a ranking discriminates at all without hard-coding any constant. On a
        single-subject bank (image-to-image cosine 0.60–0.89) the discriminating
        gap compresses by 30–70%, and that is the app's MAIN use case, not an
        edge case — so the baseline has to be measured, never assumed.

    Raises ValueError (→400) for an empty query or an unscored bank, and
    clip_text_encoder.TextEncodeError (→503) when no interpreter can run CLIP."""
    import numpy as np
    from . import clip_text_encoder
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    text = clip_text_encoder.normalize_query(query)
    if not text:
        raise ValueError('a search query is required')
    emb_by_path = _load_score_embeddings(bank)
    if not emb_by_path:
        raise ValueError('run ✨ Score first — text search ranks the embeddings '
                         'it computes')
    filters = filters or {}
    # How many rows the current filter holds AT ALL — the denominator that makes
    # "searched 120 of 400" (and therefore the unscored warning) truthful.
    filtered = _pool_query(bank_id, thresholds(), **filters).count()
    ids, E = _pool_embeddings(bank, emb_by_path, filters)
    # Encode AFTER the cheap refusals: never make someone wait on a CLIP load to
    # then be told their bank was never scored.
    qv, cached = clip_text_encoder.encode_query(text)
    base = {'query': text, 'cached': bool(cached), 'filtered': int(filtered),
            'pool': len(ids), 'unscored': max(0, int(filtered) - len(ids))}
    if not ids:
        return {**base, 'results': [], 'image_ids': [], 'score_range': None,
                'pool_median': None}
    qv = np.asarray(qv, dtype='float32')
    qv /= (float(np.linalg.norm(qv)) + 1e-8)
    sims = E @ qv                                # cosine similarity, (m,)
    order = np.argsort(-sims, kind='stable')     # desc; stable ⇒ id tie-break
    n = max(1, min(int(n), _CURATION_MAX_N))
    keep = [int(k) for k in order[:n]]
    results = [{'id': ids[k], 'score': round(float(sims[k]), 4)} for k in keep]
    # The span of what came back, plus the pool's own median — together they let
    # the UI say whether this ranking discriminates, using only numbers measured
    # on THIS bank for THIS query. An absolute band would be wrong everywhere:
    # the same model's "good" ceiling barely moves between corpora while its
    # floor climbs sharply on real photographs of people.
    score_range = ({'top': results[0]['score'], 'bottom': results[-1]['score']}
                   if results else None)
    return {**base, 'results': results,
            'image_ids': [ids[k] for k in keep], 'score_range': score_range,
            'pool_median': round(float(np.median(sims)), 4)}


def _trash_or_remove(path: str) -> str:
    """Get a rejected source file out of the user's folder, keeping it
    recoverable (OS trash → app trash → permanent unlink). The policy itself is
    app-wide and lives in ``services.trash``; this is the bank's entry point into
    it, kept as a name because the tests and the delete sweep both address it."""
    return trash.dispose(path, context='bank-rejected')


def _bank_folders(user_id, exclude_id=None) -> list:
    """[(bank, normalised realpath)] for the user's banks with a usable folder."""
    out = []
    for b in ImageBank.query.filter_by(user_id=user_id).all():
        if exclude_id is not None and b.id == exclude_id:
            continue
        if not b.source_path:
            continue
        try:
            out.append((b, os.path.normcase(os.path.realpath(b.source_path))))
        except (OSError, ValueError):
            continue
    return out


def overlapping_banks(user_id, bank_id) -> list:
    """The user's OTHER banks whose folder contains, or sits inside, this one's.

    Two banks over nested folders see the same files, and a bank never owns its
    source folder — so a delete run from one silently amputates the other. The
    UI has to be able to say so BEFORE the click. Returns
    [{'id', 'name', 'source_path', 'relation'}] with relation 'parent' (it
    contains us) or 'child' (it sits inside us)."""
    bank = get_bank(user_id, bank_id)
    if not bank or not bank.source_path:
        return []
    try:
        mine = os.path.normcase(os.path.realpath(bank.source_path))
    except (OSError, ValueError):
        return []
    out = []
    for other, theirs in _bank_folders(user_id, exclude_id=bank_id):
        if theirs == mine:
            relation = 'same'
        elif mine.startswith(theirs + os.sep):
            relation = 'parent'
        elif theirs.startswith(mine + os.sep):
            relation = 'child'
        else:
            continue
        out.append({'id': other.id, 'name': other.name,
                    'source_path': other.source_path, 'relation': relation})
    return sorted(out, key=lambda o: o['id'])


class BankSharesDataset(ValueError):
    """A destructive action was asked of a bank whose folder IS a dataset's.

    A subclass of ValueError so nothing that already catches ValueError starts
    500-ing, but a distinct type so the one route that answers 404 to "bank not
    found" can tell this apart and answer 400 with the explanation instead."""


def bank_dataset_conflict(user_id, bank_id) -> dict | None:
    """Does THIS bank already sit on a dataset's storage folder? None when it
    does not — which is every bank the guard in create_bank/_relocate_target has
    ever seen.

    The guard alone protects nobody who already has such a bank: it was created
    before the guard existed, it is in their database right now, and the click
    that hurts is 🗑 Delete rejected. So the conflict is DETECTED at open time
    (it rides in the workspace payload, which is also the 2 s poll — one
    realpath, no walk) and the destructive action refuses. Deliberately nothing
    else: the bank stays fully readable and fully triageable, and NOTHING is
    ever removed on the app's own initiative. Only the user can decide whether
    that bank should be relocated or dropped."""
    bank = get_bank(user_id, bank_id)
    if not bank or not bank.source_path:
        return None
    return path_guard.dataset_folder_conflict(bank.source_path)


def rejected_delete_preview(user_id, bank_id) -> dict | None:
    """What a 🗑 Delete rejected would actually destroy — the honest warning the
    confirmation needs. Counts the rejected files of this bank that ANOTHER bank
    also lists, per bank, by matching absolute paths against that bank's own
    inventory. None when the bank is gone.

    Returns {'rejected', 'mode', 'shared': [{'id','name','relation','files'}]}.
    ``mode`` is where the files would go, resolved the same way the deletion
    resolves it, so the dialog never promises the wrong thing."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    rows = BankImage.query.filter_by(bank_id=bank_id, status='reject').all()
    mine = {}
    for r in rows:
        p = abs_image_path(bank, r)
        if p:
            mine[os.path.normcase(p)] = True
    shared = []
    for other in overlapping_banks(user_id, bank_id):
        ob = db.session.get(ImageBank, other['id'])
        if ob is None:
            continue
        n = 0
        for (rel,) in db.session.query(BankImage.relpath).filter_by(bank_id=ob.id):
            try:
                full = os.path.normcase(os.path.realpath(
                    os.path.join(os.path.realpath(ob.source_path), rel)))
            except (OSError, ValueError):
                continue
            if full in mine:
                n += 1
        if n:
            shared.append({'id': other['id'], 'name': other['name'],
                           'relation': other['relation'], 'files': n})
    return {'rejected': len(rows), 'mode': _delete_mode(), 'shared': shared,
            # The hard stop, next to the soft warnings: another BANK losing files
            # is a warning the user may accept, a DATASET losing them is not.
            'dataset_conflict': bank_dataset_conflict(user_id, bank_id)}


def _delete_mode() -> str:
    """Where a deleted source file WOULD go, without deleting anything, so the
    confirmation can say it. Same probe as ``services.trash.disposal_mode``."""
    return trash.disposal_mode()


def delete_rejected(user_id, bank_id) -> dict:
    """Delete the SOURCE files of every status='reject' image from disk, then
    drop their bank_image rows.

    This is the ONLY bank action that writes to the user's source folder. Where
    the files land is _trash_or_remove's decision (OS trash, else the app's own
    trash, else a permanent unlink) and rides back in 'mode' — the confirmation
    dialog says it BEFORE the click, via rejected_delete_preview().

    ⚠️ A bank does not own its folder. When another bank sits over the same tree,
    these files are ITS files too and it will find them gone; the preview names
    those banks so the warning can.

    Non-rejected images are never touched. Per-file failures (permission, a path
    that escapes the bank folder) are collected and reported; they never abort
    the batch. A row is dropped only when its file is gone afterwards (deleted,
    trashed, or already absent) — a file we failed to remove keeps its row so the
    user can see and retry it. Returns
    {'mode', 'deleted', 'trashed', 'already_absent', 'rows_removed', 'skipped'}
    where 'trashed' counts everything that stayed recoverable.
    """
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if bank_jobs.running(bank_id):
        raise RuntimeError('a job is running on this bank — stop it first')
    # An install that predates the create-time guard can still hold a bank whose
    # folder IS a dataset's. Here, and only here, that stops being cosmetic: these
    # "rejects" are the dataset's images. Refuse — never silently delete, never
    # silently clean up on the user's behalf.
    conflict = bank_dataset_conflict(user_id, bank_id)
    if conflict:
        raise BankSharesDataset(
            'This bank points at a dataset\'s own image folder, so deleting its '
            'rejected files would delete images out of the dataset. Nothing was '
            f'deleted. {conflict["message"]}')

    rows = BankImage.query.filter_by(bank_id=bank_id, status='reject').all()
    out = {'mode': 'trash', 'deleted': 0, 'trashed': 0, 'already_absent': 0,
           'rows_removed': 0, 'skipped': []}
    remove_ids = []
    modes_used = set()
    for row in rows:
        path = abs_image_path(bank, row)
        if path is None:
            # relpath escapes the bank folder — refuse to touch it, keep the row.
            out['skipped'].append({'relpath': row.relpath, 'reason': 'unsafe_path'})
            continue
        if not os.path.exists(path):
            out['already_absent'] += 1
            remove_ids.append(row.id)
            continue
        try:
            mode = _trash_or_remove(path)
        except OSError as e:
            out['skipped'].append({'relpath': row.relpath, 'reason': str(e)})
            continue
        modes_used.add(mode)
        if mode == 'delete':
            out['deleted'] += 1
        else:
            out['trashed'] += 1          # OS trash or app trash — recoverable
        remove_ids.append(row.id)

    for i0 in range(0, len(remove_ids), _SQL_IN_CHUNK):
        BankImage.query.filter(
            BankImage.id.in_(remove_ids[i0:i0 + _SQL_IN_CHUNK])
        ).delete(synchronize_session=False)
    out['rows_removed'] = len(remove_ids)
    db.session.commit()
    # The pending ↩ offer points at rows this run just dropped — restoring them
    # would find nothing. Withdraw it rather than advertise a restore we cannot
    # perform (the files themselves went to a trash only the user can reach).
    bank_undo.clear(bank_id)
    # Report the WORST outcome that happened: one permanently removed file makes
    # the run 'delete', whatever the rest did. The UI wording follows this.
    for mode in ('delete', 'app_trash', 'trash'):
        if mode in modes_used:
            out['mode'] = mode
            break
    return out


# --- cooperative-cancel subprocess driver (shared by the face + score passes) --
# A bank inference pass runs for minutes over thousands of images. Its embeddings
# are cached incrementally, but a brutal proc.kill on Stop still throws away the
# in-flight slice AND leaves the UI mute. Instead we ask the child to stop CLEANLY:
# drop a sentinel file it polls between images so it flushes its cache and reports
# how much it kept; a watchdog timer hard-kills it only if it doesn't stop within
# the grace period. The child also writes a plain-text ``<cache>.count`` sidecar we
# read back here (the Flask venv has no numpy to open the .npz), so a stopped pass
# always shows an honest count — even in the rare hard-kill case.
_INFER_CANCEL_GRACE = 15.0   # seconds a cleanly-cancelled child gets before a kill
_CACHED_RE = re.compile(r'(\d+) image\(s\), (\d+) cached')


def _safe_kill(proc):
    try:
        proc.kill()
    except Exception:  # noqa: BLE001 — already gone is fine
        pass


def _read_cache_count(cache_path):
    """The count the child last flushed to ``<cache>.count``, or None."""
    try:
        with open(str(cache_path) + '.count', encoding='utf-8') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _stopped_detail(noun, data, cache_path, total):
    """The honest end-of-pass line when the user Stopped it. Prefers the child's
    own cancel counts, falls back to the flushed sidecar count, and never invents
    a number it can't back up."""
    n = data.get('cached')
    if n is None:
        n = _read_cache_count(cache_path)
    if n is None:
        return 'Stopped — progress saved to cache; relaunch to finish and cluster'
    n = int(n)
    remaining = data.get('remaining')
    remaining = int(remaining) if remaining is not None else max(0, int(total) - n)
    return (f'Stopped — {n} {noun} ({remaining} remaining); '
            'relaunch to finish and cluster')


def _drive_infer_subprocess(job, python, script, payload, cache_path,
                            progress_re, window):
    """Run an infer subprocess, streaming its stderr progress into ``job`` and
    honouring Stop cooperatively. Returns (data, stderr_tail, returncode) where
    ``data`` is the child's last JSON line (``cancelled: true`` when it stopped
    cleanly). On the first "N cached" line it sets a "resuming" hint, so relaunching
    over a partly-cached bank doesn't look like a full recompute."""
    import json
    import threading
    cancel_file = str(cache_path) + '.cancel'
    try:
        os.remove(cancel_file)   # never inherit a stale sentinel from a past run
    except OSError:
        pass
    hint = {'shown': False}
    with window:
        proc = subprocess.Popen(
            [python, script], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        killer = {'timer': None}

        def _cancel():
            # Ask for a clean stop (the child flushes + exits), and arm a watchdog
            # that hard-kills ONLY if it doesn't stop within the grace period.
            try:
                with open(cancel_file, 'w', encoding='utf-8') as f:
                    f.write('1')
            except OSError:
                pass
            t = threading.Timer(_INFER_CANCEL_GRACE, _safe_kill, args=(proc,))
            t.daemon = True
            t.start()
            killer['timer'] = t

        bank_jobs.set_cancel_hook(job, _cancel)
        stderr_tail = deque(maxlen=5)

        def _drain_stderr():
            for line in proc.stderr:
                line = line.strip()
                if line:
                    stderr_tail.append(line)
                m = progress_re.search(line)
                if m:
                    bank_jobs.progress(job, done=int(m.group(1)), total=int(m.group(2)))
                if not hint['shown']:
                    mc = _CACHED_RE.search(line)
                    if mc:
                        hint['shown'] = True
                        total, cached = int(mc.group(1)), int(mc.group(2))
                        if 0 < cached < total:
                            bank_jobs.progress(
                                job,
                                detail=f'resuming — {cached} of {total} already cached')

        t = threading.Thread(target=_drain_stderr, daemon=True)
        t.start()
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except OSError:
            pass  # process died early — surfaced through the exit path below
        stdout = proc.stdout.read()
        proc.wait()
        t.join(timeout=5)
        if killer['timer']:
            killer['timer'].cancel()
    try:
        os.remove(cancel_file)
    except OSError:
        pass
    line = next((ln for ln in reversed(stdout.splitlines())
                 if ln.strip().startswith('{')), '')
    try:
        data = json.loads(line) if line else {}
    except json.JSONDecodeError:
        data = {}
    return data, stderr_tail, proc.returncode


# --- subject (face) pass ----------------------------------------------------
_EMBED_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'face_embed_infer.py')
_PROGRESS_RE = re.compile(r'\[embed\] (\d+)/(\d+)')


def _resolve_face_device():
    """(device, use_gpu) for the face pass. 'cpu' is the safe default and never
    touches the GPU; GPU is used ONLY when the face interpreter truly exposes
    CUDA (onnxruntime-gpu installed) and the config allows it. Config
    face_scoring.device: 'auto' (default — GPU if available) | 'cpu' | 'cuda'.
    A 'cuda' request without CUDA available still degrades to CPU here, so the
    parent never opens the GPU-exclusive window for a pass that will run on CPU."""
    from .. import capabilities
    pref = str(cfg.get('face_scoring.device') or 'auto').lower()
    use_gpu = pref in ('auto', 'cuda') and capabilities.face_gpu_available()
    return ('cuda' if use_gpu else 'cpu'), use_gpu


def start_faces(app, user_id, bank_id):
    """Launch the face embedding + person clustering pass over the bank's
    non-rejected images. Needs the face-scoring extra (Setup ▸ Quality tools)."""
    from .face_similarity import is_available
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if not is_available():
        raise RuntimeError(
            'face scoring is not installed (Quality tools step in Setup)')
    total = (BankImage.query.filter_by(bank_id=bank_id)
             .filter(BankImage.status != 'reject').count())
    return bank_jobs.start(app, bank_id, 'faces', _faces_job(bank_id), total=total)


def _faces_job(bank_id):
    def run(job):
        import json as _json
        import sys
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .filter(BankImage.status != 'reject')
                .order_by(BankImage.id.asc()).all())
        by_path = {}
        for r in rows:
            p = abs_image_path(bank, r)
            if p and os.path.isfile(p):
                by_path[p] = r.id
        paths = list(by_path)
        bank_jobs.progress(job, done=0, total=len(paths), detail='face pass')
        if not paths:
            return
        _bank_dir(bank_id).mkdir(parents=True, exist_ok=True)
        th = thresholds()
        # Device: 'cpu' (default, never touches the GPU/ComfyUI) or 'cuda'.
        # 'auto' = GPU when the face interpreter actually exposes CUDA
        # (onnxruntime-gpu installed), else CPU. The GPU path is used ONLY when
        # CUDA is truly available AND must run inside the GPU-exclusive window so
        # it never competes with a training / scoring pass; a CPU pass stays out
        # of the window (it can run alongside GPU work).
        from ..gpu_window import gpu_exclusive_vision_window
        from contextlib import nullcontext
        device, use_gpu = _resolve_face_device()
        cache_path = _face_cache_path(bank_id)
        payload = _json.dumps({
            'images': paths,
            'models_root': cfg.get('face_scoring.models_root') or None,
            'cache': str(cache_path),
            'cancel_file': str(cache_path) + '.cancel',
            'threshold': th['face_threshold'],
            'device': device,
        })
        python = cfg.get('face_scoring.python') or sys.executable
        window = gpu_exclusive_vision_window(flag_ttl=1800) if use_gpu else nullcontext()
        data, stderr_tail, returncode = _drive_infer_subprocess(
            job, python, _EMBED_SCRIPT, payload, cache_path, _PROGRESS_RE, window)
        # Stopped by the user — say exactly what's kept, never a mute ✗ (the cached
        # embeddings are safe; relaunching skips them and only finishes the rest).
        if data.get('cancelled') or (bank_jobs.cancelled(job) and not data.get('ok')):
            bank_jobs.progress(job, detail=_stopped_detail(
                'face embeddings cached', data, cache_path, len(paths)))
            return
        if not data.get('ok'):
            tail = data.get('error') or (stderr_tail[-1] if stderr_tail else '')
            raise RuntimeError(tail or f'face pass produced no output '
                                       f'(rc={returncode})')
        results = data.get('results') or {}
        clusters = data.get('clusters') or {}
        done = 0
        for p, image_id in by_path.items():
            row = db.session.get(BankImage, image_id)
            if row is None:
                continue
            res = results.get(p) or {}
            row.face_state = res.get('state')
            row.face_det = res.get('det')
            row.face_cluster = clusters.get(p)
            done += 1
            if done % 200 == 0:
                db.session.commit()
        db.session.commit()
        sizes = {}
        for cid in clusters.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        multi = sum(1 for n in sizes.values() if n >= 2)
        bank_jobs.progress(job, detail=f'done — {multi} person cluster(s) '
                                       f'of 2+ images')
    return run


# --- scoring pass (aesthetic · NSFW · style) --------------------------------
_SCORE_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'bank_score_infer.py')
_SCORE_PROGRESS_RE = re.compile(r'\[score\] (\d+)/(\d+)')


def _gpu_busy_reason() -> str | None:
    """A human reason the GPU is unavailable right now, or None. Same system
    flags training and the vision window use, so a bank GPU pass never races a
    training run or a captioning pass (the 'never concurrent with a training'
    guarantee). Checked up-front for an immediate 503 rather than a doomed 202."""
    from ..job_queue import queue_manager
    if queue_manager._get_system_state('training_in_progress'):
        return 'training is running on the GPU — try again once it finishes'
    if queue_manager._get_system_state('vision_in_progress'):
        return 'a vision/GPU pass is already running — try again in a moment'
    return None


def _resolve_score_device() -> tuple:
    """(device, use_gpu) for the scoring pass — what the CHILD will actually do.

    bank_score_infer.py picks `cuda if torch.cuda.is_available() else cpu` on its
    own, so the parent asks the same interpreter the same question. It matters
    beyond a label: use_gpu decides whether we take the GPU-exclusive window,
    which unloads ComfyUI and blocks any training start for the whole pass. The
    extra ships CPU-only torch, so the honest answer is usually 'cpu' — and an
    hour of CPU work must not hold a GPU it never touches."""
    from ..capabilities import bank_scoring_gpu_available
    use_gpu = bank_scoring_gpu_available()
    return ('cuda' if use_gpu else 'cpu'), use_gpu


# CLIP ViT-L/14 measured at ~336 ms/image on CPU against ~15 ms on a recent
# card. Used only to warn, never to refuse — a slow pass is still a pass.
SCORE_CPU_MS_PER_IMAGE = 336


def score_device_info(bank_id=None) -> dict:
    """What ✨ Score will run on, and — when that is the CPU — how long the bank
    would take and whether this machine even has a card to switch to.

    The pass is not slow by accident: the scoring extra installs CPU-only torch
    on purpose (Setup builds it a small venv rather than pushing a ~2.5 GB CUDA
    download on people with no GPU). That is a defensible default, but it has to
    be VISIBLE — an unexplained hour looks like a hang, and the user cannot fix
    what nobody told them about."""
    from ..capabilities import gpu_vram_gb
    device, use_gpu = _resolve_score_device()
    out = {'device': device, 'gpu': use_gpu, 'gpu_present': False,
           'eta_minutes': None}
    if use_gpu:
        return out
    out['gpu_present'] = gpu_vram_gb() is not None
    if bank_id is not None:
        pending = (BankImage.query.filter_by(bank_id=bank_id)
                   .filter(BankImage.status != 'reject')
                   .filter(BankImage.aesthetic_score.is_(None)).count())
        # Rounded UP to a whole minute while there is anything left: "0 minutes"
        # for work that is about to start reads as "instant" and is a lie.
        out['eta_minutes'] = (max(1, round(pending * SCORE_CPU_MS_PER_IMAGE / 60000))
                              if pending else None)
    return out


def start_score(app, user_id, bank_id):
    """Launch the scoring pass (LAION aesthetic + NSFW + style clustering) over
    the bank's non-rejected images. Needs the bank-scoring extra (Setup ▸ Quality
    tools). Serialized against training/vision ONLY when it will really run on
    the GPU: refusing a CPU pass because 'the GPU is busy' would block an hour of
    work that never wanted the card."""
    from ..capabilities import probe_bank_scoring
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if not probe_bank_scoring().get('ok'):
        raise RuntimeError('bank scoring is not installed '
                           '(Quality tools step in Setup)')
    _device, use_gpu = _resolve_score_device()
    reason = _gpu_busy_reason() if use_gpu else None
    if reason:
        raise RuntimeError(reason)
    total = (BankImage.query.filter_by(bank_id=bank_id)
             .filter(BankImage.status != 'reject').count())
    return bank_jobs.start(app, bank_id, 'score', _score_job(bank_id), total=total)


def _score_job(bank_id):
    def run(job):
        import json as _json
        import sys
        from contextlib import nullcontext

        from ..gpu_window import gpu_exclusive_vision_window
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .filter(BankImage.status != 'reject')
                .order_by(BankImage.id.asc()).all())
        by_path = {}
        for r in rows:
            p = abs_image_path(bank, r)
            if p and os.path.isfile(p):
                by_path[p] = r.id
        paths = list(by_path)
        device, use_gpu = _resolve_score_device()
        # Say WHICH device, every time. On CPU this pass is ~20× slower (CLIP
        # ViT-L is the whole cost), and a progress bar crawling for an hour with
        # no explanation reads as a hang.
        bank_jobs.progress(job, done=0, total=len(paths),
                           detail=f'scoring pass ({device.upper()})')
        if not paths:
            return
        _bank_dir(bank_id).mkdir(parents=True, exist_ok=True)
        th = thresholds()
        cache_path = _score_cache_path(bank_id)
        payload = _json.dumps({
            'images': paths,
            'models_root': cfg.get('bank_scoring.models_root') or None,
            'cache': str(cache_path),
            'cancel_file': str(cache_path) + '.cancel',
            'style_threshold': th['style_threshold'],
        })
        python = cfg.get('bank_scoring.python') or sys.executable
        # The GPU-exclusive window frees ComfyUI's VRAM and blocks any training
        # start for the whole pass — so it is taken ONLY when the child really
        # runs on the card, exactly like the face pass. Holding it through an
        # hour of CPU inference was the worst of both worlds: the GPU idle and
        # unusable, the work slow anyway.
        window = (gpu_exclusive_vision_window(flag_ttl=1800) if use_gpu
                  else nullcontext())
        data, stderr_tail, returncode = _drive_infer_subprocess(
            job, python, _SCORE_SCRIPT, payload, cache_path, _SCORE_PROGRESS_RE,
            window)
        # Stopped by the user — say exactly what's kept, never a mute ✗ (the cached
        # scores/embeddings are safe; relaunching skips them and finishes the rest).
        if data.get('cancelled') or (bank_jobs.cancelled(job) and not data.get('ok')):
            bank_jobs.progress(job, detail=_stopped_detail(
                'images scored', data, cache_path, len(paths)))
            return
        if not data.get('ok'):
            tail = data.get('error') or (stderr_tail[-1] if stderr_tail else '')
            raise RuntimeError(tail or f'scoring pass produced no output '
                                       f'(rc={returncode})')
        results = data.get('results') or {}
        clusters = data.get('clusters') or {}
        done = 0
        for p, image_id in by_path.items():
            row = db.session.get(BankImage, image_id)
            if row is None:
                continue
            res = results.get(p) or {}
            row.aesthetic_score = res.get('aesthetic')
            row.nsfw_score = res.get('nsfw')
            row.style_cluster = clusters.get(p)
            done += 1
            if done % 200 == 0:
                db.session.commit()
        db.session.commit()
        sizes = {}
        for cid in clusters.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        multi = sum(1 for n in sizes.values() if n >= 2)
        ok = [r for r in results.values() if r.get('state') == 'ok']
        # Name any head that produced nothing, so a degraded pass says so out loud
        # (graceful degradation must be visible, never a silent gap).
        missing = []
        if ok and not any('aesthetic' in r for r in ok):
            missing.append('aesthetic')
        if ok and not any('nsfw' in r for r in ok):
            missing.append('NSFW')
        detail = (f'done — scored {len(ok)} image(s), '
                  f'{multi} style group(s) of 2+')
        if missing:
            detail += f' ({" + ".join(missing)} head unavailable)'
        bank_jobs.progress(job, detail=detail)
    return run


# --- watermark pass (reuses the dataset Qwen3-VL overlaid-mark detector) -----
def _watermark_scan_query(bank_id, rescan):
    """The rows the detection pass should look at.

    Not a rescan = "finish the job": rows never scanned, PLUS rows flagged
    'detected' with no bbox. The latter exist because the pass used to parse the
    box and keep only the boolean — they would be invisible to both cleaning
    levels forever, so a plain re-run adopts them instead of asking the user to
    guess. 'dismissed' rows are never re-examined (the user already ruled), even
    on a rescan — same anti-frustration rule as the dataset detector."""
    q = (BankImage.query.filter_by(bank_id=bank_id)
         .filter(BankImage.status != 'reject')
         .filter(or_(BankImage.watermark_state.is_(None),
                     BankImage.watermark_state != 'dismissed')))
    if not rescan:
        q = q.filter(or_(BankImage.watermark_state.is_(None),
                         and_(BankImage.watermark_state == 'detected',
                              BankImage.watermark_bbox.is_(None))))
    return q


def start_watermark(app, user_id, bank_id, rescan=False):
    """Launch the overlaid-watermark scan over the bank's non-rejected images,
    reusing the SAME Qwen3-VL detector the datasets use. Needs the vision model
    pulled; serialized against training/vision (503 when the GPU is held)."""
    from ..capabilities import probe_ollama_model
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    # Occupancy BEFORE the model probe, and the order is the whole point: a bank
    # that is already busy is busy whether or not Ollama answers. Probing first
    # made a busy bank report "the vision model is not available" whenever the
    # model happened to be unreachable -- the wrong reason, sending the user to
    # fix an unrelated thing while the real answer was "wait, a pass is running".
    # It also made the refusal depend on an EXTERNAL SERVICE, which is why CI
    # (no Ollama) failed a release on it while two agents read it as a flake.
    if bank_jobs.running(bank_id):
        raise bank_jobs.BankJobBusy((bank_jobs.get(bank_id) or {}).get('kind') or 'background')
    if not probe_ollama_model().get('ok'):
        raise RuntimeError('the vision model is not available '
                           '(Settings ▸ Captioning & quality)')
    reason = _gpu_busy_reason()
    if reason:
        raise RuntimeError(reason)
    return bank_jobs.start(app, bank_id, 'watermark',
                           _watermark_job(bank_id, rescan),
                           total=_watermark_scan_query(bank_id, rescan).count())


def _watermark_job(bank_id, rescan):
    def run(job):
        import json as _json
        from .face_dataset_service import WATERMARK_BBOX_PROMPT, _parse_watermark_bbox
        from .vision_ollama import describe_image_ollama, unload_vision_model
        from .vision_pool import map_vision
        from ..gpu_window import gpu_exclusive_vision_window
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = _watermark_scan_query(bank_id, rescan).order_by(BankImage.id.asc()).all()
        bank_jobs.progress(job, done=0, total=len(rows), detail='watermark scan')
        if not rows:
            return
        detected = clean = errors = checked = unanswered = 0

        def prepared():
            """Yielded on the JOB's thread, one image per free slot in the pool.
            Everything that reads the database or has a side effect lives here,
            so it stays off the workers, keeps its original order, and is only
            paid for by images the pass actually reaches (which matters: the
            discard below is destructive)."""
            for row in rows:
                # Always detect on the SOURCE pixels: a re-scan of an already
                # cleaned image drops its cleaned version first (otherwise we
                # would be asking "is there a watermark?" about our own edit).
                if row.watermark_clean_method:
                    _discard_clean_blob(bank_id, row)
                yield row, abs_image_path(bank, row)

        def ask(item):
            """WORKER thread: read the file, ask Ollama. Touches no session — the
            path was resolved above, on the owning thread. Returns None (not '')
            for a file that is gone, so the caller can tell "nothing to analyse"
            from "the model answered nothing"."""
            _row, path = item
            if not path or not os.path.isfile(path):
                return None
            with open(path, 'rb') as fh:
                return describe_image_ollama(
                    fh.read(), WATERMARK_BBOX_PROMPT, num_predict=400,
                    prefer_json=True, fmt='json', keep_alive='5m')

        with gpu_exclusive_vision_window(flag_ttl=1800):
            try:
                # The calls overlap (see vision_pool); the loop body — every
                # database write below — still runs here, on this one thread.
                for (row, _path), raw, error in map_vision(
                        prepared(), ask,
                        should_cancel=lambda: bank_jobs.cancelled(job)):
                    if error is not None:  # one bad file never sinks the pass
                        row.watermark_state = 'error'
                        errors += 1
                    elif raw is None:      # file gone: leave the row as it was
                        pass
                    # Empty output = Ollama unreachable, NOT "clean": leave the
                    # state untouched so a retry can finish it (same reasoning as
                    # the dataset detector), never falsely mark everything clean.
                    elif not raw.strip():
                        # COUNTED, not merely skipped: a pass where every image
                        # came back empty reported "done — 0 with a watermark,
                        # 0 clean", which reads as "looked at them all, found
                        # nothing" when in truth nothing could be looked at. The
                        # rows stay unscanned on purpose — the report must say so.
                        unanswered += 1
                    else:
                        bbox = _parse_watermark_bbox(raw)
                        if bbox:
                            row.watermark_state = 'detected'
                            # Keep the box — the crop/inpaint levels route on it.
                            row.watermark_bbox = _json.dumps([round(v, 4) for v in bbox])
                            detected += 1
                        else:
                            row.watermark_state = 'none'
                            row.watermark_bbox = None
                            clean += 1
                        checked += 1
                        if checked % 25 == 0:
                            db.session.commit()
                    bank_jobs.bump(job)
            finally:
                db.session.commit()
                unload_vision_model()  # hand the VRAM back to ComfyUI
        if bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail=f'cancelled — {detected} with a watermark '
                                           f'so far')
            return
        detail = f'done — {detected} with a watermark, {clean} clean'
        if unanswered:
            detail += (f', {unanswered} not analysed (the vision model returned '
                       'nothing — check Ollama in Settings, then run it again)')
        if errors:
            detail += f', {errors} unreadable'
        bank_jobs.progress(job, detail=detail)
    return run


# --- watermark cleaning: two MANUAL levels ----------------------------------
# The bank's folder belongs to the user and is never written to, so "cleaning a
# watermark" here means writing a SEPARATE cleaned blob under the bank's own
# working directory (clean/<image_id>.webp) and pointing the readers at it
# (resolved_image_path). The original therefore stays untouched by construction,
# which is what makes undo trivial and both levels risk-free.
#
# The escalation is deliberate and each level is launched BY HAND:
#   level 1 — ✂ auto-crop: CPU/PIL, invents no pixel. Only touches marks the
#             dataset router calls croppable (inside a border band, and the crop
#             still leaves a usable image). Everything else is left flagged.
#   level 2 — 🧽 inpaint: LaMa (fast, non-generative) or Klein (ComfyUI, handles
#             on-subject marks) over what is STILL flagged. allow_crop=False, so
#             a border mark that level 1 skipped (or that the user never cropped)
#             is repainted rather than cropped — level 2 is the "repaint what's
#             left" lane, not a second router.
# Both reuse the dataset routing/engines verbatim; nothing about the decision
# logic is re-implemented here.
def _clean_pool_query(bank_id):
    """Images a cleaning level can act on: still flagged, with SOMETHING to act
    on, not rejected. 'cleaned'/'dismissed'/'none' rows are out by construction.

    "Something to act on" is the stored bbox OR a hand-drawn mask: a row the
    detector left without a box (an older build) becomes cleanable the moment
    the user draws the zones themselves — that drawing IS the missing box."""
    return (BankImage.query.filter_by(bank_id=bank_id,
                                      watermark_state='detected')
            .filter(BankImage.status != 'reject')
            .filter(or_(BankImage.watermark_bbox.isnot(None),
                        BankImage.watermark_regions.isnot(None))))


def _needs_rescan_count(bank_id) -> int:
    """Rows flagged by an older build that kept no bbox — nothing can route them
    until a scan re-adopts them (see _watermark_scan_query). A row the user has
    masked by hand is NOT one of them: it no longer needs the detector."""
    return (BankImage.query.filter_by(bank_id=bank_id, watermark_state='detected')
            .filter(BankImage.status != 'reject')
            .filter(BankImage.watermark_bbox.is_(None))
            .filter(BankImage.watermark_regions.is_(None)).count())


def _discard_clean_blob(bank_id, row) -> None:
    """Forget a cleaned version: delete the blob, drop the stale thumbnail and
    clear the method so the readers fall back to the source. No commit (the
    caller owns the transaction)."""
    try:
        clean_image_path(bank_id, row.id).unlink()
    except OSError:
        pass
    drop_derived(bank_id, row.id)
    row.watermark_clean_method = None


def _clean_bbox(row):
    """The stored bbox as a 4-float tuple, or None when it's unusable."""
    try:
        import json as _json
        box = _json.loads(row.watermark_bbox or '')
    except (ValueError, TypeError):
        return None
    if not (isinstance(box, list) and len(box) == 4):
        return None
    try:
        return tuple(float(v) for v in box)
    except (TypeError, ValueError):
        return None


def _clean_regions(row):
    """THE mask both cleaning levels must act on — ``(boxes, manual, problem)``.

    A hand-drawn mask WINS over the detector's bbox. That is the entire point of
    letting the user edit it: a correction the cleaning pass then ignores is
    worse than no editor at all, because the user believes the fix landed.

    ``manual`` says the boxes came from the user (drives the routing: a hand mask
    is a REPAINT instruction — it can hold several zones and zones on the subject,
    neither of which a border crop can express, exactly like the dataset lane).
    An EMPTY manual mask returns ``([], True, None)``: an explicit "nothing to
    repaint here", never a silent fall back to the box. ``problem`` is set only
    when the stored JSON cannot be read as a mask (a genuine failure)."""
    if row.watermark_regions is not None:
        import json as _json
        try:
            stored = _json.loads(row.watermark_regions or '')
        except (ValueError, TypeError):
            return [], True, 'unreadable watermark regions'
        try:
            regions = normalize_watermark_regions(stored, allow_null=False)
        except ValueError as e:
            return [], True, f'invalid watermark regions: {e}'
        return [tuple(box) for box in regions], True, None
    bbox = _clean_bbox(row)
    return ([bbox] if bbox else []), False, None


def _source_size(bank, row):
    """(path, W, H) of the SOURCE image, or (None, 0, 0) when unreadable."""
    path = abs_image_path(bank, row)
    if not path or not os.path.isfile(path):
        return None, 0, 0
    try:
        with Image.open(path) as im:
            return path, im.width, im.height
    except (OSError, ValueError):
        return None, 0, 0


def _stage_clean_copy(bank_id, row, src_path) -> Path:
    """Put a working COPY of the source in the bank's clean/ directory and return
    it. Every editor (crop, LaMa, Klein) then works in place ON THE COPY — the
    source path is deliberately never handed to a writer."""
    dst = clean_image_path(bank_id, row.id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return dst


def start_watermark_crop(app, user_id, bank_id):
    """Level 1 — crop away every watermark that sits in a border band. Pure
    CPU/PIL, no model, no GPU: this level is always available. ValueError when
    there is nothing to crop (the UI disables the button, this is the race)."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    # Hand-masked rows are level 2's, so they don't count as work for this level:
    # launching a crop that can only skip them would report "0 cropped" and read
    # as a broken button.
    total = (_clean_pool_query(bank_id)
             .filter(BankImage.watermark_regions.is_(None)).count())
    if not total:
        if _clean_pool_query(bank_id).count():
            raise ValueError('the flagged images all carry a hand-edited mask — '
                             'use 🧽 Inpaint, which repaints the zones you drew')
        raise ValueError('no flagged image to clean — run the watermark scan first')
    return bank_jobs.start(app, bank_id, 'watermark_crop',
                           _watermark_crop_job(bank_id), total=total)


def _watermark_crop_job(bank_id):
    def run(job):
        from .face_dataset_service import _apply_watermark_crop, _route_watermark
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = _clean_pool_query(bank_id).order_by(BankImage.id.asc()).all()
        bank_jobs.progress(job, done=0, total=len(rows), detail='auto-crop')
        cropped = left = failed = 0
        try:
            for row in rows:
                if bank_jobs.cancelled(job):
                    break
                boxes, manual, _problem = _clean_regions(row)
                if manual:
                    # A hand mask is level 2's material, mask emptied or not. It
                    # can carry several zones and zones on the subject; cropping
                    # cannot express either, and quietly cropping the detector's
                    # old box would clean pixels the user did NOT point at.
                    left += 1
                    bank_jobs.bump(job)
                    continue
                bbox = boxes[0] if boxes else None
                src, width, height = _source_size(bank, row)
                if not bbox or not src:
                    failed += 1
                    bank_jobs.bump(job)
                    continue
                route, box = _route_watermark(bbox, width, height, allow_crop=True)
                if route != 'crop':
                    left += 1              # level 2's job — stays 'detected'
                    bank_jobs.bump(job)
                    continue
                dst = _stage_clean_copy(bank_id, row, src)
                if _apply_watermark_crop(str(dst), box):
                    row.watermark_state = 'cleaned'
                    row.watermark_clean_method = 'crop'
                    drop_derived(bank_id, row.id)
                    cropped += 1
                else:
                    _discard_clean_blob(bank_id, row)
                    failed += 1
                if (cropped + failed) % 25 == 0:
                    db.session.commit()
                bank_jobs.bump(job)
        finally:
            db.session.commit()
        if bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail=f'cancelled — {cropped} cropped so far')
            return
        detail = f'done — {cropped} cropped, {left} left for inpainting'
        if failed:
            detail += f', {failed} unreadable'
        bank_jobs.progress(job, detail=detail)
    return run


def _watermark_inpaint_prereq(method) -> str | None:
    """Why level 2 can't run right now, or None. Actionable text — an unavailable
    engine must say what to install, never fail silently mid-pass."""
    from . import watermark_klein, watermark_lama
    if method == 'klein':
        if not watermark_klein.is_available():
            return ('Klein inpainting needs ComfyUI running and the Klein weights '
                    '(Setup ▸ Generation models)')
        return None
    if not watermark_lama.is_available():
        return 'LaMa inpainting is not installed (Setup ▸ Quality tools)'
    return None


def start_watermark_inpaint(app, user_id, bank_id, method='auto'):
    """Level 2 — repaint what is STILL flagged after the crop level.
    ``method``: 'auto'/'lama' (LaMa, non-generative, small off-centre marks; marks
    on the subject stay flagged for manual review) or 'klein' (masked Flux.2 Klein
    through ComfyUI, which also handles on-subject marks). RuntimeError (→ 503) on
    a missing engine or a busy GPU, ValueError (→ 400) on a bad method / empty pool."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    method = (method or 'auto').lower()
    if method not in ('auto', 'lama', 'klein'):
        raise ValueError("method must be 'auto', 'lama' or 'klein'")
    total = _clean_pool_query(bank_id).count()
    if not total:
        raise ValueError('nothing left to inpaint — every flagged image is handled')
    problem = _watermark_inpaint_prereq(method)
    if problem:
        raise RuntimeError(problem)
    reason = _gpu_busy_reason()
    if reason:
        raise RuntimeError(reason)
    return bank_jobs.start(app, bank_id, 'watermark_inpaint',
                           _watermark_inpaint_job(bank_id, method), total=total)


def _watermark_inpaint_job(bank_id, method):
    def run(job):
        from contextlib import nullcontext
        from . import watermark_klein, watermark_lama
        from .face_dataset_service import _clean_inpaint_engine, _route_watermark
        from ..gpu_window import gpu_exclusive_vision_window
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = _clean_pool_query(bank_id).order_by(BankImage.id.asc()).all()
        bank_jobs.progress(job, done=0, total=len(rows), detail='inpainting')
        counts = {'inpainted': 0, 'klein': 0, 'review': 0, 'failed': 0,
                  'skipped': 0, 'empty': 0}
        error = None
        lama_ok = watermark_lama.is_available()
        klein_ok = method == 'klein' and watermark_klein.is_available()
        # LaMa on the GPU pauses ComfyUI through the exclusive vision window;
        # Klein must NOT take that window — ComfyUI owns the GPU there and
        # holding it would deadlock its worker (same split as the dataset route).
        device = 'cpu' if method == 'klein' else watermark_lama.resolve_device()
        pending = []            # (row, dst_path, [bbox]) for the single LaMa batch
        window = (gpu_exclusive_vision_window(flag_ttl=1800)
                  if device == 'cuda' else nullcontext())
        try:
            with window:
                for row in rows:
                    if bank_jobs.cancelled(job):
                        break
                    boxes, manual, problem = _clean_regions(row)
                    src, width, height = _source_size(bank, row)
                    if problem or not src:
                        counts['failed'] += 1
                        error = error or (
                            {'kind': 'failed', 'detail': problem} if problem else None)
                        bank_jobs.bump(job)
                        continue
                    if manual and not boxes:
                        # The user deleted every zone. That is an ANSWER, not a
                        # missing value: repaint nothing, and never fall back to
                        # the detector's box. The row stays flagged (and visible)
                        # so it can be masked again or dismissed.
                        counts['empty'] += 1
                        bank_jobs.bump(job)
                        continue
                    if not boxes:
                        counts['failed'] += 1
                        bank_jobs.bump(job)
                        continue
                    if manual:
                        # Hand-drawn zones bypass the router entirely — same rule
                        # as the dataset: what the user drew IS the decision, and
                        # every zone is repainted with the selected engine.
                        engine = 'klein' if method == 'klein' else 'lama'
                    else:
                        # allow_crop=False: level 2 REPAINTS what is left, including a
                        # border mark the user chose not to crop.
                        route, _box = _route_watermark(boxes[0], width, height,
                                                       allow_crop=False)
                        engine = _clean_inpaint_engine(route, method)
                    if engine == 'review':
                        counts['review'] += 1       # stays flagged, needs Klein or a human
                        bank_jobs.bump(job)
                        continue
                    if (engine == 'klein' and not klein_ok) or \
                       (engine == 'lama' and not lama_ok):
                        counts['skipped'] += 1      # engine gone since launch
                        bank_jobs.bump(job)
                        continue
                    dst = _stage_clean_copy(bank_id, row, src)
                    if engine == 'klein':
                        ok, err = watermark_klein.inpaint_watermark_klein(
                            bank.user_id, str(dst), [list(b) for b in boxes])
                        if ok:
                            row.watermark_state = 'cleaned'
                            row.watermark_clean_method = 'klein'
                            drop_derived(bank_id, row.id)
                            counts['klein'] += 1
                        else:
                            _discard_clean_blob(bank_id, row)
                            counts['skipped' if (err or {}).get('kind') == 'unavailable'
                                   else 'failed'] += 1
                            error = err or error
                        db.session.commit()
                        bank_jobs.bump(job)
                        continue
                    pending.append((row, dst, [list(b) for b in boxes]))
                    bank_jobs.bump(job)
                if pending and bank_jobs.cancelled(job):
                    # Stop means stop: the staged copies of rows we never got to
                    # repaint are thrown away rather than running a long batch
                    # after the user asked out (they stay 'detected', retryable).
                    for row, _dst, _boxes in pending:
                        _discard_clean_blob(bank_id, row)
                    pending = []
                if pending:
                    results = watermark_lama.inpaint_batch(
                        [{'image_path': str(dst), 'bboxes': boxes}
                         for _row, dst, boxes in pending], device=device)
                    for row, dst, _boxes in pending:
                        ok, err = results.get(str(dst), (
                            False, {'kind': 'failed', 'detail': 'missing inpaint result'}))
                        if ok:
                            row.watermark_state = 'cleaned'
                            row.watermark_clean_method = 'lama'
                            drop_derived(bank_id, row.id)
                            counts['inpainted'] += 1
                        else:
                            _discard_clean_blob(bank_id, row)
                            counts['skipped' if (err or {}).get('kind') == 'unavailable'
                                   else 'failed'] += 1
                            error = err or error
        finally:
            db.session.commit()
        done = counts['inpainted'] + counts['klein']
        if bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail=f'cancelled — {done} inpainted so far')
            return
        detail = f'done — {done} inpainted'
        if counts['review']:
            detail += (f", {counts['review']} on the subject "
                       '(switch the engine to Klein to repaint those)')
        if counts['empty']:
            # NOT lumped in with 'review': "on the subject" would send the user to
            # Klein for images where the honest answer is "your mask is empty".
            detail += (f", {counts['empty']} with an empty mask (draw a zone in "
                       '▶ Review, or dismiss them)')
        if counts['skipped']:
            detail += f", {counts['skipped']} skipped (engine unavailable)"
        if counts['failed']:
            detail += f", {counts['failed']} failed"
            if error and error.get('detail'):
                detail += f" — {error['detail']}"
        bank_jobs.progress(job, detail=detail)
    return run


def set_watermark_regions(user_id, bank_id, image_id, regions) -> dict | None:
    """Replace one flagged image's hand-drawn watermark mask (reported missing in
    the Bank by Qeeyana on Reddit — the Dataset had it, the Bank did not).

    ``regions`` is None (drop the override, go back to the detected box) or a list
    of normalized boxes — validated by the DATASET's validator, deliberately: one
    definition of a legal mask means the two lanes cannot drift apart. Returns the
    same payload shape the dataset route returns, None when the bank/image is
    unknown, ValueError on an illegal mask and RuntimeError when the image is no
    longer flagged (already cleaned/dismissed — an edit there would be a no-op).

    The mask is NOT cleared when a clean succeeds, unlike the dataset: the Bank's
    ↩ Undo is a first-class action (it only deletes our own blob), and handing an
    image back with its hand-drawn zones erased would mean redrawing them."""
    if not get_bank(user_id, bank_id):
        return None
    owned = BankImage.query.filter_by(id=image_id, bank_id=bank_id)
    row = owned.one_or_none()
    if not row:
        return None
    if row.watermark_state != 'detected':
        raise RuntimeError('this image is no longer flagged — nothing to mask')
    normalized = normalize_watermark_regions(regions)
    import json as _json
    stored = _json.dumps(normalized) if normalized is not None else None
    updated = (BankImage.query
               .filter_by(id=row.id, bank_id=bank_id, watermark_state='detected')
               .update({'watermark_regions': stored}, synchronize_session=False))
    if updated != 1:
        db.session.rollback()
        if owned.one_or_none() is None:
            return None
        raise RuntimeError('this image is no longer flagged — nothing to mask')
    db.session.commit()
    return _watermark_regions_payload(row)


def undo_watermark_clean(user_id, bank_id, image_ids=None) -> int:
    """Throw away cleaned versions and re-flag the images. The source was never
    modified, so undoing is just deleting our own blob — which is exactly what
    makes running both levels risk-free. ``image_ids`` empty = every cleaned
    image of the bank. Returns how many rows were restored."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    q = BankImage.query.filter_by(bank_id=bank_id, watermark_state='cleaned')
    if image_ids:
        ids = [int(i) for i in image_ids]
        q = q.filter(BankImage.id.in_(ids[:_SQL_IN_CHUNK]))
    rows = q.all()
    for row in rows:
        _discard_clean_blob(bank_id, row)
        # Back to 'detected' with its bbox intact, so it re-enters both levels
        # (e.g. to retry with the other engine).
        row.watermark_state = 'detected'
    if rows:
        db.session.commit()
    return len(rows)


def dismiss_watermarks(user_id, bank_id, image_ids) -> int:
    """Rule a flag a FALSE positive: 'detected' → 'dismissed'. Those images leave
    both cleaning levels and are never re-flagged by a later scan — without this,
    level 2 would happily repaint a legitimate logo on a T-shirt. Mirrors the
    dataset's dismiss_watermarks. Returns how many rows changed."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    ids = [int(i) for i in (image_ids or [])]
    if not ids:
        return 0
    rows = (BankImage.query
            .filter_by(bank_id=bank_id, watermark_state='detected')
            .filter(BankImage.id.in_(ids[:_SQL_IN_CHUNK])).all())
    for row in rows:
        row.watermark_state = 'dismissed'
    if rows:
        db.session.commit()
    return len(rows)


def watermark_levels(user_id, bank_id) -> dict | None:
    """Where each cleaning level stands — the numbers the UI shows per level.
    None when the bank is gone."""
    if not get_bank(user_id, bank_id):
        return None
    from .face_dataset_service import _route_watermark
    bank = db.session.get(ImageBank, bank_id)
    base = BankImage.query.filter_by(bank_id=bank_id)
    flagged = croppable = hand_masked = empty_masks = 0
    for row in _clean_pool_query(bank_id).all():
        flagged += 1
        boxes, manual, _problem = _clean_regions(row)
        if manual:
            # A hand mask never routes to the crop level (see _watermark_crop_job),
            # so it must not be counted as croppable — the ✂ button would offer
            # work it will then skip.
            hand_masked += 1
            if not boxes:
                empty_masks += 1
            continue
        # Dimensions from the scan when we have them (this runs over every flagged
        # image of a possibly huge bank — no file is opened unless it has to be).
        width, height = row.width, row.height
        if not (width and height):
            _path, width, height = _source_size(bank, row)
        if boxes and width and _route_watermark(boxes[0], width, height,
                                                allow_crop=True)[0] == 'crop':
            croppable += 1
    return {
        'scanned': base.filter(BankImage.watermark_state.isnot(None)).count(),
        # What a plain re-run would still look at. Detection resumes where it
        # stopped (the pass commits every 25 rows), but nothing said so: a
        # progress bar that restarts at 0 each run reads as "it started over
        # and is re-analysing what I already did". This is the number that
        # answers that, so the panel can say "N left to scan" out loud.
        'unscanned': _watermark_scan_query(bank_id, rescan=False).count(),
        'flagged': flagged,
        'croppable': croppable,
        'inpaintable': flagged - croppable,
        # Flagged images whose mask the user drew by hand, and how many of those
        # were deliberately emptied. Both are surfaced: an empty mask repaints
        # nothing, and a level that silently skips images is how a user ends up
        # believing a watermark was removed when it was not.
        'hand_masked': hand_masked,
        'empty_masks': empty_masks,
        'cropped': base.filter_by(watermark_clean_method='crop').count(),
        'inpainted': base.filter(
            BankImage.watermark_clean_method.in_(('lama', 'klein'))).count(),
        'dismissed': base.filter_by(watermark_state='dismissed').count(),
        'needs_rescan': _needs_rescan_count(bank_id),
        # A few already-cleaned ids so the panel can offer a before/after strip
        # (each image is served cleaned, or original with ?original=1) without a
        # second endpoint just to list them.
        'cleaned_sample': [r.id for r in
                           base.filter(BankImage.watermark_clean_method.isnot(None))
                           .order_by(BankImage.id.asc()).limit(8).all()],
    }


# --- framing pass (reuses the dataset face/bust/body/back classifier) -------
def start_framing(app, user_id, bank_id, rescan=False):
    """Classify every non-rejected image by SHOT TYPE (face / bust / body / back),
    reusing the SAME Qwen3-VL classifier the datasets use (CLASSIFY_PROMPT). Feeds
    the 📐 Framing filter chips and the coverage advice. Needs the vision model
    pulled; serialized against training/vision like the watermark pass (503 when
    the GPU is held). ``rescan`` re-classifies rows that already have a framing."""
    from ..capabilities import probe_ollama_model
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    # Occupancy BEFORE the model probe, and the order is the whole point: a bank
    # that is already busy is busy whether or not Ollama answers. Probing first
    # made a busy bank report "the vision model is not available" whenever the
    # model happened to be unreachable -- the wrong reason, sending the user to
    # fix an unrelated thing while the real answer was "wait, a pass is running".
    # It also made the refusal depend on an EXTERNAL SERVICE, which is why CI
    # (no Ollama) failed a release on it while two agents read it as a flake.
    if bank_jobs.running(bank_id):
        raise bank_jobs.BankJobBusy((bank_jobs.get(bank_id) or {}).get('kind') or 'background')
    if not probe_ollama_model().get('ok'):
        raise RuntimeError('the vision model is not available '
                           '(Settings ▸ Captioning & quality)')
    reason = _gpu_busy_reason()
    if reason:
        raise RuntimeError(reason)
    q = BankImage.query.filter_by(bank_id=bank_id).filter(BankImage.status != 'reject')
    if not rescan:
        q = q.filter(BankImage.framing.is_(None))
    return bank_jobs.start(app, bank_id, 'framing',
                           _framing_job(bank_id, rescan), total=q.count())


def _framing_job(bank_id, rescan):
    def run(job):
        from .face_dataset_service import CLASSIFY_PROMPT, _parse_classify
        from .vision_ollama import describe_image_ollama, unload_vision_model
        from .vision_pool import map_vision
        from ..gpu_window import gpu_exclusive_vision_window
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        q = (BankImage.query.filter_by(bank_id=bank_id)
             .filter(BankImage.status != 'reject'))
        if not rescan:
            q = q.filter(BankImage.framing.is_(None))
        rows = q.order_by(BankImage.id.asc()).all()
        bank_jobs.progress(job, done=0, total=len(rows), detail='framing')
        if not rows:
            return
        classified = errors = 0

        def prepared():
            """Path resolution reads the row, so it belongs on the job's own
            thread — pulled one image per free slot in the pool."""
            for row in rows:
                yield row, abs_image_path(bank, row)

        def ask(item):
            """WORKER thread: file + network only, no session. None means the
            file is gone, as opposed to '' meaning the model said nothing."""
            _row, path = item
            if not path or not os.path.isfile(path):
                return None
            with open(path, 'rb') as fh:
                return describe_image_ollama(
                    fh.read(), CLASSIFY_PROMPT, num_predict=400,
                    prefer_json=True, fmt='json', keep_alive='5m')

        with gpu_exclusive_vision_window(flag_ttl=1800):
            try:
                # The calls overlap (see vision_pool); every write below still
                # happens here, on this one thread.
                for (row, _path), raw, error in map_vision(
                        prepared(), ask,
                        should_cancel=lambda: bank_jobs.cancelled(job)):
                    if error is not None:  # one bad file never sinks the pass
                        errors += 1
                    elif raw is None:      # file gone: leave the row as it was
                        pass
                    # Empty output = Ollama unreachable, NOT "unknown": leave the
                    # framing NULL so a retry can finish it (same reasoning as the
                    # watermark/dataset classifier), never mislabel everything.
                    elif not raw.strip():
                        pass
                    else:
                        framing, _label = _parse_classify(raw)
                        row.framing = framing        # face|bust|body|back|unknown
                        classified += 1
                        if classified % 25 == 0:
                            db.session.commit()
                    bank_jobs.bump(job)
            finally:
                db.session.commit()
                unload_vision_model()  # hand the VRAM back to ComfyUI
        if bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail=f'cancelled — {classified} classified so far')
            return
        detail = f'done — {classified} classified'
        if errors:
            detail += f', {errors} unreadable'
        bank_jobs.progress(job, detail=detail)
    return run


# --- caption pass (reuses the dataset caption engines) ----------------------
def start_caption(app, user_id, bank_id, ids=None, force=False, vocabulary=None):
    """Launch the caption pass over a selection (``ids``) or, when empty, every
    non-rejected readable image. Reuses the dataset caption engines (JoyCaption /
    Ollama per Settings) through a dataset-free descriptive brick; the captions
    double as the bank's search text and ride along on promotion. Serialized
    against training/vision like the score/watermark passes (503 when the GPU is
    held). BankJobBusy when a job is already live, ValueError on a bad bank/config.

    ``vocabulary`` picks a caption REGISTER (one of face_dataset_service's
    CAPTION_VOCABULARIES: 'explicit' | 'clinical' | 'safe') — the SAME lane the
    dataset caption uses, appended as an instruction. Explicit only spells sexual
    content out when the backend runs an abliterated Ollama model; the choice rides
    per-call (the UI passes it), so a call WITHOUT it is byte-identical to before
    (no instruction appended). Richer captions also mean richer 🔍 search text."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    from .face_dataset_service import CAPTION_VOCABULARIES
    vocab = (vocabulary or '').strip().lower() or None
    if vocab and vocab not in CAPTION_VOCABULARIES:
        raise ValueError(f'invalid caption vocabulary: {vocab}')
    backend = (cfg.get('captioning.backend') or 'auto').lower()
    if backend == 'none':
        raise ValueError('no captioning backend configured (Settings ▸ Captioning & quality)')
    reason = _gpu_busy_reason()
    if reason:
        raise RuntimeError(reason)
    ids = [int(i) for i in ids] if ids else None
    q = BankImage.query.filter_by(bank_id=bank_id).filter(BankImage.status != 'reject')
    if ids is not None:
        q = q.filter(BankImage.id.in_(ids[:_SQL_IN_CHUNK]))
    if not force:
        q = q.filter(or_(BankImage.caption.is_(None), BankImage.caption == ''))
    total = q.count()
    return bank_jobs.start(app, bank_id, 'caption',
                           _caption_job(bank_id, ids, force, vocab), total=total)


def _caption_job(bank_id, ids, force, vocabulary=None):
    def run(job):
        from .face_dataset_service import caption_paths, vocabulary_instruction
        from ..gpu_window import gpu_exclusive_vision_window
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        q = BankImage.query.filter_by(bank_id=bank_id).filter(BankImage.status != 'reject')
        if ids is not None:
            rows = []
            for i0 in range(0, len(ids), _SQL_IN_CHUNK):
                rows.extend(q.filter(BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all())
            rows.sort(key=lambda r: r.id)
        else:
            rows = q.order_by(BankImage.id.asc()).all()
        if not force:
            rows = [r for r in rows if not (r.caption or '').strip()]
        by_path = {}
        for r in rows:
            p = abs_image_path(bank, r)
            if p and os.path.isfile(p):
                by_path[p] = r.id
        paths = list(by_path)
        bank_jobs.progress(job, done=0, total=len(paths), detail='captioning')
        if not paths:
            return
        captioned = 0

        def _on_caption(path, caption):
            nonlocal captioned
            row = db.session.get(BankImage, by_path.get(path))
            if row is not None:
                row.caption = caption
                db.session.commit()
                captioned += 1

        # GPU-exclusive for the whole pass, exactly like the score/watermark passes:
        # frees ComfyUI VRAM and blocks a training start for the duration.
        # The vocabulary register rides in as the SAME appended instruction the
        # dataset pass uses (None when unset → byte-identical to the plain pass).
        extra = vocabulary_instruction(vocabulary)
        with gpu_exclusive_vision_window(flag_ttl=1800):
            caption_paths(
                paths,
                extra_instructions=extra,
                should_cancel=lambda: bank_jobs.cancelled(job),
                on_caption=_on_caption,
                progress=lambda d, t: bank_jobs.progress(job, done=d, total=t))
        if bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail=f'cancelled — {captioned} captioned so far')
            return
        bank_jobs.progress(job, detail=f'done — {captioned} captioned')
    return run


# --- "Launch all" pipeline --------------------------------------------------
# The overnight funnel: the user configures it once, hits Launch all, and comes
# back to a triaged, optionally pre-captioned bank. It chains the EXISTING passes
# in the validated order. Each pass already filters status != 'reject', so
# running auto-reject BEFORE the heavy passes means score/watermark/person only
# ever touch the SURVIVORS — the costly work never pays for images we just
# dropped (the deliberate cost/quality trade-off: duplicate "keep best" therefore
# ranks on sharpness/size, not the aesthetic score that isn't computed yet).
PIPELINE_STEPS = ('scan', 'auto_reject', 'score', 'semantic_dedup', 'watermark',
                  'faces', 'framing', 'caption')
# Auto-reject inside the pipeline runs right after the quality scan, so it can
# only act on the CPU-scan flags (and duplicates). The score-derived flags
# (low_aesthetic/nsfw/watermark) have no data yet at that point.
#
# NOT every quality flag, though. `soft_detail` and `bars` are excluded ON
# PURPOSE — they are provenance HINTS, not verdicts, and their own documentation
# says so: a crisp watermark rescues an enlargement's detail ratio while a
# motion-blurred native shot sinks it, and `bars` fires on any dark-themed
# screenshot. The standalone 🧹 Auto-reject button still offers them, because
# there a human is looking at the flagged count, can undo on the spot, and the
# hint under the checkbox says "check before mass-rejecting". The pipeline is the
# opposite situation: unattended, and auto-reject runs FIRST, so anything it
# drops never reaches the score / watermark / caption passes at all — the mistake
# becomes invisible instead of reviewable. Offering a non-verdict as an overnight
# bulk rejection contradicts the measurement it is built on.
# The pipeline UI never offered these two; this makes the API agree with it.
_PIPELINE_EXCLUDED_REJECT_FLAGS = ('soft_detail', 'bars')
PIPELINE_REJECT_FLAGS = tuple(f for f in _QUALITY_FLAGS
                              if f not in _PIPELINE_EXCLUDED_REJECT_FLAGS)


def _sanitize_pipeline_steps(steps) -> list:
    """Keep only known steps, in the canonical pipeline order (the client can't
    reorder or invent a pass)."""
    want = set(steps or [])
    return [s for s in PIPELINE_STEPS if s in want]


def _score_prereq() -> str | None:
    from ..capabilities import probe_bank_scoring
    if not probe_bank_scoring().get('ok'):
        return 'bank scoring extra not installed (Setup ▸ Quality tools)'
    return None


def _watermark_prereq() -> str | None:
    from ..capabilities import probe_ollama_model
    if not probe_ollama_model().get('ok'):
        return 'vision model not available (Settings ▸ Captioning & quality)'
    return None


def _faces_prereq() -> str | None:
    from .face_similarity import is_available
    if not is_available():
        return 'face scoring extra not installed (Setup ▸ Quality tools)'
    return None


def _framing_prereq() -> str | None:
    from ..capabilities import probe_ollama_model
    if not probe_ollama_model().get('ok'):
        return 'vision model not available (Settings ▸ Captioning & quality)'
    return None


def _caption_prereq() -> str | None:
    if (cfg.get('captioning.backend') or 'auto').lower() == 'none':
        return 'no captioning backend configured (Settings ▸ Captioning & quality)'
    return None


def start_pipeline(app, user_id, bank_id, steps=None, reject_flags=None,
                   resolve_dups=False):
    """Launch the chained triage pipeline. ``steps`` selects which passes run
    (canonical order enforced); ``reject_flags`` + ``resolve_dups`` configure the
    auto-reject step. One background job like every other pass — BankJobBusy when
    one is already live, ValueError on a bad bank / empty step list."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    steps = _sanitize_pipeline_steps(steps)
    if not steps:
        raise ValueError('no pipeline steps selected')
    reject_flags = [f for f in (reject_flags or []) if f in PIPELINE_REJECT_FLAGS]
    return bank_jobs.start(
        app, bank_id, 'pipeline',
        _pipeline_job(user_id, bank_id, steps, reject_flags, bool(resolve_dups)),
        total=0)


def _bank_counts(bank_id) -> dict:
    """Live headline counts used for the per-step tallies and the final report."""
    base = BankImage.query.filter_by(bank_id=bank_id)
    dup_groups = (db.session.query(BankImage.dup_group)
                  .filter(BankImage.bank_id == bank_id,
                          BankImage.dup_group.isnot(None))
                  .distinct().count())
    style_groups = (db.session.query(BankImage.style_cluster)
                    .filter(BankImage.bank_id == bank_id,
                            BankImage.style_cluster.isnot(None))
                    .distinct().count())
    semantic_groups = (db.session.query(BankImage.semantic_dup_group)
                       .filter(BankImage.bank_id == bank_id,
                               BankImage.semantic_dup_group.isnot(None))
                       .distinct().count())
    person_groups = (db.session.query(BankImage.face_cluster)
                     .filter(BankImage.bank_id == bank_id,
                             BankImage.face_cluster.isnot(None))
                     .distinct().count())
    return {
        'total': base.count(),
        'scanned': base.filter(BankImage.quality_state.isnot(None)).count(),
        'reject': base.filter_by(status='reject').count(),
        'scored': base.filter(or_(BankImage.aesthetic_score.isnot(None),
                                  BankImage.nsfw_score.isnot(None))).count(),
        'watermark_detected': base.filter(BankImage.watermark_state == 'detected').count(),
        'framing_classified': base.filter(BankImage.framing.isnot(None)).count(),
        'captioned': base.filter(and_(BankImage.caption.isnot(None),
                                       BankImage.caption != '')).count(),
        'dup_groups': dup_groups,
        'style_groups': style_groups,
        'semantic_groups': semantic_groups,
        'person_groups': person_groups,
    }


def _pipeline_job(user_id, bank_id, steps, reject_flags, resolve_dups):
    def run(job):
        import json as _json
        import time as _time
        from ..gpu_window import GpuBusyError
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        results = []
        pipe = {'steps': list(steps), 'total_steps': len(steps),
                'index': 0, 'current': steps[0], 'results': results}

        def _sync(current=None, index=None):
            if index is not None:
                pipe['index'] = index
            if current is not None:
                pipe['current'] = current
            pipe['results'] = list(results)
            bank_jobs.set_pipeline(job, pipe)

        _sync()
        # Each entry closes with a status: 'done' | 'skipped' (reason) | 'error'.
        for i, step in enumerate(steps):
            if bank_jobs.cancelled(job):
                break
            _sync(current=step, index=i)
            bank_jobs.progress(job, done=0, total=0,
                               detail=f'step {i + 1}/{len(steps)}: {step}')
            entry = {'step': step, 'status': 'done', 'reason': None,
                     'detail': None, 'counts': {}}
            try:
                _run_pipeline_step(job, user_id, bank_id, step,
                                   reject_flags, resolve_dups, entry)
            except GpuBusyError as e:
                # A vision/training job grabbed the GPU mid-pipeline — skip this
                # pass and keep going (never wake the user for a transient clash).
                entry['status'] = 'skipped'
                entry['reason'] = f'GPU busy — {e}'
            except Exception as e:  # noqa: BLE001 — one bad pass never sinks the rest
                entry['status'] = 'error'
                entry['reason'] = f'{type(e).__name__}: {e}'
                db.session.rollback()
            # A step that executed stays 'done' even if a cancel landed at its
            # tail (its inner run already returned early); only steps we never
            # reach are recorded as cancelled, below.
            results.append(entry)
            _sync()

        cancelled = bank_jobs.cancelled(job)
        # Any step never reached (cancel, or a hard earlier break) is recorded so
        # the morning-after report has a row for every requested pass.
        reached = {e['step'] for e in results}
        for step in steps:
            if step not in reached:
                results.append({'step': step, 'status': 'cancelled' if cancelled
                                else 'skipped',
                                'reason': 'cancelled before it ran' if cancelled
                                else 'not reached', 'detail': None, 'counts': {}})
        _sync()

        report = {
            'started_at': job.get('started_at'),
            'finished_at': _time.time(),
            'cancelled': cancelled,
            'requested_steps': list(steps),
            'reject_flags': list(reject_flags),
            'resolve_dups': resolve_dups,
            'steps': results,
            'counts': _bank_counts(bank_id),
        }
        bank = db.session.get(ImageBank, bank_id)
        if bank is not None:
            bank.pipeline_report = _json.dumps(report)
            db.session.commit()
        done_n = sum(1 for e in results if e['status'] == 'done')
        skipped_n = sum(1 for e in results if e['status'] in ('skipped', 'cancelled'))
        err_n = sum(1 for e in results if e['status'] == 'error')
        tail = f'done — {done_n}/{len(steps)} steps ran'
        if skipped_n:
            tail += f', {skipped_n} skipped'
        if err_n:
            tail += f', {err_n} errored'
        if cancelled:
            tail = f'cancelled — {done_n}/{len(steps)} steps ran'
        bank_jobs.progress(job, detail=tail)
    return run


def _run_pipeline_step(job, user_id, bank_id, step, reject_flags, resolve_dups, entry):
    """Run ONE pipeline pass into ``entry``, reusing the standalone pass work.
    Prerequisite missing → entry marked 'skipped' with a reason, pipeline
    continues. Reuses each pass's inner ``run(job)`` so progress, cancellation
    and the GPU-exclusive window behave exactly as the standalone buttons."""
    if step == 'scan':
        _scan_job(bank_id, rescan=False)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'scanned': c['scanned'], 'dup_groups': c['dup_groups']}
        entry['detail'] = (job.get('detail')
                           or f"scanned {c['scanned']}, {c['dup_groups']} duplicate group(s)")
        return
    if step == 'auto_reject':
        # ONE undo offer for the whole step, not one per sub-pass: the user fired
        # "Launch all", so the unit they would take back is the auto-reject, not
        # its second half. The later steps only ADD analysis columns, so undoing
        # this one leaves them consistent.
        snap = bank_undo.Snapshot('Launch all — auto-reject')
        rejected = (apply_flags(user_id, bank_id, reject_flags, snapshot=snap)
                    if reject_flags else {})
        dup_rejected = 0
        if resolve_dups:
            dup_rejected = resolve_dups_keep_best(user_id, bank_id, snapshot=snap)
        snap.commit(bank_id)
        n = sum(rejected.values()) + dup_rejected
        entry['counts'] = {'rejected': n, 'by_flag': rejected,
                           'duplicates': dup_rejected}
        parts = [f'{v} {k}' for k, v in rejected.items() if v]
        if dup_rejected:
            parts.append(f'{dup_rejected} duplicate')
        entry['detail'] = (f"rejected {n} image(s)"
                           + (f" ({', '.join(parts)})" if parts else '')
                           + ' — manual ✓/✕ untouched')
        return
    if step == 'score':
        reason = _score_prereq() or _gpu_busy_reason()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        _score_job(bank_id)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'scored': c['scored'], 'style_groups': c['style_groups']}
        entry['detail'] = job.get('detail') or f"scored {c['scored']} image(s)"
        return
    if step == 'semantic_dedup':
        # Runs right after Score, reusing its cached embeddings (no GPU). Groups
        # crops/variants for review; resolution stays a UI action (keep best/first
        # /manual) — near-dups are fuzzier than exact dHash, so the overnight run
        # surfaces them rather than auto-rejecting. Skipped-with-reason (never a
        # mute ✗) when Score produced no embeddings.
        n = rebuild_semantic_dup_groups(bank_id)
        if n is None:
            entry['status'], entry['reason'] = 'skipped', 'run ✨ Score first — no embeddings'
            return
        entry['counts'] = {'semantic_groups': n}
        entry['detail'] = f'{n} semantic near-duplicate group(s) to review'
        return
    if step == 'watermark':
        reason = _watermark_prereq() or _gpu_busy_reason()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        _watermark_job(bank_id, rescan=False)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'watermarks': c['watermark_detected']}
        entry['detail'] = job.get('detail') or f"{c['watermark_detected']} with a watermark"
        return
    if step == 'faces':
        reason = _faces_prereq()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        _faces_job(bank_id)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'person_groups': c['person_groups']}
        entry['detail'] = job.get('detail') or f"{c['person_groups']} person cluster(s)"
        return
    if step == 'framing':
        reason = _framing_prereq() or _gpu_busy_reason()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        _framing_job(bank_id, rescan=False)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'framing_classified': c['framing_classified']}
        entry['detail'] = job.get('detail') or f"{c['framing_classified']} classified by framing"
        return
    if step == 'caption':
        reason = _caption_prereq() or _gpu_busy_reason()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        before = _bank_counts(bank_id)['captioned']
        _caption_job(bank_id, None, False)(job)
        after = _bank_counts(bank_id)['captioned']
        entry['counts'] = {'captioned': max(0, after - before), 'total_captioned': after}
        entry['detail'] = job.get('detail') or f"{after} captioned"
        return
    entry['status'], entry['reason'] = 'skipped', 'unknown step'


def resolve_dups_keep_best(user_id, bank_id, snapshot=None) -> int:
    """Auto-resolve every unresolved duplicate group keeping the best member,
    for the pipeline's auto-reject step. Returns the number REJECTED."""
    out = resolve_dups(user_id, bank_id, strategy='best', snapshot=snapshot)
    return out.get('rejected', 0)


# --- subfolders (scoping facet) ---------------------------------------------
def subfolders_payload(user_id, bank_id) -> dict | None:
    """Top-level subfolders of the bank's source folder with image counts, for
    the scoping picker. Computed once on open (not polled) — a Telegram export
    nests one folder per chat/date. '' = files at the bank root."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    from collections import Counter
    counts: Counter = Counter()
    for (rel,) in (db.session.query(BankImage.relpath)
                   .filter(BankImage.bank_id == bank_id).all()):
        counts[_subfolder_of(rel)] += 1
    items = [{'name': name, 'count': n}
             for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {'subfolders': items, 'total': sum(counts.values())}


# --- coverage advice (idea by @antonp) --------------------------------------
# A read-only ADVICE panel: from what you'd actually train on (the kept set, or
# every non-rejected image before anything is kept), it says what leans and what
# is thin for a good LoRA — purely from data the passes already computed (framing,
# person clusters, style clusters, resolution). It NEVER selects or rejects; it
# only phrases honest, non-alarmist sentences. Everything here is pure DB math,
# zero GPU — the framing part just needs the 📐 Framing pass to have run.
def _pct(n, total) -> int:
    return int(round(100 * n / total)) if total else 0


def _coverage_stats(bank_id) -> dict:
    """Everything the coverage panel needs, from the pool the user would train on:
    the KEPT images, or — before anything is kept — every non-rejected image (so
    the panel is useful from the first look). Pure aggregate SQL, no GPU."""
    base = BankImage.query.filter_by(bank_id=bank_id)
    kept_n = base.filter_by(status='keep').count()
    pool_is_kept = kept_n > 0
    crit = (BankImage.status == 'keep') if pool_is_kept \
        else (BankImage.status != 'reject')
    pool = base.filter(crit)
    total = pool.count()

    framing = _framing_counts(bank_id, extra_crit=crit)
    framing_known = sum(framing[k] for k in _FRAMINGS)
    framing_available = framing_known + framing['unknown'] > 0

    # Person clusters within the pool, biggest first (list of sizes).
    person_rows = (db.session.query(BankImage.face_cluster, func.count(BankImage.id))
                   .filter(BankImage.bank_id == bank_id, crit,
                           BankImage.face_cluster.isnot(None))
                   .group_by(BankImage.face_cluster)
                   .order_by(func.count(BankImage.id).desc()).all())
    person_sizes = [int(n) for _c, n in person_rows]
    style_rows = (db.session.query(BankImage.style_cluster, func.count(BankImage.id))
                  .filter(BankImage.bank_id == bank_id, crit,
                          BankImage.style_cluster.isnot(None))
                  .group_by(BankImage.style_cluster)
                  .order_by(func.count(BankImage.id).desc()).all())
    style_sizes = [int(n) for _c, n in style_rows]
    top_person_id = int(person_rows[0][0]) if person_rows else None

    # Resolution: how much of the pool is small (< 1 MP), where low-res caps detail.
    res_scanned = pool.filter(BankImage.width.isnot(None),
                              BankImage.height.isnot(None)).count()
    under_1mp = pool.filter(BankImage.width.isnot(None), BankImage.height.isnot(None),
                            BankImage.width * BankImage.height < 1_000_000).count()

    return {
        'pool': 'kept' if pool_is_kept else 'candidates',
        'total': total,
        'framing': framing, 'framing_known': framing_known,
        'framing_available': framing_available,
        'person': {'clusters': person_sizes, 'top_id': top_person_id,
                   'total': sum(person_sizes),
                   'singletons': sum(1 for n in person_sizes if n == 1)},
        'style': {'clusters': style_sizes, 'total': sum(style_sizes)},
        'resolution': {'scanned': res_scanned, 'under_1mp': under_1mp},
    }


def _coverage_advice(stats: dict) -> list:
    """Turn the coverage stats into a short list of honest, actionable sentences.
    Each is {'tone': 'warn'|'info', 'text': ...}. Pure function of ``stats`` (unit
    of the logic — deterministic on a known distribution). Never alarmist: a
    dominance reads as a QUESTION, a thin axis as a gentle 'add a few'."""
    total = stats['total']
    pool = stats['pool']
    noun = 'kept' if pool == 'kept' else 'candidate'
    out = []
    if total == 0:
        return [{'tone': 'info',
                 'text': 'Nothing to advise on yet — keep some images (or run a '
                         'pass) and the coverage read appears here.'}]

    # Size — most families want a couple dozen.
    if total < 20:
        out.append({'tone': 'warn',
                    'text': f'Only {total} {noun} — most LoRA families train more '
                            f'reliably with 20+ images.'})

    # Framing balance (needs the 📐 Framing pass).
    fr, known = stats['framing'], stats['framing_known']
    if not stats['framing_available']:
        out.append({'tone': 'info',
                    'text': 'Run the 📐 Framing pass to see how your face / bust / '
                            'body / back shots balance.'})
    elif known > 0:
        dom = max(_FRAMINGS, key=lambda k: fr[k])
        dom_share = fr[dom] / known
        thin = [k for k in _FRAMINGS if _pct(fr[k], known) < 10]
        if dom_share >= 0.55 and total >= 8:
            add = ' / '.join(k for k in ('body', 'back', 'bust', 'face')
                             if k in thin) or 'other angles'
            out.append({'tone': 'warn',
                        'text': f'{_pct(fr[dom], known)}% {dom} shots — add '
                                f'{add} for a fuller character.'})
        elif fr['back'] == 0 and known >= 10:
            out.append({'tone': 'info',
                        'text': 'No back views — a few help a character hold up '
                                'from behind (optional).'})

    # Person mix — a dominance is a question, not a verdict.
    ppl = stats['person']
    if ppl['total'] > 0 and ppl['clusters']:
        top = ppl['clusters'][0]
        if len(ppl['clusters']) >= 2 and top / ppl['total'] >= 0.5:
            out.append({'tone': 'info',
                        'text': f'Person #{ppl["top_id"]} is {_pct(top, ppl["total"])}% '
                                f'of the set — is this one subject, or a mix?'})
        if ppl['singletons'] >= 3 and total >= 10:
            out.append({'tone': 'info',
                        'text': f'{ppl["singletons"]} people appear only once — a '
                                f'character LoRA wants one consistent subject.'})

    # Style spread — only when it's genuinely mixed (no single dominant style).
    st = stats['style']
    if len(st['clusters']) >= 2 and st['total'] > 0 \
            and st['clusters'][0] / st['total'] < 0.7 and total >= 8:
        out.append({'tone': 'info',
                    'text': f'{len(st["clusters"])} visual styles in the set — '
                            f'mixing photoreal and illustration can dilute a LoRA.'})

    # Resolution — low-res caps the training resolution.
    res = stats['resolution']
    if res['scanned'] > 0:
        share = res['under_1mp'] / res['scanned']
        if share >= 0.3:
            out.append({'tone': 'info',
                        'text': f'{_pct(res["under_1mp"], res["scanned"])}% are under '
                                f'1 MP — low-res images cap the detail a LoRA can learn.'})

    if not out:
        out.append({'tone': 'info',
                    'text': 'Nothing stands out — your set looks reasonably balanced.'})
    # Warnings first so the panel reads worst-to-mildest.
    out.sort(key=lambda a: 0 if a['tone'] == 'warn' else 1)
    return out


def coverage(user_id, bank_id) -> dict | None:
    """The read-only coverage advice for the bank (idea by @antonp). Returns the
    distributions the panel renders plus the generated advice, or None if the bank
    is gone. Never mutates; pure DB, zero GPU."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    stats = _coverage_stats(bank_id)
    stats['advice'] = _coverage_advice(stats)
    return stats


# --- promotion --------------------------------------------------------------
def _promoted_here(dataset_id):
    """The bank_image ids the dataset STILL holds — one row per promoted image,
    written by the promotion itself (FaceDatasetImage.bank_image_id). Deleting
    the image in the dataset deletes the row, so this shrinks on its own."""
    return (db.session.query(FaceDatasetImage.bank_image_id)
            .filter(FaceDatasetImage.dataset_id == dataset_id,
                    FaceDatasetImage.bank_image_id.isnot(None)))


def _not_already_on(dataset_id):
    """The criterion 'this kept image is not already sitting on that dataset'.

    Measured, not remembered: an image counts as already there only while the
    dataset really holds a row pointing back at it. Delete that image in the
    dataset and the bank offers it again — the old one-way promoted_dataset_id
    flag never came back, so a bank could end up advertising nothing promotable
    into a dataset it had no image left in, which reads as "the bank lost my
    images".

    promoted_dataset_id survives as the LEGACY answer, for images promoted
    before the back-link existed: nothing writes it any more (a promotion clears
    it as it records the link), so it can only ever describe a pre-upgrade
    promotion, and it is dropped for good the next time that image is promoted.
    """
    return and_(BankImage.id.notin_(_promoted_here(dataset_id)),
                or_(BankImage.promoted_dataset_id.is_(None),
                    BankImage.promoted_dataset_id != dataset_id))


def _promotable_query(bank_id, dataset_id):
    """The KEPT images eligible to promote into ``dataset_id``. Per-target, not a
    global 'promoted anywhere' lock — an image promoted to dataset A stays
    promotable to B. (The dataset-side perceptual dedup on import is the real
    guard against genuine duplicates.)"""
    return (BankImage.query.filter_by(bank_id=bank_id, status='keep')
            .filter(_not_already_on(dataset_id)))


def promotable_count(user_id, bank_id, dataset_id) -> int | None:
    """How many kept images the 'promote all' path would send to ``dataset_id``
    right now — the honest number behind the modal's copy line. None = bank or
    dataset gone."""
    if not get_bank(user_id, bank_id):
        return None
    if not FaceDataset.query.filter_by(id=dataset_id, user_id=user_id).first():
        return None
    return _promotable_query(bank_id, dataset_id).count()


_IMPORT_FOLDER_SAFE = re.compile(r'[^A-Za-z0-9 _-]')


def _import_folder_for(name: str) -> str:
    """A fresh, unused folder under bank_sources_root for an imported bank.
    Suffixes -2, -3… rather than reusing a folder: two imports of the same name
    must never end up sharing (and silently merging) one set of files."""
    stem = _IMPORT_FOLDER_SAFE.sub('_', name).strip() or 'bank'
    root = cfg.bank_sources_root()
    candidate = root / stem
    i = 2
    while candidate.exists():
        candidate = root / f'{stem}-{i}'
        i += 1
    return str(candidate)


_SCRAPE_EXT = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'BMP': '.bmp'}


def _scrape_blob_name(raw: bytes) -> str | None:
    """The filename a downloaded blob gets in a bank folder: its own content
    hash. Two consequences, both wanted:

    * a resume that re-downloads the SAME bytes writes the same name, so it
      overwrites itself instead of piling up `photo (2).jpg` — idempotent without
      anyone having to decide what a duplicate is;
    * that is file IDENTITY, not curation. Near-duplicates (a re-encode, a crop,
      the same shot at another size) keep separate names and reach the bank, where
      the duplicate-group pass and the semantic pass are the ONE place that rules
      on them. The dataset outlet's dHash gate deliberately does not run here —
      two different definitions of "duplicate" over one pile is the failure mode
      this whole path exists to avoid.

    None when the bytes are not a raster image we store."""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            ext = _SCRAPE_EXT.get(im.format)
    except (OSError, ValueError):
        return None
    if not ext:
        return None
    return f'{hashlib.sha256(raw).hexdigest()[:24]}{ext}'


def scrape_import_to_bank(user_id, items, bank_id=None, name=None) -> dict:
    """🕸 Scrape → BANK: the scraper's second destination.

    Downloads the SELECTED scanned images ({'url','title'}) into a bank's source
    FOLDER, then lets the ordinary folder walk inventory them. Two modes:
    ``bank_id`` appends to an existing bank (resume — a bank points at a live
    folder, so a second scrape simply adds to the pile it already holds), while
    ``name`` creates a new bank under ``bank_sources_root()`` exactly like
    "Import to bank" does.

    Deliberately does NOT reuse `scrape_import_urls`: that path is a DATASET
    intake and rightly refuses what cannot be trained on (side < 768 px, ratio
    > 3:1) and what it judges a perceptual duplicate. A bank is the step BEFORE
    that judgement — "too small" and "near-duplicate" are verdicts its own passes
    produce, with thresholds the user moves. Filtering at download time would
    delete the evidence before the triage tool ever sees it. What IS kept from
    that path is the download itself (`_download_scrape_item`: SSRF guard,
    content-type allow-list, image-magic check, size cap) and the per-request cap.

    Returns {'bank_id', 'name', 'created', 'saved', 'already_there', 'added',
    'skipped': {...}}. ``added`` is what the folder walk actually inventoried.
    Raises ValueError (bad input) or BankJobBusy (a pass owns the bank)."""
    items = [it for it in (items or []) if isinstance(it, dict) and it.get('url')]
    if not items:
        raise ValueError('no items')
    if len(items) > SCRAPE_IMPORT_MAX:
        raise ValueError(f'max {SCRAPE_IMPORT_MAX} images per import')

    created = False
    if bank_id is not None:
        bank = get_bank(user_id, bank_id)
        if bank is None:
            raise ValueError('bank not found')
        # A live pass works off a snapshot of this bank's rows and reports against
        # a fixed total; refresh_bank also declines to walk underneath it. Adding
        # files now would land outside both — refuse in the shape the UI knows.
        if bank_jobs.running(bank.id):
            # The snapshot can vanish between the two reads (a job that finishes
            # right here); the refusal must still name something.
            snap = bank_jobs.get(bank.id) or {}
            raise bank_jobs.BankJobBusy(snap.get('kind') or 'background')
        folder = bank.source_path
        if not folder or not os.path.isdir(folder):
            raise ValueError('this bank\'s folder is unavailable — relocate it first')
        # Appending here WRITES into the bank's folder. On a legacy bank sitting
        # on a dataset that means downloading scraped files straight into the
        # dataset's training images — not destructive, but exactly the sharing
        # this guard exists to end.
        conflict = path_guard.dataset_folder_conflict(folder)
        if conflict:
            raise ValueError(
                'This bank points at a dataset\'s own image folder, so scraping '
                f'into it would drop files inside the dataset. {conflict["message"]}')
    else:
        name = (name or '').strip()
        if not name:
            raise ValueError('name is required')
        folder = _import_folder_for(name)
        os.makedirs(folder, exist_ok=True)
        bank = ImageBank(user_id=user_id, name=name, source_path=folder)
        db.session.add(bank)
        db.session.commit()
        created = True

    with ThreadPoolExecutor(max_workers=_SCRAPE_DL_WORKERS) as pool:
        downloaded = list(pool.map(_download_scrape_item, items))

    skipped: dict[str, int] = {}
    saved = already_there = 0
    for reason, raw in downloaded:
        if reason != 'ok' or not raw:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        blob_name = _scrape_blob_name(raw)
        if blob_name is None:
            skipped['not_image'] = skipped.get('not_image', 0) + 1
            continue
        dest = os.path.join(folder, blob_name)
        if os.path.exists(dest):
            already_there += 1
            continue
        try:
            with open(dest, 'wb') as fh:
                fh.write(raw)
        except OSError:
            logger.warning('bank scrape: could not write %s', blob_name, exc_info=True)
            skipped['errors'] = skipped.get('errors', 0) + 1
            continue
        saved += 1

    # ONE inventory path for every bank: the same walk that picks up files the
    # user drops in the folder by hand picks these up too. No third insert path.
    sync = refresh_bank(user_id, bank.id, force=True) or {}
    return {'bank_id': bank.id, 'name': bank.name, 'created': created,
            'saved': saved, 'already_there': already_there,
            'added': sync.get('added', 0), 'skipped': skipped}


def start_dataset_import(app, user_id, dataset_id, name):
    """The REVERSE of promote: turn a dataset back into a bank. Copies the
    dataset's KEPT images into a folder of their own and registers it as a bank
    under `name`, so the dataset's material can be re-triaged with the bank tools
    (perceptual + semantic dedup, framing, scores) without disturbing it.

    COPIES rather than pointing the bank at the dataset's live folder: the two
    would otherwise share files, and curating one would mutate the other. That
    mirrors promote, which copies in the other direction — each side owns its
    images. Kept images only, again mirroring promote (which only ever carries
    kept ones across).

    Background job: hundreds of files is a slow copy, and the bank page already
    renders bank_jobs progress. The bank row is created FIRST (empty) so the job
    has a bank_id to report against; a job that dies part-way leaves a bank
    holding exactly the images it managed to copy, never a phantom row.
    Raises ValueError (-> 400) on a missing dataset, a blank name, or nothing kept."""
    from ..models import FaceDatasetImage
    from .dataset_storage import dataset_path
    name = (name or '').strip()
    if not name:
        raise ValueError('name is required')
    ds = FaceDataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    if not ds:
        raise ValueError('dataset not found')
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .filter(FaceDatasetImage.filename.isnot(None))
            .order_by(FaceDatasetImage.id.asc()).all())
    if not rows:
        raise ValueError('nothing to import — keep some images first')
    if len(rows) > BANK_MAX_FILES:
        raise ValueError(f'too many images (max {BANK_MAX_FILES})')
    folder = _import_folder_for(name)
    os.makedirs(folder, exist_ok=True)
    bank = ImageBank(user_id=user_id, name=name, source_path=folder)
    db.session.add(bank)
    db.session.commit()
    src_dir = str(dataset_path(dataset_id))
    bank_jobs.start(
        app, bank.id, 'dataset_import',
        _dataset_import_job(bank.id, src_dir, [r.filename for r in rows]),
        total=len(rows))
    return bank.id


def _dataset_import_job(bank_id, src_dir, filenames):
    def run(job):
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        copied = missing = failed = 0
        for i, fn in enumerate(filenames, 1):
            if bank_jobs.cancelled(job):
                break
            src = os.path.join(src_dir, fn)
            if not os.path.isfile(src):
                missing += 1
                bank_jobs.bump(job)
                continue
            dest = os.path.join(bank.source_path, fn)
            try:
                shutil.copy2(src, dest)
                size = os.path.getsize(dest)
            except OSError:
                # One unreadable/locked file never sinks the whole import — the
                # bank just ends up with the rest, and the detail line says so.
                logger.warning('dataset import: copy %s failed', fn, exc_info=True)
                failed += 1
                bank_jobs.bump(job)
                continue
            db.session.add(BankImage(bank_id=bank_id, relpath=fn, file_size=size))
            copied += 1
            if i % 200 == 0:
                db.session.commit()
            bank_jobs.bump(job)
        db.session.commit()
        detail = f'{copied} image(s) imported'
        if missing:
            detail += f', {missing} missing on disk'
        if failed:
            detail += f', {failed} failed'
        bank_jobs.progress(job, detail=detail)
    return run


# --- ⬆ Promote, second destination: a NEW BANK -------------------------------
# Promotion used to lead exactly one place: a dataset. A dataset is the strict,
# training-bound container — isolating 200 candidates out of a 9 000-image dump
# to keep working on them is a different intent, and forcing it through a dataset
# commits material the user has not decided on yet.
#
# Built on the SAME machinery as "Import to bank" (start_dataset_import): a name,
# a folder of its own under bank_sources_root, a background job, and the new
# bank's id back so the UI can jump to it. What is deliberately NOT reused is
# hardlinking: run_archive.py already settled that question for this app — the
# app rewrites images IN PLACE (re-crop, "Reset to auto", watermark cleaning) and
# an in-place rewrite reuses the inode, so two "independent" banks would become
# one at the first edit. Banks never share their files. It costs the bytes.
def _promote_source_rows(bank_id, ids) -> list:
    """The rows a promotion would carry: the explicit selection, or every KEPT
    image when the selection is empty (same rule as promoting to a dataset).

    Ordered by relpath so the copy, the count and the size preview all describe
    the same set in the same order."""
    if ids:
        wanted = [int(i) for i in ids]
        rows = []
        for i0 in range(0, len(wanted), _SQL_IN_CHUNK):
            rows.extend(BankImage.query.filter(
                BankImage.bank_id == bank_id,
                BankImage.id.in_(wanted[i0:i0 + _SQL_IN_CHUNK])).all())
    else:
        rows = BankImage.query.filter_by(bank_id=bank_id, status='keep').all()
    rows.sort(key=lambda r: r.relpath)
    return rows


def selection_size(user_id, bank_id, ids) -> dict | None:
    """{'count', 'bytes'} for what a promotion would COPY — the honest weight the
    confirmation shows BEFORE the click.

    Real bytes, not an order of magnitude: today's images average ~300 KB, so
    200 of them are ~60 MB and nobody needs warning; a video bank is three orders
    of magnitude above that and the same dialog must not lie about it. Reads the
    size the scan already recorded (one column, no disk hit), and only stats the
    watermark-CLEANED blobs, which are what a promotion actually copies for those
    rows and whose size the column does not describe. None = bank gone."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    rows = _promote_source_rows(bank_id, ids)
    total = 0
    for r in rows:
        if r.watermark_clean_method:
            try:
                total += os.path.getsize(resolved_image_path(bank, r))
                continue
            except (OSError, TypeError):
                pass            # fall back to the recorded source size
        total += int(r.file_size or 0)
    return {'count': len(rows), 'bytes': total}


def start_bank_promote(app, user_id, bank_id, ids, name):
    """Copy a selection into a BRAND NEW bank named ``name``. 202 + background
    job, like every other pass; returns the new bank's id so the UI can jump to
    the bank being filled.

    The job is registered against the SOURCE bank — that is the bank the user is
    looking at, the one whose rows get marked, and the one a concurrent scan
    would race. So an already-busy source bank is the established 409, and the
    progress bar appears where the user clicked.

    Raises ValueError (-> 400) on a missing bank, a blank name or an empty
    selection, BankJobBusy (-> 409) while another pass runs on the source."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    name = (name or '').strip()
    if not name:
        raise ValueError('name is required')
    rows = _promote_source_rows(bank_id, ids)
    if not rows:
        raise ValueError('nothing to promote — keep or select some images first')
    if len(rows) > BANK_MAX_FILES:
        raise ValueError(f'too many images (max {BANK_MAX_FILES})')
    # Checked BEFORE anything is created: bank_jobs.start would raise the same
    # 409 a moment later, having already left a folder and a row behind.
    if bank_jobs.running(bank_id):
        raise bank_jobs.BankJobBusy(bank_jobs.get(bank_id)['kind'])
    folder = _import_folder_for(name)
    os.makedirs(folder, exist_ok=True)
    dest = ImageBank(user_id=user_id, name=name, source_path=folder)
    db.session.add(dest)
    db.session.commit()
    try:
        bank_jobs.start(app, bank_id, 'bank_promote',
                        _bank_promote_job(user_id, bank_id, dest.id,
                                          [r.id for r in rows]),
                        total=len(rows))
    except bank_jobs.BankJobBusy:
        _discard_promoted_bank(user_id, dest.id)   # lost the race: leave nothing
        raise
    return dest.id


def _discard_promoted_bank(user_id, dest_bank_id):
    """Unmake a destination bank that never became one. Uncommitted rows are
    rolled back first, then delete_bank takes the row, the working data and the
    copy folder (it is under bank_sources_root, so it is OURS to remove)."""
    try:
        db.session.rollback()
    except Exception:  # noqa: BLE001 — teardown must not mask the real failure
        logger.warning('bank promote: rollback failed', exc_info=True)
    try:
        delete_bank(user_id, dest_bank_id)
    except Exception:  # noqa: BLE001
        logger.warning('bank promote: could not discard the partial bank',
                       exc_info=True)


def _bank_promote_job(user_id, src_bank_id, dest_bank_id, ids):
    def run(job):
        src = db.session.get(ImageBank, src_bank_id)
        dest = db.session.get(ImageBank, dest_bank_id)
        if not src or not dest:
            return
        rows = _promote_source_rows(src_bank_id, ids)
        bank_jobs.progress(job, done=0, total=len(rows), detail='copying')
        copied, unreadable = [], 0
        for r in rows:
            if bank_jobs.cancelled(job):
                break
            # RESOLVED path: a watermark-cleaned image must land cleaned, same
            # rule as promoting to a dataset.
            p = resolved_image_path(src, r)
            try:
                with open(p, 'rb'):       # prove the SOURCE is the readable one
                    pass
            except (OSError, TypeError):
                # One unreadable/locked source costs one image, never the run.
                unreadable += 1
                bank_jobs.bump(job)
                continue
            target = os.path.join(dest.source_path, r.relpath)
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(p, target)
                size = os.path.getsize(target)
            except OSError as e:
                # The source opened, so this is the DESTINATION refusing: disk
                # full, read-only, drive unplugged. Carrying on would leave a
                # bank holding half the selection and presenting as finished —
                # the one outcome worse than failing. Unmake it and say so.
                logger.warning('bank promote: writing the copy failed',
                               exc_info=True)
                _discard_promoted_bank(user_id, dest_bank_id)
                bank_jobs.fail(job, 'Could not write the copies — the new bank '
                                    'was discarded and nothing was changed. '
                                    'Check the free space on the drive holding '
                                    "the app's data, then try again. "
                                    f'({e.strerror or "write failed"})')
                return
            db.session.add(BankImage(bank_id=dest_bank_id, relpath=r.relpath,
                                     file_size=size))
            copied.append(r.id)
            if len(copied) % 200 == 0:
                db.session.commit()
            bank_jobs.bump(job)
        if not copied:
            _discard_promoted_bank(user_id, dest_bank_id)
            bank_jobs.fail(job, 'Nothing could be copied — the new bank was '
                                'discarded. The selected files could not be read.')
            return
        db.session.commit()
        # Marked LAST, and only for what really landed: the source keeps its rows
        # (a promotion never removes anything from the bank it came from) and now
        # says where they went, exactly like a promotion to a dataset.
        for i0 in range(0, len(copied), _SQL_IN_CHUNK):
            (BankImage.query
             .filter(BankImage.bank_id == src_bank_id,
                     BankImage.id.in_(copied[i0:i0 + _SQL_IN_CHUNK]))
             .update({'promoted_bank_id': dest_bank_id},
                     synchronize_session=False))
        db.session.commit()
        detail = f'{len(copied)} image(s) copied into "{dest.name}"'
        if unreadable:
            detail += f', {unreadable} unreadable'
        bank_jobs.progress(job, detail=detail)
    return run


def start_promote(app, user_id, bank_id, ids, dataset_id):
    """Copy a selection into a dataset through the normal import path
    (normalize + perceptual dedup vs the dataset). ``ids`` empty = every KEPT
    image not already on THIS dataset. Background job (a big promotion decodes
    hundreds of files)."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    ds = FaceDataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    if not ds:
        raise ValueError('dataset not found')
    if ids:
        ids = [int(i) for i in ids]
    else:
        ids = [r.id for r in
               _promotable_query(bank_id, dataset_id)
               .order_by(BankImage.id.asc()).all()]
    if not ids:
        raise ValueError('nothing to promote — keep some images first')
    return bank_jobs.start(app, bank_id, 'promote',
                           _promote_job(user_id, bank_id, ids, dataset_id),
                           total=len(ids))


def _promote_job(user_id, bank_id, ids, dataset_id):
    def run(job):
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = []
        for i0 in range(0, len(ids), _SQL_IN_CHUNK):
            rows.extend(BankImage.query.filter(
                BankImage.bank_id == bank_id,
                BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all())
        rows.sort(key=lambda r: r.id)
        bank_jobs.progress(job, done=0, total=len(rows), detail='promoting')
        stats: dict = {}
        imported = failed = 0
        for c0 in range(0, len(rows), _PROMOTE_CHUNK):
            if bank_jobs.cancelled(job):
                break
            chunk = rows[c0:c0 + _PROMOTE_CHUNK]
            blobs, chunk_rows, caps, frms = [], [], [], []
            for r in chunk:
                # RESOLVED path: a watermark-cleaned image must reach the dataset
                # cleaned, otherwise the two cleaning levels were run for nothing.
                p = resolved_image_path(bank, r)
                try:
                    with open(p, 'rb') as fh:
                        blobs.append(fh.read())
                    chunk_rows.append(r)
                    # Carry the bank caption onto the dataset image (parallel to blobs),
                    # so a captioned selection lands already captioned.
                    caps.append(r.caption)
                    # Carry the framing the bank's classify pass already wrote, so
                    # the dataset's Composition counter is right the moment the
                    # promotion lands (it only tallies rows that HAVE a framing).
                    frms.append(r.framing)
                except (OSError, TypeError):
                    failed += 1
            if blobs:
                new_ids, bad = import_images(
                    user_id, dataset_id, blobs, dedupe=True, stats=stats,
                    captions=caps, bank_image_ids=[r.id for r in chunk_rows],
                    framings=frms)
                imported += len(new_ids)
                failed += bad
                # The dataset row now carries the link back (import_images writes
                # it, and hands it to the matched row when a dedupe skips the
                # blob), so 'already promoted here' is a fact we can re-check.
                # Clear the legacy one-way flag as we go: it would otherwise keep
                # excluding this image from the target long after the user
                # deleted it there.
                #
                # The exception is an image whose row in the dataset is already
                # credited to ANOTHER bank (both banks hold the same photo). There
                # is one column for one owner, so this bank gets no verifiable
                # trace and keeps the old flag — the alternative is offering the
                # image on every promotion, forever.
                unlinked = set(stats.get('bank_unlinked') or ())
                stats.pop('bank_unlinked', None)
                for r in chunk_rows:
                    r.promoted_dataset_id = dataset_id if r.id in unlinked else None
                db.session.commit()
            bank_jobs.bump(job, len(chunk))
        dups = stats.get('duplicates', 0)
        small = stats.get('small', 0)
        detail = f'done — {imported} imported'
        if dups:
            detail += f', {dups} already in the dataset'
        if failed:
            detail += f', {failed} failed'
        if small:
            detail += f', {small} under the recommended size'
        bank_jobs.progress(job, detail=detail)
    return run
