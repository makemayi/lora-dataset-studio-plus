"""Per-dataset backup: writing one dataset out as a self-contained ZIP
(metadata + images + reference photos) and importing such an archive back as a
NEW dataset.

Every limit in here is a zip-bomb guard, not a preference: file count, row
count, uncompressed total, per-image bytes and metadata bytes are all capped
before anything is read, and image names are validated against a strict
pattern rather than trusted from the archive.

Not to be confused with full_backup.py, which archives the whole install.

Split out of face_dataset_service.py (2026-08, Phase 7 of a multi-phase file
split) -- pure move, no behavior change.
"""
import io
import json
import ntpath
import os
import posixpath
import re
import shutil
import time
import uuid
import zipfile
from typing import BinaryIO

from . import dataset_activity
from ..extensions import db
from ..models import FaceDataset, FaceDatasetImage
from .. import config as cfg
from . import bank_transfer_metadata

# Own sentinel rather than face_dataset_service's: it is a DEFAULT ARGUMENT
# below, and a default is evaluated while this module's body runs -- before the
# borrow-back import at the bottom has bound anything. Identity is only ever
# compared inside this module, and no caller outside it passes `limit`, so a
# separate object is equivalent.
_UNSET = object()


# --- Sauvegarde / restauration complète d'un dataset ---------------------------
# ZIP portable (≠ export d'entraînement) : manifest + réglages + TOUTES les images
# avec statuts/captions/scores — pour archiver ou déplacer un dataset entre machines.
BACKUP_FORMAT = 'lds-dataset-backup'
BACKUP_VERSION = 2
_BACKUP_MAX_FILES = 1400
_BACKUP_MAX_ROWS = 600
_BACKUP_MAX_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB uncompressed (zip-bomb guard)
_BACKUP_MAX_METADATA_BYTES = 192 * 1024 * 1024
# The ZIP central directory is read BEFORE any member: a crafted archive can
# otherwise make zipfile allocate for tens of thousands of declared entries
# that do not exist. Bounded here, checked by _preflight_zip_central_directory.
_BACKUP_MAX_CENTRAL_DIRECTORY_BYTES = 16 * 1024 * 1024
# The import raster budget permits at most 16 Mi pixels.  96 MiB leaves room for
# a valid 16 MP RGBA/BMP source plus container overhead, while making a single
# archive entry incapable of filling a disk or RAM by itself.
_BACKUP_MAX_IMAGE_BYTES = 96 * 1024 * 1024
_BACKUP_NAME_RE = re.compile(r'^[\w.-]+\.(webp|jpg|jpeg|png|bmp)$', re.IGNORECASE)
_BACKUP_EXTENSION_CANONICAL = {
    '.jpg': '.jpg', '.jpeg': '.jpg', '.png': '.png', '.webp': '.webp', '.bmp': '.bmp',
}

# Champs snapshotés tels quels par ligne image (job_id/klein_model exclus : liés
# à la machine source — un backup restauré ne peut pas « regénérer »).
# Every column a restore must carry. A field missing HERE is data silently lost
# on the round trip, so it tracks upstream's list exactly.
_BACKUP_IMG_FIELDS = ('filename', 'source', 'framing', 'variation_label', 'status',
                      'caption', 'caption_short',
                      'caption_origin', 'caption_short_origin',
                      'variation_prompt', 'klein_model', 'face_score', 'face_state',
                      'upscale_ratio', 'watermark_state', 'watermark_bbox',
                      'watermark_source', 'watermark_score',
                      'watermark_regions', 'parent_image_id', 'derivation_kind',
                      'fail_reason', 'fail_kind', 'source_metadata',
                      'bank_analysis_snapshot', 'transfer_metadata')


def _preflight_zip_central_directory(
        stream, *, max_entries, max_central_bytes, label) -> None:
    """Bound EOCD-declared allocation before ``ZipFile`` parses the directory.

    ``_EndRecData`` is the bounded EOCD reader used by CPython 3.12 itself.  Keep
    a compatibility fallback for runtimes that do not expose it, and always
    rewind because ``ZipFile`` must receive the original stream from offset zero.
    """
    end_reader = getattr(zipfile, '_EndRecData', None)
    if not callable(end_reader):
        stream.seek(0)
        return
    try:
        try:
            end_record = end_reader(stream)
        except (OSError, EOFError, AttributeError, OverflowError, TypeError, ValueError,
                zipfile.BadZipFile) as exc:
            raise ValueError('not a zip file') from exc
    finally:
        try:
            stream.seek(0)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError('zip archive is not seekable') from exc
    if end_record is None:
        raise ValueError('not a zip file')
    try:
        entries = end_record[getattr(zipfile, '_ECD_ENTRIES_TOTAL', 4)]
        central_size = end_record[getattr(zipfile, '_ECD_SIZE', 5)]
    except (IndexError, TypeError) as exc:
        raise ValueError('not a zip file') from exc
    if (isinstance(entries, bool) or not isinstance(entries, int)
            or entries < 0
            or isinstance(central_size, bool) or not isinstance(central_size, int)
            or central_size < 0):
        raise ValueError('not a zip file')
    if entries > max_entries:
        raise ValueError(f'too many files in {label} (max {max_entries})')
    if central_size > max_central_bytes:
        raise ValueError(f'{label} central directory is too large')


def _preflight_backup_central_directory(stream) -> None:
    return _preflight_zip_central_directory(
        stream, max_entries=_BACKUP_MAX_FILES + 2,
        max_central_bytes=_BACKUP_MAX_CENTRAL_DIRECTORY_BYTES,
        label='backup')


def _backup_basename(value):
    """Return a portable image basename, or None for paths/invalid values."""
    if not isinstance(value, str) or not value:
        return None
    if '/' in value or '\\' in value or not _BACKUP_NAME_RE.fullmatch(value):
        return None
    # Windows treats these stems as devices even when an extension is present
    # (``NUL.jpg`` writes to NUL). Reject them on every platform so an archive
    # restored on Windows cannot commit a row whose blob was never created.
    stem = value.split('.', 1)[0].rstrip(' .')
    if _BACKUP_WINDOWS_DEVICE_RE.fullmatch(stem):
        return None
    return value


def _backup_file_generation(path):
    """Identity/generation of one regular backup source, without symlinks."""
    try:
        info = os.stat(path, follow_symlinks=False)
    except (OSError, TypeError, ValueError):
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _backup_handle_generation(stream):
    try:
        info = os.fstat(stream.fileno())
    except (OSError, AttributeError, ValueError):
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _write_backup_file_member(archive, path, archive_name, expected_generation):
    """Stream one already-identified regular file through the same open handle."""
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0)
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError('dataset changed while the backup was created') from exc
    with os.fdopen(descriptor, 'rb') as source:
        if _backup_handle_generation(source) != expected_generation:
            raise ValueError('dataset changed while the backup was created')
        with archive.open(archive_name, 'w') as destination:
            remaining = expected_generation[2]
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError('dataset changed while the backup was created')
                destination.write(chunk)
                remaining -= len(chunk)
            # The generation's exact length is part of the archive contract.  A
            # concurrent append must not escape the writer's aggregate preflight.
            if source.read(1):
                raise ValueError('dataset changed while the backup was created')
        if _backup_handle_generation(source) != expected_generation:
            raise ValueError('dataset changed while the backup was created')


def _validate_backup_limits(member_sizes=(), *, row_count=None) -> None:
    """Apply the portable-backup budgets to reader and writer inventories."""
    members = tuple(member_sizes)
    if len(members) > _BACKUP_MAX_FILES + 2:
        raise ValueError(f'too many files in backup (max {_BACKUP_MAX_FILES})')
    total = 0
    for name, size in members:
        total += size
        if total > _BACKUP_MAX_BYTES:
            raise ValueError('backup too large (max 2 GB uncompressed)')
        if (name.startswith(('images/', 'ref/'))
                and size > _BACKUP_MAX_IMAGE_BYTES):
            basename = name.rsplit('/', 1)[-1]
            raise ValueError(
                f'backup image {basename} is too large '
                f'(max {_BACKUP_MAX_IMAGE_BYTES // (1024 * 1024)} MiB per image)')
    if row_count is not None and row_count > _BACKUP_MAX_ROWS:
        raise ValueError(
            f'too many image rows in backup (max {_BACKUP_MAX_ROWS})')


def _normalized_backup_image_meta(meta, *, version=BACKUP_VERSION):
    """Validate every active FaceDatasetImage value before restore staging."""
    if not isinstance(meta, dict):
        raise ValueError('invalid backup image metadata')
    out = dict(meta)

    def optional_text(field, limit, *, allowed=None):
        value = meta.get(field)
        if value is None:
            return None
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f'invalid backup image {field}')
        if allowed is not None and value not in allowed:
            raise ValueError(f'invalid backup image {field}')
        return value

    def optional_number(field, low=None, high=None):
        value = meta.get(field)
        if value is None:
            return None
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f'invalid backup image {field}')
        number = float(value)
        if ((low is not None and number < low)
                or (high is not None and number > high)):
            raise ValueError(f'invalid backup image {field}')
        return number

    filename = meta.get('filename')
    if filename is not None and not isinstance(filename, str):
        raise ValueError('invalid backup image filename')
    if (version >= 2 and filename is not None
            and _backup_basename(filename) != filename):
        raise ValueError('invalid backup image filename')
    source = optional_text('source', 12, allowed=('generated', 'import'))
    status = optional_text(
        'status', 10, allowed=('pending', 'keep', 'reject', 'failed'))
    # v1 and small hand-built integration archives predate ``source``; use the
    # model's historical defaults without weakening validation of supplied data.
    source = source or 'generated'
    status = status or 'pending'
    out['source'], out['status'] = source, status
    out['framing'] = optional_text(
        'framing', 12, allowed=('face', 'bust', 'body', 'back', 'unknown'))
    out['variation_label'] = optional_text('variation_label', 120)
    out['caption'] = optional_text('caption', CAPTION_MAX_CHARS)
    out['caption_short'] = optional_text('caption_short', CAPTION_MAX_CHARS)
    for field, caption_field in (
            ('caption_origin', 'caption'),
            ('caption_short_origin', 'caption_short')):
        origin = optional_text(field, 16, allowed=caption_origin.VALUES)
        out[field] = origin if (out.get(caption_field) or '').strip() else None
    out['variation_prompt'] = optional_text('variation_prompt', 500)
    out['klein_model'] = optional_text('klein_model', 255)
    out['face_score'] = optional_number('face_score', -1.0, 1.0)
    out['face_state'] = optional_text(
        'face_state', 16,
        allowed=('scorable', 'no_face', 'low_det', 'too_small',
                 'extreme_pose', 'unreadable', 'error'))
    out['fail_reason'] = optional_text('fail_reason', 32768)
    out['fail_kind'] = optional_text(
        'fail_kind', 16, allowed=('refused', 'empty', 'error'))
    out['upscale_ratio'] = optional_number('upscale_ratio', 0.0, 1_000_000.0)
    out['watermark_state'] = optional_text(
        'watermark_state', 16,
        allowed=('none', 'detected', 'dismissed', 'cleaned', 'failed', 'error'))
    out['watermark_source'] = optional_text(
        'watermark_source', 16, allowed=('detector', 'vision'))
    out['watermark_score'] = optional_number('watermark_score', 0.0, 1.0)

    def box_storage(field, *, many):
        raw = meta.get(field)
        if raw is None:
            return None
        if not isinstance(raw, str) or len(raw) > 32768:
            raise ValueError(f'invalid backup image {field}')
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, RecursionError, MemoryError):
            raise ValueError(f'invalid backup image {field}')
        boxes = parsed if many else [parsed]
        if not isinstance(boxes, list) or len(boxes) > WATERMARK_REGION_LIMIT:
            raise ValueError(f'invalid backup image {field}')
        normalized = []
        for box in boxes:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f'invalid backup image {field}')
            if any(isinstance(value, bool)
                   or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in box):
                raise ValueError(f'invalid backup image {field}')
            x1, y1, x2, y2 = map(float, box)
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f'invalid backup image {field}')
            normalized.append([x1, y1, x2, y2])
        # Validation is semantic, transport remains byte-for-byte at the TEXT
        # level (including harmless whitespace) for a faithful backup roundtrip.
        return raw

    out['watermark_bbox'] = box_storage('watermark_bbox', many=False)
    out['watermark_regions'] = box_storage('watermark_regions', many=True)

    parent_id = meta.get('parent_image_id')
    if (parent_id is not None and (isinstance(parent_id, bool)
                                   or not isinstance(parent_id, int)
                                   or parent_id <= 0)):
        raise ValueError('invalid backup parent image id')
    out['parent_image_id'] = parent_id
    return out


def _normalized_backup_manifest(manifest):
    """Typed/bounded portable Dataset settings from an untrusted archive."""
    if not isinstance(manifest, dict):
        raise ValueError('invalid backup manifest')
    out = dict(manifest)

    def text_field(field, limit, *, allowed=None, required=False):
        value = manifest.get(field)
        if value is None:
            if required:
                raise ValueError(f'invalid backup {field}')
            return None
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f'invalid backup {field}')
        if allowed is not None and value not in allowed:
            raise ValueError(f'invalid backup {field}')
        return value

    out['name'] = text_field('name', 100)
    out['trigger_word'] = text_field('trigger_word', 60)
    out['kind'] = text_field(
        'kind', 16, allowed=('character', 'concept', 'style'))
    out['fidelity'] = text_field('fidelity', 8, allowed=FIDELITIES)
    out['concept_desc'] = text_field('concept_desc', 500)
    out['train_type'] = text_field('train_type', 16, allowed=TRAIN_TYPES)
    out['training_mode'] = text_field(
        'training_mode', 32, allowed=('lora', 'full_transformer')) or 'lora'
    out['train_base_model'] = text_field('train_base_model', 4096)
    variant = text_field('train_variant', 20)
    if variant is not None and not re.fullmatch(r'[A-Za-z0-9_.-]+', variant):
        raise ValueError('invalid backup train_variant')
    out['train_variant'] = variant

    def parsed_json_field(field, expected_type, max_bytes=1024 * 1024):
        raw = manifest.get(field)
        if raw is None:
            return None
        if not isinstance(raw, str) or len(raw.encode('utf-8')) > max_bytes:
            raise ValueError(f'invalid backup {field}')
        try:
            value = json.loads(
                raw,
                parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()))
        except (TypeError, ValueError, UnicodeError, RecursionError, MemoryError):
            raise ValueError(f'invalid backup {field}')
        if not isinstance(value, expected_type):
            raise ValueError(f'invalid backup {field}')
        stack = [(value, 0)]
        seen = 0
        while stack:
            current, depth = stack.pop()
            seen += 1
            if seen > 100_000 or depth > 32:
                raise ValueError(f'invalid backup {field}')
            if isinstance(current, float) and not math.isfinite(current):
                raise ValueError(f'invalid backup {field}')
            if isinstance(current, dict):
                stack.extend((item, depth + 1) for item in current.values())
            elif isinstance(current, list):
                stack.extend((item, depth + 1) for item in current)
        return raw

    out['train_settings'] = parsed_json_field('train_settings', dict)
    out['best_settings'] = parsed_json_field('best_settings', dict)
    out['concept_terms'] = parsed_json_field(
        'concept_terms', list, max_bytes=256 * 1024)
    if out['concept_terms'] is not None:
        terms = json.loads(out['concept_terms'])
        if (len(terms) > 2048
                or any(not isinstance(term, str) or len(term) > 500
                       for term in terms)):
            raise ValueError('invalid backup concept_terms')
    return out


def _read_validated_backup_image(z: zipfile.ZipFile, info: zipfile.ZipInfo,
                                 basename: str) -> bytes:
    """Return one bounded, fully-decoded static backup image.

    The outer archive's central-directory total protects the aggregate, but a
    single entry still needs its own raw cap before it is inflated.  Crucially,
    validation happens while the bytes are only in memory: a malformed image
    must never be copied into the restore staging directory and later promoted.
    """
    max_bytes = _BACKUP_MAX_IMAGE_BYTES
    if info.file_size > max_bytes:
        raise ValueError(
            f'backup image {basename} is too large '
            f'(max {max_bytes // (1024 * 1024)} MiB per image)')
    try:
        with z.open(info) as source:
            # Do not trust a crafted central-directory ``file_size`` alone.
            # Reading one extra byte limits actual decompression even if that
            # metadata is inconsistent with the entry payload.
            raw = source.read(max_bytes + 1)
    except (OSError, EOFError, RuntimeError, MemoryError, zipfile.BadZipFile,
            zlib.error, lzma.LZMAError) as exc:
        raise ValueError(f'backup image {basename} could not be read') from exc
    if len(raw) > max_bytes:
        raise ValueError(
            f'backup image {basename} is too large '
            f'(max {max_bytes // (1024 * 1024)} MiB per image)')
    try:
        content_ext = _preserved_import_extension(raw, label=f'backup image {basename}')
    except (ValueError, MemoryError) as exc:
        raise ValueError(f'backup image {basename} is invalid: {exc}') from exc
    named_ext = _BACKUP_EXTENSION_CANONICAL.get(os.path.splitext(basename)[1].lower())
    if named_ext != content_ext:
        raise ValueError(
            f'backup image {basename} extension does not match its decoded content')
    return raw


def _backup_extra_ref_names(raw, *, limit=_UNSET):
    """Parse the stored JSON list into unique portable basenames."""
    if limit is _UNSET:
        # MAX_EXTRA_REFS lives in reference_photos_service.py and only reaches
        # this module through the bottom-of-file borrow, so it is NOT bound yet
        # when this def executes. The default therefore has to resolve at CALL
        # time — upstream can spell it as a def-time default because it has no
        # split to schedule around.
        limit = MAX_EXTRA_REFS
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    if not isinstance(raw, list):
        return []
    out = []
    seen = set()
    for value in raw:
        name = _backup_basename(value)
        key = name.casefold() if name else None
        if not name or key in seen:
            continue
        seen.add(key)
        out.append(name)
        if limit is not None and len(out) >= limit:
            break
    return out


def _portable_train_base_model(value):
    """Keep model ids/relative paths, never machine-local absolute paths."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    drive, _ = ntpath.splitdrive(value)
    if (not value or drive or ntpath.isabs(value) or posixpath.isabs(value)):
        return None
    return value


def _write_backup_zip_locked(user_id: int, dataset_id: int,
                             output: BinaryIO, *, activity_token=None) -> None:
    """Self-contained backup of one dataset: manifest.json (settings) +
    images.json (rows) + ref/ + images/ files. Ordinary rows without a file are
    skipped, but small-image rescue metadata rows are retained so their pair can
    never become orphaned after restore."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    dsdir = _dataset_dir(dataset_id)
    from sqlalchemy import or_
    rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id)
            .filter(or_(FaceDatasetImage.filename.isnot(None),
                        FaceDatasetImage.derivation_kind.in_(_SMALL_IMAGE_DERIVATIONS)))
            .all())
    # Fail before touching the destination stream: an archive emitted with more
    # rows than the importer accepts is neither portable nor useful.
    _validate_backup_limits(row_count=len(rows))
    primary_ref_names = []
    ref_name_keys = set()
    for raw_name in (ds.ref_filename, ds.ref_original_filename):
        name = _backup_basename(raw_name)
        key = name.casefold() if name else None
        if (not name or key in ref_name_keys
                or not os.path.isfile(os.path.join(dsdir, name))):
            continue
        ref_name_keys.add(key)
        primary_ref_names.append(name)
    portable_extras = []
    for name in _backup_extra_ref_names(ds.ref_extra_filenames, limit=None):
        key = name.casefold()
        if (key in ref_name_keys
                or not os.path.isfile(os.path.join(dsdir, name))):
            continue
        ref_name_keys.add(key)
        portable_extras.append(name)
        if len(portable_extras) >= MAX_EXTRA_REFS:
            break
    # Extra-ref ORIGINALS travel too (as plain ref/ files, not in the manifest list):
    # restore keeps basenames, so the naming convention still ties them to their
    # extra — a restored dataset can widen a crop back out just like the source one.
    extra_originals = []
    for name in portable_extras:
        orig = extra_ref_original_name(name)
        key = orig.casefold() if orig else None
        if (not orig or key in ref_name_keys
                or not os.path.isfile(os.path.join(dsdir, orig))):
            continue
        ref_name_keys.add(key)
        extra_originals.append(orig)
    ref_names = primary_ref_names + portable_extras + extra_originals
    image_file_names = {
        name.casefold(): name for img in rows
        if (name := _backup_basename(img.filename))
        and os.path.isfile(os.path.join(dsdir, name))
    }
    file_backed_rows = [img for img in rows if _backup_basename(img.filename)]
    if len(image_file_names) != len(file_backed_rows):
        raise ValueError('dataset has duplicate or missing image files')
    collisions = ref_name_keys.intersection(image_file_names)
    if collisions:
        collision = image_file_names[next(iter(collisions))]
        raise ValueError(f'ref/image filename collision in dataset: {collision}')

    manifest = {
        'format': BACKUP_FORMAT, 'version': BACKUP_VERSION,
        'name': ds.name, 'trigger_word': ds.trigger_word,
        'kind': ds.kind, 'fidelity': ds.fidelity,
        'concept_desc': ds.concept_desc, 'concept_terms': ds.concept_terms,
        'train_type': ds.train_type,
        # Optional in backup v1: old archives omit it and restore as LoRA.
        'training_mode': (ds.training_mode
                          if ds.training_mode in ('lora', 'full_transformer')
                          else 'lora'),
        'train_base_model': _portable_train_base_model(ds.train_base_model),
        'train_variant': ds.train_variant, 'train_settings': ds.train_settings,
        'best_settings': ds.best_settings,
        'ref_filename': (_backup_basename(ds.ref_filename)
                         if _backup_basename(ds.ref_filename) in primary_ref_names else None),
        'ref_original_filename': (
            _backup_basename(ds.ref_original_filename)
            if _backup_basename(ds.ref_original_filename) in primary_ref_names else None),
        'ref_extra_filenames': json.dumps(portable_extras),
    }
    manifest_generation = tuple(
        getattr(ds, field) for field in (
            'name', 'trigger_word', 'kind', 'fidelity', 'concept_desc',
            'concept_terms', 'train_type', 'training_mode', 'train_base_model',
            'train_variant', 'train_settings', 'best_settings', 'ref_filename',
            'ref_original_filename', 'ref_extra_filenames'))
    row_generations = {
        img.id: tuple(getattr(img, field) for field in _BACKUP_IMG_FIELDS)
        for img in rows
    }
    backup_files = [
        (os.path.join(dsdir, name), f'ref/{name}') for name in ref_names
    ] + [
        (os.path.join(dsdir, name), f'images/{name}')
        for name in image_file_names.values()
    ]
    file_generations = {}
    for path, _archive_name in backup_files:
        generation = _backup_file_generation(path)
        if generation is None:
            raise ValueError('dataset file disappeared before backup')
        file_generations[path] = generation
    # backup_image_id is archive-local only. It lets restore remap parent_image_id
    # to the newly allocated row ids instead of retaining ids from the source DB.
    images_meta = []
    analysis_cache_payloads = {}
    analysis_cache_dir = _bank_analysis_cache_dir(dataset_id)
    for row_index, img in enumerate(rows, 1):
        if row_index == 1 or row_index % 25 == 0:
            dataset_activity.progress(
                activity_token, detail='sealing backup metadata')
        row = dict({'backup_image_id': img.id},
                   **{f: getattr(img, f) for f in _BACKUP_IMG_FIELDS})
        # Archive a structured, revalidated object rather than the raw TEXT
        # column. A malformed legacy/local row can never export arbitrary links.
        row['source_metadata'] = normalize_source_metadata(img.source_metadata)
        # A snapshot is durable only when it has the expected version, fingerprint
        # and bounded analysis shape.  Invalid legacy/local text is deliberately
        # omitted rather than becoming an opaque payload in a portable backup.
        snapshot = bank_transfer_metadata.parse_snapshot(img.bank_analysis_snapshot)
        if snapshot is None and img.bank_analysis_snapshot:
            try:
                declared = (img.bank_analysis_snapshot
                            if isinstance(img.bank_analysis_snapshot, dict)
                            else json.loads(img.bank_analysis_snapshot))
            except (TypeError, ValueError, UnicodeError, RecursionError, MemoryError):
                declared = None
            if isinstance(declared, dict) and declared.get('cache_ref') is not None:
                raise ValueError('invalid Bank analysis cache snapshot')
        if snapshot and snapshot.get('cache_ref'):
            cache_ref = snapshot['cache_ref']
            if not bank_transfer_metadata.is_content_addressed_cache_ref(cache_ref):
                raise ValueError(
                    'Bank analysis cache is not content-addressed; re-promote '
                    'the image before creating a backup')
            if cache_ref not in analysis_cache_payloads:
                cache_path = os.path.join(
                    analysis_cache_dir, f'{cache_ref}.npz')
                try:
                    size = os.path.getsize(cache_path)
                    if size <= 0 or size > bank_transfer_metadata.CACHE_SIDECAR_MAX_BYTES:
                        raise ValueError('Bank analysis cache sidecar is too large')
                    with open(cache_path, 'rb') as source:
                        cache_payload = source.read(
                            bank_transfer_metadata.CACHE_SIDECAR_MAX_BYTES + 1)
                except OSError as exc:
                    raise ValueError(
                        'Bank analysis cache sidecar is missing or unreadable') from exc
                if (len(cache_payload) != size
                        or bank_transfer_metadata.read_cache_sidecar_bytes(
                            cache_payload, expected_ref=cache_ref) is None):
                    raise ValueError(
                        'Bank analysis cache sidecar is malformed or has changed')
                analysis_cache_payloads[cache_ref] = cache_payload
        row['bank_analysis_snapshot'] = (
            bank_transfer_metadata.normalized_snapshot_storage(snapshot)
            if snapshot is not None else None)
        transfer_metadata = (
            bank_transfer_metadata.normalized_transfer_metadata_storage(
                img.transfer_metadata))
        if img.transfer_metadata is not None and transfer_metadata is None:
            raise ValueError('invalid Bank/Dataset transfer metadata')
        row['transfer_metadata'] = transfer_metadata
        images_meta.append(row)
    manifest_payload = json.dumps(
        manifest, ensure_ascii=False, indent=1).encode('utf-8')
    images_payload = json.dumps(
        images_meta, ensure_ascii=False, indent=1).encode('utf-8')
    if (len(manifest_payload) > _BACKUP_MAX_METADATA_BYTES
            or len(images_payload) > _BACKUP_MAX_METADATA_BYTES):
        raise ValueError('dataset backup metadata is too large')
    backup_member_sizes = [
        ('manifest.json', len(manifest_payload)),
        ('images.json', len(images_payload)),
        *((archive_name, file_generations[path][2])
          for path, archive_name in backup_files),
        *((f'analysis-cache/{cache_ref}.npz', len(payload))
          for cache_ref, payload in analysis_cache_payloads.items()),
    ]
    _validate_backup_limits(backup_member_sizes, row_count=len(images_meta))
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        total_members = 2 + len(backup_files) + len(analysis_cache_payloads)
        dataset_activity.progress(
            activity_token, done=0, total=total_members,
            detail='writing portable backup')
        z.writestr('manifest.json', manifest_payload)
        dataset_activity.bump(activity_token)
        z.writestr('images.json', images_payload)
        dataset_activity.bump(activity_token)
        for path, archive_name in backup_files:
            if _backup_file_generation(path) != file_generations[path]:
                raise ValueError('dataset changed while the backup was created')
            _write_backup_file_member(
                z, path, archive_name, file_generations[path])
            if _backup_file_generation(path) != file_generations[path]:
                raise ValueError('dataset changed while the backup was created')
            dataset_activity.bump(activity_token)
        for cache_ref, payload in sorted(analysis_cache_payloads.items()):
            # Archive exactly the immutable bytes validated above.  A live
            # sidecar replacement between validation and ZipFile.write cannot
            # smuggle different cache contents into the backup.
            z.writestr(f'analysis-cache/{cache_ref}.npz', payload)
            dataset_activity.bump(activity_token)
        # Final generation fence: activity/ingest reservations block normal app
        # writes, while this catches an already-running request or an external
        # same-folder change that crossed the reservation boundary.
        current_ds = (FaceDataset.query.filter_by(id=dataset_id, user_id=user_id)
                      .populate_existing().one_or_none())
        current_manifest_generation = (tuple(
            getattr(current_ds, field) for field in (
                'name', 'trigger_word', 'kind', 'fidelity', 'concept_desc',
                'concept_terms', 'train_type', 'training_mode',
                'train_base_model', 'train_variant', 'train_settings',
                'best_settings', 'ref_filename', 'ref_original_filename',
                'ref_extra_filenames')) if current_ds is not None else None)
        current_rows = (FaceDatasetImage.query
                        .filter(FaceDatasetImage.id.in_(tuple(row_generations)))
                        .populate_existing().all()) if row_generations else []
        current_row_generations = {
            img.id: tuple(getattr(img, field) for field in _BACKUP_IMG_FIELDS)
            for img in current_rows if img.dataset_id == dataset_id
        }
        current_row_ids = {
            row_id for row_id, in db.session.query(FaceDatasetImage.id)
            .filter_by(dataset_id=dataset_id)
            .filter(or_(FaceDatasetImage.filename.isnot(None),
                        FaceDatasetImage.derivation_kind.in_(
                            _SMALL_IMAGE_DERIVATIONS))).all()
        }
        if (current_manifest_generation != manifest_generation
                or current_row_generations != row_generations
                or current_row_ids != set(row_generations)
                or any(_backup_file_generation(path) != generation
                       for path, generation in file_generations.items())):
            raise ValueError('dataset changed while the backup was created')


def write_backup_zip(user_id: int, dataset_id: int, output: BinaryIO) -> None:
    """Create one coherent backup generation under Dataset-wide exclusion."""
    dataset_id = dataset_activity.normalize_dataset_id(dataset_id)
    with _dataset_ingest_lock(user_id, dataset_id):
        token = dataset_activity.begin_exclusive(
            dataset_id, 'backup', detail='creating portable backup')
        if token is None:
            raise ValueError(
                'dataset has work in progress; wait before creating a backup')
        try:
            return _write_backup_zip_locked(
                user_id, dataset_id, output, activity_token=token)
        finally:
            dataset_activity.end(token)


def build_backup_zip(user_id: int, dataset_id: int) -> bytes:
    """Compatibility wrapper for callers that still need an in-memory archive."""
    output = io.BytesIO()
    write_backup_zip(user_id, dataset_id, output)
    return output.getvalue()


def _coerce_archive_stream(archive):
    """Return (seekable stream, owned stream or None) without copying file uploads."""
    if isinstance(archive, (bytes, bytearray, memoryview)):
        owned = io.BytesIO(bytes(archive))
        return owned, owned
    if not hasattr(archive, 'read') or not hasattr(archive, 'seek'):
        raise ValueError('not a zip file')
    try:
        archive.seek(0)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError('zip archive is not seekable') from exc
    return archive, None


def import_backup_zip(user_id: int, archive: bytes | BinaryIO):
    """Restore a backup as a NEW dataset (never merges into an existing one).
    Hardened: manifest format/version check, per-entry filename whitelist (no
    separators/traversal), file-count and uncompressed-size caps. Returns the
    created FaceDataset."""
    stream, owned = _coerce_archive_stream(archive)
    try:
        _preflight_backup_central_directory(stream)
        try:
            z = zipfile.ZipFile(stream)
        except zipfile.BadZipFile as exc:
            raise ValueError('not a zip file') from exc
        try:
            return _import_backup_zipfile(user_id, z)
        finally:
            z.close()
    finally:
        if owned is not None:
            owned.close()


def _import_backup_zipfile(user_id: int, z: zipfile.ZipFile):
    # Validate the central directory BEFORE inflating JSON.  Previously a tiny
    # compressed manifest/images.json could bypass the image-only size total and
    # consume unbounded RAM during z.read/json.loads.
    all_infos = z.infolist()
    _validate_backup_limits(
        (info.filename, info.file_size) for info in all_infos)
    metadata = {}
    for info in all_infos:
        if info.filename not in ('manifest.json', 'images.json'):
            continue
        if info.filename in metadata:
            raise ValueError(f'duplicate {info.filename} in backup')
        if info.file_size > _BACKUP_MAX_METADATA_BYTES:
            raise ValueError(f'{info.filename} is too large')
        metadata[info.filename] = info
    if set(metadata) != {'manifest.json', 'images.json'}:
        raise ValueError('not a dataset backup (manifest.json/images.json missing or invalid)')
    try:
        manifest = json.loads(z.read(metadata['manifest.json']).decode('utf-8'))
        images_meta = json.loads(z.read(metadata['images.json']).decode('utf-8'))
    except (ValueError, UnicodeError, RecursionError, MemoryError,
            RuntimeError, NotImplementedError, EOFError, zipfile.BadZipFile,
            zlib.error, lzma.LZMAError):
        raise ValueError('not a dataset backup (manifest.json/images.json missing or invalid)')
    if not isinstance(manifest, dict):
        raise ValueError('invalid backup manifest')
    if manifest.get('format') != BACKUP_FORMAT:
        raise ValueError('not a dataset backup')
    version = manifest.get('version')
    if (isinstance(version, bool) or not isinstance(version, int)
            or version < 1):
        raise ValueError('invalid backup version')
    if version > BACKUP_VERSION:
        raise ValueError('backup made by a newer version of the app - update first')
    manifest = _normalized_backup_manifest(manifest)
    restored_training_mode = manifest['training_mode']
    if not isinstance(images_meta, list):
        raise ValueError('invalid backup image metadata')
    _validate_backup_limits(row_count=len(images_meta))
    images_meta = [
        _normalized_backup_image_meta(meta, version=version)
        for meta in images_meta
    ]
    seen_backup_ids = set()
    metadata_image_names = {}
    rescue_sources = set()
    rescue_parent_counts = {}
    for meta in images_meta:
        if not isinstance(meta, dict):
            raise ValueError('invalid backup image metadata')
        filename = meta.get('filename')
        if filename is not None and not isinstance(filename, str):
            raise ValueError('invalid backup image filename')
        if filename is not None:
            portable_name = _backup_basename(filename)
            if portable_name != filename and version >= 2:
                raise ValueError('invalid backup image filename')
            if portable_name == filename:
                filename_key = filename.casefold()
                if filename_key in metadata_image_names and version >= 2:
                    raise ValueError('duplicate backup image filename')
                metadata_image_names[filename_key] = filename
        backup_id = meta.get('backup_image_id')
        if backup_id is not None:
            if isinstance(backup_id, bool) or not isinstance(backup_id, int) or backup_id <= 0:
                raise ValueError('invalid backup image id')
            if backup_id in seen_backup_ids:
                raise ValueError('duplicate backup image id')
            seen_backup_ids.add(backup_id)
        derivation = meta.get('derivation_kind')
        if derivation not in (None, SMALL_IMAGE_SOURCE, KLEIN_SMALL_IMAGE,
                              KLEIN_IMAGE_IMPROVE):
            raise ValueError('invalid image derivation in backup')
        if (version >= 2 and filename is None
                and derivation != KLEIN_SMALL_IMAGE):
            raise ValueError('invalid metadata-only image row in backup')
        if derivation == SMALL_IMAGE_SOURCE:
            if backup_id is None or meta.get('parent_image_id') is not None:
                raise ValueError('invalid small-image source provenance')
            rescue_sources.add(backup_id)
        elif derivation == KLEIN_SMALL_IMAGE:
            parent_id = meta.get('parent_image_id')
            if backup_id is None or isinstance(parent_id, bool) or not isinstance(parent_id, int):
                raise ValueError('invalid Klein rescue provenance')
            rescue_parent_counts[parent_id] = rescue_parent_counts.get(parent_id, 0) + 1
            if rescue_parent_counts[parent_id] > 1:
                raise ValueError('multiple Klein rescue candidates for one source')
    if any(parent_id not in rescue_sources for parent_id in rescue_parent_counts):
        raise ValueError('Klein rescue candidate has no valid source')
    infos = []
    for info in all_infos:
        if info.is_dir() or info.filename in ('manifest.json', 'images.json'):
            continue
        if info.filename.startswith('analysis-cache/'):
            continue
        if info.filename.startswith(('ref/', 'images/')):
            prefix, candidate = info.filename.split('/', 1)
            base = _backup_basename(candidate)
            if base is None or info.filename != f'{prefix}/{base}':
                if version >= 2:
                    raise ValueError('invalid image/ref filename in backup')
                continue
            infos.append(info)
            continue
        if version >= 2:
            raise ValueError(f'unexpected file in backup: {info.filename}')
    if len(infos) > _BACKUP_MAX_FILES:
        raise ValueError(f'too many files in backup (max {_BACKUP_MAX_FILES})')
    archive_names = {'ref': {}, 'images': {}}
    for info in infos:
        prefix, candidate = info.filename.split('/', 1)
        name = _backup_basename(candidate)
        if name:
            key = name.casefold()
            if key in archive_names[prefix]:
                raise ValueError(
                    f'backup has duplicate {prefix} filename: {name}')
            archive_names[prefix][key] = name
    collisions = set(archive_names['ref']).intersection(archive_names['images'])
    if collisions:
        collision = archive_names['images'][next(iter(collisions))]
        raise ValueError(f'backup has colliding ref/image filename: {collision}')
    if version >= 2:
        archive_image_keys = set(archive_names['images'])
        metadata_image_keys = set(metadata_image_names)
        missing_images = metadata_image_keys.difference(archive_image_keys)
        orphan_images = archive_image_keys.difference(metadata_image_keys)
        if missing_images:
            raise ValueError(
                f'backup image is missing: {metadata_image_names[next(iter(missing_images))]}')
        if orphan_images:
            raise ValueError(
                f'unreferenced image in backup: '
                f'{archive_names["images"][next(iter(orphan_images))]}')
        for key in archive_image_keys:
            if archive_names['images'][key] != metadata_image_names[key]:
                raise ValueError('backup image filename case does not match metadata')

        required_ref_names = []
        for field in ('ref_filename', 'ref_original_filename'):
            raw_name = manifest.get(field)
            if raw_name is None:
                continue
            name = _backup_basename(raw_name)
            if name != raw_name:
                raise ValueError(f'invalid backup {field}')
            required_ref_names.append(name)
        extra_ref_names = _backup_extra_ref_names(
            manifest.get('ref_extra_filenames'))
        required_ref_names.extend(extra_ref_names)
        required_ref_keys = {name.casefold() for name in required_ref_names}
        allowed_ref_keys = set(required_ref_keys)
        for name in extra_ref_names:
            original = extra_ref_original_name(name)
            if original:
                allowed_ref_keys.add(original.casefold())
        archive_ref_keys = set(archive_names['ref'])
        missing_refs = required_ref_keys.difference(archive_ref_keys)
        orphan_refs = archive_ref_keys.difference(allowed_ref_keys)
        if missing_refs:
            raise ValueError('backup reference image is missing')
        if orphan_refs:
            raise ValueError(
                f'unreferenced reference image in backup: '
                f'{archive_names["ref"][next(iter(orphan_refs))]}')

    # Derive cache ownership only after the exact set of restorable rows/files
    # has been validated. A crafted skipped row cannot smuggle an otherwise
    # unowned sidecar into the restored Dataset folder.
    requested_cache_refs = set()
    if version >= 2:
        for meta in images_meta:
            raw_snapshot = meta.get('bank_analysis_snapshot')
            snapshot = bank_transfer_metadata.parse_snapshot(raw_snapshot)
            if raw_snapshot is not None and snapshot is None:
                raise ValueError('invalid Bank analysis snapshot in backup')
            if snapshot and snapshot.get('cache_ref'):
                cache_ref = snapshot['cache_ref']
                if not bank_transfer_metadata.is_content_addressed_cache_ref(
                        cache_ref):
                    raise ValueError(
                        'backup analysis cache reference is not content-addressed')
                requested_cache_refs.add(cache_ref)
    cache_infos = {}
    seen_cache_refs = set()
    for info in all_infos:
        if info.is_dir():
            continue
        if not info.filename.startswith('analysis-cache/'):
            continue
        match = _BACKUP_ANALYSIS_CACHE_RE.fullmatch(info.filename)
        if not match:
            if version >= 2:
                raise ValueError('invalid analysis cache filename in backup')
            continue
        cache_ref = match.group('ref')
        if cache_ref in seen_cache_refs:
            raise ValueError(f'duplicate analysis cache in backup: {cache_ref}')
        seen_cache_refs.add(cache_ref)
        if version >= 2 and not bank_transfer_metadata.is_content_addressed_cache_ref(
                cache_ref):
            raise ValueError('backup analysis cache reference is not content-addressed')
        if cache_ref not in requested_cache_refs:
            if version >= 2:
                raise ValueError(f'unreferenced analysis cache in backup: {cache_ref}')
            continue
        if (info.file_size <= 0
                or info.file_size > bank_transfer_metadata.CACHE_SIDECAR_MAX_BYTES):
            raise ValueError('analysis cache sidecar is too large')
        cache_infos[cache_ref] = info
    missing_cache_refs = requested_cache_refs.difference(cache_infos)
    if missing_cache_refs:
        raise ValueError(
            f'analysis cache missing from backup: {sorted(missing_cache_refs)[0]}')

    # Validate and bind every requested cache before creating a staging folder or
    # opening a database transaction.  A v2 restore is all-or-nothing: a bad CRC,
    # truncated NPZ, wrong SHA or invalid shape aborts the whole archive instead
    # of silently deleting cache_ref and restoring weaker metadata.
    validated_cache_payloads = {}
    for cache_ref, info in cache_infos.items():
        try:
            with z.open(info) as source:
                raw = source.read(
                    bank_transfer_metadata.CACHE_SIDECAR_MAX_BYTES + 1)
        except (OSError, EOFError, RuntimeError, MemoryError,
                zipfile.BadZipFile, zlib.error, lzma.LZMAError) as exc:
            raise ValueError('analysis cache sidecar is unreadable') from exc
        if (len(raw) != info.file_size
                or len(raw) > bank_transfer_metadata.CACHE_SIDECAR_MAX_BYTES
                or bank_transfer_metadata.read_cache_sidecar_bytes(
                    raw, expected_ref=cache_ref) is None):
            raise ValueError(
                'analysis cache sidecar is malformed or has a digest mismatch')
        validated_cache_payloads[cache_ref] = raw
    name = (manifest.get('name') or 'Restored dataset')[:100]
    trigger = (manifest.get('trigger_word') or 'restored')[:60]
    # Extract first into a sibling directory: it is on the same volume as the final
    # dataset folder, so promotion is a single rename.  The database transaction is
    # only opened after extraction succeeds; no empty dataset can become visible.
    root = str(cfg.dataset_images_root())
    staging_dir = os.path.join(root, f'.restore-{uuid.uuid4().hex}.tmp')
    os.mkdir(staging_dir)
    final_dir = None
    promoted = False
    db_started = False
    try:
        extracted_images = set()
        extracted_refs = {}
        for info in infos:
            prefix, candidate = info.filename.split('/', 1)
            base = _backup_basename(candidate)
            if not base:
                continue   # nested path or weird name -> skip, never traverse
            # Decode/verify all archive image bytes before they ever reach the
            # restore staging directory. That keeps the eventual rename/promotion
            # atomic even for compact pixel bombs or content/extension lies.
            raw = _read_validated_backup_image(z, info, base)
            with open(os.path.join(staging_dir, base), 'wb') as dst:
                # Keep this copy seam (rather than ``dst.write(raw)``): it is
                # deliberately fault-injectable by the atomic restore regression.
                shutil.copyfileobj(io.BytesIO(raw), dst, 1024 * 1024)
            if prefix == 'ref':
                extracted_refs.setdefault(base.casefold(), base)
            else:
                extracted_images.add(base)

        extracted_cache_refs = set()
        if validated_cache_payloads:
            cache_dir = os.path.join(staging_dir, '.bank-analysis-cache')
            os.mkdir(cache_dir)
            for cache_ref, raw in validated_cache_payloads.items():
                cache_path = os.path.join(cache_dir, f'{cache_ref}.npz')
                try:
                    with open(cache_path, 'wb') as dst:
                        dst.write(raw)
                        dst.flush()
                        os.fsync(dst.fileno())
                except OSError as exc:
                    raise ValueError(
                        'could not restore analysis cache sidecar') from exc
                extracted_cache_refs.add(cache_ref)

        db_started = True
        ds = create_dataset(user_id, name, trigger, kind=manifest.get('kind'),
                            concept_desc=manifest.get('concept_desc'),
                            train_type=manifest.get('train_type'), commit=False)
        for field in ('concept_terms', 'train_variant', 'train_settings',
                      'best_settings', 'fidelity'):
            setattr(ds, field, manifest.get(field))
        ds.training_mode = restored_training_mode
        ds.train_base_model = _portable_train_base_model(manifest.get('train_base_model'))
        ds.ref_filename = _backup_basename(manifest.get('ref_filename'))
        ds.ref_original_filename = _backup_basename(
            manifest.get('ref_original_filename'))
        final_dir = os.path.join(root, str(ds.id))
        if os.path.exists(final_dir):
            # Never merge with or delete a pre-existing orphan directory.
            raise RuntimeError(f'dataset folder already exists for id {ds.id}')

        n_rows = 0
        restored_rows = []
        valid_source_ids = {
            meta.get('backup_image_id') for meta in images_meta
            if isinstance(meta, dict)
            and meta.get('derivation_kind') == SMALL_IMAGE_SOURCE
            and meta.get('filename') in extracted_images
        }
        for meta in images_meta:
            if not isinstance(meta, dict):
                continue
            fn = meta.get('filename')
            derivation = meta.get('derivation_kind')
            is_candidate = derivation == KLEIN_SMALL_IMAGE
            if fn and fn not in extracted_images:
                continue
            if not fn and not is_candidate:
                continue   # only rescue candidates have meaningful metadata-only rows
            if is_candidate and meta.get('parent_image_id') not in valid_source_ids:
                continue   # never restore an orphaned candidate
            values = {f: meta.get(f) for f in _BACKUP_IMG_FIELDS
                      if f not in ('filename', 'parent_image_id')}
            # Backup input is untrusted. Unknown/invalid provenance is dropped,
            # while valid Pexels or web-search metadata is canonicalized back to JSON TEXT.
            values['source_metadata'] = _source_metadata_storage(
                values.get('source_metadata'))
            snapshot = bank_transfer_metadata.parse_snapshot(
                values.get('bank_analysis_snapshot'))
            if (snapshot and snapshot.get('cache_ref')
                    and snapshot['cache_ref'] not in extracted_cache_refs):
                raise ValueError('analysis cache sidecar was not restored')
            values['bank_analysis_snapshot'] = (
                bank_transfer_metadata.normalized_snapshot_storage(snapshot)
                if snapshot is not None else None)
            raw_transfer_metadata = values.get('transfer_metadata')
            values['transfer_metadata'] = (
                bank_transfer_metadata.normalized_transfer_metadata_storage(
                    raw_transfer_metadata))
            if (raw_transfer_metadata is not None
                    and values['transfer_metadata'] is None):
                raise ValueError('invalid Bank/Dataset transfer metadata in backup')
            if is_candidate and not fn and values.get('status') in ('pending', 'keep'):
                values['status'] = 'failed'
                values['fail_reason'] = (
                    'Klein rescue was in flight when this backup was created; '
                    'the original image is preserved, but the job must be started again.'
                )
            img = FaceDatasetImage(dataset_id=ds.id,
                                   **values,
                                   filename=fn)
            db.session.add(img)
            restored_rows.append((img, meta))
            n_rows += 1
        # Allocate new ids first, then restore the graph strictly within this backup.
        # A missing/skipped parent clears the relationship rather than pointing at an
        # unrelated row that happens to reuse the old numeric id.
        db.session.flush()
        id_map = {meta.get('backup_image_id'): img.id for img, meta in restored_rows
                  if meta.get('backup_image_id') is not None}
        for img, meta in restored_rows:
            img.parent_image_id = id_map.get(meta.get('parent_image_id'))
        # Reference fields are rebuilt exclusively from actual ref/ archive files.
        # Never retain paths, missing names, image-only files, or case variants.
        ds.ref_filename = (extracted_refs.get(ds.ref_filename.casefold())
                           if ds.ref_filename else None)
        ds.ref_original_filename = (
            extracted_refs.get(ds.ref_original_filename.casefold())
            if ds.ref_original_filename else None)
        used_ref_keys = {
            ref.casefold() for ref in (ds.ref_filename, ds.ref_original_filename) if ref
        }
        restored_extras = []
        for requested in _backup_extra_ref_names(
                manifest.get('ref_extra_filenames'), limit=None):
            key = requested.casefold()
            actual = extracted_refs.get(key)
            if not actual or key in used_ref_keys:
                continue
            used_ref_keys.add(key)
            restored_extras.append(actual)
            if len(restored_extras) >= MAX_EXTRA_REFS:
                break
        ds.ref_extra_filenames = json.dumps(restored_extras)

        os.replace(staging_dir, final_dir)
        promoted = True
        db.session.commit()
    except Exception:
        try:
            if db_started:
                db.session.rollback()
        finally:
            if promoted and final_dir:
                shutil.rmtree(final_dir, ignore_errors=True)
        raise
    finally:
        # Exists on extraction/build/promotion failure; after promotion the old path
        # is already gone.  Never leave hidden partial restores behind.
        shutil.rmtree(staging_dir, ignore_errors=True)
    logger.info(f"dataset backup restored: '{name}' -> #{ds.id} ({n_rows} image rows)")
    return ds



# --- Borrow: face_dataset_service.py primitives -----------------------------
# MUST stay at the bottom of this file, same reason as in the sibling split
# modules: this module and face_dataset_service.py import names from each
# other, and whichever side loads first must find the other fully defined by
# the time the reach-back import resolves. A name owned by ANOTHER split module
# is imported from that module directly, never through the parent's re-export.
from .face_dataset_service import (
    get_dataset, create_dataset, normalize_source_metadata, _dataset_ingest_lock,
    _bank_analysis_cache_dir,
    _source_metadata_storage, _SMALL_IMAGE_DERIVATIONS, _dataset_dir,
    KLEIN_IMAGE_IMPROVE, KLEIN_SMALL_IMAGE, SMALL_IMAGE_SOURCE, logger,
)
# Owned by sibling split modules -- imported from the owner, never through
# face_dataset_service's re-export, so no block depends on the order in which
# the parent emits them.
from .reference_photos_service import MAX_EXTRA_REFS, extra_ref_original_name
from .dataset_import_service import _preserved_import_extension
