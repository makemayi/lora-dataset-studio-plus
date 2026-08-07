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
BACKUP_VERSION = 1
_BACKUP_MAX_FILES = 600
_BACKUP_MAX_ROWS = 600
_BACKUP_MAX_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB uncompressed (zip-bomb guard)
_BACKUP_MAX_METADATA_BYTES = 4 * 1024 * 1024
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
_BACKUP_IMG_FIELDS = ('filename', 'source', 'framing', 'variation_label', 'status',
                      'caption', 'caption_short', 'variation_prompt', 'face_score', 'face_state',
                      'upscale_ratio', 'watermark_state', 'watermark_bbox',
                      'watermark_regions', 'parent_image_id', 'derivation_kind',
                      'fail_reason', 'fail_kind', 'source_metadata',
                      'bank_analysis_snapshot')


def _backup_basename(value):
    """Return a portable image basename, or None for paths/invalid values."""
    if not isinstance(value, str) or not value:
        return None
    if '/' in value or '\\' in value or not _BACKUP_NAME_RE.fullmatch(value):
        return None
    return value


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
    except (OSError, EOFError, RuntimeError, MemoryError, zipfile.BadZipFile) as exc:
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
        # MAX_EXTRA_REFS now lives in reference_photos_service.py and is only
        # borrowed back into this module's namespace by the bottom-of-file
        # re-export (see that block) — it isn't defined yet at THIS point in
        # module load, so the default must be resolved here, at call time,
        # not as a def-time default expression.
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


def write_backup_zip(user_id: int, dataset_id: int, output: BinaryIO) -> None:
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
    # backup_image_id is archive-local only. It lets restore remap parent_image_id
    # to the newly allocated row ids instead of retaining ids from the source DB.
    images_meta = []
    for img in rows:
        row = dict({'backup_image_id': img.id},
                   **{f: getattr(img, f) for f in _BACKUP_IMG_FIELDS})
        # Archive a structured, revalidated object rather than the raw TEXT
        # column. A malformed legacy/local row can never export arbitrary links.
        row['source_metadata'] = normalize_source_metadata(img.source_metadata)
        # A snapshot is durable only when it has the expected version, fingerprint
        # and bounded analysis shape.  Invalid legacy/local text is deliberately
        # omitted rather than becoming an opaque payload in a portable backup.
        row['bank_analysis_snapshot'] = bank_transfer_metadata.normalized_snapshot_storage(
            img.bank_analysis_snapshot)
        images_meta.append(row)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=1))
        z.writestr('images.json', json.dumps(images_meta, ensure_ascii=False, indent=1))
        for n in ref_names:
            p = os.path.join(dsdir, n)
            z.write(p, f'ref/{n}')
        for img in rows:
            name = _backup_basename(img.filename)
            if not name:
                continue   # metadata-only small-rescue candidate
            p = os.path.join(dsdir, name)
            if os.path.isfile(p):
                z.write(p, f'images/{name}')


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
    except (OSError, ValueError) as exc:
        raise ValueError('zip archive is not seekable') from exc
    return archive, None


def import_backup_zip(user_id: int, archive: bytes | BinaryIO):
    """Restore a backup as a NEW dataset (never merges into an existing one).
    Hardened: manifest format/version check, per-entry filename whitelist (no
    separators/traversal), file-count and uncompressed-size caps. Returns the
    created FaceDataset."""
    stream, owned = _coerce_archive_stream(archive)
    try:
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
    if len(all_infos) > _BACKUP_MAX_FILES + 2:
        raise ValueError(f'too many files in backup (max {_BACKUP_MAX_FILES})')
    if sum(info.file_size for info in all_infos) > _BACKUP_MAX_BYTES:
        raise ValueError('backup too large (max 2 GB uncompressed)')
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
    except (ValueError, UnicodeError, RecursionError, MemoryError, zipfile.BadZipFile):
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
    for field in ('name', 'trigger_word'):
        value = manifest.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f'invalid backup {field}')
    restored_training_mode = manifest.get('training_mode', 'lora')
    if restored_training_mode not in ('lora', 'full_transformer'):
        raise ValueError('invalid backup training_mode')
    if not isinstance(images_meta, list):
        raise ValueError('invalid backup image metadata')
    if len(images_meta) > _BACKUP_MAX_ROWS:
        raise ValueError(f'too many image rows in backup (max {_BACKUP_MAX_ROWS})')
    seen_backup_ids = set()
    rescue_sources = set()
    rescue_parent_counts = {}
    for meta in images_meta:
        if not isinstance(meta, dict):
            raise ValueError('invalid backup image metadata')
        filename = meta.get('filename')
        if filename is not None and not isinstance(filename, str):
            raise ValueError('invalid backup image filename')
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
    infos = [i for i in all_infos
             if not i.is_dir() and i.filename.startswith(('ref/', 'images/'))]
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
            # while valid Pexels metadata is canonicalized back to JSON TEXT.
            values['source_metadata'] = _source_metadata_storage(
                values.get('source_metadata'))
            values['bank_analysis_snapshot'] = (
                bank_transfer_metadata.normalized_snapshot_storage(
                    values.get('bank_analysis_snapshot')))
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
    get_dataset, create_dataset, normalize_source_metadata,
    _source_metadata_storage, _SMALL_IMAGE_DERIVATIONS, _dataset_dir,
    KLEIN_IMAGE_IMPROVE, KLEIN_SMALL_IMAGE, SMALL_IMAGE_SOURCE, logger,
)
# Owned by sibling split modules -- imported from the owner, never through
# face_dataset_service's re-export, so no block depends on the order in which
# the parent emits them.
from .reference_photos_service import MAX_EXTRA_REFS, extra_ref_original_name
from .dataset_import_service import _preserved_import_extension
