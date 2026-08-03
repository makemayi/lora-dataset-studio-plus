"""Reference-photo CRUD for a face dataset: the primary reference's extra
refs (Nano Banana / Klein / API-engine identity locking) and the angle
reference photos ("pose slots", Krea 2 Edit only). Split out of
face_dataset_service.py (2026-08, Phase 1 of a multi-phase file split) —
pure move, no behavior change. See
docs/superpowers/plans/2026-08-03-split-face-dataset-service-phase1.md.
"""
import io
import json
import os
import shutil
import uuid
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from ..extensions import db
from ..models import RefPoseSlot
from . import trash

# Références ADDITIONNELLES par dataset (au-delà de la principale) : servent
# UNIQUEMENT Nano Banana (multi-images d'entrée) - Klein/crop/scoring restent
# sur la principale. Cap bas pour garder des payloads API légers.
MAX_EXTRA_REFS = 3

# A reference handed to an external image API is a disposable transport copy,
# never a reason to disclose a camera master byte-for-byte.  Keep the same 2048px
# API-transport convention as ``normalize_to_webp`` and a modest raw-input cap:
# three dataset extras plus transient anchors must not turn a base64 request into
# an unbounded allocation.  The cap applies again to persistent legacy refs read
# from disk, because a restored backup can contain a preserved JPEG/PNG/BMP.
EXTERNAL_REFERENCE_MAX_BYTES = 25 * 1024 * 1024
EXTERNAL_REFERENCE_MAX_SIDE = 2048


def extra_ref_filenames(ds) -> list:
    """Références additionnelles du dataset (JSON en base, parse tolérant)."""
    try:
        v = json.loads(ds.ref_extra_filenames or '[]')
    except (ValueError, TypeError):
        return []
    return [f for f in v if isinstance(f, str)] if isinstance(v, list) else []


def sanitize_external_reference(image_bytes: bytes, *, label: str = 'reference image') -> bytes:
    """Return a bounded, upright, metadata-free WebP for an external API.

    This is an egress boundary, deliberately separate from dataset storage:
    local masters stay untouched, while every API engine receives fresh pixels
    without EXIF/XMP/GPS/ICC payloads.  It also rejects unsupported/animated
    containers and unsafe headers before asking Pillow to decode pixels.
    """
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        raise ValueError(f'{label} must be a non-empty image')
    raw = bytes(image_bytes)
    if len(raw) > EXTERNAL_REFERENCE_MAX_BYTES:
        raise ValueError(
            f'{label} is too large (max {EXTERNAL_REFERENCE_MAX_BYTES // (1024 * 1024)} MiB)')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                # This shared header validator admits only static JPEG/PNG/WebP/BMP
                # and applies the global side/pixel budget before ``load``.
                _preserved_import_header_extension(source, label=label)
                source.load()
                oriented = ImageOps.exif_transpose(source)
                has_alpha = ('A' in oriented.getbands()
                             or 'transparency' in getattr(oriented, 'info', {}))
                if has_alpha:
                    rgba = oriented.convert('RGBA')
                    clean = Image.new('RGB', rgba.size, (255, 255, 255))
                    clean.paste(rgba, mask=rgba.getchannel('A'))
                else:
                    clean = Image.new('RGB', oriented.size)
                    clean.paste(oriented.convert('RGB'))
        clean.thumbnail((EXTERNAL_REFERENCE_MAX_SIDE, EXTERNAL_REFERENCE_MAX_SIDE),
                        Image.LANCZOS)
        out = io.BytesIO()
        # ``clean`` is a fresh canvas, so no source EXIF/XMP/GPS/ICC metadata can
        # reach an external provider even on Pillow format-specific save paths.
        clean.save(out, 'WEBP', quality=92)
        payload = out.getvalue()
    except ValueError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f'{label} rejected an unsafe image header') from exc
    except (OSError, UnidentifiedImageError, SyntaxError, MemoryError) as exc:
        raise ValueError(f'{label} is unreadable') from exc
    if len(payload) > EXTERNAL_REFERENCE_MAX_BYTES:
        raise ValueError(
            f'{label} is too large after preparation '
            f'(max {EXTERNAL_REFERENCE_MAX_BYTES // (1024 * 1024)} MiB)')
    return payload


def _all_ref_bytes(ds) -> list:
    """Safe external derivatives of primary then persistent extra references.

    The primary remains mandatory.  A missing/corrupt legacy extra is skipped
    like it was before, but it can never leak raw EXIF/GPS bytes to an API.
    """
    out = [_read_external_reference(_ref_path(ds), label='primary reference')]
    for fn in extra_ref_filenames(ds):
        p = os.path.join(_dataset_dir(ds.id), fn)
        try:
            out.append(_read_external_reference(p, label='extra reference'))
        except ValueError:
            # An old/hand-edited extra is optional identity context. Do not turn
            # it into an opaque provider error, but do not send raw fallback bytes.
            logger.warning('dataset %s: skipping unavailable/unsafe extra reference', ds.id)
    return out


def _extra_ref_paths(ds) -> list:
    """Existing extra reference photos as full paths, for callers (face
    scoring) that want every reference this dataset has, not just the
    primary. Missing/deleted files are silently skipped — same tolerance
    _all_ref_bytes already applies to a corrupt/unreadable extra."""
    out = []
    for fn in extra_ref_filenames(ds):
        p = os.path.join(_dataset_dir(ds.id), fn)
        if os.path.isfile(p):
            out.append(p)
    return out


_EXTRA_REF_MARKER = '_datasetrefx_'
_EXTRA_REF_ORIG_MARKER = '_datasetrefxorig_'


def extra_ref_original_name(filename):
    """Name of the full-frame ORIGINAL kept beside an extra reference
    (`..._datasetrefx_<id>.webp` -> `..._datasetrefxorig_<id>.webp`), or None when
    the name doesn't follow the convention. A NAMING convention rather than a new
    column: extras live in `ref_extra_filenames`, a JSON list of names inside a
    schema that user databases froze long ago — deriving the companion needs no
    migration and restores from a backup as-is."""
    if not isinstance(filename, str) or _EXTRA_REF_ORIG_MARKER in filename:
        return None
    if _EXTRA_REF_MARKER not in filename:
        return None
    return filename.replace(_EXTRA_REF_MARKER, _EXTRA_REF_ORIG_MARKER, 1)


def extra_ref_crop_source(ds, filename) -> str:
    """The file the ✂ editor must display for an extra reference: the kept
    full-frame ORIGINAL when there is one, else the extra itself (still fully
    croppable — see crop_extra_ref, which snapshots it on the first crop)."""
    orig = extra_ref_original_name(filename)
    if orig and os.path.isfile(os.path.join(_dataset_dir(ds.id), orig)):
        return orig
    return filename


def add_extra_ref(user_id, dataset_id, image_bytes) -> str:
    """Ajoute une référence additionnelle. Normalisée WEBP ratio conservé, SANS
    head-crop GPU : un plan buste/corps est une bonne réf d'identité pour Nano
    Banana, et l'upload ne doit pas dépendre de la fenêtre GPU. Retourne le nom
    de fichier ; ValueError si dataset absent, réf principale manquante ou cap."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('set the primary reference first')
    extras = extra_ref_filenames(ds)
    if len(extras) >= MAX_EXTRA_REFS:
        raise ValueError(f'{MAX_EXTRA_REFS} extra references max')
    fn = f"{user_id}_datasetrefx_{uuid.uuid4().hex[:8]}.webp"
    dsdir = _dataset_dir(dataset_id)
    # Keep the full-frame ORIGINAL beside it (same deal as the primary reference):
    # ✂ Crop reads the original, so a re-crop can widen back out instead of only
    # eating further into the previous crop.
    orig_fn = extra_ref_original_name(fn)
    write_image_atomic(os.path.join(dsdir, orig_fn),
                       normalize_to_webp(image_bytes, size=2048))
    write_image_atomic(os.path.join(dsdir, fn), normalize_to_webp(image_bytes))
    ds.ref_extra_filenames = json.dumps(extras + [fn])
    db.session.commit()
    return fn


def crop_extra_ref(user_id, dataset_id, filename, x, y, w, h) -> bool:
    """Manually crop ONE extra reference to (x,y,w,h), long side capped at 1024
    (never enlarged - a smaller box keeps its own pixels).
    The box is in the crop SOURCE's pixel space (what extra_ref_crop_source names,
    i.e. what the editor displayed) and the result overwrites the extra only — the
    original stays untouched, so re-crops widen as freely as they tighten.

    `filename` is client-supplied: membership in the dataset's stored extras is the
    path guard (identical to remove_extra_ref) — nothing derived from it is opened
    before that check passes."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    extras = extra_ref_filenames(ds)
    if filename not in extras:
        return False
    dsdir = _dataset_dir(dataset_id)
    dst = os.path.join(dsdir, filename)
    if not os.path.isfile(dst):
        return False
    orig = extra_ref_original_name(filename)
    src = os.path.join(dsdir, orig) if orig else None
    if src and not os.path.isfile(src):
        # Retrofit for extras imported before originals were kept: what's on disk
        # IS still the uncropped full frame (cropping is the only thing that ever
        # rewrites an extra), so snapshotting it now costs one copy and gives those
        # datasets the same widen-back-out behaviour as a fresh import — instead of
        # "works for future imports only".
        shutil.copyfile(dst, src)
    ok, _scale = _crop_resize_file(src or dst, x, y, w, h, dst=dst)
    return ok


def remove_extra_ref(user_id, dataset_id, filename) -> bool:
    """Retire une référence additionnelle, en plaçant son fichier en corbeille."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    extras = extra_ref_filenames(ds)
    if filename not in extras:
        return False
    original_path = os.path.join(_dataset_path(dataset_id), filename)
    trashed_path = None
    if os.path.exists(original_path):
        trashed_path = trash.send_to_trash(
            original_path, context=f'dataset-{dataset_id}-extra-ref')
    try:
        ds.ref_extra_filenames = json.dumps([f for f in extras if f != filename])
        db.session.commit()
    except Exception:
        db.session.rollback()
        _restore_from_trash(trashed_path, original_path)
        raise
    # The kept original follows its extra to the trash — never leave it orphaned in
    # the dataset folder. Best effort: losing the extra itself is what matters.
    orig = extra_ref_original_name(filename)
    orig_path = os.path.join(_dataset_path(dataset_id), orig) if orig else None
    if orig_path and os.path.exists(orig_path):
        try:
            trash.send_to_trash(orig_path, context=f'dataset-{dataset_id}-extra-ref')
        except OSError:
            logger.warning(f'dataset {dataset_id}: could not trash extra-ref original {orig}')
    return True


# --- Angle reference photos (pose slots) — Krea 2 Edit only -----------------
# `front` is NOT one of these — it stays FaceDataset.ref_filename/ref_original_filename.
POSE_SLOT_KEYS = ('left45', 'right45', 'back', 'left90', 'right90')
# v1 wires only these two through the upload UI and the Krea detector's direct
# hits. The other three exist in the schema/API today (a later wave opens the
# UI button, nothing else) — 'back' already resolves through
# krea_pose_direction, it just has no enabled row to find yet on any dataset.
POSE_SLOT_ACTIVE_KEYS = ('left45', 'right45')


def _pose_slot_row(ds, pose_key):
    return RefPoseSlot.query.filter_by(dataset_id=ds.id, pose_key=pose_key).first()


def pose_slot_rows(ds) -> dict:
    """{pose_key: RefPoseSlot} for every row this dataset actually has — a
    pose_key with no upload yet is simply absent, never a placeholder row."""
    rows = RefPoseSlot.query.filter_by(dataset_id=ds.id).all()
    return {r.pose_key: r for r in rows}


def enabled_pose_slot_paths(ds) -> dict:
    """{pose_key: absolute path} for rows the user enabled AND whose file still
    exists on disk. The ONLY reader Krea generation trusts — an uploaded-but-
    disabled or since-deleted file must never be picked up silently."""
    dsdir = _dataset_dir(ds.id)
    out = {}
    for pose_key, row in pose_slot_rows(ds).items():
        if row.enabled and row.filename:
            path = os.path.join(dsdir, row.filename)
            if os.path.isfile(path):
                out[pose_key] = path
    return out


def set_pose_slot(user_id, dataset_id, pose_key, image_bytes) -> str:
    """Upload/replace ONE angle reference. Head-cropped like the PRIMARY
    reference (it substitutes for it during Krea generation) — a different
    role from the aspect-preserved extra-ref upload. Enabling is a SEPARATE,
    explicit step (set_pose_slot_enabled): uploading alone never turns a slot
    on, and re-uploading over an already-enabled slot leaves it enabled.
    Returns the new filename; ValueError on an unknown pose_key or dataset."""
    if pose_key not in POSE_SLOT_KEYS:
        raise ValueError('invalid pose_key')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    dsdir = _dataset_dir(dataset_id)
    webp = face_crop_to_square_webp(image_bytes, pad=REF_CROP_PAD, use_vision=False)
    uid = uuid.uuid4().hex[:8]
    orig_fn = f"{user_id}_datasetrefpose_{pose_key}orig_{uid}.webp"
    fn = f"{user_id}_datasetrefpose_{pose_key}_{uid}.webp"
    write_image_atomic(os.path.join(dsdir, orig_fn), normalize_to_webp(image_bytes, size=2048))
    write_image_atomic(os.path.join(dsdir, fn), webp)

    row = _pose_slot_row(ds, pose_key)
    old_fn, old_orig = (row.filename, row.original_filename) if row else (None, None)
    if row is None:
        row = RefPoseSlot(dataset_id=dataset_id, pose_key=pose_key, enabled=False)
        db.session.add(row)
    row.filename = fn
    row.original_filename = orig_fn
    db.session.commit()

    for stale in (old_fn, old_orig):
        if not stale:
            continue
        stale_path = os.path.join(dsdir, stale)
        if os.path.isfile(stale_path):
            try:
                trash.send_to_trash(stale_path, context=f'dataset-{dataset_id}-pose-{pose_key}')
            except OSError:
                logger.warning(
                    f'dataset {dataset_id}: could not trash stale pose slot file {stale}')
    return fn


def crop_pose_slot(user_id, dataset_id, pose_key, x, y, w, h) -> bool:
    """Manually crop ONE angle reference to (x,y,w,h), same contract as
    crop_reference: the box is in the ORIGINAL's pixel space, and only
    `filename` (never `original_filename`) is overwritten — so re-cropping can
    widen back out any number of times.

    If mirror_pose_slot ran first, this silently undoes it: the crop always
    re-derives `filename` from the untouched `original_filename` — known
    limitation, not handled."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    row = _pose_slot_row(ds, pose_key)
    if not row or not row.filename:
        return False
    dsdir = _dataset_dir(dataset_id)
    src = os.path.join(dsdir, row.original_filename) if row.original_filename else None
    if not src or not os.path.isfile(src):
        src = os.path.join(dsdir, row.filename)
    ok, _scale = _crop_resize_file(src, x, y, w, h, dst=os.path.join(dsdir, row.filename))
    return ok


def mirror_pose_slot(user_id, dataset_id, pose_key) -> bool:
    """Flip ONE angle reference horizontally, in place, same filename. Reuses
    the pixel primitive `_mirrored_image_bytes` (see `mirror_image`) but skips
    `_edit_image_in_place`: that wrapper is keyed to a FaceDatasetImage row and
    clears watermark metadata a pose slot doesn't have. `original_filename` (the
    pre-crop full frame) is untouched, same as `mirror_image`'s contract.

    A later crop_pose_slot call re-derives `filename` from the untouched
    `original_filename` and will silently undo this mirror — known
    limitation, not handled.

    The row lookup, path resolution and existence check all happen INSIDE the
    lock: nothing about which file gets mirrored may be decided before the
    lock is held, or a concurrent set_pose_slot on the same slot could swap
    the file out from under an in-flight mirror."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    lock = _IMAGE_PIXEL_EDIT_LOCKS[
        hash((str(user_id), 'pose', dataset_id, pose_key)) % len(_IMAGE_PIXEL_EDIT_LOCKS)]
    with lock:
        row = _pose_slot_row(ds, pose_key)
        if not row or not row.filename:
            return False
        path = os.path.join(_dataset_dir(dataset_id), row.filename)
        if not os.path.isfile(path):
            return False
        payload = _mirrored_image_bytes(path)
        write_image_atomic(path, payload)
    return True


def set_pose_slot_enabled(user_id, dataset_id, pose_key, enabled) -> bool:
    """Explicit user opt-in/out. Refused (False) when nothing has been
    uploaded to this pose_key yet — there is nothing to enable."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    row = _pose_slot_row(ds, pose_key)
    if not row or not row.filename:
        return False
    row.enabled = bool(enabled)
    db.session.commit()
    return True


def remove_pose_slot(user_id, dataset_id, pose_key) -> bool:
    """Delete ONE angle reference (row + both files, trashed not unlinked)."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    row = _pose_slot_row(ds, pose_key)
    if not row:
        return False
    dsdir = _dataset_path(dataset_id)
    fn, orig = row.filename, row.original_filename
    original_path = os.path.join(dsdir, fn) if fn else None
    trashed_path = None
    if original_path and os.path.exists(original_path):
        trashed_path = trash.send_to_trash(
            original_path, context=f'dataset-{dataset_id}-pose-{pose_key}')
    try:
        db.session.delete(row)
        db.session.commit()
    except Exception:
        db.session.rollback()
        _restore_from_trash(trashed_path, original_path)
        raise
    # The kept original follows its file to the trash — never leave it orphaned in
    # the dataset folder. Best effort: losing the main file itself is what matters.
    orig_path = os.path.join(dsdir, orig) if orig else None
    if orig_path and os.path.exists(orig_path):
        try:
            trash.send_to_trash(orig_path, context=f'dataset-{dataset_id}-pose-{pose_key}')
        except OSError:
            logger.warning(f'dataset {dataset_id}: could not trash pose slot original {orig}')
    return True


# --- Borrow: face_dataset_service.py primitives -----------------------------
# MUST stay at the bottom of this file, for the same reason face_dataset_service.py
# keeps ITS reverse borrow-back import at ITS bottom (see the comment there):
# this module and face_dataset_service.py import specific names from each
# other — a genuine two-way dependency. Loading either module in isolation
# (e.g. `import reference_photos_service` with nothing else imported yet)
# transitively loads the other, whose own bottom import then reaches back
# into THIS module for the names defined above. If this import ran at the
# top of the file instead, that reach-back would find a partially loaded
# module — none of the constants/functions above would exist yet — and raise
# ImportError. Placing it last guarantees this module has already defined
# every name it owns by the time either side's import actually resolves,
# regardless of which of the two modules gets imported first.
from .face_dataset_service import (
    get_dataset, _ref_path, _dataset_dir, _dataset_path,
    _read_external_reference, _preserved_import_header_extension,
    _restore_from_trash, _crop_resize_file, _mirrored_image_bytes,
    face_crop_to_square_webp, write_image_atomic, normalize_to_webp,
    REF_CROP_PAD, _IMAGE_PIXEL_EDIT_LOCKS, logger,
)
