"""Subject trim — the batch: masks, manifests, apply, undo.

Split from the decision module on purpose. Everything here touches disk, the
database or SAM3; everything in `subject_trim` is pure. The withdrawn
2026-08-16 wave put this in `dataset_generation_service` (already past 4000
lines) and cropped in the same pass that computed the frames, so the first time
anyone saw a crop it had already replaced the original.

THE TWO-PHASE SHAPE, AND WHY
----------------------------
`build_preview` runs the masks and writes `trim-preview.json`. It writes no
pixels. `apply_preview` takes the image ids the user confirmed, reads their
frames back out of that manifest, and only then crops. The client never sends
coordinates: it says which rows it accepted, so a stale or hand-edited request
cannot ask for an arbitrary crop.

Applying moves each original to the app Trash and appends to `trim-undo.json`,
so a whole batch can still be reverted after the fact — the review catches
"this one should not be cropped", the undo catches "it looked right and was
not".
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass

from PIL import Image, ImageOps

from ..models import FaceDatasetImage
from . import auto_mask, dataset_activity, image_encoding, trash
from .face_dataset_service import (
    get_dataset, write_image_atomic, _dataset_path, _valid_icc_profile,
)
from .subject_trim import (
    LONG_EDGE, largest_bbox, output_size, skip_reason, trim_frame,
)

logger = logging.getLogger(__name__)

# Owned by dataset_activity, which composes STOPPABLE_KINDS out of it — the same
# way dataset_generation_service reads IMPROVE_KINDS rather than restating it. A
# second copy here could not disagree in value, but it could disagree in SCOPE
# the day the arming scope changes, and only one of the two would be found.
TRIM_KINDS = dataset_activity.TRIM_KINDS
PREVIEW_FILE = 'trim-preview.json'
UNDO_FILE = 'trim-undo.json'
MASK_PROMPT = 'person'


@dataclass
class PreviewRow:
    image_id: int
    filename: str


def _row_id(item):
    """A manifest row's image id, or ``None`` when it has none that parses.

    Tolerant on purpose: these manifests are disk state that survives upgrades
    and can be hand-edited, and the one thing a malformed row may never do is
    raise out of a batch that has already moved somebody's originals to Trash.
    """
    try:
        return int(item['image_id'])
    except (KeyError, TypeError, ValueError):
        return None


def _preview_path(dataset_id):
    return os.path.join(_dataset_path(dataset_id), PREVIEW_FILE)


def _undo_path(dataset_id):
    return os.path.join(_dataset_path(dataset_id), UNDO_FILE)


def _read_json(path):
    """The manifest at `path`, or ``None``.

    A missing file and an UNREADABLE one both answer ``None`` — the callers
    have nothing useful to do with the difference — but only one of them is
    normal. An unreadable `trim-undo.json` silently stops the UI offering the
    undo while the originals sit in Trash, so it says so in the log. Only the
    basename is logged: this file's contents are a user's photo names.
    """
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            loaded = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.exception('subject trim: %s is unreadable', os.path.basename(path))
        return None
    # Every manifest this module writes is an object, and every caller reaches
    # for a key. A hand-edited file holding a list would otherwise crash them
    # with an AttributeError instead of the "no preview" they know how to say.
    if not isinstance(loaded, dict):
        logger.warning('subject trim: %s is not a manifest', os.path.basename(path))
        return None
    return loaded


def _write_json(path, payload):
    """Publish a manifest in one step: it is either the old one or the new one.

    The same shape `write_image_atomic` and `auto_mask` use, for the same
    reason — `trim-undo.json` is rewritten after every single image now, so a
    process death lands in that window far more often than it used to, and a
    truncated undo manifest is a batch of originals nobody can put back.
    """
    tmp = f'{path}.part'
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def build_preview(ds_dir, rows, progress=None, should_stop=None):
    """Mask every row and return ``{'items': [...]}`` without writing pixels.

    One item per row, ALWAYS — a skipped image keeps its entry and its reason
    so the review screen can show what was passed over and why. `frame` is
    ``None`` only when no frame could be computed at all.

    All the masking happens in ONE child process (`auto_mask.masks_for`); the
    per-row loop that follows is arithmetic. `progress(done, total, phase=None)`
    is called throughout: with a `phase` other than 'masking' while the child
    is still starting up or loading its checkpoint (no `done` to report yet),
    then with the running count once masking itself starts.
    """
    paths = [os.path.join(ds_dir, row.filename) for row in rows]
    total = len(rows)

    def _on_mask_progress(record):
        if not progress:
            return
        phase = record.get('phase')
        if phase == 'masking':
            done = int(record.get('done') or 0)
            # The child only ever sees the cache MISSES, so its total is short
            # by however many were already cached — those are done before it
            # starts. Report against the selection the user actually picked.
            child_total = int(record.get('total') or total) or total
            already = max(0, total - child_total)
            progress(min(total, already + done), total, 'masking')
        else:
            progress(0, total, phase)

    masks = auto_mask.masks_for(paths, MASK_PROMPT,
                                on_progress=_on_mask_progress,
                                should_stop=should_stop)
    items = []
    for row, path in zip(rows, paths):
        item = {'image_id': row.image_id, 'filename': row.filename,
                 'image': None, 'frame': None, 'out': None, 'skip': None}
        mask_path, reason = masks.get(path, (None, 'failed'))
        if reason:
            # Three different answers, three different words. 'no-match' is
            # about the picture, 'stopped' is about the user, 'failed' is about
            # the engine — folding the middle one into the last is how a Stop
            # comes back looking like a crash.
            item['skip'] = {'no-match': 'no-subject-found',
                            'cancelled': 'stopped'}.get(reason, 'failed')
        else:
            try:
                with Image.open(mask_path) as mask:
                    iw, ih = mask.width, mask.height
                    bbox = largest_bbox(mask)
                item['image'] = [iw, ih]
                if bbox is None:
                    item['skip'] = 'no-subject-found'
                else:
                    frame = trim_frame(iw, ih, bbox)
                    item['frame'] = list(frame)
                    item['out'] = list(output_size(frame[2], frame[3]))
                    item['skip'] = skip_reason(iw, ih, frame)
            except Exception:                   # noqa: BLE001
                logger.exception('subject trim preview: image %s failed', row.image_id)
                item['skip'] = 'failed'
        items.append(item)
    progress and progress(total, total)
    return {'items': items}


def preview_report(dataset_id):
    """The pending preview manifest, or ``None``."""
    return _read_json(_preview_path(dataset_id))


def clear_preview(dataset_id):
    try:
        os.remove(_preview_path(dataset_id))
    except OSError:
        pass


def start_preview(app, user_id, dataset_id, image_ids):
    """Run the masks over ``image_ids`` in the background and leave a preview
    manifest behind. Returns the launch snapshot ``{'queued': n}``.

    Raises ValueError (400) on an unknown dataset or an empty selection,
    RuntimeError (409) when a trim pass is already running, and
    `auto_mask.AutoMaskUnavailable` (409) when the mask environment is not set
    up — refused UP FRONT, because a per-image failure would be counted as a
    per-image skip and the batch would silently propose nothing.
    """
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if dataset_activity.running(dataset_id, TRIM_KINDS):
        raise RuntimeError('a trim pass is already running on this dataset')
    if not auto_mask.is_available():
        raise auto_mask.AutoMaskUnavailable(
            'the automatic-masking environment is not set up')
    wanted = {int(i) for i in (image_ids or [])}
    if not wanted:
        raise ValueError('no images selected')
    rows = [PreviewRow(r.id, r.filename)
            for r in FaceDatasetImage.query.filter_by(dataset_id=dataset_id).all()
            if r.id in wanted and r.filename]
    total = len(rows)
    if not total:
        raise ValueError('no images selected')
    token = dataset_activity.begin(dataset_id, 'trim', total=total,
                                    detail=f'Measuring crops… 0/{total}')
    clear_preview(dataset_id)

    def _run():
        try:
            with app.app_context():
                ds_dir = _dataset_path(dataset_id)
                manifest = build_preview(
                    ds_dir, rows,
                    progress=lambda done, tot, phase=None: dataset_activity.progress(
                        token, done=done,
                        detail=('Loading the masking model…'
                                if phase and phase != 'masking'
                                else f'Measuring crops… {done}/{tot}')),
                    should_stop=lambda: dataset_activity.cancel_requested(
                        dataset_id, TRIM_KINDS))
                _write_json(_preview_path(dataset_id), manifest)
        except Exception:   # noqa: BLE001 — a crash here must not strand the banner
            logger.exception('trim preview failed on dataset %s', dataset_id)
            _write_json(_preview_path(dataset_id),
                        {'items': [], 'error': 'The crop preview failed. See the '
                                               'server log for the details.'})
        finally:
            dataset_activity.end(token)
            dataset_activity.clear_cancel(dataset_id, TRIM_KINDS)

    if app.config.get('TESTING'):
        _run()
    else:
        threading.Thread(target=_run, daemon=True,
                          name=f'trim-preview-{dataset_id}').start()
    return {'queued': total}


def trim_report(dataset_id):
    """The last applied batch's undo manifest — ``{batch, entries, counts}`` —
    or ``None``. The UI reads it to decide whether to offer the undo.

    ONE batch deep, on purpose: applying a FRESH preview overwrites it. A retry
    over the rows a partial apply left behind is the same batch, not a new one,
    and extends this manifest instead of replacing it — see `apply_preview`.
    """
    return _read_json(_undo_path(dataset_id))


def apply_preview(user_id, dataset_id, image_ids):
    """Crop the confirmed rows in place and return
    ``{'trimmed', 'refused', 'failed'}``.

    Coordinates come from the manifest, never from the caller — the caller only
    says which rows it accepted. A row the manifest marked as skipped is
    REFUSED rather than cropped: the rule already rejected it, and a stale
    screen must not be able to talk the server out of that. An id that is not
    in the manifest at all is refused for the same reason and counted the same
    way — there is no frame to honour, so nothing is guessed.

    Each replaced original goes to the app Trash and is appended to the undo
    manifest — on disk, after every single image — so the batch can be reverted
    in one action afterwards even if the process dies halfway through it.

    The preview is CLAIMED for the duration and handed back holding whatever
    was not trimmed, so two overlapping applies cannot both act on the same
    rows and a pass in which most rows failed does not cost a masking run.
    Raises ValueError when there is no preview to claim.
    """
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # The caller's own input is parsed BEFORE the claim below. Anything that
    # can be refused without touching disk state should be.
    wanted = {int(i) for i in (image_ids or [])}
    # CLAIM the manifest before reading a single row. `apply_preview` has no
    # activity guard, so a double-clicked Apply or a retried POST used to run
    # twice over the same rows: A trashes the original and writes crop₁, B
    # trashes CROP₁ and writes crop₂, and the undo pointer ends up naming the
    # intermediate while the true original sits in a timestamped Trash folder
    # nothing references. `os.replace` is atomic on both platforms, so exactly
    # one of the two gets the rows and the loser is told there is no preview.
    claim = _preview_path(dataset_id) + '.applying'
    try:
        os.replace(_preview_path(dataset_id), claim)
    except OSError:
        raise ValueError('no crop preview to apply — run the preview first')
    manifest = _read_json(claim)
    items = (manifest or {}).get('items') or []
    if not items:
        # Nothing to apply — but the claim must not swallow what the review
        # screen is still showing (a failed preview keeps an `error` manifest).
        try:
            os.replace(claim, _preview_path(dataset_id))
        except OSError:
            logger.exception('subject trim: could not release the preview claim')
        raise ValueError('no crop preview to apply — run the preview first')
    by_id = {_row_id(it): it for it in items if _row_id(it) is not None}
    ds_dir = _dataset_path(dataset_id)
    counts = {'trimmed': 0, 'refused': 0, 'failed': 0}
    # A retry over what a partial apply left behind CONTINUES the same undo
    # batch. The preview now keeps its untrimmed rows (below), so a second
    # Apply is an ordinary thing to click — and without this stamp it would
    # write a fresh undo manifest over the first pass's, leaving those
    # originals in Trash with nothing in the app pointing at them. A preview
    # built by `build_preview` carries no stamp, so it opens a new batch and
    # the undo stays one batch deep, exactly as `trim_report` describes.
    batch = manifest.get('batch') or uuid.uuid4().hex
    previous = trim_report(dataset_id) or {}
    carried = previous.get('entries')
    entries = (list(carried)
               if previous.get('batch') == batch and isinstance(carried, list)
               else [])

    def _record():
        # The undo manifest lands after EVERY image, not once at the end. The
        # originals are already in Trash by then; a restart or an OOM halfway
        # through a lossless batch used to leave them there with no manifest at
        # all, and the trash folders are named by timestamp, not by filename,
        # so putting them back meant opening every one by hand. One small JSON
        # write per image is noise next to a LANCZOS resize plus that encode.
        # `counts` is this CALL's tally; the manifest's `trimmed` describes the
        # manifest, so it counts the entries in it — the two differ only after
        # a continuation, and neither may ever disagree with what it names.
        _write_json(_undo_path(dataset_id),
                    {'batch': batch, 'entries': entries,
                     'counts': dict(counts, trimmed=len(entries))})

    def _dispose():
        """Hand the preview back and drop the claim. Runs on EVERY exit.

        The preview keeps every row that was NOT trimmed instead of being
        binned wholesale: confirming 40 rows and having 1 succeed used to throw
        away a manifest that cost a full 3.4 GB-checkpoint masking run. It also
        closes the retry hole from the other side — a second Apply structurally
        cannot re-crop a row that is no longer in the preview. And because it
        is a `finally`, an unexpected escape leaves the preview readable under
        its own name rather than stranded under the claim, minus whatever the
        batch already did.
        """
        if entries:
            _record()           # once more, with the final counts
        done = {e.get('image_id') for e in entries}
        remaining = [it for it in items if _row_id(it) not in done]
        if remaining:
            _write_json(_preview_path(dataset_id),
                        dict(manifest, items=remaining, batch=batch))
        try:
            os.remove(claim)
        except OSError:
            pass

    # Sorted, not set order: the undo manifest is a record of what happened to
    # a user's files, and a record whose order changes run to run is one nobody
    # can diff against the screen they clicked.
    try:
        for image_id in sorted(wanted):
            item = by_id.get(image_id)
            if not item or item.get('skip') or not item.get('frame'):
                counts['refused'] += 1
                continue
            try:
                # Inside the try, all of it. Unpacking a frame with the wrong arity
                # or a non-numeric value used to raise straight out of the loop —
                # after the previous images had already been trashed. The manifest
                # is server-written today, but it is disk state that survives
                # upgrades and can be hand-edited. `basename` for the same reason:
                # a `..` component would otherwise aim send_to_trash at a file
                # outside the dataset folder.
                filename = os.path.basename(item['filename'])
                path = os.path.join(ds_dir, filename)
                fx, fy, fw, fh = (int(v) for v in item['frame'])
                with Image.open(path) as opened:
                    # The frame describes the picture as it was MEASURED. Anything
                    # that rewrites the same file in between — mirror, rotate, a
                    # manual crop, a watermark inpaint — leaves coordinates that no
                    # longer describe it, and the clamp below would turn that into
                    # a plausible-looking wrong crop instead of a refusal.
                    if item.get('image') and list(opened.size) != list(item['image']):
                        raise ValueError(
                            f'{filename} changed since the crop was measured')
                    # The file keeps its own format — a `.png` must not come back
                    # holding WEBP bytes — and its colour profile.
                    fmt = image_encoding.format_for_path(path, opened)
                    opened.load()
                    icc = _valid_icc_profile(opened.info.get('icc_profile'))
                    # The frame is in RAW RASTER space: the mask children
                    # (infer/sam3_mask_infer.py, infer/mask_infer.py) open the file
                    # and convert, they never exif_transpose. Cutting the raw
                    # buffer and THEN baking the orientation into the cut is the
                    # same picture — `Image.crop` carries `info['exif']` through,
                    # so the two commute — and it is the only order that lands on
                    # the region the mask actually measured. Cropping the oriented
                    # image instead put an ordinary portrait phone photo
                    # (orientation 6, raw 4:3 landscape) somewhere else entirely,
                    # and the clamp absorbed the overflow so nothing raised.
                    box = (max(0, fx), max(0, fy),
                           min(opened.width, fx + fw), min(opened.height, fy + fh))
                    if box[2] <= box[0] or box[3] <= box[1]:
                        raise ValueError(f'empty crop box {item["frame"]}')
                    oriented = ImageOps.exif_transpose(opened.crop(box))
                    # Narrow the mode BEFORE resampling: Pillow silently drops to
                    # nearest-neighbour on paletted images.
                    crop = oriented.convert(image_encoding.resample_mode(oriented))
                out_w, out_h = output_size(crop.width, crop.height, LONG_EDGE)
                if (out_w, out_h) != crop.size:
                    crop = crop.resize((out_w, out_h), Image.LANCZOS)
                buf = io.BytesIO()
                image_encoding.save_edit(crop, buf, fmt, image_encoding.LOSSLESS,
                                         icc_profile=icc)
                # ENCODE, then move the original, then publish. Everything that can
                # fail for a reason other than the disk has already failed by this
                # line, with the original still in its folder; and the entry is on
                # DISK before the write, so even a write that dies mid-way leaves
                # an undo that can put the original back.
                trashed = trash.send_to_trash(
                    path, context=f'dataset-{dataset_id}-trim-{image_id}')
                entries[:] = [e for e in entries if e.get('image_id') != image_id]
                entries.append({'image_id': image_id, 'filename': filename,
                                'trashed': trashed})
                # Counted here, with the entry, so `trimmed` and the entry list can
                # never disagree. A publish that fails after this point is logged
                # and stays counted: its original is in Trash with an undo entry
                # naming it, which is the state the count has to describe.
                counts['trimmed'] += 1
                _record()
                write_image_atomic(path, buf.getvalue())
            except Exception:                       # noqa: BLE001
                logger.exception('subject trim: image %s failed', image_id)
                if not (entries and entries[-1].get('image_id') == image_id):
                    counts['failed'] += 1
    finally:
        _dispose()
    return counts


def restore_trim_batch(user_id, dataset_id):
    """Whole-batch undo: every original the last applied batch trashed comes
    back and overwrites the cropped file; the manifest is cleared.

    A missing original — already restored, or the Trash emptied by hand — is
    counted as ``gone``, not raised: half a restore is still worth having.
    Returns ``{'restored', 'gone', 'failed'}``.

    ``gone`` and ``failed`` are different answers and are kept apart, because
    only one of them is final. Gone means there is nothing left to put back and
    the entry is dropped. Failed means the original is still sitting in Trash
    and the restore ERRORED — an antivirus scanner holding the just-written
    crop open (the documented reason `trash.py` retries at all), or a
    cross-device `shutil.move` — so the entry is KEPT and the manifest is
    rewritten with the survivors. Deleting the manifest in that case is how the
    only remaining copy of a photo stops being reachable from the app; a retry
    is all that is needed instead.
    """
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    manifest = trim_report(dataset_id)
    if not manifest or not manifest.get('entries'):
        return {'restored': 0, 'gone': 0, 'failed': 0}
    ds_dir = _dataset_path(dataset_id)
    restored = gone = failed = 0
    survivors = []
    for e in manifest['entries']:
        original = e.get('trashed')
        # basename, as on the way in: the destination of an undo is a file in
        # this dataset's folder and nowhere else.
        filename = os.path.basename(e.get('filename') or '')
        target = os.path.join(ds_dir, filename) if filename else None
        # No filename is no destination, so such an entry counts as gone rather
        # than being retried forever: the copy in Trash is the only one left,
        # and deleting it is the one outcome an undo may never produce.
        if not (original and target and os.path.exists(original)):
            gone += 1
            continue
        try:
            if os.path.exists(target):
                trash.send_to_trash(
                    target, context=f'dataset-{dataset_id}-trim-restore')
            # shutil.move, not os.replace: the dataset folder is relocatable
            # (`paths.dataset_images_root`), so it and the trash can live on
            # different drives — where a rename raises and an original
            # sitting safely in Trash would be reported gone.
            shutil.move(original, target)
            restored += 1
        except Exception:                       # noqa: BLE001
            # Only the basename: a log line naming a user's machine layout is
            # one they cannot paste into a bug report.
            logger.exception('subject trim restore: %s', filename)
            failed += 1
            survivors.append(e)
    if survivors:
        # `dict(manifest, ...)`, not a fresh one: the batch stamp has to survive
        # a partial restore, or a later apply on the same batch would count this
        # manifest as somebody else's and write over the entries still in it.
        _write_json(_undo_path(dataset_id),
                    dict(manifest, entries=survivors,
                         counts=dict(manifest.get('counts') or {},
                                     trimmed=len(survivors))))
    else:
        try:
            os.remove(_undo_path(dataset_id))
        except OSError:
            pass
    return {'restored': restored, 'gone': gone, 'failed': failed}
