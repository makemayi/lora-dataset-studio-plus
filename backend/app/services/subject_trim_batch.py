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


def _preview_path(dataset_id):
    return os.path.join(_dataset_path(dataset_id), PREVIEW_FILE)


def _undo_path(dataset_id):
    return os.path.join(_dataset_path(dataset_id), UNDO_FILE)


def _read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False)


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
    """The last applied batch's undo manifest — ``{entries, counts}`` — or
    ``None``. The UI reads it to decide whether to offer the undo.

    ONE batch deep, on purpose: applying again overwrites it. That is only
    reachable through a fresh preview, because applying clears the pending one.
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
    manifest, so the batch can be reverted in one action afterwards.
    """
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    manifest = preview_report(dataset_id)
    if not manifest or not manifest.get('items'):
        raise ValueError('no crop preview to apply — run the preview first')
    wanted = {int(i) for i in (image_ids or [])}
    by_id = {int(it['image_id']): it for it in manifest['items']}
    ds_dir = _dataset_path(dataset_id)
    entries = []
    counts = {'trimmed': 0, 'refused': 0, 'failed': 0}
    # Sorted, not set order: the undo manifest is a record of what happened to
    # a user's files, and a record whose order changes run to run is one nobody
    # can diff against the screen they clicked.
    for image_id in sorted(wanted):
        item = by_id.get(image_id)
        if not item or item.get('skip') or not item.get('frame'):
            counts['refused'] += 1
            continue
        path = os.path.join(ds_dir, item['filename'])
        fx, fy, fw, fh = (int(v) for v in item['frame'])
        try:
            with Image.open(path) as opened:
                # The file keeps its own format — a `.png` must not come back
                # holding WEBP bytes — and its colour profile.
                fmt = image_encoding.format_for_path(path, opened)
                opened.load()
                icc = _valid_icc_profile(opened.info.get('icc_profile'))
                # EXIF-bake first: the frame was measured on the mask, which
                # auto_mask produces from the ORIENTED image, so a crop against
                # the raw buffer would land somewhere else entirely.
                oriented = ImageOps.exif_transpose(opened)
                # Narrow the mode BEFORE resampling: Pillow silently drops to
                # nearest-neighbour on paletted images.
                src = oriented.convert(image_encoding.resample_mode(oriented))
            box = (max(0, fx), max(0, fy),
                   min(src.width, fx + fw), min(src.height, fy + fh))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f'empty crop box {item["frame"]}')
            crop = src.crop(box)
            out_w, out_h = output_size(crop.width, crop.height, LONG_EDGE)
            if (out_w, out_h) != crop.size:
                crop = crop.resize((out_w, out_h), Image.LANCZOS)
            buf = io.BytesIO()
            image_encoding.save_edit(crop, buf, fmt, image_encoding.LOSSLESS,
                                     icc_profile=icc)
            # ENCODE, then move the original, then publish. Everything that can
            # fail for a reason other than the disk has already failed by this
            # line, with the original still in its folder; and the entry is
            # recorded BEFORE the write, so even a write that dies mid-way
            # leaves an undo that can put the original back.
            trashed = trash.send_to_trash(
                path, context=f'dataset-{dataset_id}-trim-{image_id}')
            entries.append({'image_id': image_id, 'filename': item['filename'],
                            'trashed': trashed})
            write_image_atomic(path, buf.getvalue())
            counts['trimmed'] += 1
        except Exception:                       # noqa: BLE001
            logger.exception('subject trim: image %s failed', image_id)
            counts['failed'] += 1
    if entries:
        _write_json(_undo_path(dataset_id), {'entries': entries, 'counts': counts})
        # The preview is spent only once a file has actually moved. A pass that
        # wrote nothing — every row refused, or every one failed — leaves it
        # alone: rebuilding it costs a full masking run, and a stale click must
        # not be able to bin one.
        clear_preview(dataset_id)
    return counts


def restore_trim_batch(user_id, dataset_id):
    """Whole-batch undo: every original the last applied batch trashed comes
    back and overwrites the cropped file; the manifest is cleared.

    A missing original — already restored, or the Trash emptied by hand — is
    counted as ``gone``, not raised: half a restore is still worth having.
    Returns ``{'restored', 'gone'}``.
    """
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    manifest = trim_report(dataset_id)
    if not manifest or not manifest.get('entries'):
        return {'restored': 0, 'gone': 0}
    ds_dir = _dataset_path(dataset_id)
    restored = gone = 0
    for e in manifest['entries']:
        original = e.get('trashed')
        filename = e.get('filename')
        target = os.path.join(ds_dir, filename) if filename else None
        try:
            # No filename is no destination, so such an entry counts as gone
            # rather than being removed: the copy in Trash is the only one left,
            # and deleting it is the one outcome an undo may never produce.
            if original and target and os.path.exists(original):
                if os.path.exists(target):
                    trash.send_to_trash(
                        target, context=f'dataset-{dataset_id}-trim-restore')
                # shutil.move, not os.replace: the dataset folder is relocatable
                # (`paths.dataset_images_root`), so it and the trash can live on
                # different drives — where a rename raises and an original
                # sitting safely in Trash would be reported gone.
                shutil.move(original, target)
                restored += 1
            else:
                gone += 1
        except Exception:                       # noqa: BLE001
            logger.exception('subject trim restore: %s', original)
            gone += 1
    try:
        os.remove(_undo_path(dataset_id))
    except OSError:
        pass
    return {'restored': restored, 'gone': gone}
