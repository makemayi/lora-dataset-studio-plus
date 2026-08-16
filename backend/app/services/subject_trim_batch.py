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

Applying moves each original to the OS Trash and appends to `trim-undo.json`,
so a whole batch can still be reverted after the fact — the review catches
"this one should not be cropped", the undo catches "it looked right and was
not".
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass

from PIL import Image, ImageOps

from ..models import FaceDatasetImage
from . import auto_mask, dataset_activity, image_encoding, trash
from .face_dataset_service import get_dataset, _dataset_path
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
