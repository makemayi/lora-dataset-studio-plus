"""Watermark detection and auto-correction for a face dataset (V1).

Scraped images often carry an OVERLAID watermark (site logo, URL, @username,
studio text) that the LoRA would learn. Detection is a Qwen3-VL bbox; removal is
routed by cost/risk -- crop a border-band mark, LaMa/Klein-inpaint a small
off-center one, else leave it for manual review.

Split out of face_dataset_service.py (2026-08, Phase 2 of a multi-phase file
split) -- pure move, no behavior change. The three helpers this block reads but
does not own (`_parse_watermark_bbox`, `_watermark_regions_payload`,
`_watermark_route_payload`) stay in face_dataset_service.py on purpose: they sit
far from this block in that file and image_bank_service.py imports two of them
from there, so moving them would widen a pure-move diff into an API change.
"""
from decimal import Decimal
from functools import wraps
import io
import json
import math
import os
import re
import shutil
import tempfile
import time

from PIL import Image, ImageOps, UnidentifiedImageError

from ..extensions import db
from ..models import FaceDataset, FaceDatasetImage
from .. import config as cfg
from . import dataset_activity, image_encoding
# Straight from the prompt module, not borrowed at the bottom: face_variations
# has no dependency on this module, so there is no cycle to schedule around.
from .face_variations import WATERMARK_BBOX_PROMPT


def _serialize_dataset_ingest(fn):
    """Late-bound twin of face_dataset_service._serialize_dataset_ingest.

    A DECORATOR runs while this module's body executes, which is before the
    borrow-back import at the bottom of this file has bound anything -- so the
    real decorator cannot simply be borrowed like the other primitives. This
    resolves it at CALL time instead and delegates, rather than restating the
    locking, so there is exactly one implementation of the ingest lock. Safe
    because that decorator is stateless: it takes the per-(user, dataset) lock
    when the wrapped function runs and holds nothing between calls.
    """
    @wraps(fn)
    def wrapped(*args, **kwargs):
        from . import face_dataset_service
        return face_dataset_service._serialize_dataset_ingest(fn)(*args, **kwargs)
    return wrapped


# --- Watermark auto-correction (V1) ----------------------------------------
# Scraped images often carry an OVERLAID watermark (site logo, URL, @username, studio
# text) that the LoRA would learn. V1 = detect (Qwen3-VL bbox) then route removal by
# cost/risk: CROP a border-band mark (PIL pur, invents no pixel), LaMa-inpaint a small
# off-center mark (non-generative, only masked pixels change), else leave it for manual
# review. NO YOLO, NO generative inpaint -- those are V2.
WATERMARK_BORDER_BAND = 0.20       # a mark within this outer strip is croppable
WATERMARK_MAX_INPAINT_AREA = 0.10  # bbox area above this fraction -> manual review
WATERMARK_MIN_SIDE = 768           # never crop a side below this (ai-toolkit only downscales)
WATERMARK_REGION_LIMIT = 32
WATERMARK_REGION_MIN_SIDE = 0.005


def normalize_watermark_regions(value, *, allow_null=True) -> list[list[float]] | None:
    if value is None:
        if allow_null:
            return None
        raise ValueError('regions must be a list')
    if not isinstance(value, list) or len(value) > WATERMARK_REGION_LIMIT:
        raise ValueError('regions must contain at most 32 boxes')
    out = []
    for box in value:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError('each region must be [x1,y1,x2,y2]')
        try:
            invalid_number = any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) for v in box
            )
        except OverflowError:
            invalid_number = True
        if invalid_number:
            raise ValueError('region coordinates must be finite numbers')
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('region coordinates must be ordered within [0,1]')
        min_side = Decimal(str(WATERMARK_REGION_MIN_SIDE))
        if (Decimal(str(x2)) - Decimal(str(x1)) < min_side
                or Decimal(str(y2)) - Decimal(str(y1)) < min_side):
            raise ValueError('region is too small')
        out.append([round(v, 4) for v in (x1, y1, x2, y2)])
    return out


def set_watermark_regions(user_id, dataset_id, image_id, regions) -> dict | None:
    """Atomically replace a detected image's manual watermark-region override."""
    _guard_not_bank_export(dataset_id)
    owned_query = (FaceDatasetImage.query
                   .join(FaceDataset, FaceDatasetImage.dataset_id == FaceDataset.id)
                   .filter(FaceDatasetImage.id == image_id,
                           FaceDatasetImage.dataset_id == dataset_id,
                           FaceDataset.user_id == str(user_id)))
    img = owned_query.one_or_none()
    if not img:
        return None
    if img.watermark_state != 'detected':
        raise RuntimeError('image is no longer detected')
    normalized = normalize_watermark_regions(regions)
    stored = json.dumps(normalized) if normalized is not None else None
    updated = (FaceDatasetImage.query
               .filter_by(id=img.id, watermark_state='detected')
               .update({'watermark_regions': stored}, synchronize_session=False))
    if updated != 1:
        db.session.rollback()
        if owned_query.one_or_none() is None:
            return None
        raise RuntimeError('image is no longer detected')
    db.session.commit()
    return _watermark_regions_payload(img)


def _route_watermark(bbox, W, H, *, min_side=WATERMARK_MIN_SIDE, allow_crop=True):
    """Decide how to remove the watermark at normalized `bbox` (x1,y1,x2,y2) on a
    W x H image. Returns ('crop', (left, top, right, bottom)) | ('lama', None) |
    ('review', None). PURE function (no I/O) so the routing is unit-testable.

    CROP (default, invents no pixel) when the mark sits ENTIRELY inside one outer
    border band (<= WATERMARK_BORDER_BAND of the side) AND the resulting crop keeps
    BOTH sides >= min_side -- we cut the band up to the mark's INNER edge. LaMa when
    the mark is small (area <= WATERMARK_MAX_INPAINT_AREA) and does not straddle the
    image center. Otherwise (large, or on the central subject with no safe crop) ->
    manual review, never a risky auto-edit.

    allow_crop=False (the "Allow auto-crop" preference turned off, or a per-image
    "force inpaint" from the review lightbox) SKIPS the crop branches entirely: a
    border mark then falls through to the inpaint/review logic below and is repainted
    (LaMa/Klein per the chosen engine) instead of cropped. Nothing else changes -- the
    min_side guard still governs whether crop is ever offered when it IS allowed."""
    x1, y1, x2, y2 = bbox
    px1, py1, px2, py2 = x1 * W, y1 * H, x2 * W, y2 * H
    band = WATERMARK_BORDER_BAND
    # Border-band crops, tried top/bottom/left/right. The kept box is (left,top,right,bottom).
    if allow_crop:
        if y2 <= band and (H - py2) >= min_side and W >= min_side:        # top band
            return 'crop', (0, int(round(py2)), W, H)
        if y1 >= 1 - band and py1 >= min_side and W >= min_side:          # bottom band
            return 'crop', (0, 0, W, int(round(py1)))
        if x2 <= band and (W - px2) >= min_side and H >= min_side:        # left band
            return 'crop', (int(round(px2)), 0, W, H)
        if x1 >= 1 - band and px1 >= min_side and H >= min_side:          # right band
            return 'crop', (0, 0, int(round(px1)), H)
    # Not a safe border crop (off-band, or the crop would fall below min_side).
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    overlaps_center = (x1 < 0.5 < x2) and (y1 < 0.5 < y2)
    if area <= WATERMARK_MAX_INPAINT_AREA and not overlaps_center:
        return 'lama', None
    return 'review', None


def _preserve_original(path) -> bool:
    """Durably preserve ``path`` before any destructive watermark edit.

    Returns ``True`` only when a usable sibling ``.orig`` exists.  A direct
    ``copy2(path, backup)`` can leave a truncated backup if the disk fills up;
    treating that as success would allow a subsequent crop/inpaint to destroy
    the only intact master.  Copy through a sibling temporary file and promote
    it atomically instead.  Existing backups are deliberately never overwritten
    (they are the older, recoverable master from a prior clean pass).
    """
    stem, ext = os.path.splitext(path)
    backup = f'{stem}.orig{ext or ".webp"}'
    if os.path.exists(backup):
        # A prior interrupted *old* implementation could have left a partial
        # .orig behind. Do not assume its mere existence makes an edit safe.
        try:
            if not os.path.isfile(backup) or os.path.getsize(backup) <= 0:
                raise OSError('existing backup is empty or not a regular file')
            with Image.open(backup) as check:
                check.verify()
            return True
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            logger.error('watermark: refusing edit; existing backup is unusable for %s: %s',
                         path, exc)
            return False

    staged_backup = None
    try:
        fd, staged_backup = tempfile.mkstemp(
            prefix=f'.{os.path.basename(backup)}.preserve-', suffix='.part',
            dir=os.path.dirname(path),
        )
        os.close(fd)
        shutil.copy2(path, staged_backup)
        # ``copy2`` returning does not guarantee that the data was flushed to
        # disk. Fsync the staged bytes before making the backup visible.
        # Windows requires a writable descriptor for fsync; the staged copy is
        # complete at this point, so ``rb+`` does not alter its bytes.
        with open(staged_backup, 'rb+') as handle:
            os.fsync(handle.fileno())
        with Image.open(staged_backup) as check:
            check.verify()
        os.replace(staged_backup, backup)
        staged_backup = None
        return True
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        logger.error('watermark: could not preserve original %s; refusing edit: %s', path, exc)
        return False
    finally:
        if staged_backup:
            try:
                os.unlink(staged_backup)
            except OSError:
                pass


def _stage_oriented_watermark_edit(path) -> str | None:
    """Create an upright, metadata-free sibling for a destructive watermark pass.

    Browser/VLM boxes live in visual orientation, whereas the master may still
    carry camera EXIF.  LaMa and Klein edit a path in place, so they must receive
    a disposable, EXIF-transposed file.  The caller promotes this sibling only
    after the engine reports success; a failure must leave the master byte-for-
    byte untouched.
    """
    staged = None
    try:
        with Image.open(path) as opened:
            fmt = image_encoding.format_for_path(path, opened)
            opened.load()
            icc = _valid_icc_profile(opened.info.get('icc_profile'))
            oriented = ImageOps.exif_transpose(opened)
            payload = io.BytesIO()
            image_encoding.save_edit(oriented, payload, fmt, image_encoding.LOSSLESS,
                                     icc_profile=icc)
        suffix = os.path.splitext(path)[1] or '.webp'
        fd, staged = tempfile.mkstemp(
            prefix=f'.{os.path.basename(path)}.wm-orient-', suffix=suffix,
            dir=os.path.dirname(path),
        )
        with os.fdopen(fd, 'wb') as fh:
            fh.write(payload.getvalue())
            fh.flush()
            os.fsync(fh.fileno())
        return staged
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        logger.warning('watermark: could not stage EXIF-oriented edit for %s: %s', path, exc)
        if staged:
            try:
                os.unlink(staged)
            except OSError:
                pass
        return None


def _promote_staged_watermark_edit(staged_path, live_path) -> bool:
    """Atomically replace a master only after a staged engine result verifies."""
    try:
        expected = image_encoding.format_for_path(live_path)
        with Image.open(staged_path) as check:
            if (check.format or '').upper() != expected:
                raise OSError('staged result format does not match its extension')
            check.verify()
        os.replace(staged_path, live_path)
        return True
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        logger.warning('watermark: could not promote staged edit for %s: %s', live_path, exc)
        return False


def _discard_staged_watermark_edit(staged_path) -> None:
    try:
        os.unlink(staged_path)
    except OSError:
        pass


def _apply_watermark_crop(path, box) -> bool:
    """Crop `path` to `box` (left,top,right,bottom px) WITHOUT resizing -- the whole
    point of the crop route is that it invents no pixel (the aspect-ratio change is
    absorbed by ai-toolkit's bucketing). Returns bool.

    Because nothing is resampled here, PNG/WebP/BMP retain the surviving pixels
    losslessly under ``image_encoding``. JPEG has no lossless write path and is
    deliberately re-encoded at the documented high-quality 4:4:4 setting. It used
    to re-save every format as WebP q92, which quietly re-compressed the ENTIRE
    image to remove a band at its edge."""
    try:
        with Image.open(path) as opened:
            fmt = image_encoding.format_for_path(path, opened)
            opened.load()
            icc = _valid_icc_profile(opened.info.get('icc_profile'))
            # `box` is visual/browser (and VLM) space. If called directly rather
            # than through clean_watermarks, keep the same orientation contract.
            im = ImageOps.exif_transpose(opened)
    except (OSError, ValueError):
        return False
    box = (max(0, int(box[0])), max(0, int(box[1])),
           min(im.width, int(box[2])), min(im.height, int(box[3])))
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        return False
    out = io.BytesIO()
    image_encoding.save_edit(im.crop(box), out, fmt, image_encoding.LOSSLESS,
                             icc_profile=icc)
    write_image_atomic(path, out.getvalue())
    return True


def detect_watermarks(user_id, dataset_id, *, include_dismissed=False, backend=None,
                      should_cancel=None, report=None):
    """Scan the KEPT images for an overlaid watermark and persist watermark_state
    ('detected'|'none') + watermark_bbox (JSON normalized box). Returns
    {'detected': n, 'none': n, 'checked': n} — that dict is the route's response
    shape and four tests pin it EXACTLY, so anything else the caller needs travels
    through ``report`` (a dict this fills in) or the route's own keys.

    TWO ROUTES, and which one runs is decided by ``watermark_detect.backend`` —
    the same setting the bank reads, resolved by the same function, because two
    screens obeying two rules is the defect this replaced:

    * the dedicated DETECTOR extra (SigLIP2 ranks, Grounding DINO locates) — no
      Ollama needed, ~0.14 s/image, and it writes a SCORE;
    * the vision model (Qwen3-VL), exactly as before — one chat question per
      image, ~1.7 s, no score. This is what 'auto' picks when the extra is not
      installed, so an untouched install behaves identically to yesterday.

    ``should_cancel`` is polled BETWEEN images: what was already judged is
    committed and kept, the rest simply stays unscanned and a later run finishes
    it (detect looks at every kept row on every pass).

    Images the user already judged NOT a watermark ('dismissed', a false positive
    ruled out in the review lightbox) are SKIPPED so a re-run never re-flags them --
    that's the anti-frustration point. Pass include_dismissed=True to re-examine them
    (a deliberate "check everything again"). CALLER decides on the GPU-exclusive
    vision window: the vision route always needs it, the detector only when it
    actually runs on CUDA."""
    _guard_not_bank_export(dataset_id)
    from . import watermark_detector
    resolution = (backend if isinstance(backend, dict)
                  else watermark_detector.resolve_backend(backend))
    if report is not None:
        report.update(resolution)
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return {'detected': 0, 'none': 0, 'checked': 0}
    rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
            .filter(FaceDatasetImage.filename.isnot(None)).all())
    # Ids, not ORM objects: see _live_image_row. Both loops commit per image, so
    # every row they have not reached is expired, and deleting a tile from the
    # grid mid-scan used to kill the scan.
    row_ids = [img.id for img in rows]
    if resolution['backend'] == 'detector':
        return _detect_watermarks_detector(
            dataset_id, row_ids, include_dismissed=include_dismissed,
            should_cancel=should_cancel, report=report)
    return _detect_watermarks_vision(
        dataset_id, row_ids, include_dismissed=include_dismissed,
        should_cancel=should_cancel, report=report)


def _detect_watermarks_vision(dataset_id, row_ids, *, include_dismissed,
                              should_cancel, report):
    """The original Qwen3-VL pass, unchanged except for the cancel poll."""
    try:
        from .vision_ollama import describe_image_ollama, unload_vision_model
    except ImportError:
        raise RuntimeError('vision (Ollama) service not configured/available yet')
    counts = {'detected': 0, 'none': 0, 'checked': 0}
    # Deliberately NOT a key in `counts`: that dict is this route's response
    # shape and four tests pin it exactly. A counter that is zero on every
    # ordinary run does not justify changing an API contract — and surfacing it
    # usefully would mean a UI decision, not a silent extra field. Logged below.
    vanished = 0
    stopped = False
    # Persistent progress indicator (survives a page reload); try/finally clears it
    # even if the vision pass raises → no phantom "Scanning…" spinner.
    token = dataset_activity.begin(dataset_id, 'watermark_detect', total=len(row_ids))
    try:
        for i, image_id in enumerate(row_ids):
            # Between images, never inside one: the current inference finishes,
            # everything already committed stays, and the pass unwinds through
            # the SAME cleanup as a normal end (model unload, indicator end).
            if should_cancel and should_cancel():
                stopped = True
                break
            dataset_activity.progress(token, done=i + 1)
            img = _live_image_row(image_id)
            if img is None:      # deleted while the pass ran
                vanished += 1
                continue
            # Dismissed = a confirmed false positive; don't waste a vision call re-asking
            # (and never silently re-flag it) unless the caller opts back in.
            if not include_dismissed and img.watermark_state == 'dismissed':
                continue
            path = _img_path(img)
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as fh:
                raw = describe_image_ollama(fh.read(), WATERMARK_BBOX_PROMPT, num_predict=400,
                                            prefer_json=True, fmt='json',
                                            keep_alive=_VISION_BATCH_KEEPALIVE)
            if not (raw or '').strip():
                # Vision unreachable/empty != "no watermark" (same reasoning as
                # classify_images): leave the state UNTOUCHED (retry possible) instead
                # of falsely marking every image clean when Ollama is just down.
                continue
            img.watermark_regions = None
            # Stamp WHICH route ruled, on every row this pass touches — a dataset
            # can hold verdicts from both (promotion carries a bank's across).
            img.watermark_source = 'vision'
            img.watermark_score = None          # this route has no score
            bbox = _parse_watermark_bbox(raw)
            if bbox:
                img.watermark_state = 'detected'
                img.watermark_bbox = json.dumps([round(v, 4) for v in bbox])
                counts['detected'] += 1
            else:
                img.watermark_state = 'none'
                img.watermark_bbox = None
                counts['none'] += 1
            counts['checked'] += 1
            db.session.commit()
    finally:
        unload_vision_model()  # rend la VRAM a ComfyUI en fin de batch
        dataset_activity.end(token)
    if vanished:
        logger.info('watermark detect: %s image(s) were deleted while the pass ran, '
                    'skipped', vanished)
    if report is not None:
        # located == detected here: the vision route never flags without a box
        # (no box parsed IS the "clean" answer).
        report.update({'stopped': stopped, 'located': counts['detected'],
                       'unlocated': 0, 'errors': 0})
    return counts


def _detect_watermarks_detector(dataset_id, row_ids, *, include_dismissed,
                                should_cancel, report):
    """The same pass run by the dedicated detector extra. Deliberately the same
    SHAPE as the vision pass — same skip rules, same per-image commit, same
    survives-a-deletion discipline — because the two must be interchangeable.

    Two structural differences the caller has to know about:

    * a single child process holds both models (loading them costs ~10 s), so a
      stop travels as a sentinel FILE the child polls between images rather than
      a kill — killing a process mid-forward is how a stop becomes a half-write;
    * this route can legitimately answer "detected, position unknown" when the
      locator finds nothing. The vision route cannot. That row is flagged with a
      NULL bbox, counted apart in ``report['unlocated']``, and the screen says so
      — 🧽 Clean has no box to route on and would otherwise stamp it 'failed'.
    """
    from . import watermark_detector
    counts = {'detected': 0, 'none': 0, 'checked': 0}
    planned = []
    for image_id in row_ids:
        img = _live_image_row(image_id)
        if img is None:
            continue
        if not include_dismissed and img.watermark_state == 'dismissed':
            continue
        path = _img_path(img)
        if not path or not os.path.exists(path):
            continue
        planned.append((image_id, path))
    located = unlocated = errors = vanished = 0
    stopped = False
    if not planned:
        if report is not None:
            report.update({'stopped': False, 'located': 0, 'unlocated': 0, 'errors': 0})
        return counts
    by_path = {}
    for image_id, path in planned:
        # A dataset can hold the same file twice; pop one waiting row per verdict
        # so both do not land on the first row.
        by_path.setdefault(path, []).append(image_id)

    cancel_dir = tempfile.mkdtemp(prefix='lds-wmdet-ds-')
    cancel_file = os.path.join(cancel_dir, 'cancel')

    def _cancelled():
        if not (should_cancel and should_cancel()):
            return False
        try:                                # the child polls for this file
            open(cancel_file, 'wb').close()
        except OSError:
            pass
        return True

    token = dataset_activity.begin(dataset_id, 'watermark_detect', total=len(planned))
    try:
        for path, state, score, regions, _error in watermark_detector.scan(
                [p for _i, p in planned], should_cancel=_cancelled,
                cancel_file=cancel_file):
            dataset_activity.bump(token)
            waiting = by_path.get(path) or []
            image_id = waiting.pop(0) if waiting else None
            img = _live_image_row(image_id) if image_id is not None else None
            if img is None:                 # deleted while it was being analysed
                vanished += 1
                continue
            img.watermark_source = 'detector'
            img.watermark_score = (round(float(score), 4) if score is not None else None)
            if state == 'error':
                # One unreadable file never sinks the pass, and it is NOT "clean":
                # the row keeps whatever state it had so a retry can finish it.
                errors += 1
                db.session.commit()
                continue
            img.watermark_regions = None
            if state == 'detected':
                img.watermark_state = 'detected'
                if regions:
                    # ONE box, the child's first — it orders them
                    # most-peripheral-first precisely because this line takes one
                    # (see the bank's identical write for the full reasoning).
                    img.watermark_bbox = json.dumps(
                        [round(float(v), 4) for v in regions[0][:4]])
                    located += 1
                else:
                    img.watermark_bbox = None
                    unlocated += 1
                counts['detected'] += 1
            else:
                img.watermark_state = 'none'
                img.watermark_bbox = None
                counts['none'] += 1
            counts['checked'] += 1
            db.session.commit()
        stopped = bool(should_cancel and should_cancel())
    except watermark_detector.DetectorUnavailable as e:
        # The extra probed OK but could not actually run (weights half downloaded,
        # a torch that no longer imports there). Everything already judged is
        # committed; say what happened and name the way out instead of failing
        # silently or, worse, marking unscanned rows clean.
        db.session.commit()
        logger.warning('dataset watermark detect: detector unavailable (%s)', e)
        raise RuntimeError(
            f'the watermark detector could not run ({e}). Nothing was mis-flagged — '
            'the images it had not reached are still unscanned. Set Settings ▸ '
            'Captioning & quality ▸ Watermark detection to "Vision model" to finish '
            'the pass without it.') from e
    finally:
        db.session.commit()
        dataset_activity.end(token)
        shutil.rmtree(cancel_dir, ignore_errors=True)
    if vanished:
        logger.info('watermark detect: %s image(s) were deleted while the pass ran, '
                    'skipped', vanished)
    if report is not None:
        report.update({'stopped': stopped, 'located': located,
                       'unlocated': unlocated, 'errors': errors})
    return counts


def dismiss_watermarks(user_id, dataset_id, image_ids):
    """Mark 'detected' images as 'dismissed' -- the user ruled, in the review lightbox,
    that the flag is a FALSE positive. Dismissed images drop the 🚩 badge, leave the
    Clean batch, and are skipped by future detect passes (see detect_watermarks) so
    they're never re-flagged. Only 'detected' rows of THIS dataset transition (ids that
    don't belong / aren't detected are silently ignored, like batch_image_action).
    Returns the number of rows dismissed. The bbox is kept (harmless, and a later
    include_dismissed re-scan overwrites it)."""
    _guard_not_bank_export(dataset_id)
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    ids = [int(i) for i in (image_ids or [])
           if isinstance(i, (int, float, str)) and str(i).lstrip('-').isdigit()]
    if not ids:
        return 0
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, watermark_state='detected')
            .filter(FaceDatasetImage.id.in_(ids)).all())
    for img in rows:
        img.watermark_state = 'dismissed'
        img.watermark_regions = None
    if rows:
        db.session.commit()
    return len(rows)


def _clean_inpaint_engine(route, method):
    """Which inpaint engine a NON-crop image gets, given the batch `method`
    ('auto'|'lama'|'klein'). Crop-routed images always crop (invents no pixel) — this
    only decides how a mark is *repainted*:
      - method 'klein' → Klein for both the small-off-center ('lama') route AND the
        on-subject ('review') route, so review becomes actionable (the whole V2 point);
      - otherwise → LaMa for 'lama', and 'review' stays manual review (unchanged V1)."""
    if method == 'klein':
        return 'klein'
    return 'lama' if route == 'lama' else 'review'


@_serialize_dataset_ingest
def clean_watermarks(user_id, dataset_id, image_ids=None, device='cpu', method='auto',
                     allow_crop=None):
    """Apply the crop/inpaint/review routing to every image marked 'detected'. Returns
    ({'cropped', 'inpainted', 'inpainted_klein', 'needs_review', 'failed', 'skipped'},
    error|None) -- same tuple contract as score_dataset_faces: `error` is None unless an
    inpaint that was ATTEMPTED failed (never a silent swallow). Crop stays in PIL.

    `allow_crop` gates the border-crop route (see _route_watermark). None (the default)
    resolves the persisted `watermark.allow_crop` preference, so a plain call and the
    batch Clean button both honour Settings; the review lightbox passes an explicit
    True/False to force crop or inpaint for ONE image. When False, a border mark is
    repainted (LaMa/Klein per `method`) instead of cropped -- nothing else changes.

    `method` selects the inpaint engine (the batch UI's LaMa|Klein toggle):
      - 'auto'/'lama' → LaMa (fast, non-generative) for small off-center marks; on-subject
        marks stay 'review'. Uses the resolved CPU/GPU `device`; GPU mode is protected by
        the route's exclusive window.
      - 'klein' → masked Flux.2 Klein inpaint + pixel-space composite for the off-center
        AND the on-subject marks (making 'review' actionable). Each image is one serialized
        ComfyUI round-trip; `device` is irrelevant (ComfyUI owns the GPU).

    LaMa absent (probe False) is NOT an error: LaMa-routed images are counted as
    `skipped` (crop still runs) so the UI can nudge "install the ML extras". Klein absent
    is likewise `skipped`.

    image_ids (optional): restrict the pass to this subset -- the review lightbox cleans
    ONE image at a time. The filter still requires watermark_state='detected' AND
    dataset ownership, so a stale/foreign id is a no-op (never touches another dataset,
    never re-edits an already-cleaned image). None = every detected image (bulk button)."""
    from . import watermark_lama, watermark_klein
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # None = "no explicit choice" -> fall back to the persisted preference (default
    # True), so the batch button follows Settings; the lightbox passes a real bool.
    if allow_crop is None:
        allow_crop = bool(cfg.get('watermark.allow_crop'))
    q = (FaceDatasetImage.query
         .filter_by(dataset_id=dataset_id, watermark_state='detected')
         .filter(FaceDatasetImage.filename.isnot(None)))
    if image_ids is not None:
        ids = [int(i) for i in (image_ids or [])
               if isinstance(i, (int, float, str)) and str(i).lstrip('-').isdigit()]
        q = q.filter(FaceDatasetImage.id.in_(ids or [-1]))   # empty subset -> match nothing
    rows = q.all()
    row_ids = [img.id for img in rows]
    out = {'cropped': 0, 'inpainted': 0, 'inpainted_klein': 0, 'needs_review': 0,
           'failed': 0, 'skipped': 0}
    # NOT a key in `out`: that dict is the route's response shape and existing
    # tests pin it. 'skipped' already means "engine unavailable" and must not be
    # overloaded with "the image no longer exists". Logged at the end instead.
    vanished = 0
    error = None
    lama_ok = watermark_lama.is_available()
    klein_ok = method == 'klein' and watermark_klein.is_available()
    # The Klein model this DATASET runs on — the same pick ✨ improve and Klein
    # generation use. A watermark clean overwrites the image in place, so running
    # it on a model the dataset did not choose is the one lane where the swap
    # cannot be spotted afterwards by comparing to a source.
    klein_model = dataset_klein_model(ds)
    if klein_ok and klein_model:
        # Refuse the WHOLE pass by name, before a single file is touched: every
        # image would fail identically, and a half-cleaned dataset is worse than
        # an untouched one. None (never chose) skips this — nothing was promised.
        from . import klein_edit_helper as keh
        if not keh.klein_model_on_disk(klein_model):
            raise keh.KleinModelGone(klein_model)
    # (image_id, live_path, staged_path, bboxes, manual_regions). An ID, not an
    # ORM row: this list is carried across the whole per-image loop AND across
    # the LaMa batch, which runs for minutes -- by the time the tail loop writes,
    # those rows have been expired by a dozen commits and any one of them may
    # have been deleted from the grid. See _live_image_row.
    lama_pending = []

    def _backup_failed(img, staged_path=None):
        """Record a fail-closed preservation error and discard any edit staging."""
        nonlocal error
        if staged_path:
            _discard_staged_watermark_edit(staged_path)
        # Retain manual regions for a future retry, but never present this row
        # as clean/detected when no recoverable pre-edit master exists.
        img.watermark_state = 'failed'
        out['failed'] += 1
        error = {
            'kind': 'failed',
            'detail': 'could not preserve original; master was left unchanged',
        }

    def _run_klein(img, path, boxes, manual):
        """One serialized Klein inpaint through an upright disposable sibling."""
        nonlocal error
        if not klein_ok:
            out['skipped'] += 1               # leave 'detected' (Klein not ready)
            return
        staged = _stage_oriented_watermark_edit(path)
        if not staged:
            if not manual:                    # retain manual regions for a retry
                img.watermark_state = 'failed'
            out['failed'] += 1
            error = {'kind': 'failed', 'detail': 'could not stage image EXIF orientation'}
            return
        if not _preserve_original(path):
            _backup_failed(img, staged)
            return
        try:
            ok, err = watermark_klein.inpaint_watermark_klein(user_id, staged, boxes,
                                                              klein_model=klein_model)
            if ok and _promote_staged_watermark_edit(staged, path):
                img.watermark_state = 'cleaned'
                if manual:
                    img.watermark_regions = None
                out['inpainted_klein'] += 1
            elif ok:
                if not manual:
                    img.watermark_state = 'failed'
                out['failed'] += 1
                error = {'kind': 'failed', 'detail': 'could not promote staged watermark edit'}
            elif err and err.get('kind') == 'unavailable':
                out['skipped'] += 1
            else:
                if not manual:                # keep manual retry metadata (like LaMa)
                    img.watermark_state = 'failed'
                out['failed'] += 1
                if err:
                    error = err
        finally:
            _discard_staged_watermark_edit(staged)
    # Persistent progress indicator (survives a page reload). The device is included
    # so the UI can honestly state whether ComfyUI is paused for the GPU pass.
    device_label = 'GPU' if device == 'cuda' else 'CPU'
    token = dataset_activity.begin(
        dataset_id, 'watermark_clean', total=len(rows),
        detail=f'Cleaning watermarks on {device_label}…')
    try:
        for i, image_id in enumerate(row_ids):
            dataset_activity.progress(token, done=i + 1)
            img = _live_image_row(image_id)
            if img is None:      # deleted while the pass ran
                vanished += 1
                continue
            path = _img_path(img)
            if img.watermark_regions is not None:
                try:
                    regions = normalize_watermark_regions(
                        _safe_json(img.watermark_regions), allow_null=False,
                    )
                except ValueError as e:
                    out['failed'] += 1
                    error = {'kind': 'failed',
                             'detail': f'invalid watermark regions: {e}'}
                    db.session.commit()
                    continue
                if not regions:
                    out['needs_review'] += 1
                    db.session.commit()
                    continue
                if not os.path.exists(path):
                    out['failed'] += 1
                    db.session.commit()
                    continue
                if method == 'klein':
                    _run_klein(img, path, regions, True)
                    db.session.commit()
                    continue
                if not lama_ok:
                    out['skipped'] += 1
                    db.session.commit()
                    continue
                staged = _stage_oriented_watermark_edit(path)
                if not staged:
                    out['failed'] += 1
                    error = {'kind': 'failed',
                             'detail': 'could not stage image EXIF orientation'}
                    db.session.commit()
                    continue
                if not _preserve_original(path):
                    _backup_failed(img, staged)
                    db.session.commit()
                    continue
                lama_pending.append((img.id, path, staged, regions, True))
                continue
            bbox = _safe_json(img.watermark_bbox)
            if not (isinstance(bbox, list) and len(bbox) == 4):
                # Flagged, position unknown. The detector cascade produces this
                # legitimately (its locator found nothing) and promotion carries
                # it in from a bank; stamping 'failed' would DESTROY a correct
                # flag over a missing coordinate. It goes to manual review, where
                # a zone can be drawn — the same answer the bank gives.
                out['needs_review'] += 1
                db.session.commit()
                continue
            if not os.path.exists(path):
                img.watermark_state = 'failed'
                out['failed'] += 1
                db.session.commit()
                continue
            try:
                with Image.open(path) as im:
                    # Stored detection boxes are in the browser/VLM's upright
                    # coordinate space, never the raw camera raster. This branch
                    # may route to review/no-op, so keep it header-only until an
                    # actual crop/staging edit needs the pixels.
                    W, H = image_encoding.visual_size_from_header(im)
            except (OSError, ValueError):
                img.watermark_state = 'failed'
                out['failed'] += 1
                db.session.commit()
                continue
            route, box = _route_watermark(tuple(bbox), W, H, allow_crop=allow_crop)
            if route == 'crop':
                if not _preserve_original(path):
                    _backup_failed(img)
                elif _apply_watermark_crop(path, box):
                    # NOTE dHash: the perceptual hash used for import-dedupe is recomputed
                    # ON THE FLY from the file (_existing_dhashes / _dhash), NOT stored in a
                    # column -- there is no stored dHash to leave untouched. So after a crop
                    # the dedupe compares against the CLEANED pixels; re-importing the same
                    # watermarked visual is NOT guaranteed to dedupe against it (a border
                    # crop shifts the whole hash). Preserving the original-dHash behaviour the
                    # spec asks for would need a new stored column -> deferred (out of V1 scope).
                    img.watermark_state = 'cleaned'
                    out['cropped'] += 1
                else:
                    img.watermark_state = 'failed'
                    out['failed'] += 1
            else:
                engine = _clean_inpaint_engine(route, method)
                if engine == 'klein':
                    _run_klein(img, path, [bbox], False)
                elif engine == 'lama':
                    if not lama_ok:
                        out['skipped'] += 1      # leave state='detected' (crop-only mode)
                    else:
                        staged = _stage_oriented_watermark_edit(path)
                        if staged:
                            if _preserve_original(path):
                                lama_pending.append((img.id, path, staged, [bbox], False))
                            else:
                                _backup_failed(img, staged)
                        else:
                            img.watermark_state = 'failed'
                            out['failed'] += 1
                            error = {'kind': 'failed',
                                     'detail': 'could not stage image EXIF orientation'}
                else:  # 'review' -> stays 'detected' so the badge/count keep flagging it
                    out['needs_review'] += 1
            db.session.commit()
        if lama_pending:
            try:
                if len(lama_pending) == 1:
                    _pid, live_path, staged_path, boxes, manual = lama_pending[0]
                    if manual:
                        ok, err = watermark_lama.inpaint_watermarks(
                            staged_path, boxes,
                            **({'device': device} if device != 'cpu' else {}))
                    else:
                        ok, err = watermark_lama.inpaint_watermark(
                            staged_path, boxes[0],
                            **({'device': device} if device != 'cpu' else {}))
                    results = {staged_path: (ok, err)}
                else:
                    results = watermark_lama.inpaint_batch(
                        [{'image_path': staged_path, 'bboxes': boxes}
                         for _pid, _live_path, staged_path, boxes, _manual in lama_pending],
                        device=device,
                    )
                for pending_id, live_path, staged_path, _boxes, manual in lama_pending:
                    img = _live_image_row(pending_id)
                    if img is None:
                        # Deleted while the batch ran: there is no row left to
                        # point at the repainted file, so drop the staged edit
                        # rather than promote it over a master nobody owns.
                        _discard_staged_watermark_edit(staged_path)
                        vanished += 1
                        continue
                    ok, err = results.get(
                        staged_path,
                        (False, {'kind': 'failed', 'detail': 'missing inpaint result'}),
                    )
                    if ok and _promote_staged_watermark_edit(staged_path, live_path):
                        img.watermark_state = 'cleaned'
                        if manual:
                            img.watermark_regions = None
                        out['inpainted'] += 1
                    elif ok:
                        if not manual:
                            img.watermark_state = 'failed'
                        out['failed'] += 1
                        error = {'kind': 'failed',
                                 'detail': 'could not promote staged watermark edit'}
                    elif err and err.get('kind') == 'unavailable':
                        out['skipped'] += 1
                    else:
                        # Manual correction regions are user-authored retry metadata. Keep
                        # the image detected when LaMa fails so Clean can be retried.
                        if not manual:
                            img.watermark_state = 'failed'
                        out['failed'] += 1
                        if err:
                            error = err
                    db.session.commit()
            except Exception as exc:  # engine/process faults must not leak a staged edit
                logger.exception('watermark: LaMa execution failed for dataset %s', dataset_id)
                error = {'kind': 'failed', 'detail': f'watermark inpaint failed: {exc}'}
                for pending_id, _live_path, _staged_path, _boxes, manual in lama_pending:
                    img = _live_image_row(pending_id)
                    if img is None:
                        vanished += 1
                        continue
                    if not manual:
                        img.watermark_state = 'failed'
                    out['failed'] += 1
                    db.session.commit()
            finally:
                # The engine can crash before returning a result; in that case its
                # disposable EXIF-oriented copy still has to disappear, while the
                # master remains exactly where it was.
                for _pid, _live_path, staged_path, _boxes, _manual in lama_pending:
                    _discard_staged_watermark_edit(staged_path)
        if vanished:
            logger.info('watermark clean: %s image(s) were deleted while the pass '
                        'ran, skipped', vanished)
        return out, error
    finally:
        dataset_activity.end(token)


@_serialize_dataset_ingest
def restore_watermark_original(user_id, dataset_id, image_id) -> dict | None:
    """Undo a watermark Clean on ONE image: copy the preserved `<stem>.orig<ext>` back
    over the current file and flip the row from 'cleaned' (or 'failed') back to
    'detected', so it re-enters the Clean set and the user can re-clean it -- e.g. retry
    with the OTHER engine, or re-edit the zones. Returns a payload dict (state + planned
    route + regions) on success, None when the image isn't found/owned, and raises
    FileNotFoundError when no original was preserved (the image was never cleaned, or the
    sibling was removed) -> the route maps that to a 404.

    Design: the `.orig` is KEPT after a restore. It stays the single source of truth for
    the original pixels, so any number of clean -> restore -> clean cycles never loses it:
    _preserve_original is write-once (guarded by os.path.exists), so a later re-clean sees
    the existing sibling and won't overwrite it with an already-edited image. bbox/regions
    are preserved as-is (a crop/inpaint doesn't move the normalized box, and the user may
    want to re-clean the same zones). The crop route shrinks the image; restoring the
    .orig also restores the ORIGINAL dimensions -- nothing stored depends on them (the
    planned-route recompute reads the file live in _payload_watermark_route)."""
    owned_query = (FaceDatasetImage.query
                   .join(FaceDataset, FaceDatasetImage.dataset_id == FaceDataset.id)
                   .filter(FaceDatasetImage.id == image_id,
                           FaceDatasetImage.dataset_id == dataset_id,
                           FaceDataset.user_id == str(user_id)))
    img = owned_query.one_or_none()
    if not img or not img.filename:
        return None
    path = _img_path(img)
    stem, ext = os.path.splitext(path)
    backup = f'{stem}.orig{ext or ".webp"}'
    if not os.path.exists(backup):
        raise FileNotFoundError('no original to restore')
    shutil.copy2(backup, path)   # bring the watermarked original back in place
    # Re-flag as 'detected' so the badge/Clean count pick it up again; bbox and manual
    # regions are left exactly as stored (re-cleanable, possibly with the other engine).
    img.watermark_state = 'detected'
    db.session.commit()
    return {'watermark_state': img.watermark_state,
            **_watermark_route_payload(img),
            **_watermark_regions_payload(img)}

# --- Borrow: face_dataset_service.py primitives -----------------------------
# MUST stay at the bottom of this file, for the same reason reference_photos_service.py
# keeps ITS borrow at ITS bottom (see the comment there): this module and
# face_dataset_service.py import names from each other -- a genuine two-way
# dependency. Loading either one in isolation transitively loads the other,
# whose own bottom import then reaches back here for the names defined above.
# At the top of the file that reach-back would find a partially loaded module
# and raise ImportError; last, every name this module owns already exists no
# matter which side gets imported first.
from .face_dataset_service import (
    get_dataset, batch_image_action, dataset_klein_model,
    write_image_atomic, _img_path, _dhash, _existing_dhashes, _safe_json,
    _valid_icc_profile, _VISION_BATCH_KEEPALIVE,
    _guard_not_bank_export, _live_image_row,
    _watermark_regions_payload, _watermark_route_payload,
    logger,
)
# Straight from the module that OWNS them, not via face_dataset_service's
# re-export: routing a cross-module borrow through the parent would make the
# ORDER of the parent's re-export blocks load-bearing.
from .dataset_import_service import (
    classify_images, _dhash, _existing_dhashes, _parse_watermark_bbox,
)
