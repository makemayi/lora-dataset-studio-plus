"""The geometry of the person crop, verbatim from the operator's own script.

The h3-video-dataset workflow batch-cropped a bank around its largest detected
person with `data/bank_sources/__-3/_crop_tools/crop_persons.py`; this module is
that script's geometry, moved into the app so the pass can carry the same rules.
Only the I/O changed hands: the detector lives in `infer/person_box_infer.py`,
the disk writes belong to the image bank service, and what is left here is what
that script was actually ABOUT — picking one rectangle per picture.

THE RULES, unchanged from the script:

  * every detection box for "person. human body." is collected;
  * boxes whose IoU is at least `IOU_MERGE` are merged into their union — the
    detector names one person several times (the prompt repeats synonyms), and
    merging by overlap keeps genuinely distinct people apart;
  * the LARGEST merged box wins. One subject per picture, chosen by area;
  * the box is padded by `MARGIN` (a fraction of its larger side, per side);
  * the crop is the tightest window with the SOURCE image's aspect ratio that
    contains the padded box, centred on it, clamped to the image — so the crop
    changes WHAT fills the frame, never the SHAPE of the frame, and a dataset
    keeps a single aspect ratio end to end.

Pure functions over numbers, like `video_frame_select` — no decode, no disk —
because these are the parts worth testing (a box on an edge, a box bigger than
the picture, a clamped window) and they are exactly the parts a batch script
never got to test.
"""

from __future__ import annotations

IOU_MERGE = 0.5        # merge boxes whose IoU >= this (phrase duplicates)
#: The framings promote-to-dataset can emit. 'full' is the native picture;
#: 'half' and 'face' are crops cut from it around a measured face box.
PROMOTION_FRAMINGS = ('full', 'half', 'face')
MARGIN = 0.03          # 3% of the box's larger side, added per side


def iou(a, b) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    ba = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + ba - inter)


def merge_boxes(boxes, iou_merge: float = IOU_MERGE):
    """Merge near-duplicate boxes (same person named twice) into unions,
    keeping distinct persons separate. Result sorted by area, biggest first."""
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    merged = []
    for box in boxes:
        for existing in merged:
            if iou(existing, box) >= iou_merge:
                existing[0] = min(existing[0], box[0])
                existing[1] = min(existing[1], box[1])
                existing[2] = max(existing[2], box[2])
                existing[3] = max(existing[3], box[3])
                break
        else:
            merged.append(list(box))
    return merged


def largest_person_box(boxes, iou_merge: float = IOU_MERGE):
    """The largest merged box, or None when the picture holds no person."""
    merged = merge_boxes(boxes, iou_merge)
    return merged[0] if merged else None


def ar_window(box, img_w: int, img_h: int, margin: float = MARGIN):
    """Tightest window with the image's aspect ratio that contains the padded
    box, clamped to the image. Returns (x1, y1, x2, y2) in pixels."""
    x1, y1, x2, y2 = box
    pw, ph = x2 - x1, y2 - y1
    pad = margin * max(pw, ph)
    x1, y1 = max(0.0, x1 - pad), max(0.0, y1 - pad)
    x2, y2 = min(float(img_w), x2 + pad), min(float(img_h), y2 + pad)
    pw, ph = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    ar = img_w / img_h
    cw = max(pw, ph * ar)
    ch = cw / ar
    # center on the box, then clamp so the window stays inside the image
    wx1 = cx - cw / 2.0
    wy1 = cy - ch / 2.0
    wx1 = min(max(wx1, 0.0), img_w - cw)
    wy1 = min(max(wy1, 0.0), img_h - ch)
    return (round(wx1), round(wy1), round(wx1 + cw), round(wy1 + ch))


# ── the batch the pass runs ──────────────────────────────────────────────────
#
# The pass hands the subprocess a slice of the bank at a time: one model load
# per slice, a progress update per slice, and a slice that fails is reported
# without sinking the run. 24 is comfortably inside a GPU's batch appetite at
# the 480/800 detect size, and small enough that the progress bar moves.

BATCH = 24


def batch_timeout(n: int) -> int:
    """Seconds one subprocess slice may take. The FIRST slice loads the model
    (tens of seconds on CPU, a few on GPU); later slices only infer."""
    return 120 + 45 * n


# ── promotion framings ────────────────────────────────────────────────────────
#
# The promote-to-dataset pass can emit, beside the full frame, two face-centred
# crops built from one measured face box. The geometry mirrors the video lane's
# `video_frame_select.half_window`/`face_window` (same numbers where the intent
# is the same), with ONE operator rule the video lane does not have:
#
#   A FACE CROP MUST PUT THE FACE AT AT LEAST 60 % OF THE PICTURE — that is the
#   operator's own bar for a usable face still ("不低于60%画面大小"). A window
#   computed at exactly that tightness, clamped back by a frame edge until the
#   face box no longer FITS inside it, is refused (None) rather than shipped as
#   a loose crop wearing the name.

def _normalised_box(face):
    """(x0, y0, x1, y1) pixels from a normalised box dict, or None."""
    try:
        x0, y0 = float(face['nx0']), float(face['ny0'])
        x1, y1 = float(face['nx1']), float(face['ny1'])
    except (KeyError, TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def half_window(w, h, face, face_scale: float = 1.0):
    """The waist-up window for one normalised face box, or None.

    Head-room of half a face above the brows, ~6 face-heights down the body
    (head to waist), 3:4 portrait, centred on the face horizontally, clamped
    to the frame. Refused when the face box would not sit entirely inside the
    window — a half-body crop that beheads or hip-checks its subject is not a
    framing, it is a bug with a label."""
    box = _normalised_box(face)
    if box is None:
        return None
    fx0, fy0, fx1, fy1 = box
    fh = (fy1 - fy0) * h
    cx = (fx0 + fx1) / 2.0 * w
    top = max(0.0, fy0 * h - fh * 0.5)
    ch = min(h - top, fh * 6.0)
    cw = min(float(w), ch * 3.0 / 4.0)
    left = min(max(0.0, cx - cw / 2.0), float(w) - cw)
    win = (left, top, left + cw, top + ch)
    # The face must land entirely inside what we ship.
    if not (win[0] <= fx0 * w and fx1 * w <= win[2]
            and win[1] <= fy0 * h and fy1 * h <= win[3]):
        return None
    return tuple(round(v) for v in win)


def face_window(w, h, face, min_face_frac: float = 0.60):
    """The square close-up window, or None when the bar cannot be met.

    The side is chosen so the face's height is AT LEAST `min_face_frac` of the
    window (side = face height / fraction), never larger than the frame, then
    centred on the face and clamped. If clamping would push the face box out of
    its own close-up — a face hard against a frame edge — the crop is refused:
    shipping it would quietly drop below the operator's 60 % bar, and a crop
    named "face" that is mostly shoulder is exactly the dishonest output this
    function exists to prevent."""
    box = _normalised_box(face)
    if box is None:
        return None
    fx0, fy0, fx1, fy1 = box
    pw, ph = (fx1 - fx0) * w, (fy1 - fy0) * h
    if pw <= 0 or ph <= 0:
        return None
    side = ph / min_face_frac
    side = min(side, float(w), float(h))
    cx, cy = (fx0 + fx1) / 2.0 * w, (fy0 + fy1) / 2.0 * h
    left = min(max(0.0, cx - side / 2.0), float(w) - side)
    top = min(max(0.0, cy - side / 2.0), float(h) - side)
    win = (left, top, left + side, top + side)
    if not (win[0] <= fx0 * w and fx1 * w <= win[2]
            and win[1] <= fy0 * h and fy1 * h <= win[3]):
        return None
    return tuple(round(v) for v in win)
