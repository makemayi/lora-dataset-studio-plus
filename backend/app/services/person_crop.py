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
