"""Subject trim — the crop rule, and nothing else.

Everything a crop decision depends on except the bytes: the largest connected
mask region, the frame that removes background around it, and the two rules
that decide a crop is not worth doing. Pure on purpose — a test pins every rule
without a SAM3, a ComfyUI or a file on disk.

WHAT THE 2026-08-16 WAVE GOT WRONG, AND WHAT REPLACED IT
--------------------------------------------------------
It kept the subject bbox's OWN aspect and never padded to a target ratio. For a
standing person that bbox is about 1:4, so a 2160x3240 photo produced a
257x1024 sliver. Correctly placed, and useless. Hence ASPECT_FLOOR: the frame
may be any shape it likes as long as it is not narrower than 9:16, and it
reaches the floor by taking background sideways — never by cutting height.

It also clamped the frame to the canvas by shrinking the WHOLE frame
proportionally, so removing excess width removed height too, off a subject that
exactly filled it. That is what cut people and faces in half. Here each axis is
clamped alone, which structurally cannot cut the subject: width only overshoots
after background has been ADDED, so what a width clamp removes is that
background; height is the subject's own height, so it only overshoots when the
subject already spans the canvas.

Its only guard was "skip when the subject is under 15% of the image height",
which skipped exactly the case this feature exists for and let through photos
whose subject already filled 92% of the frame. Both guards below measure the
OUTPUT instead.
"""
from __future__ import annotations

# 9:16. Chosen with the user over 1:2 (thinner, not in the app's ✂ vocabulary)
# and 2:3 (a tenth more background on a standing figure). It is a FLOOR, never
# a target: a frame already wider is left alone, because a wide frame here means
# no background was added.
ASPECT_FLOOR = 9 / 16

# A 1% sliver so the mask's own edge never clips hair or fingertips. This trim
# removes background; it does not compose a shot.
MARGIN = 1.01

# Keep 85% of the original area or more and there was nothing to remove — the
# re-encode and the downscale would be the only effect.
MAX_KEEP_FRACTION = 0.85

# Under 256 original pixels on the short edge there is not enough of the subject
# to train on, and a speck of mis-detected mask cannot pass it either.
MIN_SHORT_EDGE = 256

# Matches the manual ✂ crop: cap the LONG side, never upscale.
LONG_EDGE = 1024


def largest_bbox(mask):
    """``(x, y, w, h)`` of the LARGEST connected white region of a PIL mask, or
    ``None`` when the mask has no white pixels.

    Connectedness is 8-neighbour, so a person split by a thin shadow gap still
    counts as one region. `auto_mask.mask_for` returns ONE mask with every
    person white, so in a group photo this picks one of them and the others may
    be cut — decided with the user: for a character LoRA the bystander is
    unwanted anyway.
    """
    import cv2
    import numpy as np
    arr = np.asarray(mask.convert('L'))
    _ok, binary = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
    num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    if num <= 1:                        # background component only
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1     # +1: skip background
    return (int(stats[idx, cv2.CC_STAT_LEFT]),
            int(stats[idx, cv2.CC_STAT_TOP]),
            int(stats[idx, cv2.CC_STAT_WIDTH]),
            int(stats[idx, cv2.CC_STAT_HEIGHT]))


def trim_frame(image_w, image_h, bbox, *, floor=ASPECT_FLOOR, margin=MARGIN):
    """A crop frame ``(x, y, w, h)`` inside an ``image_w x image_h`` canvas.

    Height is the subject's own height plus the sliver and is never used to
    reach the floor. Width starts the same way and grows sideways only if the
    frame would be narrower than ``floor``. The frame is centred on the subject
    and each axis is clamped to the canvas independently.
    """
    x, y, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        raise ValueError('empty subject bbox')
    fw = float(bw) * margin
    fh = float(bh) * margin
    if fw / fh < floor:
        fw = fh * floor                 # widen only; height never moves for this
    # Integer frame first, each axis capped alone — see the module docstring on
    # why a proportional shrink is not used here.
    fw_i = min(image_w, max(1, round(fw)))
    fh_i = min(image_h, max(1, round(fh)))
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    fx_i = int(min(max(round(cx - fw_i / 2.0), 0), image_w - fw_i))
    fy_i = int(min(max(round(cy - fh_i / 2.0), 0), image_h - fh_i))
    return (fx_i, fy_i, fw_i, fh_i)


def skip_reason(image_w, image_h, frame, *, max_keep=MAX_KEEP_FRACTION,
                min_short_edge=MIN_SHORT_EDGE):
    """``None`` when the crop is worth doing, otherwise a stable reason code:
    ``'nothing-much-to-remove'`` or ``'subject-too-small'``.

    Order matters. A frame that keeps almost the whole image is reported as
    having nothing to remove even when it is also small, because that is the
    honest reason and the other one would send someone looking at the mask.
    """
    _x, _y, fw, fh = frame
    if fw * fh >= image_w * image_h * max_keep:
        return 'nothing-much-to-remove'
    if min(fw, fh) < min_short_edge:
        return 'subject-too-small'
    return None


def output_size(w, h, long_edge=LONG_EDGE):
    """The saved size for a crop of ``w x h``: the long side capped at
    ``long_edge``, aspect kept, never upscaled."""
    longest = max(w, h)
    if longest <= long_edge:
        return (w, h)
    r = long_edge / float(longest)
    return (max(1, round(w * r)), max(1, round(h * r)))
