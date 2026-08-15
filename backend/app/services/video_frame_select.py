"""Choosing WHICH frames of a video become dataset images.

The naive version of this — "keep the N sharpest frames" — produces N nearly
identical pictures, because sharpness is strongly autocorrelated in time: the
sharpest frame's neighbours are the second, third and fourth sharpest. A LoRA
trained on that learns one instant of one shot very well and nothing else.

So sharpness only ever RANKS here. What makes the selection usable is the three
constraints applied around it, in this order:

  1. HARD FILTERS decide what is admissible at all — exposure, and (when the
     face pass ran) that a face is present, large enough, and not in extreme
     profile. A frame that fails one of these can never be rescued by being
     sharp, so it leaves the pool before ranking.
  2. A MINIMUM TIME GAP is what actually breaks the autocorrelation. Two frames
     40 ms apart are the same picture whatever their scores say.
  3. DEDUP BY EMBEDDING catches what the gap cannot: a locked-off shot where the
     subject is still, and a cut back to the same framing thirty seconds later.

WHY THIS FILE HAS NO I/O. Decoding is the expensive, stateful, hard-to-test part
and it lives elsewhere; everything here is a pure function over readings that a
decoder already produced. That is what lets the interesting cases — "every frame
is blurred", "the face is there but tiny", "two candidates are 30 ms apart" — be
tested without a video file.

WHAT IT NEVER DOES. It never pads the result to reach `limit`. Returning three
frames when six were asked for is the honest answer to a shot that only holds
three usable moments; returning six means three of them are there because the
caller asked for a number, and nothing downstream can tell which three.

The per-frame reading is the SAME shape `video_metrics.summarise` consumes
(`luma`, `sharp`), plus the timestamp and the optional face reading, so a
decoder can feed both from one pass.
"""

from __future__ import annotations

from .video_metrics import _LUMA_SANE_HIGH, _LUMA_SANE_LOW

# Face gates. `DET_MIN` and `YAW_MAX` MIRROR `backend/infer/face_score_infer.py`
# and are kept equal to it by `test_video_frame_select.py` — two tables that
# drift apart are worse than one, and this module admitting a frame the scorer
# then refuses to talk about is exactly that drift.
#
# YAW_MAX moved 40° -> 70° there on 2026-08-15: antelopev2 still extracts a
# discriminative embedding out to roughly 70°, so a 3/4 profile is usable rather
# than noise. Those frames stay out of auto-triage on the front end; here they
# are admissible, and the sharpness ranking decides whether they win a slot.
DET_MIN = 0.50
YAW_MAX = 70.0
# ...but the SIZE floor is deliberately higher than the scorer's 0.06. That one
# answers "is there a face here at all"; a training crop has to answer "is there
# an identity here", and a face occupying 6 % of the frame's short edge carries
# almost no identity signal once it is resized for training.
FACE_BBOX_MIN = 0.12

MIN_GAP_S = 0.75          # frames closer than this are the same picture
DEDUP_MAX_COSINE = 0.92   # above this, two frames are the same shot


def _face_reason(face):
    """Why this face reading disqualifies the frame, or None."""
    if face is None:
        return None                     # the pass did not run — see select_frames
    if not face.get('ok', True):
        return 'no_face'
    det = face.get('det')
    if det is None or det < DET_MIN:
        return 'low_det'
    bbox = face.get('bbox_frac')
    if bbox is None or bbox < FACE_BBOX_MIN:
        return 'face_too_small'
    yaw = face.get('yaw')
    if yaw is not None and abs(yaw) > YAW_MAX:
        return 'extreme_profile'
    return None


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def select_frames(frames, *, limit, min_gap_s=MIN_GAP_S,
                  face_bbox_min=FACE_BBOX_MIN,
                  dedup_max_cosine=DEDUP_MAX_COSINE,
                  require_face=True):
    """Pick at most ``limit`` frames out of one clip's readings.

    ``frames`` is a list of dicts in decode order:
        t           seconds into the SOURCE (float — never a frame index, see
                    below), sharp, luma, and optionally
        face        {'ok', 'det', 'bbox_frac', 'yaw'} from the face pass
        embedding   a sequence of floats, for the dedup stage

    TIMESTAMPS, NEVER FRAME INDICES. Scraped material is routinely
    variable-frame-rate, where index n corresponds to no fixed instant — the
    same reason `VideoClip.start_frame` is stored as informative only and never
    cut from.

    ``require_face=False`` skips the face gates entirely, which is what a caller
    does when the face pass is unavailable. A frame carrying no ``face`` key is
    NOT treated as a failure even when ``require_face`` is on: absent evidence
    is not evidence of absence, and silently dropping every frame because a
    scorer did not run would look exactly like "this video has no faces".

    Returns ``{'picked': [...], 'rejected': {reason: count}}``. `picked` carries
    each chosen reading plus the ``reason`` it survived on, newest constraint
    last, so a caller can explain a thin result instead of just showing one.
    """
    if limit is None or limit <= 0:
        return {'picked': [], 'rejected': {'limit_is_zero': len(frames or [])}}

    rejected = {}

    def drop(reason, n=1):
        rejected[reason] = rejected.get(reason, 0) + n

    pool = []
    for f in (frames or []):
        luma = f.get('luma')
        if luma is None or not (_LUMA_SANE_LOW <= luma <= _LUMA_SANE_HIGH):
            drop('exposure')
            continue
        if f.get('sharp') is None:
            drop('unmeasured')
            continue
        if require_face:
            reason = _face_reason(f.get('face'))
            if reason:
                drop(reason)
                continue
        pool.append(f)

    # Rank by sharpness, then by time so a tie is resolved the same way twice.
    pool.sort(key=lambda f: (-f['sharp'], f.get('t', 0.0)))

    picked = []
    for f in pool:
        if len(picked) >= limit:
            drop('over_limit')
            continue
        t = f.get('t')
        if t is not None and any(
                p.get('t') is not None and abs(p['t'] - t) < min_gap_s
                for p in picked):
            drop('too_close')
            continue
        emb = f.get('embedding')
        if emb is not None and any(
                p.get('embedding') is not None
                and _cosine(emb, p['embedding']) > dedup_max_cosine
                for p in picked):
            drop('duplicate')
            continue
        picked.append(f)

    # Chronological output: a dataset ordered by descending sharpness reads as
    # shuffled, and every consumer downstream wants source order.
    picked.sort(key=lambda f: f.get('t', 0.0))
    return {'picked': picked, 'rejected': rejected}


def spread_quota(per_source_counts, total_limit):
    """Split a total frame budget across sources without letting one dominate.

    ``per_source_counts`` maps a source id to how many frames it could supply.
    A single long video will otherwise eat the entire budget and the dataset
    becomes one video's lighting, one video's grade and one video's wardrobe —
    the failure that `max_per_source` already guards against on the clip side.

    Largest-remainder over equal shares: everyone gets the same base, whatever
    they cannot use flows back, and the leftover goes to whoever has most to
    offer. Returns {source_id: quota}, and never promises a source more than it
    said it has.
    """
    sources = {k: max(0, int(v)) for k, v in (per_source_counts or {}).items()}
    if not sources or total_limit is None or total_limit <= 0:
        return {k: 0 for k in sources}

    quota = {k: 0 for k in sources}
    remaining = int(total_limit)
    # Repeatedly hand out an equal share; sources that cannot take theirs return
    # it to the pot, so a budget is never lost to a source that had one frame.
    while remaining > 0:
        hungry = [k for k in sources if quota[k] < sources[k]]
        if not hungry:
            break
        share = remaining // len(hungry)
        if share == 0:
            # Fewer frames left than sources: give them out one at a time, to
            # the sources with the most to offer, so the tail is deterministic.
            for k in sorted(hungry, key=lambda k: (-sources[k], str(k))):
                if remaining == 0:
                    break
                quota[k] += 1
                remaining -= 1
            break
        for k in hungry:
            take = min(share, sources[k] - quota[k])
            quota[k] += take
            remaining -= take
    return quota
