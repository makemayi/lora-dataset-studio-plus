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
# The SIZE floor mirrors the scorer too, and a first draft here got it wrong.
# It shipped at 0.12 on the argument that a small face "carries almost no
# identity signal once it is resized for training". Measured on dataset 4
# (2026-08-16) that argument is false: faces at 2-5 % of the photo — full-body
# shots, which is exactly what a LoRA set holds — scored 0.48-0.89, the same
# distribution as the rows already called scorable. A 0.12 floor would have
# thrown away most of the full-body variety a set needs, for a reason that does
# not survive measurement. Callers who want a stricter crop pass `face_bbox_min`.
FACE_BBOX_MIN = 0.02

MIN_GAP_S = 0.75          # frames closer than this are the same picture
DEDUP_MAX_COSINE = 0.92   # above this, two frames are the same shot

# ── The character gates ──────────────────────────────────────────────────────
# Off by default; a CHARACTER dataset turns them on, because that is the one
# kind where a frame that is technically fine and shows the wrong face — or a
# face too small to hold detail — is worse than no frame at all.
#
# FACE PIXELS, NOT A FRACTION. `bbox_frac` is an AREA fraction, so the same 3 %
# is a ~100 px face in a 4K frame and a ~20 px face at 720p. One trains, the
# other is a smudge the model learns as this person's face. The fraction gates
# presence; only the pixel count gates usability, and it needs the frame's own
# dimensions to be computed at all.
MIN_FACE_PX = 96.0
# "Somebody is here" vs "THIS person is here". Only applied when a reference
# exists — with none, there is nothing to be similar to and the gate would
# silently reject everything.
MIN_SIM = 0.35


def face_pixels(frame):
    """The face's approximate edge in PIXELS, or None when it cannot be known.

    `bbox_frac` is an area fraction of the whole picture, so the edge is its
    square root scaled by the frame's own size. Without the frame dimensions the
    answer is None and the caller must not guess one — a fabricated pixel count
    is worse than an ungated frame, because it looks like a measurement.
    """
    face = frame.get('face') or {}
    bbox = face.get('bbox_frac')
    w, h = frame.get('w'), frame.get('h')
    if not bbox or not w or not h:
        return None
    return (float(bbox) * float(w) * float(h)) ** 0.5


def _face_reason(frame, *, face_bbox_min=None, min_face_px=None, min_sim=None):
    """Why this frame's face reading disqualifies it, or None."""
    face = frame.get('face')
    if face is None:
        return None                     # the pass did not run — see select_frames
    if not face.get('ok', True):
        return 'no_face'
    det = face.get('det')
    if det is None or det < DET_MIN:
        return 'low_det'
    bbox = face.get('bbox_frac')
    if bbox is None or bbox < (FACE_BBOX_MIN if face_bbox_min is None
                               else face_bbox_min):
        return 'face_too_small'
    yaw = face.get('yaw')
    if yaw is not None and abs(yaw) > YAW_MAX:
        return 'extreme_profile'
    if min_face_px:
        px = face_pixels(frame)
        # Unknown is NOT a pass: a character set asked for a measured floor, and
        # letting an unmeasurable frame through would quietly disable the gate
        # for exactly the decoder that forgot to report its own size.
        if px is None:
            return 'face_px_unknown'
        if px < min_face_px:
            return 'face_too_few_pixels'
    if min_sim is not None:
        sim = face.get('sim')
        # Only meaningful when the scorer had a reference. `sim` absent means it
        # did not compute one, which is not the same as "does not resemble".
        if sim is not None and sim < min_sim:
            return 'wrong_person'
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
                  require_face=True, min_face_px=None, min_sim=None,
                  sharp_tolerance=None, face_tolerance=None):
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

    # Adaptive sharpness floors: a blur score only means something relative to
    # the clip it came from (resolution and codec shift the scale), so the floor
    # is max × tolerance, not an absolute number. A zero max (a degenerate
    # clip) disables the gate — the exposure filter already owns pure-black
    # frames. The face floor needs readings to exist at all; absent face_sharp
    # everywhere is absent evidence, never a rejection.
    sharps = [f.get('sharp') for f in (frames or []) if f.get('sharp') is not None]
    max_sharp = max(sharps) if sharps else 0.0
    face_sharps = [
        ((f.get('face') or {}).get('face_sharp'))
        for f in (frames or []) if (f.get('face') or {}).get('face_sharp') is not None
    ]
    max_face_sharp = max(face_sharps) if face_sharps else None

    pool = []
    for f in (frames or []):
        luma = f.get('luma')
        if luma is None or not (_LUMA_SANE_LOW <= luma <= _LUMA_SANE_HIGH):
            drop('exposure')
            continue
        if f.get('sharp') is None:
            drop('unmeasured')
            continue
        if sharp_tolerance and max_sharp > 0 and f['sharp'] < max_sharp * sharp_tolerance:
            drop('too_blurry')
            continue
        if require_face:
            reason = _face_reason(f, face_bbox_min=face_bbox_min,
                                  min_face_px=min_face_px, min_sim=min_sim)
            if reason:
                drop(reason)
                continue
            # Face-region sharpness gate: the frame can be globally sharp while
            # the face is blurry (it moved during the exposure). Adaptive like
            # the global gate; absent face_sharp = no evidence, never a reject.
            face_sharp = (f.get('face') or {}).get('face_sharp')
            if (face_tolerance and face_sharp is not None and max_face_sharp
                    and face_sharp < max_face_sharp * face_tolerance):
                drop('face_blurry')
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
