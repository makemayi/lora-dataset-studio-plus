"""Who gives which framing, and when a picture may give more than one.

The rule, in the operator's words: "我会给你总数,如果总数超过了 bank 的图片数量,
那么就可以一图多用,否则不行". So the allocator has two passes — one picture one
framing while the pool covers the request, then reuse in rounds when it does not.

Pure on purpose: the plan endpoint and the job both call this, and a plan that
promised something the run then did not do would be worse than no plan at all.
"""

# Order matters twice: it is the tie-break when two framings are equally
# starved, and it is the order a reused picture offers its remaining framings.
FRAMINGS = ('face', 'half', 'full')

# The bank's head-angle facet buckets. A build that spends its quotas on
# whatever ranked first trains a LoRA that has seen one angle; walking the
# buckets ROUND-ROBIN spends the same quotas across the head angles instead.
# Unmeasured rows (no 'angle', or None) go LAST: a measurement the app holds
# must inform the mix, and an absent one must not block it.
ANGLE_BUCKETS = ('frontal', 'three_quarter', 'profile', 'behind')


def interleave_by_angle(pictures):
    """Round-robin the pictures over their angle buckets, rank kept within a
    bucket, unmeasured last (original order). A list with no 'angle' keys is
    returned in its original order — the interleave is invisible to callers
    that cannot measure angles."""
    buckets = {a: [] for a in ANGLE_BUCKETS}
    unmeasured = []
    for p in pictures:
        bucket = buckets.get(p.get('angle'))
        if bucket is None:
            unmeasured.append(p)
        else:
            bucket.append(p)
    groups = [buckets[a] for a in ANGLE_BUCKETS] + [unmeasured]
    out = []
    i = 0
    while any(groups):
        g = groups[i % len(groups)]
        if g:
            out.append(g.pop(0))
        i += 1
    return out


def _allowed(picture, remaining, already_given):
    if not picture.get('has_face_box'):
        candidates = ('full',)
    else:
        candidates = FRAMINGS
    return [f for f in candidates
            if remaining.get(f, 0) > 0 and f not in already_given]


def allocate(pictures, quotas):
    """Assign framings to pictures, best-ranked first within an angle bucket.

    ``pictures``: ``[{'id': int, 'has_face_box': bool, 'angle': str|None}]``
    ranked best first. The optional ``angle`` (the bank's head-angle facet)
    round-robins the walk across buckets so the spent quotas spread over the
    head angles; pictures without one keep their relative order, last.
    ``quotas``:   ``{'face': int, 'half': int, 'full': int}`` — a missing or
                  zero entry means that framing is not wanted at all.

    Returns ``{'assignments': [(picture_id, framing)], 'counts': {framing: n},
    'reused': n, 'shortfall': {framing: n}}``. ``reused`` counts PICTURES that
    gave more than one framing, which is the number the plan shows.
    """
    remaining = {f: int(quotas.get(f) or 0) for f in FRAMINGS}
    remaining = {f: n for f, n in remaining.items() if n > 0}
    # Pass 1 walks the buckets round-robin (see interleave_by_angle): when the
    # caps later bite, the pictures that survive follow the head-angle mix
    # instead of whoever's aesthetic score ranked first.
    pictures = interleave_by_angle(pictures)
    counts = {}
    given = {}                                   # picture id -> [framing, ...]
    assignments = []

    def _take(picture, framing):
        remaining[framing] -= 1
        if remaining[framing] == 0:
            del remaining[framing]
        counts[framing] = counts.get(framing, 0) + 1
        given.setdefault(picture['id'], []).append(framing)
        assignments.append((picture['id'], framing))

    # Pass 1 — one picture, one framing. Each takes the wanted framing with the
    # smallest running count so the split tracks the requested ratio instead of
    # filling `face` first and starving the rest.
    for picture in pictures:
        if not remaining:
            break
        candidates = _allowed(picture, remaining, given.get(picture['id'], ()))
        if not candidates:
            continue
        _take(picture, min(candidates,
                           key=lambda f: (counts.get(f, 0), FRAMINGS.index(f))))

    # Pass 2 — reuse, in ROUNDS, best picture first. One extra framing per
    # picture per round: 40 pictures cannot cover 90 images with one extra each,
    # and the rounds are what let the best pictures give all three.
    while remaining:
        progressed = False
        for picture in pictures:
            if not remaining:
                break
            already = given.get(picture['id'])
            if not already:
                continue                         # untouched in pass 1: unusable
            candidates = _allowed(picture, remaining, already)
            if not candidates:
                continue
            _take(picture, min(candidates,
                               key=lambda f: (counts.get(f, 0), FRAMINGS.index(f))))
            progressed = True
        if not progressed:
            break                                # nothing left to give

    wanted = {f: int(quotas.get(f) or 0) for f in FRAMINGS}
    shortfall = {f: n - counts.get(f, 0) for f, n in wanted.items()
                 if n - counts.get(f, 0) > 0}
    return {'assignments': assignments, 'counts': counts,
            'reused': sum(1 for v in given.values() if len(v) > 1),
            'shortfall': shortfall}
