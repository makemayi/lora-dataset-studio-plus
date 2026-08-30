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


def _allowed(picture, remaining, already_given):
    if not picture.get('has_face_box'):
        candidates = ('full',)
    else:
        candidates = FRAMINGS
    return [f for f in candidates
            if remaining.get(f, 0) > 0 and f not in already_given]


def allocate(pictures, quotas):
    """Assign framings to pictures, best-ranked first.

    ``pictures``: ``[{'id': int, 'has_face_box': bool}]`` ranked best first.
    ``quotas``:   ``{'face': int, 'half': int, 'full': int}`` — a missing or
                  zero entry means that framing is not wanted at all.

    Returns ``{'assignments': [(picture_id, framing)], 'counts': {framing: n},
    'reused': n, 'shortfall': {framing: n}}``. ``reused`` counts PICTURES that
    gave more than one framing, which is the number the plan shows.
    """
    remaining = {f: int(quotas.get(f) or 0) for f in FRAMINGS}
    remaining = {f: n for f, n in remaining.items() if n > 0}
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
