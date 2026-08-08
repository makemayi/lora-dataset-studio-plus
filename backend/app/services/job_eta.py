"""How long a bank pass still has to run — measured, not extrapolated from birth.

WHY NOT ``done / (now - started_at)``.  Every long pass in this app changes
speed by an order of magnitude *during* the run, so an average taken from the
start is not a rough estimate, it is a wrong one:

* ✨ Score resumes from its cache.  A real run reported ``resuming — 20847 of
  21220 already cached``: twenty thousand rows swallowed in seconds, then 373
  images paid for on the GPU.  The average-since-start says "a few seconds" for
  the ninety minutes that follow.
* Every model-backed pass spends its first seconds loading weights, at zero
  images per second.  That dead time never leaves an average-since-start.
* A measured window is right: 6.4 images/s over 45 s predicted 1 h 27, and 1 h
  27 is what it took.

So the rate comes from a SLIDING WINDOW of recent ``(instant, done)`` samples,
bounded in both count and age, held in the job dict (in memory, dies with the
process, like the rest of ``bank_jobs``).

THE THREE THINGS THIS MODULE REFUSES TO DO

1. **It does not speak before it is sure.**  The first samples of any pass are
   noise.  A banner that says "3 hours" and then "20 minutes" two minutes later
   has not given the user a number, it has taught them not to read the number —
   and that damage is permanent for the whole feature.  An estimate is published
   only after it has agreed with itself, inside a band as wide as the DISPLAY
   rounding, for :data:`STABLE_SECONDS` continuous seconds.  Until then the
   caller shows "estimating…", which promises nothing.
2. **It does not average across a phase boundary.**  ✨ Score runs child
   inference, then writes ~21 000 rows, then groups styles (181 s measured on
   23 000 images).  Seconds-per-item in one of those says nothing about the
   next, so the window is CLEARED whenever the work being counted changes —
   detected from the numbers themselves (``total`` changed, or ``done`` went
   backwards), never from the free-text detail, which several passes rewrite
   mid-phase.  After the first such transition the answer is explicitly scoped
   to the current step (:func:`read` returns ``scope='phase'``) so the caller
   can say so out loud.
3. **It does not estimate what nobody is counting.**  A phase that publishes
   ``done=0, total=0`` (the style grouping, the semantic dedup comparison) has
   no unit of work, and a duration invented for it would be a guess wearing a
   number's clothes.  Those get ``state='none'`` — silence, exactly like the
   counter that already refuses to print a bare "0" there.

TIME IS ALWAYS INJECTED.  Every function takes ``now``; nothing here reads the
clock.  A test that has to sleep to exercise a rate is a test that goes flaky
the moment the machine is busy — which, on a machine running several passes, is
precisely when this code matters.
"""

# The rate is measured over at most this much recent history.  Long enough to
# ride out one slow image, short enough that a real slowdown shows up fast.
WINDOW_SECONDS = 60.0
# ...and never over less than this, however fast the items arrive.
MIN_WINDOW_SECONDS = 20.0
# ...nor over fewer than this many completed items.  Both floors matter: 5 items
# in 0.3 s is a burst out of a cache, not a speed.
MIN_WINDOW_ITEMS = 5
# Reports closer together than this are folded into the newest sample.  A fast
# pass calls in thousands of times a minute and every one of those points would
# otherwise be stored and re-scanned; half a second of resolution is far finer
# than anything the estimate can express.
SAMPLE_MIN_GAP = 0.5
# Ring bound, belt and braces alongside the age pruning above.
MAX_SAMPLES = 256

# BURST GUARD.  The window is a good smoother, but it is only honest while the
# work inside it ran at ONE speed.  ✨ Score resuming from cache swallows twenty
# thousand rows in about two seconds and then crawls; for the next minute the
# window straddles both, and the rate it reports belongs to neither.  So: if one
# half of the window carries this many times the items of the other, the window
# is not measuring a speed and nothing is published from it.  Checked in both
# directions — a burst that lands in the RECENT half flatters the estimate just
# as badly as one in the old half.
BURST_RATIO = 4

# STABILITY, and why these two numbers.  The estimate is published once its
# spread has stayed inside `high <= low * STABLE_BAND + STABLE_SLACK` for
# STABLE_SECONDS.  STABLE_BAND is 1.35 because that is roughly ONE DISPLAY
# BUCKET wide at the coarse end (2 h vs 2 h 30, 20 min vs 25 min): the criterion
# is calibrated to what the user can actually SEE change, not to an abstract
# statistic.  An estimate that holds inside a band the rounding cannot resolve
# will not visibly jump — which is the entire promise.  STABLE_SLACK keeps short
# estimates from failing the ratio test on noise that is seconds wide.
STABLE_BAND = 1.35
STABLE_SLACK = 5.0
STABLE_SECONDS = 15.0

# No progress report for this long and the published number stops being a
# measurement: a wedged pass must not keep promising "about 20 minutes".
STALE_SAMPLE_SECONDS = 90.0

# What :func:`read` can answer.
ETA_NONE = 'none'              # nothing countable here — say nothing
ETA_ESTIMATING = 'estimating'  # countable, but not yet trustworthy
ETA_READY = 'ready'            # a number that has held still


def new_state():
    """A fresh estimator, before the job has reported anything."""
    return {
        'samples': [],        # [(instant, done)], oldest first, age-bounded
        'anchor': None,       # when the newest sample slot was opened
        'done': 0,
        'total': None,        # None = never observed; a real 0 IS a phase
        'phases': 1,
        'eta': None,          # last computed seconds remaining
        'published': False,   # has this phase earned the right to show a number
        'streak_low': None,
        'streak_high': None,
        'streak_started': None,
        'last_at': None,
    }


def _start_phase(state):
    state['samples'] = []
    state['anchor'] = None
    state['eta'] = None
    state['published'] = False
    state['streak_low'] = None
    state['streak_high'] = None
    state['streak_started'] = None


def _window(samples):
    """(span_seconds, items_done) across the kept samples."""
    if len(samples) < 2:
        return 0.0, 0
    return samples[-1][0] - samples[0][0], samples[-1][1] - samples[0][1]


def _is_burst(samples):
    """Does this window straddle a speed change instead of measuring a speed?"""
    if len(samples) < 3:
        # Two points cannot disagree with themselves.  The item and span floors
        # are what hold the line here.
        return False
    middle = (samples[0][0] + samples[-1][0]) / 2.0
    split = 1
    while split < len(samples) - 1 and samples[split][0] < middle:
        split += 1
    older = samples[split][1] - samples[0][1]
    newer = samples[-1][1] - samples[split][1]
    high, low = max(older, newer), min(older, newer)
    return high >= MIN_WINDOW_ITEMS and high >= BURST_RATIO * max(low, 1)


def observe(state, done, total, now):
    """Record one progress report.  Cheap: called on every ``bump``."""
    done = int(done or 0)
    total = int(total or 0)

    if state['total'] is not None and (total != state['total'] or done < state['done']):
        # The work being counted changed under us.  Anything measured before
        # this instant describes different work; keeping it would produce a
        # duration for a phase that is already over.
        state['phases'] += 1
        _start_phase(state)
    state['done'] = done
    state['total'] = total
    state['last_at'] = now

    samples = state['samples']
    # Open a new sample slot only when something completed AND the newest slot
    # has been open for at least one resolution step.  Otherwise MOVE the newest
    # point.  Two behaviours ride on that: a fast pass does not store thousands
    # of points a minute, and a STALL drags the newest point forward in time
    # without adding items — which lowers the rate instead of leaving it frozen
    # at its last healthy value.  The age is measured from when the slot was
    # OPENED, never from the point itself: that point keeps moving, so comparing
    # against it meant the gap was never reached and the window stayed one
    # sample wide forever.
    if samples and not (done != samples[-1][1]
                        and now - state['anchor'] >= SAMPLE_MIN_GAP):
        samples[-1] = (now, done)
    else:
        samples.append((now, done))
        state['anchor'] = now
    if len(samples) > MAX_SAMPLES:
        del samples[:-MAX_SAMPLES]
    cutoff = now - WINDOW_SECONDS
    while len(samples) > 2 and samples[1][0] < cutoff:
        samples.pop(0)

    raw = _raw_estimate(state, done, total)
    if raw is not None:
        state['eta'] = raw
    elif not state['published']:
        state['eta'] = None
    _update_streak(state, raw, now)
    return state


def _raw_estimate(state, done, total):
    if total <= 0 or done >= total:
        return None
    samples = state['samples']
    span, items = _window(samples)
    # The item floor guards the FIRST publication — five items in a burst is not
    # a speed.  Once a number is on screen the floor drops to one, because the
    # alternative is worse: a pass that decays to two items a minute would fall
    # under the floor, stop refreshing, and leave its old, far-too-short promise
    # standing in front of the user for as long as the crawl lasts.  One item
    # over a full window is a slow rate, not an absent one.
    floor = 1 if state['published'] else MIN_WINDOW_ITEMS
    if span < MIN_WINDOW_SECONDS or items < floor:
        return None
    if _is_burst(samples):
        return None
    return (total - done) * span / items


def _update_streak(state, raw, now):
    if state['published']:
        return
    if raw is None:
        state['streak_started'] = None
        state['streak_low'] = None
        state['streak_high'] = None
        return
    if state['streak_started'] is None:
        state['streak_low'] = state['streak_high'] = raw
        state['streak_started'] = now
        return
    low = min(state['streak_low'], raw)
    high = max(state['streak_high'], raw)
    if high > low * STABLE_BAND + STABLE_SLACK:
        # It is still moving.  Start counting again from here — not from zero
        # history, from THIS value, which is our best current belief.
        state['streak_low'] = state['streak_high'] = raw
        state['streak_started'] = now
        return
    state['streak_low'] = low
    state['streak_high'] = high
    if now - state['streak_started'] >= STABLE_SECONDS:
        state['published'] = True


def read(state, now):
    """(state, seconds_or_None, scope) for the snapshot the UI polls.

    ``scope`` is ``'phase'`` once this job has been seen to change phase, so the
    caller can bound its sentence to the current step instead of implying it
    covers the whole pass.
    """
    scope = 'phase' if state['phases'] > 1 else 'job'
    total = state['total'] or 0
    if total <= 0 or state['done'] >= total:
        return ETA_NONE, None, scope
    fresh = (state['last_at'] is not None
             and now - state['last_at'] <= STALE_SAMPLE_SECONDS)
    if state['published'] and state['eta'] is not None and fresh:
        # Raw seconds. Deciding that eight hours is "more than a day" territory,
        # or that 113 minutes reads as "about 2 hours", is a DISPLAY judgement
        # and lives with the display.
        return ETA_READY, state['eta'], scope
    return ETA_ESTIMATING, None, scope
