// 🔤 Text search — the wording that keeps a SIMILARITY RANKING from reading as a
// filter, kept out of the JSX so `node --test` can exercise it.
//
// The honesty problem this file exists for: a text search returns the N closest
// images, and "closest" is never "matching". Every image in the bank has some
// cosine to every phrase, so a result list ALWAYS comes back full, however wrong.
// If the UI just shows pictures, the user reads "these are the ones that match"
// — and a bad query produces a confident, wrong-looking answer with no signal.
// So three things are always said: that it is a ranking, how strong the top and
// bottom of it are, and how many images could not be searched at all.

// ── Why there is NO absolute quality scale here ──────────────────────────────
// CLIP text/image cosines are not spread over 0..1 and their useful band is far
// lower and narrower than intuition suggests. Measured on this codebase's exact
// model (ViT-L-14 / openai, CPU) over stock photographs: a CORRECT top-1 match
// peaked at 0.20–0.23, while unrelated pairs sat at 0.09–0.16. So the whole
// usable range was ≈0.09–0.23.
//
// That kills the obvious designs:
//   • an absolute threshold — a "sensible-looking" 0.25 cut returns NOTHING,
//     ever, including on perfect matches, and reads as "search is broken";
//   • a percentage — showing the best possible result as "22% match" reads as a
//     failure when it is in fact the ceiling;
//   • hard-coding 0.23 as "the top" — that figure came from five stock photos.
//     A bank of portraits, or of one person, will sit somewhere else entirely.
//
// So nothing here is absolute. Strength is expressed as the SPREAD WITHIN the
// returned set: how much weaker the last result is than the best one, as a
// fraction of the best. That question ("can I trust the tail?") is the one the
// user actually has, and it needs no universal constant to answer — it stays
// correct whatever band a particular bank happens to live in.
// The yardstick is measured, never assumed: `pool_median` is what a TYPICAL
// image of this bank scores for this query. So:
//     lift   = top - pool_median   → how far the best result stands out at all
//     spread = top - bottom        → how much the returned set varies internally
// Both are in the same (unknown, per-bank) units, so their RATIO is meaningful
// while neither value alone is. This matters most in the app's main use case: a
// bank dominated by one subject has image-to-image cosines of 0.60–0.89, and the
// discriminating gap of a query compresses by 30–70%. A rule written against an
// absolute band would simply never fire there; a ratio against the measured
// median fires exactly when the ranking stops separating anything.
export const SPREAD_BANDS = [
  { maxRatio: 0.35, label: 'all about equally close' },
  { maxRatio: 0.80, label: 'the last ones are noticeably looser' },
  { maxRatio: Infinity, label: 'the tail is much weaker than the top' },
]

// Below this fraction of the top score, the best result is barely above a
// typical image of the bank — the ranking is not really separating anything.
const FLAT_LIFT_RATIO = 0.05

/**
 * Describe a result set relative to what this bank typically scores for this
 * query. Returns '' when there is nothing to say (missing data).
 */
export function spreadLabel(range, poolMedian) {
  const top = Number(range?.top)
  const bottom = Number(range?.bottom)
  if (!Number.isFinite(top) || !Number.isFinite(bottom) || top <= 0) return ''
  const median = Number(poolMedian)
  if (!Number.isFinite(median)) {
    // No baseline: fall back to the set's own drop, still a pure ratio.
    const drop = (top - bottom) / top
    return (SPREAD_BANDS.find((b) => drop / 0.3 <= b.maxRatio)
      || SPREAD_BANDS[SPREAD_BANDS.length - 1]).label
  }
  const lift = top - median
  if (lift <= 0 || lift / top < FLAT_LIFT_RATIO) {
    return 'barely above what any image here scores — the order is a hint at best'
  }
  const ratio = (top - bottom) / lift
  return (SPREAD_BANDS.find((b) => ratio <= b.maxRatio)
    || SPREAD_BANDS[SPREAD_BANDS.length - 1]).label
}

/**
 * The one-line summary announced to screen readers and shown above the grid.
 * Never claims a match: it says "closest", gives the range, and names what was
 * left out.
 */
export function summarize(result) {
  if (!result) return ''
  const shown = (result.image_ids || []).length
  if (!shown) {
    return `No scored image to rank for “${result.query}”. `
      + unsearchableNote(result)
  }
  const r = result.score_range || {}
  const spread = spreadLabel(r, result.pool_median)
  const parts = [
    `${shown} closest of ${result.pool} for “${result.query}”, best first.`,
    // Raw cosines, shown as what they are and never as a percentage: on this
    // model even a perfect match tops out around 0.2, so "22%" would read as a
    // failure. The SPREAD is the part that is actually actionable.
    `Similarity ${fmt(r.top)} down to ${fmt(r.bottom)}${spread ? ` — ${spread}` : ''}.`,
    'Search brings the likeliest images to the front; it does not select them. '
      + 'Every image scores something against every phrase.',
  ]
  const note = unsearchableNote(result)
  if (note) parts.push(note)
  return parts.join(' ')
}

function fmt(v) {
  const n = Number(v)
  return Number.isFinite(n) ? n.toFixed(2) : '—'
}

/**
 * The load-bearing warning. An image with no ✨ Score embedding cannot be found
 * by ANY phrase — staying silent about it lets the user conclude their picture
 * is missing from the bank.
 */
export function unsearchableNote(result) {
  const unscored = Number(result?.unscored) || 0
  if (unscored <= 0) return ''
  return `${unscored} of ${result.filtered} image(s) in this filter have no ✨ Score `
    + 'embedding yet and could NOT be searched — run ✨ Score to include them.'
}

/**
 * What to tell the user BEFORE they press the button, so a slow first search is
 * never an unexplained freeze. Mirrors the measured cost: ~8.5 s to load CLIP,
 * ~20 ms per phrase afterwards.
 */
export function readinessHint(status) {
  if (!status) return ''
  if (!status.available) {
    return status.reason || 'Text search is unavailable on this install.'
  }
  if (status.weights_warning) return status.weights_warning
  if (status.warm) return 'Search model is loaded — results are instant.'
  return 'First search loads the search model (about 10 seconds, on the CPU). '
    + 'After that, searches are instant.'
}

/** The pending-state label, so the wait is attributed to the right cause. */
export function pendingLabel(status) {
  return status?.warm ? 'Searching…' : 'Loading the search model…'
}

/**
 * What CLIP genuinely cannot do. Shown in the panel, not buried in a doc,
 * because a user who trusts a wrong answer here silently builds a bad dataset.
 * These are the documented, reproducible weaknesses of CLIP-style contrastive
 * models — not a disclaimer for legal comfort.
 */
// Measured on this exact model, not folklore:
//  • counting — "two people" beat "one person" on a two-person image by 0.001,
//    i.e. noise. It separates "one" from "several" at best, nothing finer.
//  • negation — on a photo of an astronaut WEARING a helmet:
//      "an astronaut with a helmet"    0.212
//      "an astronaut without a helmet" 0.217   ← the negation scored HIGHER
//      "an astronaut"                  0.219
//    CLIP does not penalise "without", it ignores it. That is not imprecision,
//    it is a silent inversion: ask for "without glasses" and you get glasses,
//    with no signal that anything went wrong. Hence the phrasing advice below —
//    describe what IS in the frame.
export const CLIP_LIMITS = [
  'Counting — “two people” barely outranks one person; expect “one vs several” at best.',
  'Negation — “without glasses” returns glasses. The word “without” is ignored, not applied.',
  'Spatial relations — “to the left of” carries almost no meaning.',
]

export function limitsSentence() {
  return 'Best at subjects, settings, styles and framing. It cannot count, '
    + 'cannot handle “without”, and ignores left/right — so describe what IS in '
    + 'the shot rather than what is missing.'
}

/**
 * Clamp the requested result count the same way the sibling selectors do.
 */
export function clampN(value) {
  const n = Number(value)
  if (!Number.isFinite(n)) return 1
  return Math.max(1, Math.min(2000, Math.round(n)))
}
