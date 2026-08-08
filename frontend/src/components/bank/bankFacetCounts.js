// 🗃️ Bank chip counters — WHICH number a chip prints, and WHEN to go get it.
//
// The chips used to print the workspace payload's bank-wide totals whatever else
// was filtered. Measured on a 50 397-image bank with ✕ Rejected picked: the row
// offered "📺 Noisy 1 136" and "🔞 NSFW 20 540" over grids of 527 and 3 364
// rows. True about the bank, and describing nothing the user could see.
//
// So there are TWO maps per facet from here on, and they are not
// interchangeable:
//   * the FILTERED count (`/api/bank/:id/facets`, measured with every other
//     filter in force and the facet's own value lifted) — what the chip PRINTS;
//   * the bank-wide count (the payload) — what decides a chip EXISTS at all.
// Driving visibility off the filtered map instead would make rows of chips
// vanish as soon as a filter emptied them, which is the one thing a filter row
// must never do: you would have no way back.

// Every facet key of the workspace's filter state. `cluster`, `style` and
// `subfolder` are meaningful at 0 / '' (cluster 0 is a cluster, '' is the bank
// root), so they are tested against null, not for truthiness.
const NULLABLE = ['cluster', 'style', 'subfolder']
const TRUTHY = ['status', 'flag', 'search', 'exclude', 'tags', 'resBucket',
  'framing', 'origin', 'medium', 'angle']

/** Is anything narrowing the grid right now? `sort` is NOT a filter — it
 *  reorders the same rows, so it never changes a count. */
export function isFacetFiltered(filter) {
  if (!filter) return false
  return TRUTHY.some((k) => !!filter[k])
    || NULLABLE.some((k) => filter[k] != null)
}

/** A short string that changes exactly when the DATA behind the chips moved —
 *  a decision, an import, or a pass landing. The facet fetch keys off this
 *  instead of the payload object, whose identity changes on every 2 s job poll:
 *  re-measuring seven facets twice a second to print the same numbers is the
 *  kind of cost that only shows up on someone else's 100 000-image bank. */
export function facetDataKey(counts, thresholds) {
  const moved = counts ? ['total', 'pending', 'keep', 'reject', 'scanned',
    'scored', 'watermark_scanned', 'framing_classified', 'medium_classified',
    'angle_measured'].map((k) => counts[k] || 0).join('.') : ''
  // The 🎚 thresholds belong in here even though they move no counter: the flag
  // verdicts are recomputed from stored scores at read time, so lowering
  // "sharpness min" re-slices 🌫 Blurry with every count in the payload
  // unchanged. Leave them out and the chips keep the numbers of the thresholds
  // you just stopped using.
  if (!thresholds) return moved
  return `${moved}|${Object.keys(thresholds).sort()
    .map((k) => `${k}=${thresholds[k]}`).join(',')}`
}

const EMPTY = {}

/** The maps the chip row reads: `.print` for the numbers, `.wide` for whether a
 *  chip is offered at all. `facets` is the filtered answer, or null while none
 *  has been fetched (no filter, or the request has not landed yet) — in which
 *  case both are the same bank-wide truth and nothing on screen moves. */
export function chipCounts(payload, facets) {
  const wide = {
    flags: payload?.flags || EMPTY,
    resBuckets: payload?.res_buckets || EMPTY,
    framing: payload?.framing || EMPTY,
    origins: payload?.origins || EMPTY,
    mediums: payload?.mediums || EMPTY,
    angles: payload?.angles || EMPTY,
  }
  const print = facets ? {
    flags: facets.flags || EMPTY,
    resBuckets: facets.res_buckets || EMPTY,
    framing: facets.framing || EMPTY,
    origins: facets.origins || EMPTY,
    mediums: facets.mediums || EMPTY,
    angles: facets.angles || EMPTY,
  } : wide
  return { print, wide, filtered: !!facets }
}
