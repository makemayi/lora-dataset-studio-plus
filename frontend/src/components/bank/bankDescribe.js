/* Turning the server's answer into a change of the visible controls.
 *
 * The grid's filter state and the API's facet names are NOT spelled the same
 * (`resBucket` here, `res_bucket` there). One map, in one place: a second copy of
 * this correspondence is how a field silently stops arriving — the summary would
 * still say "resolution above 2 MP" over a grid that never narrowed.
 *
 * Nothing here applies anything. It returns a patch the caller hands to the same
 * `setF` a chip click uses, so the request lands in controls the user can read and
 * undo, and the grid's own counters — not the model — say how many images it holds.
 */

/* API facet -> filter key in this component. Only these cross the boundary; a
 * field outside this map is one the server should never have sent (it validates
 * against the same vocabulary) and is reported rather than applied. */
const KEY = {
  status: 'status', flag: 'flag', medium: 'medium', framing: 'framing',
  angle: 'angle', origin: 'origin', res_bucket: 'resBucket',
  search: 'search', exclude: 'exclude',
}

/* Every filter key the translator is allowed to touch, so applying a new reading
 * CLEARS what a previous one set. Without this, two requests in a row compose into
 * a filter nobody asked for: "amateur" then "anime portraits" would keep the first
 * request's medium and return nothing, with a summary describing only the second. */
export const TOUCHED = Object.values(KEY)

export function toFilterPatch(res) {
  const from = (res && res.filter) || {}
  const patch = {}
  for (const k of TOUCHED) patch[k] = null       // reset the lane first — see above
  for (const [apiKey, value] of Object.entries(from)) {
    const key = KEY[apiKey]
    if (key) patch[key] = value
  }
  if (res && res.sort) patch.sort = res.sort
  return patch
}

/* The lines shown next to the request. Three registers, deliberately distinct:
 * what it DID (understood), what it COULD NOT (unsupported), and what it tried
 * that this bank does not have (dropped). Collapsing them into one paragraph is
 * how "I ignored half your sentence" starts reading like a success. */
export function describeSummary(res) {
  if (!res) return { understood: [], unsupported: [], dropped: [], refused: true }
  return {
    understood: res.understood || [],
    unsupported: res.unsupported || [],
    dropped: res.dropped || [],
    refused: !!res.refused,
  }
}

/* One sentence for the headline. It never claims a count: the chips below carry
 * the measured number, and repeating a number the model did not compute is how a
 * guess acquires the authority of a measurement. */
export function headline(res) {
  const s = describeSummary(res)
  if (s.refused && !s.understood.length) {
    return s.unsupported.length
      ? 'Nothing in that request maps onto what this bank has measured yet'
      : 'That did not come back as something the filters can express'
  }
  return `Filters set from your request — the counts below are measured, not guessed`
}
