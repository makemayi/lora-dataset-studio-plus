/** 🎬 Keeping and rejecting shots — selection, and the ONE footgun of this API.
 *
 * THE FOOTGUN: `POST /video-bank/<id>/triage` with an empty (or absent) `ids`
 * means EVERY CLIP IN THE BANK. That convention is deliberate and it is what
 * "Reject everything, then keep what's good" needs — but it turns the most
 * ordinary front-end bug in existence, "the selection was empty and we posted it
 * anyway", into a silent bulk overwrite of hundreds of decisions.
 *
 * So no component builds that body by hand. `triagePayload` REFUSES an empty
 * selection (returns null) and `triageAllPayload` is the separate, explicit way
 * to say "all of them". Two functions, because the difference between them is
 * an afternoon of triage.
 *
 * PURE: no JSX, no fetch.
 */

export const TRIAGE_STATUSES = ['pending', 'keep', 'reject']

/** Add or remove one id. Works on an array (what the components hold) and keeps
 * insertion order, so "12 selected" always lists the same twelve. */
export function toggleSelection(selected, id) {
  const list = Array.isArray(selected) ? selected : []
  return list.includes(id) ? list.filter((x) => x !== id) : [...list, id]
}

/** Shift-click range over the CURRENTLY VISIBLE ids, which is the only order the
 * user can see. Selecting over the underlying database order would pick clips
 * that are not on screen. */
export function selectRange(selected, visibleIds, anchorId, targetId) {
  const ids = Array.isArray(visibleIds) ? visibleIds : []
  const a = ids.indexOf(anchorId)
  const b = ids.indexOf(targetId)
  if (a < 0 || b < 0) return toggleSelection(selected, targetId)
  const [from, to] = a <= b ? [a, b] : [b, a]
  const span = ids.slice(from, to + 1)
  const out = Array.isArray(selected) ? [...selected] : []
  for (const id of span) if (!out.includes(id)) out.push(id)
  return out
}

/** The body for "apply this status to what I selected", or NULL when nothing is
 * selected. Null is not an oversight: posting `{ids: []}` would retag the entire
 * bank. Callers must treat null as "there is nothing to do". */
export function triagePayload(selectedIds, status, reason) {
  const ids = (Array.isArray(selectedIds) ? selectedIds : []).filter((v) => v != null)
  if (ids.length === 0) return null
  if (!TRIAGE_STATUSES.includes(status)) return null
  const body = { ids, status }
  if (status === 'reject' && (reason || '').trim()) body.reason = reason.trim()
  return body
}

/** The body for "apply this to EVERY clip in the bank" — the server's empty-ids
 * convention, spelled out at the call site so it can never be reached by
 * accident. `ids: []` is sent explicitly rather than omitted: the wire then says
 * what was meant. */
export function triageAllPayload(status, reason) {
  if (!TRIAGE_STATUSES.includes(status)) return null
  const body = { ids: [], status }
  if (status === 'reject' && (reason || '').trim()) body.reason = reason.trim()
  return body
}

/** The confirmation an "everything" action must show. A bulk retag is not
 * undoable in this lane, and the count is what makes it real. */
export function triageAllConfirmation(status, total) {
  const verb = { keep: 'Keep', reject: 'Reject', pending: 'Reset' }[status] || status
  return `${verb} ALL ${total} shot${total === 1 ? '' : 's'} in this bank?\n\n`
    + 'This replaces every decision you have already made here, including the ones '
    + 'that are not currently on screen.'
}

/** The filter chips. `all` is a UI-only value and must never reach the query —
 * see videoBankApi.videoClipsUrl, which drops it. */
export const STATUS_FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'pending', label: 'To triage' },
  { key: 'keep', label: 'Kept' },
  { key: 'reject', label: 'Rejected' },
]

/** How many the current filter holds, straight off the bank's counters — so the
 * chips carry numbers without a request each. */
export function statusFilterCount(counts, key) {
  const c = counts || {}
  if (key === 'all') return Number(c.clips) || 0
  return Number(c[key]) || 0
}

/** "No shot matches …" — worded so an empty result never reads as a broken grid.
 * The reason it is empty is the whole message. */
export function emptyGridMessage({ status, sourceName, counts }) {
  const c = counts || {}
  if (!Number(c.sources)) return 'This bank has no files yet.'
  if (!Number(c.clips)) return 'No shots yet — run ▶ Run everything to cut your files into shots.'
  if (sourceName && status && status !== 'all') {
    return `No ${statusWord(status)} shot in ${sourceName}.`
  }
  if (sourceName) return `No shot in ${sourceName}.`
  if (status && status !== 'all') return `No ${statusWord(status)} shot in this bank.`
  return 'No shots on this page.'
}

function statusWord(status) {
  return { pending: 'untriaged', keep: 'kept', reject: 'rejected' }[status] || status
}

/** Pagination maths for the grid's "Load more". Kept here because "did the page
 * come back short?" is the only reliable end-of-list signal the API gives. */
export function hasMore({ loaded, total }) {
  return Number(loaded) < Number(total || 0)
}
