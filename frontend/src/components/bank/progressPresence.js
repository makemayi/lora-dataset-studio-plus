/* What the bank's progress zone should show, given the last snapshot we have
 * and whether the server is currently reachable.
 *
 * The bug: a failed poll left the page with no fresh snapshot, and "no
 * snapshot" was rendered as "nothing". A pass running happily in the bank's
 * server-side thread therefore looked identical to a pass that had stopped.
 * There are FOUR states here, not two — the two extra ones are the honest
 * "I don't know" the UI was missing.
 *
 * Plain .js (no JSX) so `node --test` can exercise it.
 */

export const PROGRESS_HIDDEN = 'hidden';     // nothing running, and we know it
export const PROGRESS_RUNNING = 'running';   // fresh snapshot, job in flight
export const PROGRESS_STALE = 'stale';       // last known snapshot + lost contact
export const PROGRESS_UNKNOWN = 'unknown';   // no snapshot at all + lost contact

/**
 * @param {object|null|undefined} activity last known `payload.activity`
 * @param {boolean} offline                is the server currently unreachable
 */
export function progressPresence(activity, offline = false) {
  const live = !!activity && !activity.finished;
  if (live) return offline ? PROGRESS_STALE : PROGRESS_RUNNING;
  return offline ? PROGRESS_UNKNOWN : PROGRESS_HIDDEN;
}
