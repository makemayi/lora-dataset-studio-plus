/* 📡 One place that knows whether the browser can still reach this server.
 *
 * THE BUG THIS EXISTS FOR
 * -----------------------
 * A phone left an "Analyze all" running, locked, came back. Every 2 s poll had
 * failed meanwhile, each one firing its own "Connection lost" banner — ten of
 * them stacked over the whole app — while the progress bar was simply GONE,
 * because a failed poll leaves the page with no snapshot to draw.
 *
 * Both halves are the same design mistake: an UNKNOWN state was rendered as an
 * EMPTY one, and an expected, automatic retry was announced like a user action
 * that just failed. The job itself was fine — bank passes run in a server-side
 * thread and survive the page leaving.
 *
 * THE MODEL
 * ---------
 *  • A network failure opens an "offline episode". Further failures do not
 *    re-announce it — they belong to the same episode.
 *  • A background poll NEVER announces. Its whole voice is the persistent
 *    indicator this store drives.
 *  • A user-triggered call announces ONCE per episode: the user pressed a
 *    button and is owed an answer, but pressing it five times owes five
 *    answers, not fifty.
 *  • Any response at all — even a 500 — proves the server is reachable and
 *    closes the episode, which is worth exactly one "Back online".
 *
 * Deliberately framework-free (no React import): `node --test` cannot parse
 * JSX, and every rule above is testable arithmetic.
 */

const INITIAL = Object.freeze({
  online: true,
  /** Failures accumulated in the current episode (0 while online). */
  failures: 0,
  /** Timestamp the current episode opened, or null while online. */
  offlineSince: null,
  /** Timestamp of the last recovery — lets a UI show "back" transiently. */
  restoredAt: null,
});

let state = INITIAL;
/* Whether the CURRENT episode has already been announced to the user. Kept out
   of `state` on purpose: it is bookkeeping for the notifier, not something a
   view should render or re-render on. */
let announced = false;

const listeners = new Set();

function commit(next) {
  state = next;
  for (const fn of [...listeners]) fn(state);
}

export function getConnectionState() {
  return state;
}

/** Subscribe to connection changes. Returns the unsubscribe function. */
export function subscribeConnection(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/**
 * Record a request that never reached the server.
 * @returns {boolean} true when THIS failure is worth a visible notification.
 */
export function reportRequestFailure({ background = false, now = Date.now() } = {}) {
  const opening = state.online;
  commit({
    online: false,
    failures: state.failures + 1,
    offlineSince: opening ? now : state.offlineSince,
    restoredAt: state.restoredAt,
  });
  if (background) return false;
  if (announced) return false;
  announced = true;
  return true;
}

/**
 * Record a request that got a response (any status — reachability is the point).
 * @returns {boolean} true when this closes an offline episode, i.e. the one
 *                    moment worth saying "Back online".
 */
export function reportRequestSuccess({ now = Date.now() } = {}) {
  if (state.online) return false;
  announced = false;
  commit({ online: true, failures: 0, offlineSince: null, restoredAt: now });
  return true;
}

/** Test seam — also used if the app ever needs a clean slate after a reload. */
export function resetConnectionStatus() {
  announced = false;
  commit(INITIAL);
}
