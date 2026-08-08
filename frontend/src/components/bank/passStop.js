/* What the Bank's Stop button SAYS — before the click, at the click, and after.
 *
 * Measured on a live 36 925-image bank while ✨ Score was writing its rows:
 *
 *   POST /api/bank/<id>/cancel      79 ms
 *   GET  /api/bank/<id>            2745 ms   ← the banner's own payload
 *
 * The button was never broken. It was MUTE: a bare `<button>Stop</button>` with
 * no state of its own, so the only possible feedback came from a poll two and a
 * half seconds away. Click, nothing, click again — one session logged SEVEN
 * cancel POSTs inside 20 ms.
 *
 * Everything here is deliberately pure and local. The fix cannot depend on the
 * server answering, because the whole complaint is that the server's ANSWER is
 * what is slow; the request itself already lands instantly.
 *
 * The two sentences, though, come FROM the server (`activity.stop_cost` /
 * `stop_wait`) and are relayed verbatim. The price of stopping is not a
 * property of the pass — the front end only knows `kind` — it is a property of
 * the phase, and only the worker knows which phase it is in this second. ✨
 * Score stops three different ways inside one run. Guessing from `kind` here
 * would produce a confident sentence that is wrong two thirds of the time.
 */

/** True once the user has asked this job to stop — by our own click, or
 *  because the server already reports the flag set (a stop asked in another
 *  tab, or before this bar was mounted). */
export function stopRequested(activity, requestedKey) {
  if (!activity) return false
  if (activity.cancelled) return true
  const key = jobKey(activity)
  return Boolean(key && requestedKey && key === requestedKey)
}

/** Identity of the running job, so a click on one pass does not leave the next
 *  pass's button pre-disabled. `started_at` alone is enough in practice, but a
 *  pipeline reuses one job across steps and must keep its requested state. */
export function jobKey(activity) {
  if (!activity) return null
  const started = activity.started_at
  if (started == null) return null
  return `${activity.kind || 'job'}:${started}`
}

export function stopLabel(requested) {
  return requested ? 'Stopping…' : 'Stop'
}

/** The line under the bar. Before the click it is the PRICE; after it, what the
 *  pass is finishing. Empty when the phase promised nothing — a generic
 *  reassurance we cannot back up is worse than silence.
 *
 *  One exception to the silence rule: once the click has landed, a pass with no
 *  phase-specific wait still owes the user the fact that the request WAS taken.
 *  That sentence names no duration, because none is knowable here. */
export function stopNote(activity, requested) {
  if (!activity) return ''
  if (requested) {
    return activity.stop_wait
      || 'Stopping — the request is in; the pass stops at its next safe point.'
  }
  return activity.stop_cost || ''
}
