/** 🗃️ ↩ Undo the last bulk decision — pure helpers (no JSX, so `node --test`
 * can run them).
 *
 * Marking hundreds of images in one gesture is the bank's biggest lever. The
 * backend now snapshots the (status, reason) of every row a bulk action flips
 * and offers ONE step back, per bank, in the payload the workspace already
 * polls (`payload.undo` = {label, count, at}) — which is why the bar survives a
 * reload: the decision it takes back lives in the database, not in this tab.
 *
 * These helpers own the WORDING, and the wording is where the honesty lives.
 * Two rules run through the whole file:
 *   1. never promise what the server did not do — a restore that put 340 of 400
 *      rows back says so, and names what it left alone;
 *   2. never offer an undo for an action that cannot be undone cleanly. The
 *      server simply publishes no offer for 🗑 Delete rejected or ⬆ Promote,
 *      so there is nothing here to accidentally render.
 */

/** The offer to show, or null. Defensive about the payload shape: a count of 0
 * is "nothing was actually flipped", which must not draw a bar. */
export function undoOffer(payload) {
  const undo = payload?.undo
  if (!undo) return null
  const count = Number(undo.count) || 0
  if (count <= 0) return null
  const label = String(undo.label || 'Last bulk action')
  return { label, count, at: Number(undo.at) || 0 }
}

/** The bar's sentence. Kept to one line at 400 px: the action, how many images
 * it moved, and the boundary of the promise (this session only). */
export function undoBannerText(offer) {
  if (!offer) return ''
  return `${offer.label} — ${offer.count} image${offer.count === 1 ? '' : 's'}`
}

/** The tooltip/second line: what pressing it will do, and what it will not.
 * Users deserve the restart caveat BEFORE they rely on the net, not after. */
export const UNDO_HINT = 'Puts those images back exactly as they were. '
  + 'One step only, and only until the app restarts.'

/** The honest ledger after a restore. Never a bare "done": every image that did
 * not make it back is counted, and the ones a newer decision now owns are named
 * so the user can go and look at them. */
export function undoResultMessage(result) {
  if (!result) return { type: 'error', text: 'Undo failed.' }
  const restored = Number(result.restored) || 0
  const missing = Number(result.missing) || 0
  const conflicts = Number(result.conflicts) || 0
  const total = Number(result.total) || restored + missing + conflicts
  const names = Array.isArray(result.conflict_names) ? result.conflict_names : []

  if (restored === total && !missing && !conflicts) {
    return {
      type: 'success',
      text: `↩ Restored ${restored} image${restored === 1 ? '' : 's'} to what they were.`,
    }
  }
  const parts = [`↩ Restored ${restored} of ${total} images.`]
  if (missing) {
    parts.push(`${missing} ${missing === 1 ? 'is' : 'are'} no longer in the bank.`)
  }
  if (conflicts) {
    const shown = names.slice(0, 3).join(', ')
    parts.push(`${conflicts} changed since and ${conflicts === 1 ? 'was' : 'were'} left alone`
      + (shown ? ` (${shown}${conflicts > names.slice(0, 3).length ? ', …' : ''}).` : '.'))
  }
  return { type: restored ? 'info' : 'error', text: parts.join(' ') }
}
