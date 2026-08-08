/** 🎬 Playing one detected shot — pure helpers (no JSX, so `node --test` runs them).
 *
 * THE GRID CONTAINS NO <video> AT ALL. That is the whole design, and it is worth
 * stating because the obvious alternative is everywhere: a grid of tiny previews
 * with `preload="none"`, mounted on hover.
 *
 * Two things rule it out. First, the video bank writes no clip files — cutting a
 * shot means re-encoding it, and we only pay that at promotion, for the shots
 * actually kept. So there is no small file to hover; a preview would have to seek
 * into a multi-gigabyte rush. Second, Chrome caps WebMediaPlayers at about 60 in
 * total, which leaves roughly 40 usable <video> elements on a page — and past
 * that, new elements simply never load, with no error of any kind. A bank holds
 * hundreds of shots.
 *
 * So: JPEG thumbnails in the grid, and playback in a lightbox holding exactly ONE
 * <video>, pointed at the SOURCE with a media fragment. The browser fetches only
 * the requested range, and the player ceiling stops being something to work
 * around.
 */

/** How many <video> elements may be mounted at once. One, by construction. */
export const MAX_MOUNTED_PLAYERS = 1

/** `<base>#t=start,end`, or null when the range is not playable.
 *
 * Returning null matters more than it looks: a malformed fragment does not throw
 * and does not warn — the browser ignores it and plays the WHOLE file. On a
 * two-hour rush that is the worst outcome available, so an inverted range is
 * refused here rather than handed to the element.
 */
export function clipFragmentSrc(base, startS, endS) {
  const start = Math.max(0, Number(startS))
  const end = Number(endS)
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null
  // Strip any fragment already on the base: re-deriving a src from an
  // already-fragmented one would produce "#t=1,2#t=3,4", which parses as neither.
  const clean = String(base).split('#')[0]
  return `${clean}#t=${trimNumber(start)},${trimNumber(end)}`
}

/** Trailing-zero-free decimal — sub-second precision is kept (rounding a bound to
 * the nearest second moves a quarter of a two-second clip) but 5.0 reads as 5. */
function trimNumber(value) {
  return String(Number(value.toFixed(3)))
}

/** Non-null when something has mounted more players than the design allows — a
 * guard against a future grid quietly reintroducing inline players. */
export function playerBudgetWarning(mounted) {
  if (mounted <= MAX_MOUNTED_PLAYERS) return null
  return `${mounted} video players are mounted; this view is designed to hold one.`
}

/** Does the player element have to be recreated?
 *
 * Yes whenever the clip changes. Assigning a new `#t=` to a LIVE <video> is not
 * reliable across browsers — several ignore the fragment once the resource is
 * loaded, and the viewer silently watches the previous clip's range while the
 * caption says otherwise. Remounting is the only portable way to re-seek.
 *
 * No when nothing changed: remounting on every render restarts playback from the
 * head, which makes the lightbox unusable.
 */
export function shouldRemountPlayer(current, next) {
  if (!current || !next) return current !== next
  return current.sourceId !== next.sourceId || current.start !== next.start
}

/** "0:41 – 0:46 (5.1s)" — where the shot sits in its source, and how long it is.
 *
 * Position first, because that is what lets someone find the moment in the
 * original file; the id of a row means nothing to them. */
export function clipLabel(startS, endS) {
  return `${timecode(startS)} – ${timecode(endS)} (${(endS - startS).toFixed(1)}s)`
}

/** m:ss, or h:mm:ss past the hour — rushes are routinely longer than an hour. */
function timecode(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0))
  const s = String(total % 60).padStart(2, '0')
  const m = Math.floor(total / 60) % 60
  const h = Math.floor(total / 3600)
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`
}
