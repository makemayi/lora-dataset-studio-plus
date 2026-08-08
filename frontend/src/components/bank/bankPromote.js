/* ⬆ Promote — the DECIDABLE part of the two-destination dialog, kept free of
 * JSX so `node --test` can run it.
 *
 * Promoting used to lead exactly one place: a dataset. A dataset is the strict,
 * training-bound container — pulling 200 candidates out of a 9 000-image dump to
 * keep working on them is a different intent, and a dataset commits material the
 * user has not decided on yet. So there are two doors now, and the copy has to
 * say plainly which one is open and what it costs.
 *
 * The cost line is not decoration. Images average ~300 KB, so 200 of them are
 * ~60 MB and nobody needs a warning — but a video bank is three orders of
 * magnitude above that, and the same sentence has to stay true when it is. So
 * the dialog states a MEASURED figure from the server, or says it is still
 * measuring; it never guesses one.
 */

export const PROMOTE_DESTINATIONS = ['dataset', 'bank']

const UNITS = ['B', 'KB', 'MB', 'GB', 'TB']

/** Human weight for a byte count. null/undefined/negative → null, so callers
 *  render "measuring…" instead of a confident "0 B". */
export function formatWeight(bytes) {
  // Number(null) is 0 — an absent measurement must NOT render as "0 B", which
  // reads as "this costs nothing" at the exact moment nothing is known.
  if (typeof bytes !== 'number') return null
  const n = bytes
  if (!Number.isFinite(n) || n < 0) return null
  if (n < 1024) return `${Math.round(n)} B`
  let v = n
  let i = 0
  while (v >= 1024 && i < UNITS.length - 1) { v /= 1024; i += 1 }
  // One decimal below 10 (1.4 GB reads better than 1 GB), none above.
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${UNITS[i]}`
}

/** How many images the promotion will carry, or null while unknown.
 *
 *  A grid selection is authoritative and known client-side. With no selection
 *  the set is "every kept image not already there", which only the server can
 *  count — and for a dataset it is per-target, so it is unknown until one is
 *  picked. */
export function promoteCount({ useSelection, selectedCount, promotable, size }) {
  if (useSelection) return selectedCount
  if (Number.isFinite(promotable)) return promotable
  if (Number.isFinite(size?.count)) return size.count
  return null
}

/** The weight sentence, or null when there is nothing honest to say.
 *
 *  Only for the NEW BANK destination: that copy has no de-duplication step, so
 *  the measured size IS the disk cost. A dataset promotion preserves the bytes
 *  too, but may skip near-duplicates; quoting the whole source weight there
 *  would therefore overstate what is actually written. */
export function weightNotice({ destination, size }) {
  if (destination !== 'bank') return null
  if (!size) return 'Measuring what that weighs on disk…'
  const w = formatWeight(size.bytes)
  if (w == null) return 'Measuring what that weighs on disk…'
  return `That is ${w} of image files copied onto your disk — the two banks never share a file, `
    + 'so each one owns its own copy.'
}

/** The main copy line of the dialog, per destination. */
export function promoteSummary({ destination, useSelection, selectedCount,
  promotable, size, datasetChosen }) {
  const n = promoteCount({ useSelection, selectedCount, promotable, size })
  const many = n == null ? 'The kept image(s)' : `The ${n} image(s)`
  if (destination === 'bank') {
    return `${many} will be COPIED into a brand-new bank with their analysis and `
      + 'keep/pending/reject decisions intact, so you can continue working on them '
      + 'apart. This bank keeps every one of them, marked as promoted.'
  }
  if (!useSelection && !datasetChosen) {
    return 'Kept image(s) not yet in the chosen dataset will be COPIED into it'
      + ' byte-for-byte so their Bank analysis can travel with them; near-duplicates'
      + ' already in the dataset are skipped.'
      + ' The bank and its source folder are left as they are.'
  }
  const scoped = useSelection ? `The ${selectedCount} selected image(s)`
    : (n == null ? 'The kept image(s) not yet in this dataset'
      : `The ${n} kept image(s) not yet in this dataset`)
  return `${scoped} will be COPIED into the dataset byte-for-byte so their Bank`
    + ' analysis can travel with them; near-duplicates already in the dataset are'
    + ' skipped. The bank and its source folder are left as they are.'
}

/** May the confirm button arm? A dataset needs a target, a new bank needs a
 *  name — and neither may fire twice. */
export function canStartPromote({ destination, datasetId, bankName, busy }) {
  if (busy) return false
  if (destination === 'bank') return String(bankName || '').trim().length > 0
  return !!datasetId
}

/** Label of the confirm button — it has to name what it is about to make. */
export function promoteButtonLabel({ destination, busy }) {
  if (busy) return 'Starting…'
  return destination === 'bank' ? 'Create bank' : 'Promote'
}
