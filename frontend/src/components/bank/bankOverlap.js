/* 🗃 Banks that share files — the DECIDABLE part, kept free of JSX so
 * `node --test` can run it.
 *
 * Nothing stops two banks pointing at nested folders, and while you are only
 * triaging that is harmless: statuses live on the bank, not on the file. It
 * stops being harmless at 🗑 Delete rejected, the one action that removes the
 * FILES — the other bank simply finds them gone. These helpers turn the backend
 * facts into the sentences that have to be said BEFORE that click.
 */

/* Where the files end up is not a bank question — every destructive
 * confirmation in the app says it, so the sentence lives in utils and is
 * re-exported here for the callers (and tests) that already address it. */
export { deleteDestination, isRecoverable } from '../../utils/deletionWording.js'

/** The one-line notice for a bank created over a folder another bank already
 *  covers — or null when there is nothing to say. */
export function overlapNotice(overlaps) {
  const list = (overlaps || []).filter((o) => o && o.name)
  if (!list.length) return null
  const names = list.map((o) => `“${o.name}”`).join(', ')
  return `Heads up: this folder overlaps ${names}. Both banks list the same files, `
    + 'so 🗑 Delete rejected in one removes them from the other too.'
}

/** A bank whose source folder IS a dataset's storage folder — the one overlap
 *  that is never negotiable.
 *
 *  Two banks over one folder is a WARNING: the user may well mean it, and the
 *  worst case is another bank's triage. A bank over a DATASET is different in
 *  kind — the "rejected" files are the dataset's training images, and the app
 *  copies in both directions precisely so the two never share. The server
 *  refuses the click; this turns that refusal into a sentence, and into a
 *  disabled button, so nobody types DELETE to be told no.
 *
 *  Installs made before the guard existed still hold such banks, which is why
 *  this reads a fact off the payload instead of trusting creation-time checks.
 *  Nothing is ever deleted or repaired on the app's own initiative — the bank
 *  stays fully readable, and only the user decides to move or drop it.
 *
 *  Returns {blocked, title, text}; blocked=false when there is no conflict. */
export function datasetConflictBlock(conflict) {
  if (!conflict) return { blocked: false, title: '', text: '' }
  return {
    blocked: true,
    title: '⛔ This bank sits on a dataset’s image folder',
    text: conflict.message
      || 'Deleting these files would delete images out of a dataset. A bank and a '
      + 'dataset must never share files — open the dataset and use 🗃 Import to bank, '
      + 'which copies them into a bank of their own.',
  }
}

/** The blocking warning inside the delete confirmation: which OTHER banks lose
 *  files, and how many. null when this bank is on its own. */
export function sharedFilesWarning(preview) {
  const shared = (preview?.shared || []).filter((s) => s && s.files > 0)
  if (!shared.length) return null
  const parts = shared.map((s) => `${s.files} of them are also in “${s.name}”`)
  return `${parts.join('; ')}. Deleting them here removes them from that bank too `
    + '— its own decisions on those images are lost with the files.'
}

/** Total files another bank would lose, across every overlapping bank. Used to
 *  decide how loud the confirmation has to be. */
export function sharedFileCount(preview) {
  return (preview?.shared || []).reduce((n, s) => n + (Number(s?.files) || 0), 0)
}

/** May the delete confirmation arm — and what to say while it may not.
 *
 *  A destructive control has to fail CLOSED when its evidence is missing. Every
 *  warning in this dialog is MADE of the preview: which other bank loses files,
 *  and whether they land somewhere recoverable. With no preview, all of that
 *  simply does not render — the ⚠ banner vanishes, the destination silently
 *  falls back to "deleted for good", and the button arms anyway. That is the
 *  protection disappearing at the exact moment it could not do its job, so the
 *  button stays disabled until the answer is in.
 *
 *  Returns {ready, state: 'checking'|'failed'|'ready', title, text}. */
export function deletePreviewState(preview) {
  if (preview?.failed) {
    return {
      ready: false,
      state: 'failed',
      title: '⚠ Could not check what this delete would do',
      text: 'The app could not ask where these files would go, nor whether another '
        + 'bank shares them. Nothing is deleted while that is unknown — close this, '
        + 'make sure the app is still running, and open it again.',
    }
  }
  if (!preview) {
    return {
      ready: false,
      state: 'checking',
      title: 'Checking what this delete would do…',
      text: 'Asking where the files would go, and whether another bank shares them.',
    }
  }
  return { ready: true, state: 'ready', title: '', text: '' }
}
