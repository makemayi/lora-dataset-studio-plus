import { folderSyncNote } from './bankSync'
import { MoveIcon } from '../common/icons'

/** 🗃️ The state of a bank's SOURCE FOLDER as of the last automatic walk.
 *
 * Shown on the bank card and in the workspace header. It only appears when
 * something is off — files listed in the bank that are no longer on disk, or a
 * folder that has gone away entirely. Both cases are reported, never acted on:
 * the bank keeps every row and every decision, because a disconnected drive
 * must not be able to erase a triage.
 *
 * Both cases also share the same usual cause — the folder moved — so when the
 * caller passes ``onRelocate`` the note carries the FIX and not only the
 * diagnosis: the user is already reading this line, which is where the button
 * belongs. */
export default function FolderSyncNote({ sync, onRelocate }) {
  const note = folderSyncNote(sync)
  if (!note) return null
  const tone = note.tone === 'error'
    ? 'border-rose-400/40 bg-rose-50 text-rose-600'
    : 'border-amber-200 bg-amber-50 text-amber-700'
  return (
    <div className={`rounded-md border px-2 py-1 text-xs ${tone}`}>
      <p>{note.tone === 'error' ? '⚠️ ' : 'ℹ️ '}{note.text}</p>
      {note.canRelocate && onRelocate && (
        <button type="button" onClick={onRelocate}
          className="mt-1 inline-flex items-center gap-1.5 rounded-full border border-current px-2 py-0.5 text-xs font-semibold hover:bg-white/10">
          <MoveIcon className="h-3.5 w-3.5 shrink-0" /> Move folder…
        </button>
      )}
    </div>
  )
}
