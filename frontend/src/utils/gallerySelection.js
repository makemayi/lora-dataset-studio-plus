/* 🖼 The checkpoint gallery's selection + delete wording — the DECIDABLE part,
 * kept free of JSX so `node --test` can run it.
 *
 * A checkpoint can hold thirty-odd renders and most of them are misses, so the
 * gallery deletes in BATCHES: pick, then one confirmation for the lot. Two
 * things about that are easy to get wrong and impossible to see in a screenshot,
 * hence this module:
 *
 *  1. selection must survive a refresh that no longer lists an image (something
 *     else deleted it, the limit moved) — a stale id in the set would arm a
 *     delete for something the user cannot see;
 *  2. the confirmation must state BOTH consequences before the button arms: the
 *     images also leave the Test Studio (one row, two surfaces), and WHERE the
 *     files go. Missing evidence downgrades the promise, it never hides it.
 */
import { deleteDestination, isRecoverable } from './deletionWording.js';

/** Add/remove one id. Returns a NEW Set (React state, never mutated in place). */
export function toggleGalleryImage(selected, id) {
  const next = new Set(selected || []);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
}

/** Drop every selected id the gallery no longer lists. Called after each load:
 *  an image deleted from the Test Studio, or pushed past the fetch limit, must
 *  not stay armed for deletion behind the user's back. */
export function pruneGallerySelection(selected, images) {
  const alive = new Set((images || []).map((i) => i?.id));
  const next = new Set();
  for (const id of selected || []) if (alive.has(id)) next.add(id);
  return next;
}

/** Select-all / clear-all over what is currently listed. */
export function allGalleryImageIds(images) {
  return new Set((images || []).map((i) => i?.id).filter((id) => id != null));
}

/** 📌 What the pinned bottom bar carries, for a given panel state.
 *
 *  `Select` used to live in the header while `Select all` and `Delete` lived in
 *  a footer bar that only existed once the mode was ON. Entering the mode was
 *  therefore a reach to the far corner of a panel whose every other control was
 *  under the thumb — on a phone, the most expensive gesture here. So the bar is
 *  PERMANENT (from the moment there is something to act on) and simply fills up
 *  when the mode turns on. Permanent also means it does not appear under the
 *  finger when the mode starts: the gate stays exactly where it was tapped.
 *
 *  Two guarantees that a `{picking && …}` used to carry for free now have to be
 *  decided here, which is why this is a function and not an inline ternary:
 *   • an EMPTY gallery shows no bar at all — no destructive control, and no
 *     dead `Select` offering to pick from nothing;
 *   • the destructive half exists only inside the mode AND stays inert until
 *     something is picked, so the gate and the delete can never chain into one
 *     accidental double-tap despite now sitting on the same row.
 *
 *  Returns {shown, toggleLabel, togglePressed, showsDelete, deleteDisabled,
 *  selectAllLabel}. */
export function galleryActionBar({
  status, picking, imageCount, selectedCount, busy,
} = {}) {
  const listed = Math.max(0, Number(imageCount) || 0);
  const picked = Math.max(0, Number(selectedCount) || 0);
  const shown = status === 'ready' && listed > 0;
  return {
    shown,
    toggleLabel: picking ? 'Done' : 'Select',
    togglePressed: !!picking,
    showsDelete: shown && !!picking,
    deleteDisabled: picked === 0 || !!busy,
    selectAllLabel: listed > 0 && picked === listed ? 'Clear' : 'Select all',
  };
}

/** 📌 Does THIS thumbnail carry a "pin to canvas" affordance?
 *
 *  WHY THIS IS A DECISION AND NOT AN INLINE `&&`
 *  ---------------------------------------------
 *  Pinning an image onto the board shipped as an action inside the VIEWER: you
 *  had to open an image to find out it could leave the modal. The person who
 *  asked for the feature could not find it — and an action nobody finds is an
 *  action that does not exist. The toolbar hint mentioned it, but it is a
 *  seven-clause line that `lg:` hides on anything narrower than a laptop, with
 *  the pin last in it.
 *
 *  So the action moves onto the tile, where the images are. Two rules keep the
 *  tile from becoming a control panel, and both are the kind that a rewrite
 *  quietly loosens — hence here, under test:
 *
 *   • NOT in Select mode. That mode exists to arm a batch delete, and its whole
 *     safety story is "outside it a tap only zooms, inside it a tap only picks".
 *     A third target on the tile while a delete is being armed is exactly the
 *     mis-tap the mode was built to prevent.
 *   • ONLY where a board exists to pin onto. The panel opens from the canvas
 *     (which passes `onPin`) and from screens that have no board at all; there,
 *     the button would promise something that cannot happen.
 *
 *  Returns a boolean. */
export function galleryTilePin({ picking, canPin } = {}) {
  return !!canPin && !picking;
}

/** The confirmation text for a batch — everything the click will do, in order of
 *  surprise. `mode` is the backend's announced destination ('trash' |
 *  'app_trash' | anything else), read from the gallery payload BEFORE the click.
 *
 *  Returns {title, lines: [string], destructive} — `destructive` true when the
 *  files would NOT be recoverable, which is what makes the dialog shout. */
export function galleryDeleteConfirmation(count, mode) {
  const n = Math.max(0, Number(count) || 0);
  const what = n === 1 ? 'this image' : `these ${n} images`;
  const lines = [
    // The consequence that would otherwise be discovered afterwards: these rows
    // ARE the Test Studio's cells — there is only one of each in the database.
    `They are the same images as in the Test Studio grid, so ${
      n === 1 ? 'it disappears' : 'they disappear'} from there too.`,
    isRecoverable(mode)
      ? `The file${n === 1 ? '' : 's'} go to ${deleteDestination(mode)}, so ${
        n === 1 ? 'it' : 'they'} can be restored.`
      : `The file${n === 1 ? '' : 's'} go ${deleteDestination(mode)}.`,
  ];
  return {
    title: `Delete ${what}?`,
    lines,
    destructive: !isRecoverable(mode),
  };
}

/** What actually happened, for the toast. Reads the backend's report rather than
 *  assuming the request did what it asked: a locked or shared file can leave a
 *  row behind, and saying "deleted" then would be a lie. */
export function galleryDeleteSummary(result) {
  const removed = Number(result?.rows_removed) || 0;
  const skipped = (result?.skipped || []).length;
  const head = removed === 1 ? '1 image deleted' : `${removed} images deleted`;
  const parts = [removed ? head : 'Nothing was deleted'];
  if (result?.mode && removed) {
    parts.push(isRecoverable(result.mode)
      ? `moved to ${deleteDestination(result.mode)}`
      : 'removed for good');
  }
  if (skipped) parts.push(`${skipped} skipped`);
  return parts.join(' · ');
}
