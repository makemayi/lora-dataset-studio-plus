/* 🖼🗑 Deleting from a checkpoint's gallery — the parts a screenshot cannot show.
 *
 * The picking itself is trivial; what is not is (a) a selection that outlives
 * the images it points at, (b) a confirmation that forgets to say the images
 * also leave the Test Studio, and (c) a toast that claims a success the backend
 * did not report. Each of those is a silent wrong, so each is pinned here.
 *
 * The panel's own guarantees that live in the JSX (deletion unreachable without
 * Select mode, the phone layout) are pinned in lineagePanelsResponsive.test.js,
 * which reads the file as text — node --test cannot parse JSX.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  allGalleryImageIds, galleryActionBar, galleryDeleteConfirmation, galleryDeleteSummary,
  galleryTilePin, pruneGallerySelection, toggleGalleryImage,
} from './gallerySelection.js';

const IMAGES = [{ id: 3 }, { id: 7 }, { id: 11 }];

test('picking toggles without mutating the previous set', () => {
  const a = new Set([3]);
  const b = toggleGalleryImage(a, 7);
  assert.deepEqual([...a], [3]);           // React state is never mutated
  assert.deepEqual([...b].sort(), [3, 7]);
  assert.deepEqual([...toggleGalleryImage(b, 3)], [7]);
});

test('select all / clear all cover exactly what is listed', () => {
  assert.deepEqual([...allGalleryImageIds(IMAGES)].sort((x, y) => x - y), [3, 7, 11]);
  assert.deepEqual([...allGalleryImageIds(null)], []);
  // A row with no id (never happens today, but a payload change is cheap) is not
  // turned into an `undefined` in the delete request.
  assert.deepEqual([...allGalleryImageIds([{ id: 5 }, {}])], [5]);
});

test('a refresh drops ids the gallery no longer lists', () => {
  // Deleted from the Test Studio while the panel was open: the id must not stay
  // armed, or the next 🗑 click deletes something invisible.
  const kept = pruneGallerySelection(new Set([3, 999]), IMAGES);
  assert.deepEqual([...kept], [3]);
  assert.deepEqual([...pruneGallerySelection(new Set([3]), [])], []);
});

test('the confirmation says the Test Studio loses them too', () => {
  const one = galleryDeleteConfirmation(1, 'trash');
  assert.match(one.title, /Delete this image\?/);
  assert.ok(one.lines.some((l) => /Test Studio/.test(l)),
    'the shared-row consequence must be stated before the click');
  const many = galleryDeleteConfirmation(4, 'trash');
  assert.match(many.title, /Delete these 4 images\?/);
});

test('the confirmation names WHICH of the three destinations, before arming', () => {
  const os = galleryDeleteConfirmation(2, 'trash');
  assert.ok(os.lines.some((l) => /Recycle Bin/.test(l)));
  assert.equal(os.destructive, false);

  const app = galleryDeleteConfirmation(2, 'app_trash');
  assert.ok(app.lines.some((l) => /app's Trash/.test(l)));
  assert.equal(app.destructive, false);

  // Unknown / missing mode fails LOUD, not silently reassuring: an answer that
  // never arrived must not read as "recoverable".
  const gone = galleryDeleteConfirmation(2, undefined);
  assert.ok(gone.lines.some((l) => /deleted for good/.test(l)));
  assert.equal(gone.destructive, true);
});

test('the summary reports what the backend did, not what was asked', () => {
  assert.match(galleryDeleteSummary({ rows_removed: 3, mode: 'app_trash', skipped: [] }),
    /3 images deleted · moved to the app's Trash/);
  assert.match(galleryDeleteSummary({ rows_removed: 1, mode: 'trash', skipped: [] }),
    /^1 image deleted/);
  // A locked file keeps its row: the toast has to admit it rather than claim a
  // clean sweep.
  assert.match(
    galleryDeleteSummary({ rows_removed: 1, mode: 'trash', skipped: [{ id: 9 }] }),
    /1 skipped/);
  assert.match(galleryDeleteSummary({ rows_removed: 0, skipped: [{ id: 9 }] }),
    /^Nothing was deleted · 1 skipped$/);
  assert.match(galleryDeleteSummary(null), /Nothing was deleted/);
});

/* 📌 The pinned action bar. `Select` used to sit in the HEADER while everything
 * it leads to sat in a footer bar that only existed once the mode was on — so
 * entering the mode was a reach to the top of a panel whose every other control
 * was under the thumb. The bar is now PERMANENT (as soon as there is something
 * to act on) and simply fills up. Which means two promises now live here rather
 * than in a `{picking && …}` that used to carry them for free:
 *   • an EMPTY gallery still shows no bar at all — no destructive control, and
 *     no dead `Select` on a panel with nothing to select;
 *   • the destructive half only exists inside the mode, and stays inert until
 *     something is picked, so `Select` then `Delete` cannot chain into a delete.
 */
test('the action bar appears only once there is something to act on', () => {
  // Empty, still loading, or failed: no bar, hence no destructive control.
  assert.equal(galleryActionBar({ status: 'ready', imageCount: 0 }).shown, false);
  assert.equal(galleryActionBar({ status: 'loading', imageCount: 12 }).shown, false);
  assert.equal(galleryActionBar({ status: 'error', imageCount: 12 }).shown, false);
  assert.equal(galleryActionBar({ status: 'ready', imageCount: 1 }).shown, true);
  // An empty gallery cannot even offer the gate into deletion.
  assert.equal(galleryActionBar({ status: 'ready', imageCount: 0, picking: true }).showsDelete,
    false);
});

test('the delete half only exists in Select mode, and is inert until a pick', () => {
  const rest = galleryActionBar({ status: 'ready', imageCount: 4, selectedCount: 0 });
  assert.equal(rest.showsDelete, false);           // at rest the bar carries Select alone
  assert.equal(rest.togglePressed, false);
  assert.equal(rest.toggleLabel, 'Select');

  // Entering the mode selects nothing, so the very next tap cannot delete.
  const armed = galleryActionBar({ status: 'ready', imageCount: 4, picking: true, selectedCount: 0 });
  assert.equal(armed.showsDelete, true);
  assert.equal(armed.deleteDisabled, true);
  assert.equal(armed.toggleLabel, 'Done');
  assert.equal(armed.togglePressed, true);

  const picked = galleryActionBar({ status: 'ready', imageCount: 4, picking: true, selectedCount: 1 });
  assert.equal(picked.deleteDisabled, false);
  // A delete already in flight re-disables it — no double submit.
  assert.equal(galleryActionBar({
    status: 'ready', imageCount: 4, picking: true, selectedCount: 1, busy: true,
  }).deleteDisabled, true);
});

/* 📌 Pin to canvas, from the grid.
 *
 * The action shipped inside the VIEWER only: you had to open an image to learn
 * it could leave the modal, and the person who asked for the feature never
 * found it. It now sits on the tile — but a tile is also the target of a batch
 * delete, and that collision is what is worth pinning. */
test('a tile offers 📌 only outside Select mode, and only with a board to pin onto', () => {
  assert.equal(galleryTilePin({ canPin: true }), true);

  // Select mode arms a DELETE. Its safety story is "outside it a tap zooms,
  // inside it a tap picks" — a third target while a delete is being armed is
  // exactly the mis-tap the mode exists to prevent.
  assert.equal(galleryTilePin({ canPin: true, picking: true }), false);

  // No board (the panel also opens from screens that have no canvas): the
  // button would promise something that cannot happen.
  assert.equal(galleryTilePin({ canPin: false }), false);
  assert.equal(galleryTilePin({}), false);
  assert.equal(galleryTilePin(), false);
});

test('select-all flips to Clear exactly when everything listed is picked', () => {
  const at = (selectedCount) => galleryActionBar({
    status: 'ready', imageCount: 3, picking: true, selectedCount,
  }).selectAllLabel;
  assert.equal(at(0), 'Select all');
  assert.equal(at(2), 'Select all');
  assert.equal(at(3), 'Clear');
});
