/* What a running batch is allowed to lock — decided by its SCOPE, not by whether
 * anything is running at all.
 *
 * PURE JS (no JSX) so `node --test` exercises it directly.
 *
 * THE BUG THIS REPLACES
 * ---------------------
 * `bulkBusy` was `busy || …`, where `busy` was true for ANY live activity. So
 * regenerating ONE tile — which registers a `generate` activity — greyed out the
 * buttons on every other tile in the grid. Reported as: 点击了一个图片重新生成
 * 后，其他的都不能点了.
 *
 * That was never the rule the comment claimed. `bulkBusy` exists because "a
 * running pass owns the pixels, the statuses and the files, and a second writer
 * would race it". That is TRUE of a dataset-wide pass — a caption run, a
 * watermark sweep, a face-analysis pass all walk every row and write to it. It
 * is NOT true of a regenerate: that job owns exactly one row, the one already
 * showing its own in-progress state.
 *
 * WHY QUEUEING IS ALREADY SOLVED, AND NOBODY NOTICED
 * --------------------------------------------------
 * A second local regenerate does not need a new task list. `job_queue` is a
 * single worker: enqueue two and the second runs when the first finishes. The
 * `generate` indicator is a COUNT of in-flight rows (`_sync_generate_activity`),
 * so it already reads "3 pending" without any change. The only thing refusing a
 * second click was this flag.
 *
 * And a local job plus an API job do not even queue — different schedulers
 * entirely, see generationLanes.js.
 */

/** Passes that walk the WHOLE dataset and write as they go. A second writer
 *  really would race these, so they keep locking the grid. */
export const DATASET_WIDE_KINDS = Object.freeze([
  'caption',
  'recaption',
  'classify',
  'analyze_faces',
  'watermark_detect',
  'watermark_clean',
]);

/** Passes that fan out over specific ROWS and own only those. Each affected tile
 *  already renders its own in-progress state, which is the correct-scoped lock.
 *
 *  `generate` covers a whole batch AND a single regenerate — the registry cannot
 *  tell them apart, because it tracks a pending COUNT rather than a batch. That
 *  is fine here: either way the rows it owns are exactly the pending ones. */
export const PER_IMAGE_KINDS = Object.freeze(['generate', 'improve']);

/** True when this activity should lock the whole grid. Unknown kinds lock it:
 *  a pass this build has never heard of is assumed to own everything, because
 *  the failure mode of guessing the other way is a corrupted row, while the
 *  failure of guessing this way is a wait. */
export function locksWholeDataset(activity) {
  const kind = activity?.kind;
  if (!kind) return false;
  return !PER_IMAGE_KINDS.includes(kind);
}

/** The dataset-wide activity to blame for a locked grid, or null. Given the
 *  full `activities` list so a per-image batch running alongside a dataset-wide
 *  one does not hide the one actually responsible. */
export function blockingActivity(activities) {
  const list = Array.isArray(activities) ? activities : [];
  return list.find(locksWholeDataset) || null;
}
