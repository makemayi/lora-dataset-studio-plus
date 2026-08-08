/* What the flagged pile actually contains — pure, so node --test can hold it to
   its word without mounting the workspace.

   WHY THIS FILE EXISTS. The number printed next to a bulk action must be
   LITERALLY what the click will change. The bank paid for that lesson once
   already (see autoRejectReadiness.js: "🌫 Blurry 5 930" next to a button that
   rejected 0), and the dataset's flagged pile has two ways to repeat it:

   * a small-image RESCUE row makes the whole batch fail — the server refuses the
     entire request with a 400 before writing anything, so ONE flagged rescue
     winner means zero rejections and an error toast;
   * a 'failed' row is skipped server-side (the batch loop `continue`s before it
     counts), so it inflates a promise it never keeps.

   Both are excluded here, and when the two numbers differ the UI says so rather
   than quietly showing the smaller one. */
// Explicit extension: node --test resolves this module for real (no bundler).
import { isSmallImageRescueRow } from '../../utils/smallImageRescue.js';

const isFlagged = (image) => image?.watermark_state === 'detected';

/** True when the flag carries a POSITION — a stored auto box, or hand-drawn
    zones. Only the detector cascade can flag without one (its locator found
    nothing); 🧽 Clean has nothing to route on for those. */
export function hasWatermarkPosition(image) {
  const regions = image?.watermark_regions;
  if (Array.isArray(regions)) return regions.length > 0;
  const bbox = image?.watermark_bbox;
  return Array.isArray(bbox) && bbox.length === 4;
}

/** The flagged images a "Reject all" click would REALLY move to reject. */
export function rejectableFlagged(images) {
  return (images || []).filter(
    (i) => isFlagged(i) && !isSmallImageRescueRow(i) && i.status !== 'failed');
}

/**
 * Everything the curation row needs about the flagged pile, in one pass.
 *   flagged      — 🚩 total (what "Review flagged"/"Clean" show)
 *   rejectable / rejectableIds — what "Reject all flagged" acts on
 *   heldBack     — flagged - rejectable (rescue pairs + failed rows)
 *   unlocated    — flagged with no position at all
 *   dismissed    — ruled false positives (skipped by every later scan)
 *   bySource     — who judged: { detector, vision, unknown }
 */
export function summarizeFlagged(images) {
  const list = images || [];
  const flagged = list.filter(isFlagged);
  const rejectable = rejectableFlagged(list);
  const bySource = { detector: 0, vision: 0, unknown: 0 };
  for (const image of flagged) {
    if (image.watermark_source === 'detector') bySource.detector += 1;
    else if (image.watermark_source === 'vision') bySource.vision += 1;
    else bySource.unknown += 1;
  }
  return {
    flagged: flagged.length,
    rejectable: rejectable.length,
    rejectableIds: rejectable.map((i) => i.id),
    heldBack: flagged.length - rejectable.length,
    unlocated: flagged.filter((i) => !hasWatermarkPosition(i)).length,
    dismissed: list.filter((i) => i.watermark_state === 'dismissed').length,
    bySource,
  };
}

/**
 * The confirmation text. It states the exact number, the way back, and the ONE
 * thing rejecting destroys: the flags themselves (batch_image_action clears
 * watermark_state/bbox/regions on a reject, so "Review flagged" is empty
 * afterwards and nothing records which images had been flagged).
 */
export function rejectFlaggedConfirmText(summary) {
  const { rejectable, heldBack } = summary;
  const lines = [
    `Reject ${rejectable} flagged image(s)?`,
    '',
    'They leave the training set but stay on disk — bring any of them back with '
    + 'Show ▸ Rejected in the grid, then ✓ Keep.',
    '',
    'Rejecting also clears their watermark flags: 🔍 Review flagged will be empty '
    + 'afterwards. Re-run 🧽 Find watermarks to flag them again.',
  ];
  if (heldBack > 0) {
    lines.push('', `${heldBack} more flagged image(s) are NOT included (small-image `
      + 'rescue pairs and failed rows) — settle those in their own review.');
  }
  return lines.join('\n');
}

/**
 * "Who decided this was a watermark?" — the bank's sentence, in the dataset's
 * words. Returns '' when there is nothing to disambiguate (one source, no
 * position-less flag), so the row stays quiet on the ordinary run.
 */
export function flaggedSourceNote(summary) {
  const { bySource, unlocated, flagged } = summary;
  if (!flagged) return '';
  const parts = [];
  if (bySource.detector) parts.push(`${bySource.detector} by the watermark detector`);
  if (bySource.vision) parts.push(`${bySource.vision} by the vision model`);
  if (bySource.unknown) parts.push(`${bySource.unknown} before the source was recorded`);
  const mixed = parts.length > 1;
  if (!mixed && !unlocated) return '';
  const head = mixed ? `Judged ${parts.join(', ')}.` : '';
  const tail = unlocated
    ? `${unlocated} flagged without a position — 🧽 Clean cannot route those; `
      + 'draw a zone in 🔍 Review flagged.'
    : '';
  return [head, tail].filter(Boolean).join(' ');
}
