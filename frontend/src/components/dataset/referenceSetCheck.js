/** Reading the reference-set self-check report.
 *
 * The check exists because candidate scoring cannot perform it. `sim` is the MAX
 * over the references (best-match-of-N), so a photo of the WRONG person is never
 * outvoted — it wins that max and RAISES every candidate's score instead of
 * lowering it. A wrong reference is silent in the numbers it corrupts, and the
 * only place it shows is against the other references.
 *
 * Kept as pure helpers (no JSX), like `extraRefs.js` next door, so the branches
 * that matter — "flagged" vs "clear" vs "never compared" — are directly testable
 * without a DOM.
 */

/** The row for one tile, or null when the report says nothing about it. */
export function verdictFor(report, slot, filename) {
  const rows = report && report.refs;
  if (!Array.isArray(rows)) return null;
  const match = rows.find((r) => (slot === 'primary'
    ? r.slot === 'primary'
    : r.filename === filename));
  return match || null;
}

/** Was this reference actually compared against the others?
 *
 * Three ways to be un-compared, and all of them must stay distinct from "agrees":
 * no usable face here, no OTHER usable face to compare with, or a report that
 * predates this field. */
export function wasCompared(row) {
  if (!row) return false;
  return typeof row.agreement === 'number';
}

/** Hover text for one reference tile. Never claims agreement it did not measure. */
export function verdictTitle(report, row) {
  if (!row) return '';
  const state = row.state;
  if (state && state !== 'scorable' && state !== 'extreme_pose') {
    return `Not compared — no usable face in this photo (${state}).`;
  }
  if (!wasCompared(row)) return 'Not compared — nothing here to compare it against.';
  const floor = report && report.floor;
  return row.flagged
    ? `Agreement ${row.agreement} with the other references, under the ${floor} floor — this may not be the same person.`
    : `Agreement ${row.agreement} with the other references.`;
}

/** The one-line verdict for the whole set.
 *
 * `compared` comes from the backend rather than being recounted here, because the
 * two can legitimately differ: a row can carry an agreement the caller cannot see
 * (an older report shape), and "how many were compared" is the backend's answer.
 * `lowest` is null when nothing was compared — NOT 0, which would render as a
 * measured floor-scraping score. */
export function summarise(report) {
  const rows = Array.isArray(report && report.refs) ? report.refs : [];
  const agreements = rows.filter(wasCompared).map((r) => r.agreement);
  return {
    compared: (report && report.compared) || 0,
    flaggedCount: rows.filter((r) => r.flagged).length,
    lowest: agreements.length ? Math.min(...agreements) : null,
  };
}
