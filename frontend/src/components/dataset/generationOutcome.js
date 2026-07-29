// What a finished generation run actually produced — counted, not guessed.
//
// WHY THIS FILE EXISTS
// --------------------
// A dataset run can end with images missing and nothing on screen saying so. The
// grid shows one small failed tile per missing image, clamped to a few lines of
// 9px text; twelve refusals in a forty-image run therefore looked like "the app
// generated fewer images than I asked for" and read as a bug in LDS. It was not
// a bug: Gemini's output filter had refused twelve of them, and nobody was told.
//
// So the count is done here, from the rows the workspace already polls, and
// stated once in full sentences. Two rules it must never break:
//
//  * a REFUSAL and a MALFUNCTION are counted apart. They have opposite remedies
//    ("this content will not pass" vs "your key/quota/network is wrong"), and a
//    single number covering both would be wrong for one of them every time.
//  * only what the backend actually classified is counted. Rows written before
//    `fail_kind` existed carry null and are counted as "unclassified" — never
//    folded into either bucket to make a total look tidier.
//
// The backend writes fail_kind (see models.FaceDatasetImage): 'refused' |
// 'empty' | 'error' | null. Those are stored keys — never rename them here.

export const FAIL_REFUSED = 'refused';
export const FAIL_EMPTY = 'empty';
export const FAIL_ERROR = 'error';

/** Count a dataset's image rows by outcome. Pure; tolerates junk input. */
export function summarizeGeneration(images) {
  const counts = {
    total: 0, made: 0, failed: 0, refused: 0, empty: 0, errored: 0, unclassified: 0,
    // Which engines did the refusing — so the notice can name the provider it is
    // describing instead of generalising a policy that belongs to one of them.
    refusedEngines: [],
  };
  if (!Array.isArray(images)) return counts;
  const engines = new Set();
  for (const img of images) {
    if (!img || typeof img !== 'object') continue;
    counts.total += 1;
    if (img.status !== 'failed') {
      if (img.filename) counts.made += 1;
      continue;
    }
    counts.failed += 1;
    if (img.fail_kind === FAIL_REFUSED) {
      counts.refused += 1;
      if (img.engine) engines.add(img.engine);
    } else if (img.fail_kind === FAIL_EMPTY) counts.empty += 1;
    else if (img.fail_kind === FAIL_ERROR) counts.errored += 1;
    else counts.unclassified += 1;
  }
  counts.refusedEngines = [...engines].sort();
  return counts;
}

/**
 * The headline for the refusal notice, or '' when there is nothing to say.
 *
 * Says how many were refused AND how many were made, because "12 refused" alone
 * reads as a broken run when 28 images are sitting right there. Deliberately
 * silent about remedies: the filter is not configurable and not deterministic,
 * so there is no advice that would be true (see backend/app/services/nanobanana.py).
 */
export function refusalHeadline(counts) {
  const c = counts || {};
  const refused = c.refused || 0;
  if (refused < 1) return '';
  const n = `${refused} image${refused === 1 ? '' : 's'}`;
  const made = c.made || 0;
  const kept = made > 0 ? ` ${made} generated normally.` : '';
  return `${n} in this dataset ${refused === 1 ? 'was' : 'were'} refused by the`
    + ` provider's content filter, not lost to an error.${kept}`;
}

/**
 * The one-line note for the malfunction bucket, or '' when there is none.
 * Separate from the headline on purpose — merging them is exactly the mistake
 * this module was written to stop.
 */
export function failureHeadline(counts) {
  const c = counts || {};
  const broken = (c.errored || 0) + (c.empty || 0);
  if (broken < 1) return '';
  const n = `${broken} image${broken === 1 ? '' : 's'}`;
  return `${n} failed for a different reason (connection, key, quota or an`
    + ' unexplained empty answer) — open a failed tile to read its message.';
}
