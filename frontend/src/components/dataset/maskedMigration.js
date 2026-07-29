/* One-time carry-over of the LEGACY `masked` browser preference.
 *
 * `masked` (person masking, background at 10 % loss weight) used to live in this
 * browser's localStorage under `trainMasked_v1`, and reached the server only as a
 * launch parameter. It is now a persisted DATASET setting. People already have a
 * value in that key, and it can contradict the new stored default — so the
 * arbitration matters more than the plumbing:
 *
 *   • The dataset default is UNCHANGED (ON, and forced OFF for concept/style and
 *     slider mode). No dataset silently changes behaviour by upgrading. That rules
 *     out the worst outcome: silently DISABLING masking someone believed active.
 *
 *   • The only population whose effective behaviour changes is the browser that
 *     had explicitly turned masking OFF (`'0'`). Silently adopting that value
 *     would spread one browser's choice onto every dataset in the install;
 *     silently ignoring it would start paying for a rembg pass per image that
 *     nobody asked for. Both are the exact defect being fixed, so we do NEITHER:
 *     we DISCLOSE it once, in the panel, and let the user answer. Nothing is
 *     written until they click.
 *
 *   • A browser whose key AGREES with the new default ('1') has nothing to
 *     disclose — the notice would be pure noise. The key is just cleared.
 *
 * The key is never renamed and never deleted before it is answered, so a
 * downgrade to an older build still reads exactly what it wrote.
 */

export const LEGACY_MASKED_KEY = 'trainMasked_v1';

/** The legacy browser value: true, false, or null when this browser never had one
 *  (or storage is unreadable — private mode, SSR). */
export function readLegacyMasked(storage) {
  try {
    const v = storage?.getItem(LEGACY_MASKED_KEY);
    if (v === null || v === undefined) return null;
    return v !== '0';
  } catch { return null; }
}

/** Forget the legacy key. Called only once the carry-over has been answered (or
 *  is known to be redundant) — never before. */
export function clearLegacyMasked(storage) {
  try { storage?.removeItem(LEGACY_MASKED_KEY); } catch { /* private mode */ }
}

/** What to do about the legacy key for the dataset currently open.
 *
 *  'prompt' — this browser turned masking OFF, the dataset never answered, and the
 *             setting applies here: show the notice and let the user choose.
 *  'clear'  — nothing left to disclose (the key agrees with the default, or this
 *             dataset already carries an explicit answer): drop the key silently.
 *  'none'   — say nothing yet (settings not loaded, no legacy key, or a dataset
 *             where masking is refused by design and the choice cannot apply).
 */
export function maskedCarryOverAction(storage, adv) {
  if (!adv) return 'none';                       // settings not loaded yet
  const legacy = readLegacyMasked(storage);
  if (legacy === null) return 'none';            // this browser never had one
  if (legacy === true) return 'clear';           // agrees with the default
  if (adv.masked_supported === false) return 'none';   // concept/style/slider: ask elsewhere
  if (adv.masked_stored !== null && adv.masked_stored !== undefined) return 'clear';
  return 'prompt';
}
