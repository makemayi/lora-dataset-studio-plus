/* What the app-wide ComfyUI recovery banner says.
 *
 * The bug this exists for: the recovery barrier is GLOBAL — one stalled prompt
 * blocks every local generation in the app — but its only resolution used to be
 * the Stop button of the dataset that happened to own the job. Working on any
 * other dataset meant an endless "a previous ComfyUI job has an unresolved
 * remote state", with the fix invisible from where you were standing.
 *
 * So the banner's job is to answer, from anywhere: what is stuck, where it
 * lives, since when, and what one click will do about it. Pure functions here,
 * pixels in ComfyRecoveryBanner.jsx.
 */

const MINUTE = 60 * 1000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** "12 minutes" / "3 hours" / "2 days", or null when the server sent no date.
 *  Deliberately coarse: the point is "this is old, from an earlier session",
 *  not a stopwatch. */
export function stalledForText(iso, now = Date.now()) {
  if (!iso) return null;
  const started = Date.parse(iso);
  if (Number.isNaN(started)) return null;
  const elapsed = now - started;
  if (elapsed < 2 * MINUTE) return 'a moment';
  if (elapsed < HOUR) return `${Math.round(elapsed / MINUTE)} minutes`;
  if (elapsed < DAY) {
    const hours = Math.round(elapsed / HOUR);
    return hours === 1 ? '1 hour' : `${hours} hours`;
  }
  const days = Math.round(elapsed / DAY);
  return days === 1 ? '1 day' : `${days} days`;
}

/** Name the stuck job the way its owner would recognise it. */
function jobDescription(recovery) {
  const name = recovery.dataset_name;
  const label = recovery.variation_label;
  if (recovery.run_id && !name) return 'A Test Studio run';
  if (name && label) return `A generation of “${label}” in “${name}”`;
  if (name) return `A generation in “${name}”`;
  if (label) return `A generation of “${label}”`;
  return 'A generation';
}

/**
 * @returns null when nothing is blocking, else the banner's content:
 *   {tone, headline, detail, actionLabel, canConfirm, datasetId, datasetName}
 */
export function recoveryBannerModel(state, { now = Date.now() } = {}) {
  const recovery = state?.recovery;
  if (!recovery) return null;

  if (recovery.kind === 'unreadable') {
    // Nothing here is actionable in one click, and pretending otherwise would
    // send the user clicking a button that can only refuse.
    return {
      tone: 'error',
      headline: 'ComfyUI recovery record is unreadable',
      detail: recovery.detail
        || 'LDS found an invalid ComfyUI recovery record. Restart LDS and check the '
           + 'server log before starting new generations.',
      actionLabel: null,
      canConfirm: false,
      datasetId: null,
      datasetName: null,
    };
  }

  const since = stalledForText(recovery.stalled_since, now);
  const sinceText = since ? ` It has been paused for ${since}.` : '';
  const restarted = recovery.kind === 'unknown_submit'
    // No prompt id exists to check, so this one genuinely needs the user's word.
    ? ' LDS cannot identify the remote job, so it needs you to confirm the restart.'
    : '';
  return {
    tone: 'warning',
    headline: 'A paused ComfyUI job is blocking new generations',
    detail: `${jobDescription(recovery)} stopped without a known outcome.${sinceText}`
      + ' Restart ComfyUI if it is not running, then clear the paused job.'
      + restarted,
    actionLabel: 'I restarted ComfyUI — clear it',
    canConfirm: recovery.can_confirm_restart !== false,
    datasetId: recovery.dataset_id ?? null,
    datasetName: recovery.dataset_name ?? null,
  };
}

/** The one-line toast for a clear that happened on its own — shown once per
 *  notice id, so a 20-second poll doesn't repeat it forever. */
export function autoClearedMessage(state, seenId) {
  const notice = state?.auto_cleared;
  if (!notice?.id || notice.id === seenId) return null;
  return { id: notice.id, message: notice.message };
}
