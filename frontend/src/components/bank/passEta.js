/* "about 2 hours left" — turning the server's remaining-seconds into a sentence.
 *
 * The measurement lives in `backend/app/services/job_eta.py`, which publishes
 * three fields on every job snapshot: `eta_state` ('none' | 'estimating' |
 * 'ready'), `eta_seconds`, and `eta_scope` ('job' | 'phase'). Everything here is
 * the DISPLAY half of the same contract, and it has exactly two jobs.
 *
 * ROUND HONESTLY. "1 h 53 min left" claims a precision the estimate does not
 * have, and a user who sees that number miss by nine minutes concludes the
 * feature is broken — where "about 2 hours" would have been right. So the
 * buckets get coarser as the number gets bigger: minutes to the nearest minute
 * under ten, to the nearest five under forty-five, half hours after that, whole
 * hours past four, and "more than a day" beyond a day, because past that point
 * there is no decision the exact figure would change.
 *
 * SAY NOTHING RATHER THAN GUESS. `eta_state: 'none'` is a pass with nothing
 * countable in front of it — the style grouping that runs for three minutes on
 * done=0/total=0, a phase that never declared a total. It renders as empty
 * string, the same rule `jobProgress` already applies to the bare "0".
 * 'estimating' is the honest in-between: we know there is work left, we do not
 * yet trust our own number, and we say so instead of printing one that will
 * change.
 *
 * Plain .js (no JSX) so `node --test` can execute all of it.
 */

/** Seconds → a coarse English duration, or null when there is nothing to say. */
export function formatEtaSeconds(seconds) {
  // `Number(null)` is 0, and 0 renders as "under a minute" — which would put a
  // duration on every pass that publishes none.
  if (seconds === null || seconds === undefined || seconds === '') return null;
  const s = Number(seconds);
  if (!Number.isFinite(s) || s < 0) return null;
  if (s < 45) return 'under a minute';
  if (s < 90) return 'about a minute';
  const minutes = s / 60;
  if (minutes < 10) return `about ${Math.max(2, Math.round(minutes))} minutes`;
  if (minutes < 45) return `about ${Math.round(minutes / 5) * 5} minutes`;
  const hours = s / 3600;
  if (hours < 4) {
    // Half-hour buckets are as fine as this deserves to get. Below four hours
    // the half still carries information ("2 hours 30" is a different plan from
    // "2 hours"); above it, it is noise.
    const half = Math.round(hours * 2) / 2;
    if (half === 1) return 'about an hour';
    const whole = Math.floor(half);
    const unit = whole === 1 ? 'hour' : 'hours';
    return half === whole
      ? `about ${whole} ${unit}`
      : `about ${whole} ${unit} 30 minutes`;
  }
  if (hours < 24) return `about ${Math.round(hours)} hours`;
  return 'more than a day';
}

/**
 * The remaining-time clause for a live job snapshot, or '' when there is none.
 *
 * "in this step" is not decoration. ✨ Score runs child inference, then writes
 * ~21 000 rows, then groups styles; each of those has its own speed, and the
 * estimator only ever measures the one it is inside. Once the server has SEEN a
 * phase change it says so through `eta_scope`, and the sentence stops implying
 * it covers the whole pass.
 */
export function etaPhrase(activity) {
  if (!activity || activity.finished || activity.error || activity.cancelled) return '';
  const state = activity.eta_state;
  if (state === 'ready') {
    const text = formatEtaSeconds(activity.eta_seconds);
    if (!text) return '';
    return activity.eta_scope === 'phase' ? `${text} left in this step` : `${text} left`;
  }
  if (state === 'estimating') return 'estimating time left…';
  return '';
}
