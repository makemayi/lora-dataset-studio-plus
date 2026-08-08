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

/** The concrete places a ComfyUI address goes wrong. Ordered by how often they
 *  are the answer, and each one names what to look at rather than what to feel:
 *  a fresh install with an auto-detected URL has no way to know which of these
 *  it is until someone lists them. */
const CONNECTION_CHECKS = [
  'Open the ComfyUI window you actually use and copy the address from its browser'
  + ' bar — LDS must point at that exact host and port.',
  'ComfyUI started without --listen only answers on its own machine. If it runs on'
  + ' another PC, start it with --listen and put that machine’s address here.',
  'LDS in Docker with ComfyUI on the host: use http://host.docker.internal:8188 —'
  + ' inside a container, 127.0.0.1 is the container itself.',
  'The address lives in Settings ▸ Local tools ▸ ComfyUI API URL.',
];

/** The banner when LDS cannot talk to ComfyUI at all, or null.
 *
 * This is the half that was missing. A barrier says a job is paused; it says
 * nothing about whether the two programs are in touch. A fresh install was told
 * "a paused ComfyUI job is blocking new generations" on its very first Generate,
 * while its ComfyUI logged no incoming connection at all — so the user went
 * hunting for a flag he had to pass, which is the one thing that could not have
 * helped (jerkyjunky, Discord). When the server reports the link as down, the
 * connection is the headline and the paused job is a footnote.
 */
function connectionFirstModel(recovery) {
  const connection = recovery.connection;
  if (!connection || connection.reachable !== false) return null;
  const url = connection.url || '';
  const unconfigured = connection.status === 'unconfigured' || !url;
  // An unconfirmed submission is the fresh-install shape: LDS asked ComfyUI to
  // start a job and never heard back, so it holds that job rather than risk
  // running it twice. Saying "a paused job is blocking you" first, to someone
  // who has never generated anything, is what made this unreadable.
  const footnote = recovery.kind === 'unknown_submit'
    ? 'A generation LDS could not confirm ComfyUI ever accepted is on hold behind'
      + ' this — that silence is the same problem, not a second one. Clear it with'
      + ' the button below once ComfyUI answers.'
    : 'A paused generation is waiting behind this. LDS clears it by itself as soon'
      + ' as ComfyUI answers again and no longer knows the job.';
  return {
    tone: 'error',
    headline: unconfigured
      ? 'LDS has no ComfyUI address to reach'
      : `LDS cannot reach ComfyUI at ${url}`,
    detail: connection.hint
      || (unconfigured
        ? 'Set the ComfyUI API URL before generating anything — nothing local can run'
          + ' until LDS knows where ComfyUI is.'
        : 'Fix the connection first: nothing local can generate until LDS can talk to'
          + ' ComfyUI, and LDS got no answer from that address.'),
    checks: CONNECTION_CHECKS,
    footnote,
    actionLabel: 'I restarted ComfyUI — clear it',
    canConfirm: recovery.can_confirm_restart !== false,
    datasetId: recovery.dataset_id ?? null,
    datasetName: recovery.dataset_name ?? null,
  };
}

/**
 * @returns null when nothing is blocking, else the banner's content:
 *   {tone, headline, detail, checks, footnote, actionLabel, canConfirm,
 *    datasetId, datasetName}
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
      checks: [],
      footnote: null,
      actionLabel: null,
      canConfirm: false,
      datasetId: null,
      datasetName: null,
    };
  }

  // Then, before naming the paused job: is LDS even talking to ComfyUI? If it
  // is not, every other sentence here is a red herring. (An unreadable record
  // above stays first — it survives ComfyUI coming back, so it must not be
  // hidden behind a connection the user is about to fix.)
  const unreachable = connectionFirstModel(recovery);
  if (unreachable) return unreachable;

  const since = stalledForText(recovery.stalled_since, now);
  const sinceText = since ? ` It has been paused for ${since}.` : '';
  // The two kinds need genuinely different sentences. A known prompt id is
  // checkable, so restarting ComfyUI is the whole fix and LDS finishes the job
  // on its own; an unknown submission has no id to check, so it will still be
  // waiting for a person once ComfyUI is back.
  const next = recovery.kind === 'unknown_submit'
    ? ' Restart ComfyUI if it is not running. LDS cannot identify the remote job,'
      + ' so it needs you to confirm the restart before clearing it.'
    : ' Restart ComfyUI if it is not running — LDS clears this by itself once'
      + ' ComfyUI answers and no longer knows the job.';
  return {
    tone: 'warning',
    headline: 'A paused ComfyUI job is blocking new generations',
    detail: `${jobDescription(recovery)} stopped without a known outcome.${sinceText}${next}`,
    checks: [],
    footnote: null,
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
