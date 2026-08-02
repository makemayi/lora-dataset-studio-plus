/* Starting a bank pass FROM A BUTTON: whether the button can work right now,
 * what to say when the server refuses, and what the pass actually produced.
 *
 * THE BUG THIS REPLACES. The 🎚 threshold panel offers "↻ Re-group duplicates"
 * next to the duplicate knobs. A bank runs ONE background job at a time, so
 * while any other pass is in flight that button cannot possibly work — yet it
 * looked exactly like a button that could, and the only feedback was the
 * server's own sentence, "a scan job is already running on this bank", pasted
 * into a red toast. That sentence names no progress, no remedy, and no place to
 * go. And when the click DID work, the pass returned 202 and said nothing: a
 * re-group over stored hashes finishes in under the 2 s poll, so the workspace's
 * completion toast (which only fires for a job it observed alive) never fired
 * either. Both halves of the click were silent.
 *
 * THREE FUNCTIONS, ONE RULE EACH:
 *   passButtonState  — a button that cannot work must not look like one, and
 *                      must say WHICH pass is holding the bank and where it is.
 *   busyRefusal      — a 409 is rephrased into blocker + progress + remedy.
 *                      The raw server string is never shown.
 *   passOutcome      — a finished pass reports its NUMBERS, from the counts the
 *                      bank payload already carries.
 *
 * No new progress mechanism lives here. Liveness comes from `progressPresence`
 * (the offline/stale patron already shipped) fed with the SAME `payload.activity`
 * snapshot the Bank's progress bar reads, and the numbers come from the SAME
 * `payload.dup` / `payload.semantic_dup` summaries the duplicate chips count.
 *
 * Plain .js (no JSX) so `node --test` can execute all of it.
 */
import { progressPresence, PROGRESS_RUNNING, PROGRESS_STALE } from './progressPresence.js';

/* Job kind (as `bank_jobs` stores it, and as the 409 body reports it in
   `busy_kind`) → how a human names that pass. Same emoji + words as the button
   that starts it, so "the ✨ Score pass is running" points at something the user
   can see on screen. Unknown kinds fall back to a neutral phrase rather than
   leaking an internal identifier. */
export const JOB_LABELS = {
  scan: '🔎 Quality scan',
  faces: '👥 Face pass',
  score: '✨ Score pass',
  semantic_dedup: '✂ Crops & variants',
  watermark: '🚩 Watermark scan',
  watermark_crop: '🚩 Watermark crop',
  watermark_inpaint: '🚩 Watermark repaint',
  framing: '📐 Framing pass',
  caption: '🏷️ Captioning',
  promote: '⬆ Promotion',
  bank_promote: '⬆ Copy into a new bank',
  import: '📥 Import',
  pipeline: '🚀 Launch all',
  delete_rejected: '🗑 Delete rejected',
};

export function jobLabel(kind) {
  return JOB_LABELS[kind] || 'Another pass';
}

/* The POST path the re-run button uses → the job kind that path produces. The
   two differ (`semantic-dedup` starts a `semantic_dedup` job), and every screen
   that matches one against the other needs the SAME table. */
export const ENDPOINT_JOB_KIND = {
  scan: 'scan',
  faces: 'faces',
  score: 'score',
  'semantic-dedup': 'semantic_dedup',
};

/** "137 / 412", or "137" when the pass could not count its work up front, or
 *  '' when there is nothing numeric to show. */
export function jobProgress(activity) {
  if (!activity) return '';
  const done = Number(activity.done);
  if (!Number.isFinite(done)) return '';
  const total = Number(activity.total);
  return Number.isFinite(total) && total > 0 ? `${done} / ${total}` : `${done}`;
}

/** Is a job holding this bank right now? STALE counts: we lost contact, but the
 *  pass keeps running server-side and starting a second one would still 409. */
export function bankIsBusy(activity, offline = false) {
  const presence = progressPresence(activity, offline);
  return presence === PROGRESS_RUNNING || presence === PROGRESS_STALE;
}

/* Where the Stop button is. Named once so the refusal and the disabled reason
   cannot drift apart, and so renaming the control is one edit. */
const STOP_HINT = 'Wait for it to finish, or press Stop in the progress bar at the top of the bank.';

/** One sentence naming the blocker and its progress, e.g.
 *  "✨ Score pass is running on this bank — 137 / 412". No remedy: callers add
 *  the part that fits their surface. */
export function busyLine({ kind, activity } = {}) {
  const k = kind || activity?.kind;
  const label = jobLabel(k);
  const progress = jobProgress(activity);
  const detail = activity && !activity.finished ? usefulDetail(label, activity.detail) : null;
  let line = `${label} is running on this bank`;
  if (progress) line += ` — ${progress}`;
  if (detail) line += `${progress ? ' · ' : ' — '}${detail}`;
  return line;
}

/* Several passes set a `detail` that is simply their own name ("quality scan"),
   which reads as a stutter once the line already opens with "🔎 Quality scan
   is running". Only a detail that ADDS something is worth the width — and width
   is exactly what a 400 px phone does not have. */
function usefulDetail(label, detail) {
  if (!detail) return null;
  const d = String(detail).trim();
  const stripped = label.replace(/^\P{L}+/u, '').toLowerCase();
  return d.toLowerCase() === stripped ? null : d;
}

/**
 * What to show INSTEAD of a 409's server text. `kind` is the `busy_kind` the
 * route now returns; `activity` is the last snapshot we have, for the numbers.
 * Never returns the server string — that is the whole point.
 */
export function busyRefusal({ kind, activity } = {}) {
  return `${busyLine({ kind, activity })}. ${STOP_HINT}`;
}

/**
 * Whether a re-run button can work, and what it must say if it cannot.
 * @returns {{disabled: boolean, reason: string|null, pending: boolean}}
 */
export function passButtonState({ activity, offline = false, pending = false } = {}) {
  if (pending) {
    return { disabled: true, pending: true, reason: 'Starting…' };
  }
  if (bankIsBusy(activity, offline)) {
    return { disabled: true, pending: false, reason: busyRefusal({ activity }) };
  }
  return { disabled: false, pending: false, reason: null };
}

/* What a pass over this endpoint counts. Both forms are spelled OUT: the plural
   of "group of the same shot" is not that string with an s stuck on the end, and
   a naive pluraliser gets it wrong in exactly the place a user reads first. Only
   the two grouping passes have a summary in the payload; the others report
   through the job's own `detail`, which they already write in plain words. */
const OUTCOME_NOUN = {
  scan: ['duplicate group', 'duplicate groups'],
  'semantic-dedup': ['group of the same shot', 'groups of the same shot'],
};

/** Which payload summary holds this endpoint's result. */
export function summaryKeyFor(endpoint) {
  if (endpoint === 'scan') return 'dup';
  if (endpoint === 'semantic-dedup') return 'semantic_dup';
  return null;
}

function pick([one, many], n) {
  return n === 1 ? one : many;
}

function counted(summary, noun) {
  const g = Number(summary.groups) || 0;
  const i = Number(summary.images) || 0;
  return `${g} ${pick(noun, g)} · ${i} ${pick(['image', 'images'], i)}`;
}

function sameCounts(a, b) {
  return !!a && !!b && Number(a.groups) === Number(b.groups)
    && Number(a.images) === Number(b.images);
}

/**
 * The line a finished pass leaves behind in the panel.
 * @param {string} endpoint  the POST path that was run
 * @param {object|null} before  the summary captured at click time
 * @param {object|null} after   the summary now
 * @param {object|null} activity  the finished job snapshot, when we still have it
 * @returns {{tone: 'ok'|'warn'|'error', text: string}}
 */
export function passOutcome({ endpoint, before, after, activity } = {}) {
  if (activity?.error) {
    return { tone: 'error', text: `Didn't finish — ${activity.error}` };
  }
  if (activity?.cancelled) {
    return { tone: 'warn', text: 'Stopped before it finished — the groups are whatever it had reached.' };
  }
  const noun = OUTCOME_NOUN[endpoint];
  if (noun && after) {
    const now = counted(after, noun);
    if (!before) return { tone: 'ok', text: `Done — ${now}.` };
    if (sameCounts(before, after)) {
      // The honest outcome of a threshold nudge that changed nothing. Silence
      // here would read as "it didn't run".
      return { tone: 'ok', text: `Done — ${now}. Unchanged: your new value groups exactly the same images.` };
    }
    return {
      tone: 'ok',
      text: `Done — ${now} (was ${Number(before.groups) || 0} · ${Number(before.images) || 0}).`,
    };
  }
  if (activity?.detail) {
    const d = String(activity.detail);
    return { tone: 'ok', text: d.charAt(0).toUpperCase() + d.slice(1) };
  }
  return { tone: 'ok', text: 'Done.' };
}

/**
 * Has the pass we launched come back? True when the bank carries no live job,
 * or carries a FINISHED snapshot — either way the counts on screen are final.
 * @param {object|null} activity the bank's current job snapshot
 * @param {string} endpoint      the pass we are waiting on
 */
export function passSettled(activity, endpoint) {
  if (!activity) return true;
  if (!activity.finished) {
    // A live job of a DIFFERENT kind means ours never started (409) or already
    // gave way; either way we are not waiting on it any more.
    return activity.kind !== ENDPOINT_JOB_KIND[endpoint];
  }
  return true;
}

/** The finished snapshot to quote in the outcome, or null when it belongs to
 *  some other pass (or was already purged). */
export function settledActivity(activity, endpoint) {
  if (!activity || !activity.finished) return null;
  return activity.kind === ENDPOINT_JOB_KIND[endpoint] ? activity : null;
}
