/* What a ▶ Continue submission does to the dialog it was fired from.
 *
 * The three hosts of ContinueDialog (the dataset Checkpoints panel, the cloud
 * Runs hub, the LoRA Canvas) used to CLOSE the dialog before posting. That was
 * never a design: the toast container shipped at z-[100] under every modal, so
 * an error raised over an open dialog was invisible, and closing first was the
 * only way to be heard. The container is z-[10000] now (Toast.contract.test.js
 * keeps it there) and the workaround has a price of its own — a refusal threw
 * away the lane, the resume checkpoint, the extra steps and the five folded
 * settings, with no hint of WHICH choice was refused.
 *
 * So: a refusal keeps the dialog open and is rendered INSIDE it, next to the
 * inputs that caused it. Success is the only thing that closes it.
 *
 * The three hosts read three different channels — one hook that never throws
 * and returns {ok:false,error}, one apiFetch that throws on 400/409, and a
 * confirm-and-retry helper that reports a user decline. This turns all three
 * into the same two-field answer, so the hosts cannot drift apart on what a
 * refusal means.
 *
 * NOT in scope here: the confirmable refusals (UNCAPTIONED:, MISMATCH_CAPTION:
 * …). Those are a QUESTION, not a refusal — the app already answers them with
 * postWithConfirmations / runConfirmableTrainingRequest, whose window.confirm
 * draws above everything and whose add-a-flag-once rule stops a server that
 * ignores the flag from looping. Only what comes out of those helpers reaches
 * this function. */
/* The generic half of this rule now lives in submitOutcome.js — four more
   dialogs had the same close-before-posting bug, and a second notion of "what
   is a refusal" is how they would drift apart again. What stays HERE is the
   training-specific part: stripping the wire markers. */
// Explicit .js: this module is imported by `node --test` (no bundler resolution).
import { submitAttemptOutcome } from './submitOutcome.js';
import { CONFIRMABLE_REFUSALS, READINESS_REFUSAL } from './trainingRefusals.js';

const MARKERS = [...CONFIRMABLE_REFUSALS, READINESS_REFUSAL].map(([marker]) => marker);

/* The message a human should read. A confirmable refusal that reaches the
   dialog has already been answered once and came back anyway (the server did
   not honour the flag), so its wire marker is noise on screen — strip it, the
   same way the confirm prompt does. */
export function continueRefusalMessage(raw, fallback = 'Continue failed') {
  let s = String(raw ?? '').trim();
  if (!s) return fallback;
  const hit = MARKERS.find((m) => s.includes(m));
  if (hit) s = s.replace(hit, '').trim();
  return s || fallback;
}

/* continueAttemptOutcome({ response, thrown, declined }) -> { close, error }
 *
 *  declined  the user answered "no" at a confirm. Nothing failed and nothing
 *            started: keep the dialog and its inputs, say nothing (the question
 *            they just answered IS the explanation).
 *  thrown    apiFetch rejected (400/409/network) — the Runs hub and the board.
 *  response  the never-throwing hook shape: {ok:false, error, hint?}.
 *
 * `close` is true for success ONLY. A caller that ignores it and closes anyway
 * is back to the bug this exists to remove. */
export function continueAttemptOutcome({ response, thrown, declined } = {}) {
  return submitAttemptOutcome({
    response, thrown, declined, fallback: 'Continue failed', clean: continueRefusalMessage,
  });
}

export default continueAttemptOutcome;
