/* What a modal submission does to the modal it was fired from.
 *
 * ONE rule for every dialog in the app, extracted from continueOutcome.js when
 * the same bug was found in four more places. The shape is always the same:
 * a dialog collects something (a caption, a prompt, seven checkboxes, a folder
 * path), posts it, and used to CLOSE ITSELF FIRST — so a refusal threw the
 * input away and the user re-typed it blind.
 *
 * That order was never a design. The toast container shipped at z-[100], UNDER
 * every modal, so an error raised over an open dialog was invisible and closing
 * first was the only way to be heard. Toast.jsx is z-[10000] now
 * (Toast.contract.test.js keeps it there) and the workaround outlived its reason.
 *
 * So: a refusal keeps the modal open and is rendered INSIDE it, next to the
 * inputs that caused it. Success is the only thing that closes it.
 *
 * The app reads three different channels — useDataset's postJson which never
 * throws and returns {ok:false,error}, fetchClient's apiFetch which throws on
 * 4xx/5xx, and the confirm-and-retry helpers which report a user decline. This
 * turns all three into the same two-field answer so the dialogs cannot drift
 * apart on what a refusal means.
 *
 * NOT in scope: the confirmable refusals (UNCAPTIONED:, MISMATCH_CAPTION: …).
 * Those are a QUESTION, not a refusal — postWithConfirmations /
 * runConfirmableTrainingRequest already answer them, and their add-the-flag-ONCE
 * rule is what stops a server that ignores the flag from looping. Only what
 * comes out of those helpers reaches this function.
 */

/* submitAttemptOutcome({ response, thrown, declined, fallback, clean }) -> { close, error }
 *
 *  declined  the user answered "no" at a confirm. Nothing failed and nothing
 *            started: keep the modal and its inputs, say nothing (the question
 *            they just answered IS the explanation).
 *  thrown    the request rejected (4xx/5xx/network).
 *  response  the never-throwing shape: {ok:false, error, hint?}.
 *  fallback  what to say when the refusal carries no words of its own.
 *  clean     OPTIONAL (raw, fallback) => string, for a caller whose wire format
 *            carries markers a human should not read (see continueOutcome.js).
 *
 * `close` is true for success ONLY. A caller that ignores it and closes anyway
 * is back to the bug this exists to remove. */
export function submitAttemptOutcome({
  response, thrown, declined, fallback = 'Request failed', clean,
} = {}) {
  const say = (raw) => (clean ? clean(raw, fallback) : (String(raw ?? '').trim() || fallback));
  if (declined) return { close: false, error: null };
  if (thrown) return { close: false, error: say(thrown?.message ?? thrown) };
  // No answer at all is NOT a success. A handler that returns undefined has told
  // us nothing, and closing on nothing is exactly how the input got destroyed.
  if (!response) return { close: false, error: `${fallback} — no answer from the server.` };
  if (response.ok === false) {
    const msg = say(response.error);
    return { close: false, error: response.hint ? `${msg} — ${response.hint}` : msg };
  }
  return { close: true, error: null };
}

/* The one-liner every dialog actually calls: run the submission, whatever
   channel it speaks, and get back {close, error}. Having this here rather than
   in each dialog is what keeps a `try` without a `catch` — a network blip that
   escapes as an unhandled rejection, leaving the modal frozen on "Saving…" —
   from being re-invented four times. */
export async function attemptModalSubmit(fn, { fallback, clean } = {}) {
  try {
    return submitAttemptOutcome({ response: await fn(), fallback, clean });
  } catch (thrown) {
    return submitAttemptOutcome({ thrown, fallback, clean });
  }
}

export default submitAttemptOutcome;
