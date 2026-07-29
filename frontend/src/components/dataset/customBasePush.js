/* What the cloud dialog's "Custom base" card says, and whether it may offer the
 * one-time push at all.
 *
 * JSX-free on purpose (node --test cannot parse JSX) — and worth extracting for
 * more than testability: two of the three states this decides were WRONG in a
 * way that cost trust rather than pixels.
 *
 *  * a base carried over from another model family was reported as "the local
 *    file is unavailable (missing) — restore it to push", under a push button.
 *    The file was never missing. `train_base_model` is one column shared by
 *    every family, so a Z-Image merge stayed attached to a Krea 2 dataset, and
 *    the readiness probe resolved that Z-Image name as if it were a Krea
 *    absolute path, found nothing, and blamed the disk. The server now answers
 *    `foreign_family`, and this card must say THAT, not offer an upload;
 *  * a genuinely missing local file must never enable the push either — the
 *    upload has nothing to send (the server refuses too; this is the button).
 */

/* Decide the whole card from the readiness payload.
 * Returns {kind, message, warning, canPush, showPush}:
 *   kind    — 'checking' | 'error' | 'ready' | 'no_token' | 'token_invalid'
 *             | 'foreign' | 'pushing' | 'push'
 *   canPush — may the push button be clicked
 *   showPush— should the push button be rendered at all */
export function customBasePushView({ state, checkError, pushing } = {}) {
  if (checkError) {
    return { kind: 'error', message: checkError, canPush: false, showPush: false };
  }
  if (!state) {
    return { kind: 'checking', message: 'Checking your custom base on Hugging Face…',
      canPush: false, showPush: false };
  }
  if (state.ready) {
    return { kind: 'ready', message: null, canPush: false, showPush: false };
  }
  // The family mismatch outranks every other reason: there is no token to add,
  // no file to restore and nothing to upload — the selection itself is wrong.
  if (state.reason === 'foreign_family') {
    return {
      kind: 'foreign',
      message: state.foreign_base_message
        || 'This base was chosen for another model family — pick a base for this '
           + 'family, or train on the official one.',
      canPush: false,
      showPush: false,
    };
  }
  if (state.reason === 'no_token' || state.reason === 'token_invalid') {
    return { kind: state.reason, message: null, canPush: false, showPush: false };
  }
  if (pushing) {
    return { kind: 'pushing', message: null, canPush: false, showPush: false };
  }
  const message = state.reason === 'size_mismatch'
    ? 'Your local custom base changed since it was pushed — push it again to update the private copy.'
    : state.reason === 'file_missing'
      ? 'The private repo exists but is missing the file this variant needs — push again to add it.'
      : 'This run uses custom weights the pod cannot download yet.';
  return {
    kind: 'push',
    message,
    // The one-time push reads the local file. Absent, there is nothing to send:
    // the button is offered (so the requirement is visible) but never clickable.
    warning: state.local_available ? null
      : `The local file is unavailable (${state.local_reason || 'missing'}) — restore it to push.`,
    canPush: !!state.local_available,
    showPush: true,
  };
}
