/* ▶ Continue must survive a refusal — on all THREE of its hosts.
 *
 * ContinueDialog collects a lane, a resume checkpoint, an extra-steps count and
 * five folded settings. Every host closed it BEFORE posting, so a refusal threw
 * all of that away and the user re-typed it blind, with no way to know WHICH
 * choice was refused. That order was a workaround for the toast container
 * sitting under every modal (fixed: Toast.jsx is z-[10000], Toast.contract.test.js
 * guards it), and it outlived its reason.
 *
 * Contract, per host: the request goes out with the dialog still open; a refusal
 * lands INSIDE it; only a success closes it. The hosts stay different in every
 * other way — the dataset panel keeps its preflight gate and its accumulating
 * confirm-and-retry, the Runs hub keeps postWithConfirmations on the local lane —
 * and this file is what keeps them from diverging on the refusal path.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const dialog = read('./ContinueDialog.jsx');
const panel = read('./TrainingPanel.jsx');
const hub = read('../../pages/CloudRunsPage.jsx');
const canvas = read('../canvas/LineageCanvas.jsx');

/* Comments are stripped before any ordering check: the comment explaining why a
   dialog no longer closes first is exactly where the old call gets written down
   in prose (the Toast contract test learned this the hard way). */
const code = (t) => t.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/(^|[^:])\/\/[^\n]*/g, '$1');

/* The body of one handler, comments removed, minus its cancel line — cancelling
   legitimately closes before anything is awaited. */
function handler(src, start, end) {
  const s = code(src);
  const a = s.indexOf(start);
  assert.ok(a > 0, `handler not found: ${start}`);
  const b = s.indexOf(end, a);
  assert.ok(b > a, `handler end not found: ${end}`);
  return s.slice(a, b).replace(/if \(![^\n]*payload\)[^\n]*\n/, '');
}

function assertPostsBeforeClosing(bodyText, closeCall, host) {
  const firstAwait = bodyText.indexOf('await ');
  const closed = bodyText.indexOf(closeCall);
  assert.ok(firstAwait > 0, `${host}: no request awaited in the submit handler?`);
  assert.ok(closed > firstAwait,
    `${host}: the dialog is still closed (${closeCall}) BEFORE the request is awaited — `
    + 'a refusal would again discard the lane, the checkpoint and the settings the user picked.');
}

test('the dialog can show a refusal inside itself, next to the inputs that caused it', () => {
  assert.match(dialog, /error = null/, 'ContinueDialog must take an optional `error` prop');
  // an assertive live region, so it is announced and not just drawn
  assert.match(dialog, /role="alert"[\s\S]{0,600}\{error\}/,
    'the error must render in a role="alert" region inside the dialog');
  // Two 400-px measurements, both of which made the message unreadable:
  //  • the card is a flex column with max-h-[90vh], so the alert was SQUASHED to
  //    a 20-px sliver of clipped text once "Adjust settings" was unfolded;
  //  • it landed below the fold, because the card scrolls inside itself.
  assert.match(dialog, /className="shrink-0 rounded-lg border border-red-500\/40/);
  assert.match(dialog, /if \(error && card\) card\.scrollTop = card\.scrollHeight;/);
  // …and a long backend refusal scrolls inside its own box instead of pushing
  // ▶ Continue off the screen.
  assert.match(dialog, /max-h-28 overflow-y-auto/);
});

test('a request in flight cannot be dismissed out from under itself', () => {
  // Escape / backdrop / ✕ / Cancel all route through one guarded dismiss: the
  // dataset panel raises its preflight modal ON TOP of this one, and a stray
  // Escape used to cancel both at once.
  assert.match(dialog, /const dismiss = \(\) => \{ if \(!busy\) onResolve\(null\); \}/);
  assert.equal((code(dialog).match(/onResolve\(null\)/g) || []).length, 1,
    'every dismissal path must go through dismiss(), not call onResolve(null) directly');
});

test('all three hosts read the SAME answer out of a submission', () => {
  for (const [host, src] of [['panel', panel], ['Runs hub', hub], ['canvas', canvas]]) {
    assert.match(src, /continueAttemptOutcome/, `${host} must classify the answer with the shared rule`);
    assert.match(src, /error=\{continueError\}/, `${host} must feed the refusal back into the dialog`);
  }
});

test('the dataset panel posts with the dialog open, and keeps its preflight + confirm loop', () => {
  const body = handler(panel, 'const runContinue = async (payload)', 'const askResumeOrFresh');
  assertPostsBeforeClosing(body, 'setContinueOpen(false)', 'dataset panel');
  // what makes this host different must survive untouched
  assert.match(body, /await preflightOk\(\{ lane, trainType: checkpointTrainType/);
  assert.match(body, /runConfirmableTrainingRequest\(/);
  assert.match(body, /confirmableRetryFlag\(error, 'Continue anyway \(force\)'\)/);
  // a preflight that stops the launch must SAY so in the dialog, not just toast
  assert.match(body, /setContinueError\(/);
});

test('the Runs hub posts with the dialog open, and keeps postWithConfirmations on the local lane', () => {
  const body = handler(hub, 'const submitContinue = async (payload)', 'const shareConfig');
  assertPostsBeforeClosing(body, 'setContinueRunTarget(null)', 'Runs hub');
  assert.match(hub, /return postWithConfirmations\(/,
    'the local lane keeps the confirm-and-retry loop — a refusal is not a question');
});

test('the canvas posts with the dialog open, and gains the confirm loop it never had', () => {
  const body = handler(canvas, 'const submitContinue = useCallback', '}, [continueTarget');
  assertPostsBeforeClosing(body, 'setContinueTarget(null)', 'canvas');
  // The board's local lane hits the very same caption/quality guards as the two
  // other hosts; it used to render "UNCAPTIONED: …" as a dead-end error with no
  // way to answer it. Same helper, no second loop.
  assert.match(canvas, /postWithConfirmations/);
});

test('the preflight modal is drawn ABOVE the dialog that opened it, and where it is seen', () => {
  // Both sat at z-[9990]; the dialog portals to document.body and therefore came
  // last in the DOM, so keeping it open would have hidden the preflight report
  // behind it — and the promise it awaits would never resolve ("Starting…"
  // forever). It also has to leave the panel's hideable section, for the same
  // reason ContinueDialog does (ContinueDialogVisibility.test.js).
  const preflight = read('./PreflightModal.jsx');
  assert.match(preflight, /z-\[9992\]/);
  assert.match(panel, /\{preflightReport && createPortal\(\(/);
});
