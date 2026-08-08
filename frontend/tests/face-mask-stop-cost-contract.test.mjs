/* What the face-mask preview's Stop button PROMISES, asserted as wording.

   Asked while watching "Looking for faces… analyzing image 4 of 153": there was
   no way out. The button is the easy half; the hard half is that stopping must
   not quietly cost more than waiting.

   The detector runs in a subprocess that loads antelopev2 before image 1 and
   exits with the pass, so that load is re-paid on every start (measured ~4-5 s
   warm on the reference machine, tens of seconds cold, ~350 MB downloaded on the
   very first run). Which means the cost of stopping is NOT constant — it depends
   on where the pass currently is — and a single "are you sure?" could not tell
   the truth in both cases. So:

   * before any image is analyzed, the only loss is the load, and it says so;
   * once images are analyzed, they are kept, and it says how many;
   * either way the re-load is stated rather than glossed;
   * the start button then ADVERTISES the credit, because a resume the user
     cannot see is a resume they have no reason to trust.
*/
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  previewRunning, previewStartLabel, previewStatusLabel, previewStopCost,
  previewStopLabel, previewStoppedNotice,
} from '../src/utils/faceMaskProgress.js';

const job = (o) => ({
  phase: 'starting', done: 0, total: 0, error: null, finished: false,
  stopping: false, stopped: false, ...o,
});

test('during the model load, Stop says the only thing lost is the load', () => {
  for (const phase of ['starting', 'downloading', 'loading']) {
    const cost = previewStopCost(job({ phase, total: 153 }));
    assert.match(cost, /nothing has been analyzed yet/i,
      `${phase} must say no image is at stake`);
    // The re-load is the real price and must be named even here — this is the
    // case where it is the ONLY price.
    assert.match(cost, /pays it over|detector load/i);
    assert.doesNotMatch(cost, /\bkept\b/,
      `${phase} must not promise kept work when there is none`);
  }
});

test('while images are being analyzed, Stop names how many are kept and that the detector reloads', () => {
  const cost = previewStopCost(job({ phase: 'detecting', done: 47, total: 153 }));
  assert.match(cost, /47 images already analyzed are kept/i);
  assert.match(cost, /re-loads the detector/i);
  assert.match(cost, /carries on from where it stopped/i);
});

test('the count in the promise is the analyzed count, never the total', () => {
  // The failure this guards is the tempting one: showing 153 because that is the
  // number already on screen, and promising to keep work that was never done.
  const cost = previewStopCost(job({ phase: 'detecting', done: 4, total: 153 }));
  assert.match(cost, /\b4 images\b/);
  assert.doesNotMatch(cost, /153/);
});

test('one analyzed image is singular — the promise is read at image 1', () => {
  const cost = previewStopCost(job({ phase: 'detecting', done: 1, total: 153 }));
  assert.match(cost, /1 image already analyzed is kept/i);
});

test('there is no cost line when no pass is running', () => {
  assert.equal(previewStopCost(null), '');
  assert.equal(previewStopCost(job({ finished: true })), '');
  assert.equal(previewStopCost(job({ error: 'boom' })), '');
});

test('a requested stop reads as "stopping", not as already stopped', () => {
  // The child only looks at the request between two images, and during the model
  // load it cannot look at all. Claiming "Stopped" while the counter is visibly
  // still moving is the one lie the user can catch in the act.
  const j = job({ phase: 'detecting', done: 12, total: 153, stopping: true });
  assert.equal(previewStopLabel(j), 'Stopping…');
  assert.match(previewStatusLabel(j), /stopping/i);
  assert.ok(previewRunning(j), 'the pass is still running until the child hands back');
  assert.equal(previewStopLabel(job({ phase: 'detecting' })), 'Stop');
});

test('a stopped pass is reported as stopped, not as done', () => {
  const j = job({ phase: 'detecting', done: 47, total: 153, finished: true, stopped: true });
  assert.equal(previewStatusLabel(j), 'Stopped.');
  assert.ok(!previewRunning(j));
  assert.match(previewStoppedNotice(j, { done: 47, total: 153 }),
    /47 of 153 images are kept/i);
  assert.match(previewStoppedNotice(j, { done: 47, total: 153 }),
    /continues from there/i);
});

test('stopping before image 1 admits there was nothing to keep', () => {
  const j = job({ phase: 'loading', finished: true, stopped: true });
  assert.match(previewStoppedNotice(j, null), /nothing to keep/i);
  // No invented credit: claiming kept work here would be the dishonest version
  // of the same button.
  assert.doesNotMatch(previewStoppedNotice(j, null), /\d+ of \d+/);
});

test('a finished pass shows no stopped notice', () => {
  assert.equal(previewStoppedNotice(job({ finished: true }), { done: 3, total: 3 }), '');
  assert.equal(previewStoppedNotice(null, null), '');
});

test('the start button advertises the resume credit instead of offering a fresh pass', () => {
  assert.equal(previewStartLabel({ done: 47, total: 153 }, false),
    '▶ Resume — 47 of 153 already analyzed');
  // Same offer whether or not an older preview is on screen: what matters is the
  // banked work, not what is drawn.
  assert.equal(previewStartLabel({ done: 47, total: 153 }, true),
    '▶ Resume — 47 of 153 already analyzed');
});

test('with nothing banked the button is the plain one it always was', () => {
  assert.equal(previewStartLabel(null, false), '👁 Preview the mask');
  assert.equal(previewStartLabel(null, true), 'Refresh preview');
  // A zero credit is not a credit — this is the shape the server sends once the
  // kept set moved and the bank was dropped.
  assert.equal(previewStartLabel({ done: 0, total: 153 }, false), '👁 Preview the mask');
});

test('everything already analyzed still needs a start, and says so', () => {
  // The banked set covers the whole run: there is a preview to draw but no
  // detection left to pay for.
  assert.equal(previewStartLabel({ done: 153, total: 153 }, false), '▶ Resume — finishing up');
});
