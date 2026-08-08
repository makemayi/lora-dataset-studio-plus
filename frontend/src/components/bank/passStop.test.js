/* The Bank Stop button's three states, as data.
 *
 * Written from a measured session: `POST /cancel` answered in 79 ms while the
 * banner's own payload took 2 745 ms, and the log shows SEVEN cancel POSTs
 * inside 20 ms. Nothing was broken — the button simply looked identical after
 * the click, so the only rational move was to click again.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import { jobKey, stopLabel, stopNote, stopRequested } from './passStop.js';

const running = (o = {}) => ({
  kind: 'score', done: 2200, total: 36925, finished: false, cancelled: false,
  started_at: 1000, detail: 'writing 36925 score(s) to the database…',
  stop_cost: 'Scores already written stay.',
  stop_wait: 'Stopping — finishing the current batch of 200 rows, then saving.',
  ...o,
});

test('the click alone flips the button, with no server round trip', () => {
  const job = running();
  assert.equal(stopRequested(job, null), false);
  assert.equal(stopLabel(false), 'Stop');
  // What the click stores is the job identity, and that is all it takes.
  assert.equal(stopRequested(job, jobKey(job)), true);
  assert.equal(stopLabel(true), 'Stopping…');
});

test('a stop asked elsewhere still disarms this button at the next poll', () => {
  // `cancelled` is set in memory by the cancel route, so it arrives whatever
  // tab (or preflight dialog) asked for it. Re-showing an armed button here is
  // how the seventh POST gets sent.
  assert.equal(stopRequested(running({ cancelled: true }), null), true);
});

test('the request does not carry over to the NEXT pass', () => {
  // Same bank, same `kind`, new run: the button must come back alive without an
  // effect, a cleanup or a parent reset.
  const first = running();
  const second = running({ started_at: 2000 });
  assert.equal(stopRequested(second, jobKey(first)), false);
  assert.equal(stopRequested(first, jobKey(first)), true);
});

test('a pipeline keeps its requested state across its steps', () => {
  // 🚀 Launch all runs every step under ONE job, so `kind`/`started_at` hold
  // still and a stop asked at step 2 must not re-arm at step 3.
  const step2 = { kind: 'pipeline', started_at: 500, pipeline: { index: 1 } };
  const step3 = { kind: 'pipeline', started_at: 500, pipeline: { index: 2 } };
  assert.equal(stopRequested(step3, jobKey(step2)), true);
});

test('before the click the line is the PRICE, verbatim from the phase', () => {
  assert.equal(stopNote(running(), false), 'Scores already written stay.');
});

test('after the click the line is what the pass is finishing', () => {
  assert.equal(
    stopNote(running(), true),
    'Stopping — finishing the current batch of 200 rows, then saving.');
});

test('a phase that promises nothing says nothing before the click', () => {
  // Requirement: the sentence VARIES per phase or it is absent. A generic
  // "some work may be lost" would be a guess dressed as information — and the
  // front end cannot do better, it only knows `kind`.
  assert.equal(stopNote(running({ stop_cost: null, stop_wait: null }), false), '');
  assert.equal(stopNote({ kind: 'caption', started_at: 1 }, false), '');
});

test('after the click, even a silent phase confirms the request was taken', () => {
  const seen = stopNote(running({ stop_wait: null }), true);
  assert.match(seen, /stopping/i);
  // No invented duration: "in a few seconds" is exactly the promise this layer
  // has no way to keep.
  assert.doesNotMatch(seen, /\bsecond|\bminute|\bsoon\b/i);
});

test('no job means no note and no key', () => {
  assert.equal(stopNote(null, true), '');
  assert.equal(stopNote(null, false), '');
  assert.equal(jobKey(null), null);
  assert.equal(stopRequested(null, 'score:1'), false);
});

test('a job with no started_at cannot be keyed, and stays clickable', () => {
  // Legacy snapshots (and a few hand-built test mappings) have no timestamp.
  // Failing OPEN here is right: a button that will not disarm is recoverable,
  // one that is stuck disabled over a running pass is not.
  const legacy = { kind: 'score', done: 1, total: 2 };
  assert.equal(jobKey(legacy), null);
  assert.equal(stopRequested(legacy, null), false);
});
