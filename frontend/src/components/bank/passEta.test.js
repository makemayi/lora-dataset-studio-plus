import test from 'node:test';
import assert from 'node:assert/strict';
import { formatEtaSeconds, etaPhrase } from './passEta.js';
import { busyLine } from './bankPassRun.js';

test('durations are rounded to buckets a user can act on', () => {
  assert.equal(formatEtaSeconds(3), 'under a minute');
  assert.equal(formatEtaSeconds(44), 'under a minute');
  assert.equal(formatEtaSeconds(80), 'about a minute');
  assert.equal(formatEtaSeconds(4 * 60), 'about 4 minutes');
  assert.equal(formatEtaSeconds(23 * 60), 'about 25 minutes');
  assert.equal(formatEtaSeconds(60 * 60), 'about an hour');
  assert.equal(formatEtaSeconds(2 * 3600), 'about 2 hours');
  assert.equal(formatEtaSeconds(3.9 * 3600), 'about 4 hours');
  assert.equal(formatEtaSeconds(9 * 3600), 'about 9 hours');
  assert.equal(formatEtaSeconds(40 * 3600), 'more than a day');
});

test('a measured 1 h 53 is announced as about 2 hours, not as 1 h 53', () => {
  // False precision on an estimate is a lie about the estimate's quality: the
  // user who watches "1 h 53" miss by nine minutes concludes the number is
  // broken, where "about 2 hours" would have been right all along.
  const text = formatEtaSeconds(113 * 60);
  assert.equal(text, 'about 2 hours');
  assert.ok(!/\d\d? *(min|h) *\d/.test(text));
});

test('the half-hour bucket keeps its singular', () => {
  assert.equal(formatEtaSeconds(95 * 60), 'about 1 hour 30 minutes');
  assert.equal(formatEtaSeconds(150 * 60), 'about 2 hours 30 minutes');
});

test('a garbage or missing duration produces nothing at all', () => {
  assert.equal(formatEtaSeconds(null), null);
  assert.equal(formatEtaSeconds(undefined), null);
  assert.equal(formatEtaSeconds(-5), null);
  assert.equal(formatEtaSeconds('soon'), null);
});

test('a ready estimate reads as a sentence', () => {
  assert.equal(
    etaPhrase({ eta_state: 'ready', eta_seconds: 7200, eta_scope: 'job' }),
    'about 2 hours left');
});

test('once the pass has changed phase the sentence says which step it means', () => {
  // ✨ Score is inference, then ~21 000 row writes, then style grouping. An
  // unqualified "about 20 minutes left" over the second of those would be a
  // claim about the whole pass that nothing measured.
  assert.equal(
    etaPhrase({ eta_state: 'ready', eta_seconds: 1200, eta_scope: 'phase' }),
    'about 20 minutes left in this step');
});

test('an unsettled estimate promises nothing', () => {
  assert.equal(etaPhrase({ eta_state: 'estimating' }), 'estimating time left…');
});

test('a phase with nothing to count says nothing', () => {
  // The style grouping runs 181 s on done=0/total=0. Same rule the counter
  // already applies to the bare "0": silence beats a guess.
  assert.equal(etaPhrase({ eta_state: 'none', done: 0, total: 0 }), '');
});

test('a snapshot from a build with no estimator renders no clause', () => {
  assert.equal(etaPhrase({ done: 5, total: 10 }), '');
});

test('a finished, failed or stopped pass carries no remaining time', () => {
  const ready = { eta_state: 'ready', eta_seconds: 600, eta_scope: 'job' };
  assert.equal(etaPhrase({ ...ready, finished: true }), '');
  assert.equal(etaPhrase({ ...ready, error: 'boom' }), '');
  assert.equal(etaPhrase({ ...ready, cancelled: true }), '');
});

test('the busy refusal now says how long the blocker needs', () => {
  const line = busyLine({
    activity: {
      kind: 'score', done: 137, total: 412,
      eta_state: 'ready', eta_seconds: 1500, eta_scope: 'job',
    },
  });
  assert.equal(line, '✨ Score pass is running on this bank — 137 / 412 · about 25 minutes left');
});

test('the busy refusal keeps its shape when there is no estimate yet', () => {
  const line = busyLine({
    activity: { kind: 'score', done: 137, total: 412, eta_state: 'none' },
  });
  assert.equal(line, '✨ Score pass is running on this bank — 137 / 412');
});
