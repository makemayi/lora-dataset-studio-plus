import test from 'node:test';
import assert from 'node:assert/strict';
import {
  progressPresence, PROGRESS_HIDDEN, PROGRESS_RUNNING, PROGRESS_STALE, PROGRESS_UNKNOWN,
} from './progressPresence.js';

const running = { kind: 'scan', done: 42, total: 300, finished: false };

test('online + a running job → the normal progress bar', () => {
  assert.equal(progressPresence(running, false), PROGRESS_RUNNING);
});

test('a failed poll KEEPS the last known progress instead of blanking it', () => {
  // This is the regression: the snapshot survives, the zone stays populated,
  // and the loss of contact is added ON TOP — never substituted for the state.
  assert.equal(progressPresence(running, true), PROGRESS_STALE);
});

test('offline with no snapshot says "I do not know", not "nothing is running"', () => {
  assert.equal(progressPresence(null, true), PROGRESS_UNKNOWN);
  assert.equal(progressPresence(undefined, true), PROGRESS_UNKNOWN);
});

test('online with nothing running stays silent — no permanent clutter', () => {
  assert.equal(progressPresence(null, false), PROGRESS_HIDDEN);
  assert.equal(progressPresence({ finished: true }, false), PROGRESS_HIDDEN);
});

test('a job that FINISHED before the outage is not resurrected as stale', () => {
  assert.equal(progressPresence({ finished: true }, true), PROGRESS_UNKNOWN);
});

test('offline defaults to false — existing callers keep today’s behaviour', () => {
  assert.equal(progressPresence(running), PROGRESS_RUNNING);
  assert.equal(progressPresence(null), PROGRESS_HIDDEN);
});
