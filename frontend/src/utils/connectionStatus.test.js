import test, { beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  getConnectionState, subscribeConnection, reportRequestFailure,
  reportRequestSuccess, resetConnectionStatus,
} from './connectionStatus.js';

beforeEach(() => resetConnectionStatus());

test('a background poll failing is NEVER announced', () => {
  for (let i = 0; i < 10; i += 1) {
    assert.equal(reportRequestFailure({ background: true }), false);
  }
  assert.equal(getConnectionState().online, false, 'but it does flip the state');
  assert.equal(getConnectionState().failures, 10);
});

test('a user-triggered failure IS announced — the app must not go mute', () => {
  assert.equal(reportRequestFailure({ background: false }), true);
});

test('a user action failing while a poll already went dark still speaks up', () => {
  reportRequestFailure({ background: true });          // silent
  reportRequestFailure({ background: true });          // silent
  assert.equal(reportRequestFailure({ background: false }), true,
    'the click deserves an answer even though the episode was already open');
});

test('repeated user failures announce ONCE per outage, not once per attempt', () => {
  assert.equal(reportRequestFailure({ background: false }), true);
  assert.equal(reportRequestFailure({ background: false }), false);
  assert.equal(reportRequestFailure({ background: false }), false);
});

test('coming back announces exactly once, then goes quiet', () => {
  reportRequestFailure({ background: true });
  assert.equal(reportRequestSuccess(), true);
  assert.equal(reportRequestSuccess(), false);
  assert.equal(getConnectionState().online, true);
  assert.equal(getConnectionState().failures, 0);
  assert.equal(getConnectionState().offlineSince, null);
});

test('a success while already online announces nothing', () => {
  assert.equal(reportRequestSuccess(), false);
});

test('a new outage after a recovery is announced again', () => {
  reportRequestFailure({ background: false });
  reportRequestSuccess();
  assert.equal(reportRequestFailure({ background: false }), true);
});

test('offlineSince marks the START of the episode, not the latest failure', () => {
  reportRequestFailure({ background: true, now: 1000 });
  reportRequestFailure({ background: true, now: 5000 });
  assert.equal(getConnectionState().offlineSince, 1000);
});

test('subscribers see the transitions and can unsubscribe', () => {
  const seen = [];
  const off = subscribeConnection((s) => seen.push(s.online));
  reportRequestFailure({ background: true });
  reportRequestSuccess();
  off();
  reportRequestFailure({ background: true });
  assert.deepEqual(seen, [false, true]);
});
