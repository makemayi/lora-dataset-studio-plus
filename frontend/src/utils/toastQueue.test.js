import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MAX_VISIBLE_TOASTS, pushToast, dropToast, sweepToasts, toastLabel,
} from './toastQueue.js';

const entry = (message, type = 'error', id = 1, expiresAt = 1000) =>
  ({ id, message, type, expiresAt });

test('ten identical failures produce ONE banner carrying the count', () => {
  let list = [];
  for (let i = 1; i <= 10; i += 1) {
    list = pushToast(list, entry('Connection lost. Please check your network.', 'error', i, 1000 + i));
  }
  assert.equal(list.length, 1, 'the repeated message must not stack');
  assert.equal(list[0].count, 10);
  assert.equal(toastLabel(list[0]), 'Connection lost. Please check your network. (10×)');
});

test('a merged toast keeps the id of the first emission (its ✕ still works)', () => {
  let list = pushToast([], entry('Boom', 'error', 7));
  list = pushToast(list, entry('Boom', 'error', 8));
  assert.equal(list[0].id, 7);
  assert.deepEqual(dropToast(list, 7), []);
});

test('merging pushes the expiry out, so repeats do not die on the first timer', () => {
  let list = pushToast([], entry('Boom', 'error', 1, 1000));
  list = pushToast(list, entry('Boom', 'error', 2, 9000));
  assert.equal(list[0].expiresAt, 9000);
  assert.equal(sweepToasts(list, 5000).length, 1, 'still alive at t=5000');
  assert.equal(sweepToasts(list, 9001).length, 0);
});

test('same text in a different tone is a different notification', () => {
  let list = pushToast([], entry('Saved', 'success', 1));
  list = pushToast(list, entry('Saved', 'info', 2));
  assert.equal(list.length, 2);
});

test('the visible cap holds even with all-different messages', () => {
  let list = [];
  for (let i = 1; i <= 12; i += 1) list = pushToast(list, entry(`Failure #${i}`, 'error', i));
  assert.equal(list.length, MAX_VISIBLE_TOASTS);
  // Oldest dropped, newest kept.
  assert.equal(list.at(-1).message, 'Failure #12');
  assert.equal(list[0].message, 'Failure #9');
});

test('a sticky toast is not evicted by a burst of transient ones', () => {
  let list = pushToast([], { id: 0, message: 'Pinned', type: 'info', expiresAt: null });
  for (let i = 1; i <= 12; i += 1) list = pushToast(list, entry(`Noise #${i}`, 'error', i));
  assert.equal(list.length, MAX_VISIBLE_TOASTS);
  assert.ok(list.some((t) => t.message === 'Pinned'), 'sticky survived the burst');
});

test('sweep drops expired toasts and keeps sticky ones', () => {
  const list = [
    { id: 1, message: 'a', type: 'info', expiresAt: 100 },
    { id: 2, message: 'b', type: 'info', expiresAt: null },
  ];
  assert.deepEqual(sweepToasts(list, 500).map((t) => t.id), [2]);
});

test('pushToast never mutates the list it was given', () => {
  const before = [entry('a', 'error', 1)];
  const snapshot = JSON.stringify(before);
  pushToast(before, entry('a', 'error', 2));
  pushToast(before, entry('b', 'error', 3));
  assert.equal(JSON.stringify(before), snapshot);
});
