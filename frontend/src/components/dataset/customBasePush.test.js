import test from 'node:test';
import assert from 'node:assert/strict';

import { customBasePushView } from './customBasePush.js';

/* The reported modal, verbatim: "Custom base: bigLove_zt3.safetensors" /
 * "⚠ The local file is unavailable (missing) — restore it to push." / a
 * "⬆ Push custom base to Hugging Face" button. The file was NOT missing: a
 * Z-Image merge name had stayed attached to a dataset switched to Krea 2, and
 * the readiness probe resolved it as a Krea absolute path. */
test('a base from another family is named as such, and offers no push', () => {
  const v = customBasePushView({
    state: {
      ready: false,
      reason: 'foreign_family',
      local_available: false,
      foreign_base_message: '“bigLove_zt3.safetensors” was chosen for another model '
        + 'family, not Krea 2 — a Krea 2 run cannot load it, so this run uses the '
        + 'official Krea 2 base. Pick a Krea 2 base to change that.',
    },
  });
  assert.equal(v.kind, 'foreign');
  assert.equal(v.showPush, false);
  assert.equal(v.canPush, false);
  assert.match(v.message, /another model family/);
  assert.doesNotMatch(v.message, /missing/);
  assert.doesNotMatch(v.message, /restore/);
});

test('the family mismatch outranks a missing token', () => {
  // Both are true on a fresh install; only one of them is the user's problem.
  const v = customBasePushView({
    state: { ready: false, reason: 'foreign_family', local_available: false,
      foreign_base_message: 'another model family' },
  });
  assert.equal(v.kind, 'foreign');
});

test('an absent local file never enables the push', () => {
  const v = customBasePushView({
    state: { ready: false, reason: null, local_available: false,
      local_reason: 'weights_missing' },
  });
  assert.equal(v.kind, 'push');
  assert.equal(v.showPush, true, 'the requirement stays visible');
  assert.equal(v.canPush, false, 'but there is nothing to upload');
  assert.match(v.warning, /weights_missing/);
});

test('a present local file enables the push', () => {
  const v = customBasePushView({
    state: { ready: false, reason: null, local_available: true },
  });
  assert.equal(v.canPush, true);
  assert.equal(v.warning, null);
  assert.match(v.message, /cannot download yet/);
});

test('an already-pushed base shows no push at all', () => {
  const v = customBasePushView({ state: { ready: true, repo_id: 'me/lds-base-h1' } });
  assert.equal(v.kind, 'ready');
  assert.equal(v.showPush, false);
});

test('size mismatch and file_missing keep their own explanations', () => {
  assert.match(
    customBasePushView({ state: { reason: 'size_mismatch', local_available: true } }).message,
    /changed since it was pushed/);
  assert.match(
    customBasePushView({ state: { reason: 'file_missing', local_available: true } }).message,
    /missing the file this variant needs/);
});

test('token problems are handed to their own copy, with no push offered', () => {
  for (const reason of ['no_token', 'token_invalid']) {
    const v = customBasePushView({ state: { ready: false, reason, local_available: true } });
    assert.equal(v.kind, reason);
    assert.equal(v.showPush, false);
  }
});

test('a check error and a pending check never offer a push', () => {
  assert.equal(customBasePushView({ checkError: 'Network error' }).showPush, false);
  assert.equal(customBasePushView({}).kind, 'checking');
});

test('an in-flight push does not re-offer the button', () => {
  const v = customBasePushView({
    state: { ready: false, reason: null, local_available: true }, pushing: true,
  });
  assert.equal(v.kind, 'pushing');
  assert.equal(v.showPush, false);
});
