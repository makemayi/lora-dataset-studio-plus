import test from 'node:test';
import assert from 'node:assert/strict';
import { autoClearedMessage, recoveryBannerModel, stalledForText } from './comfyRecovery.js';

const NOW = Date.parse('2026-08-02T12:00:00Z');

/* The lived case: job 5359 died on dataset 40 when the disk filled, and the
   user was working on dataset 44. Nothing on his screen ever mentioned dataset
   40 — the banner's whole job is to say WHERE the stuck job lives. */
const OTHER_DATASET = {
  recovery: {
    kind: 'prompt',
    job_id: 'job-5359',
    dataset_id: 40,
    dataset_name: 'Anna',
    variation_label: 'portrait',
    stalled_since: '2026-08-02T11:40:00Z',
    can_confirm_restart: true,
  },
};

test('nothing blocking -> no banner', () => {
  assert.equal(recoveryBannerModel(null), null);
  assert.equal(recoveryBannerModel({ recovery: null }), null);
});

test('names the dataset and variation the stuck job belongs to', () => {
  const model = recoveryBannerModel(OTHER_DATASET, { now: NOW });
  assert.match(model.detail, /Anna/);
  assert.match(model.detail, /portrait/);
  assert.equal(model.datasetId, 40);
  assert.equal(model.canConfirm, true);
  assert.match(model.actionLabel, /restarted ComfyUI/);
});

test('says how long it has been stuck, so "from an earlier session" is obvious', () => {
  const model = recoveryBannerModel(OTHER_DATASET, { now: NOW });
  assert.match(model.detail, /paused for 20 minutes/);
});

test('a missing timestamp drops the duration instead of inventing one', () => {
  const model = recoveryBannerModel(
    { recovery: { ...OTHER_DATASET.recovery, stalled_since: null } }, { now: NOW });
  assert.doesNotMatch(model.detail, /paused for/);
  assert.match(model.detail, /Anna/);
});

test('an unknown submission says why LDS needs the user\'s word', () => {
  const model = recoveryBannerModel(
    { recovery: { ...OTHER_DATASET.recovery, kind: 'unknown_submit' } }, { now: NOW });
  assert.match(model.detail, /cannot identify the remote job/);
  assert.equal(model.canConfirm, true);
});

/* A known prompt id resolves itself: promising the user a click they don't
   need — or leaving them to think a restart is not enough — is the whole
   reason this text is a tested function and not a string in JSX. */
test('a known prompt says the restart alone is enough', () => {
  const model = recoveryBannerModel(OTHER_DATASET, { now: NOW });
  assert.match(model.detail, /clears this by itself/);
  assert.doesNotMatch(model.detail, /cannot identify/);
});

test('an unreadable record offers no button it could only refuse', () => {
  const model = recoveryBannerModel({ recovery: { kind: 'unreadable',
    detail: 'LDS found an invalid ComfyUI recovery record.' } });
  assert.equal(model.canConfirm, false);
  assert.equal(model.actionLabel, null);
  assert.equal(model.tone, 'error');
});

test('a job with no dataset still produces a sentence, not "undefined"', () => {
  const model = recoveryBannerModel({ recovery: { kind: 'prompt', job_id: 'j' } });
  assert.match(model.detail, /A generation stopped without a known outcome/);
  assert.doesNotMatch(model.detail, /undefined|null/);
});

test('stalledForText stays coarse', () => {
  assert.equal(stalledForText('2026-08-02T11:59:30Z', NOW), 'a moment');
  assert.equal(stalledForText('2026-08-02T11:30:00Z', NOW), '30 minutes');
  assert.equal(stalledForText('2026-08-02T11:00:00Z', NOW), '1 hour');
  assert.equal(stalledForText('2026-08-01T12:00:00Z', NOW), '1 day');
  assert.equal(stalledForText(null, NOW), null);
  assert.equal(stalledForText('not a date', NOW), null);
});

test('the automatic-clear toast fires once per notice, not once per poll', () => {
  const state = { auto_cleared: { id: 'n1', message: 'cleared automatically' } };
  assert.deepEqual(autoClearedMessage(state, null),
    { id: 'n1', message: 'cleared automatically' });
  assert.equal(autoClearedMessage(state, 'n1'), null);
  assert.equal(autoClearedMessage({ auto_cleared: null }, null), null);
});
