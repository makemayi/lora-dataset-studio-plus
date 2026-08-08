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

/* The fresh install, from the report that is this feature's spec: URL and
   ComfyUI folder auto-detected, model files found, first Generate answered "A
   paused comfyui job is blocking new generation" — and his ComfyUI logged NO
   incoming connection at all, so he went looking for a special flag he had to
   pass. Nothing on that screen could have told him LDS was talking to the wrong
   address (jerkyjunky, Discord). */
const UNREACHABLE = {
  recovery: {
    kind: 'unknown_submit',
    job_id: 'job-1',
    can_confirm_restart: true,
    connection: { reachable: false, url: 'http://127.0.0.1:8188',
      status: 'unreachable', hint: null },
  },
};

test('an unreachable ComfyUI is the headline, and it names the address', () => {
  const model = recoveryBannerModel(UNREACHABLE, { now: NOW });
  assert.match(model.headline, /cannot reach ComfyUI at http:\/\/127\.0\.0\.1:8188/);
  assert.doesNotMatch(model.headline, /paused/i);
});

test('the unreachable banner lists what to check, including the two invisible ones', () => {
  const { checks } = recoveryBannerModel(UNREACHABLE, { now: NOW });
  assert.ok(checks.length >= 3);
  assert.ok(checks.some((c) => /--listen/.test(c)));          // bound to 127.0.0.1
  assert.ok(checks.some((c) => /host\.docker\.internal/.test(c)));
  assert.ok(checks.some((c) => /Settings/.test(c)));
});

test('the paused job is demoted to a footnote, not the accusation', () => {
  const model = recoveryBannerModel(UNREACHABLE, { now: NOW });
  assert.match(model.footnote, /could not confirm ComfyUI ever accepted/);
  assert.doesNotMatch(model.detail, /paused for/);
  assert.equal(model.canConfirm, true);   // still the way out, once it answers
});

test('a known prompt behind a dead link says LDS finishes it once ComfyUI answers', () => {
  const model = recoveryBannerModel({ recovery: { ...UNREACHABLE.recovery, kind: 'prompt' } });
  assert.match(model.headline, /cannot reach ComfyUI/);
  assert.match(model.footnote, /clears it by itself/);
});

test('no configured address says THAT, instead of "cannot reach at "', () => {
  const model = recoveryBannerModel({ recovery: { ...UNREACHABLE.recovery,
    connection: { reachable: false, url: '', status: 'unconfigured',
      hint: 'Set the ComfyUI API URL in Settings ▸ Local tools.' } } });
  assert.match(model.headline, /no ComfyUI address/);
  assert.doesNotMatch(model.headline, /at\s*$/);
  assert.match(model.detail, /Set the ComfyUI API URL/);
});

test('a reachable ComfyUI keeps the paused-job story word for word', () => {
  const model = recoveryBannerModel({ recovery: { ...OTHER_DATASET.recovery,
    connection: { reachable: true, url: 'http://127.0.0.1:8188', status: 'ok', hint: null } },
  }, { now: NOW });
  assert.equal(model.headline, 'A paused ComfyUI job is blocking new generations');
  assert.deepEqual(model.checks, []);
  assert.equal(model.footnote, null);
});

/* A server that predates this field, and the poll's own "nobody asked" value.
   Neither may be read as "unreachable" — inventing a connection failure is as
   wrong as hiding one. */
test('no connection verdict at all leaves the banner exactly as it was', () => {
  assert.equal(recoveryBannerModel(OTHER_DATASET, { now: NOW }).headline,
    'A paused ComfyUI job is blocking new generations');
  assert.equal(recoveryBannerModel({ recovery: { ...OTHER_DATASET.recovery,
    connection: null } }, { now: NOW }).headline,
  'A paused ComfyUI job is blocking new generations');
});

test('an unreadable record outranks the connection — it survives ComfyUI coming back', () => {
  const model = recoveryBannerModel({ recovery: { kind: 'unreadable',
    detail: 'LDS found an invalid ComfyUI recovery record.',
    connection: { reachable: false, url: 'http://127.0.0.1:8188', status: 'unreachable' } } });
  assert.match(model.headline, /unreadable/);
  assert.equal(model.canConfirm, false);
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
