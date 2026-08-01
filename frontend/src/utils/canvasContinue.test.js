import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
  canvasContinueLanes, canvasContinueRefusal, canvasContinueRequest,
  canvasContinueRow, canvasContinueSettings, canvasContinueSteps,
} from './canvasContinue.js';

const canvas = fs.readFileSync(new URL('../components/canvas/LineageCanvas.jsx', import.meta.url), 'utf8');

// A cloud lineage node as `_lineage_node` serialises it (cloud branch: run_id +
// status), with three harvested saves.
const CLOUD = {
  record_id: 11, source: 'cloud', run_id: 7, status: 'done',
  dataset_id: 3, train_type: 'zimage', variant: 'turbo', base_model: '', steps: 3500,
  checkpoints: [{ step: 1500 }, { step: 2500 }, { step: 3500, final: true }],
  config: { optimizer: 'adamw8bit', lr: 5e-5, rank: 32, alpha: 16, save_every: 500 },
};
// A local node: NO run_id at all — record_id is the universal key.
const LOCAL = {
  record_id: 12, source: 'local', status: null,
  dataset_id: 4, train_type: 'sdxl', variant: 'base', base_model: 'sdxl.safetensors', steps: 2000,
  checkpoints: [{ step: 1000 }, { step: 2000, final: true }],
  config: { optimizer: 'prodigy', lr: 1, rank: 64 },
};
const OPEN = { aitoolkitValid: true, configured: true, limit: 2, actives: [] };

// --- what the dialog opens ON ----------------------------------------------

test('the resumable steps come from the node’s OWN saves, ascending and distinct', () => {
  assert.deepEqual(canvasContinueSteps(CLOUD), [1500, 2500, 3500]);
  assert.deepEqual(canvasContinueSteps(LOCAL), [1000, 2000]);
  // a node with nothing to resume yields an empty list, never a fabricated step
  assert.deepEqual(canvasContinueSteps({ checkpoints: [] }), []);
  assert.deepEqual(canvasContinueSteps({ checkpoints: [{ step: 0 }, { step: null }] }), []);
});

test('the hub row is matched on record_id — the key BOTH payloads share', () => {
  const rows = [{ record_id: 12, masked: false, settings: { optimizer: 'adamw', lr: 1e-4 } },
    { record_id: 11, masked: true, settings: null }];
  assert.equal(canvasContinueRow(LOCAL, rows).masked, false);
  assert.equal(canvasContinueRow(CLOUD, rows).record_id, 11);
  // a run outside the hub's window is simply unknown, not mis-matched
  assert.equal(canvasContinueRow({ record_id: 99 }, rows), null);
  assert.equal(canvasContinueRow(CLOUD, null), null);
});

test('the LR hint reads the snapshot’s own spelling (`lr`), and invents nothing', () => {
  const s = canvasContinueSettings(CLOUD);
  assert.equal(s.optimizer, 'adamw8bit');
  assert.equal(s.learning_rate, 5e-5);
  assert.equal(s.rank, 32);
  assert.equal(s.alpha, 16);
  assert.equal(s.save_every, 500);
  // `sample_every` is only stamped when truthy — absent must stay absent so the
  // dialog falls back to its own default rather than showing a made-up cadence.
  assert.equal('sample_every' in s, false);
  // a run that recorded no settings yields {} — the dialog then shows the
  // documented default instead of a fabricated rate
  assert.deepEqual(canvasContinueSettings({ config: null }), {});
  // the hub row wins when present (same snapshot, re-read live)
  assert.equal(
    canvasContinueSettings(CLOUD, { settings: { optimizer: 'prodigy', lr: 1 } }).optimizer,
    'prodigy');
});

// --- the lanes --------------------------------------------------------------

test('a finished cloud run offers BOTH lanes from the board', () => {
  const lanes = canvasContinueLanes(CLOUD, CLOUD.checkpoints[1], OPEN);
  assert.equal(lanes.cloud.available, true);
  assert.equal(lanes.local.available, true);
});

test('a LOCAL run offers both lanes too — the board is the only surface that can', () => {
  // The Runs hub refuses a local node outright (cloud-only continue), the
  // dataset panel refuses a pill outside its active selection. The board has
  // dataset_id on every node, so continue-local can seed a pod from it.
  const lanes = canvasContinueLanes(LOCAL, LOCAL.checkpoints[0], OPEN);
  assert.equal(lanes.local.available, true);
  assert.equal(lanes.cloud.available, true);
});

test('a closed lane keeps its slot and states the reason — never a hidden option', () => {
  const noToolkit = canvasContinueLanes(CLOUD, null, { ...OPEN, aitoolkitValid: false });
  assert.equal(noToolkit.local.available, false);
  assert.match(noToolkit.local.reason, /ai-toolkit/);
  assert.equal(noToolkit.cloud.available, true);

  const noKey = canvasContinueLanes(CLOUD, null, { ...OPEN, configured: false });
  assert.equal(noKey.cloud.available, false);
  assert.match(noKey.cloud.reason, /vast\.ai API key/);

  const busy = canvasContinueLanes(LOCAL, null, { ...OPEN, localActive: { current: {} } });
  assert.equal(busy.local.available, false);
  assert.match(busy.local.reason, /already running on this machine/);
});

test('a cloud run that is not linked here cannot be relaunched in the cloud', () => {
  const lanes = canvasContinueLanes({ ...CLOUD, run_id: null }, CLOUD.checkpoints[0], OPEN);
  assert.equal(lanes.cloud.available, false);
  assert.match(lanes.cloud.reason, /not linked on this machine/);
});

// --- the checkpoint whose file is gone --------------------------------------

test('a save that is no longer on disk closes the lanes that need the FILE', () => {
  // A cloud run's cloud lane re-seeds from its own staging by run_id, so it
  // survives; every lane that uploads or resumes the local file does not. The
  // asymmetry is the honest answer, not a blanket refusal.
  const lanes = canvasContinueLanes(CLOUD, { step: 2500, present: false }, OPEN);
  assert.equal(lanes.local.available, false);
  assert.match(lanes.local.reason, /no longer on this machine/);
  assert.match(lanes.local.reason, /continue in the cloud/);
  assert.equal(lanes.cloud.available, true);
});

test('a LOCAL run’s missing save closes BOTH lanes — there is no copy anywhere', () => {
  const lanes = canvasContinueLanes(LOCAL, { step: 1000, present: false }, OPEN);
  assert.equal(lanes.local.available, false);
  assert.equal(lanes.cloud.available, false);
  assert.match(lanes.cloud.reason, /no file to send to a pod/);
});

test('a run with no checkpoint at all is refused up front, with the reason', () => {
  assert.match(canvasContinueRefusal({ ...CLOUD, checkpoints: [] }, null), /no checkpoint/);
  assert.match(canvasContinueRefusal(null, null), /unknown/);
  assert.equal(canvasContinueRefusal(CLOUD, CLOUD.checkpoints[0]), null);
  // a local node with no dataset can address no endpoint at all
  assert.match(canvasContinueRefusal({ ...LOCAL, dataset_id: null }, null), /dataset is unknown/);
});

// --- the routing ------------------------------------------------------------

test('cloud lane on a CLOUD run relaunches that run by id', () => {
  const req = canvasContinueRequest(CLOUD,
    { lane: 'cloud', extraSteps: 1000, fromStep: 2500 }, { steps: [1500, 2500, 3500] });
  assert.equal(req.url, '/api/dataset/train/cloud/continue');
  assert.deepEqual(req.body, {
    run_id: 7, extra_steps: 1000, from_step: 2500, resume_mode: 'weights_only',
  });
});

test('cloud lane on a LOCAL run seeds a pod from the local file (continue-local)', () => {
  const req = canvasContinueRequest(LOCAL,
    { lane: 'cloud', extraSteps: 500, fromStep: 1000 },
    { steps: [1000, 2000], masked: false });
  assert.equal(req.url, '/api/dataset/4/train/cloud/continue-local');
  assert.equal(req.body.from_step, 1000);
  assert.equal(req.body.base_model, 'sdxl.safetensors');
  assert.equal(req.body.train_type, 'sdxl');
  assert.equal(req.body.variant, 'base');
  // the SOURCE run's masking, not a board-wide default
  assert.equal(req.body.masked, false);
});

test('local lane resumes through the dataset endpoint, for either source', () => {
  const fromCloud = canvasContinueRequest(CLOUD,
    { lane: 'local', extraSteps: 1000, fromStep: 1500 }, { steps: [1500, 2500, 3500], masked: true });
  assert.equal(fromCloud.url, '/api/dataset/3/train/continue');
  assert.equal(fromCloud.body.from_step, 1500);
  assert.equal(fromCloud.body.masked, true);

  const fromLocal = canvasContinueRequest(LOCAL,
    { lane: 'local', extraSteps: 1000, fromStep: null }, { steps: [1000, 2000] });
  assert.equal(fromLocal.url, '/api/dataset/4/train/continue');
});

test('the CHOSEN checkpoint rides the request — never "whatever is newest"', () => {
  // THE regression this whole feature exists for: the board shows several runs
  // sharing one lane's run dir, so an implicit "resume in place" would continue
  // a different run than the card that was clicked. The dialog nulls fromStep
  // for the newest save; the board re-materialises it.
  const explicit = canvasContinueRequest(CLOUD,
    { lane: 'local', extraSteps: 1000, fromStep: 1500 }, { steps: [1500, 2500, 3500] });
  assert.equal(explicit.body.from_step, 1500, 'the clicked, earlier step must be sent');

  const newest = canvasContinueRequest(CLOUD,
    { lane: 'local', extraSteps: 1000, fromStep: null }, { steps: [1500, 2500, 3500] });
  assert.equal(newest.body.from_step, 3500,
    'the newest save of THIS run must be named, not left to the lane');
});

test('overrides ride only when the dialog produced some', () => {
  const bare = canvasContinueRequest(CLOUD, { lane: 'cloud', extraSteps: 1000 }, { steps: [3500] });
  assert.equal('overrides' in bare.body, false);
  const withOv = canvasContinueRequest(CLOUD,
    { lane: 'cloud', extraSteps: 1000, overrides: { lr_factor: 0.5 } }, { steps: [3500] });
  assert.deepEqual(withOv.body.overrides, { lr_factor: 0.5 });
});

test('resume mode and opaque bundle id survive all board routes', () => {
  const req = canvasContinueRequest(LOCAL, {
    lane: 'local',
    extraSteps: 500,
    fromStep: 1000,
    resumeMode: 'full_state',
    stateBundleId: '0123456789abcdef0123456789abcdef',
  }, { steps: [1000, 2000] });
  assert.equal(req.body.resume_mode, 'full_state');
  assert.equal(req.body.state_bundle_id, '0123456789abcdef0123456789abcdef');
});

test('an unaddressable run yields no request rather than a wrong one', () => {
  assert.equal(canvasContinueRequest(null, { lane: 'local' }, {}), null);
  assert.equal(canvasContinueRequest({ ...LOCAL, dataset_id: null },
    { lane: 'local', extraSteps: 1000 }, {}), null);
});

// --- the board wiring (contract) -------------------------------------------

test('the board opens the SHARED ContinueDialog — no third continue form', () => {
  assert.match(canvas, /import ContinueDialog from '\.\.\/dataset\/ContinueDialog'/);
  assert.match(canvas, /<ContinueDialog/);
  assert.match(canvas, /lanes=\{continueLanes\}/);
  assert.match(canvas, /initialFromStep=\{continueTarget\.step\}/);
  assert.match(canvas, /checkpoints=\{continueTarget\.node\.checkpoints \|\| \[\]\}/);
});

test('the popover’s ▶ Continue is a real button on the board, not a sentence', () => {
  // It used to be greyed text telling the user to go to another page. A handler
  // is what makes checkpointActionModel render the live row.
  assert.match(canvas, /onContinue=\{handleContinueCheckpoint\}/);
  assert.match(canvas, /continueSource="any"/);
  assert.doesNotMatch(canvas, /continueReason="Continue from here: open this run/);
});

test('the board surfaces the backend’s refusal instead of swallowing it', () => {
  // postJson THROWS on a 400/409 — the very refusal a vanished checkpoint
  // produces ("no local checkpoint at step N"). Without the catch the click
  // looks dead, which is the bug the Runs hub already paid for once.
  assert.match(canvas, /await postWithConfirmations\(\(b\) => postJson\(req\.url, b\)/);
  assert.match(canvas, /catch \(e\) \{[\s\S]*?continueAttemptOutcome\(\{ thrown: e \}\)/);
  // an explicit {ok:false} body is a refusal too, not a success toast — both
  // shapes go through the one classifier (utils/continueOutcome.js).
  assert.match(canvas, /continueAttemptOutcome\(d === null \? \{ declined: true \} : \{ response: d \}\)/);
  // …and the dialog is NO LONGER dismissed before the request. It used to be, to
  // work around a toast container that rendered under every modal (fixed:
  // Toast.jsx is z-[10000]) — at the price of discarding the lane, the resume
  // checkpoint, the extra steps and the five folded settings on every refusal.
  // The ordering contract for all three hosts lives in
  // components/dataset/ContinueDialogRefusal.contract.test.js.
  assert.doesNotMatch(canvas, /setContinueTarget\(null\);\s*\n\s*if \(!payload\) return;/);
  assert.match(canvas, /if \(!outcome\.close\) \{ setContinueError\(outcome\.error\); return; \}/);
});

test('the dialog waits for the lane inputs before it seeds itself', () => {
  // MEASURED regression: ContinueDialog resolves its lane and its resume-from in
  // useState initialisers — once, on mount. Mounting it before the Runs payload
  // landed made a configured cloud lane read "needs a vast.ai API key",
  // pre-selected Local for a CLOUD run, and posted to the local endpoint.
  assert.match(canvas, /\{continueTarget && continueRuns && \(\s*<ContinueDialog/);
  // and a failed read must still SETTLE (to {}), or the dialog would never open
  assert.match(canvas, /\.catch\(\(\) => setContinueRuns\(\{\}\)\)/);
});

test('the board never guesses a lane input it failed to read', () => {
  // The lane guards are fed from the Runs-hub payload + capabilities, fetched
  // ONLY when the dialog opens — a board that polled them on idle would drain a
  // phone drawing a static graph.
  assert.match(canvas, /apiFetch\('\/api\/dataset\/train\/cloud\/runs\?limit=50'\)/);
  // ai-toolkit's own probe is the app-wide one — no second request for it
  assert.match(canvas, /const \{ caps \} = useCapabilities\(\)/);
  assert.match(canvas, /aitoolkitValid: caps\?\.aitoolkit\?\.valid/);
  assert.match(canvas, /canvasContinueLanes\(continueTarget\.node, continueTarget\.pill/);
});
