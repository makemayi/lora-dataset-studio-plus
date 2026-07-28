import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
  MACHINE_SCOPE_CHECKS, checksForLane, laneOfPayload, normalizeLane, preflightUrl,
} from './preflightLane.js';

const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');
const modal = fs.readFileSync(new URL('./PreflightModal.jsx', import.meta.url), 'utf8');

// --- the URL contract ---------------------------------------------------------

test('no lane is sent unless it is the cloud one — the existing callers must not move', () => {
  // The workspace readiness badge calls this route with no lane; its payload has
  // to stay byte-for-byte what it was.
  assert.equal(preflightUrl(7, { trainType: 'krea', variant: 'turbo' }),
    '/api/dataset/7/train/preflight?train_type=krea&variant=turbo');
  assert.equal(preflightUrl(7, { trainType: 'krea', variant: 'turbo', lane: 'local' }),
    '/api/dataset/7/train/preflight?train_type=krea&variant=turbo');
  assert.equal(preflightUrl(7, {}), '/api/dataset/7/train/preflight');
});

test('the cloud lane rides in the query string', () => {
  assert.equal(preflightUrl(7, { trainType: 'zimage', variant: 'base', lane: 'cloud' }),
    '/api/dataset/7/train/preflight?train_type=zimage&variant=base&lane=cloud');
});

test('anything unrecognised falls back to the local lane', () => {
  assert.equal(normalizeLane(undefined), 'local');
  assert.equal(normalizeLane('CLOUD'), 'local');   // exact match only, never guessed
  assert.equal(normalizeLane('cloud'), 'cloud');
  assert.equal(laneOfPayload({ lane: 'cloud' }), 'cloud');
  assert.equal(laneOfPayload({ lane: 'local' }), 'local');
  assert.equal(laneOfPayload(null), 'local');
});

// --- the guard-rail -----------------------------------------------------------

test('GPU-memory and torch-build rows never reach a cloud lane', () => {
  // The whole point: those two read THIS machine. On a cloud-only install they
  // would fire on every launch, and a warning that cries wolf makes people click
  // through the nine that do not.
  const rows = [
    { id: 'leaks', scope: 'dataset' },
    { id: 'vram', scope: 'machine' },
    { id: 'torch_arch', scope: 'machine' },
    { id: 'face_mask', scope: 'dataset' },
  ];
  assert.deepEqual(checksForLane(rows, 'cloud').map((c) => c.id), ['leaks', 'face_mask']);
  assert.deepEqual(checksForLane(rows, 'local').map((c) => c.id),
    ['leaks', 'vram', 'torch_arch', 'face_mask']);
  assert.deepEqual(MACHINE_SCOPE_CHECKS, ['vram', 'torch_arch']);
});

test('a machine row with no scope field is still dropped by id', () => {
  // Defence against an older backend answering a newer frontend.
  assert.deepEqual(checksForLane([{ id: 'vram' }, { id: 'triage' }], 'cloud')
    .map((c) => c.id), ['triage']);
});

test('face_mask is NOT machine-scope — it is the inverse trap', () => {
  // InsightFace runs locally at export and the masks are uploaded with the
  // images: missing here means the PAID run trains unmasked.
  assert.ok(!MACHINE_SCOPE_CHECKS.includes('face_mask'));
});

// --- the call sites that were skipping the gate -------------------------------

test('the cloud launch runs the preflight before spending money', () => {
  assert.match(panel, /const launchCloud = async \(gpuName\) => \{\s*(\/\/[^\n]*\n\s*)*if \(!\(await preflightOk\(\{ lane: 'cloud' \}\)\)\) return;/);
});

test('▶ Continue runs it too, on whichever lane it resumes', () => {
  // The gap nobody had reported: runContinue skipped the preflight on BOTH lanes.
  assert.match(panel, /const lane = laneOfPayload\(payload\);/);
  assert.match(panel, /await preflightOk\(\{ lane, trainType: checkpointTrainType,\s*variant: checkpointVariant \}\)/);
});

test('the modal names the cloud lane and keeps its fix-in-place lists', () => {
  assert.match(modal, /lane = 'local'/);
  assert.match(modal, /Before training\{inCloud \? ' in the cloud' : ''\}/);
  // Editable leaking captions and rejectable duplicate pairs stay: without them
  // the cloud modal would just be a dialog box with nothing to act on.
  assert.match(modal, /ds\.setCaption\(li\.id, e\.target\.value\)/);
  assert.match(modal, /reject\(im\.id\)/);
});
