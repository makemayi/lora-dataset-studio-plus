import test from 'node:test';
import assert from 'node:assert/strict';
import {
  framePromoteProblem, framePromotePayload, frameScopeLabel,
  frameCeilingHint, frameFaceNote, FRAMES_PER_CLIP_MAX,
} from './videoFramePromote.js';

const ok = { name: 'faces', framesPerClip: 3, requireFace: false, refDatasetId: null };

test('a nameless request is refused before it is sent', () => {
  assert.match(framePromoteProblem({ ...ok, name: '   ' }), /Name the dataset/);
});

test('a frame budget has to be a whole positive number', () => {
  for (const bad of [0, -1, 2.5, NaN, null, 'three']) {
    assert.ok(framePromoteProblem({ ...ok, framesPerClip: bad }),
      `${bad} should be refused`);
  }
  assert.equal(framePromoteProblem(ok), null);
});

test('the ceiling is named in the refusal, not just enforced', () => {
  const why = framePromoteProblem({ ...ok, framesPerClip: FRAMES_PER_CLIP_MAX + 1 });
  assert.match(why, new RegExp(String(FRAMES_PER_CLIP_MAX)));
});

test('face filtering without a reference dataset is refused in the dialog', () => {
  assert.match(framePromoteProblem({ ...ok, requireFace: true, refDatasetId: null }),
    /reference photo/);
  assert.equal(
    framePromoteProblem({ ...ok, requireFace: true, refDatasetId: 4 }), null);
});

test('the payload omits caps rather than sending null', () => {
  const body = framePromotePayload({ name: 'x', framesPerClip: 2 });
  assert.equal('total_limit' in body, false);
  assert.equal('max_per_source' in body, false);
  assert.equal('ids' in body, false);
  assert.equal('ref_dataset_id' in body, false);
  assert.equal(body.frames_per_clip, 2);
  assert.equal(body.require_face, false);
});

test('a zero cap is treated as "no cap", not as a cap of zero', () => {
  const body = framePromotePayload({ name: 'x', framesPerClip: 1, totalLimit: 0,
    maxPerSource: 0 });
  assert.equal('total_limit' in body, false);
  assert.equal('max_per_source' in body, false);
});

test('the reference is a dataset ID and only rides along with the filter on', () => {
  const off = framePromotePayload({ name: 'x', framesPerClip: 1,
    requireFace: false, refDatasetId: 4 });
  assert.equal('ref_dataset_id' in off, false);
  const on = framePromotePayload({ name: 'x', framesPerClip: 1,
    requireFace: true, refDatasetId: '4' });
  assert.equal(on.ref_dataset_id, 4, 'sent as a number, never a path');
});

test('no request can carry a file path', () => {
  const body = framePromotePayload({ name: 'x', framesPerClip: 1,
    requireFace: true, refDatasetId: 4 });
  assert.equal('refs' in body, false);
});

test('the name is trimmed on the way out', () => {
  assert.equal(framePromotePayload({ name: '  x  ', framesPerClip: 1 }).name, 'x');
});

test('the scope label counts selection first, then the kept clips', () => {
  assert.match(frameScopeLabel(1, 40), /1 selected clip\b/);
  assert.match(frameScopeLabel(3, 40), /3 selected clips/);
  assert.match(frameScopeLabel(0, 40), /all 40 kept clips/);
  assert.match(frameScopeLabel(0, 1), /all 1 kept clip\b/);
});

test('the ceiling is stated as a ceiling, in the words the user needs', () => {
  const hint = frameCeilingHint({ frames_ceiling: 120 });
  assert.match(hint, /120/);
  assert.match(hint, /not a promise/);
});

test('no ceiling means no sentence rather than a zero', () => {
  assert.equal(frameCeilingHint({}), '');
  assert.equal(frameCeilingHint(null), '');
});

test('a run without the face filter says so; a filtered one stays quiet', () => {
  assert.match(frameFaceNote({ face_filtered: false }), /face filter is OFF/);
  assert.equal(frameFaceNote({ face_filtered: true }), '');
});
