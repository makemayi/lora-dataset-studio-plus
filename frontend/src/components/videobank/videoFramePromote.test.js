import test from 'node:test';
import assert from 'node:assert/strict';
import {
  framePromoteProblem, framePromotePayload, frameScopeLabel,
  frameCeilingHint, frameFaceNote, FRAMES_PER_CLIP_MAX,
} from './videoFramePromote.js';

const ok = { name: 'faces', framesPerClip: 3, personMode: 'none', refDatasetId: null };

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

test('identity without a reference dataset is refused; person does not need one', () => {
  assert.match(framePromoteProblem({ ...ok, personMode: 'identity', refDatasetId: null }),
    /reference photo/);
  assert.equal(
    framePromoteProblem({ ...ok, personMode: 'identity', refDatasetId: 4 }), null);
  assert.equal(
    framePromoteProblem({ ...ok, personMode: 'person', refDatasetId: null }), null);
});

test('the payload omits caps rather than sending null', () => {
  const body = framePromotePayload({ name: 'x', framesPerClip: 2 });
  assert.equal('total_limit' in body, false);
  assert.equal('max_per_source' in body, false);
  assert.equal('ids' in body, false);
  assert.equal('ref_dataset_id' in body, false);
  assert.equal(body.frames_per_clip, 2);
  assert.equal(body.person_mode, 'identity');
  assert.equal(body.sharp_tolerance, 0.6);
});

test('a zero cap is treated as "no cap", not as a cap of zero', () => {
  const body = framePromotePayload({ name: 'x', framesPerClip: 1, totalLimit: 0,
    maxPerSource: 0 });
  assert.equal('total_limit' in body, false);
  assert.equal('max_per_source' in body, false);
});

test('the reference is a dataset ID and only rides along with the filter on', () => {
  const off = framePromotePayload({ name: 'x', framesPerClip: 1,
    personMode: 'person', refDatasetId: 4 });
  assert.equal('ref_dataset_id' in off, false);
  const on = framePromotePayload({ name: 'x', framesPerClip: 1,
    personMode: 'identity', refDatasetId: '4' });
  assert.equal(on.ref_dataset_id, 4, 'sent as a number, never a path');
});

test('no request can carry a file path', () => {
  const body = framePromotePayload({ name: 'x', framesPerClip: 1,
    personMode: 'identity', refDatasetId: 4 });
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

test('the payload carries person_mode and the two tolerance dials', () => {
  const body = framePromotePayload({ name: 'n', framesPerClip: 3, personMode: 'person',
    sharpTolerance: 0.65, faceTolerance: 0.6, refDatasetId: '' });
  assert.equal(body.person_mode, 'person');
  assert.equal(body.sharp_tolerance, 0.65);
  assert.equal(body.face_tolerance, 0.6);
  assert.equal('require_face' in body, false);
});

test('tolerances default to 0.6 when absent', () => {
  const body = framePromotePayload({ name: 'n', framesPerClip: 3, personMode: 'none' });
  assert.equal(body.sharp_tolerance, 0.6);
  assert.equal(body.face_tolerance, 0.6);
});
