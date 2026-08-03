import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const src = readFileSync(new URL('./useDataset.js', import.meta.url), 'utf8');

test('setPoseSlot posts multipart form data to /ref/pose/<poseKey>', () => {
  assert.match(src, /const setPoseSlot = useCallback/);
  assert.match(src, /\/ref\/pose\/\$\{poseKey\}/);
});

test('cropPoseSlot and mirrorPoseSlot and togglePoseSlotEnabled and removePoseSlot all exist', () => {
  assert.match(src, /const cropPoseSlot = useCallback/);
  assert.match(src, /const mirrorPoseSlot = useCallback/);
  assert.match(src, /const togglePoseSlotEnabled = useCallback/);
  assert.match(src, /const removePoseSlot = useCallback/);
});

test('mirrorPoseSlot and removePoseSlot hit their own /mirror and /delete routes', () => {
  assert.match(src, /\/ref\/pose\/\$\{poseKey\}\/mirror/);
  assert.match(src, /\/ref\/pose\/\$\{poseKey\}\/delete/);
});

test('togglePoseSlotEnabled posts {enabled} to the /enable route', () => {
  const body = src.slice(src.indexOf('const togglePoseSlotEnabled'),
                         src.indexOf('const togglePoseSlotEnabled') + 400);
  assert.match(body, /\/ref\/pose\/\$\{poseKey\}\/enable/);
  assert.match(body, /enabled/);
});

test('all five pose slot actions are exported from the hook', () => {
  const exportsIdx = src.lastIndexOf('return {');
  const exportsBlock = src.slice(exportsIdx);
  for (const name of ['setPoseSlot', 'cropPoseSlot', 'mirrorPoseSlot',
                      'togglePoseSlotEnabled', 'removePoseSlot']) {
    assert.match(exportsBlock, new RegExp(`\\b${name}\\b`), `${name} must be exported`);
  }
});
