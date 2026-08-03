import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const src = readFileSync(new URL('./PoseSlotPanel.jsx', import.meta.url), 'utf8');

test('renders one card per POSE_SLOT_KEYS entry, in a fixed order', () => {
  assert.match(src, /left45/);
  assert.match(src, /right45/);
  assert.match(src, /\bback\b/);
  assert.match(src, /left90/);
  assert.match(src, /right90/);
});

test('only left45 and right45 are interactive — the other three are disabled placeholders', () => {
  assert.match(src, /ACTIVE_POSE_KEYS|left45.*right45/);
  assert.match(src, /coming soon|Coming soon|敬请期待/i);
});

test('each active card offers upload, crop, mirror and an enabled checkbox', () => {
  assert.match(src, /onUploadPoseSlot|onSetPoseSlot/);
  assert.match(src, /onCropPoseSlot/);
  assert.match(src, /onMirrorPoseSlot/);
  assert.match(src, /type="checkbox"/);
});

test('the mirror action is labelled as a 180-degree horizontal flip', () => {
  assert.match(src, /180|mirror|flip/i);
});
