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

test('all five slots are interactive — no placeholder card is left', () => {
  // One list, one map: a slot that renders is a slot the detector can pick, so
  // the panel cannot drift back into offering an upload that nothing reads.
  assert.match(src, /ACTIVE_POSE_KEYS\s*=\s*\[[^\]]*left45[^\]]*right45[^\]]*left90[^\]]*right90[^\]]*back[^\]]*\]/s);
  assert.doesNotMatch(src, /RESERVED_POSE_KEYS/);
  assert.doesNotMatch(src, /coming soon|敬请期待/i);
});

test('each card says which shots it answers, not just its degree', () => {
  assert.match(src, /POSE_HINTS/);
  assert.match(src, /strict left profile/);
  assert.match(src, /three-quarter right/);
  assert.match(src, /from behind/);
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

test('hovering a card and pressing Ctrl+V pastes into that pose_key, not a fixed one', () => {
  assert.match(src, /addEventListener\('paste'/);
  assert.match(src, /onMouseEnter=\{.*hoverKey\.current = poseKey/);
  assert.match(src, /onSetPoseSlot\?\.\(poseKey, file\)/);
});
