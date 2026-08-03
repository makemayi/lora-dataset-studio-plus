import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const src = readFileSync(new URL('./ReferencePanel.jsx', import.meta.url), 'utf8');
const workspaceSrc = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');

test('ReferencePanel renders PoseSlotPanel alongside the extra-ref row', () => {
  assert.match(src, /import PoseSlotPanel from '\.\/PoseSlotPanel'/);
  assert.match(src, /<PoseSlotPanel/);
});

test('pose slot props are threaded through from ReferencePanel props', () => {
  assert.match(src, /poseSlots/);
  assert.match(src, /onSetPoseSlot/);
});

test('all six pose slot props round-trip by name through ReferencePanel', () => {
  for (const name of ['poseSlots', 'onSetPoseSlot', 'onCropPoseSlot', 'onMirrorPoseSlot',
                      'onTogglePoseSlotEnabled', 'onRemovePoseSlot']) {
    assert.match(src, new RegExp(`\\b${name}\\b`), `${name} must appear in ReferencePanel.jsx`);
  }
});

test('DatasetWorkspace wires pose slot props to the useDataset hook and a dedicated crop modal', () => {
  assert.match(workspaceSrc, /poseSlots=\{d\.pose_slots/);
  assert.match(workspaceSrc, /onSetPoseSlot=\{ds\.setPoseSlot\}/);
  assert.match(workspaceSrc, /const \[poseSlotCrop, setPoseSlotCrop\] = useState/);
  assert.match(workspaceSrc, /poseSlotCrop && d\.pose_slots\?\.\[poseSlotCrop\]\?\.filename/);
});

test('the pose slot crop modal uses the square defaultAspect, like the primary reference', () => {
  const modalStart = workspaceSrc.indexOf('poseSlotCrop && d.pose_slots');
  const modalBlock = workspaceSrc.slice(modalStart, modalStart + 500);
  assert.match(modalBlock, /defaultAspect=\{1\}/);
  assert.match(modalBlock, /original_filename \|\| .*\.filename/);
  assert.match(modalBlock, /ds\.cropPoseSlot\(poseSlotCrop, box\)/);
});

test('hovering the primary ref or extra-refs row and pressing Ctrl+V pastes an image', () => {
  assert.match(src, /import \{ imageFromClipboard \} from '\.\/clipboardImage'/);
  assert.match(src, /addEventListener\('paste'/);
  assert.match(src, /refHover\.current = true/);
  assert.match(src, /extraHover\.current = true/);
  assert.match(src, /onSetRef\(file, \{ autoCrop/);
  assert.match(src, /onAddExtraRef\?\.\(file\)/);
});
