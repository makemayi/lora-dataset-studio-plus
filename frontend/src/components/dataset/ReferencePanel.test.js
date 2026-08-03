import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const src = readFileSync(new URL('./ReferencePanel.jsx', import.meta.url), 'utf8');

test('ReferencePanel renders PoseSlotPanel alongside the extra-ref row', () => {
  assert.match(src, /import PoseSlotPanel from '\.\/PoseSlotPanel'/);
  assert.match(src, /<PoseSlotPanel/);
});

test('pose slot props are threaded through from ReferencePanel props', () => {
  assert.match(src, /poseSlots/);
  assert.match(src, /onSetPoseSlot/);
});
