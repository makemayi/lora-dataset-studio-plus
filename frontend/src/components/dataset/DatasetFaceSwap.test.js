import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const hook = readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');
const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');
const grid = readFileSync(new URL('./DatasetGrid.jsx', import.meta.url), 'utf8');
const gridItem = readFileSync(new URL('./DatasetGridItem.jsx', import.meta.url), 'utf8');

test('dataset hook exposes a face-swap action posting to /face-swap', () => {
  assert.ok(hook.includes('`/api/dataset/image/${imageId}/face-swap`'));
  const start = hook.indexOf('const faceSwapImage = useCallback');
  assert.ok(start !== -1, 'faceSwapImage not defined');
  assert.match(hook, /regenerate, faceSwapImage, analyzeFaces, scoreFace/);
});

test('workspace passes hasRef and the face-swap action into the grid', () => {
  assert.match(workspace, /onFaceSwap=\{ds\.faceSwapImage\}/);
  assert.match(workspace, /hasRef=\{!!d\.ref_filename\}/);
});

test('grid forwards onFaceSwap/hasRef down to each tile', () => {
  assert.match(grid, /onFaceSwap, hasRef = false/);
  assert.match(grid, /onFaceSwap=\{onFaceSwap\} hasRef=\{hasRef\}/);
});

test('grid item renders a gated face-swap button', () => {
  assert.match(gridItem, /const canFaceSwap = hasRef && !!img\.filename;/);
  assert.match(gridItem, /\{canFaceSwap && onFaceSwap && \(/);
  assert.match(gridItem, /e\.stopPropagation\(\); onFaceSwap\(img\.id\)/);
  assert.match(gridItem, /Swap this tile's face with the reference image/);
  assert.match(gridItem, /🎭↔/);
});
