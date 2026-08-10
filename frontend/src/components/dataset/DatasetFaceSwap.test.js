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
  assert.match(grid, /onFaceSwap=\{onFaceSwap\} swapBusy=\{Boolean\(swappingIds\?\.has\(img\.id\)\)\}/);
  assert.match(grid, /hasRef=\{hasRef\}/);
});

test('grid item renders a gated face-swap button', () => {
  assert.match(gridItem, /const canFaceSwap = hasRef && !!img\.filename;/);
  assert.match(gridItem, /\{canFaceSwap && onFaceSwap && \(/);
  assert.match(gridItem, /e\.stopPropagation\(\); onFaceSwap\(img\.id\)/);
  assert.match(gridItem, /Swap this tile's face with the reference image/);
  assert.match(gridItem, /🎭↔/);
});

/* THE BUTTON MUST GO BUSY WHILE THE REQUEST IS OUT.
   On the app-mask lane the server spends ~20 s masking before the row changes
   at all, so "disabled once the tile is pending" is far too late: the button
   still looks clickable through the whole wait, and it gets clicked — one tile,
   three swaps queued. The hook holds a synchronous ref so the second click is
   dropped before React has even re-rendered. */
test('the swap button is disabled while its own request is in flight', () => {
  assert.match(gridItem, /swapBusy = false/);
  assert.match(gridItem, /disabled=\{busy \|\| swapBusy\}/);
  assert.match(gridItem, /aria-busy=\{swapBusy\}/);
  // Both labels stay MOUNTED (Chrome auto-translate rewrites text nodes; a
  // ternary swap throws NotFoundError) — see CLAUDE.md.
  assert.match(gridItem, /<span hidden=\{swapBusy\}>🎭↔<\/span>/);
  assert.match(gridItem, /<span hidden=\{!swapBusy\}/);
});

test('the hook drops a second click before React can re-render', () => {
  const hook = readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');
  assert.match(hook, /const swappingRef = useRef\(new Set\(\)\)/);
  assert.match(hook, /if \(swappingRef\.current\.has\(imageId\)\) return \{ ok: false/);
  // ...and it is released whatever happened, or the tile stays dead forever.
  assert.match(hook, /swappingRef\.current\.delete\(imageId\)/);
});
