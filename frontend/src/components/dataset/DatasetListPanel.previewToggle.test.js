import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const panel = readFileSync(new URL('./DatasetListPanel.jsx', import.meta.url), 'utf8');

test('library preview visibility is persistent and suppresses image mounts', () => {
  assert.match(panel, /const PREVIEWS_VISIBLE_KEY = 'datasetLibraryPreviewsVisible';/);
  assert.match(panel, /localStorage\.getItem\(PREVIEWS_VISIBLE_KEY\) !== '0'/);
  assert.match(panel, /localStorage\.setItem\(PREVIEWS_VISIBLE_KEY, showPreviews \? '1' : '0'\)/);
  assert.match(panel, /aria-pressed=\{showPreviews\}/);
  assert.match(panel, /title=\{showPreviews \? 'Hide image previews' : 'Show image previews'\}/);

  const tile = panel.slice(panel.indexOf('function DatasetTile'), panel.indexOf('/** Compact row'));
  const row = panel.slice(panel.indexOf('function DatasetRow'), panel.indexOf('/** The creation form'));
  const imageGuard = /showPreviews && d\.ref_filename \?\s*\(\s*<img/s;

  assert.match(tile, imageGuard);
  assert.match(row, imageGuard);
  assert.equal((panel.match(/showPreviews=\{showPreviews\}/g) || []).length, 2);
});
