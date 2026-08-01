import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const src = readFileSync(new URL('./VariationCatalog.jsx', import.meta.url), 'utf8');

test('VariationCatalog renders a Quick generate entry point next to the presets', () => {
  assert.match(src, /QuickGenerateDialog/);
  assert.match(src, /🎲/);
});

test('the dialog receives the same engine/generate wiring the manual Generate button uses', () => {
  assert.match(src, /quickGenerateCompose=\{/);
  assert.match(src, /onGenerate=\{onGenerate\}/);
});

test('the Quick generate button click opens the dialog', () => {
  assert.match(src, /onClick=\{\(\) => setQuickGenOpen\(true\)\}/);
});

test('the dialog mount is guarded by quickGenOpen, so its on-mount pool fetch never fires while closed', () => {
  // Specific to the QuickGenerateDialog mount (not just any `quickGenOpen` use
  // in the file): the guard must wrap the JSX that renders the dialog.
  assert.match(src, /\{quickGenOpen\s*&&\s*\(\s*<QuickGenerateDialog/);
});
