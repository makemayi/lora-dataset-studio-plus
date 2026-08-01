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
