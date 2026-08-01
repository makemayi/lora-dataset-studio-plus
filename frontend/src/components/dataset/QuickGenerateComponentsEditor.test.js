import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const src = readFileSync(new URL('./QuickGenerateComponentsEditor.jsx', import.meta.url), 'utf8');

test('a JSON editor for quick-generate custom components exists and uses the hook actions', () => {
  assert.match(src, /saveQuickGenerateCustomComponents/);
  assert.match(src, /quickGenerateComponents/);
});

test('the editor loads the round-tripped custom entries, not a blind blank object', () => {
  // Regression guard for the data-loss bug: the initial load must read
  // d.custom (what the GET route now returns for round-tripping), never
  // unconditionally reset to '{}' regardless of what was previously saved.
  assert.match(src, /d\.custom/);
});

const catalogSrc = readFileSync(
  new URL('./VariationCatalog.jsx', import.meta.url), 'utf8');

test('the editor is mounted near the shot-catalog import UI in VariationCatalog', () => {
  assert.match(catalogSrc, /QuickGenerateComponentsEditor/);
});
