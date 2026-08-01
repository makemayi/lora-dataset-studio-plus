import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const src = readFileSync(new URL('./QuickGenerateDialog.jsx', import.meta.url), 'utf8');

test('dialog is a controlled modal with an onResolve(payload|null) contract', () => {
  assert.match(src, /role="dialog"/);
  assert.match(src, /aria-modal="true"/);
  assert.match(src, /onResolve/);
});

test('total count input is capped at the same 200 the backend enforces', () => {
  assert.match(src, /max=["']200["']/);
});

test('framing sliders cover face, bust and body only (back excluded, per spec)', () => {
  assert.match(src, /face/);
  assert.match(src, /bust/);
  assert.match(src, /\bbody\b/);
  assert.doesNotMatch(src, /framing.*back|back.*framing/i);
});

test('submit composes then generates, in that order, with no preview step', () => {
  const composeIdx = src.indexOf('quickGenerateCompose');
  const generateIdx = src.indexOf('onGenerate(');
  assert.ok(composeIdx !== -1 && generateIdx !== -1);
  assert.ok(composeIdx < generateIdx, 'compose must be called before generate');
});
