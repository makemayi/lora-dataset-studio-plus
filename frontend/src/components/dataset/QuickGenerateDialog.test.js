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

test('submit() re-entry guard blocks a second call while one is in flight (no duplicate job submission)', () => {
  const submitBody = src.slice(src.indexOf('const submit = async () => {'), src.indexOf('const activeFramings'));
  assert.match(submitBody, /if\s*\(\s*submitting\s*\)\s*return\s*;/,
    'submit() must bail out immediately when a submission is already in flight');
  // The guard has to run BEFORE setSubmitting(true), otherwise it can never fire.
  const guardIdx = submitBody.search(/if\s*\(\s*submitting\s*\)\s*return\s*;/);
  const setSubmittingIdx = submitBody.indexOf('setSubmitting(true)');
  assert.ok(guardIdx !== -1 && setSubmittingIdx !== -1 && guardIdx < setSubmittingIdx,
    'the re-entry guard must precede setSubmitting(true)');
});

test('normalizeTo100 uses a floor + largest-remainder allocation, which cannot go negative', () => {
  const fnBody = src.slice(src.indexOf('function normalizeTo100'), src.indexOf('export default function'));
  // Floors are non-negative by construction (Math.floor of a non-negative
  // exact share); the old bug came from per-key Math.round() plus an
  // unclamped final drift correction that could push the last key negative.
  assert.match(fnBody, /Math\.floor/, 'shares must be built from a floor, not round+drift');
  assert.doesNotMatch(fnBody, /\+=\s*drift/, 'must not reintroduce the unclamped drift-correction line');
  // The changed key itself must be clamped into [0, 100] rather than trusted
  // as-is, and every "other" share must come from a non-negative floor array
  // that only ever has +1 applied to it (never assigned a raw, unclamped
  // value), so no key in the returned object can be negative.
  assert.match(fnBody, /Math\.max\(0,\s*Math\.min\(100,\s*changedValue\)\)/);
  assert.match(fnBody, /shares\[order\[i\]\[1\]\]\s*\+=\s*1/);
});
