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

test('imports LOCAL_ENGINES from engineSelection.js instead of hardcoding the local-engine list again', () => {
  assert.match(src, /import\s*\{[^}]*LOCAL_ENGINES[^}]*\}\s*from\s*['"]\.\/engineSelection\.js['"]/);
  // The array literal itself must not be re-typed anywhere in this file.
  assert.doesNotMatch(src, /\[\s*['"]klein['"]\s*,\s*['"]krea['"]\s*\]/);
});

test('an NSFW ratio slider exists, gated behind the nsfwMode prop', () => {
  assert.match(src, /nsfwRatio/);
  assert.match(src, /const \[nsfwRatio, setNsfwRatio\] = useState\(0\)/);
  // The slider JSX itself must be conditioned on nsfwMode, not just present
  // somewhere in the file — this proves the gate wraps the actual control.
  const nsfwSection = src.slice(src.indexOf('{nsfwMode && ('), src.indexOf('Dialog-owned engine picker'));
  assert.match(nsfwSection, /type="range"/);
  assert.match(nsfwSection, /setNsfwRatio/);
});

test('the dialog owns its engine selection (dialogEngines), seeded from the engines prop, not the raw prop, for compose/generate', () => {
  assert.match(src, /const \[dialogEngines, setDialogEngines\] = useState\(\(\) => \[\.\.\.engines\]\)/);
  const submitBody = src.slice(src.indexOf('const submit = async () => {'), src.indexOf('const activeFramings'));
  assert.match(submitBody, /engineBatches\(variations,\s*dialogEngines,\s*engineMode\)/,
    'engineBatches must be called with the dialog-owned selection, not the raw engines prop');
  assert.doesNotMatch(submitBody, /engineBatches\(variations,\s*engines,\s*engineMode\)/,
    'using the raw engines prop directly would defeat the point of a dialog-owned, NSFW-filterable selection');
});

test('nsfw_ratio is threaded through to the quickGenerateCompose payload', () => {
  const submitBody = src.slice(src.indexOf('const submit = async () => {'), src.indexOf('const activeFramings'));
  assert.match(submitBody, /quickGenerateCompose\(\{[\s\S]*nsfw_ratio:\s*nsfwRatio/);
});

test('non-local engines are excluded from the offered picker while the NSFW ratio is active', () => {
  const offeredBody = src.slice(src.indexOf('const offeredEngines'), src.indexOf('const submit = async'));
  assert.match(offeredBody, /LOCAL_ENGINES\.includes\(name\)/,
    'the offered-engines computation must gate on LOCAL_ENGINES when nsfwRatio > 0');
  assert.match(offeredBody, /nsfwRatio\s*<=\s*0\s*\|\|\s*LOCAL_ENGINES\.includes\(name\)/);
});

test('an empty local-engine selection while NSFW is active is surfaced, not silently submitted', () => {
  assert.match(src, /noLocalEngineForNsfw/);
  assert.match(src, /nsfwMode\s*&&\s*nsfwRatio\s*>\s*0\s*&&\s*dialogEngines\.length\s*===\s*0/);
  assert.match(src, /dialogEngines\.length\s*===\s*0/);  // also folded into the Generate `disabled` check
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
