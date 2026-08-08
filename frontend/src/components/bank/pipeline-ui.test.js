import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { defaultPipelineStepKeys } from './bankSemanticEngine.js';

const dialog = fs.readFileSync(new URL('./LaunchAllDialog.jsx', import.meta.url), 'utf8');
const report = fs.readFileSync(new URL('./PipelineReport.jsx', import.meta.url), 'utf8');
const ws = fs.readFileSync(new URL('./BankWorkspace.jsx', import.meta.url), 'utf8');

test('the launch dialog posts the three config keys the backend expects', () => {
  assert.match(dialog, /steps:\s*\[\.\.\.steps\]/);
  assert.match(dialog, /reject_flags:\s*autoRejectOn\s*\?\s*\[\.\.\.rejectFlags\]\s*:\s*\[\]/);
  assert.match(dialog, /resolve_dups:\s*autoRejectOn\s*&&\s*resolveDups/);
});

test('the overnight dialog offers no non-verdict flag; the attended button prints its caveat', () => {
  // The unattended funnel must never offer to bulk-reject on a provenance HINT.
  const list = dialog.match(/const QUALITY_FLAGS = \[([^\]]*)\]/);
  assert.ok(list, 'found the dialog flag list');
  assert.doesNotMatch(list[1], /soft_detail/);
  assert.doesNotMatch(list[1], /bars/);
  // The standalone 🧹 Auto-reject still offers them — with the caveat SHOWN,
  // not left in a title= tooltip nobody sees on a phone.
  assert.match(ws, /QUALITY_REJECT_FLAGS = \['blur', 'noise', 'uniform', 'small', 'soft_detail', 'bars'\]/);
  assert.match(ws, /\{FLAG_HINT\[f\] && \(/);
  assert.match(ws, /check before mass-rejecting/);
});

test('captioning is OFF by default; auto-reject defaults to blur+uniform and keep-best dedup', () => {
  const ready = Object.fromEntries([
    'scan', 'auto_reject', 'score', 'semantic_index', 'semantic_dedup',
    'watermark', 'faces', 'framing', 'caption',
  ].map((key) => [key, true]));
  const defaults = defaultPipelineStepKeys('siglip2', ready);
  assert.equal(defaults.includes('caption'), false);
  assert.equal(defaults.includes('scan'), true);
  assert.equal(defaults.includes('auto_reject'), true);
  assert.match(dialog, /defaultPipelineStepKeys\(engine, ready\)/);
  assert.match(dialog, /new Set\(\['blur',\s*'uniform'\]\)/);
  assert.match(dialog, /useState\(true\)/);            // resolveDups defaults on
});

test('a heavy pass whose tool is not ready is auto-unchecked and flagged "will skip"', () => {
  assert.match(dialog, /score:\s*!!caps\?\.bank_scoring/);
  assert.match(dialog, /watermark:\s*!!visionReady/);
  assert.match(dialog, /faces:\s*!!caps\?\.face_scoring/);
  assert.equal(defaultPipelineStepKeys('siglip2', { scan: true, semantic_index: false })
    .includes('semantic_index'), false);
  assert.match(dialog, /will skip/);
});

test('Launch all receives the Bank engine and only offers the SigLIP2 index step there', () => {
  assert.match(ws, /<LaunchAllDialog[\s\S]*?semanticEngine=\{semanticState\.engine\}/);
  assert.match(dialog, /pipelineStepKeys\(engine\)/);
  assert.match(dialog, /key:\s*'semantic_index'/);
});

test('the progress bar understands the pipeline kind (step X/N + per-step chips)', () => {
  assert.match(ws, /kind === 'pipeline' \? activity\.pipeline/);
  assert.match(ws, /step \$\{\(pipe\.index \?\? 0\) \+ 1\}\/\$\{pipe\.total_steps\}/);
  assert.match(ws, /pipe\.results\.map/);
});

test('the report renders per-step status and is fed from the persisted payload field', () => {
  assert.match(report, /STATUS_STYLE/);
  assert.match(report, /skipped/);
  assert.match(report, /cancelled/);
  assert.match(report, /error/);
  // The workspace shows it only when idle, from the persisted field.
  assert.match(ws, /payload\.pipeline_report/);
  assert.match(ws, /<PipelineReport/);
});

test('launching posts to the pipeline endpoint', () => {
  assert.match(ws, /\/api\/bank\/\$\{bankId\}\/pipeline/);
});
