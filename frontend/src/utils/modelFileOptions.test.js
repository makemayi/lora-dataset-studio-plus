/**
 * The list behind the model-file pickers, and the one case a plain dropdown gets
 * wrong: a value pinned in the config that is NOT on disk.
 *
 * These assertions are about a DECISION, not a shape. "Pinned but absent stays
 * first, stays selected, and the engine refuses" is the rule that exists because
 * a silent fallback once trained a LoRA on a third-party finetune — so each test
 * here names the behaviour it would take to reintroduce that.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  sameModelRef, isScanned, buildModelOptions, filterModelOptions,
  emptyScanMessage, pinnedModelGapReason, PIN_SLOT_LABELS,
} from './modelFileOptions.js';

const FILES = ['klein/flux-2-klein-9b-fp8.safetensors', 'Krea/krea2_turbo_fp8_scaled.safetensors'];

test('a name matches whatever separator and case the user pasted', () => {
  assert.ok(sameModelRef('Krea\\krea2_turbo_fp8_scaled.safetensors',
    'krea/KREA2_TURBO_FP8_SCALED.safetensors'));
  assert.ok(!sameModelRef('', ''));
  assert.ok(isScanned('KLEIN/FLUX-2-KLEIN-9B-FP8.safetensors', FILES));
});

test('a bare filename an existing install typed is NOT accused of being missing', () => {
  // The setting this replaces has always resolved a Krea base by BASENAME, so
  // this value works today. Badging it "not found" on the day the picker ships
  // is the picker crying wolf at a correct install.
  assert.ok(isScanned('krea2_turbo_fp8_scaled.safetensors', FILES));
  assert.equal(
    buildModelOptions('krea2_turbo_fp8_scaled.safetensors', FILES).pinnedMissing, false);
});

test('a pinned file that is NOT on disk stays first, stays flagged, is never dropped', () => {
  const { options, pinnedMissing } = buildModelOptions('krea/my-own-build.safetensors', FILES);
  assert.equal(pinnedMissing, true);
  assert.equal(options[0].name, 'krea/my-own-build.safetensors',
    'the pinned value must open the list — a list that opens on something else '
    + 'invites picking another file by accident');
  assert.equal(options[0].missing, true);
  assert.equal(options.length, FILES.length + 1, 'no scanned file was dropped');
});

test('a value that IS on disk adds nothing and flags nothing', () => {
  const { options, pinnedMissing } = buildModelOptions(FILES[1], FILES);
  assert.equal(pinnedMissing, false);
  assert.equal(options.length, FILES.length);
});

test('an empty or still-loading scan never cries "not found"', () => {
  // ComfyUI down, unconfigured, or a slow mount still answering: telling someone
  // their file is missing here sends them to re-download what they already have.
  assert.equal(buildModelOptions('x.safetensors', []).pinnedMissing, false);
  assert.equal(buildModelOptions('x.safetensors', FILES, { loading: true }).pinnedMissing, false);
});

test('typing filters, but opening a filled field shows the whole folder', () => {
  const { options } = buildModelOptions(FILES[0], FILES);
  assert.equal(filterModelOptions(options, null, FILES[0]).length, 2);
  assert.equal(filterModelOptions(options, FILES[0], FILES[0]).length, 2,
    'the query equals the current value — that is not a search');
  assert.equal(filterModelOptions(options, 'turbo', FILES[0]).length, 2,
    'the current value is always kept reachable');
  assert.equal(filterModelOptions(options, 'turbo', 'nothing').length, 1);
});

test('every empty state names a DIFFERENT next action', () => {
  const hint = 'ComfyUI’s models/vae';
  const loading = emptyScanMessage({ loading: true, error: false, count: 0, folderHint: hint });
  const down = emptyScanMessage({ loading: false, error: true, count: 0, folderHint: hint });
  const empty = emptyScanMessage({ loading: false, error: false, count: 0, folderHint: hint });
  assert.ok(loading && down && empty);
  assert.notEqual(down, empty, 'ComfyUI unreachable and an empty folder are different problems');
  assert.ok(empty.includes(hint), 'the empty state must say WHERE to put the file');
  assert.equal(emptyScanMessage({ loading: false, error: false, count: 3, folderHint: hint }), null);
});

test('the refusal names the FILE the user chose, not just the slot', () => {
  const msg = pinnedModelGapReason('Krea 2 Edit', [
    { slot: 'base_model', configured: 'krea/my-own-build.safetensors' },
  ]);
  assert.ok(msg.includes('my-own-build.safetensors'),
    '"base model missing" would send the user to download a model they already have');
  assert.ok(msg.includes(PIN_SLOT_LABELS.base_model));
  assert.ok(/auto-detection/.test(msg), 'the way out has to be in the sentence');
  assert.equal(pinnedModelGapReason('Klein', []), null);
  assert.equal(pinnedModelGapReason('Klein', undefined), null);
});
