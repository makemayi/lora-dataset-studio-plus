import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  denseActions, denseFileRows, denseGuidanceLine, denseHubLine, denseModelTitle,
  denseStudioTarget, denseWhereChip, fmtBytes, STUDIO_NEEDS_A_LORA,
} from './denseModels.js';

const local = (over = {}) => ({
  run_id: 146, dataset_id: 3, train_type: 'krea', variant: 'Raw', steps: 3000,
  active: false, can_quantize: false, can_send_to_comfyui: true, can_delete: true,
  master: {
    filename: 'Krea_full_x.safetensors', path: '/store/run_146/Krea_full_x.safetensors',
    size_bytes: 26e9, step: null, is_final: true, total_candidates: 1, others: [],
  },
  fp8: {
    filename: 'Krea_full_x_fp8.safetensors', size_bytes: 13e9,
    in_comfyui: false, comfyui_name: null,
  },
  hub: { repo_id: 'acme/dense-146', url: 'https://huggingface.co/acme/dense-146',
    status: 'available' },
  inference_hint: { guidance_scale: 4, steps: 25 },
  ...over,
});

const hubOnly = (over = {}) => ({
  run_id: 90, dataset_id: 3, train_type: 'krea', variant: 'Raw', active: false,
  master: null, fp8: null, can_quantize: false,
  hub: { repo_id: 'acme/dense-90', url: 'https://huggingface.co/acme/dense-90',
    status: 'available', weight_filename: 'Krea_full_y.safetensors' },
  inference_hint: { guidance_scale: 4, steps: 25 },
  ...over,
});

// --- identity ---------------------------------------------------------------

test('the title names the family AND the variant — Raw and Turbo want different settings', () => {
  assert.equal(denseModelTitle(local()), 'Krea 2 · Raw — run #146');
  assert.equal(denseModelTitle({ train_type: 'krea' }), 'Krea 2');
  assert.equal(denseModelTitle(null), 'Full model');
});

test('the where-chip mirrors the canvas vocabulary, never "missing" for a Hub model', () => {
  assert.equal(denseWhereChip(local()).label, 'On this computer');
  assert.equal(denseWhereChip({}).label, 'Not found');
  // A run holding only the twin is still "on this computer".
  assert.equal(denseWhereChip({ fp8: { filename: 'a' } }).label, 'On this computer');
  // A Hub model is still never "missing" — but "On Hugging Face" is a claim
  // about NOW, and nothing has asked. What the run did is what may be stated.
  assert.equal(denseWhereChip(hubOnly()).label, 'Delivered to Hugging Face');
  assert.match(denseWhereChip(hubOnly()).title, /has not been checked/);
});

test('the where-chip speaks in the present tense only once something has asked', () => {
  assert.equal(denseWhereChip(hubOnly(), { state: 'present' }).label, 'On Hugging Face');
  assert.equal(denseWhereChip(hubOnly(), { state: 'gone' }).label, 'No copy left');
  // "Could not check" is our failure, not the repository's. It must read like
  // the unchecked state, never like the absent one.
  assert.equal(denseWhereChip(hubOnly(), { state: 'unknown' }).label,
    'Delivered to Hugging Face');
  // A model on disk answers before the Hub is consulted at all.
  assert.equal(denseWhereChip(local(), { state: 'gone' }).label, 'On this computer');
});

// --- the distinction this panel exists to make -------------------------------

test('the fp8 twin comes FIRST and is named as what ComfyUI loads', () => {
  const rows = denseFileRows(local());
  assert.deepEqual(rows.map((r) => r.kind), ['fp8', 'master']);
  assert.match(rows[0].role, /ComfyUI loads/);
});

test('the master says it is for re-training and that it is never sent to ComfyUI', () => {
  const master = denseFileRows(local()).find((r) => r.kind === 'master');
  assert.match(master.role, /train again or resume from/);
  assert.match(master.role, /never sent\s+to ComfyUI/);
  assert.equal(master.stateLabel, 'Keep to re-train');
});

test('a run that saved several 26 GB files says which one it takes, and over what', () => {
  const rows = denseFileRows(local({
    master: { filename: 'Krea_x_000002750.safetensors', size_bytes: 26e9,
      step: 2750, is_final: false, total_candidates: 3, others: ['a', 'b'] },
  }));
  assert.match(rows.find((r) => r.kind === 'master').choice,
    /the step 2750 checkpoint, chosen over 2 other checkpoints/);
});

test('one file only means one row, and no rows at all for a Hub-only run', () => {
  assert.equal(denseFileRows(local({ master: null })).length, 1);
  assert.equal(denseFileRows(hubOnly()).length, 0);
});

test('the fp8 row has THREE states, because "delivered" is not "loadable"', () => {
  const off = denseFileRows(local())[0];
  const on = denseFileRows(local({
    fp8: { filename: 'f', size_bytes: 1, in_comfyui: true, delivered: true,
      comfyui_name: 'krea\\f' },
  }))[0];
  // The install with no ComfyUI configured: the app put the file in its OWN
  // models folder and said so. Calling that "not there yet" would offer a Send
  // that has already happened, and the click would answer "already there".
  const fallback = denseFileRows(local({
    fp8: { filename: 'f', size_bytes: 1, in_comfyui: false, delivered: true,
      comfyui_name: null },
  }))[0];
  assert.equal(off.stateLabel, 'Not in ComfyUI yet');
  assert.equal(on.stateLabel, '✓ In ComfyUI');
  assert.equal(fallback.stateLabel, 'Delivered — move it into ComfyUI');
  assert.equal(fallback.state, 'delivered');
});

// --- what the buttons may do -------------------------------------------------

test('sending is offered only for a twin that has not been delivered yet', () => {
  assert.ok(denseActions(local()).send);
  assert.equal(denseActions(local({
    fp8: { filename: 'f', in_comfyui: true, delivered: true, comfyui_name: 'k\\f' },
  })).send, null);
  // Delivered to the app's own folder counts: otherwise the button never clears
  // on an install with no ComfyUI configured.
  assert.equal(denseActions(local({
    fp8: { filename: 'f', in_comfyui: false, delivered: true, comfyui_name: null },
  })).send, null);
  assert.equal(denseActions(local({ fp8: null })).send, null);
});

test('an active run offers nothing and says why', () => {
  const a = denseActions(local({ active: true, can_quantize: true }));
  assert.equal(a.send, null);
  assert.equal(a.quantize, null);
  assert.match(a.activeNote, /still working/);
});

test('a Hub-only master can still be quantized — the job downloads it first', () => {
  assert.ok(denseActions(hubOnly()).quantize.enabled);
  // "Could not check" must not take the button away: the repository is very
  // probably fine and the user is merely offline.
  assert.equal(denseActions(hubOnly(), { state: 'unknown' }).quantize.enabled, true);
});

test('quantizing is refused, with its reason, once the repository is measured gone', () => {
  const gone = denseActions(hubOnly(), { state: 'gone' }).quantize;
  // Still offered as a control, so the reason has somewhere to live — the click
  // is what is refused. "Quantizing fetches it first" cannot be honoured here.
  assert.equal(gone.enabled, false);
  assert.match(gone.reason, /repository it would be downloaded from is gone/);
  // A master on THIS computer never depended on the Hub.
  assert.equal(denseActions(local({ fp8: null, can_quantize: true }),
    { state: 'gone' }).quantize.enabled, true);
});

test('a run with a twin already is not offered a second quantization', () => {
  assert.equal(denseActions(local()).quantize, null);
});

// --- the Hugging Face line: one sentence, chosen from the state --------------
//
// The bug this section pins down shipped for weeks and was green the whole
// time: the status came from `entry.hub.status` and the sentence came from
// `rows.length`, so a run whose repository the backend had VERIFIED empty
// printed "· missing — the model is there, not on this computer". Two
// assertions on one line, and nothing that ever compared them. Every test here
// therefore asserts the SENTENCE against the state, never one or the other.

test('a verified-missing repository can never be followed by "the model is there"', () => {
  for (const status of ['missing', 'verification_pending', 'pending', 'not_requested', '']) {
    for (const presence of [null, { state: 'unknown' }, { state: 'gone' }]) {
      const line = denseHubLine(hubOnly({ hub: { repo_id: 'acme/x', status } }), presence);
      assert.doesNotMatch(line.text, /the model is there|still holds/,
        `status=${status} presence=${presence?.state || 'unchecked'} asserted a presence`);
    }
  }
});

test('with nothing asked yet, the line dates the record instead of stating the present', () => {
  const line = denseHubLine(hubOnly({
    hub: { repo_id: 'acme/dense-90', status: 'available', checked_at: '2026-07-11T09:12:33' },
  }));
  assert.equal(line.state, 'unchecked');
  assert.equal(line.stateLabel, 'not re-checked');
  assert.match(line.text, /Delivered and verified on 2026-07-11 — not re-checked since\./);
  assert.match(line.text, /Nothing from this run is on this computer\./);
  // Undated delivery still reads as a past event, never as a current fact.
  assert.match(denseHubLine(hubOnly()).text, /at the end of the run — not re-checked since/);
});

test('only a live answer earns the present tense', () => {
  const present = denseHubLine(hubOnly(), { state: 'present' });
  assert.equal(present.stateLabel, 'checked just now');
  assert.match(present.text, /still holds this model/);
  assert.match(present.text, /quantizing downloads it from there first/);
  assert.equal(present.tone, 'ok');
});

test('"could not check" says so first, and is never an absence', () => {
  const line = denseHubLine(hubOnly(), {
    state: 'unknown',
    detail: 'Hugging Face could not be reached, so the repository was not checked.',
  });
  assert.equal(line.stateLabel, 'could not check');
  assert.match(line.text, /^Hugging Face could not be reached/);
  assert.doesNotMatch(line.text, /deleted|gone|no longer/);
  // The record still appears — dated, and behind the failure, so it cannot be
  // misread as the answer we failed to get.
  assert.match(line.text, /not re-checked since/);
});

test('a repository verified gone names what is LEFT, which is the whole question', () => {
  const nothing = denseHubLine(hubOnly(), { state: 'gone' });
  assert.equal(nothing.tone, 'error');
  assert.equal(nothing.stateLabel, 'not found');
  assert.match(nothing.text, /no longer returns this repository/);
  assert.match(nothing.text, /no recoverable model left/);
  assert.match(nothing.text, /training it again is the only way back/i);

  // The master is here: this is an inconvenience, not a loss, and saying
  // otherwise would send someone re-renting a GPU they do not need.
  const safe = denseHubLine(local(), { state: 'gone' });
  assert.match(safe.text, /full-precision master is still on this computer, so nothing is lost/);

  // Only the twin: still generates, can never be continued. The difference
  // costs eight hours of GPU to rediscover.
  const twinOnly = denseHubLine(local({ master: null }), { state: 'gone' });
  assert.match(twinOnly.text, /fp8 twin is still on this computer/);
  assert.match(twinOnly.text, /no longer be trained again, merged or re-quantized/);
});

test('a run still working has not failed to deliver — it has not got there yet', () => {
  const line = denseHubLine(hubOnly({
    active: true, hub: { repo_id: 'acme/x', status: 'pending' },
  }));
  assert.match(line.text, /still working — nothing has reached this repository yet/);
  assert.doesNotMatch(line.text, /never confirmed/);
});

test('a run that never had a repository has no line at all', () => {
  assert.equal(denseHubLine(local({ hub: null })), null);
  assert.equal(denseHubLine({}, { state: 'gone' }), null);
});

// --- the Raw sampler settings ------------------------------------------------

test('the guidance line carries the undistilled settings, and degrades quietly', () => {
  assert.equal(denseGuidanceLine({ guidance_scale: 4, steps: 25 }), 'CFG 4 · 25 steps');
  assert.equal(denseGuidanceLine({ guidance_scale: 4 }), 'CFG 4');
  assert.equal(denseGuidanceLine(null), '');
  assert.equal(denseGuidanceLine({}), '');
});

// --- the honest limit --------------------------------------------------------

test('a Studio target exists only once ComfyUI can really load the file', () => {
  assert.equal(denseStudioTarget(local()), null);          // twin not in ComfyUI
  const t = denseStudioTarget(local({
    fp8: { filename: 'f', in_comfyui: true, comfyui_name: 'krea\\f_fp8.safetensors' },
  }));
  assert.deepEqual(t, { base: 'krea\\f_fp8.safetensors', family: 'krea', datasetId: 3 });
});

test('the Studio limit is stated with its workaround, not just as a refusal', () => {
  assert.match(STUDIO_NEEDS_A_LORA, /strength to 0/);
  assert.match(STUDIO_NEEDS_A_LORA, /bare model/);
});

// --- sizes -------------------------------------------------------------------

test('sizes read in the unit that matches the file, and never render "0 GB"', () => {
  assert.equal(fmtBytes(26e9), '26.0 GB');
  assert.equal(fmtBytes(13.4e9), '13.4 GB');
  assert.equal(fmtBytes(0), '');
  assert.equal(fmtBytes(null), '');
  assert.equal(fmtBytes('nope'), '');
  // A file below a kilobyte exists; "0 kB" would say it does not — and a
  // truncated weight file is exactly when a size is worth reading.
  assert.equal(fmtBytes(137), '137 B');
  assert.equal(fmtBytes(2400), '2 kB');
});

// --- the panel's own source: two invariants worth pinning --------------------

const panel = readFileSync(
  fileURLToPath(new URL('./DenseModelsPanel.jsx', import.meta.url)), 'utf8');

test('no control in the panel can send the MASTER to ComfyUI', () => {
  // Every place that triggers a send sits behind `row.kind === 'fp8'`. A future
  // edit that moves one out of that guard fails here rather than in production,
  // with 26 GB landing in a model folder.
  const calls = [...panel.matchAll(/askSend\(entry\.run_id\)/g)].map((m) => m.index);
  assert.ok(calls.length >= 1, 'the panel must offer a send at all');
  for (const at of calls) {
    const before = panel.slice(0, at);
    assert.ok(before.lastIndexOf("row.kind === 'fp8'") > before.lastIndexOf('<FileRow'),
      'the send button must sit inside the fp8 branch of a file row');
  }
  // And the master's own row carries no send affordance of any kind.
  assert.ok(!/master[\s\S]{0,400}Send to ComfyUI/.test(panel));
});

test('the card survives 400 px: long names wrap, rows wrap, nothing scrolls sideways', () => {
  assert.ok(panel.includes('break-all'), 'weight filenames must be allowed to wrap');
  assert.ok((panel.match(/flex-wrap/g) || []).length >= 4,
    'the header, the action rows and the plan buttons all wrap');
  assert.ok(!panel.includes('overflow-x-scroll'));
  assert.ok(!panel.includes('whitespace-nowrap'));
});
