/**
 * The OneTrainer lane's Advanced options are mostly decorative: only three of
 * the panel's ~40 fields (learning rate, resolution, dual captions) reach a
 * OneTrainer run. `backend/app/services/onetrainer_service.py` (418824c0)
 * declares the truth via GET /api/train/onetrainer/settings-status; this file
 * pins that the panel actually GREYS OUT what the declaration says is inert,
 * on the OneTrainer lane only, and leaves everything alone on ai-toolkit.
 *
 * `renderToStaticMarkup` never runs effects, so neither the local-trainer pick
 * (a `<select>` the user would otherwise have to click) nor the settings-status
 * fetch can happen inside a mount test. TrainingPanel.jsx therefore grew two
 * test-only optional props — `trainerOverride` and `otSettingStatusInitial` —
 * that seed those two pieces of state before the first render. Both default to
 * null/'ai_toolkit' and are never passed by the real app (App.jsx has no
 * knowledge of them), so production behaviour is unchanged.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

const { default: TrainingPanel } = await import(
  '../src/components/dataset/TrainingPanel.jsx')
const { CapabilitiesContext } = await import(
  '../src/context/CapabilitiesContext.jsx')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
const { MemoryRouter } = await import('react-router')

const CAPS = {
  configured: true,
  engines: { nanobanana: false, chatgpt: false, openrouter: false, qwen: false, klein: false },
  comfyui: { reachable: false, api_url: '', models: {} },
  ollama: { reachable: false, installed: false, binary_path: '', url: '', vision_model: '', vision_model_ready: false },
  aitoolkit: { configured: true, valid: true },
  captioners: { joycaption: false, ollama: false },
  face_scoring: false,
  python: { version: '', ml_supported: true, ml_range: '3.10-3.12' },
  masks: true,
  watermark_inpaint: false,
  watermark_allow_crop: true,
  training_visible: true,
  cloud_training: false,
  studio_visible: false,
}

// The methods the panel only ever calls from inside an effect or an event
// handler — neither runs in a static render — so a no-op is enough for each.
const noop = () => Promise.resolve(null)
const ds = {
  currentId: 3,
  data: { subject_type: 'person', best_settings_loras: [] },
  trainBaseInfo: noop,
  setTrainSettings: noop,
  setDatasetTrainingMode: noop,
  setDatasetTrainType: noop,
  listCheckpoints: () => Promise.resolve([]),
  continueTrainingInCloud: noop,
  continueTraining: noop,
  train: noop,
  stopTraining: noop,
  prepareBase: noop,
  deleteCheckpoint: noop,
  importCheckpoint: noop,
}

function renderPanel(props = {}) {
  return renderToStaticMarkup(createElement(MemoryRouter, null,
    createElement(CapabilitiesContext.Provider, { value: { caps: CAPS, loading: false, refresh: noop } },
      createElement(ToastProvider, null,
        createElement(TrainingPanel, {
          ds, keptCount: 15, kind: 'character', onCheckpointsChange: () => {},
          ...props,
        })))))
}

/* SERVER declaration exactly as backend/app/services/onetrainer_service.py
   (418824c0) ships it, trimmed to the keys this file exercises — kept as DATA
   here, not re-derived, so a client-side copy of the server's wording would
   still fail the "server text, not a client restate" assertion below. */
const OT_STATUS = {
  learning_rate: { state: 'applies', why: '' },
  resolution: { state: 'applies', why: '' },
  dual_captions: { state: 'applies', why: '' },
  rank: { state: 'pinned', why: 'OneTrainer runs at rank 32 on this lane. The rank chosen here is used by ai-toolkit only.' },
  alpha: { state: 'pinned', why: 'Pinned to equal the rank (scale 1.0).' },
  network_type: { state: 'pinned', why: 'This lane reads Settings ▸ OneTrainer ▸ PEFT type, not this control.' },
  optimizer: { state: 'preset', why: '' },
  dropout: { state: 'preset', why: '' },
  timestep_type: { state: 'preset', why: '' },
  ema: { state: 'preset', why: '' },
  grad_accum: { state: 'preset', why: '' },
}

const PRESET_SENTENCE = "Decided by OneTrainer's own Krea 2 preset — this app deliberately does not override it."

/* React's static-markup renderer HTML-entity-escapes attribute text (the
   apostrophe in PRESET_SENTENCE becomes `&#x27;`), so every match below reads
   through this rather than comparing raw strings. */
const decode = (s) => s
  .replace(/&#x27;|&#39;/g, "'").replace(/&amp;/g, '&')
  .replace(/&quot;/g, '"').replace(/&mdash;/g, '—')

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

test('ai-toolkit lane: nothing is greyed', () => {
  const html = renderPanel({ trainerOverride: 'ai_toolkit', otSettingStatusInitial: OT_STATUS })
  assert.ok(html.includes('aria-label="Optimizer"'), 'the optimizer control did not render')
  assert.ok(!/aria-label="Optimizer"[^>]*disabled=""/.test(html), 'optimizer must not be disabled on ai-toolkit')
  assert.ok(!/aria-label="Network dropout"[^>]*disabled=""/.test(html), 'dropout must not be disabled on ai-toolkit')
  assert.ok(!/aria-label="LoRA rank"[^>]*disabled=""/.test(html), 'rank must not be disabled on ai-toolkit')
  assert.ok(!html.includes(PRESET_SENTENCE), 'the preset sentence must not appear on ai-toolkit')
})

test('OneTrainer lane: a preset-owned control is disabled and carries the shared sentence', () => {
  const html = decode(renderPanel({ trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS }))
  // Optimizer is 'preset' in the declaration above.
  const optMatch = html.match(/<select[^>]*aria-label="Optimizer"[^>]*>/)
  assert.ok(optMatch, 'optimizer select not found')
  assert.match(optMatch[0], /disabled=""/, 'a preset-owned control must be disabled on the OneTrainer lane')
  assert.match(optMatch[0], new RegExp(`title="${escapeRe(PRESET_SENTENCE)}"`),
    'a preset-owned control must show the shared sentence')
})

test('OneTrainer lane: a pinned control carries the SERVER reason, not a client-side one', () => {
  const html = renderPanel({ trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS })
  const rankMatch = html.match(/<select[^>]*aria-label="LoRA rank"[^>]*>/)
  assert.ok(rankMatch, 'rank select not found')
  assert.match(rankMatch[0], /disabled=""/, 'the pinned rank control must be disabled on OneTrainer')
  // A distinctive fragment of the SERVER's own sentence — a hand-written
  // client duplicate of this wording would not reproduce it verbatim.
  assert.match(rankMatch[0], /OneTrainer runs at rank 32 on this lane/,
    'the pinned control must show the server-provided reason, not a client-authored one')
  assert.ok(!rankMatch[0].includes(PRESET_SENTENCE),
    'a pinned control must not fall back to the generic preset sentence')
})

test('OneTrainer lane: the three "applies" controls are not disabled', () => {
  const html = renderPanel({ trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS })
  const resMatch = html.match(/<select[^>]*aria-label="Training resolution"[^>]*>/)
  assert.ok(resMatch, 'resolution select not found')
  assert.doesNotMatch(resMatch[0], /disabled=""/, 'resolution applies on OneTrainer and must stay enabled')
  const dualMatch = html.match(/<input[^>]*aria-label="Dual long \+ short captions"[^>]*>/)
  assert.ok(dualMatch, 'dual captions checkbox not found')
  assert.doesNotMatch(dualMatch[0], /disabled=""/, 'dual captions applies on OneTrainer and must stay enabled')
  // learning_rate is the third 'applies' key, but this panel has no editable
  // learning-rate control on the LoRA/OneTrainer arm at all (the only "Learning
  // rate" input in this file belongs to the separate, cloud-only full-transformer
  // arm) — so there is nothing here to assert enabled. Recorded, not invented.
})

test('an unknown key fails safe as preset-owned', () => {
  const html = decode(renderPanel({ trainerOverride: 'onetrainer', otSettingStatusInitial: {} }))
  const emaMatch = html.match(/<select[^>]*aria-label="EMA \(exponential moving average\)"[^>]*>/)
  assert.ok(emaMatch, 'EMA select not found')
  assert.match(emaMatch[0], /disabled=""/, 'a key the endpoint never mentions must fail safe to preset (disabled)')
  assert.match(emaMatch[0], new RegExp(escapeRe(PRESET_SENTENCE)))
})

/* --- the learning-rate control ------------------------------------------
 *
 * It is the ONE setting both lanes honour and neither exposed: without it the
 * rate is the family-fixed 1e-4 unless a preset happens to carry one, so a
 * dataset trained at 3e-4 by hand could not be reproduced in the app at all.
 * These pin that it exists, that it is NOT greyed on the OneTrainer lane (it
 * is one of the three that apply), and that it stands down for the optimisers
 * that set the rate themselves.
 */

test('the learning rate has a control at all', () => {
  const html = renderPanel()
  assert.match(html, /aria-label="Learning rate"/,
    'the rate reaches every run; a panel without this field cannot reproduce a hand-set one')
})

test('the learning rate is NOT greyed on the OneTrainer lane', () => {
  // It is declared 'applies'. Greying it would be the opposite error to the one
  // this file exists to catch: hiding a control that genuinely works.
  const html = renderPanel({ trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS })
  const field = html.slice(html.indexOf('aria-label="Learning rate"') - 400,
                           html.indexOf('aria-label="Learning rate"') + 200)
  assert.doesNotMatch(decode(field), new RegExp(PRESET_SENTENCE.slice(0, 30)),
    'learning_rate applies on this lane and must not carry the preset-owned reason')
})

const lrField = (html) => {
  const at = html.indexOf('aria-label="Learning rate"')
  assert.notEqual(at, -1, 'the learning-rate field is gone')
  // Back to the opening `<input`, forward to its close: the class string carries
  // Tailwind's `disabled:` VARIANTS, so a naive /disabled/ over a window around
  // the field matches whether or not the control is actually disabled. That is
  // how the first version of this test passed against a field that was enabled.
  return html.slice(html.lastIndexOf('<input', at), html.indexOf('/>', at) + 2)
}

test('an adaptive optimiser disables the field instead of pretending it matters', () => {
  // prodigy drives the LR itself — _lr_from_settings returns the lr≈1.0
  // convention whatever is stored, so an editable box would be a lie.
  const off = lrField(renderPanel({ advOverride: { optimizer: 'prodigy' } }))
  assert.match(off, /\sdisabled=""/, 'prodigy ignores a fixed rate; the control must stand down')

  const on = lrField(renderPanel({ advOverride: { optimizer: 'adamw8bit' } }))
  assert.doesNotMatch(on, /\sdisabled=""/,
    'and it must be editable for the optimisers that DO read it — without this '
    + 'half the assertion above passes against a field that is always disabled')
})

/* --- epochs and batch, in OneTrainer's own vocabulary --------------------
 *
 * OneTrainer counts EPOCHS. The app used to hide that behind
 * `epochs = ceil(steps / images)` with the batch pinned to 1 so the arithmetic
 * held — an approximation, and one that made a hand-set batch inexpressible.
 * These pin that the panel now asks in OneTrainer's words on that lane, and
 * asks in nobody's words on the other one.
 */

test('the epoch and batch fields exist only on the OneTrainer lane', () => {
  const ot = renderPanel({ trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS })
  assert.match(ot, /aria-label="OneTrainer epochs"/)
  assert.match(ot, /aria-label="OneTrainer batch size"/)

  // ai-toolkit thinks in steps and never reads these; showing them there would
  // be the same lie in the other direction.
  const ai = renderPanel()
  assert.doesNotMatch(ai, /aria-label="OneTrainer epochs"/)
  assert.doesNotMatch(ai, /aria-label="OneTrainer batch size"/)
})

test('the derived step count is shown, and counts the batch', () => {
  // 10 epochs over 15 kept images at batch 3 = ceil(10*15/3) = 50 steps. The
  // old derivation ignored the batch entirely; a label that did the same would
  // be worse than none, because it would look authoritative.
  const html = renderPanel({
    trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS,
    advOverride: { epochs: 10, batch_size: 3 },
  })
  assert.match(html, /≈ 50 optimizer steps/)
})

test('with no epochs set the panel says the count comes from steps', () => {
  // Showing a step count back while the server is deriving epochs FROM steps
  // would be circular.
  const html = renderPanel({ trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS })
  assert.match(decode(html), /epochs derived from the step count above/)
})

test('a batch larger than the dataset is called out', () => {
  const html = renderPanel({
    trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS,
    advOverride: { epochs: 5, batch_size: 40 },
  })
  assert.match(decode(html), /larger than the 15 images/)
})

/* --- the text encoders --------------------------------------------------- */

test('the text-encoder rates appear only on the OneTrainer lane', () => {
  const ot = renderPanel({ trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS })
  assert.match(ot, /aria-label="OneTrainer text encoder 1 learning rate"/)
  assert.match(ot, /aria-label="OneTrainer text encoder 2 learning rate"/)
  assert.doesNotMatch(renderPanel(), /OneTrainer text encoder/)
})

test('an empty text-encoder rate reads as frozen, not as zero', () => {
  // "Training at 0" and "not training" would otherwise be two ways to say the
  // same thing, and only one of them would be true.
  const html = renderPanel({ trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS })
  const te1 = html.slice(html.lastIndexOf('<input', html.indexOf('text encoder 1')),
                         html.indexOf('/>', html.indexOf('text encoder 1')) + 2)
  assert.match(te1, /placeholder="frozen"/)
})

test('a text-encoder rate at or above the main rate is called out', () => {
  // The ordinary way a character LoRA ends up welded to the words in its
  // captions. Said at the moment of setting it, not in a doc nobody opens.
  const html = decode(renderPanel({
    trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS,
    advOverride: { learning_rate: 0.0003, te2_lr: 0.0003 },
  }))
  assert.match(html, /at or above the main 0\.0003/)

  const quiet = decode(renderPanel({
    trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS,
    advOverride: { learning_rate: 0.0003, te2_lr: 0.00001 },
  }))
  assert.doesNotMatch(quiet, /at or above the main/,
    'a sane fraction must not warn — a warning that always fires teaches nothing')
})

/* --- the schedule, once the backend started honouring it ------------------
 *
 * lr_scheduler and warmup moved from 'preset' to 'applies' when
 * build_job_config learned to map this app's vocabulary into OneTrainer's enum.
 * The panel must follow the declaration, not a memory of it — that is the whole
 * point of serving the status from the module that writes the config.
 */

test('a control follows the declaration when a setting stops being preset-owned', () => {
  const nowApplies = { ...OT_STATUS, lr_scheduler: { state: 'applies', why: '' } }
  const html = renderPanel({ trainerOverride: 'onetrainer', otSettingStatusInitial: nowApplies })
  const sched = html.match(/<select[^>]*aria-label="Learning-rate schedule"[^>]*>/)
  assert.ok(sched, 'the schedule select is gone')
  assert.doesNotMatch(sched[0], /disabled=""/,
    'the server now says this applies; a panel that kept greying it would be '
    + 'the same drift the declaration exists to end, pointing the other way')

  // And the converse, so this is not a test that passes on any input.
  const stillPreset = renderPanel({ trainerOverride: 'onetrainer', otSettingStatusInitial: OT_STATUS })
  const greyed = stillPreset.match(/<select[^>]*aria-label="Learning-rate schedule"[^>]*>/)
  assert.match(greyed[0], /disabled=""/)
})
