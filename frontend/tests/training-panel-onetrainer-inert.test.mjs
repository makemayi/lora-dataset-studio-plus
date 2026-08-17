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
