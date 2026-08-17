/**
 * The Advanced-options panel groups itself by OWNERSHIP from the two-sided
 * declaration at `backend/app/services/training_settings_map.py` (served over
 * GET /api/train/settings-map): shared settings, then the block belonging to
 * the lane in use, then the other lane's block — kept, collapsed, labelled,
 * never hidden outright. This file pins that grouping, plus the disabled/
 * enabled state and the SERVER-provided reason each control carries.
 *
 * The one-sided predecessor of this map (`/api/train/onetrainer/settings-status`,
 * a `{state: 'applies'|'pinned'|'preset'}` shape) still exists server-side and
 * is untouched — this file no longer exercises it. The three states are now
 * `applies` / `pinned` / `absent`, declared per lane; `preset` does not exist
 * in the new module (see `training_settings_map.py`'s own docstring for why
 * the complement could not simply be taken).
 *
 * `renderToStaticMarkup` never runs effects, so neither the local-trainer pick
 * (a `<select>` the user would otherwise have to click) nor the settings-map
 * fetch can happen inside a mount test. TrainingPanel.jsx therefore grows
 * three test-only optional props — `trainerOverride`, `settingsMapInitial`
 * (replacing the old `otSettingStatusInitial`) and `advOverride` — that seed
 * those pieces of state before the first render. All default to null/
 * 'ai_toolkit' and are never passed by the real app (App.jsx has no knowledge
 * of them), so production behaviour is unchanged: ai-toolkit by default, the
 * map fetched live, nothing disabled until it resolves.
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

/* --- the fixture -----------------------------------------------------------
 *
 * SERVER declaration exactly as backend/app/services/training_settings_map.py
 * ships it, trimmed to the keys and groups this file exercises — kept as DATA
 * here, not re-derived, so a client-side copy of the server's wording would
 * still fail the "server text, not a client restate" assertion below. The
 * GROUPS order and labels are copied verbatim too, because one of the tests
 * below exists specifically to pin that the panel follows this array's order
 * rather than a hardcoded one of its own.
 */
const GROUPS = [
  { key: 'core', label: 'Core' },
  { key: 'iteration', label: 'Iteration' },
  { key: 'network', label: 'Network' },
  { key: 'optimisation', label: 'Optimisation' },
  { key: 'text_encoders', label: 'Text encoders' },
  { key: 'memory', label: 'Memory' },
  { key: 'quality', label: 'Quality' },
]

const APPLIES = { state: 'applies', why: '' }
const ABSENT = { state: 'absent', why: '' }

const RANK_PINNED_WHY = 'OneTrainer runs at rank 32 on this lane. The rank chosen here is '
  + 'used by ai-toolkit only.'
const ALPHA_PINNED_WHY = 'Pinned to equal the rank (scale 1.0). A LoRA trained at the '
  + 'preset’s own alpha against this app’s rank came out ~1/32 of its intended strength.'
const NETWORK_TYPE_PINNED_WHY = 'This lane reads Settings ▸ OneTrainer ▸ PEFT type, not this control.'

const SETTINGS_MAP = {
  groups: GROUPS,
  lanes: {
    ai_toolkit: {
      learning_rate: { ...APPLIES, group: 'core' },
      resolution: { ...APPLIES, group: 'core' },
      dual_captions: { ...APPLIES, group: 'core' },
      lr_scheduler: { ...APPLIES, group: 'optimisation' },
      warmup: { ...APPLIES, group: 'optimisation' },
      min_snr_gamma: { ...APPLIES, group: 'quality' },
      rank: { ...APPLIES, group: 'network' },
      alpha: { ...APPLIES, group: 'network' },
      network_type: { ...APPLIES, group: 'network' },
      dropout: { ...APPLIES, group: 'network' },
      optimizer: { ...APPLIES, group: 'optimisation' },
      grad_accum: { ...APPLIES, group: 'optimisation' },
      timestep_type: { ...APPLIES, group: 'quality' },
      ema: { ...APPLIES, group: 'quality' },
      content_or_style: { ...APPLIES, group: 'quality' },
      do_differential_guidance: { ...APPLIES, group: 'quality' },
      differential_guidance_scale: { ...APPLIES, group: 'quality' },
      quantize: { ...APPLIES, group: 'memory' },
      quantize_te: { ...APPLIES, group: 'memory' },
      low_vram: { ...APPLIES, group: 'memory' },
      lokr_factor: { ...APPLIES, group: 'network' },
      epochs: { ...ABSENT, group: 'iteration' },
      batch_size: { ...ABSENT, group: 'iteration' },
      te1_lr: { ...ABSENT, group: 'text_encoders' },
      te2_lr: { ...ABSENT, group: 'text_encoders' },
    },
    onetrainer: {
      learning_rate: { ...APPLIES, group: 'core' },
      resolution: { ...APPLIES, group: 'core' },
      dual_captions: { ...APPLIES, group: 'core' },
      lr_scheduler: { ...APPLIES, group: 'optimisation' },
      warmup: { ...APPLIES, group: 'optimisation' },
      min_snr_gamma: { ...APPLIES, group: 'quality' },
      rank: { state: 'pinned', why: RANK_PINNED_WHY, group: 'network' },
      alpha: { state: 'pinned', why: ALPHA_PINNED_WHY, group: 'network' },
      network_type: { state: 'pinned', why: NETWORK_TYPE_PINNED_WHY, group: 'network' },
      dropout: { ...ABSENT, group: 'network' },
      optimizer: { ...ABSENT, group: 'optimisation' },
      grad_accum: { ...ABSENT, group: 'optimisation' },
      timestep_type: { ...ABSENT, group: 'quality' },
      ema: { ...ABSENT, group: 'quality' },
      content_or_style: { ...ABSENT, group: 'quality' },
      do_differential_guidance: { ...ABSENT, group: 'quality' },
      differential_guidance_scale: { ...ABSENT, group: 'quality' },
      quantize: { ...ABSENT, group: 'memory' },
      quantize_te: { ...ABSENT, group: 'memory' },
      low_vram: { ...ABSENT, group: 'memory' },
      lokr_factor: { ...ABSENT, group: 'network' },
      epochs: { ...APPLIES, group: 'iteration' },
      batch_size: { ...APPLIES, group: 'iteration' },
      te1_lr: { ...APPLIES, group: 'text_encoders' },
      te2_lr: { ...APPLIES, group: 'text_encoders' },
    },
  },
}

/* React's static-markup renderer HTML-entity-escapes attribute text (an
   apostrophe becomes `&#x27;`, `▸` becomes `&#x25B8;`), so every match below
   reads through this rather than comparing raw strings. */
const decode = (s) => s
  .replace(/&#x27;|&#39;/g, "'").replace(/&amp;/g, '&')
  .replace(/&quot;/g, '"').replace(/&mdash;/g, '—')
  .replace(/&#x25B8;|&#9656;/gi, '▸')

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/* Back to the opening tag, forward to its close: the class string carries
   Tailwind's `disabled:` VARIANTS (`disabled:opacity-50`), so a naive
   /disabled/ over a window around a field matches whether or not the control
   is actually disabled. Assert on the attribute, inside the element itself. */
function fieldAround(html, ariaLabel) {
  const at = html.indexOf(`aria-label="${ariaLabel}"`)
  assert.notEqual(at, -1, `${ariaLabel} not found`)
  const openAt = html.lastIndexOf('<', at)
  const closeSelf = html.indexOf('/>', at)
  const closeTag = html.indexOf('>', at)
  const close = (closeSelf !== -1 && closeSelf < closeTag + 40) ? closeSelf + 2 : closeTag + 1
  return html.slice(openAt, close)
}

test('ai-toolkit lane: nothing is greyed', () => {
  const html = decode(renderPanel({ trainerOverride: 'ai_toolkit', settingsMapInitial: SETTINGS_MAP }))
  assert.doesNotMatch(fieldAround(html, 'Optimizer'), /\sdisabled=""/, 'optimizer must not be disabled on ai-toolkit')
  assert.doesNotMatch(fieldAround(html, 'Network dropout'), /\sdisabled=""/, 'dropout must not be disabled on ai-toolkit')
  assert.doesNotMatch(fieldAround(html, 'LoRA rank'), /\sdisabled=""/, 'rank must not be disabled on ai-toolkit')
  assert.doesNotMatch(fieldAround(html, 'Network type'), /\sdisabled=""/, 'network type must not be disabled on ai-toolkit')
  assert.ok(!html.includes(RANK_PINNED_WHY), 'the OneTrainer-only pin reason must not appear on ai-toolkit')
})

test('OneTrainer lane: a setting absent from this lane sits in the other lane’s collapsed block, disabled', () => {
  const html = decode(renderPanel({ trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP }))
  assert.match(fieldAround(html, 'Optimizer'), /\sdisabled=""/, 'a setting absent on OneTrainer must be disabled there')
  // The reason is a fact ABOUT the current lane ("OneTrainer does not read
  // this"), not the other lane's own pin reason — those are different claims.
  const optField = fieldAround(html, 'Optimizer')
  assert.match(optField, /OneTrainer does not read this setting/)
})

test('the other-lane block is a collapsed <details>, and its heading names the lane', () => {
  const html = decode(renderPanel({ trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP }))
  assert.match(html, /ai-toolkit only — this lane does not read these/,
    'the collapsed block must name the OTHER lane (ai-toolkit) while viewing OneTrainer')
  const summaryAt = html.indexOf('ai-toolkit only')
  const detailsAt = html.lastIndexOf('<details', summaryAt)
  assert.notEqual(detailsAt, -1, 'the other-lane block must be a <details> element')
  const summaryTagAt = html.lastIndexOf('<summary', summaryAt)
  assert.ok(summaryTagAt > detailsAt, 'the naming heading must be inside a <summary>, i.e. collapsed by default')
})

test('OneTrainer lane: a pinned control carries the SERVER reason, not a client-side one', () => {
  const html = decode(renderPanel({ trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP }))
  const rankField = fieldAround(html, 'LoRA rank')
  assert.match(rankField, /\sdisabled=""/, 'the pinned rank control must be disabled on OneTrainer')
  // A distinctive fragment of the SERVER's own sentence — a hand-written
  // client duplicate of this wording would not reproduce it verbatim.
  assert.match(rankField, /OneTrainer runs at rank 32 on this lane/,
    'the pinned control must show the server-provided reason, not a client-authored one')
  assert.ok(!rankField.includes('does not read this setting'),
    'a pinned control must not fall back to the generic absent-lane sentence')
})

test('OneTrainer lane: the three "applies" controls are not disabled', () => {
  const html = decode(renderPanel({ trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP }))
  assert.doesNotMatch(fieldAround(html, 'Training resolution'), /\sdisabled=""/, 'resolution applies on OneTrainer and must stay enabled')
  assert.doesNotMatch(fieldAround(html, 'Dual long + short captions'), /\sdisabled=""/, 'dual captions applies on OneTrainer and must stay enabled')
  assert.doesNotMatch(fieldAround(html, 'Learning rate'), /\sdisabled=""/, 'learning_rate applies on OneTrainer and must stay enabled')
})

test('an unknown key fails safe as inert, with a generic reason', () => {
  // Empty per-lane maps: this app's own module could not declare the key —
  // the same "has never heard of it" case `training_settings_map.status()`
  // documents. Neither lane claims EMA, so it has no honest home in either
  // lane's own block; it must still render, disabled, rather than vanish.
  const html = decode(renderPanel({
    trainerOverride: 'onetrainer',
    settingsMapInitial: { groups: GROUPS, lanes: { ai_toolkit: {}, onetrainer: {} } },
  }))
  const emaField = fieldAround(html, 'EMA (exponential moving average)')
  assert.match(emaField, /\sdisabled=""/, 'a key neither lane mentions must fail safe to disabled')
  assert.match(emaField, /OneTrainer does not read this setting/)
})

/* --- the learning-rate control ------------------------------------------
 *
 * It is the ONE setting both lanes honour and neither exposed: without it the
 * rate is the family-fixed 1e-4 unless a preset happens to carry one, so a
 * dataset trained at 3e-4 by hand could not be reproduced in the app at all.
 * These pin that it exists, that it is NOT greyed on the OneTrainer lane (it
 * is shared — applies on both), and that it stands down for the optimisers
 * that set the rate themselves.
 */

test('the learning rate has a control at all', () => {
  const html = renderPanel()
  assert.match(html, /aria-label="Learning rate"/,
    'the rate reaches every run; a panel without this field cannot reproduce a hand-set one')
})

test('the learning rate is NOT greyed on the OneTrainer lane', () => {
  const html = decode(renderPanel({ trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP }))
  assert.doesNotMatch(fieldAround(html, 'Learning rate'), /\sdisabled=""/,
    'learning_rate is shared (applies on both lanes) and must not be disabled by the lane')
})

const lrField = (html) => fieldAround(html, 'Learning rate')

test('an adaptive optimiser disables the field instead of pretending it matters', () => {
  // prodigy drives the LR itself — _lr_from_settings returns the lr≈1.0
  // convention whatever is stored, so an editable box would be a lie. This is
  // independent of the lane classification above: the field can be shared
  // AND still stand down for an optimiser that ignores it.
  const off = lrField(renderPanel({ advOverride: { optimizer: 'prodigy' } }))
  assert.match(off, /\sdisabled=""/, 'prodigy ignores a fixed rate; the control must stand down')

  const on = lrField(renderPanel({ advOverride: { optimizer: 'adamw8bit' } }))
  assert.doesNotMatch(on, /\sdisabled=""/,
    'and it must be editable for the optimisers that DO read it — without this '
    + 'half the assertion above passes against a field that is always disabled')
})

/* --- epochs and batch, in OneTrainer's own vocabulary ---------------------
 *
 * OneTrainer counts EPOCHS. These fields live OUTSIDE the Expert block (they
 * are gated purely on `trainer === 'onetrainer'`, unaffected by this wave's
 * restructuring), so they are unaffected by the map itself — pinned here
 * so a future change to either area is caught by the other's tests.
 */

test('the epoch and batch fields exist only on the OneTrainer lane', () => {
  const ot = renderPanel({ trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP })
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
    trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP,
    advOverride: { epochs: 10, batch_size: 3 },
  })
  assert.match(html, /≈ 50 optimizer steps/)
})

test('with no epochs set the panel says the count comes from steps', () => {
  // Showing a step count back while the server is deriving epochs FROM steps
  // would be circular.
  const html = renderPanel({ trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP })
  assert.match(decode(html), /epochs derived from the step count above/)
})

test('a batch larger than the dataset is called out', () => {
  const html = renderPanel({
    trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP,
    advOverride: { epochs: 5, batch_size: 40 },
  })
  assert.match(decode(html), /larger than the 15 images/)
})

/* --- the text encoders --------------------------------------------------- */

test('the text-encoder rates appear only on the OneTrainer lane', () => {
  const ot = renderPanel({ trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP })
  assert.match(ot, /aria-label="OneTrainer text encoder 1 learning rate"/)
  assert.match(ot, /aria-label="OneTrainer text encoder 2 learning rate"/)
  assert.doesNotMatch(renderPanel(), /OneTrainer text encoder/)
})

test('an empty text-encoder rate reads as frozen, not as zero', () => {
  // "Training at 0" and "not training" would otherwise be two ways to say the
  // same thing, and only one of them would be true.
  const html = renderPanel({ trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP })
  const te1 = html.slice(html.lastIndexOf('<input', html.indexOf('text encoder 1')),
                         html.indexOf('/>', html.indexOf('text encoder 1')) + 2)
  assert.match(te1, /placeholder="frozen"/)
})

test('a text-encoder rate at or above the main rate is called out', () => {
  // The ordinary way a character LoRA ends up welded to the words in its
  // captions. Said at the moment of setting it, not in a doc nobody opens.
  const html = decode(renderPanel({
    trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP,
    advOverride: { learning_rate: 0.0003, te2_lr: 0.0003 },
  }))
  assert.match(html, /at or above the main 0\.0003/)

  const quiet = decode(renderPanel({
    trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP,
    advOverride: { learning_rate: 0.0003, te2_lr: 0.00001 },
  }))
  assert.doesNotMatch(quiet, /at or above the main/,
    'a sane fraction must not warn — a warning that always fires teaches nothing')
})

/* --- shared vs. lane-owned vs. the other lane, section by section -------- */

test('a shared setting appears in the Shared section on BOTH lanes', () => {
  // dual_captions applies on both lanes in the map, so it must sit under the
  // "Shared" heading (and be enabled) whichever lane is current.
  for (const lane of ['ai_toolkit', 'onetrainer']) {
    const html = decode(renderPanel({ trainerOverride: lane, settingsMapInitial: SETTINGS_MAP }))
    const sharedAt = html.indexOf('>Shared<')
    assert.notEqual(sharedAt, -1, `Shared heading missing on ${lane}`)
    const dualAt = html.indexOf('aria-label="Dual long + short captions"')
    assert.ok(dualAt > sharedAt, `dual captions must render after the Shared heading on ${lane}`)
    assert.doesNotMatch(fieldAround(html, 'Dual long + short captions'), /\sdisabled=""/)
  }
})

test('switching the lane MOVES a control between sections rather than removing it', () => {
  // network_type applies on ai-toolkit and is pinned on OneTrainer — never
  // absent on either — so it must be present, enabled, under "ai-toolkit
  // settings" on one lane, and present, disabled, under "OneTrainer settings"
  // on the other. It must never disappear.
  const ai = decode(renderPanel({ trainerOverride: 'ai_toolkit', settingsMapInitial: SETTINGS_MAP }))
  assert.match(ai, /aria-label="Network type"/, 'network type must render on ai-toolkit')
  assert.doesNotMatch(fieldAround(ai, 'Network type'), /\sdisabled=""/)
  assert.match(ai, /ai-toolkit settings/)

  const ot = decode(renderPanel({ trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP }))
  assert.match(ot, /aria-label="Network type"/, 'network type must still render on OneTrainer, not disappear')
  assert.match(fieldAround(ot, 'Network type'), /\sdisabled=""/)
  assert.match(ot, /OneTrainer settings/)
})

test('groups render in the server’s order, and an empty group is not rendered', () => {
  // Swap Network and Quality relative to the fixture above: both groups are
  // non-empty in the ai-toolkit "this lane" section (network_type/alpha/
  // dropout vs. ema/timestep_type), so the panel must follow the array, not a
  // hardcoded idea of which comes first.
  const swapped = {
    ...SETTINGS_MAP,
    groups: [
      { key: 'core', label: 'Core' },
      { key: 'iteration', label: 'Iteration' },
      { key: 'quality', label: 'Quality' },
      { key: 'network', label: 'Network' },
      { key: 'optimisation', label: 'Optimisation' },
      { key: 'text_encoders', label: 'Text encoders' },
      { key: 'memory', label: 'Memory' },
    ],
  }
  const html = decode(renderPanel({ trainerOverride: 'ai_toolkit', settingsMapInitial: swapped }))
  const qualityAt = html.indexOf('>Quality<')
  const networkAt = html.indexOf('>Network<')
  assert.notEqual(qualityAt, -1)
  assert.notEqual(networkAt, -1)
  assert.ok(qualityAt < networkAt, 'Quality was placed before Network in the map and must render first')

  // "Iteration" and "Text encoders" have nothing to show inside the Expert
  // block on the ai-toolkit lane (epochs/batch/TE live outside it, gated on
  // trainer==='onetrainer' alone) — an empty group must not print a heading.
  assert.ok(!html.includes('>Iteration<'), 'an empty group must not render its heading')
  assert.ok(!html.includes('>Text encoders<'), 'an empty group must not render its heading')
})

test('min-SNR gamma is a SHARED control, present on both lanes', () => {
  // Both trainers read it, each in its own shape, so it belongs with the
  // cross-lane settings rather than inside either lane's block. A control that
  // appeared on only one lane would misdescribe a setting that reaches both.
  assert.match(renderPanel(), /aria-label="Min-SNR gamma"/)
  const ot = decode(renderPanel({ trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP }))
  assert.doesNotMatch(fieldAround(ot, 'Min-SNR gamma'), /\sdisabled=""/, 'it applies on OneTrainer too')
  assert.match(fieldAround(ot, 'Min-SNR gamma'), /placeholder="off"/, 'empty reads as off, not as zero')
  const sharedAt = ot.indexOf('>Shared<')
  assert.ok(ot.indexOf('aria-label="Min-SNR gamma"') > sharedAt, 'min-SNR gamma must render under Shared')
})

/* --- branches that only render under a non-default `adv`, mounted here so a
   broken one cannot hide behind tests that never reach it (the Settings page
   white-screen this rule exists for: a source-text test cannot tell a removed
   branch from a broken one). Neither has its own aria-label test elsewhere in
   this suite. --- */

test('the LoKr factor field renders when Network is set to LoKr', () => {
  const html = renderPanel({
    trainerOverride: 'ai_toolkit', settingsMapInitial: SETTINGS_MAP,
    advOverride: { network_type: 'lokr' },
  })
  assert.match(html, /aria-label="LoKr decomposition factor"/)
})

test('the Krea community recipe card renders when the family supports it', () => {
  const html = renderPanel({
    trainerOverride: 'ai_toolkit', settingsMapInitial: SETTINGS_MAP,
    advOverride: { krea_recipe_supported: true },
  })
  assert.match(html, /aria-label="Krea content or style balance"/)
  assert.match(html, /aria-label="Enable Krea differential guidance"/)
  assert.match(html, /aria-label="Krea differential guidance scale"/)

  // And on OneTrainer, where all three are absent: still present, disabled,
  // inside the collapsed other-lane block — not gone.
  const ot = decode(renderPanel({
    trainerOverride: 'onetrainer', settingsMapInitial: SETTINGS_MAP,
    advOverride: { krea_recipe_supported: true },
  }))
  assert.match(fieldAround(ot, 'Krea content or style balance'), /\sdisabled=""/)
})
