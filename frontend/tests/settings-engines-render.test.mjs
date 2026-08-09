/**
 * The Engines settings section, RENDERED — because a section that throws is a
 * settings page that will not open at all, and every other test in this area
 * reads the file as text.
 *
 * It is mounted in each ChatGPT auth lane, since each one takes a different
 * branch of the card (the ComfyUI lane draws an extra paragraph, and drops the
 * dollar wording), and a branch no test asks for is a branch that ships broken.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { default: EnginesSection } = await import(
  '../src/components/settings/EnginesSection.jsx')

const CONFIG = {
  engines: {
    default: 'krea',
    enabled: ['krea', 'minimax_h3'],
    chatgpt_auth: 'auto',
    chatgpt_comfy_quality: 'high',
    chatgpt_subscription_model: 'gpt-5.4-mini',
    openrouter_model: 'google/gemini-3-pro-image',
    chatgpt_image_model: '',
    qwen_model: '',
  },
  klein: {}, krea: {}, minimax_h3: {}, seedvr2: {}, improve: { engine: 'klein' },
  identity_prompts: {}, comfyui: {},
}

const render = (overrides = {}) => renderToStaticMarkup(createElement(EnginesSection, {
  config: { ...CONFIG, engines: { ...CONFIG.engines, ...(overrides.engines || {}) } },
  configDefaults: CONFIG,
  setField: () => {},
  toggleEngine: () => {},
  caps: {},
  refreshCaps: () => {},
  toast: { error: () => {}, success: () => {} },
  secretsPresence: {},
  secretInputs: {},
  setSecretInputs: () => {},
  testResults: {},
  recordTestResult: () => {},
  saveSecretIfPending: () => {},
  handleDeleteSecret: () => {},
}))

test('the section renders in every ChatGPT auth lane', () => {
  for (const lane of ['auto', 'api', 'subscription', 'comfyui']) {
    const html = render({ engines: { chatgpt_auth: lane } })
    assert.match(html, /ChatGPT engine auth/, lane)
  }
})

test('the comfy.org key field is offered, and needs no Test button to render', () => {
  const html = render()
  assert.match(html, /comfy\.org API key/)
  assert.match(html, /platform\.comfy\.org/)
})

test('the ComfyUI lane states its two costs where the choice is made', () => {
  const html = render({ engines: { chatgpt_auth: 'comfyui' } })
  assert.match(html, /OpenAI image node/)
  assert.match(html, /queue/i)                 // it holds a ComfyUI queue slot
  assert.match(html, /NSFW/)                   // ...and is still not local
})

test('the quality dial is only offered on the lane it applies to', () => {
  const on = render({ engines: { chatgpt_auth: 'comfyui' } })
  assert.match(on, /chatgpt-comfy-quality/)
  assert.match(on, /cheapest and fastest/)
  for (const lane of ['auto', 'api', 'subscription']) {
    assert.doesNotMatch(render({ engines: { chatgpt_auth: lane } }),
      /chatgpt-comfy-quality/, lane)
  }
})

/* The Base URL field draws DIFFERENT text depending on whether it is filled, and
   the filled branch is the one carrying the privacy warning — the one nobody
   would notice missing, because the field works either way. Both are mounted. */
test('the ChatGPT Base URL field renders blank, pointing at OpenAI', () => {
  const html = render({ engines: { chatgpt_base_url: '' } })
  assert.match(html, /ChatGPT \(OpenAI\) Base URL/)
  assert.match(html, /Blank = OpenAI itself/)
  assert.doesNotMatch(html, /go to that operator/)
})

test('a filled Base URL says out loud that a third party sees the photos', () => {
  const html = render({ engines: { chatgpt_base_url: 'https://gateway.example.com' } })
  assert.match(html, /Your reference photos go to that operator, not to OpenAI/)
  // ...and points at the field before the key, since 401/404 are what a wrong
  // Base URL returns and also what a bad key returns.
  assert.match(html, /401 or 404, suspect this field before your key/)
})

/* The dataset side of the same class of bug: the angle-reference panel lost its
   placeholder branch when all five slots opened, and a source-text test cannot
   tell a removed branch from a broken one. */
const { default: PoseSlotPanel } = await import(
  '../src/components/dataset/PoseSlotPanel.jsx')

test('the angle-reference panel renders all five slots, filled or empty', () => {
  const html = renderToStaticMarkup(createElement(PoseSlotPanel, {
    datasetId: 1,
    poseSlots: { left45: { filename: 'a.webp', enabled: true },
                 back: { filename: null, enabled: false } },
    busy: false,
    onSetPoseSlot: () => {}, onCropPoseSlot: () => {}, onMirrorPoseSlot: () => {},
    onTogglePoseSlotEnabled: () => {}, onRemovePoseSlot: () => {},
  }))
  for (const label of ['Left 45', 'Right 45', 'Left 90', 'Right 90', 'Back']) {
    assert.ok(html.includes(label), label)
  }
  assert.ok(!/coming soon/i.test(html))
})
