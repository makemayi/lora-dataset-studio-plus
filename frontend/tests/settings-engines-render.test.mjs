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

/* The page is a rail + detail: only the selected entry renders. `focusId` is the
   deep-link id SettingsPage hands down, and it also decides which entry opens —
   so a test asserting on a field says which field it means, exactly as a user
   arriving from Settings search would. Default 'keys' matches the live page. */
const CHATGPT = 'chatgpt-auth-mode'   // opens the ChatGPT rail entry
const MODELS = 'engine-image-models'  // opens Image models

const render = (overrides = {}) => renderToStaticMarkup(createElement(EnginesSection, {
  config: { ...CONFIG, engines: { ...CONFIG.engines, ...(overrides.engines || {}) },
            ...(overrides.klein ? { klein: overrides.klein } : {}) },
  focusId: overrides.focusId,
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
    const html = render({ engines: { chatgpt_auth: lane }, focusId: CHATGPT })
    assert.match(html, /ChatGPT engine auth/, lane)
  }
})

test('the comfy.org key field is offered, and needs no Test button to render', () => {
  const html = render()   // an ENGINE_SECRET, so the default API keys entry
  assert.match(html, /comfy\.org API key/)
  assert.match(html, /platform\.comfy\.org/)
})

test('the ComfyUI lane states its two costs where the choice is made', () => {
  const html = render({ engines: { chatgpt_auth: 'comfyui' }, focusId: CHATGPT })
  assert.match(html, /OpenAI image node/)
  assert.match(html, /queue/i)                 // it holds a ComfyUI queue slot
  assert.match(html, /NSFW/)                   // ...and is still not local
})

test('the quality dial is only offered on the lane it applies to', () => {
  // Match the CONTROL, not the string: the rail button advertises the ids it
  // owns in data-focus-gate, so a bare /chatgpt-comfy-quality/ now matches the
  // nav on every lane and the negatives below would pass for the wrong reason.
  const dial = /id="chatgpt-comfy-quality"/
  const on = render({ engines: { chatgpt_auth: 'comfyui' }, focusId: CHATGPT })
  assert.match(on, dial)
  assert.match(on, /cheapest and fastest/)
  for (const lane of ['auto', 'api', 'subscription']) {
    assert.doesNotMatch(render({ engines: { chatgpt_auth: lane }, focusId: CHATGPT }),
      dial, lane)
  }
})

/* The Base URL field shows one of two hints depending on whether it is filled.
   It first shipped as a ternary, which CRASHED the whole Settings section for
   anyone reading the page through Chrome's auto-translate: Translate rewrites
   text nodes into its own <font> wrappers, so React's removeChild threw
   "NotFoundError: The node to be removed is not a child of this node" on the
   keystroke that filled the field, and the error boundary swallowed the section.

   The fix is structural — BOTH hints stay mounted and only `hidden` flips — so
   the invariant worth pinning is not "which text appears" but "the text is
   always there, in both states". A test cannot run Google Translate; it CAN
   refuse the shape that Translate breaks. */
test('the ChatGPT Base URL field renders blank, pointing at OpenAI', () => {
  const html = render({ engines: { chatgpt_base_url: '' }, focusId: MODELS })
  assert.match(html, /ChatGPT \(OpenAI\) Base URL/)
  assert.match(html, /Blank = OpenAI itself/)
})

test('a filled Base URL says out loud that a third party sees the photos', () => {
  const html = render({ engines: { chatgpt_base_url: 'https://gateway.example.com' }, focusId: MODELS })
  assert.match(html, /Your reference photos go to that operator, not to OpenAI/)
  // ...and points at the field before the key, since 401/404 are what a wrong
  // Base URL returns and also what a bad key returns.
  assert.match(html, /401 or 404, suspect this field before your key/)
})

test('the Nano Banana Base URL field renders in both states', () => {
  assert.match(render({ engines: { nanobanana_base_url: '' }, focusId: MODELS }),
    /Nano Banana \(Gemini\) Base URL/)
  assert.match(render({ engines: { nanobanana_base_url: 'https://gw.example.com' }, focusId: MODELS }),
    /Your reference photos go to that operator, not to Google/)
})

/* BOTH gateway fields, checked the same way. Two fields, four hints, and every
   one of them must survive a value change without a text node being removed. */
test('both Base URL hints stay in the DOM in BOTH states, toggled by hidden', () => {
  const cases = [
    ['chatgpt_base_url', /not to OpenAI/, /Applies to the API-key lane only/],
    ['nanobanana_base_url', /not to Google/, /Applies to this engine only/],
  ]
  for (const [key, warning, scope] of cases) {
    for (const value of ['', 'https://gateway.example.com']) {
      const html = render({ engines: { [key]: value }, focusId: MODELS })
      // Present either way: a ternary would drop one, and dropping a text node
      // is exactly what kills a translated page.
      assert.match(html, warning, `${key}: warning missing for ${value || 'blank'}`)
      assert.match(html, scope, `${key}: scope note missing for ${value || 'blank'}`)
    }
  }
  // Two fields, so exactly two hidden spans whichever way each one is set.
  const hiddenSpans = (html) => (html.match(/<span hidden=""/g) || []).length
  const blank = render({ engines: { chatgpt_base_url: '', nanobanana_base_url: '' }, focusId: MODELS })
  const filled = render({
    engines: { chatgpt_base_url: 'https://a.example.com',
               nanobanana_base_url: 'https://b.example.com' },
    focusId: MODELS,
  })
  assert.equal(hiddenSpans(blank), 2)
  assert.equal(hiddenSpans(filled), 2)
  assert.notEqual(blank, filled)
})

/* Face swap LoRAs. Both states mount: empty (the placeholder line) and filled
   (rows with a combobox, a strength slider and the reorder buttons). */
test('the face swap LoRA card renders empty and with rows', () => {
  const empty = render({ engines: {}, focusId: MODELS })
  assert.match(empty, /Face swap LoRAs \(optional\)/)
  assert.match(empty, /the face swap runs with just the LoRAs its own graph names/)

  const filled = renderToStaticMarkup(createElement(EnginesSection, {
    config: {
      ...CONFIG,
      klein: { face_swap_loras: [{ file: 'a.safetensors', strength: 0.8 },
                                 { file: 'b.safetensors', strength: 1.2 }] },
    },
    focusId: MODELS,
    configDefaults: CONFIG,
    setField: () => {}, toggleEngine: () => {}, caps: {}, refreshCaps: () => {},
    toast: { error: () => {}, success: () => {} },
    secretsPresence: {}, secretInputs: {}, setSecretInputs: () => {},
    testResults: {}, recordTestResult: () => {}, saveSecretIfPending: () => {},
    handleDeleteSecret: () => {},
  }))
  assert.match(filled, /2\/8 in the chain/)
  assert.match(filled, /Face swap LoRA 1 strength/)
  assert.doesNotMatch(filled, /the face swap runs with just the LoRAs/)
})

test('a row duplicating a LoRA the swap graph already loads is flagged', () => {
  const withGraphLoras = (file, graphLoras) => renderToStaticMarkup(
    createElement(EnginesSection, {
      config: { ...CONFIG, klein: { face_swap_loras: [{ file, strength: 1 }] } },
      focusId: MODELS,
      configDefaults: CONFIG,
      faceSwapGraphLoras: graphLoras,
      setField: () => {}, toggleEngine: () => {}, caps: {}, refreshCaps: () => {},
      toast: { error: () => {}, success: () => {} },
      secretsPresence: {}, secretInputs: {}, setSecretInputs: () => {},
      testResults: {}, recordTestResult: () => {}, saveSecretIfPending: () => {},
      handleDeleteSecret: () => {},
    }))
  const own = ['klein\\Klein2-9B-SmartCharacterSwap.safetensors']
  const warning = /face swap graph already loads this LoRA/

  assert.match(withGraphLoras(own[0], own), warning)
  // A separator/case difference must not dodge it, same as the server.
  assert.match(withGraphLoras('KLEIN/klein2-9b-smartcharacterswap.safetensors', own),
    warning)
  assert.doesNotMatch(withGraphLoras('something-else.safetensors', own), warning)
  // No server list yet (first paint) must not invent a warning.
  assert.doesNotMatch(withGraphLoras(own[0], []), warning)
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


/* ── Deep links must still land (2026-08-09) ──────────────────────────────────
   The page now shows ONE rail entry at a time, so a help topic or Settings-search
   result pointing at a field in a different entry would scroll to a page showing
   something else. `railItemForFocus` is what prevents that, and this is the test
   that keeps the two lists in step: every anchor the help registry publishes for
   the `engines` section has to be claimed by exactly one entry. Adding a setting
   without adding its id here is the regression, and it is invisible by eye. */
const { railItemForFocus, RAIL_ITEMS } = await import(
  '../src/components/settings/EnginesSection.jsx')
const { helpTopics } = await import('../src/help/helpRegistry.js')

test('every help anchor on this page is owned by a rail entry', () => {
  const anchors = helpTopics
    .filter((t) => t.app && t.app.route === '/settings/engines' && t.app.focus)
    .map((t) => t.app.focus)
  assert.ok(anchors.length >= 5, `expected engine help topics, saw ${anchors.length}`)
  const orphans = anchors.filter((a) => !railItemForFocus(a))
  assert.deepEqual(orphans, [],
    'these help anchors open a rail entry that does not contain them')
})

test('no dom id is claimed by two rail entries', () => {
  const seen = new Map()
  for (const item of RAIL_ITEMS) {
    for (const id of item.owns || []) {
      assert.equal(seen.get(id), undefined,
        `${id} is claimed by both ${seen.get(id)} and ${item.id}`)
      seen.set(id, item.id)
    }
  }
})

test('an unknown focus id falls back instead of blanking the page', () => {
  assert.equal(railItemForFocus('not-a-real-id'), null)
  assert.equal(railItemForFocus(null), null)
  assert.match(render({ focusId: 'not-a-real-id' }), /API keys/)
})
