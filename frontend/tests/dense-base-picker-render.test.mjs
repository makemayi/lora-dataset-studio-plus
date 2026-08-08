/**
 * The full-model base picker, RENDERED — not matched as source text.
 *
 * The owner's report was "I still can't see where to put the turbo option".
 * The fix adds a control; a control that no test ever executes is a control
 * that can ship with a ReferenceError in it and a green suite behind it (see
 * tests/support/mountJsx.mjs for the two white screens that taught this).
 *
 * The whole TrainingPanel cannot be mounted into full-model mode: it only
 * enters that mode from an effect, and effects do not run under
 * renderToStaticMarkup. So the picker is its own exported component, and this
 * file renders it in each of the three states a dense run can be in.
 *
 * What is asserted is what a user must be able to SEE or must not be misled
 * by — the Turbo option exists, the summary names the base that will really be
 * trained, and a picked checkpoint disables a switch that would otherwise
 * offer a choice with no effect.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

/* ⚠️ Dynamic — the hooks that teach Node to read .jsx are installed while
   mountJsx.mjs is evaluated, and a static import would already be linked. */
const { DenseBasePicker } =
  await import('../src/components/dataset/TrainingPanel.jsx')
const { fullTransformerBaseLabel } =
  await import('../src/utils/trainingMode.js')

const INSTALLED = [
  { value: '', label: 'Official recipe' },
  { value: 'D:/models/krea2-raw-bf16.safetensors', label: 'Krea 2 Raw bf16' },
]

const noop = () => {}

function render(props) {
  return renderToStaticMarkup(createElement(DenseBasePicker, {
    variant: 'base', setVariant: noop, base: '', setBase: noop,
    customBase: false, setCustomBase: noop,
    currentBases: INSTALLED, customSupported: true,
    ...props,
  }))
}

test('the Turbo option exists, and so does the way to reach a local checkpoint', () => {
  const html = render({ baseSummary: fullTransformerBaseLabel({ variant: 'base' }) })
  assert.match(html, /aria-label="Krea 2 base for full-model training"/)
  assert.match(html, /<option value="turbo"[^>]*>Turbo \(few-step\)<\/option>/)
  assert.match(html, /<option value="base"[^>]*>Raw \(recommended\)<\/option>/)
  // The catalog entry and the local-file escape hatch both reach the list.
  assert.match(html, /Krea 2 Raw bf16/)
  assert.match(html, /Custom weights/)
})

test('the summary names the base that will really be trained, in all three states', () => {
  for (const [variant, base, expected] of [
    ['base', '', 'official Krea 2 Raw'],
    ['turbo', '', 'official Krea 2 Turbo'],
    ['turbo', 'D:/models/mine.safetensors', 'custom: mine.safetensors'],
  ]) {
    const summary = fullTransformerBaseLabel({ variant, baseModel: base })
    assert.equal(summary, expected)
    // Rendered from the SAME helper the panel feeds it, so a change of wording
    // cannot leave the screen and the resolver disagreeing.
    assert.ok(render({ variant, base, baseSummary: summary })
      .includes(`This run will train <b class="text-content-muted font-medium">${expected}</b>`),
    `the picker must state "${expected}"`)
  }
})

test('a picked checkpoint disables the Raw/Turbo switch instead of ignoring it', () => {
  // Backend truth: _krea_name_or_path returns the custom path whatever the
  // variant says. A live switch would offer a choice with no effect.
  const withCustom = render({ variant: 'turbo', base: 'D:/models/mine.safetensors' })
  const switchTag = withCustom.match(
    /<select[^>]*aria-label="Krea 2 base for full-model training"[^>]*>/)[0]
  assert.match(switchTag, /disabled=""/)
  // …and the picker below it stays usable, or there would be no way back.
  const fileTag = withCustom.match(
    /<select[^>]*aria-label="Full-model base checkpoint"[^>]*>/)[0]
  assert.doesNotMatch(fileTag, /disabled/)
  assert.match(withCustom, /Raw\/Turbo switch does not apply/)
  // …and it is live again as soon as the official lane is selected.
  assert.doesNotMatch(render({ variant: 'turbo', base: '' }),
    /Raw\/Turbo switch does not apply/)
})

test('typing a path is possible, and the placeholder carries no machine path', () => {
  const html = render({ customBase: true, base: '' })
  assert.match(html, /aria-label="Full-model custom weights path"/)
  // A public repo: the hint must not look like anyone's drive.
  assert.doesNotMatch(html, /[A-Z]:\\\\?Users/i)
})

test('the mechanical refusal is stated where the base is chosen, not after paying', () => {
  const html = render({})
  assert.match(html, /scaled fp8 export cannot be loaded/)
  assert.match(html, /bf16\/fp16 build/)
})

test('a base advisory from the picker above is surfaced with its severity', () => {
  const err = render({ baseNote: { level: 'error', text: 'not a Krea 2 checkpoint' } })
  assert.match(err, /not a Krea 2 checkpoint/)
  assert.match(err, /text-red-300/)
  const warn = render({ baseNote: { level: 'warn', text: 'heads up' } })
  assert.match(warn, /text-amber-300/)
})

test('the panel still renders the picker inside its full-model arm', () => {
  // The markup moved out; the WIRING must not. Without this the component
  // could be perfect and unreachable — which was the original bug.
  const panel = readPanel()
  const denseArm = panel.slice(
    panel.indexOf('FULL_TRANSFORMER_ADVANCED_BRANCH_START'),
    panel.indexOf('LORA_ADVANCED_CONTROLS_START'))
  assert.match(denseArm, /DENSE_BASE_PICKER_START/)
  assert.match(denseArm, /<DenseBasePicker/)
  assert.match(denseArm, /baseSummary=\{denseBaseSummary\}/)
})

function readPanel() {
  return readFileSync(
    new URL('../src/components/dataset/TrainingPanel.jsx', import.meta.url), 'utf8')
}
