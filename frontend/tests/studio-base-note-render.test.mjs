/**
 * What the Studio actually SHOWS when its Krea base was not the one Setup
 * installs — rendered, not grepped.
 *
 * The note exists for exactly one install: the one with a single Krea file on
 * disk. That install has no ALTERNATIVE to offer, so the base picker is not
 * rendered at all (`zModels.length > 1`). A note placed inside that guard would
 * be correct in the payload, correct in the test that reads the payload, and
 * invisible to the only person who needs it. Only a render can tell those apart,
 * so this mounts the component with an EMPTY model list and asserts the sentence
 * is on screen anyway.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { default: AxisPickers } = await import(
  '../src/components/dataset/studio/AxisPickers.jsx')

const NOTE = 'The Krea 2 Turbo base Setup installs (krea2_turbo_fp8_scaled.safetensors) '
  + 'is not on this machine, so the Studio renders on the best Krea 2 build it found '
  + 'here: someones_turbo_repack.safetensors.'

const render = (props) => renderToStaticMarkup(createElement(AxisPickers, {
  zModels: null, effectiveModels: [], onToggleModel: () => {},
  aspects: null, effectiveAspects: [], onToggleAspect: () => {},
  cfgChoices: null, effectiveCfgs: [], onToggleCfg: () => {},
  stepsChoices: null, effectiveSteps: [], onToggleStep: () => {},
  steps2Choices: null, effectiveSteps2: [], onToggleStep2: () => {},
  fmt: (v) => String(v),
  ...props,
}))

test('the note reaches the install that has no base picker at all', () => {
  const html = render({ baseNote: NOTE, zModels: [] })
  assert.ok(html.includes('krea2_turbo_fp8_scaled.safetensors'),
    'the note vanished on the one install it was written for')
  assert.ok(!html.includes('Base model (multi)'),
    'this install has nothing to pick between — the picker must stay hidden')
})

test('the note also shows above a picker that IS rendered', () => {
  const html = render({
    baseNote: NOTE,
    zModels: [{ value: '', label: 'Default – someones_turbo_repack' },
      { value: 'Krea\\other.safetensors', label: 'other' }],
  })
  assert.ok(html.includes('krea2_turbo_fp8_scaled.safetensors'))
  assert.ok(html.includes('Base model (multi)'))
  // Above, not below: the reader must know what the default IS before choosing.
  assert.ok(html.indexOf('krea2_turbo_fp8_scaled') < html.indexOf('Base model (multi)'))
})

test('nothing is rendered when there is nothing to say', () => {
  const html = render({ baseNote: null })
  assert.equal(html.includes('text-amber-300/80'), false,
    'a null note must not leave an empty coloured paragraph behind')
})

test('the note wraps instead of pushing the panel sideways at 400 px', () => {
  // The panel is read on a phone. A 90-character filename inside a fixed-width
  // column is a horizontal scrollbar on the whole page unless it may break.
  const html = render({ baseNote: NOTE, zModels: [] })
  const paragraph = html.slice(html.indexOf('<p'), html.indexOf('</p>'))
  assert.match(paragraph, /break-words/)
  assert.match(paragraph, /leading-snug/)
})
