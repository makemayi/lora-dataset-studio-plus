/**
 * What the two Krea calibration sliders actually SHOW — rendered, not grepped.
 *
 * The source-text contract next to the component
 * (src/components/dataset/kreaTuningPanel.test.js) can prove the sliders are
 * wired. It cannot prove the one thing a user reads: whether the number on
 * screen matches the value in state, and whether "Reset to default" is there
 * exactly when the dial is off its shipped value. That button's PRESENCE is the
 * only "you changed this" marker in the panel — a version that always renders,
 * or never renders, would pass every regex and tell the user nothing.
 *
 * `renderToStaticMarkup` runs no effects and fires no events; this asks the one
 * question it can answer — given this state, what is on screen.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { KreaDial } = await import('../src/components/dataset/VariationCatalog.jsx')
// The help badge beside each label calls useNavigate(), so the dial only renders
// inside a router — mounting it without one is not a state a user can reach.
const { MemoryRouter } = await import('react-router')
const { refBoostDescription, identityStrengthDescription } = await import(
  '../src/utils/kreaDials.js')

const dial = (props) => renderToStaticMarkup(
  createElement(MemoryRouter, null, createElement(KreaDial, {
    id: 'krea-ref-boost',
    label: 'Reference pull',
    topic: 'krea.ref_boost',
    value: 0.25,
    min: 0,
    max: 10,
    step: 0.25,
    description: refBoostDescription(0.25),
    defaultValue: 0.25,
    onChange: () => {},
    children: 'How hard the source latent is pushed back in.',
    ...props,
  })))

test('the slider renders its own value, its bounds and what that value means', () => {
  const html = dial({ value: 3, description: refBoostDescription(3) })
  assert.match(html, /type="range"/)
  assert.match(html, /min="0"/)
  assert.match(html, /max="10"/)
  assert.match(html, /step="0.25"/)
  assert.match(html, /value="3"/)
  // The number is never alone: a bare 3 is not a setting.
  assert.ok(html.includes('reference-dominated'),
    'the dial position is not explained on screen')
  assert.ok(html.includes('(3)'), 'the current value is not shown beside the label')
})

test('a dial sitting on its default offers no reset — the button IS the marker', () => {
  const html = dial({ value: 0.25, defaultValue: 0.25 })
  assert.ok(!html.includes('Reset to default'),
    'an inert reset button on an unchanged dial is noise')
})

test('a dial moved off its default offers the way back, naming the value', () => {
  const html = dial({ value: 4, defaultValue: 0.25, description: refBoostDescription(4) })
  assert.ok(html.includes('Reset to default'), 'no way back from a changed dial')
  // The accessible name says WHICH field and WHICH value — a dozen identical
  // "Reset to default" buttons on one screen announce nothing.
  assert.match(html, /aria-label="Reset to default: Reference pull, 0\.25"/)
})

test("a config sent by a backend with no config_defaults hides the button", () => {
  // defaultValueAt returns undefined there. Offering a reset we cannot honour
  // would write `undefined` into the settings.
  const html = dial({ value: 4, defaultValue: undefined })
  assert.ok(!html.includes('Reset to default'))
})

test('the identity dial renders its own vocabulary, including 0 = off', () => {
  const off = dial({
    id: 'krea-identity-lora-strength',
    label: 'Identity LoRA strength',
    topic: 'krea.identity_lora_strength',
    value: 0, min: 0, max: 1.5, step: 0.05, defaultValue: 1,
    description: identityStrengthDescription(0),
  })
  assert.ok(off.includes('identity LoRA off'), '0 must read as off, not as a small number')
  assert.match(off, /aria-label="Reset to default: Identity LoRA strength, 1"/)
  const trained = dial({
    id: 'krea-identity-lora-strength',
    label: 'Identity LoRA strength',
    value: 1, min: 0, max: 1.5, step: 0.05, defaultValue: 1,
    description: identityStrengthDescription(1),
  })
  assert.ok(trained.includes('trained for'))
  assert.ok(!trained.includes('Reset to default'))
})

test('the label is tied to the input, so the slider is reachable by its name', () => {
  const html = dial({ id: 'krea-ref-boost' })
  assert.match(html, /for="krea-ref-boost"/)
  assert.match(html, /id="krea-ref-boost"/)
})
