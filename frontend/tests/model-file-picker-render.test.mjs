/**
 * What the model-file picker actually SHOWS — rendered, not grepped.
 *
 * The pure-logic test next to the util can prove buildModelOptions keeps a
 * pinned-but-absent value first. It cannot prove the one thing a user sees:
 * that the value is still IN the input (not blanked), and that the "not found"
 * badge is on screen. A version that computed the right options and rendered an
 * empty box would pass every regex in the util test.
 *
 * `renderToStaticMarkup` runs no effects and fires no events, so the scan is
 * passed in as props — which is exactly how the component receives it in the
 * app (the hook is called by the card, the component is dumb).
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const ModelFilePicker = (await import('../src/components/settings/ModelFilePicker.jsx')).default

const FILES = ['klein/flux-2-klein-9b-fp8.safetensors', 'Krea/krea2_turbo_fp8_scaled.safetensors']

const pick = (props) => renderToStaticMarkup(createElement(ModelFilePicker, {
  id: 'krea-base-model',
  ariaLabel: 'Krea base model file',
  value: '',
  onChange: () => {},
  placeholder: 'auto — finds a Krea 2 Turbo/Raw build',
  files: FILES,
  folder: 'ComfyUI’s models/unet',
  loading: false,
  error: false,
  rescan: () => {},
  rescanning: false,
  ...props,
}))

test('a pinned file that is not on disk is still in the field, and flagged', () => {
  const html = pick({ value: 'krea/my-own-build.safetensors' })
  assert.ok(html.includes('value="krea/my-own-build.safetensors"'),
    'the pinned value was blanked — that is the silent reset this feature exists to prevent')
  assert.ok(html.includes('not found'), 'nothing on screen says the file is absent')
})

test('a value that exists gets no alarm badge', () => {
  const html = pick({ value: FILES[1] })
  assert.ok(html.includes(`value="${FILES[1]}"`))
  assert.ok(!html.includes('not found'), 'a present file must not be accused')
})

test('an empty value renders the placeholder and no badge', () => {
  const html = pick({ value: '' })
  assert.ok(html.includes('placeholder='))
  assert.ok(!html.includes('not found'))
})

test('the field is a combobox and keeps its rescan control', () => {
  const html = pick({})
  assert.match(html, /role="combobox"/)
  assert.ok(html.includes('Rescan model files for Krea base model file'),
    'no way to pick up a file dropped in while the panel was open')
})

test('rendering while the scan is still running does not throw and accuses nothing', () => {
  // The state every install is in for the first moment, and the one a slow or
  // remote model mount can stay in for seconds. It must never read as "missing".
  const html = pick({ value: 'krea/whatever.safetensors', files: [], loading: true })
  assert.ok(!html.includes('not found'))
})
