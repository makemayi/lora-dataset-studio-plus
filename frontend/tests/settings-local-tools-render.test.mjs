/**
 * The Local tools settings section, RENDERED — the vision-server card gained a
 * second engine (llama.cpp) whose fields live in a `hidden` branch. A branch no
 * test mounts is a branch that ships broken (and a settings page that throws is
 * a settings page nobody can open), so both engine lanes are mounted here.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { default: LocalToolsSection } = await import(
  '../src/components/settings/LocalToolsSection.jsx')

const CONFIG = (visionBackend) => ({
  comfyui: { api_url: '', base_dir: '' },
  ollama: { url: 'http://127.0.0.1:11434', vision_model: 'm:8b' },
  llamacpp: { url: 'http://127.0.0.1:8080', vision_model: '' },
  vision: { backend: visionBackend },
  aitoolkit: {},
  onetrainer: {},
  comfyFolders: {},
})

const render = (backend) => renderToStaticMarkup(createElement(LocalToolsSection, {
  config: CONFIG(backend),
  setField: () => {},
  configDefaults: {},
  testResults: {},
  recordTestResult: () => {},
  saveConfigSection: () => {},
  caps: { ollama: { reachable: true, installed: true }, llamacpp: { reachable: false } },
  refreshCaps: () => {},
  toast: { error: () => {}, success: () => {} },
  // SecretField props (spread into the card via {...props}); presence values
  // are irrelevant here — the mount only has to not throw.
  secretsPresence: {},
  secretInputs: {},
  setSecretInputs: () => {},
  saveSecretIfPending: () => {},
  handleDeleteSecret: () => {},
}))

test('the vision-server card renders with the engine select in both lanes', () => {
  for (const backend of ['ollama', 'llamacpp']) {
    const html = render(backend)
    assert.match(html, /id="vision-backend"/)
    assert.match(html, /value="ollama"/)
    assert.match(html, /value="llamacpp"/)
  }
})

// React emits the branch div's hidden flag BEFORE its class attribute.
const OLLAMA_BRANCH = '<div class="space-y-4 pt-2">'
const LLAMACPP_BRANCH = '<div hidden="" class="space-y-4 pt-2">'

test('Ollama lane: ollama fields visible, llama.cpp branch hidden but mounted', () => {
  const html = render('ollama')
  assert.ok(html.includes(OLLAMA_BRANCH), 'ollama branch must not be hidden')
  assert.ok(html.includes(LLAMACPP_BRANCH), 'llamacpp branch must be mounted AND hidden')
  assert.match(html, /id="llamacpp-url"/)   // mounted, never unmounted
})

test('llama.cpp lane: its fields visible, ollama branch hidden but mounted', () => {
  const html = render('llamacpp')
  assert.ok(html.includes(LLAMACPP_BRANCH), 'ollama branch must be hidden')
  assert.ok(html.includes(OLLAMA_BRANCH), 'ollama branch must stay mounted')
  assert.match(html, /Start\s+llama-server yourself/)
  assert.match(html, /id="ollama-url"/)     // mounted, never unmounted
})
