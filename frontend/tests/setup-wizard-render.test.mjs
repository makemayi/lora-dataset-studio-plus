/**
 * The setup wizard, RENDERED — the first thing that has ever executed this
 * 1,400-line page. It is the first screen a new install shows, and until now
 * only one of its child components (InstallEverything) was ever mounted by a
 * test: a ReferenceError anywhere else in it would have reached a user before
 * it reached the suite.
 *
 * It also pins the 2026-08-10 pass: the wizard shares the app's card surface
 * and button shapes (components/common/surfaces.js) instead of carrying its own
 * outlined boxes and its own copy of the input class.
 */
import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { MemoryRouter } = await import('react-router')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
const { CapabilitiesContext } = await import('../src/context/CapabilitiesContext.jsx')
const { default: SetupPage } = await import('../src/pages/SetupPage.jsx')

const CAPS = { engines: {}, comfyui: {}, ollama: {}, python: {} }

const render = () => renderToStaticMarkup(
  createElement(MemoryRouter, null,
    createElement(ToastProvider, null,
      createElement(CapabilitiesContext.Provider,
        { value: { caps: CAPS, loading: false, refresh: () => {} } },
        createElement(SetupPage)))))

/* ⚠️ Read this before trusting a green run. The wizard renders "Loading setup…"
   until GET /api/setup/config answers, and `useEffect` does not run server-side
   — so this mounts the page and its module graph, and reaches the loading
   branch only. It catches an import cycle, a bad module-level constant and a
   crash on the way in; it does NOT execute the six screens. Making those
   testable means lifting them out of the page, which is a refactor, not a
   restyle. */
test('the wizard mounts, and reaches its loading branch', () => {
  assert.match(render(), /Loading setup…/)
})

test('the wizard uses the shared surfaces, not its own outlined boxes', () => {
  const source = readFileSync(new URL('../src/pages/SetupPage.jsx', import.meta.url), 'utf8')
  assert.match(source, /from '\.\.\/components\/common\/surfaces'/)
  // Its own duplicate of the input class is gone…
  assert.doesNotMatch(source, /^const INPUT_CLASS =/m)
  // …and no card in it draws an outline any more.
  assert.doesNotMatch(source, /rounded-xl border border-border bg-surface/)
  assert.doesNotMatch(source, /bg-white\/5/)
})
