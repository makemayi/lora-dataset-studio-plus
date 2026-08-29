/**
 * A workspace this install cannot use yet is DISABLED, never absent.
 *
 * Reported 2026-08-29: "LoRA Studio — why can't I see this page any more?".
 * ComfyUI had been closed to free VRAM for a training run, and `studio_visible`
 * went false — which removed Test Studio from the bar entirely, along with Runs
 * and Canvas. Nothing said the app had taken them away, which ones had gone, or
 * how to get them back; the answer was only readable in capabilities.py.
 *
 * Mounted, not grepped: the whole point is what the bar RENDERS in each state,
 * and both variants have to survive being drawn (they stay mounted and are
 * swapped by class, because a ternary that unmounts one is what Chrome
 * auto-translate turns into a removeChild crash).
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { GatedNavItem } = await import('../src/App.jsx')
// NavLink resolves its path against the router, so the item needs one to render
// at all — an in-memory one is enough, nothing here navigates.
const { MemoryRouter } = await import('react-router')

const item = (available) => renderToStaticMarkup(createElement(
  MemoryRouter, { initialEntries: ['/datasets'] },
  createElement(
    GatedNavItem,
    { to: '/studio', available, hint: 'ComfyUI is not reachable — start it.' },
    'Test Studio',
  ),
))

test('an unavailable workspace still shows, greyed, carrying its reason', () => {
  const html = item(false)
  assert.match(html, /Test Studio/, 'the destination stays on screen')
  assert.match(html, /aria-disabled="true"/)
  assert.match(html, /ComfyUI is not reachable/, 'the reason travels with it')
  assert.match(html, /cursor-not-allowed/)
  // The real link is the one hidden in this state — never removed.
  assert.match(html, /<a[^>]*class="hidden"/)
})

test('an available workspace is an ordinary link', () => {
  const html = item(true)
  assert.match(html, /<a[^>]*href="\/studio"/)
  assert.doesNotMatch(html, /<a[^>]*class="hidden"/)
  // …and the disabled twin is the hidden one, still mounted.
  assert.match(html, /<span[^>]*class="hidden"/)
})

test('both states keep the label text mounted', () => {
  // Two copies of the label in each state: one shown, one hidden. This is the
  // property that keeps auto-translate from crashing the header on a flip.
  for (const available of [true, false]) {
    const copies = item(available).match(/Test Studio/g) || []
    assert.equal(copies.length, 2, `both variants render in state ${available}`)
  }
})

test('an unknown capability is not treated as a missing one', async () => {
  // `caps` starts as an all-false PLACEHOLDER and the first probe measured 37s
  // on a real machine (it lists ComfyUI's models and asks /object_info). For
  // those 37 seconds the header greyed out workspaces on an install where
  // ComfyUI was running, and lit the Setup dot as if nothing were configured —
  // reported as "why does the menu go grey and why does it always want setup?".
  const { readFileSync } = await import('node:fs')
  const src = readFileSync(new URL('../src/App.jsx', import.meta.url), 'utf8')
  assert.match(src, /const \{ caps, loading: capsUnknown \} = useCapabilities\(\)/)
  // The two trainer-gated items read the unknown as "not yet". (Test Studio is
  // gated on the cheap liveness poll instead — see comfyui-alive-gate.test.mjs.)
  assert.equal((src.match(/available=\{capsUnknown \|\| Boolean\(/g) || []).length, 2)
  assert.match(src, /const setupNeedsAttention = !capsUnknown && !recommendedMet\(caps\)/)
  assert.doesNotMatch(src, /available=\{Boolean\(caps\./,
    'no gated item may read the placeholder as an answer')
})
