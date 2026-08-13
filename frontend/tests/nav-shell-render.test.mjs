/**
 * The header, RENDERED. It is on every screen in the app, so a ReferenceError
 * in one of its branches is a white screen everywhere, not on one page — and a
 * source-text test cannot tell a removed branch from a broken one
 * (CLAUDE.md ▸ UI changes, rule 6).
 *
 * It also pins the three things the 2026-08-10 restyle is FOR, so a later
 * "tidy-up" cannot quietly undo them:
 *   · every workspace carries an icon from one set (the bar used to mix three
 *     emoji with two bare labels),
 *   · the active workspace is a primary-filled pill, not just differently-coloured text,
 *   · the bar separates itself by elevation — no permanent bottom hairline.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { MemoryRouter } = await import('react-router')
const { CapabilitiesContext } = await import('../src/context/CapabilitiesContext.jsx')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
const { NavBar } = await import('../src/App.jsx')

// Two rigs: a fresh install (no training, no studio → three of the five
// workspaces are gated off) and a fully configured one.
const CAPS_EMPTY = { engines: {}, comfyui: {}, ollama: {} }
const CAPS_FULL = {
  engines: {}, comfyui: {}, ollama: {},
  training_visible: true, cloud_training: true, studio_visible: true,
}

// ToastProvider: the update-check button asks for a toast on click, and
// `useToast` throws outside it even when nothing is ever clicked.
const render = (caps, path = '/datasets') => renderToStaticMarkup(
  createElement(MemoryRouter, { initialEntries: [path] },
    createElement(ToastProvider, null,
      createElement(CapabilitiesContext.Provider,
        { value: { caps, loading: false, refresh: () => {} } },
        createElement(NavBar)))))

test('the bar renders in both capability rigs, on every route it styles', () => {
  for (const [label, caps] of [['fresh install', CAPS_EMPTY], ['configured', CAPS_FULL]]) {
    for (const path of ['/datasets', '/bank', '/cloud', '/canvas', '/settings', '/guide']) {
      const html = render(caps, path)
      assert.ok(html.includes('LoRA Dataset Studio'), `${label} @ ${path}`)
    }
  }
})

test('every workspace link carries an icon, and no emoji are left in the bar', () => {
  const html = render(CAPS_FULL)
  for (const icon of ['datasets', 'bank', 'runs', 'canvas', 'studio']) {
    assert.match(html, new RegExp(`data-icon="${icon}"`), `${icon} icon missing`)
  }
  // The three the icon set replaced. A mixed bar (some links decorated, some
  // not) is what this change was undoing.
  for (const emoji of ['🗃️', '🏋️', '◉', '🎁', '⬆', '☰']) {
    assert.ok(!html.includes(emoji), `${emoji} is back in the header`)
  }
})

test('the current workspace is a primary-filled pill', () => {
  const html = render(CAPS_FULL, '/bank')
  const link = html.match(/<a[^>]*href="\/bank"[^>]*>/)?.[0]
  assert.ok(link, 'no /bank link in the bar')
  assert.match(link, /rounded-full/)
  assert.match(link, /bg-primary/)
  // …and an inactive one is not.
  const inactive = html.match(/<a[^>]*href="\/canvas"[^>]*>/)?.[0]
  assert.ok(inactive && !/bg-primary/.test(inactive), '/canvas looks active too')
})

test('the header separates by elevation, not by a permanent hairline', () => {
  const html = render(CAPS_FULL)
  const header = html.slice(0, html.indexOf('>') + 1)
  assert.match(header, /sticky/)
  assert.ok(!/border-b/.test(header), 'the bottom border is back')
})

test('the state glyphs stay mounted — a ternary here is the translate crash', () => {
  const html = render(CAPS_FULL)
  // Update check: the arrow and its busy spinner both exist, one hidden.
  assert.match(html, /data-icon="update"/)
  assert.match(html, /data-icon="spinner"/)
  // Mobile menu: hamburger and ✕ both exist, one hidden.
  assert.match(html, /data-icon="menu"/)
  assert.match(html, /data-icon="close"/)
  assert.match(html, /hidden=""/)
})
