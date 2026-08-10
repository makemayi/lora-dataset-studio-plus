/**
 * The Guide, rendered. Nothing mounted this page before — it is 200 KB of
 * markdown behind a chapter rail, a mobile chip row, two "on this page" navs
 * and a prev/next footer, and every one of those branches was only ever checked
 * by eye. A source-text test cannot tell a removed branch from a broken one
 * (CLAUDE.md ▸ UI changes, rule 6).
 *
 * It also pins what the 2026-08-10 pass changed: the panels separate by
 * elevation, and the current chapter is a filled pill like every other
 * "where am I" in the app.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { MemoryRouter, Routes, Route } = await import('react-router')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
const { default: GuidePage } = await import('../src/pages/GuidePage.jsx')

const at = (path, props = {}) => renderToStaticMarkup(
  createElement(MemoryRouter, { initialEntries: [path] },
    createElement(ToastProvider, null,
      createElement(Routes, null,
        createElement(Route, {
          path: '/guide/:section', element: createElement(GuidePage, props),
        }),
        createElement(Route, {
          path: '/help', element: createElement(GuidePage, { helpOnly: true }),
        })))))

test('every chapter renders, and the help-only variant too', () => {
  for (const id of ['getting-started', 'using-the-app', 'dataset-guide',
    'settings-reference', 'troubleshooting']) {
    const html = at(`/guide/${id}`)
    assert.match(html, /Field manual/, id)
    assert.match(html, /min read/, id)
  }
  assert.match(at('/help'), /Getting help/)
})

test('the chapter you are reading is a filled pill, the others are plain', () => {
  const html = at('/guide/troubleshooting')
  const current = html.match(/<button[^>]*aria-current="page"[^>]*>/g) || []
  assert.ok(current.length >= 1, 'no chapter marked current')
  for (const button of current) assert.match(button, /bg-surface-raised/)
})

test('the chapter header separates by elevation, not by an outline', () => {
  const html = at('/guide/getting-started')
  const header = html.match(/<header[^>]*>/)?.[0]
  assert.ok(header, 'no chapter header')
  assert.match(header, /shadow-\[/)
  assert.ok(!/border border-border/.test(header), 'the header outline is back')
})

test('a chapter in the middle offers both neighbours', () => {
  const html = at('/guide/dataset-guide')
  assert.match(html, /Previous/)
  assert.match(html, /Next/)
  // The first chapter has no previous — the slot stays empty rather than
  // shifting "Next" to the left half.
  assert.ok(!at('/guide/getting-started').includes('>Previous<'))
})
