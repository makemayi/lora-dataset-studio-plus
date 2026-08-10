/**
 * The two bank LIST pages, rendered — and the surface language they now share
 * with Settings and the nav bar pinned down.
 *
 * The 2026-08-10 pass took the outline off every card (the tokens are already an
 * elevation system — see components/common/surfaces.js), replaced the emoji that
 * stood in for controls (📦 move, ✕ remove, ➕ create, 🔄 rescan) with the drawn
 * icon set, and gave both pages the mono eyebrow every other section has. All
 * three are easy to undo by accident from a "tidy the classes" edit, so they are
 * assertions rather than a comment.
 *
 * The CARDS are rendered directly: both pages show "Loading…" until the server
 * answers, so rendering a page alone executes none of the card markup.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { MemoryRouter } = await import('react-router')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
const { default: BankPage, BankCard } = await import('../src/pages/BankPage.jsx')
const { default: VideoBankPage, VideoBankCard } = await import('../src/pages/VideoBankPage.jsx')

const BANK = {
  id: 3,
  name: 'Telegram export 07/2026',
  source_path: 'D:\\unsorted\\telegram',
  total: 812,
  preview_ids: [1, 2, 3],
  counts: { keep: 120, pending: 600, reject: 92 },
  scanned: 812,
}

const page = (Component) => renderToStaticMarkup(
  createElement(MemoryRouter, null,
    createElement(ToastProvider, null, createElement(Component))))

const card = (element) => renderToStaticMarkup(
  createElement(MemoryRouter, null, createElement(ToastProvider, null, element)))

test('both list pages render', () => {
  assert.match(page(BankPage), /<h1[^>]*>Image bank<\/h1>/)
  assert.match(page(VideoBankPage), /Video bank/)
})

test('every list page carries the mono eyebrow the rest of the app uses', () => {
  for (const html of [page(BankPage), page(VideoBankPage)]) {
    assert.match(html, /tracking-\[0\.18em\][^>]*>bank</)
  }
})

test('a bank card renders, with drawn icons instead of emoji controls', () => {
  const html = card(createElement(BankCard, {
    bank: BANK, onOpen: () => {}, onRelocate: () => {}, onRemove: () => {},
  }))
  for (const icon of ['move', 'close', 'arrow-right']) {
    assert.match(html, new RegExp(`data-icon="${icon}"`), `${icon} icon missing`)
  }
  for (const emoji of ['📦', '✕', '⏳']) {
    assert.ok(!html.includes(emoji), `${emoji} is back on the bank card`)
  }
})

test('a running pass still says so on the card', () => {
  const html = card(createElement(BankCard, {
    bank: { ...BANK, activity: { kind: 'quality', finished: false } },
    onOpen: () => {}, onRelocate: () => {}, onRemove: () => {},
  }))
  assert.match(html, /quality…/)
  assert.match(html, /data-icon="spinner"/)
})

test('cards separate by elevation — no outline on either kind', () => {
  const rigs = [
    card(createElement(BankCard, {
      bank: BANK, onOpen: () => {}, onRelocate: () => {}, onRemove: () => {},
    })),
    card(createElement(VideoBankCard, {
      bank: { ...BANK, counts: { total: 3, kept: 1 } }, onOpen: () => {}, onRemove: () => {},
    })),
  ]
  for (const html of rigs) {
    const li = html.match(/<li[^>]*>/)?.[0]
    assert.ok(li, 'no card element')
    assert.match(li, /shadow-\[/)
    assert.ok(!/border border-border/.test(li), 'the card outline is back')
  }
})
