/**
 * The dataset library, RENDERED — the app's landing page, and until now only
 * checked as source text (tests/dataset-library-contract.test.mjs). It has four
 * branches that never met a renderer: the empty state, the photo tile, the
 * compact row and the new-dataset form.
 *
 * It also pins the 2026-08-10 restyle: cards separate by elevation, the
 * controls are pills, and the emoji that stood in for the per-card actions
 * (⬇ 💾 ⚙️ 🗑) are drawn icons.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { MemoryRouter } = await import('react-router')   // the help badges navigate
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
const { default: DatasetListPanel } = await import(
  '../src/components/dataset/DatasetListPanel.jsx')

const DATASETS = [
  {
    id: 1, name: 'Emma', kind: 'character', trigger_word: 'zchar_emma',
    ref_filename: 'ref.png', images_total: 40, images_kept: 32, images_captioned: 32,
    trained_families: ['krea'],
  },
  {
    id: 2, name: 'ink-wash', kind: 'style', trigger_word: null,
    images_total: 18, images_kept: 12, images_captioned: 4, trained_families: [],
  },
]

const noop = () => {}
const render = (datasets) => renderToStaticMarkup(
  createElement(MemoryRouter, null,
    createElement(ToastProvider, null,
      createElement(DatasetListPanel, {
      datasets,
      onOpen: noop, onCreate: noop, onDelete: noop, onRestore: noop,
        onExportZip: noop, onExportBackup: noop, onSettingsSave: noop,
        backup: null,
      }))))

test('an empty library renders its hero instead of an empty page', () => {
  const html = render([])
  assert.match(html, /No datasets yet/)
  assert.match(html, /Reference photo/)   // the 3-step strip
})

test('a populated library renders its cards', () => {
  const html = render(DATASETS)
  assert.match(html, /Emma/)
  assert.match(html, /ink-wash/)
  // Style datasets say so instead of showing an empty trigger.
  assert.match(html, /always-on/)
})

test('the per-card actions are drawn icons, not emoji', () => {
  const html = render(DATASETS)
  for (const icon of ['download', 'bank', 'settings', 'trash']) {
    assert.match(html, new RegExp(`data-icon="${icon}"`), `${icon} icon missing`)
  }
  for (const emoji of ['⬇', '💾', '⚙️', '🗑']) {
    assert.ok(!html.includes(emoji), `${emoji} is back on a library card`)
  }
})

test('library cards separate by elevation, not by an outline', () => {
  const html = render(DATASETS)
  const card = html.match(/<div class="library-card [^"]*"/)?.[0]
  assert.ok(card, 'no library card in the markup')
  assert.match(card, /shadow-\[/)
  assert.ok(!/border border-border/.test(card), 'the card outline is back')
})
