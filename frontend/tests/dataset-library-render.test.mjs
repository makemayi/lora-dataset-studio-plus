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

/* THE COVER IS ROUND, AND ITS BOX IS SQUARE — the two go together.
   It used to be a 4:3 banner with `object-cover`, and a reference photo is a
   square head crop: the crop ate the top and bottom of every one of them, so
   the library was a wall of half faces. A square box takes a square image
   whole, and the circle is then only the shape. */
test('a dataset cover is a circle in a square box, not a cropped banner', () => {
  const html = render(DATASETS)
  assert.match(html, /aspect-square[^"]*rounded-full[^"]*object-cover/)
  assert.doesNotMatch(html, /aspect-\[4\/3\]/)
  // The image is still the reference photo, still decorative (the card's own
  // button carries the accessible name).
  assert.match(html, /\/api\/dataset\/1\/img\/ref\.png/)
})

test('a dataset with no reference photo still gets a round initial', () => {
  const html = render(DATASETS)
  // The second fixture has no ref_filename: it falls back to the gradient
  // initial, which must be the same shape as the photo it stands in for.
  assert.match(html, /grid aspect-square[^"]*rounded-full/)
  assert.match(html, />I</)
})

test('the cover answers the pointer, and the placeholder answers it the same way', () => {
  /* The whole card is the button, so the lift is driven by `group-hover` on the
     card rather than by hovering the image itself — otherwise the corner of a
     card would feel dead. The global prefers-reduced-motion reset in index.css
     neutralises the transition, so no `motion-safe:` prefix is needed here. */
  const html = render(DATASETS)
  const hovers = html.match(/group-hover:scale-105/g) || []
  assert.equal(hovers.length, 2, 'the photo and the initial must lift alike')
  assert.match(html, /group-hover:ring-2 group-hover:ring-indigo-400\/70/)
  // A circle that grows must not be clipped by the box it grows in.
  assert.match(html, /flex items-center justify-center overflow-hidden bg-surface-raised/)
})

test('export, backup, edit and delete are one cluster of icon buttons', () => {
  /* They are the same weight of action — taken rarely, never in bulk — so they
     get the same weight of control. The labelled export bar that used to sit
     under every card cost a row of height on every tile for two of them. */
  const html = render(DATASETS)
  for (const label of [/Export training ZIP for Emma/, /Export portable backup for Emma/,
                       /Edit settings for Emma/, /Delete the dataset Emma/]) {
    assert.match(html, label)
  }
  // The labelled bar under the card is gone — its own container, not the word
  // "Backup", which the library-wide backup menu in the header also uses.
  assert.doesNotMatch(html, /library-card__actions grid grid-cols-2/)
  // ...and all four now live in the one hover-revealed cluster.
  const cluster = html.slice(html.indexOf('library-card__actions absolute'))
  assert.match(cluster.slice(0, 3000), /data-icon="download"[\s\S]*data-icon="trash"/)
  // A dataset with nothing kept cannot export a training ZIP, and the button
  // says why rather than going quietly grey.
  const empty = render([{ ...DATASETS[0], id: 7, images_kept: 0, images_captioned: 0 }])
  assert.match(empty, /Keep at least one image before exporting a training ZIP/)
  assert.match(empty, /disabled=""/)
})
