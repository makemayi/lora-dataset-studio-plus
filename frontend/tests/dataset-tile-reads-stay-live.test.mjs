/**
 * A DATASET STAYS READABLE WHILE IT WORKS.
 *
 * The complaint: "during an image generation every action around the dataset is
 * blocked — opening the lightbox, deleting a photo…". One `busy` flag guarded
 * the whole tile, so a pass that owns the pixels also switched off the button
 * that OPENS an image and the tick that SELECTS one. Neither writes anything.
 *
 * The decision, taken explicitly rather than inferred: reads come back; writes
 * — deletion included — stay refused, but stop being mute.
 *
 * These assertions are on the RENDERED markup, not on the source text: whether
 * an attribute survives the props it is computed from is exactly what a regex
 * over the .jsx cannot answer, and `onView` being withheld one layer up in
 * DatasetGrid is precisely the kind of second lock a source assertion misses.
 * The same modelled precedent it follows —
 * `backend/tests/test_bank_job_reservations.py::
 *  test_reserved_destination_allows_reads_and_cancel_but_refuses_writes_and_delete`
 * — pins reads and writes in ONE test for the same reason.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { render } from './support/mountJsx.mjs'

const { default: DatasetGridItem } = await import(
  '../src/components/dataset/DatasetGridItem.jsx')
const { datasetBusyReason } = await import(
  '../src/components/dataset/datasetBusyReason.js')

const IMG = {
  id: 7, filename: 'shot.png', status: 'keep', source: 'import',
  caption: 'a caption', variation_label: 'portrait',
}

const GENERATING = {
  kind: 'generate', done: 12, total: 64, started_at: 0,
}

const tile = (props) => render(DatasetGridItem, {
  img: IMG, datasetId: 3,
  onStatus: () => {}, onCaption: () => {}, onCrop: () => {}, onDelete: () => {},
  onView: () => {}, onToggleSelect: () => {}, onMirror: () => {},
  ...props,
})

/* The `<button …>` / `<input …>` opening tag that CONTAINS `needle` — the
   nearest one before it, not merely the nearest `<`: several of these controls
   wrap their glyph in an aria-hidden <span>, and stopping at that span would
   read the attributes of the wrong element. */
const tagAround = (markup, needle) => {
  const at = markup.indexOf(needle)
  assert.ok(at >= 0, `not found in the markup: ${needle}`)
  const open = Math.max(markup.lastIndexOf('<button', at),
    markup.lastIndexOf('<input', at))
  assert.ok(open >= 0, `no control encloses: ${needle}`)
  return markup.slice(open, markup.indexOf('>', open) + 1)
}

/* The ATTRIBUTE, never the Tailwind class: every one of these buttons carries
   `disabled:cursor-not-allowed`, so a bare /disabled/ would pass on a control
   that is perfectly clickable — a test measuring a proxy instead of the
   property. React renders the boolean attribute as ` disabled=""`. */
const DISABLED_ATTR = / disabled=""/

/* The tiles' write buttons, each found by something that does NOT move when
   the refusal takes over its title and aria-label. */
const DELETE_BTN = 'bg-red-700/80'
const CROP_BTN = '✂'
const MIRROR_BTN = '⇆'

test('a running pass no longer switches off inspecting or ticking', () => {
  const busyMarkup = tile({ busy: true, busyReason: datasetBusyReason(GENERATING) })

  const inspect = tagAround(busyMarkup, 'Inspect portrait full screen')
  assert.doesNotMatch(inspect, DISABLED_ATTR,
    'the inspect button is a pure read and must stay clickable during a pass')

  const tick = tagAround(busyMarkup, 'Select portrait for bulk actions')
  assert.doesNotMatch(tick, DISABLED_ATTR,
    'ticking writes nothing on the server and must stay available during a pass')
})

test('the writes stay refused — deletion included, by decision', () => {
  const busyMarkup = tile({ busy: true, busyReason: datasetBusyReason(GENERATING) })
  for (const needle of [DELETE_BTN, CROP_BTN, MIRROR_BTN]) {
    assert.match(tagAround(busyMarkup, needle), DISABLED_ATTR,
      `${needle} touches the dataset and must stay refused while a pass runs`)
  }
})

test('a refused write names the pass holding it, with its progress', () => {
  const reason = datasetBusyReason(GENERATING)
  assert.equal(
    reason,
    '⚡ Variation generation is running on this dataset — 12 / 64. '
    + 'Wait for it to finish, or press ⏹ Stop in the banner at the top of the workspace.')

  const busyMarkup = tile({ busy: true, busyReason: reason })
  // BOTH channels, on every refused write: `title` is what a mouse user reads,
  // `aria-label` is what a screen reader announces. Asserting only that the
  // sentence appears SOMEWHERE in the tag passes when one of the two silently
  // reverts — which is exactly what a mutation of the title alone proved.
  for (const [name, needle] of [['delete', DELETE_BTN], ['crop', CROP_BTN],
    ['mirror', MIRROR_BTN]]) {
    const tag = tagAround(busyMarkup, needle)
    assert.match(tag, /title="⚡ Variation generation is running on this dataset — 12 \/ 64\./,
      `${name} must name the pass on hover`)
    assert.match(tag, /aria-label="⚡ Variation generation is running on this dataset — 12 \/ 64\./,
      `${name} must name the pass to a screen reader`)
  }
})

test('an idle tile keeps its own words — the refusal is not always-on', () => {
  const idle = tile({})
  assert.match(tagAround(idle, DELETE_BTN), /Delete permanently/)
  assert.doesNotMatch(idle, /is running on this dataset/)
  assert.doesNotMatch(tagAround(idle, 'Inspect portrait full screen'), DISABLED_ATTR)
  assert.doesNotMatch(tagAround(idle, DELETE_BTN), DISABLED_ATTR)
})

test('a pass that cannot be stopped from here does not point at a Stop button', () => {
  // `training_export` is not in the backend's STOPPABLE_KINDS, and no Stop
  // control for it exists on this screen. Advice you cannot act on is worse
  // than none.
  const reason = datasetBusyReason({ kind: 'training_export', done: 0, total: 0 })
  assert.match(reason, /^🎓 Training export is running on this dataset\./)
  assert.doesNotMatch(reason, /Stop/)
  assert.match(reason, /inspecting and selecting stay available/)
})

test('a locally-tracked lock with no server snapshot still says something', () => {
  const reason = datasetBusyReason(null)
  assert.match(reason, /^Another pass is running on this dataset\./)
  assert.doesNotMatch(reason, /undefined/)
})

/* ── The grid, one layer up ───────────────────────────────────────────────────
   The tile's `disabled` was only half the lock. DatasetGrid also WITHHELD the
   handlers themselves (`onView={bulkBusy ? undefined : onView}`,
   `onToggleSelect={… && !bulkBusy && …}`), so lifting `disabled` alone would
   have produced a live-looking button that does nothing — the exact failure
   mode a source-text assertion on the tile cannot see. */
const { default: DatasetGrid } = await import(
  '../src/components/dataset/DatasetGrid.jsx')
const { CapabilitiesProvider } = await import('../src/context/CapabilitiesContext.jsx')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
const { renderToStaticMarkup, createElement } = await import('./support/mountJsx.mjs')

const gridHtml = (props) => renderToStaticMarkup(
  createElement(ToastProvider, null,
    createElement(CapabilitiesProvider, null,
      createElement(DatasetGrid, {
        images: [IMG], datasetId: 3,
        onStatus: () => {}, onCaption: () => {}, onCrop: () => {}, onDelete: () => {},
        onView: () => {}, onBatch: () => {}, onMirror: () => {},
        ...props,
      }))))

test('a dataset pass leaves the grid inspectable and tickable', () => {
  const html = gridHtml({ busy: true, activity: GENERATING })
  assert.match(html, /Inspect portrait full screen/)
  assert.doesNotMatch(tagAround(html, 'Inspect portrait full screen'), DISABLED_ATTR)
  // The tick box is RENDERED at all — the grid used to withhold onToggleSelect,
  // which removed the checkbox from the DOM entirely.
  assert.match(html, /Select portrait for bulk actions/)
  assert.doesNotMatch(tagAround(html, 'Select portrait for bulk actions'), DISABLED_ATTR)
  // And the writes are still refused, named.
  assert.match(tagAround(html, DELETE_BTN), DISABLED_ATTR)
  assert.match(tagAround(html, DELETE_BTN), /Variation generation is running/)
})

test('what the pass blocks is said in words, not only in a tooltip', () => {
  // A title is unreadable on a touch screen, which is where "nothing works"
  // was reported. One visible line, above the grid.
  const html = gridHtml({ busy: true, activity: GENERATING })
  assert.match(html, /Edits, captions and deletes wait for the pass above to finish/)
  assert.match(html, /inspecting an image and ticking a selection still work/)
  // It appears only while something is actually holding the dataset.
  assert.doesNotMatch(gridHtml({}), /Edits, captions and deletes wait/)
})
