import test from 'node:test'
import assert from 'node:assert/strict'

import { render } from './support/mountJsx.mjs'

const { default: SubjectTrimReview } = await import(
  '../src/components/dataset/SubjectTrimReview.jsx')

const ITEMS = [
  { image_id: 1, filename: 'a.png', image: [2000, 3000], frame: [400, 100, 900, 1600], out: [576, 1024], skip: null },
  { image_id: 2, filename: 'b.png', image: [2000, 3000], frame: [10, 10, 1980, 2980], out: [680, 1024], skip: 'nothing-much-to-remove' },
  { image_id: 3, filename: 'c.png', image: [1000, 1000], frame: null, out: null, skip: 'no-subject-found' },
]

const review = (over = {}) => render(SubjectTrimReview, {
  datasetId: 4, preview: { items: ITEMS },
  onApply: async () => {}, onDiscard: async () => {}, ...over,
})

test('every proposed crop renders, and the skipped ones are separated', () => {
  const html = review()
  assert.match(html, /1 proposed/)
  assert.match(html, /a\.png/)
  assert.match(html, /2 left alone/)
  // Both skip reasons are shown as text, not as codes.
  assert.match(html, /Nothing much to remove/)
  assert.match(html, /No subject found/)
  assert.doesNotMatch(html, /nothing-much-to-remove/)
})

test('a skipped row carries no checkbox', () => {
  const checkboxes = review().match(/type="checkbox"/g) || []
  assert.equal(checkboxes.length, 1, 'only the one applicable row is selectable')
})

test('an empty preview renders the empty state rather than a blank panel', () => {
  assert.match(review({ preview: { items: [] } }), /Nothing to crop/)
})

test('a preview where every row was skipped says so', () => {
  const html = review({ preview: { items: [ITEMS[1], ITEMS[2]] } })
  assert.match(html, /Nothing to crop/)
  assert.match(html, /Nothing much to remove/)
})

test('a stopped row says the user stopped it, not that it failed', () => {
  // A Stop is not a failure. The server keeps them apart; so must the screen.
  const stopped = { image_id: 9, filename: 'd.png', image: [800, 800], frame: null, out: null, skip: 'stopped' }
  const html = review({ preview: { items: [stopped] } })
  assert.match(html, /you stopped/i)
  assert.doesNotMatch(html, /could not be measured/i)
})

test('a crashed measuring pass shows its error instead of "nothing to crop"', () => {
  // The server writes an EMPTY manifest carrying `error` when the pass throws,
  // so that a crash has somewhere to land. Ignoring it would render the same
  // screen a successful no-op produces.
  const html = review({ preview: { items: [], error: 'The crop preview failed.' } })
  assert.match(html, /The crop preview failed\./)
})

test('both state texts stay mounted rather than being swapped', () => {
  // Chrome auto-translate rewrites text nodes and React then throws
  // NotFoundError: removeChild on a ternary swap. Both variants render; one
  // carries hidden.
  assert.match(review(), /hidden=""/)
})

test('the after pane is offset by the frame, not just scaled', () => {
  // The whole point of the pane: it must show WHAT WILL BE KEPT. A pane that
  // scaled without offsetting would show the top-left corner of every photo and
  // look plausible on a centred subject.
  assert.match(review(), /left:\s*-44\.4/)      // -400/900 * 100
})
