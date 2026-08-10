/**
 * A FACE SWAP THAT CAME BACK EMPTY IS NOT SILENT.
 *
 * A swap overwrites the tile in place, so when it is cancelled or fails the
 * server puts the tile's previous picture back. That restore is invisible by
 * construction — the tile shows the same photo it showed before — and the only
 * trace the server leaves is `fail_reason` on a row that has a file and is not
 * `failed`. That combination is what this badge renders, and nothing else
 * produces it: a genuinely failed row has no file and shows its reason in the
 * placeholder instead.
 *
 * Rendered, not read off the source: whether the badge survives the props it is
 * computed from is exactly the question a regex over the .jsx cannot answer.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { render } from './support/mountJsx.mjs'

const { default: DatasetGridItem } = await import(
  '../src/components/dataset/DatasetGridItem.jsx')

const BASE = {
  id: 11, filename: 'shot.png', status: 'keep', source: 'generated',
  variation_label: 'portrait',
}

const tile = (img) => render(DatasetGridItem, {
  img: { ...BASE, ...img }, datasetId: 3,
  onStatus: () => {}, onCaption: () => {}, onCrop: () => {}, onDelete: () => {},
  onView: () => {}, onToggleSelect: () => {}, onMirror: () => {},
})

test('a restored tile says so, and carries the reason', () => {
  const html = tile({ fail_reason: 'The face swap was cancelled — the original image was restored.' })
  assert.match(html, /↩ restored/)
  assert.match(html, /the original image was restored/)
})

test('an ordinary tile draws no restore badge', () => {
  assert.doesNotMatch(tile({}), /↩ restored/)
})

test('a genuinely failed tile keeps its own placeholder, not the badge', () => {
  const html = tile({ filename: null, status: 'failed', fail_reason: 'ComfyUI said no' })
  assert.doesNotMatch(html, /↩ restored/)
  assert.match(html, /⚠ failed/)
  assert.match(html, /ComfyUI said no/)
})
