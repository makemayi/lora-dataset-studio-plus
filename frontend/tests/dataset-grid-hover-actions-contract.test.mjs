import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const frontend = path.resolve(here, '..')
const item = fs.readFileSync(path.join(frontend, 'src/components/dataset/DatasetGridItem.jsx'), 'utf8')
const css = fs.readFileSync(path.join(frontend, 'src/index.css'), 'utf8')

test('all image-card control groups share the hover-action contract', () => {
  // The tile is now the photo itself: the surface token carries radius+clip,
  // and the resting shadow is swapped for the selection glow (no ring).
  assert.match(item, /dataset-grid-item \$\{TILE_SURFACE\}/)
  assert.doesNotMatch(item, /rounded-\[28px\]/)
  assert.doesNotMatch(item, /bg-white\/60/)
  // Three groups: bulk-select tick, the hover toolbar, and the keep/reject
  // row. The caption is deliberately NOT here — it gates through its own
  // .dataset-grid-item__caption class so a touch screen does not resurrect
  // the always-on bottom plate (see index.css).
  assert.ok((item.match(/dataset-grid-item__actions/g) || []).length >= 3)
  assert.match(item, /dataset-grid-item__caption[^"']*absolute inset-x-2 bottom-14/)
})

test('fine pointers hide controls without reflow and hover or focus reveals them', () => {
  assert.match(css, /@media \(hover: hover\) and \(pointer: fine\)/)
  assert.match(css, /\.dataset-grid-item:hover \.dataset-grid-item__actions/)
  assert.match(css, /\.dataset-grid-item:focus-within \.dataset-grid-item__actions/)
  assert.match(css, /visibility: hidden/)
  assert.match(css, /pointer-events: none/)
  assert.doesNotMatch(css, /\.dataset-grid-item \.dataset-grid-item__actions\s*\{[^}]*display:\s*none/s)
})
