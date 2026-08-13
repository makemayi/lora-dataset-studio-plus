import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { CARD_SURFACE } from '../src/components/common/surfaces.js'
import tailwindConfig from '../tailwind.config.js'

const here = dirname(fileURLToPath(import.meta.url))
const frontend = join(here, '..')
const read = (rel) => readFileSync(join(frontend, rel), 'utf8')

test('the page ground carries a static ambient-light veil', () => {
  const css = read('src/index.css')
  assert.match(css, /body::before\s*\{[^}]*position:\s*fixed/s)
  assert.match(css, /body::before\s*\{[^}]*pointer-events:\s*none/s)
  assert.match(css, /body::before\s*\{[^}]*radial-gradient/s)
  // static — no animation on the veil, so reduced-motion is satisfied by construction
  assert.doesNotMatch(css, /body::before\s*\{[^}]*animation:/s)
})

test('cards catch a top light edge via a CSS class, not an opacity modifier', () => {
  const css = read('src/index.css')
  assert.match(css, /\.card-light-edge\s*\{[^}]*linear-gradient\(180deg,\s*rgba\(255/s)
})

test('CARD_SURFACE carries the light edge', () => {
  assert.match(CARD_SURFACE, /card-light-edge/)
})
