import assert from 'node:assert/strict'
import test from 'node:test'

import {
  IMPROVE_DERIVATION,
  improvementBadge,
  improvementStateByParent,
} from './improveCandidates.js'

const candidate = (over = {}) => ({
  id: 99, derivation_kind: IMPROVE_DERIVATION, status: 'pending',
  parent_image_id: 1, filename: 'out.png', ...over,
})

test('a finished candidate marks its SOURCE, not itself', () => {
  // The point of the badge: from the source you were looking at, nothing had
  // changed — the result lands on its own tile somewhere else in the grid.
  const state = improvementStateByParent([
    { id: 1, derivation_kind: null, status: 'keep', filename: 'src.webp' },
    candidate(),
  ])
  assert.equal(state.get(1), 'ready')
  assert.equal(state.get(99), undefined)
})

test('a candidate still rendering says so instead of promising a result', () => {
  const state = improvementStateByParent([candidate({ filename: null })])
  assert.equal(state.get(1), 'generating')
})

test('a candidate already reviewed leaves no badge behind', () => {
  for (const status of ['keep', 'reject', 'failed']) {
    const state = improvementStateByParent([candidate({ status })])
    assert.equal(state.get(1), undefined, `${status} must clear the badge`)
  }
})

test('ready wins over generating when a source has both', () => {
  // A file you can look at is the state that should stop you re-running the
  // pass; a second one still cooking does not change that.
  const state = improvementStateByParent([
    candidate({ id: 98, filename: null }),
    candidate({ id: 99, filename: 'out.png' }),
  ])
  assert.equal(state.get(1), 'ready')
  const reversed = improvementStateByParent([
    candidate({ id: 99, filename: 'out.png' }),
    candidate({ id: 98, filename: null }),
  ])
  assert.equal(reversed.get(1), 'ready')
})

test('rows that are not improve candidates are ignored', () => {
  const state = improvementStateByParent([
    { id: 5, derivation_kind: 'klein_small_image', status: 'pending',
      parent_image_id: 1, filename: 'x.png' },
    { id: 6, derivation_kind: IMPROVE_DERIVATION, status: 'pending',
      parent_image_id: null, filename: 'y.png' },
  ])
  assert.equal(state.size, 0)
})

test('junk input never throws — the grid must render', () => {
  assert.equal(improvementStateByParent(undefined).size, 0)
  assert.equal(improvementStateByParent(null).size, 0)
  assert.equal(improvementStateByParent([null, undefined]).size, 0)
})

test('the badge explains what NOT reviewing costs, and never lies', () => {
  const ready = improvementBadge('ready')
  assert.match(ready.text, /review/i)
  // The reason the badge exists: people re-ran the pass on images that already
  // had a result waiting, paying GPU time for a duplicate.
  assert.match(ready.title, /again would just make a second copy/i)
  assert.match(ready.title, /untouched/i)
  const generating = improvementBadge('generating')
  assert.match(generating.text, /upscaling/i)
  assert.doesNotMatch(generating.text, /review/i)
  assert.equal(improvementBadge(undefined), null)
  assert.equal(improvementBadge('keep'), null)
})

test('the stored derivation kind stays the legacy one, for BOTH engines', () => {
  // It is written in user databases and predates the second engine. A SeedVR2
  // result carries it too; the engine is told by the candidate's own label.
  assert.equal(IMPROVE_DERIVATION, 'klein_image_improve')
})
