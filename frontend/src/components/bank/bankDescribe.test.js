import test from 'node:test'
import assert from 'node:assert/strict'
import { toFilterPatch, describeSummary, headline, TOUCHED } from './bankDescribe.js'

test('the API name and the filter key are mapped, not assumed equal', () => {
  // `res_bucket` there, `resBucket` here. Assuming they match is how a field
  // silently stops arriving while the summary keeps claiming it.
  const patch = toFilterPatch({ filter: { res_bucket: 'res_2_4', medium: 'photo' } })
  assert.equal(patch.resBucket, 'res_2_4')
  assert.equal(patch.medium, 'photo')
  assert.ok(!('res_bucket' in patch), 'the API spelling must not leak into the state')
})

test('a new reading CLEARS what the previous one set', () => {
  // Two requests in a row must not compose. "amateur" then "anime portraits"
  // would otherwise keep the first medium, return nothing, and show a summary
  // describing only the second.
  const patch = toFilterPatch({ filter: { medium: 'anime' } })
  for (const key of TOUCHED) {
    if (key !== 'medium') assert.equal(patch[key], null, `${key} was left behind`)
  }
  assert.equal(patch.medium, 'anime')
})

test('a field outside the map is not applied', () => {
  // The server validates against the same vocabulary, so this should be
  // unreachable — which is exactly why it must not be silently trusted here.
  const patch = toFilterPatch({ filter: { medium: 'photo', mood: 'moody' } })
  assert.ok(!('mood' in patch))
  assert.equal(patch.medium, 'photo')
})

test('the sort rides along only when the server sent one', () => {
  assert.equal(toFilterPatch({ filter: {}, sort: 'aesthetic_asc' }).sort, 'aesthetic_asc')
  assert.ok(!('sort' in toFilterPatch({ filter: {} })), 'an absent sort must not reset the current one')
})

test('the three registers stay separate', () => {
  // "understood", "could not", and "tried but this bank has none" answer three
  // different questions. Merged into one line, a half-read request reads as a win.
  const s = describeSummary({
    understood: ['photographic'], unsupported: ['outdoors'],
    dropped: ['medium=render3d holds 0 images'], refused: false,
  })
  assert.deepEqual(s.understood, ['photographic'])
  assert.deepEqual(s.unsupported, ['outdoors'])
  assert.deepEqual(s.dropped, ['medium=render3d holds 0 images'])
})

test('the headline never states a number the model did not compute', () => {
  const line = headline({ understood: ['photographic'], refused: false })
  assert.ok(!/\d/.test(line), `a count appeared in the headline: ${line}`)
  assert.match(line, /measured, not guessed/)
})

test('a request that maps onto nothing says so instead of claiming success', () => {
  const line = headline({ understood: [], unsupported: ['outdoors'], refused: true })
  assert.match(line, /Nothing in that request/)
})
