import test from 'node:test'
import assert from 'node:assert/strict'
import { reasonLabel, summarizeRejections } from './videoFrameReasons.js'

test('every reason the selector can emit has a label', () => {
  for (const r of ['too_blurry', 'face_blurry', 'no_face', 'low_det',
                   'face_too_small', 'extreme_profile', 'face_px_unknown',
                   'face_too_few_pixels', 'wrong_person', 'exposure',
                   'unmeasured', 'too_close', 'duplicate', 'over_limit',
                   'limit_is_zero']) {
    assert.ok(reasonLabel(r), `missing label for ${r}`)
  }
})

test('unknown reasons fall back to their own name instead of crashing', () => {
  assert.equal(reasonLabel('brand_new_reason'), 'brand_new_reason')
})

test('summarizeRejections drops zero counts and orders by count desc', () => {
  const lines = summarizeRejections({ too_blurry: 68, face_blurry: 25, exposure: 0 })
  assert.equal(lines.length, 2)
  assert.equal(lines[0].label, 'Blurry')
  assert.equal(lines[0].count, 68)
  assert.equal(lines[1].label, 'Blurry face')
})

test('summarizeRejections tolerates junk', () => {
  assert.deepEqual(summarizeRejections(null), [])
  assert.deepEqual(summarizeRejections({}), [])
})
