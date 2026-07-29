import test from 'node:test'
import assert from 'node:assert/strict'

import {
  canEditMask, initialMask, maskStatus, maskPayload, applyMaskResponse,
} from './bankWatermarkMask.js'

const BBOX = [0.1, 0.02, 0.4, 0.06]
const DRAWN = [[0.05, 0.02, 0.2, 0.08], [0.44, 0.46, 0.52, 0.53]]

test('only a still-flagged image can be masked', () => {
  assert.equal(canEditMask({ watermark_state: 'detected' }), true)
  for (const state of ['cleaned', 'dismissed', 'none', 'error', null, undefined]) {
    assert.equal(canEditMask({ watermark_state: state }), false, `state ${state}`)
  }
  assert.equal(canEditMask(null), false)
})

test('an untouched image opens on the detected box, marked as NOT hand-drawn', () => {
  const m = initialMask({
    watermark_state: 'detected', watermark_bbox: BBOX,
    watermark_regions: null, effective_watermark_regions: [BBOX],
  })
  assert.deepEqual(m.regions, [BBOX])
  assert.equal(m.manual, false)
  // A defensive copy: dragging a zone must not mutate the payload we were given.
  m.regions[0][0] = 0.9
  assert.equal(BBOX[0], 0.1)
})

test('a hand-edited image opens on what the user drew', () => {
  const m = initialMask({
    watermark_state: 'detected', watermark_bbox: BBOX,
    watermark_regions: DRAWN, effective_watermark_regions: DRAWN,
  })
  assert.deepEqual(m.regions, DRAWN)
  assert.equal(m.manual, true)
})

test('an emptied mask stays empty — it never falls back to the detected box', () => {
  const m = initialMask({
    watermark_state: 'detected', watermark_bbox: BBOX,
    watermark_regions: [], effective_watermark_regions: [],
  })
  assert.deepEqual(m.regions, [])
  assert.equal(m.manual, true)
})

test('a legacy flag with no box at all opens empty rather than crashing', () => {
  const m = initialMask({ watermark_state: 'detected' })
  assert.deepEqual(m.regions, [])
  assert.equal(m.manual, false)
})

test('the status line says what each level will DO with this mask', () => {
  const drawn = maskStatus({ regions: DRAWN, manual: true })
  assert.equal(drawn.tone, 'ok')
  assert.match(drawn.text, /2 hand-drawn zones/)
  assert.match(drawn.text, /Inpaint/)
  assert.match(drawn.text, /Auto-crop/)      // it skips hand-masked images: say so

  const emptied = maskStatus({ regions: [], manual: true })
  assert.equal(emptied.tone, 'warn')
  assert.match(emptied.text, /nothing/i)     // an empty mask cleans NOTHING

  const auto = maskStatus({ regions: [BBOX], manual: false })
  assert.equal(auto.tone, 'info')
  assert.match(auto.text, /detected/i)

  const nothing = maskStatus({ regions: [], manual: false })
  assert.equal(nothing.tone, 'warn')
  assert.match(nothing.text, /draw/i)
})

test('the status text is English and free of leftover placeholders', () => {
  for (const state of [{ regions: DRAWN, manual: true }, { regions: [], manual: true },
    { regions: [BBOX], manual: false }, { regions: [], manual: false }]) {
    const { text } = maskStatus(state)
    assert.ok(text.length > 10)
    assert.doesNotMatch(text, /undefined|NaN|\bTODO\b/)
  }
})

test('saving sends rounded boxes; resetting sends an explicit null', () => {
  assert.deepEqual(maskPayload([[0.10004999, 0.2, 0.30001, 0.4]]).regions,
    [[0.1, 0.2, 0.3, 0.4]])
  assert.deepEqual(maskPayload([]).regions, [])   // empty is a VALUE, not a reset
  assert.equal(maskPayload(null).regions, null)
})

test('the server answer is what the editor re-renders from', () => {
  const img = { id: 7, watermark_state: 'detected', watermark_bbox: BBOX,
    watermark_regions: null, effective_watermark_regions: [BBOX] }
  const next = applyMaskResponse(img, {
    ok: true, watermark_regions: DRAWN, effective_watermark_regions: DRAWN,
  })
  assert.deepEqual(next.watermark_regions, DRAWN)
  assert.deepEqual(initialMask(next).regions, DRAWN)
  assert.equal(initialMask(next).manual, true)
  assert.equal(next.id, 7)                     // the rest of the row is preserved
  assert.equal(img.watermark_regions, null)    // and the input is not mutated

  // A reset answer (null override) puts the detected box back.
  const back = applyMaskResponse(next, {
    ok: true, watermark_regions: null, effective_watermark_regions: [BBOX],
  })
  assert.equal(back.watermark_regions, null)
  assert.deepEqual(initialMask(back).regions, [BBOX])
  assert.equal(initialMask(back).manual, false)

  // A malformed/failed answer changes nothing rather than wiping the mask.
  assert.deepEqual(applyMaskResponse(next, null), next)
  assert.deepEqual(applyMaskResponse(next, { ok: false, error: 'nope' }), next)
})
