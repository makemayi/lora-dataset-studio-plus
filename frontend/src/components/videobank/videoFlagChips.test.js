import test from 'node:test'
import assert from 'node:assert/strict'

import {
  FLAG_LABELS, filterByFlag, flagChips, flagFilterNote,
} from './videoMetricsFilter.js'

/** ⚑ The flag chips — the half of the verdict story that was missing.
 *
 * `flagCounts` and `filterByFlag` shipped with the quality wave, were tested,
 * and were wired to NOTHING: every verdict in the video lane was a badge you
 * could read one shot at a time and could not act on in bulk. That is tolerable
 * for "too much motion" and useless for "you already have this shot" — the whole
 * value of a duplicate verdict is rejecting the pile in one gesture.
 *
 * What these tests hold is the part that can quietly become a lie: the chips
 * count the LOADED page, and they have to say so.
 */

const clip = (id, flags, metrics = { metrics_state: 'ok' }) => ({ id, flags, metrics })

test('a chip appears for each flag present, biggest first', () => {
  const chips = flagChips([
    clip(1, ['duplicate']), clip(2, ['duplicate']), clip(3, ['watermark']),
  ])
  assert.deepEqual(chips.map((c) => [c.flag, c.count]),
    [['duplicate', 2], ['watermark', 1]])
})

test('a flag nothing carries gets no chip', () => {
  const chips = flagChips([clip(1, ['still'])])
  assert.deepEqual(chips.map((c) => c.flag), ['still'])
})

test('every chip carries a human label, never a raw key', () => {
  for (const chip_ of flagChips([clip(1, ['duplicate', 'watermark', 'soft_start'])])) {
    assert.equal(chip_.label, FLAG_LABELS[chip_.flag])
    assert.ok(!chip_.label.includes('_'), `${chip_.flag} reaches the UI raw`)
  }
})

test('"not measured yet" is offered as a chip of its own', () => {
  // The state no verdict can express, and the one people most need to select.
  const chips = flagChips([{ id: 1, flags: [], metrics: null }])
  assert.deepEqual(chips.map((c) => c.flag), ['unmeasured'])
})

test('a chip selects exactly the shots carrying its flag', () => {
  const clips = [clip(1, ['duplicate']), clip(2, []), clip(3, ['duplicate', 'soft'])]
  assert.deepEqual(filterByFlag(clips, 'duplicate').map((c) => c.id), [1, 3])
})

test('the counts say when they cover only the loaded page', () => {
  assert.match(flagFilterNote(120, 900), /120 shots loaded, not all 900/)
})

test('…and stay silent once everything is loaded', () => {
  assert.equal(flagFilterNote(90, 90), '')
  assert.equal(flagFilterNote(90, 0), '')
})
