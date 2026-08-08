import test from 'node:test'
import assert from 'node:assert/strict'

import {
  FLAG_LABELS, flagCounts, filterByFlag, payloadFromDraft, thresholdFields,
} from '../src/components/videobank/videoMetricsFilter.js'

// 🎚 The duration cut is the only one in the panel that reads a number the
// DETECTOR wrote rather than one the metrics pass measured. Everything below is
// about the consequence: it has to survive a clip that was never measured, which
// is precisely the clip the flash-cut clutter is made of.

function field() {
  return thresholdFields().find((f) => f.key === 'min_duration_s')
}

test('the panel offers the duration cut, and it flags the short side', () => {
  const f = field()
  assert.ok(f, 'min_duration_s has no row in the panel table')
  assert.equal(f.flag, 'brief')
  assert.equal(f.direction, 'below')
})

test('the hint says the short clips are refused at promotion anyway', () => {
  // Without this the field reads as a filter that costs the user clips. It does
  // not: a clip under the target profile's frame count never lands, cut or no
  // cut. The field buys them the chance to see it first.
  assert.match(field().hint, /promot/i)
})

test('brief and too_short are two different words for two different things', () => {
  // `too_short` is the promotion's refusal — an impossibility of the target
  // profile. Reusing it here would tell a user that lowering this number buys
  // them a dataset, which it does not.
  assert.equal(FLAG_LABELS.brief !== undefined, true)
  assert.ok(!thresholdFields().some((f) => f.flag === 'too_short'))
})

test('an unmeasured clip still counts the flags it does carry', () => {
  // The old count skipped every flag on a clip with no metrics, which was right
  // while every flag needed the metrics pass. A duration flag on a never-measured
  // clip would have been visible in the grid and absent from the chip counts —
  // the user would filter on a chip that says 0.
  const counts = flagCounts([
    { metrics: null, flags: ['brief'] },
    { metrics: null, flags: [] },
    { metrics: { metrics_state: 'ok' }, flags: ['still'] },
  ])

  assert.equal(counts.brief, 1)
  assert.equal(counts.still, 1)
  assert.equal(counts.flagged, 2)
  assert.equal(counts.unmeasured, 2, 'unmeasured is still a state of its own')
})

test('the brief chip selects the unmeasured clips carrying it', () => {
  const clips = [{ id: 1, metrics: null, flags: ['brief'] },
                 { id: 2, metrics: null, flags: [] }]

  assert.deepEqual(filterByFlag(clips, 'brief').map((c) => c.id), [1])
})

test('the duration cut rides in the dry-run payload', () => {
  const payload = payloadFromDraft({ min_duration_s: 1.5, motion_floor: null })

  assert.deepEqual(payload, { min_duration_s: 1.5 })
})
