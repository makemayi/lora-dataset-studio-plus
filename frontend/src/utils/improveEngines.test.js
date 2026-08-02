import assert from 'node:assert/strict'
import test from 'node:test'

import {
  availableImproveEngines,
  describeImproveLaunch,
  improveBatchLabel,
  improveConfirmMessage,
  improveEngine,
  improveEngineBlockedReason,
  IMPROVE_ENGINES,
} from './improveEngines.js'

test('every engine states what it does to the ORIGINAL, not just that it improves', () => {
  // Issue #32 is exactly this distinction. A summary that only promised
  // "sharper" on both would leave the user picking blind, which is the bug.
  const klein = improveEngine('klein')
  const seedvr2 = improveEngine('seedvr2')
  assert.match(klein.summary, /shift|change/i)
  assert.match(seedvr2.summary, /keeps the original/i)
  assert.notEqual(klein.summary, seedvr2.summary)
  for (const engine of IMPROVE_ENGINES) {
    assert.ok(engine.id && engine.label && engine.action && engine.summary)
  }
})

test('an unknown engine id falls back to Klein rather than blowing up a label', () => {
  assert.equal(improveEngine('nonsense').id, 'klein')
  assert.equal(improveEngine(undefined).id, 'klein')
})

test('SeedVR2 only appears once it is ready; Klein always does', () => {
  const ids = (caps) => availableImproveEngines(caps).map((e) => e.id)
  assert.deepEqual(ids(undefined), ['klein'])
  assert.deepEqual(ids({ comfyui: {} }), ['klein'])
  assert.deepEqual(ids({ comfyui: { seedvr2_ready: false } }), ['klein'])
  assert.deepEqual(ids({ comfyui: { seedvr2_ready: true } }), ['klein', 'seedvr2'])
})

test('a blocked engine says WHY, and points at where the fix lives', () => {
  assert.equal(improveEngineBlockedReason('klein', {
    engines: { klein: true }, eligibleCount: 3,
  }), null)
  assert.match(improveEngineBlockedReason('klein', {
    engines: { klein: false }, eligibleCount: 3,
  }), /not available/i)
  assert.match(improveEngineBlockedReason('seedvr2', {
    caps: { comfyui: { seedvr2_ready: false } }, eligibleCount: 3,
  }), /Setup/)
  assert.match(improveEngineBlockedReason('seedvr2', {
    caps: { comfyui: { seedvr2_ready: true } }, eligibleCount: 0,
  }), /eligible/i)
})

test('the confirm carries the engine trade-off and the skip count', () => {
  const msg = improveConfirmMessage('seedvr2', {
    eligibleCount: 12, excludedCount: 3, exclusionSummary: 'already improved',
  })
  assert.match(msg, /12 image\(s\)/)
  assert.match(msg, /keeps the original/i)
  assert.match(msg, /3 selected image\(s\) will be skipped: already improved/)
  assert.match(msg, /Original images stay unchanged/)
  const klein = improveConfirmMessage('klein', { eligibleCount: 1 })
  assert.match(klein, /Klein/)
  assert.doesNotMatch(klein, /will be skipped/)
})

test('the launch toast names the engine the SERVER ran, not the button pressed', () => {
  assert.match(describeImproveLaunch({ queued: 4, engine: 'seedvr2' }), /^SeedVR2: processing 4/)
  assert.match(describeImproveLaunch({ queued: 4, skipped: 2, engine: 'klein' }),
    /^Klein: processing 4 image\(s\) in the background · 2 not eligible/)
  // A server that echoes nothing still produces a sentence, not "undefined".
  assert.match(describeImproveLaunch({ queued: 1 }), /^Klein: processing 1/)
})

test('the progress label reads the running batch engine', () => {
  assert.equal(improveBatchLabel(null), null)
  assert.equal(improveBatchLabel({ kind: 'caption', total: 4, done: 1 }), null)
  assert.equal(improveBatchLabel({ kind: 'improve', engine: 'seedvr2', total: 9, done: 2 }),
    '🔍 SeedVR2 2/9')
  assert.equal(improveBatchLabel({ kind: 'improve', engine: 'klein', total: 0, done: 0 }),
    '✨ Klein…')
  assert.equal(improveBatchLabel({ kind: 'improve', engine: 'seedvr2', cancelling: true }),
    '🔍 Stopping…')
})
