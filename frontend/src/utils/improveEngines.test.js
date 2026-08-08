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
  lightboxImproveButtons,
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

// --- The lightbox's per-image buttons ---------------------------------------
// A user's case, from a screenshot: a DRAWN dataset where the amber note already
// warns that Klein's instruction pulls anime skin towards realism — and SeedVR2,
// the pass that does not do that, was offered in the selection toolbar but not
// in the lightbox, which is where you are when you are looking at that one image.

const READY = { comfyui: { seedvr2_ready: true } }

test('the lightbox offers BOTH engines once SeedVR2 is installed', () => {
  const ids = lightboxImproveButtons({ caps: READY, engines: { klein: true } })
    .map((b) => b.id)
  assert.deepEqual(ids, ['klein', 'seedvr2'])
})

test('SeedVR2 absent from the lightbox until it is installed', () => {
  const ids = lightboxImproveButtons({ caps: { comfyui: {} }, engines: { klein: true } })
    .map((b) => b.id)
  assert.deepEqual(ids, ['klein'])
})

test('the amber anime warning belongs to Klein ALONE', () => {
  // It is about Klein's instruction ("detailed texture, sharp details"). SeedVR2
  // sends no instruction, so repeating it there would be false — and would warn
  // people off the exact pass that solves their problem.
  const byId = Object.fromEntries(
    lightboxImproveButtons({ caps: READY, engines: { klein: true } })
      .map((b) => [b.id, b]))
  assert.equal(byId.klein.showKleinNote, true)
  assert.equal(byId.seedvr2.showKleinNote, false)
})

test('each button carries its own trade-off sentence, never the other one', () => {
  const byId = Object.fromEntries(
    lightboxImproveButtons({ caps: READY, engines: { klein: true } })
      .map((b) => [b.id, b]))
  assert.match(byId.klein.title, /shift/i)
  assert.match(byId.seedvr2.title, /keeps the original/i)
  assert.doesNotMatch(byId.seedvr2.title, /shift/i)
})

test('an engine that cannot run is disabled and SAYS why, per engine', () => {
  const byId = Object.fromEntries(
    lightboxImproveButtons({ caps: { comfyui: { seedvr2_ready: true } },
      engines: { klein: false } }).map((b) => [b.id, b]))
  assert.equal(byId.klein.disabled, true)
  assert.match(byId.klein.title, /not available/i)
  // ...while the OTHER engine stays clickable. A shared disabled flag would have
  // greyed out the working pass because the broken one is broken.
  assert.equal(byId.seedvr2.disabled, false)
})

test('image-level state blocks every engine, and says so before the engine name', () => {
  for (const state of [{ improvePending: true }, { improving: true },
    { improveReady: true }, { busy: true }]) {
    const buttons = lightboxImproveButtons({
      caps: READY, engines: { klein: true }, ...state })
    assert.ok(buttons.every((b) => b.disabled),
      `${JSON.stringify(state)} must block both engines`)
  }
  const [klein] = lightboxImproveButtons({
    caps: READY, engines: { klein: true }, improveReady: true })
  assert.equal(klein.label, '✓ Review improvement first')
  const [running] = lightboxImproveButtons({
    caps: READY, engines: { klein: true }, improving: true })
  assert.match(running.label, /Improving…/)
})

test('the idle labels name the engine, matching the selection toolbar', () => {
  const labels = lightboxImproveButtons({ caps: READY, engines: { klein: true } })
    .map((b) => b.label)
  assert.deepEqual(labels, ['✨ Improve via Klein', '🔍 Upscale via SeedVR2'])
})
