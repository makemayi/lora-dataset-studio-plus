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
  // The 'klein' lane used to be pinned on the word "shift", because it WAS the
  // engine that shifted skin and colour. It is Krea 2 + SeedVR2 now, and its
  // ColorTransfer stage exists to put the original's tone back — so the
  // invariant is not that word, it is that each summary says what happens to
  // the ORIGINAL rather than only promising "sharper".
  assert.match(klein.summary, /colour|recolour|original/i)
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

test('SeedVR2 only appears once it is ready; both rewrites always do', () => {
  const ids = (caps) => availableImproveEngines(caps).map((e) => e.id)
  // Both REWRITE engines are always listed: when one cannot run its button
  // carries the reason, and an engine that vanishes teaches nobody why. Only
  // SeedVR2 is gated, because until it is installed it is a setup task.
  assert.deepEqual(ids(undefined), ['klein', 'klein_hq'])
  assert.deepEqual(ids({ comfyui: {} }), ['klein', 'klein_hq'])
  assert.deepEqual(ids({ comfyui: { seedvr2_ready: false } }), ['klein', 'klein_hq'])
  assert.deepEqual(ids({ comfyui: { seedvr2_ready: true } }),
    ['klein', 'klein_hq', 'seedvr2'])
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
  // Named by what it RUNS. The id behind it is still 'klein' (stored in rows),
  // but a confirm dialog naming an engine this lane stopped using is how the
  // user who wrote the pipeline failed to find it in their own app.
  assert.match(klein, /Krea 2 \+ SeedVR2/)
  assert.doesNotMatch(klein, /will be skipped/)
})

test('the launch toast names the engine the SERVER ran, not the button pressed', () => {
  assert.match(describeImproveLaunch({ queued: 4, engine: 'seedvr2' }), /^SeedVR2: processing 4/)
  assert.match(describeImproveLaunch({ queued: 4, skipped: 2, engine: 'klein' }),
    /^Krea 2 \+ SeedVR2: processing 4 image\(s\) in the background · 2 not eligible/)
  // A server that echoes nothing still produces a sentence, not "undefined".
  assert.match(describeImproveLaunch({ queued: 1 }), /^Krea 2 \+ SeedVR2: processing 1/)
})

test('the progress label reads the running batch engine', () => {
  assert.equal(improveBatchLabel(null), null)
  assert.equal(improveBatchLabel({ kind: 'caption', total: 4, done: 1 }), null)
  assert.equal(improveBatchLabel({ kind: 'improve', engine: 'seedvr2', total: 9, done: 2 }),
    '🔍 SeedVR2 2/9')
  // The stored engine id is still 'klein'; what it RENDERS is the pipeline's
  // real name, which is the whole point of the relabel.
  assert.equal(improveBatchLabel({ kind: 'improve', engine: 'klein', total: 0, done: 0 }),
    '✨ Krea 2 + SeedVR2…')
  assert.equal(improveBatchLabel({ kind: 'improve', engine: 'seedvr2', cancelling: true }),
    '🔍 Stopping…')
})

// --- The lightbox's per-image buttons ---------------------------------------
// A user's case, from a screenshot: a DRAWN dataset where the amber note already
// warns that Klein's instruction pulls anime skin towards realism — and SeedVR2,
// the pass that does not do that, was offered in the selection toolbar but not
// in the lightbox, which is where you are when you are looking at that one image.

const READY = { comfyui: { seedvr2_ready: true } }

test('the lightbox offers EVERY engine once SeedVR2 is installed', () => {
  const ids = lightboxImproveButtons({ caps: READY, engines: { klein: true } })
    .map((b) => b.id)
  assert.deepEqual(ids, ['klein', 'klein_hq', 'seedvr2'])
})

test('SeedVR2 absent from the lightbox until it is installed', () => {
  const ids = lightboxImproveButtons({ caps: { comfyui: {} }, engines: { klein: true } })
    .map((b) => b.id)
  assert.deepEqual(ids, ['klein', 'klein_hq'])
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
  // Each button carries ITS OWN sentence and never the other engine's — the
  // point of the test, unchanged by the pipeline swap behind the 'klein' id.
  assert.match(byId.klein.title, /Krea 2/i)
  assert.match(byId.seedvr2.title, /keeps the original/i)
  assert.doesNotMatch(byId.seedvr2.title, /Krea 2/i)
  assert.notEqual(byId.klein.title, byId.seedvr2.title)
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
  assert.deepEqual(labels,
    ['✨ Improve via Krea 2 + SeedVR2', '🩹 Improve via Flux.2 Klein 9B',
     '🔍 Upscale via SeedVR2'])
})
