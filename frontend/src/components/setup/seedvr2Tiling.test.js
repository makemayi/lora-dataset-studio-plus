import assert from 'node:assert/strict'
import test from 'node:test'

import { ceilingLine, laneForTarget, tilingStatus, TTP_PACK, TTP_URL } from './seedvr2Tiling.js'

const caps = (comfyui) => ({ comfyui: { reachable: true, ...comfyui } })

test('an absent pack reads as OPTIONAL, never as something broken', () => {
  const s = tilingStatus(caps({ seedvr2_tiling_ready: false,
    seedvr2_tiling_nodes_missing: ['TTP_Image_Tile_Batch', 'TTP_Image_Assy'] }))
  assert.equal(s.state, 'absent')
  assert.match(s.text, /^Optional\./)
  // ...and it says what still works, because it does.
  assert.match(s.text, /upscales still run/)
  assert.match(s.text, new RegExp(TTP_PACK))
})

test('half the classes present means UPDATE and restart, not install', () => {
  // An old build of the pack is a different problem from a missing one, and
  // telling someone to install what they already have is the failure mode this
  // whole detection style exists to avoid.
  const s = tilingStatus(caps({ seedvr2_tiling_ready: false,
    seedvr2_tiling_nodes_missing: ['TTP_Image_Assy'] }))
  assert.equal(s.state, 'restart')
  assert.match(s.text, /update it/i)
  assert.match(s.text, /TTP_Image_Assy/)
})

test('a ready pack says what it changes, in the user’s terms', () => {
  const s = tilingStatus(caps({ seedvr2_tiling_ready: true }))
  assert.equal(s.state, 'ready')
  assert.match(s.text, /tiles/)
})

test('an unreachable ComfyUI is "cannot tell", not "missing"', () => {
  // This lane's probe fails CLOSED, so silence here is honest ignorance.
  const s = tilingStatus({ comfyui: { reachable: false } })
  assert.equal(s.state, 'unknown')
  assert.doesNotMatch(s.text, /Optional/)
  assert.match(s.text, /Start ComfyUI/)
})

test('junk capabilities never throw — the Setup page must render', () => {
  for (const c of [undefined, null, {}, { comfyui: null }]) {
    assert.ok(tilingStatus(c).state)
    assert.doesNotThrow(() => ceilingLine(c))
  }
})

test('an unknown card gets SILENCE, never an invented ceiling', () => {
  for (const mp of [undefined, null, 0, -3, 'lots']) {
    assert.equal(ceilingLine(caps({ seedvr2_ceiling_mp: mp })), null)
  }
})

test('the ceiling sentence changes with the lane that is actually available', () => {
  const without = ceilingLine(caps({ seedvr2_ceiling_mp: 6.4, seedvr2_tiling_ready: false }))
  assert.match(without, /6\.4 MP/)
  assert.match(without, /run out of memory/)
  const with_ = ceilingLine(caps({ seedvr2_ceiling_mp: 6.4, seedvr2_tiling_ready: true }))
  assert.match(with_, /anything larger is tiled/)
  // With tiling available the warning is gone: keeping it would scare people
  // away from a limit that no longer applies to them.
  assert.doesNotMatch(with_, /run out of memory/)
})

// --- what today's settings will actually do ---------------------------------
// The hole these close: the crossover is STRICT and derived as 1.5x the tile,
// so it lands exactly on numbers people type (1536 at the 1024 default, 768 at
// a 512 tile). A target sitting ON it ran full-frame in silence — no tiling and
// no explanation — which reads as "my setting did nothing".

test('a target sitting EXACTLY on the crossover says so, and says it runs whole', () => {
  const line = laneForTarget('auto', 1536, 1536)
  assert.match(line, /exactly at the 1536 px crossover/)
  assert.match(line, /runs full-frame/)
  // The silence is the defect, so the sentence must also carry a way out.
  assert.match(line, /raise the target above 1536 px/)
  assert.match(line, /Start tiling above/)
  assert.match(line, /Always tile large frames/)
})

test('one pixel over the crossover is tiled, and the sentence flips with it', () => {
  assert.match(laneForTarget('auto', 1538, 1536), /is tiled/)
  assert.doesNotMatch(laneForTarget('auto', 1538, 1536), /full-frame/)
  // ...and below it reads as "below", not "exactly at".
  assert.match(laneForTarget('auto', 1080, 1536), /below the 1536 px crossover/)
})

test('the boundary moves with the tile size, not with a hardcoded 1536', () => {
  // Someone who drops the tile to 512 to survive on 8 GB gets a 768 crossover —
  // and 768 is itself a round number people type.
  assert.match(laneForTarget('auto', 768, 768), /exactly at the 768 px crossover/)
})

test('the two explicit modes describe themselves and ignore the crossover', () => {
  assert.match(laneForTarget('never', 2160, 1536), /Nothing is tiled/)
  assert.match(laneForTarget('always', 800, 1536), /bigger than one tile/)
})

test('no number to reason about means silence, never a made-up sentence', () => {
  for (const bad of [undefined, null, 0, -2, 'big']) {
    assert.equal(laneForTarget('auto', bad, 1536), null)
    assert.equal(laneForTarget('auto', 1536, bad), null)
  }
})

test('the pack is named and linkable', () => {
  assert.equal(TTP_PACK, 'Comfyui_TTP_Toolset')
  assert.match(TTP_URL, /^https:\/\/github\.com\/TTPlanetPig\/Comfyui_TTP_Toolset$/)
})
