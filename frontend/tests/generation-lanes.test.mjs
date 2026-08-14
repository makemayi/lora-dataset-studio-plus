/* The lane rule that lets a ComfyUI batch and a ChatGPT batch run at once.
   Everything here is about ONE claim: a running batch may block its own lane
   and nothing else. */
import assert from 'node:assert/strict'
import test from 'node:test'

const {
  LANE_LOCAL, LANE_API, laneOf, lanesFor, busyLanes,
  launchBlockedReason, engineCardLocked,
} = await import('../src/components/dataset/generationLanes.js')

test('every shipped engine lands in exactly one lane', () => {
  assert.equal(laneOf('klein'), LANE_LOCAL)
  assert.equal(laneOf('krea'), LANE_LOCAL)
  assert.equal(laneOf('minimax_h3'), LANE_LOCAL)
  assert.equal(laneOf('chatgpt'), LANE_API)
  assert.equal(laneOf('nanobanana'), LANE_API)
  assert.equal(laneOf('openrouter'), LANE_API)
  assert.equal(laneOf('qwen'), LANE_API)
})

test('an engine this build does not know inherits no permissions', () => {
  assert.equal(laneOf('some_future_engine'), null)
  assert.equal(lanesFor(['some_future_engine']).size, 0)
  // ...and therefore is never blocked BY a lane, nor blocks one.
  assert.equal(launchBlockedReason(['some_future_engine'],
                                   [{ kind: 'generate', engine: 'klein' }]), null)
})

test('a local batch blocks local engines and NOT the API lane', () => {
  const busy = [{ kind: 'generate', engine: 'klein', done: 3, total: 40 }]
  assert.match(launchBlockedReason(['krea'], busy), /already running on your GPU/)
  assert.equal(launchBlockedReason(['chatgpt'], busy), null)
  assert.equal(engineCardLocked('minimax_h3', busy), true)
  assert.equal(engineCardLocked('chatgpt', busy), false)
})

test('an API batch blocks the API lane and NOT the GPU', () => {
  const busy = [{ kind: 'generate', engine: 'chatgpt', done: 1, total: 4 }]
  assert.match(launchBlockedReason(['nanobanana'], busy), /on the API lane/)
  assert.equal(launchBlockedReason(['klein'], busy), null)
  assert.equal(engineCardLocked('chatgpt', busy), true)
  assert.equal(engineCardLocked('klein', busy), false)
})

test('a mixed selection is blocked when EITHER of its lanes is busy', () => {
  const busy = [{ kind: 'generate', engine: 'chatgpt' }]
  assert.ok(launchBlockedReason(['klein', 'chatgpt'], busy))
})

test('a batch with no engine counts as local', () => {
  // Caption passes, watermark sweeps and the improve batch all drive ComfyUI or
  // the local ML environments. Guessing the other way would let an API launch
  // collide with the one thing on this machine that IS serialized.
  const busy = [{ kind: 'caption', done: 10, total: 200 }]
  assert.deepEqual([...busyLanes(busy)], [LANE_LOCAL])
  assert.ok(launchBlockedReason(['klein'], busy))
  assert.equal(launchBlockedReason(['chatgpt'], busy), null)
})

test('nothing running blocks nothing', () => {
  for (const activities of [null, undefined, []]) {
    assert.equal(launchBlockedReason(['klein', 'chatgpt'], activities), null)
    assert.equal(engineCardLocked('klein', activities), false)
  }
})

test('the reason names the batch in the way', () => {
  const reason = launchBlockedReason(['krea'],
    [{ kind: 'generate', engine: 'minimax_h3' }])
  assert.match(reason, /minimax_h3/)
})
