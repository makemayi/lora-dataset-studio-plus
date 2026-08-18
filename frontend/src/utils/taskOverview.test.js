import test from 'node:test'
import assert from 'node:assert/strict'
import { normalizeOverview, badgeCounts, STATUS_LABEL } from './taskOverview.js'

test('normalizeOverview tolerates junk and derives running/paused counts', () => {
  const empty = normalizeOverview(null)
  assert.deepEqual(empty.tasks, [])
  assert.equal(empty.status, null)

  const raw = {
    status: { summary: { queued: 2, paused: 5, running: 1, today_done: 34, today_failed: 1 } },
    tasks: [
      { job_id: 'a', status: 'awaiting_comfyui' },
      { job_id: 'b', status: 'pending' },
      { job_id: 'c', status: 'failed' },
      { job_id: 'd', status: 'completed' },
    ],
  }
  const o = normalizeOverview(raw)
  assert.equal(o.status.summary.paused, 5)
  assert.equal(o.runningCount, 2)   // a + b (paused counts as in-progress)
  assert.equal(o.failedCount, 1)
  assert.deepEqual(o.failedIds, ['c'])
})

test('STATUS_LABEL covers every queue status', () => {
  for (const s of ['pending', 'awaiting_comfyui', 'processing', 'sent_to_comfy',
                   'completed', 'failed', 'cancelled', 'stalled', 'running']) {
    assert.ok(STATUS_LABEL[s], `missing label for ${s}`)
  }
})

test('badgeCounts hides failures already seen and zeroes when nothing runs', () => {
  const o = { runningCount: 3, failedIds: ['f1', 'f2'] }
  assert.deepEqual(badgeCounts(o, new Set(['f1'])), { running: 3, failed: 1 })
  assert.deepEqual(badgeCounts({ runningCount: 0, failedIds: [] }, new Set()),
                   { running: 0, failed: 0 })
})

test('topaz queued status gets a label and counts as in-progress', () => {
  assert.equal(STATUS_LABEL.queued, 'Queued')
  const o = normalizeOverview({ tasks: [{ job_id: 'topaz-1', status: 'queued' }] })
  assert.equal(o.runningCount, 1)
})
