import assert from 'node:assert/strict'
import test from 'node:test'
import { bulkActionMessage, createBulkActionGate } from './bulkActionGate.js'

test('bulk action gate refuses a double click until the first request finishes', () => {
  const gate = createBulkActionGate()
  const first = gate.begin('delete', 12)
  assert.deepEqual(first, { action: 'delete', count: 12 })
  assert.equal(gate.begin('delete', 12), null)
  assert.equal(bulkActionMessage(gate.active), 'Deleting 12 images…')

  gate.finish(first)
  assert.equal(gate.active, null)
  assert.equal(bulkActionMessage(gate.active), '')
  assert.ok(gate.begin('delete', 3), 'actions are available again after finally')
})

test('delete progress is singular for one image', () => {
  assert.equal(bulkActionMessage({ action: 'delete', count: 1 }), 'Deleting 1 image…')
})

test('only the matching request token may release the guard', () => {
  const gate = createBulkActionGate()
  const first = gate.begin('delete', 2)
  gate.finish({ action: 'delete', count: 2 })
  assert.equal(gate.active, first)
  gate.finish(first)
  assert.equal(gate.active, null)
})
