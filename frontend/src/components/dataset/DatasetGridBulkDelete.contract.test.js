import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const grid = readFileSync(new URL('./DatasetGrid.jsx', import.meta.url), 'utf8')
const hook = readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8')

test('bulk delete has an immediate ref guard in addition to rendered busy state', () => {
  assert.match(grid, /bulkActionGateRef\.current\.active/)
  assert.match(grid, /const token = bulkActionGateRef\.current\.begin\(action, actionIds\.length\)/)
  assert.match(grid, /bulkActionGateRef\.current\.finish\(token\)/)
  assert.match(grid, /finally \{/)
})

test('selection remains until the request returns and failures stay retryable', () => {
  const action = grid.slice(grid.indexOf('const act = async'), grid.indexOf('// Hand the whole selection'))
  const awaited = action.indexOf('await onBatch')
  const cleared = action.indexOf('setSelected(new Set())')
  assert.ok(awaited >= 0, 'the action actually awaits the batch request')
  assert.ok(cleared >= 0, 'the success path clears the selection')
  assert.ok(awaited < cleared, 'selection clears only after the awaited request returns')
  assert.match(action, /typeof affected === 'number'/)
})

test('the in-flight delete is announced and every bulk action is disabled', () => {
  assert.match(grid, /role="status" aria-live="polite" aria-atomic="true"/)
  assert.match(grid, /bulkActionMessage\(bulkAction\)/)
  assert.match(grid, /const bulkBusy = busy \|\| launchingImprove \|\| !!bulkAction/)
  assert.match(grid, /disabled=\{bulkBusy\}/)
  // The tick boxes are withheld while an action is SPENDING the selection —
  // each of those snapshots the ids at click time and clears the selection on
  // return, so anything ticked meanwhile would be silently thrown away.
  // Deliberately NOT `bulkBusy`: a dataset pass (`busy` — generation,
  // captioning, a watermark scan) was handed its own list of images when it
  // started and cannot be shifted by a later tick, so it no longer freezes the
  // selection. See tests/dataset-tile-reads-stay-live.test.mjs.
  assert.match(grid, /const selectionLocked = launchingImprove \|\| !!bulkAction \|\| autoTriageApplying/)
  assert.match(grid, /onBatch && !selectionLocked && !isSmallImageRescueRow/)
  assert.match(grid, /const toggle = \(id\) => \{\s*\n\s*if \(selectionLocked\) return;/)
})

test('delete completion gets a delete-specific toast', () => {
  assert.match(hook, /action === 'delete'/)
  assert.match(hook, /d\.affected === 1 \? 'image' : 'images'/)
  assert.match(hook, /deleted`/)
})
