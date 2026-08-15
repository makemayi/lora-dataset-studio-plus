/* Which running batch is allowed to lock the grid.
   One claim: a pass that walks the WHOLE dataset locks it; a pass that owns
   specific rows does not, because those rows already show their own state. */
import assert from 'node:assert/strict'
import test from 'node:test'

const {
  DATASET_WIDE_KINDS, PER_IMAGE_KINDS, locksWholeDataset, blockingActivity,
} = await import('../src/components/dataset/activityScope.js')

test('a dataset-wide pass locks the grid — it writes to every row', () => {
  for (const kind of DATASET_WIDE_KINDS) {
    assert.equal(locksWholeDataset({ kind }), true, kind)
  }
})

test('regenerating one tile does not lock the other tiles', () => {
  // The reported bug, stated as a test: 点击了一个图片重新生成后，其他的都不能点了
  assert.equal(locksWholeDataset({ kind: 'generate', done: 1, total: 1 }), false)
  assert.equal(locksWholeDataset({ kind: 'improve' }), false)
  for (const kind of PER_IMAGE_KINDS) {
    assert.equal(locksWholeDataset({ kind }), false, kind)
  }
})

test('nothing running locks nothing', () => {
  for (const activity of [null, undefined, {}, { kind: '' }]) {
    assert.equal(locksWholeDataset(activity), false)
  }
})

test('an unknown kind locks the grid, deliberately', () => {
  // Guessing "per-image" for a pass we do not know risks a corrupted row;
  // guessing "dataset-wide" risks a wait. Take the wait.
  assert.equal(locksWholeDataset({ kind: 'some_future_pass' }), true)
})

test('the blamed activity is the dataset-wide one, not whatever is newest', () => {
  const activities = [
    { kind: 'generate', engine: 'klein', done: 2, total: 5 },   // newest
    { kind: 'caption', done: 10, total: 200 },
  ]
  assert.equal(blockingActivity(activities).kind, 'caption')
})

test('a grid with only per-image work running has nothing to blame', () => {
  assert.equal(blockingActivity([{ kind: 'generate' }, { kind: 'improve' }]), null)
  for (const activities of [null, undefined, []]) {
    assert.equal(blockingActivity(activities), null)
  }
})
