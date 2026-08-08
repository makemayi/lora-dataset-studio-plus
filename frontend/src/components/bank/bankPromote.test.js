import test from 'node:test'
import assert from 'node:assert/strict'

import {
  PROMOTE_DESTINATIONS, canStartPromote, formatWeight, promoteButtonLabel,
  promoteCount, promoteSummary, weightNotice,
} from './bankPromote.js'

test('two destinations, dataset first', () => {
  assert.deepEqual(PROMOTE_DESTINATIONS, ['dataset', 'bank'])
})

test('formatWeight scales, and refuses to invent a number', () => {
  assert.equal(formatWeight(0), '0 B')
  assert.equal(formatWeight(900), '900 B')
  assert.equal(formatWeight(1024 * 300), '300 KB')
  assert.equal(formatWeight(1024 * 1024 * 62), '62 MB')
  // video-scale: the case the line exists for
  assert.equal(formatWeight(1024 * 1024 * 1024 * 1.44), '1.4 GB')
  assert.equal(formatWeight(null), null)
  assert.equal(formatWeight(undefined), null)
  assert.equal(formatWeight(-1), null)
})

test('the weight is announced for a new bank, and only once measured', () => {
  const notice = weightNotice({ destination: 'bank', size: { count: 200, bytes: 65e6 } })
  assert.match(notice, /62 MB/)
  assert.match(notice, /never share a file/)
  // not measured yet -> say so rather than show a confident 0
  assert.match(weightNotice({ destination: 'bank', size: null }), /Measuring/)
  assert.match(weightNotice({ destination: 'bank', size: { count: 2 } }), /Measuring/)
})

test('a dataset promotion never quotes a weight that de-duplication can reduce', () => {
  assert.equal(weightNotice({ destination: 'dataset', size: { bytes: 65e6 } }), null)
})

test('promoteCount prefers the selection, then the per-target count', () => {
  assert.equal(promoteCount({ useSelection: true, selectedCount: 200 }), 200)
  assert.equal(promoteCount({ useSelection: false, promotable: 42 }), 42)
  assert.equal(promoteCount({ useSelection: false, size: { count: 7 } }), 7)
  assert.equal(promoteCount({ useSelection: false }), null)
})

test('the new-bank summary says analysis and triage decisions travel', () => {
  const s = promoteSummary({ destination: 'bank', useSelection: true, selectedCount: 200 })
  assert.match(s, /200 image\(s\)/)
  assert.match(s, /COPIED/)
  assert.match(s, /analysis/)
  assert.match(s, /keep\/pending\/reject decisions intact/)
  assert.match(s, /keeps every one of them, marked as promoted/)
})

test('the dataset summary promises byte preservation for Bank analysis', () => {
  const s = promoteSummary({
    destination: 'dataset', useSelection: false, promotable: 12, datasetChosen: true,
  })
  assert.match(s, /12 kept image\(s\) not yet in this dataset/)
  assert.match(s, /byte-for-byte/)
  assert.match(s, /Bank analysis can travel/)
  assert.match(s, /source folder are left as they are/)
  const none = promoteSummary({ destination: 'dataset', useSelection: false, datasetChosen: false })
  assert.match(none, /chosen dataset/)
})

test('the button arms only when its destination is actually specified', () => {
  assert.equal(canStartPromote({ destination: 'dataset', datasetId: '' }), false)
  assert.equal(canStartPromote({ destination: 'dataset', datasetId: '3' }), true)
  assert.equal(canStartPromote({ destination: 'bank', bankName: '   ' }), false)
  assert.equal(canStartPromote({ destination: 'bank', bankName: 'Candidates' }), true)
  // ...and never twice
  assert.equal(canStartPromote({ destination: 'bank', bankName: 'C', busy: true }), false)
})

test('the button names what it makes', () => {
  assert.equal(promoteButtonLabel({ destination: 'bank' }), 'Create bank')
  assert.equal(promoteButtonLabel({ destination: 'dataset' }), 'Promote')
  assert.equal(promoteButtonLabel({ destination: 'bank', busy: true }), 'Starting…')
})
