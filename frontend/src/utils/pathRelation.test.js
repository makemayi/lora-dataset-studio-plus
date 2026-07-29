import test from 'node:test'
import assert from 'node:assert/strict'
import {
  datasetFolderNotice, looksWindows, normalizePath, pathRelation,
} from './pathRelation.js'

const DS = [{ id: 7, name: 'Lola', storage_path: 'C:\\LDS\\data\\datasets\\7' }]

test('the same Windows folder spelled differently is the same folder', () => {
  const a = 'C:\\LDS\\data\\datasets\\7'
  assert.equal(pathRelation(a, 'C:/LDS/data/datasets/7'), 'same')     // separators
  assert.equal(pathRelation(a, 'C:\\lds\\DATA\\Datasets\\7'), 'same') // case
  assert.equal(pathRelation(a, 'C:\\LDS\\data\\datasets\\7\\'), 'same')
  assert.equal(pathRelation(a, '"C:\\LDS\\data\\datasets\\7"'), 'same')  // Copy as path
  assert.equal(pathRelation(a, 'C:\\LDS\\data\\datasets\\8\\..\\7'), 'same')
})

test('POSIX keeps its case — two spellings really are two folders', () => {
  assert.equal(pathRelation('/srv/lds/datasets/7', '/srv/lds/datasets/7'), 'same')
  assert.equal(pathRelation('/srv/LDS/datasets/7', '/srv/lds/datasets/7'), null)
})

test('containment fires only on a separator boundary', () => {
  assert.equal(pathRelation('/a/data2', '/a/data'), null)
  assert.equal(pathRelation('/a/data/7', '/a/data'), 'inside')
  assert.equal(pathRelation('/a/data', '/a/data/7'), 'contains')
})

test('nothing to compare never claims a relation', () => {
  assert.equal(pathRelation('', '/a'), null)
  assert.equal(pathRelation(null, undefined), null)
  assert.equal(normalizePath('   '), '')
  assert.equal(normalizePath('/..'), '/')          // never climbs past the root
})

test('a drive letter or a UNC share is what makes a path Windows', () => {
  assert.equal(looksWindows('C:\\x'), true)
  assert.equal(looksWindows('\\\\nas\\share'), true)
  assert.equal(looksWindows('/home/me/a\\b'), false)   // legal POSIX filename
})

test('a dataset folder is flagged, and the notice says what to do instead', () => {
  const n = datasetFolderNotice('C:/LDS/data/datasets/7', DS)
  assert.equal(n.datasetId, 7)
  assert.equal(n.relation, 'same')
  assert.match(n.text, /Lola/)
  assert.match(n.text, /Import to bank/)
})

test('a subfolder of a dataset, and a parent of it, are both flagged', () => {
  assert.equal(datasetFolderNotice('C:/LDS/data/datasets/7/sub', DS).relation, 'inside')
  assert.equal(datasetFolderNotice('C:/LDS/data', DS).relation, 'contains')
})

test('a legitimate folder is never flagged', () => {
  assert.equal(datasetFolderNotice('D:/scrape/telegram-dump', DS), null)
  assert.equal(datasetFolderNotice('C:/LDS/data/datasets2', DS), null)
  assert.equal(datasetFolderNotice('', DS), null)
  assert.equal(datasetFolderNotice('C:/x', [{ id: 1 }]), null)   // no path known
})
