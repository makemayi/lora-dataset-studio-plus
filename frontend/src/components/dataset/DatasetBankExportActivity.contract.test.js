import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8')

test('Dataset to Bank copy has a named, count-aware activity banner', () => {
  assert.match(workspace, /bank_export: `Copying into a Bank…\$\{prog\}`/)
  assert.match(workspace, /\['bank_export', 'bank_import', 'training_export'\]/)
})

test('Dataset copy and training freeze activities are named and never claim GPU use', () => {
  assert.match(workspace, /\|\| act\.kind === 'bank_export'/)
  assert.match(workspace, /\|\| act\.kind === 'bank_import'/)
  assert.match(workspace, /\|\| act\.kind === 'training_export'/)
  assert.match(workspace, /bank_import: `Copying images from a Bank…\$\{prog\}`/)
  assert.match(workspace, /training_export: 'Freezing the Dataset for training…'/)
})
