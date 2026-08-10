/**
 * The workspace rail's recent-variations strip, RENDERED.
 *
 * It is the only surface in the app that shows output from OTHER datasets, and
 * the one thing it must never do is occupy the rail when there is nothing to
 * show — the rail's dead space is what it was built to use, not to double.
 */
import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'

import { render } from './support/mountJsx.mjs'

const { default: RecentVariations, RECENT_LIMIT } = await import(
  '../src/components/dataset/RecentVariations.jsx')

test('it draws nothing at all until something has been generated', () => {
  // The fetch never resolves in this environment, so this is the first paint on
  // a fresh install — where an empty heading and an empty row would be worse
  // than the dead space it replaces.
  assert.equal(render(RecentVariations, { onOpen: () => {} }).trim(), '')
})

test('the strip asks for a bounded number of images', () => {
  const src = readFileSync(
    new URL('../src/components/dataset/RecentVariations.jsx', import.meta.url), 'utf8')
  assert.ok(RECENT_LIMIT > 0 && RECENT_LIMIT <= 12)
  assert.match(src, /recent-images\?limit=\$\{RECENT_LIMIT\}/)
  // A shortcut is not worth a toast: it shows faces or it shows nothing.
  assert.match(src, /\.catch\(\(\) => \{\}\)/)
})

test('each face is a round button into its own dataset', () => {
  const src = readFileSync(
    new URL('../src/components/dataset/RecentVariations.jsx', import.meta.url), 'utf8')
  assert.match(src, /onClick=\{\(\) => onOpen\?\.\(img\.dataset_id\)\}/)
  assert.match(src, /rounded-full/)
  // The frame is pulled up because these are portraits — a face detection per
  // thumbnail would cost more than the feature is worth.
  assert.match(src, /objectPosition: '50% 28%'/)
  // The dataset you are already in is marked rather than hidden.
  assert.match(src, /aria-current=\{here \? 'true' : undefined\}/)
})

test('the rail mounts it under the checklist, not in place of it', () => {
  const workspace = readFileSync(
    new URL('../src/components/dataset/DatasetWorkspace.jsx', import.meta.url), 'utf8')
  assert.match(workspace, /<RecentVariations onOpen=\{ds\.open\} currentId=\{d\.id\}/)
  const checklist = workspace.indexOf('<GuidedChecklist')
  assert.ok(checklist > 0 && workspace.indexOf('<RecentVariations') > checklist)
})
