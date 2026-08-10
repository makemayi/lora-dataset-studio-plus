/**
 * The workspace rail's recent-variations strip, RENDERED WITH ITEMS.
 *
 * The first version of this file only asserted the EMPTY case, because the
 * component both fetched and rendered and a static render never runs an effect.
 * That is precisely how it shipped broken: `apiFetch` resolves to the parsed
 * body, not to a Response, so the fetch stored nothing and the strip was empty
 * forever — and the one test that could run passed, because "renders nothing"
 * was what it asserted.
 *
 * The list is a pure function of its items now, so the case that matters is the
 * one being tested.
 */
import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'

import { render } from './support/mountJsx.mjs'

const { default: RecentVariations, RecentVariationsList, RECENT_LIMIT } =
  await import('../src/components/dataset/RecentVariations.jsx')

const IMAGES = [
  { image_id: 9, dataset_id: 3, dataset_name: 'Ana', filename: 'a 1.png', label: 'Bust' },
  { image_id: 8, dataset_id: 5, dataset_name: 'Bea', filename: 'b1.png', label: '' },
]

test('every face is a round button into its own dataset', () => {
  const html = render(RecentVariationsList, { images: IMAGES, onOpen: () => {} })
  assert.match(html, /Recent/)
  assert.match(html, /rounded-full/)
  // The filename is URL-encoded — a space in it must not break the src.
  assert.match(html, /\/api\/dataset\/3\/img\/a%201\.png/)
  assert.match(html, /\/api\/dataset\/5\/img\/b1\.png/)
  assert.match(html, /Open Ana — Bust/)
  // A dataset with no label still gets a usable title.
  assert.match(html, /Open Bea/)
})

test('the dataset you are already in is marked, not hidden', () => {
  const html = render(RecentVariationsList,
    { images: IMAGES, onOpen: () => {}, currentId: 3 })
  assert.match(html, /aria-current="true"/)
  assert.match(html, /this dataset/)
  // ...and it is the only one so marked.
  assert.equal(html.match(/aria-current="true"/g).length, 1)
})

test('it draws nothing at all when there is nothing to show', () => {
  /* A fresh install must not pay a permanently empty section — the rail's dead
     space is what this replaces, not what it doubles. */
  assert.equal(render(RecentVariationsList, { images: [], onOpen: () => {} }).trim(), '')
  assert.equal(render(RecentVariationsList, { images: null, onOpen: () => {} }).trim(), '')
  // The fetching wrapper renders nothing before its request resolves, too.
  assert.equal(render(RecentVariations, { onOpen: () => {} }).trim(), '')
})

test('the wrapper reads apiFetch the way apiFetch actually behaves', () => {
  const src = readFileSync(
    new URL('../src/components/dataset/RecentVariations.jsx', import.meta.url), 'utf8')
  assert.ok(RECENT_LIMIT > 0 && RECENT_LIMIT <= 12)
  assert.match(src, /recent-images\?limit=\$\{RECENT_LIMIT\}/)
  // It resolves to the PARSED BODY and throws on a bad status; treating it as a
  // Response is what broke the first version. (Matched on the CALL, not on the
  // prose: the file explains that mistake in its own header.)
  assert.match(src, /\.then\(\(d\) => \{ if \(alive\) setImages/)
  assert.doesNotMatch(src, /then\(\(r\) => \(r\.ok/)
  assert.match(src, /\.catch\(\(\) => \{\}\)/)
})

test('the rail mounts it under the checklist, not in place of it', () => {
  const workspace = readFileSync(
    new URL('../src/components/dataset/DatasetWorkspace.jsx', import.meta.url), 'utf8')
  assert.match(workspace, /<RecentVariations onOpen=\{ds\.open\} currentId=\{d\.id\}/)
  const checklist = workspace.indexOf('<GuidedChecklist')
  assert.ok(checklist > 0 && workspace.indexOf('<RecentVariations') > checklist)
})
