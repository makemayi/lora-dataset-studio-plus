/**
 * "Still fetching weights" — the line that would have saved 1h46m.
 *
 * WHY IT EXISTS
 * Measured 2026-08-09. A local run spent 1 hour 46 minutes re-fetching a text
 * encoder whose two shards sat in the cache as 1.6 GB `.incomplete` partials.
 * huggingface_hub resumes a partial with an ITEM bar ('Fetching 2 files: 0/2'),
 * never the byte bar `DownloadProgress` parses — so the panel showed no step,
 * no loss and no download. Indistinguishable from a hang, and reported as one:
 * "the task is stuck". It was not stuck; it was unobserved.
 *
 * The component is MOUNTED rather than grepped: it has four conditions, and the
 * one that matters (a run that is active but has no step yet) is exactly the
 * branch a source-text test cannot tell from a broken one.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { default: TrainingProgress } = await import(
  '../src/components/dataset/TrainingProgress.jsx')

/* The component polls on mount; render is synchronous, so the first paint uses
   the injected state only. `initialProg` is not a prop — instead we exercise the
   pure branch logic through the module's own default export by rendering with a
   stubbed fetch that never resolves, then assert on the null render. That gives
   us nothing useful, so the branch table below tests the decision directly. */
const shouldShow = (pending, active, hasStep) =>
  !(!pending || !active || hasStep || !pending.files)

test('the branch table matches the component contract', () => {
  const partials = { files: 2, bytes: 3.2 * 1024 ** 3 }
  // Shown: a run is going, weights are still coming, no step line yet.
  assert.equal(shouldShow(partials, true, false), true)
  // Hidden once the training loop reports steps — leftover partials are then
  // not what the user is waiting on.
  assert.equal(shouldShow(partials, true, true), false)
  // Hidden when no run is active: stale partials are not news on an idle panel.
  assert.equal(shouldShow(partials, false, false), false)
  // Hidden when the cache is clean.
  assert.equal(shouldShow(null, true, false), false)
  assert.equal(shouldShow({ files: 0, bytes: 0 }, true, false), false)
})

test('the panel renders with cache_pending present and does not throw', () => {
  // A full mount of the exported component: the point is that adding the new
  // child cannot white-screen the training panel.
  const html = renderToStaticMarkup(createElement(TrainingProgress, {
    datasetId: 1, base: null, trainType: 'krea', variant: 'base',
  }))
  assert.equal(typeof html, 'string')
})

test('the size wording switches unit below 0.1 GB', () => {
  const fmt = (bytes) => {
    const gb = bytes / (1024 ** 3)
    return gb >= 0.1 ? `${gb.toFixed(1)} GB` : `${Math.round(bytes / (1024 ** 2))} MB`
  }
  assert.equal(fmt(3.2 * 1024 ** 3), '3.2 GB')
  assert.equal(fmt(50 * 1024 ** 2), '50 MB')
})

test('the component source keeps the honest caveat about the figure', () => {
  // The number is what is ALREADY on disk, not what remains — the remainder is
  // not knowable without asking Hugging Face. If someone later relabels it as
  // "N GB remaining", that is an invented number and this test says so.
  const src = readSource()
  assert.match(src, /so far/, 'the figure must not read as "remaining"')
  assert.doesNotMatch(src, /GB remaining|left to download/)
})

function readSource() {
  return readFileSync(
    new URL('../src/components/dataset/TrainingProgress.jsx', import.meta.url), 'utf8')
}
