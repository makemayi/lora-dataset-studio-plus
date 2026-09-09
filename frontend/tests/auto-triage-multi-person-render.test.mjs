/**
 * The 👥👥 Multi-person one-click reject, RENDERED. The button lives in the
 * auto-triage bar and must survive three traps the source cannot show:
 * a decided dataset whose multi-person photos are all still there (the bar
 * used to unmount itself and take the fix with it), the count excluding both
 * rejected and UNMEASURED rows (NULL n_faces is "not counted", never "single
 * person"), and hooks running before the early return.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { AutoTriageBar } = await import(
  '../src/components/dataset/DatasetGrid.jsx')

const img = (id, over = {}) => ({
  id, status: 'pending', filename: `f${id}.png`,
  face_state: 'scorable', face_score: 0.5, n_faces: 1, ...over,
})

const bar = (images, props = {}) => renderToStaticMarkup(createElement(
  AutoTriageBar, {
    images, datasetId: 1, faceThresholds: { green: 0.4 },
    onBatch: () => {}, busy: false, applying: false,
    onApplyingChange: () => {}, ...props,
  }))

test('the multi-person button counts every not-yet-rejected multi-face image', () => {
  const html = bar([
    img(1, { n_faces: 2 }),                 // pending multi — counted
    img(2, { n_faces: 3, status: 'keep' }), // KEPT multi — counted too
    img(3, { n_faces: 3, status: 'reject' }), // already gone — not counted
    img(4, { n_faces: 1 }),                 // single — never counted
    img(5, { n_faces: null }),              // unmeasured — never counted
  ])
  assert.match(html, /👥👥 Multi-person \(/)
  assert.match(html, /\(2\)/)
})

test('a decided dataset with only multi-person photos left still shows the fix', () => {
  // Every image decided: the replay scope is empty, the old early return
  // unmounted the whole bar — and the fix with it.
  const html = bar([
    img(1, { n_faces: 2, status: 'keep' }),
    img(2, { n_faces: 2, status: 'reject' }),
  ])
  assert.match(html, /👥👥 Multi-person \(/)
  assert.match(html, /\(1\)/)
})

test('no multi-person images anywhere means no button', () => {
  const html = bar([img(1, { n_faces: 1 }), img(2, { n_faces: null })])
  assert.doesNotMatch(html, /Multi-person/)
})
