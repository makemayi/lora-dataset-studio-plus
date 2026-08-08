/**
 * 🖼🖼 A group of pinned images must RENDER — in every state it can be put in.
 *
 * The bug this pins down, reported from a real board: dragging one picture out
 * of a group blanked the app, `ReferenceError: barH is not defined`. Moving the
 * title bar into `CanvasGroupBar` took `const barH = groupBarHeight(...)` with
 * it and left the drag-out hint — the one block that only renders WHILE you are
 * pulling a picture off the strip — reading a binding nothing declared any more.
 *
 * Nothing caught it, and the reason is worth writing down: every canvas test
 * read these components as TEXT and matched regexes. A regex sees an attribute;
 * it cannot see a renderer it never ran. The file parsed, the suite was green,
 * and the first user gesture that reached that branch threw.
 *
 * So this file EXECUTES the component (tests/support/mountJsx.mjs) instead of
 * reading it, and it does so in each of the two states the board can put a
 * group in — `dropHint: null` and `dropHint: 'leaving'`. The second one is the
 * one that shipped broken, and it stayed broken precisely because no test had
 * ever passed that prop.
 *
 * The fixture is built by the REAL layout helper rather than hand-written, so
 * a group here has the shape a group has on the board.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { layoutImageNodes } from '../src/utils/canvasImageGroups.js'
import { groupBarHeight } from '../src/utils/canvasNodeChrome.js'
import { render } from './support/mountJsx.mjs'

/* ⚠️ Dynamic, and it has to be: the hooks that teach Node to read .jsx are
   installed while mountJsx.mjs is EVALUATED, and a static import of a .jsx file
   would already have been loaded by then — the whole graph is linked before the
   first line of any module runs. */
const { default: CanvasImageGroup } =
  await import('../src/components/canvas/CanvasImageGroup.jsx')
const { default: CanvasGroupBar } =
  await import('../src/components/canvas/CanvasGroupBar.jsx')

const img = (id, x, y, w, h, extra = {}) => ({
  imageId: id, x, y, w, h, visible: true, groupId: null, groupPos: null,
  image: { url: `/i/${id}.png`, record_id: id, step: 1000 * id }, ...extra,
})

const STRIP = [
  img(1, 100, 300, 200, 150, { groupId: 'g1', groupPos: 0 }),
  img(2, 300, 300, 200, 150, { groupId: 'g1', groupPos: 1 }),
]

const group = layoutImageNodes(STRIP).find((r) => r.kind === 'group')
const props = (extra) => ({
  group, datasetId: 7, laneName: 'a dataset', boardScale: 0.4, ...extra,
})

/* --- the crash, in as many words ------------------------------------------ */

test('a group RENDERS while a picture is being dragged out of it', () => {
  // This is the exact call that threw ReferenceError on a user's board. It is
  // a test that fails by throwing: any orphan binding in that branch lands
  // here, whatever it is called next time.
  const html = render(CanvasImageGroup, props({ dropHint: 'leaving' }))
  assert.match(html, /data-testid="canvas-group-drop-hint"/,
    'the drag-out hint is the whole point of this state')
  assert.match(html, /Drag it off the group to take it out/)
})

test('…and the hint is SIZED, not merely present', () => {
  // `barH` came back as a number or it did not come back: an undefined one
  // would render font-size:NaNpx, which is a hint nobody can read — a second,
  // quieter way for this branch to be broken while a "does it throw" test
  // stays green.
  const html = render(CanvasImageGroup, props({ dropHint: 'leaving' }))
  const sized = /font-size:\s*([\d.]+)px/.exec(html)
  assert.ok(sized, 'the hint must carry a numeric font-size')
  assert.ok(Number(sized[1]) > 0)
  // …and it is the SAME counter-scaled number the bar's own label uses, which
  // is why it is recomputed from the helper instead of guessed: at 40 % zoom
  // the hint has to grow exactly as the bar does.
  const barH = groupBarHeight(0.4, group.h)
  assert.equal(Number(sized[1]), Math.max(9, barH * 0.42))
})

test('the resting state renders too, and draws every member', () => {
  const html = render(CanvasImageGroup, props({ dropHint: null }))
  // On the testid, not on the sentence: a MEMBER's aria-label carries the same
  // words at rest, on purpose — the instruction has to reach a screen reader
  // too, not only the eye that sees the overlay.
  assert.doesNotMatch(html, /data-testid="canvas-group-drop-hint"/, 'no hint at rest')
  assert.equal(html.match(/data-canvas-image=""/g).length, STRIP.length)
})

test('a blended member draws its provenance badge without throwing', () => {
  // Another prop-only branch, and branches with no test are how this file's
  // bug got in. blendNotes is a Map, keyed by image id.
  const html = render(CanvasImageGroup, props({
    dropHint: 'leaving', blendNotes: new Map([[1, '2 of 3 sources on the board']]),
  }))
  assert.match(html, /2 of 3 sources on the board/)
})

/* --- the other half of the same extraction -------------------------------- */

test('the bar the drag-out hint was extracted from renders on its own', () => {
  const html = render(CanvasGroupBar, { group, datasetId: 7, boardScale: 0.4 })
  assert.match(html, /data-canvas-group-bar=""/)
  assert.match(html, /2 images/)
  assert.match(html, /Export grid/)
})

test('the crashing state renders at every zoom the board reaches', () => {
  // The zoom is the one input the hint's arithmetic depends on — its size comes
  // from the counter-scaled groupBarHeight — so it is the axis worth sweeping.
  // ✦ Fit lands around 24-45 % on a 400-px screen, which is where every chrome
  // bug on this board has been found.
  for (const boardScale of [0.1, 0.24, 0.4, 1, 2.5]) {
    const html = render(CanvasImageGroup, props({ boardScale, dropHint: 'leaving' }))
    const px = Number(/font-size:\s*([\d.]+)px/.exec(html)[1])
    assert.ok(px >= 9, `the hint must stay legible at ${boardScale} (got ${px})`)
  }
})
