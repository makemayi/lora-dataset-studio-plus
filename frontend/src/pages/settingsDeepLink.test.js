import test from 'node:test'
import assert from 'node:assert/strict'
import { shouldScrollToSection, focusNeedsRescroll } from './settingsDeepLink.js'

const base = { hasSection: true, hasFocus: false, loading: false, panelTop: 900, viewportHeight: 800 }

test('a section deep link below the fold scrolls to the panel', () => {
  // The reported case: "Settings › Image engines →" on a phone, where the rail
  // stacks above the panel and the section lands off-screen.
  assert.equal(shouldScrollToSection(base), true)
})

test('a panel already on screen never jumps', () => {
  // Desktop: the grid puts the panel beside the rail, already at the top.
  assert.equal(shouldScrollToSection({ ...base, panelTop: 120, viewportHeight: 800 }), false)
})

test('a panel scrolled just off the top still gets pulled back', () => {
  assert.equal(shouldScrollToSection({ ...base, panelTop: -300 }), true)
})

test('a ?focus= link keeps the scroll for its own field', () => {
  // Two scrolls would fight, and the exact field is the better target.
  assert.equal(shouldScrollToSection({ ...base, hasFocus: true }), false)
})

test('bare /settings does not move the page', () => {
  assert.equal(shouldScrollToSection({ ...base, hasSection: false }), false)
})

test('nothing moves while the settings are still loading', () => {
  // The panel is not rendered yet, so its position is meaningless.
  assert.equal(shouldScrollToSection({ ...base, loading: true }), false)
})

test('missing measurements are treated as "do not move"', () => {
  assert.equal(shouldScrollToSection({ ...base, panelTop: undefined }), false)
  assert.equal(shouldScrollToSection({ ...base, viewportHeight: undefined }), false)
  assert.equal(shouldScrollToSection(), false)
})

/* The arrival correction. Measured case: the Engines deep link left the improve
   prompt 116 px below an 800 px fold two seconds in, because panels above it were
   still rendering — and the highlight had already expired. */
test('a field left below the fold by late-rendering panels is re-scrolled', () => {
  assert.equal(focusNeedsRescroll({ top: 916, height: 86, viewportHeight: 800 }), true)
})

test('a field partly cut off at either edge still counts as not arrived', () => {
  assert.equal(focusNeedsRescroll({ top: 760, height: 86, viewportHeight: 800 }), true)  // bottom
  assert.equal(focusNeedsRescroll({ top: -10, height: 86, viewportHeight: 800 }), true)  // top
  assert.equal(focusNeedsRescroll({ top: 10, height: 86, viewportHeight: 800 }), true)   // under the header
})

test('a field comfortably in view is LEFT ALONE — the reader may have scrolled', () => {
  assert.equal(focusNeedsRescroll({ top: 513, height: 86, viewportHeight: 800 }), false)
  assert.equal(focusNeedsRescroll({ top: 407, height: 86, viewportHeight: 900 }), false)
})

test('missing or nonsense measurements never trigger a scroll', () => {
  assert.equal(focusNeedsRescroll(), false)
  assert.equal(focusNeedsRescroll({ top: 10, height: 86 }), false)
  assert.equal(focusNeedsRescroll({ top: 10, height: 86, viewportHeight: 0 }), false)
})
