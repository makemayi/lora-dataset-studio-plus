import test from 'node:test';
import assert from 'node:assert/strict';
import {
  chromeScale, chromeScreenSize, isNodeControlTarget, nodePointerIntent,
} from './canvasNodeChrome.js';

/* The ✕ that "did not work" on a phone.

   The first test is the bug, in one number: at the zoom the report came from,
   the close button measured about ten screen pixels. Nothing was broken —
   nothing was reachable. */

// A stand-in for a DOM node: `closest` answers from a list of selectors the
// element is inside of. Enough to pin the rules without a DOM.
const el = (...inside) => ({
  closest: (sel) => (inside.includes(sel) ? { sel } : null),
});

const CLOSE_BUTTON = el('[data-canvas-image] button', '[data-canvas-image]');
const RESIZE_CORNER = el('[data-canvas-image-resize]', '[data-canvas-image]');
const PICTURE = el('[data-canvas-image]');

// ---- THE bug --------------------------------------------------------------

test('the ✕ is a real touch target at the zoom the board is actually read at', () => {
  // 65 % — the zoom on the report. Untouched, the 28-unit button lands at 18 px.
  assert.ok(28 * 0.65 < 24, 'the raw control is under any touch guideline');
  // Counter-scaled, it is back to its nominal size under the finger.
  assert.ok(chromeScreenSize(0.65, 320) >= 24,
    `got ${chromeScreenSize(0.65, 320)} px at 65 %`);
  // And at the far end of the range — a board fitted to twenty runs. Here the
  // cluster cap bites before the ideal does (a 320-unit pin is only 77 px on
  // screen at that zoom, and two 28-px buttons would cover most of it), so the
  // honest claim is "about four times bigger than the bug", not "28 px".
  assert.ok(chromeScreenSize(0.24, 320) >= 20,
    `got ${chromeScreenSize(0.24, 320)} px at 24 %`);
  assert.ok(chromeScreenSize(0.24, 320) > 3 * (14 * 0.24),
    'and far bigger than the glyph it replaces');
});

test('zooming IN does not blow the control up', () => {
  assert.equal(chromeScale(1, 320), 1);
  assert.equal(chromeScale(2.5, 320), 1, 'never smaller than its board size');
});

test('the control CLUSTER never eats the picture it decorates', () => {
  // A contact-sheet thumbnail on a board fitted very far out: a constant-size
  // pair of buttons would cover the whole tile. The cap wins, and the fact that
  // the target is then below the touch guideline is a consequence of the tile
  // being 45 px wide — there is no honest way around that, and it is stated
  // here rather than hidden. Zooming in is the answer, and it works.
  const tiny = chromeScale(0.28, 160);
  assert.ok(tiny * 64 <= 160 * 0.7 + 1e-9,
    'the 🔍+✕ cluster stays under 70 % of the tile');
  assert.ok(tiny < 1 / 0.28, 'the cap, not the ideal, is what applies');
  // A full-size pin at the same zoom has room for more, and gets it.
  assert.ok(chromeScreenSize(0.28, 320) > chromeScreenSize(0.28, 160));
});

test('a nonsense zoom is not allowed to produce a nonsense control', () => {
  assert.equal(chromeScale(0, 320), 1);
  assert.equal(chromeScale(-3, 320), 1);
  assert.equal(chromeScale(NaN, 320), 1);
  assert.equal(chromeScale(0.5, NaN), 2, 'no node width ⇒ no cap, just the ideal');
});

// ---- the guard: a control is never a gesture ------------------------------

test('a press on the ✕ is a control, never a drag and never a long press', () => {
  assert.equal(nodePointerIntent(CLOSE_BUTTON, 'touch'), 'control');
  assert.equal(nodePointerIntent(CLOSE_BUTTON, 'mouse'), 'control');
  assert.ok(isNodeControlTarget(CLOSE_BUTTON));
});

test('the resize corner answers at once, finger or mouse', () => {
  assert.equal(nodePointerIntent(RESIZE_CORNER, 'touch'), 'resize');
  assert.equal(nodePointerIntent(RESIZE_CORNER, 'mouse'), 'resize');
  assert.equal(isNodeControlTarget(RESIZE_CORNER), false,
    'the corner is a GESTURE, not a button: it wants the pointer capture');
});

test('the picture itself still moves — the fix must not cost the drag', () => {
  assert.equal(nodePointerIntent(PICTURE, 'mouse'), 'move');
  assert.equal(nodePointerIntent(PICTURE, 'pen'), 'move');
  assert.equal(nodePointerIntent(PICTURE, 'touch'), 'press',
    'touch pans until the long press picks the node up');
  assert.ok(!isNodeControlTarget(PICTURE));
});

test('something with no closest() at all does not throw', () => {
  assert.equal(isNodeControlTarget(null), false);
  assert.equal(isNodeControlTarget({}), false);
  assert.equal(nodePointerIntent(null, 'touch'), 'press');
});
