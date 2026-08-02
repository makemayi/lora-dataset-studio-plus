import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CLUSTER_UNITS, chromeScale, chromeScreenSize, clusterBox, groupBarHeight,
  isNodeControlTarget, nodePointerIntent,
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
  assert.ok(tiny * CLUSTER_UNITS <= 160 * 0.7 + 1e-9,
    'the 🔍+⬇+✕ cluster stays under 70 % of the tile');
  assert.ok(tiny < 1 / 0.28, 'the cap, not the ideal, is what applies');
  // A full-size pin at the same zoom has room for more, and gets it.
  assert.ok(chromeScreenSize(0.28, 320) > chromeScreenSize(0.28, 160));
});

test('⬇ Download joined the cluster WITHOUT shrinking ✕ and 🔍', () => {
  // The regression this pins: the cap is spent on the cluster's WIDTH, so a row
  // that grew from two buttons to three would have spent 50 % more of it and
  // taken every target down with it — at 24 % zoom, from 20 px to 15.7 px,
  // which is the old unhittable-✕ bug creeping back in. So the third control
  // wraps onto a second line and the width budget is untouched.
  assert.equal(CLUSTER_UNITS, 64, 'two columns wide, whatever is in it');
  assert.equal(clusterBox(3).maxWidth, CLUSTER_UNITS);
  assert.equal(clusterBox(3).rows, 2, 'the third control goes UNDER, not beside');
  assert.equal(clusterBox(2).rows, 1);
  // …so the guarantees at every zoom the board is read at are the ones the
  // two-button cluster already made.
  assert.ok(chromeScreenSize(0.45, 320) >= 24,
    `got ${chromeScreenSize(0.45, 320)} px at 45 %`);
  assert.ok(chromeScreenSize(0.24, 320) >= 20,
    `got ${chromeScreenSize(0.24, 320)} px at 24 %`);
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

// ---- 🖼🖼 the grip of a GROUP of pinned images ----------------------------

const GROUP_BAR = el('[data-canvas-group-bar]', '[data-canvas-group]');
const GROUP_CLOSE = el('[data-canvas-group-bar] button', '[data-canvas-group-bar]',
  '[data-canvas-group]');

test('a group’s title bar moves the whole strip, on any pointer type', () => {
  assert.equal(nodePointerIntent(GROUP_BAR, 'mouse'), 'group-move');
  assert.equal(nodePointerIntent(GROUP_BAR, 'touch'), 'group-move',
    'the bar is the only grip a group has — a finger must not have to wait');
  assert.equal(isNodeControlTarget(GROUP_BAR), false, 'it is a gesture: it wants the capture');
});

test('a group’s own ✕ is a button, exactly like a picture’s', () => {
  assert.equal(nodePointerIntent(GROUP_CLOSE, 'touch'), 'control');
  assert.ok(isNodeControlTarget(GROUP_CLOSE),
    'without this the frame captures the pointer and the ✕ never hears the click');
});

test('the group bar stays a finger-sized grip at the zoom the board is read at', () => {
  // 400 board units tall, read at 24 % — the far end of "zoomed out".
  const h = groupBarHeight(0.24, 400);
  assert.ok(h * 0.24 >= 24, `the bar measured ${(h * 0.24).toFixed(1)} px on screen`);
  // Zoomed IN it must not balloon: a constant screen size, never smaller.
  assert.equal(groupBarHeight(2, 400), 26);
});

test('the group bar never eats the strip it labels', () => {
  // A tiny strip at a tiny zoom: the counter-scale is capped by the strip.
  assert.ok(groupBarHeight(0.05, 100) <= Math.max(26, 100 * 0.35) + 1e-9);
});

test('groupBarHeight survives nonsense', () => {
  assert.equal(groupBarHeight(0, 400), 26);
  assert.equal(groupBarHeight(NaN, 400), 26);
  assert.ok(Number.isFinite(groupBarHeight(1, NaN)));
});

/* 🪪 A control that lives in the zoomed world WITHOUT belonging to a pinned
   node — the lane header's reference thumbnail is the first one. Caught in the
   browser, not by a test: the button was correct, its handler was correct, and
   clicking it did nothing, because the frame captured the pointer and the click
   that followed was retargeted away from it. The generic marker exists so the
   next thing added to the world opts out instead of rediscovering this. */
const LANE_CONTROL = el('[data-canvas-control]');

test('a control in the world opts out of the board gesture by marker alone', () => {
  assert.ok(isNodeControlTarget(LANE_CONTROL),
    'no capture ⇒ the click reaches the button');
  assert.equal(nodePointerIntent(LANE_CONTROL, 'mouse'), 'control');
  assert.equal(nodePointerIntent(LANE_CONTROL, 'touch'), 'control',
    'and it never arms the long press either');
});
