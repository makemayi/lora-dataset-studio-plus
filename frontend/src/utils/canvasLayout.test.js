import assert from 'node:assert/strict';
import test from 'node:test';
import {
  INITIAL_MIN_SCALE, LANE_GAP, LANE_HEADER_H, MAX_SCALE, MIN_SCALE,
  clampScale, clampView, fitView, initialView, panBy, pinchCenter, pinchDistance,
  stackLanes, toWorld, viewTransform, zoomAt,
} from './canvasLayout.js';

// ---- stackLanes ------------------------------------------------------------

test('an empty board has no lanes and no size', () => {
  assert.deepEqual(stackLanes([]), { lanes: [], width: 0, height: 0 });
  assert.deepEqual(stackLanes(null), { lanes: [], width: 0, height: 0 });
});

test('one lane sits at the top, its graph below its header', () => {
  const { lanes, width, height } = stackLanes([{ datasetId: 1, width: 500, height: 200 }]);
  assert.equal(lanes.length, 1);
  assert.equal(lanes[0].y, 0);
  assert.equal(lanes[0].graphY, LANE_HEADER_H);
  assert.equal(lanes[0].x, 0);
  assert.equal(width, 500);
  // No trailing gap: the board ends where the content ends.
  assert.equal(height, LANE_HEADER_H + 200);
});

test('lanes stack without overlapping, and the board is as wide as the widest', () => {
  const { lanes, width, height } = stackLanes([
    { datasetId: 1, width: 400, height: 100 },
    { datasetId: 2, width: 900, height: 250 },
    { datasetId: 3, width: 300, height: 80 },
  ]);
  assert.deepEqual(lanes.map((l) => l.datasetId), [1, 2, 3]);
  for (let i = 1; i < lanes.length; i += 1) {
    const prevBottom = lanes[i - 1].graphY + lanes[i - 1].height;
    assert.ok(lanes[i].y >= prevBottom, `lane ${i} overlaps the one above`);
    assert.equal(lanes[i].y - prevBottom, LANE_GAP);
  }
  assert.equal(width, 900);
  assert.equal(height, lanes[2].graphY + 80);
});

test('a lane whose graph has no size still keeps its place', () => {
  // A dataset still loading (or one whose runs all vanished) must not make the
  // lanes below it jump when its real size arrives.
  const { lanes } = stackLanes([
    { datasetId: 1, width: 0, height: 0 },
    { datasetId: 2, width: 300, height: 120 },
  ]);
  assert.equal(lanes[0].height, 0);
  assert.equal(lanes[1].y, LANE_HEADER_H + 0 + LANE_GAP);
});

test('nonsense sizes degrade to zero instead of NaN-ing the board', () => {
  const { lanes, width, height } = stackLanes([{ datasetId: 1, width: 'x', height: -50 }]);
  assert.equal(lanes[0].width, 0);
  assert.equal(lanes[0].height, 0);
  assert.equal(width, 0);
  assert.equal(height, LANE_HEADER_H);
});

// ---- scale -----------------------------------------------------------------

test('clampScale keeps the zoom usable and survives nonsense', () => {
  assert.equal(MAX_SCALE, 5, 'the canvas can zoom in to 500%');
  assert.equal(clampScale(0.0001), MIN_SCALE);
  assert.equal(clampScale(50), MAX_SCALE);
  assert.equal(clampScale(0.8), 0.8);
  assert.equal(clampScale(undefined), 1);
  assert.equal(clampScale(NaN), 1);
});

// ---- fitView ---------------------------------------------------------------

test('a board bigger than the frame is scaled down to fit both axes', () => {
  const v = fitView({ width: 2000, height: 1000 }, { width: 800, height: 600 }, { padding: 0 });
  assert.ok(v.scale < 1);
  assert.equal(v.scale, 800 / 2000);          // width is the binding axis here
  assert.ok(2000 * v.scale <= 800 + 0.001);
  assert.ok(1000 * v.scale <= 600 + 0.001);
});

test('height can be the binding axis too', () => {
  const v = fitView({ width: 400, height: 4000 }, { width: 800, height: 600 }, { padding: 0 });
  assert.equal(v.scale, 600 / 4000);
});

test('a small board is never magnified past 1', () => {
  const v = fitView({ width: 100, height: 100 }, { width: 900, height: 700 }, { padding: 0 });
  assert.equal(v.scale, 1);
  // …and it is centred rather than pinned to a corner.
  assert.equal(v.tx, (900 - 100) / 2);
  assert.equal(v.ty, (700 - 100) / 2);
});

test('an empty board or an unmeasured frame answers a safe identity view', () => {
  for (const v of [
    fitView({ width: 0, height: 0 }, { width: 800, height: 600 }),
    fitView({ width: 500, height: 500 }, { width: 0, height: 0 }),
  ]) {
    assert.equal(v.scale, 1);
    assert.ok(Number.isFinite(v.tx) && Number.isFinite(v.ty));
  }
});

// ---- initialView -----------------------------------------------------------

test('the board never OPENS below the legibility floor, even on a phone', () => {
  // A three-dataset board fits a phone frame at ~35 %, where a run card is a
  // few pixels tall. Opening closer and letting the user scroll is the deal.
  const world = { width: 1100, height: 1200 };
  const phone = { width: 370, height: 480 };
  const fit = fitView(world, phone);
  const open = initialView(world, phone);
  assert.ok(fit.scale < INITIAL_MIN_SCALE, 'precondition: this board does fit very small');
  assert.equal(open.scale, INITIAL_MIN_SCALE);
});

test('a board that fits comfortably opens exactly at its fit', () => {
  const world = { width: 900, height: 500 };
  const frame = { width: 1200, height: 800 };
  assert.equal(initialView(world, frame).scale, fitView(world, frame).scale);
});

test('content taller than the frame opens at the TOP, not centred', () => {
  const open = initialView({ width: 300, height: 4000 }, { width: 900, height: 500 }, { padding: 16 });
  assert.equal(open.ty, 16);
});

test('content that fits is centred HORIZONTALLY but still opens at the top', () => {
  const open = initialView({ width: 200, height: 100 }, { width: 900, height: 500 }, { padding: 16 });
  assert.equal(open.tx, (900 - 200) / 2);
  assert.equal(open.ty, 16);
});

test('a short board on a phone wastes no sky above its first lane', () => {
  // The frame is sized from the viewport, not from the content, so vertical
  // centring here spent a third of a 400-px screen on nothing. The first lane
  // must start within one padding of the top whatever the board's height.
  const phone = { width: 400, height: 420 };
  for (const height of [120, 240, 400, 2000]) {
    const open = initialView({ width: 700, height }, phone, { padding: 16 });
    assert.equal(open.ty, 16, `board ${height} tall opened at ty=${open.ty}`);
  }
});

// ---- zoom / pan ------------------------------------------------------------

test('zooming keeps the point under the cursor exactly where it is', () => {
  const view = { scale: 0.7, tx: 40, ty: -25 };
  const anchor = { x: 310, y: 190 };
  const before = toWorld(view, anchor.x, anchor.y);
  const zoomed = zoomAt(view, 1.6, anchor);
  const after = toWorld(zoomed, anchor.x, anchor.y);
  assert.ok(Math.abs(after.x - before.x) < 1e-9, 'x drifted');
  assert.ok(Math.abs(after.y - before.y) < 1e-9, 'y drifted');
});

test('zooming past the ceiling still does not shove the board', () => {
  // The translate must be recomputed from the CLAMPED scale — computing it from
  // the requested one is how a canvas jumps when you keep pinching at max zoom.
  const view = { scale: MAX_SCALE, tx: 10, ty: 10 };
  const anchor = { x: 200, y: 100 };
  const zoomed = zoomAt(view, 4, anchor);
  assert.equal(zoomed.scale, MAX_SCALE);
  assert.ok(Math.abs(zoomed.tx - view.tx) < 1e-9);
  assert.ok(Math.abs(zoomed.ty - view.ty) < 1e-9);
});

test('zooming out past the floor is clamped as well', () => {
  const out = zoomAt({ scale: MIN_SCALE, tx: 0, ty: 0 }, 0.01, { x: 0, y: 0 });
  assert.equal(out.scale, MIN_SCALE);
});

test('panBy adds a screen delta and preserves the scale', () => {
  assert.deepEqual(panBy({ scale: 0.5, tx: 10, ty: 20 }, -30, 15),
    { scale: 0.5, tx: -20, ty: 35 });
});

// ---- clampView -------------------------------------------------------------

test('the board can never be flung completely out of the frame', () => {
  const world = { width: 1000, height: 800 };
  const viewport = { width: 600, height: 400 };
  const far = clampView({ scale: 1, tx: 99999, ty: -99999 }, world, viewport, { keep: 80 });
  // At least `keep` px of content stay inside on each axis.
  assert.ok(far.tx <= viewport.width - 80);
  assert.ok(far.tx + world.width >= 80);
  assert.ok(far.ty <= viewport.height - 80);
  assert.ok(far.ty + world.height >= 80);
});

test('a view already inside the frame is left alone', () => {
  const v = { scale: 1, tx: 20, ty: 30 };
  assert.deepEqual(clampView(v, { width: 500, height: 300 }, { width: 600, height: 400 }), v);
});

test('clamping an empty board is a no-op, not a division by zero', () => {
  const v = { scale: 1, tx: 5, ty: 7 };
  assert.deepEqual(clampView(v, { width: 0, height: 0 }, { width: 600, height: 400 }), v);
});

// ---- transform + pinch helpers --------------------------------------------

test('viewTransform translates first, then scales (the order toWorld inverts)', () => {
  assert.equal(viewTransform({ scale: 0.5, tx: 12, ty: -4 }),
    'translate(12px, -4px) scale(0.5)');
  assert.equal(viewTransform(null), 'translate(0px, 0px) scale(1)');
});

test('pinch helpers measure distance and midpoint', () => {
  assert.equal(pinchDistance({ x: 0, y: 0 }, { x: 3, y: 4 }), 5);
  assert.deepEqual(pinchCenter({ x: 0, y: 10 }, { x: 4, y: 0 }), { x: 2, y: 5 });
  assert.equal(pinchDistance(null, null), 0);
});
