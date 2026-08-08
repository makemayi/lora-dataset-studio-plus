import test from 'node:test';
import assert from 'node:assert/strict';
import {
  IMG_MAX, IMG_MIN, IMG_REACH, clampImageBox, defaultImageSpot, imageNodeEdges,
  imageNodeExtent, nudgeImageNode, openGeometry, slideBelow, spotBesideCard,
  toImageNodeMap, visibleImageNodes,
} from './canvasImageNodes.js';

/* 🖼 Images pinned on the ◉ LoRA Canvas.

   The headline assertion is the first test: closing a pinned image and opening
   it again must restore the position AND the size it was closed at. Everything
   below it guards a pointer that outlives what it points at. */

const row = (imageId, over = {}) => ({
  image_id: imageId, x: 10, y: 20, w: 300, h: 300, visible: true,
  image: { id: imageId, url: `/img/${imageId}.png`, record_id: 7, step: 2500 },
  ...over,
});

// ---- THE assertion --------------------------------------------------------

test('re-opening a closed image restores its position AND its size', () => {
  // Pinned, dragged to a deliberate spot and resized.
  const placed = toImageNodeMap([row(42, { x: 640, y: 275.5, w: 420, h: 310 })]);
  // Closed: the geometry stays, only `visible` flips.
  const closed = toImageNodeMap([
    { ...placed[42], image_id: 42, visible: false }]);
  assert.equal(visibleImageNodes(closed).length, 0, 'a closed image leaves the board');

  // Re-opened from the gallery. The fallback is where a NEW pin would land; a
  // remembered node must ignore it completely.
  const geo = openGeometry(closed, 42, { x: 0, y: 0, w: 320, h: 320 });
  assert.deepEqual(geo, { x: 640, y: 275.5, w: 420, h: 310 });
});

test('an image never pinned before opens at the fallback spot', () => {
  const geo = openGeometry({}, 99, { x: 312, y: 40, w: 320, h: 320 });
  assert.deepEqual(geo, { x: 312, y: 40, w: 320, h: 320 });
});

// ---- geometry -------------------------------------------------------------

test('a node is clamped into a usable size', () => {
  assert.deepEqual(clampImageBox({ x: 0, y: 0, w: 9000, h: 8000 }),
    { x: 0, y: 0, w: IMG_MAX, h: IMG_MAX });
  assert.deepEqual(clampImageBox({ x: 1, y: 2, w: 3, h: 4 }),
    { x: 1, y: 2, w: IMG_MIN, h: IMG_MIN });
});

/* THE assertion of free placement. Until this, x and y were floored at zero, so
   a picture could be dragged down and right but never up or left: the lane's own
   corner was a wall, and "put this render above its run, or beside the lane
   above it" was simply not expressible. */
test('a picture goes above and left of its lane, not just below and right', () => {
  assert.deepEqual(clampImageBox({ x: -420, y: -260, w: 320, h: 320 }),
    { x: -420, y: -260, w: 320, h: 320 });
});

test('the reach is a rail on both sides, so one corrupt row cannot collapse Fit', () => {
  const far = clampImageBox({ x: 1e9, y: -1e9, w: 320, h: 320 });
  assert.deepEqual([far.x, far.y], [IMG_REACH, -IMG_REACH]);
  // And it is genuinely far: nothing placed by hand ever reaches it.
  assert.ok(IMG_REACH > 50000);
});

test('nonsense geometry degrades to a default instead of an unreachable node', () => {
  const box = clampImageBox({ x: NaN, y: Infinity, w: null, h: 'big' });
  assert.equal(Number.isFinite(box.x) && Number.isFinite(box.y), true);
  assert.equal(box.w >= IMG_MIN && box.h >= IMG_MIN, true);
});

test('rows with an unusable id or no image are dropped, not defaulted', () => {
  const m = toImageNodeMap([
    row(1), { image_id: null, x: 0, y: 0, w: 200, h: 200 },
    { image_id: 2, x: 0, y: 0, w: 200, h: 200 },   // no `image` -> nothing to draw
  ]);
  assert.deepEqual(Object.keys(m), ['1']);
});

test('the lane grows to hold its pinned images, so Fit cannot crop them', () => {
  const nodes = visibleImageNodes(toImageNodeMap([
    row(1, { x: 600, y: 100, w: 400, h: 300 }),
    row(2, { x: 10, y: 800, w: 200, h: 200 })]));
  assert.deepEqual(imageNodeExtent(nodes),
    { minX: 0, minY: 0, width: 1000, height: 1000 });
  assert.deepEqual(imageNodeExtent([]),
    { minX: 0, minY: 0, width: 0, height: 0 });
});

/* The other half of "Fit cannot crop them", and the one free placement created:
   a picture parked ABOVE its lane reaches past the lane's origin, and a board
   measured only from that origin downwards would frame itself with the picture
   floating off the top of it. */
test('a picture parked above or left of its lane is reported as overhang', () => {
  const nodes = visibleImageNodes(toImageNodeMap([
    row(1, { x: -300, y: -180, w: 320, h: 320 }),
    row(2, { x: 500, y: 40, w: 200, h: 200 })]));
  assert.deepEqual(imageNodeExtent(nodes),
    { minX: -300, minY: -180, width: 700, height: 240 });
});

test('overhang is never positive — the lane always contains its own origin', () => {
  const nodes = visibleImageNodes(toImageNodeMap([
    row(1, { x: 400, y: 400, w: 200, h: 200 })]));
  const ext = imageNodeExtent(nodes);
  assert.deepEqual([ext.minX, ext.minY], [0, 0]);
});

// ---- the link back to the checkpoint that made the image ------------------

const graph = {
  nodes: [{
    node: { record_id: 7 }, x: 0, y: 0, cellH: 120,
    checkpoints: [{ step: 2500, x: 40, y: 90, w: 60, h: 20 },
      { step: 3500, x: 110, y: 90, w: 60, h: 20 }],
  }],
};

test('a pinned image is joined to the exact checkpoint that produced it', () => {
  const nodes = visibleImageNodes(toImageNodeMap([row(42, { x: 400, y: 60, w: 300, h: 300 })]));
  const edges = imageNodeEdges(nodes, graph);
  assert.equal(edges.length, 1);
  const e = edges[0];
  // Leaves the RIGHT edge of the step-2500 pill, lands on the LEFT edge of the
  // node, mid-height — the same parent→child reading as every lineage edge.
  assert.deepEqual([e.x1, e.y1], [100, 100]);
  assert.deepEqual([e.x2, e.y2], [400, 210]);
  assert.equal(typeof e.d, 'string');
  assert.equal(e.onSpine, false);
  assert.equal(e.superseded, false);
  assert.notEqual(e.parentId, e.childId, 'edge keys must be unique per end');
});

test('an image whose checkpoint is no longer on the board draws no edge', () => {
  // The run was deleted from the board, or its dataset was unticked.
  const nodes = visibleImageNodes(toImageNodeMap([
    row(42, { image: { id: 42, url: '/a.png', record_id: 999, step: 2500 } })]));
  assert.deepEqual(imageNodeEdges(nodes, graph), []);
  // …and the node itself still renders. A missing edge is not a missing image.
  assert.equal(nodes.length, 1);
});

test('an image with no step at all is joined to nothing rather than guessed', () => {
  const nodes = visibleImageNodes(toImageNodeMap([
    row(42, { image: { id: 42, url: '/a.png', record_id: 7, step: null } })]));
  assert.deepEqual(imageNodeEdges(nodes, graph), []);
});

// ---- first placement ------------------------------------------------------

test('a new pin lands beside its source card, and never on top of another pin', () => {
  const first = defaultImageSpot(graph, 7, 2500, []);
  assert.equal(first.x > 0, true, 'to the right of the card, not on it');
  const second = defaultImageSpot(graph, 7, 2500, [{ ...first }]);
  assert.equal(second.y >= first.y + first.h, true, 'stacked below the one already there');
  assert.equal(second.x, first.x);
});

test('a pin whose source card is gone still gets a spot on the board', () => {
  const spot = defaultImageSpot(graph, 404, 2500, []);
  assert.equal(Number.isFinite(spot.x) && Number.isFinite(spot.y), true);
  assert.equal(spot.w >= IMG_MIN, true);
});

// ---- keyboard -------------------------------------------------------------

test('arrow keys move a node and +/- resize it, so a mouse is not required', () => {
  const n = { x: 100, y: 100, w: 300, h: 300 };
  assert.deepEqual(nudgeImageNode(n, 'ArrowRight', false),
    { x: 116, y: 100, w: 300, h: 300 });
  assert.deepEqual(nudgeImageNode(n, 'ArrowUp', true),
    { x: 100, y: 36, w: 300, h: 300 });
  assert.equal(nudgeImageNode(n, '+', false).w, 332);
  assert.equal(nudgeImageNode(n, '-', false).h, 268);
  assert.equal(nudgeImageNode(n, 'a', false), null, 'an unrelated key is not swallowed');
});

/* The keyboard reaches everywhere the mouse now does. Anyone who does not use a
   pointer would otherwise have half the board: a picture could be nudged down
   and right for ever and never back up past its lane's corner. */
test('the keyboard can leave the lane too, and still cannot beat the size clamp', () => {
  assert.deepEqual(nudgeImageNode({ x: 0, y: 0, w: 300, h: 300 }, 'ArrowLeft', true),
    { x: -64, y: 0, w: 300, h: 300 });
  assert.deepEqual(nudgeImageNode({ x: 0, y: 0, w: 300, h: 300 }, 'ArrowUp', false),
    { x: 0, y: -16, w: 300, h: 300 });
  assert.equal(nudgeImageNode({ x: 0, y: 0, w: IMG_MAX, h: IMG_MAX }, '+', true).w, IMG_MAX);
  assert.equal(nudgeImageNode({ x: 0, y: 0, w: IMG_MIN, h: IMG_MIN }, '-', true).h, IMG_MIN);
});

/* The two placement primitives ✦ Tidy up shares with a hand-dropped pin, so a
   strip and a lone picture cannot disagree about where "beside its run" is. */
test('spotBesideCard answers to the right of the card, at the card top', () => {
  assert.deepEqual(spotBesideCard(graph, 7), { x: 264 + 48, y: 0 });
  const orphan = spotBesideCard(graph, 404);
  assert.ok(orphan.x >= 264, 'a pin whose card is gone parks past the lane, not on it');
});

test('slideBelow drops past what is taken and carries the real footprint', () => {
  const strip = { x: 0, y: 0, w: 1600, h: 400 };
  assert.deepEqual(slideBelow(strip, []), strip);
  const moved = slideBelow(strip, [{ x: 900, y: 0, w: 200, h: 200 }]);
  assert.ok(moved.y >= 200, 'it went below the blocker it really overlapped');
  assert.equal(moved.w, 1600, 'a strip is not shrunk to one picture on the way');
});
