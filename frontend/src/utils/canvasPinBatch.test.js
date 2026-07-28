import test from 'node:test';
import assert from 'node:assert/strict';
import {
  PIN_BATCH_MAX, batchTileSize, boardObstacles, pinBatchAnnouncement,
  pinBatchLabel, pinBatchPending, pinBatchPendingAcrossLanes, placeImageBatch,
} from './canvasPinBatch.js';
import { CARD_W } from './lineageGraph.js';

/* 📌 Pin all — the one assertion that decides whether this feature is worth
   having: NOTHING may end up on top of anything else.

   Everything here works in BOARD units, never in screen pixels. The user's own
   board was at 24 % zoom when the feature was asked for, and a rule expressed in
   pixels would have passed there and failed at 100 %. */

// ---- a realistic board ----------------------------------------------------
// Four runs, two generations, several checkpoint pills each — the shape the
// automatic tree produces, with the fields placeImageBatch actually reads.

const pill = (step, x, y) => ({ step, x, y, w: 60, h: 20 });

const card = (recordId, x, y, steps) => ({
  node: { record_id: recordId },
  x,
  y,
  cellH: 64 + 8 + 20,
  checkpoints: steps.map((s, i) => pill(s, x + i * 66, y + 72)),
});

const GRAPH = {
  nodes: [
    card(106, 22, 22, [2500, 3000]),
    card(107, 22, 160, [2000]),
    card(114, 348, 22, [1500, 2000]),
    card(117, 348, 160, [4000]),
  ],
  edges: [],
  width: 700,
  height: 300,
};

const img = (id, recordId, step) => ({
  id, dataset_id: 3, record_id: recordId, step, url: `/img/${id}.png`,
});

// The lot from the report: five images, four different runs.
const LOT = [
  img(501, 106, 2500),
  img(502, 107, 2000),
  img(503, 114, 2000),
  img(504, 114, 2000),
  img(505, 117, 4000),
];

const rectsOverlap = (a, b) => (
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
);

/** Every pair of rectangles on the board, checked. Not three chosen cases: the
 *  full cross-product, so a placement that happens to be lucky cannot pass. */
function assertNoOverlap(rects) {
  for (let i = 0; i < rects.length; i += 1) {
    for (let j = i + 1; j < rects.length; j += 1) {
      assert.ok(!rectsOverlap(rects[i], rects[j]),
        `${JSON.stringify(rects[i])} overlaps ${JSON.stringify(rects[j])}`);
    }
  }
}

// ---- THE assertion --------------------------------------------------------

test('THE assertion: a lot placed on a loaded board overlaps NOTHING', () => {
  // A board that is already busy: five images pinned by hand, all over it.
  const existing = [
    { imageId: 900, x: 700, y: 20, w: 320, h: 320, visible: true },
    { imageId: 901, x: 700, y: 400, w: 320, h: 320, visible: true },
    { imageId: 902, x: 60, y: 420, w: 260, h: 260, visible: true },
    { imageId: 903, x: 1100, y: 120, w: 400, h: 400, visible: true },
    { imageId: 904, x: 380, y: 500, w: 180, h: 180, visible: true },
  ];
  const res = placeImageBatch({ graph: GRAPH, existing, images: LOT });

  assert.equal(res.placed.length, 5, 'all five got a spot');
  assert.equal(res.skipped.length, 0);

  // Cards (card + its pill block), pinned images, and the new tiles: one flat
  // list, every pair checked.
  assertNoOverlap([
    ...boardObstacles(GRAPH, existing),
    ...res.placed.map((p) => ({ x: p.x, y: p.y, w: p.w, h: p.h })),
  ]);
});

test('no overlap holds for a lot of thirty from a single checkpoint', () => {
  const many = Array.from({ length: 30 }, (_, i) => img(1000 + i, 114, 2000));
  const res = placeImageBatch({ graph: GRAPH, existing: [], images: many });
  assert.equal(res.placed.length, 30);
  assertNoOverlap([
    ...boardObstacles(GRAPH, []),
    ...res.placed.map((p) => ({ x: p.x, y: p.y, w: p.w, h: p.h })),
  ]);
});

test('no overlap holds when the batch is bigger than the board is wide', () => {
  // Every image from a different run, including runs that are NOT on the board.
  const wide = Array.from({ length: 18 }, (_, i) => img(2000 + i, 106 + i, 2500));
  const existing = [{ imageId: 1, x: 0, y: 900, w: 1400, h: 300, visible: true }];
  const res = placeImageBatch({ graph: GRAPH, existing, images: wide });
  assertNoOverlap([
    ...boardObstacles(GRAPH, existing),
    ...res.placed.map((p) => ({ x: p.x, y: p.y, w: p.w, h: p.h })),
  ]);
});

// ---- near its own source --------------------------------------------------

test('each image lands under its OWN checkpoint, not all in one heap', () => {
  const res = placeImageBatch({ graph: GRAPH, existing: [], images: LOT });
  const by = new Map(res.placed.map((p) => [p.imageId, p]));

  // Two images from the same checkpoint share a column…
  assert.equal(by.get(503).x, by.get(504).x, 'same source ⇒ same column');
  assert.notEqual(by.get(503).y, by.get(504).y, 'stacked, not superimposed');

  // …and four different sources produce four different columns.
  const columns = new Set(res.placed.map((p) => p.x));
  assert.equal(columns.size, 4, 'one column per source checkpoint');

  // A source on the left of the tree keeps its images on the left.
  assert.ok(by.get(501).x < by.get(503).x,
    'the run at depth 0 is left of the run at depth 1');
});

test('an image whose run is not on the board still gets a spot', () => {
  const res = placeImageBatch({ graph: GRAPH, existing: [], images: [img(700, 999, 1)] });
  assert.equal(res.placed.length, 1);
  assert.ok(res.placed[0].x >= 0 && res.placed[0].y >= 0);
});

// ---- determinism ----------------------------------------------------------

test('the same board and the same lot give exactly the same positions', () => {
  const existing = [{ imageId: 900, x: 700, y: 20, w: 320, h: 320, visible: true }];
  const a = placeImageBatch({ graph: GRAPH, existing, images: LOT });
  // Same lot, listed in another order: the answer must not depend on it.
  const b = placeImageBatch({ graph: GRAPH, existing, images: [...LOT].reverse() });
  const key = (r) => r.placed.map((p) => `${p.imageId}@${p.x},${p.y},${p.w}`).sort().join('|');
  assert.equal(key(a), key(b));
});

// ---- already pinned -------------------------------------------------------

test('an image already ON the board is neither moved nor duplicated', () => {
  const laneMap = {
    503: { imageId: 503, x: 999, y: 999, w: 320, h: 320, visible: true },
  };
  const pending = pinBatchPending(LOT, laneMap);
  assert.deepEqual(pending.pending.map((i) => i.id), [501, 502, 504, 505]);
  assert.equal(pending.already, 1);

  const res = placeImageBatch({ graph: GRAPH, existing: [laneMap[503]], images: pending.pending });
  assert.ok(!res.placed.some((p) => p.imageId === 503), 'not re-placed');
  // And its own rectangle is still where the user left it.
  assertNoOverlap([
    { x: 999, y: 999, w: 320, h: 320 },
    ...res.placed.map((p) => ({ x: p.x, y: p.y, w: p.w, h: p.h })),
  ]);
});

test('a CLOSED image comes back at its remembered spot when that spot is free', () => {
  const laneMap = {
    501: { imageId: 501, x: 1600, y: 40, w: 300, h: 300, visible: false },
  };
  const pending = pinBatchPending(LOT, laneMap);
  assert.equal(pending.pending.length, 5, 'a closed image is part of the lot again');
  const res = placeImageBatch({
    graph: GRAPH, existing: [], images: pending.pending, remembered: laneMap });
  const back = res.placed.find((p) => p.imageId === 501);
  assert.deepEqual({ x: back.x, y: back.y, w: back.w, h: back.h },
    { x: 1600, y: 40, w: 300, h: 300 });
});

test('a remembered spot that would collide is overruled — no overlap wins', () => {
  const laneMap = {
    501: { imageId: 501, x: 22, y: 22, w: 300, h: 300, visible: false },  // on top of card 106
  };
  const res = placeImageBatch({
    graph: GRAPH, existing: [], images: [img(501, 106, 2500)], remembered: laneMap });
  const back = res.placed[0];
  assert.notDeepEqual({ x: back.x, y: back.y }, { x: 22, y: 22 });
  assertNoOverlap([
    ...boardObstacles(GRAPH, []),
    { x: back.x, y: back.y, w: back.w, h: back.h },
  ]);
});

// ---- when there is no more room -------------------------------------------

test('a lot over the cap places the cap and SAYS what it refused', () => {
  const huge = Array.from({ length: PIN_BATCH_MAX + 7 }, (_, i) => img(3000 + i, 106, 2500));
  const res = placeImageBatch({ graph: GRAPH, existing: [], images: huge });
  assert.equal(res.placed.length, PIN_BATCH_MAX);
  assert.equal(res.skipped.length, 7);
  assert.ok(res.skipped.every((s) => s.reason === 'over-cap'));

  const said = pinBatchAnnouncement(res);
  assert.match(said, new RegExp(`${PIN_BATCH_MAX} images pinned`));
  assert.match(said, /7 left out/);
  assert.match(said, /gallery/i, 'the refusal names the way to get the rest');
});

test('nothing pinned is announced as nothing, never as a success', () => {
  const res = placeImageBatch({ graph: GRAPH, existing: [], images: [] });
  assert.deepEqual(res.placed, []);
  assert.match(pinBatchAnnouncement(res), /Nothing/i);
});

// ---- the tile size --------------------------------------------------------

test('a big lot lands as a contact sheet, a small one at full size', () => {
  assert.equal(batchTileSize(1), 320);
  assert.ok(batchTileSize(30) < batchTileSize(3), 'thirty images are thumbnails');
  assert.ok(batchTileSize(40) >= 96, 'never below the grabbable floor');
  // The chosen size is what the placed tiles actually carry.
  const many = Array.from({ length: 20 }, (_, i) => img(4000 + i, 106, 2500));
  const res = placeImageBatch({ graph: GRAPH, existing: [], images: many });
  assert.equal(res.size, batchTileSize(20));
  assert.ok(res.placed.every((p) => p.w === res.size && p.h === res.size));
});

// ---- the button's own words -----------------------------------------------

test('the button says how many it will put down', () => {
  assert.equal(pinBatchLabel(5), '📌 Pin all 5 to the board');
  assert.equal(pinBatchLabel(1), '📌 Pin this image to the board');
  assert.equal(pinBatchLabel(0), '');
});

// ---- the obstacles are the whole board ------------------------------------

// ---- the button's count, across the whole board ---------------------------

test('the count is what is NOT on the board yet, lane by lane', () => {
  const candidates = [
    { id: 501, datasetId: 3 }, { id: 502, datasetId: 3 },
    { id: 601, datasetId: 8 },
  ];
  const lanes = {
    3: { 501: { imageId: 501, visible: true }, 502: { imageId: 502, visible: false } },
    8: {},
  };
  const { pending, already } = pinBatchPendingAcrossLanes(candidates, lanes);
  assert.deepEqual(pending.map((p) => p.id), [502, 601], 'a CLOSED one still counts');
  assert.equal(already, 1);
  assert.equal(pinBatchLabel(pending.length), '📌 Pin all 2 to the board');
});

test('a lot entirely on the board already offers nothing', () => {
  const lanes = { 3: { 501: { visible: true } } };
  const { pending } = pinBatchPendingAcrossLanes([{ id: 501, datasetId: 3 }], lanes);
  assert.equal(pending.length, 0);
  assert.equal(pinBatchLabel(pending.length), '', 'no label ⇒ no button');
});

test('a run card and its pill block are ONE obstacle, pinned images are the rest', () => {
  const obstacles = boardObstacles(GRAPH, [
    { imageId: 900, x: 700, y: 20, w: 320, h: 320, visible: true },
    { imageId: 901, x: 0, y: 0, w: 320, h: 320, visible: false },
  ]);
  assert.equal(obstacles.length, 5, '4 cards + 1 VISIBLE image');
  const cardRect = obstacles.find((o) => o.x === 22 && o.y === 22);
  assert.equal(cardRect.w, CARD_W);
  assert.ok(cardRect.h >= 64 + 20, 'the pills under the card are inside the rect');
});
