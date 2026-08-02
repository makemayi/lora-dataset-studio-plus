import test from 'node:test';
import assert from 'node:assert/strict';
import {
  PIN_BATCH_MAX, batchTileSize, boardObstacles, pinBatchAnnouncement,
  groupPinnedBatchBySource, groupPinnedBatchTogether, pinBatchLabel, pinBatchPending,
  pinBatchPendingAcrossLanes, placeImageBatch,
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

const placed = (id, recordId, step) => ({
  imageId: id, x: id * 10, y: 500, w: 320, h: 320, image: img(id, recordId, step),
});

test('successive pins from the same checkpoint automatically form one strip', () => {
  const first = { imageId: 1, x: 10, y: 10, w: 320, h: 320, visible: true,
    groupId: null, groupPos: null, image: img(1, 106, 2500) };
  const result = groupPinnedBatchBySource({ nodes: [first], placed: [placed(2, 106, 2500)] });
  assert.equal(result.rows[0].groupId, result.rows[1].groupId);
  assert.deepEqual(result.rows.map((r) => r.groupPos), [0, 1]);
});

test('a later generation joins the homogeneous group for that checkpoint', () => {
  const nodes = [1, 2].map((id, pos) => ({ imageId: id, x: id * 10, y: 10,
    w: 320, h: 320, visible: true, groupId: 'g1', groupPos: pos,
    image: img(id, 106, 2500) }));
  const result = groupPinnedBatchBySource({ nodes, placed: [placed(3, 106, 2500)] });
  assert.ok(result.rows.every((r) => r.groupId === 'g1'));
  assert.deepEqual(result.rows.map((r) => r.groupPos), [0, 1, 2]);
});

test('different checkpoints and unknown sources never auto-group', () => {
  const result = groupPinnedBatchBySource({ placed: [
    placed(1, 106, 2500), placed(2, 106, 3000),
    { ...placed(3, null, null), image: { id: 3, url: '/img/3.png' } },
  ] });
  assert.ok(result.rows.every((r) => r.groupId == null));
});

test('Pin all concatenates one generated lot even when checkpoints differ', () => {
  const result = groupPinnedBatchTogether({ placed: [
    placed(1, 82, 500), placed(2, 82, 1000), placed(3, 86, 6000),
  ] });
  assert.ok(result.rows[0].groupId);
  assert.ok(result.rows.every((r) => r.groupId === result.rows[0].groupId));
  assert.deepEqual(result.rows.map((r) => r.groupPos), [0, 1, 2]);
  assert.ok(result.undoRows.every((r) => r.visible === false && r.groupId == null));
});

test('separate Pin all gestures get separate groups', () => {
  const first = groupPinnedBatchTogether({ placed: [placed(1, 82, 500), placed(2, 82, 1000)] });
  const second = groupPinnedBatchTogether({
    nodes: first.rows,
    placed: [placed(3, 82, 1500), placed(4, 82, 2000)],
  });
  assert.notEqual(first.rows[0].groupId, second.rows[0].groupId);
});

test('a manual mixed-checkpoint group is never used as an automatic target', () => {
  const nodes = [
    { imageId: 1, x: 10, y: 10, w: 320, h: 320, visible: true,
      groupId: 'mixed', groupPos: 0, image: img(1, 106, 2500) },
    { imageId: 2, x: 20, y: 10, w: 320, h: 320, visible: true,
      groupId: 'mixed', groupPos: 1, image: img(2, 107, 2500) },
  ];
  const result = groupPinnedBatchBySource({ nodes,
    placed: [placed(3, 106, 2500), placed(4, 106, 2500)] });
  assert.equal(result.rows.find((r) => r.imageId === 1), undefined);
  assert.notEqual(result.rows.find((r) => r.imageId === 3).groupId, 'mixed');
});

test('undo restores existing membership and closes newly pinned images', () => {
  const nodes = [1, 2].map((id, pos) => ({ imageId: id, x: id * 10, y: 10,
    w: 320, h: 320, visible: true, groupId: 'g1', groupPos: pos,
    image: img(id, 106, 2500) }));
  const result = groupPinnedBatchBySource({ nodes, placed: [placed(3, 106, 2500)] });
  assert.deepEqual(result.undoRows.map((r) => [r.imageId, r.visible, r.groupId, r.groupPos]),
    [[1, true, 'g1', 0], [2, true, 'g1', 1], [3, false, null, null]]);
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

/* ── Two runs are two lots, and a strip reads by epoch ──────────────────────

   The two things a board built to COMPARE checkpoints has to get right, and
   the two it was getting wrong:

     • a second generation fired at the same checkpoint had its pictures
       APPENDED to the first generation's strip, so two runs read as one lot.
       The only thing a strip could be keyed on was "same checkpoint", which is
       true of both runs. `run_id` — already in the database, now published by
       the gallery — is the thing that actually differs;
     • the order inside a strip came from `${record_id}:${step}` compared as
       TEXT, which puts step 1000 before step 500. An epoch axis sorted
       alphabetically is not an epoch axis. */

const runImg = (id, recordId, step, runId) => ({
  id, dataset_id: 3, record_id: recordId, step, run_id: runId,
  url: `/img/${id}.png`,
});
const runPlaced = (id, recordId, step, runId) => ({
  imageId: id, x: id * 10, y: 500, w: 320, h: 320,
  image: runImg(id, recordId, step, runId),
});
// Pin one picture the way the gallery does, and fold the result back into the
// board — the sequence that produced the reported bug.
const pinOneByOne = (laneMap, image) => {
  const grouped = groupPinnedBatchBySource({
    nodes: Object.values(laneMap),
    placed: [{ imageId: image.id, x: 0, y: 0, w: 320, h: 320, image }],
  });
  for (const r of grouped.rows) laneMap[r.imageId] = r;
  return laneMap;
};

test('two generations of the SAME checkpoint stay two strips', () => {
  const lane = {};
  // Run A, pinned one by one from the checkpoint gallery.
  pinOneByOne(lane, runImg(11, 7, 500, 'runA'));
  pinOneByOne(lane, runImg(12, 7, 500, 'runA'));
  // Run B — a LATER generation, same checkpoint, same dataset.
  pinOneByOne(lane, runImg(21, 7, 500, 'runB'));
  pinOneByOne(lane, runImg(22, 7, 500, 'runB'));

  const groupOf = (id) => lane[id].groupId;
  assert.ok(groupOf(11), 'run A formed a strip');
  assert.equal(groupOf(11), groupOf(12));
  assert.ok(groupOf(21), 'run B formed its own strip');
  assert.equal(groupOf(21), groupOf(22));
  assert.notEqual(groupOf(11), groupOf(21),
    'run B must NOT be concatenated onto run A');
});

test('images with no run id keep the old per-checkpoint behaviour', () => {
  // A board that predates the published run id must draw what it always drew:
  // successive pins of one checkpoint still gather into one strip.
  const lane = {};
  pinOneByOne(lane, { id: 31, dataset_id: 3, record_id: 7, step: 500 });
  pinOneByOne(lane, { id: 32, dataset_id: 3, record_id: 7, step: 500 });
  assert.ok(lane[31].groupId);
  assert.equal(lane[31].groupId, lane[32].groupId);
});

test('a strip reads in TRAINING order, not alphabetical order', () => {
  // 1000 < 500 as text; as epochs it is the other way round.
  const result = groupPinnedBatchTogether({ placed: [
    runPlaced(1, 82, 1000, 'r1'), runPlaced(2, 82, 500, 'r1'),
    runPlaced(3, 82, 2000, 'r1'), runPlaced(4, 82, 1500, 'r1'),
  ] });
  const strip = [...result.rows].sort((a, b) => a.groupPos - b.groupPos);
  assert.deepEqual(strip.map((r) => r.image.step), [500, 1000, 1500, 2000]);
  assert.deepEqual(strip.map((r) => r.groupPos), [0, 1, 2, 3]);
});

test('a picture pinned later at an EARLIER epoch lands at the head of the strip', () => {
  const lane = {};
  pinOneByOne(lane, runImg(41, 7, 2000, 'runC'));
  pinOneByOne(lane, runImg(42, 7, 3000, 'runC'));
  pinOneByOne(lane, runImg(43, 7, 1000, 'runC'));
  const strip = Object.values(lane).sort((a, b) => a.groupPos - b.groupPos);
  assert.deepEqual(strip.map((r) => r.image.step), [1000, 2000, 3000]);
});

test('a lot that carries two runs is SPLIT, never concatenated', () => {
  const result = groupPinnedBatchTogether({ placed: [
    runPlaced(1, 82, 500, 'r1'), runPlaced(2, 82, 1000, 'r1'),
    runPlaced(3, 82, 500, 'r2'), runPlaced(4, 82, 1000, 'r2'),
  ] });
  const by = Object.fromEntries(result.rows.map((r) => [r.imageId, r.groupId]));
  assert.equal(by[1], by[2]);
  assert.equal(by[3], by[4]);
  assert.notEqual(by[1], by[3]);
});

test('the band lays its columns out in epoch order', () => {
  // One card, so every checkpoint anchors at the same x and the tie-break is
  // the whole answer. Left to right must be 500, 1000, 1500 — it was
  // 1000, 1500, 500 while the tie was broken by the alphabet.
  const graph = { nodes: [{ x: 0, y: 0, cellH: 100, node: { record_id: 9 } }] };
  const { placed: out } = placeImageBatch({ graph, existing: [], images: [
    img(51, 9, 1000), img(52, 9, 500), img(53, 9, 1500),
  ] });
  const xOf = (id) => out.find((p) => p.imageId === id).x;
  assert.ok(xOf(52) < xOf(51), 'step 500 sits left of step 1000');
  assert.ok(xOf(51) < xOf(53), 'step 1000 sits left of step 1500');
});

test('an over-cap lot keeps the EARLY epochs and says what it dropped', () => {
  const graph = { nodes: [{ x: 0, y: 0, cellH: 100, node: { record_id: 9 } }] };
  const images = Array.from({ length: PIN_BATCH_MAX + 3 },
    (_, i) => img(600 + i, 9, (i + 1) * 100));
  const res = placeImageBatch({ graph, existing: [], images });
  assert.equal(res.placed.length, PIN_BATCH_MAX);
  assert.equal(res.skipped.length, 3);
  const droppedSteps = res.skipped.map((s) => s.image.step);
  const keptSteps = res.placed.map((p) => p.image.step);
  assert.ok(Math.min(...droppedSteps) > Math.max(...keptSteps),
    'the tail of the training is what gets refused, not an arbitrary slice');
});
