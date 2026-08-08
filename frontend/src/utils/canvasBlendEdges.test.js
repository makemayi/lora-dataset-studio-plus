import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import {
  blendEdgesFor, blendSourcesNote, pillWorldBox, stackMembersOf,
} from './canvasBlendEdges.js';

/* 🧬 Generation provenance: a blended picture descends from N pills at once,
   routinely across lanes. What is pinned here is the arithmetic and — mostly —
   the refusals: what happens when a source cannot be placed. */

const lane = (datasetId, x, graphY, cards) => ({ datasetId, x, graphY, graph: { nodes: cards } });
const card = (recordId, pills) => ({ node: { record_id: recordId }, checkpoints: pills });
const pill = (step, x, y) => ({ step, x, y, w: 40, h: 10 });

const LANES = [
  lane(1, 0, 100, [card(11, [pill(2000, 10, 20)])]),
  lane(4, 0, 600, [card(22, [pill(1000, 10, 30)])]),
];

const blendImage = (members) => ({
  record_id: 11, step: 2000,
  extra_loras: JSON.stringify(members.map((m) => ({ ...m, combined: true, strength: 0.6 }))),
});

const node = (imageId, datasetId, image) => ({ imageId, datasetId, x: 300, y: 40, w: 100, h: 80, image });

test('a blend draws one edge per stacked source, ACROSS lanes', () => {
  // The head keeps the ordinary image→pill edge it already had; this module
  // only adds the other parents, or the pair would carry two connectors.
  const { edges } = blendEdgesFor(
    [node(7, 1, blendImage([{ dataset_id: 4, record_id: 22, step: 1000 }]))], LANES);
  assert.equal(edges.length, 1);
  const [e] = edges;
  assert.equal(e.parentId, 'ck:4:22:1000');
  assert.equal(e.childId, 'img:7');
  assert.equal(e.blend, true);
  // WORLD coordinates: the source pill's lane sits at graphY 600, the picture's
  // at 100. An edge computed lane-locally would join two unrelated points.
  assert.equal(e.x1, 50);              // pill.x 10 + pill.w 40
  assert.equal(e.y1, 600 + 30 + 5);    // lane.graphY + pill.y + h/2
  assert.equal(e.x2, 300);
  assert.equal(e.y2, 100 + 40 + 40);   // lane.graphY + node.y + h/2
  assert.ok(e.d && typeof e.d === 'string');
  assert.equal(e.onSpine, false);
  assert.equal(e.superseded, false);
});

test('a picture that is not a blend adds nothing at all', () => {
  const plain = { record_id: 11, step: 2000, extra_loras: null };
  const { edges, unresolved } = blendEdgesFor([node(7, 1, plain)], LANES);
  assert.deepEqual(edges, []);
  assert.equal(unresolved.size, 0, 'only blends are ever counted');
  // An always-on LoRA is not a stacked source either.
  const alwaysOn = { extra_loras: JSON.stringify([{ filename: 'x', strength: 1 }]) };
  assert.deepEqual(stackMembersOf(alwaysOn), []);
});

test('a source that cannot be PLACED draws no edge, and is counted as missing', () => {
  // Run deleted, dataset unticked in the filter, or a blend older than the day
  // members started recording their origin — all the same answer: no edge.
  const cases = [
    { dataset_id: 4, record_id: 999, step: 1000 },   // no such card
    { dataset_id: 4, record_id: 22, step: 4242 },    // card is there, step is not
    { dataset_id: 77, record_id: 22, step: 1000 },   // dataset not on the board
    { dataset_id: 4, record_id: null, step: null },  // legacy: origin never recorded
  ];
  for (const m of cases) {
    const { edges, unresolved } = blendEdgesFor([node(7, 1, blendImage([m]))], LANES);
    assert.deepEqual(edges, [], `${JSON.stringify(m)} must draw nothing`);
    assert.deepEqual(unresolved.get(7), { placed: 1, total: 2 }, 'the head still counts');
  }
});

test('the badge says what is missing, in sources and never without the total', () => {
  assert.equal(blendSourcesNote({ placed: 1, total: 2 }), '1 of 2 sources is not on the board');
  assert.equal(blendSourcesNote({ placed: 1, total: 3 }), '2 of 3 sources are not on the board');
  // Nothing missing = nothing said. A badge that always speaks is noise.
  assert.equal(blendSourcesNote({ placed: 2, total: 2 }), null);
  assert.equal(blendSourcesNote(null), null);
  assert.equal(blendSourcesNote({ placed: 0, total: 0 }), null);
});

test('a mixed stack places what it can and reports the rest', () => {
  const { edges, unresolved } = blendEdgesFor([node(7, 1, blendImage([
    { dataset_id: 4, record_id: 22, step: 1000 },     // placeable
    { dataset_id: 4, record_id: 22, step: 9999 },     // not
  ]))], LANES);
  assert.equal(edges.length, 1);
  assert.deepEqual(unresolved.get(7), { placed: 2, total: 3 });
  assert.equal(blendSourcesNote(unresolved.get(7)), '1 of 3 sources is not on the board');
});

test('a corrupt or truncated extra_loras degrades to "no provenance", never a throw', () => {
  for (const raw of ['{not json', '[]', 'null', '', undefined, 42]) {
    assert.deepEqual(stackMembersOf({ extra_loras: raw }), [], `${raw} must be survivable`);
  }
  assert.deepEqual(stackMembersOf(null), []);
  // …and an already-parsed array is accepted too, so a caller that decoded it
  // once does not have to re-encode it to be understood.
  assert.equal(stackMembersOf({ extra_loras: [{ combined: true, dataset_id: 4 }] }).length, 1);
});

/* --- how the layer is DRAWN (source contract: node --test cannot parse JSX) - */

const src = (rel) => readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');

test('the provenance layer is under everything and never takes a click', () => {
  const canvas = src('components/canvas/LineageCanvas.jsx');
  const layer = canvas.indexOf('data-testid="canvas-provenance-layer"');
  const lanes = canvas.indexOf('{world.lanes.map((lane) => (');
  assert.ok(layer > 0, 'the layer must exist');
  assert.ok(layer < lanes, 'it must be the FIRST child, i.e. under the lanes');
  // THE rule, learnt the hard way on the group bar: an edge is context, not
  // content. A layer spanning the whole board that could take a pointer would
  // make everything under it unusable.
  assert.match(canvas.slice(Math.max(0, layer - 400), layer + 400), /pointer-events-none/);
});

test('a provenance edge is violet, and stays violet whatever the hover does', () => {
  const edges = src('components/dataset/lineageEdges.jsx');
  assert.match(edges, /id="lds-edge-blend"/);
  // Chosen BEFORE the spine/superseded branches, so hovering a run cannot
  // repaint a generation edge as training lineage — that would claim a descent
  // which did not happen.
  assert.match(edges, /e\.blend \? 'lds-edge-blend'/);
});

test('the badge is rendered from the note, and only when there IS one', () => {
  const node = src('components/canvas/CanvasImageNode.jsx');
  assert.match(node, /blendNote && \(/);
  assert.match(node, /data-testid="canvas-blend-note"/);
  // It must never eat a click either — it sits over the picture it describes.
  assert.match(node, /pointer-events-none absolute bottom-0 left-0/);
});

test('pillWorldBox is the single place lane offsets are applied', () => {
  assert.deepEqual(pillWorldBox(LANES, 1, 11, 2000), { x: 10, y: 120, w: 40, h: 10 });
  assert.equal(pillWorldBox(LANES, 1, 11, null), null);
  assert.equal(pillWorldBox(LANES, 1, null, 2000), null);
  assert.equal(pillWorldBox([], 1, 11, 2000), null);
  // Dataset ids arrive as strings from DOM datasets in places; both must match.
  assert.ok(pillWorldBox(LANES, '1', 11, 2000));
});
