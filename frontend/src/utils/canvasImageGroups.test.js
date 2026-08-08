import test from 'node:test';
import assert from 'node:assert/strict';
import {
  edgeAnchors, extractFromGroup, groupMembers, layoutImageNodes, mergeIntoGroup,
  mergeTargetAt, nextGroupId, shouldExtract,
} from './canvasImageGroups.js';

/* 🖼🖼 Grouped pinned images — the arithmetic.

   "Drop one image onto another and they become ONE node, side by side, with no
   border between them, no limit on how many; drag one out and it is its own
   node again."

   None of that can be tested through the component (`node --test` does not
   parse JSX), and all of it is the part that ages badly: which node a drop
   lands on, what order the strip ends up in, what a leaver gets back, what the
   ones that stay behind keep. So it all lives here. */

const img = (id) => ({ id, dataset_id: 1, record_id: 10, step: 100 * id,
  url: `/img/${id}.png` });

/** One pinned node as toImageNodeMap publishes it. */
const node = (id, box, group = null, pos = null) => ({
  imageId: id, x: box.x, y: box.y, w: box.w, h: box.h,
  visible: true, groupId: group, groupPos: pos, image: img(id),
});

const SQUARE = { x: 100, y: 100, w: 320, h: 320 };

// A three-member group already on the board: A anchors it.
const trio = () => ([
  node(1, { x: 100, y: 100, w: 320, h: 320 }, 'g1', 0),
  node(2, { x: 700, y: 100, w: 480, h: 320 }, 'g1', 1),   // 3:2, wider
  node(3, { x: 900, y: 500, w: 320, h: 640 }, 'g1', 2),   // 1:2, taller
]);

const byId = (list, id) => list.find((n) => n.imageId === id);
const groupOf = (layout) => layout.find((r) => r.kind === 'group');

// ---- 1. merging: two, then three, then four, then ten ---------------------

test('dropping an image onto another makes ONE group of two', () => {
  const nodes = [node(1, SQUARE), node(2, { x: 900, y: 400, w: 320, h: 320 })];
  const rows = mergeIntoGroup(nodes, 2, 1, 'after');
  const map = new Map(rows.map((r) => [r.imageId, r]));
  assert.equal(map.get(1).groupId, map.get(2).groupId, 'both carry one group id');
  assert.ok(map.get(1).groupId, 'and it is a real id, not null');
  assert.equal(map.get(1).groupPos, 0);
  assert.equal(map.get(2).groupPos, 1);
});

test('a third, a fourth and a tenth image join the same group — no limit', () => {
  let nodes = [node(1, SQUARE), node(2, SQUARE)];
  nodes = applyRows(nodes, mergeIntoGroup(nodes, 2, 1, 'after'));
  for (let id = 3; id <= 10; id += 1) {
    nodes = [...nodes, node(id, { x: 2000, y: 2000, w: 320, h: 320 })];
    nodes = applyRows(nodes, mergeIntoGroup(nodes, id, id - 1, 'after'));
  }
  const members = groupMembers(nodes, nodes[0].groupId);
  assert.equal(members.length, 10, 'ten images in one group');
  assert.deepEqual(members.map((m) => m.groupPos), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'positions stay dense and ordered, however many joined');
  assert.deepEqual(members.map((m) => m.imageId), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
});

test('dropping on the LEFT half of a member inserts before it', () => {
  const nodes = trio();
  const rows = mergeIntoGroup([...nodes, node(9, SQUARE)], 9, 2, 'before');
  const merged = applyRows([...nodes, node(9, SQUARE)], rows);
  assert.deepEqual(groupMembers(merged, 'g1').map((m) => m.imageId), [1, 9, 2, 3]);
});

test('a group id never collides with one already on the board', () => {
  const nodes = [...trio(), node(7, SQUARE), node(8, SQUARE)];
  // Image 1 already anchors 'g1'; a new group anchored on 1 would clash.
  const id = nextGroupId(nodes, 1);
  assert.notEqual(id, 'g1');
  assert.ok(id.startsWith('g1'), `expected a g1-derived id, got ${id}`);
});

test('joining a group while already in another leaves the first one behind', () => {
  const nodes = [...trio(), node(7, SQUARE), node(8, { x: 50, y: 900, w: 320, h: 320 })];
  const after = applyRows(nodes, mergeIntoGroup(nodes, 7, 8, 'after'));
  const after2 = applyRows(after, mergeIntoGroup(after, 3, 7, 'after'));
  assert.equal(groupMembers(after2, 'g1').length, 2, 'g1 is down to two');
  assert.equal(byId(after2, 3).groupId, byId(after2, 7).groupId);
});

// ---- 2. the strip: side by side, no border, one continuous band -----------

test('members are laid out edge to edge — no gap at all between two images', () => {
  const g = groupOf(layoutImageNodes(trio()));
  const [a, b, c] = g.members;
  assert.equal(a.x + a.w, b.x, 'the second starts exactly where the first ends');
  assert.equal(b.x + b.w, c.x, 'and the third exactly where the second ends');
  assert.equal(g.w, a.w + b.w + c.w, 'the group is exactly its members');
});

test('every member shares the anchor’s height and keeps its own aspect ratio', () => {
  const g = groupOf(layoutImageNodes(trio()));
  assert.equal(g.h, 320, 'the group is as tall as its anchor');
  for (const m of g.members) assert.equal(m.h, 320);
  assert.equal(g.members[0].w, 320, '1:1 stays square');
  assert.equal(g.members[1].w, 480, '3:2 stays 3:2');
  assert.equal(g.members[2].w, 160, '1:2 stays 1:2');
});

test('members use the generated image format instead of a letterboxed node box', () => {
  const portrait = { ...node(1, SQUARE, 'g1', 0), image: { ...img(1), aspect: '9:16' } };
  const landscape = { ...node(2, SQUARE, 'g1', 1), image: { ...img(2), aspect: '16:9' } };
  const g = groupOf(layoutImageNodes([portrait, landscape]));

  assert.deepEqual(g.members.map((member) => member.w), [180, 569]);
  assert.equal(g.members[0].x + g.members[0].w, g.members[1].x,
    'the real formats still meet edge to edge');
  assert.equal(g.w, 749);
});

test('missing or invalid image formats fall back to remembered node geometry', () => {
  const invalid = { ...node(1, { ...SQUARE, w: 480 }, 'g1', 0),
    image: { ...img(1), aspect: '9:0' } };
  const missing = node(2, { ...SQUARE, w: 160 }, 'g1', 1);
  const g = groupOf(layoutImageNodes([invalid, missing]));

  assert.deepEqual(g.members.map((member) => member.w), [480, 160]);
});

test('the group sits where its anchor sits — the others’ own x/y are ignored', () => {
  const g = groupOf(layoutImageNodes(trio()));
  assert.equal(g.x, 100);
  assert.equal(g.y, 100);
  assert.equal(g.members[0].x, 100);
});

test('ungrouped images are laid out untouched, one renderable each', () => {
  const layout = layoutImageNodes([node(1, SQUARE), node(2, { x: 9, y: 9, w: 100, h: 100 })]);
  assert.equal(layout.length, 2);
  assert.ok(layout.every((r) => r.kind === 'single'));
  assert.equal(layout[1].w, 100);
});

test('a group of ONE cannot exist in a layout — it draws as a plain image', () => {
  const layout = layoutImageNodes([node(1, SQUARE, 'g1', 0)]);
  assert.deepEqual(layout.map((r) => r.kind), ['single']);
});

// ---- 3. taking one out ---------------------------------------------------

test('dragging a member out drops the group from three members to two', () => {
  const nodes = trio();
  const after = applyRows(nodes, extractFromGroup(nodes, 2, { x: 1500, y: 900 }));
  assert.equal(byId(after, 2).groupId, null);
  assert.equal(byId(after, 2).groupPos, null);
  assert.deepEqual(groupMembers(after, 'g1').map((m) => m.imageId), [1, 3]);
  assert.deepEqual(groupMembers(after, 'g1').map((m) => m.groupPos), [0, 1],
    'the ones left close the gap');
});

test('the image taken out lands where it was dropped, at its OWN size again', () => {
  const nodes = trio();
  const after = applyRows(nodes, extractFromGroup(nodes, 3, { x: 1500, y: 900 }));
  const out = byId(after, 3);
  assert.equal(out.x, 1500);
  assert.equal(out.y, 900);
  // In the strip it rendered 160×320; on its own it is 320×640 again.
  assert.equal(out.w, 320);
  assert.equal(out.h, 640);
});

test('taking the ANCHOR out does not make the rest of the strip jump', () => {
  const nodes = trio();
  const before = groupOf(layoutImageNodes(nodes));
  const secondWas = before.members[1];
  const after = applyRows(nodes, extractFromGroup(nodes, 1, { x: 40, y: 40 }));
  const now = groupOf(layoutImageNodes(after));
  assert.equal(now.x, before.x, 'the strip stays put');
  assert.equal(now.y, before.y);
  assert.equal(now.h, before.h, 'and keeps its height');
  assert.equal(now.members[0].w, secondWas.w,
    'the new anchor draws exactly as wide as it did a moment ago');
});

test('a group that falls to ONE member dissolves back into a plain image', () => {
  const pair = [node(1, SQUARE, 'g1', 0),
    node(2, { x: 700, y: 100, w: 480, h: 320 }, 'g1', 1)];
  const after = applyRows(pair, extractFromGroup(pair, 2, { x: 1500, y: 900 }));
  assert.equal(byId(after, 1).groupId, null, 'the survivor is no longer in a group');
  assert.equal(byId(after, 1).groupPos, null);
  assert.deepEqual(layoutImageNodes(after).map((r) => r.kind), ['single', 'single']);
});

/* The strip is ANCHORED and re-flows inside its own box: taking a member out
   never moves the strip, the ones to its right slide left to close the gap.
   Which settles the two-member case too — the survivor lands at the strip's
   own origin, at the size it was drawn at, not at the slot it happened to
   occupy. The alternative (each survivor frozen where it drew) would let a
   strip creep rightwards across a board with every extraction, and would still
   have to answer "so where does the LAST one go?" with the same origin. */
test('the survivor of a dissolved group keeps the strip’s spot and its drawn size', () => {
  // 2 anchors, 1 follows. Taking 2 out must leave 1 at the STRIP's origin.
  const pair = [node(2, { x: 700, y: 100, w: 480, h: 320 }, 'g1', 0),
    node(1, { x: 100, y: 100, w: 320, h: 640 }, 'g1', 1)];   // 1:2, drawn at 160×320
  const drewAt = groupOf(layoutImageNodes(pair)).members[1];
  const after = applyRows(pair, extractFromGroup(pair, 2, { x: 20, y: 20 }));
  const survivor = byId(after, 1);
  assert.equal(survivor.x, 700, 'the strip’s own x, not the slot it sat in');
  assert.equal(survivor.y, 100);
  assert.equal(survivor.w, drewAt.w, 'and exactly the size it was drawn at');
  assert.equal(survivor.h, drewAt.h);
  assert.equal(survivor.w, 160, 'NOT its long-forgotten pre-merge 320');
});

test('a middle member leaving closes the gap without moving the strip', () => {
  const nodes = trio();
  const before = groupOf(layoutImageNodes(nodes));
  const after = applyRows(nodes, extractFromGroup(nodes, 2, { x: 4000, y: 0 }));
  const now = groupOf(layoutImageNodes(after));
  assert.equal(now.x, before.x);
  assert.equal(now.members[1].x, now.members[0].x + now.members[0].w,
    'the third slid left onto the second’s edge');
});

// ---- 4. the gesture: move the group vs take one out ----------------------

test('a member dragged INSIDE the group is not an extraction', () => {
  const box = groupOf(layoutImageNodes(trio()));
  assert.equal(shouldExtract(box, { x: box.x + 10, y: box.y + 10 }), false);
  assert.equal(shouldExtract(box, { x: box.x + box.w - 1, y: box.y + box.h - 1 }), false);
});

test('a member dragged clear of the group IS an extraction', () => {
  const box = groupOf(layoutImageNodes(trio()));
  assert.equal(shouldExtract(box, { x: box.x + box.w + 1, y: box.y + 10 }), true);
  assert.equal(shouldExtract(box, { x: box.x + 10, y: box.y - 1 }), true);
  assert.equal(shouldExtract(box, { x: box.x - 1, y: box.y + 10 }), true);
});

test('shouldExtract answers false for a node that is in no group', () => {
  assert.equal(shouldExtract(null, { x: 0, y: 0 }), false);
});

// ---- 5. what the drop feedback shows ------------------------------------

test('a dragged node over another one names the target and the side', () => {
  const nodes = [node(1, SQUARE), node(2, { x: 900, y: 400, w: 320, h: 320 })];
  const layout = layoutImageNodes(nodes);
  // Centre in the LEFT half of node 1.
  const hit = mergeTargetAt(layout, 2, { x: 150, y: 200 });
  assert.equal(hit.targetImageId, 1);
  assert.equal(hit.side, 'before');
  assert.equal(hit.count, 2, 'it announces what the group would become');
});

test('the right half of the target inserts after it', () => {
  const layout = layoutImageNodes([node(1, SQUARE), node(2, { x: 900, y: 400, w: 320, h: 320 })]);
  assert.equal(mergeTargetAt(layout, 2, { x: 390, y: 200 }).side, 'after');
});

test('a node over EMPTY board has no merge target', () => {
  const layout = layoutImageNodes([node(1, SQUARE), node(2, { x: 900, y: 400, w: 320, h: 320 })]);
  assert.equal(mergeTargetAt(layout, 2, { x: 5000, y: 5000 }), null);
});

test('a node is never a merge target for itself', () => {
  const layout = layoutImageNodes([node(1, SQUARE)]);
  assert.equal(mergeTargetAt(layout, 1, { x: 150, y: 200 }), null);
});

test('dropping onto an existing group aims at the member under the pointer', () => {
  const layout = layoutImageNodes([...trio(), node(9, { x: 3000, y: 0, w: 320, h: 320 })]);
  const g = groupOf(layout);
  const second = g.members[1];
  const hit = mergeTargetAt(layout, 9, { x: second.x + 10, y: second.y + 10 });
  assert.equal(hit.targetImageId, 2);
  assert.equal(hit.side, 'before');
  assert.equal(hit.groupId, 'g1');
  assert.equal(hit.count, 4, 'three plus the one being dropped');
});

// ---- 6. persistence: the board comes back as it was ---------------------

test('the rows a merge writes are enough to rebuild the strip after a reload', () => {
  const nodes = [node(1, SQUARE), node(2, { x: 900, y: 400, w: 480, h: 320 })];
  const rows = mergeIntoGroup(nodes, 2, 1, 'after');
  // What survives a round-trip through the API: these five fields per row.
  const reloaded = rows.map((r) => node(r.imageId, r, r.groupId, r.groupPos));
  const g = groupOf(layoutImageNodes(reloaded));
  assert.equal(g.members.length, 2);
  assert.deepEqual(g.members.map((m) => m.node.imageId), [1, 2]);
  assert.equal(g.w, 320 + 480);
});

test('a merge never rewrites a member’s remembered size — that is the way back', () => {
  const nodes = [node(1, SQUARE), node(2, { x: 900, y: 400, w: 480, h: 900 })];
  const rows = mergeIntoGroup(nodes, 2, 1, 'after');
  const moved = rows.find((r) => r.imageId === 2);
  assert.equal(moved.w, 480, 'its own width is kept for the day it leaves');
  assert.equal(moved.h, 900);
  assert.equal(moved.x, 900, 'and so is where it came from');
});

// ---- 7. degenerate input ------------------------------------------------

test('a member with a nonsense aspect ratio does not poison the strip', () => {
  const bad = [node(1, SQUARE, 'g1', 0), node(2, { x: 0, y: 0, w: 320, h: 0 }, 'g1', 1)];
  const g = groupOf(layoutImageNodes(bad));
  assert.ok(Number.isFinite(g.w) && g.w > 0, `group width was ${g.w}`);
  assert.ok(g.members.every((m) => Number.isFinite(m.w) && m.w > 0));
});

test('members with the same groupPos still get a stable, repeatable order', () => {
  const tied = [node(5, SQUARE, 'g1', 0), node(3, SQUARE, 'g1', 0)];
  const once = groupMembers(tied, 'g1').map((m) => m.imageId);
  const twice = groupMembers([...tied].reverse(), 'g1').map((m) => m.imageId);
  assert.deepEqual(once, twice);
});

test('extracting an image that is in no group changes nothing', () => {
  assert.deepEqual(extractFromGroup([node(1, SQUARE)], 1, { x: 5, y: 5 }), []);
});

// ---- the links back to the source checkpoints -----------------------------
/* One thing on screen gets one thread. A strip is one object to the eye and to
   every gesture, so a line per member fanned eight connectors out of eight
   points along one band — and free placement makes those lines long, which is
   exactly when the ink starts to matter. */

const source = (id, recordId, step) => ({
  imageId: id, x: 0, y: 0, w: 200, h: 200, visible: true,
  groupId: 'g', groupPos: id, image: { id, url: `/i/${id}.png`, record_id: recordId, step },
});

test('a strip whose pictures share a checkpoint draws ONE link, from the strip', () => {
  const strip = [source(1, 10, 500), source(2, 10, 500), source(3, 10, 500)];
  const layout = layoutImageNodes(strip);
  const anchors = edgeAnchors(layout);
  assert.equal(anchors.length, 1, 'one object, one thread');
  const band = layout.find((r) => r.kind === 'group');
  assert.deepEqual([anchors[0].x, anchors[0].y, anchors[0].w, anchors[0].h],
    [band.x, band.y, band.w, band.h], 'it leaves the STRIP, not a tile inside it');
});

test('…and a strip built from SEVERAL checkpoints owns up to every one of them', () => {
  // 📌 Pin all groups a whole generation run — routinely three epochs of the
  // same card. One line to one pill would attribute the other two to it.
  const strip = [source(1, 10, 500), source(2, 10, 1000), source(3, 10, 1000),
    source(4, 10, 1500)];
  const anchors = edgeAnchors(layoutImageNodes(strip));
  assert.deepEqual(anchors.map((n) => n.image.step), [500, 1000, 1500]);
  // Every one of them still leaves from the same point — a strip with threads,
  // never a comb.
  assert.equal(new Set(anchors.map((n) => `${n.x}:${n.y}`)).size, 1);
});

test('a lone picture still answers for itself, at the box it is drawn in', () => {
  const anchors = edgeAnchors(layoutImageNodes([node(1, SQUARE)]));
  assert.equal(anchors.length, 1);
  assert.deepEqual([anchors[0].x, anchors[0].y], [SQUARE.x, SQUARE.y]);
});

/** Apply the rows a merge/extract produced back onto the node list, the way the
 *  page does. */
function applyRows(nodes, rows) {
  const map = new Map(rows.map((r) => [r.imageId, r]));
  return nodes.map((n) => (map.has(n.imageId)
    ? { ...n, ...map.get(n.imageId) } : n));
}
