/* 📌 Pin ALL the images a run just produced onto the ◉ LoRA Canvas, in one go.

   A finished generation says "5 images ready" and names the checkpoints they
   joined. Getting them onto the board meant opening each of those galleries and
   pinning the pictures one by one — for a lot spanning four runs, four panels
   and five gestures to see the thing the run was fired to let you see.

   This module is the whole decision layer of the one-click version. Pure, no
   JSX and no DOM, because `node --test` cannot parse JSX and the part that
   decides whether the feature is usable is geometry, not markup.

   ⚠️ EVERYTHING HERE IS IN BOARD UNITS, never in screen pixels. The board is
   zoomable (a real board is often read at 24 %); a rule written in pixels would
   be right at one zoom level and wrong at every other one.

   ── The placement, and why this one ──────────────────────────────────────────

   The requirement is not "tidy", it is "nothing may ever be on top of anything
   else" — a lot dropped onto a board that already carries cards, checkpoint
   pills and hand-placed pictures. Three candidates were on the table: a spiral
   around each source, a free-space search from the source's own spot, and a
   band under the tree. The first two share one flaw: they place INTO the tree,
   so they can only work by searching around obstacles, and the result reads as
   scatter — you cannot tell at a glance which picture came from which run, which
   is the only reason to put them on the board at all.

   So: a CONTACT-SHEET BAND below the lane, one COLUMN per source checkpoint,
   each column starting at the x of its own source and sliding right until it
   finds a free column.

     • no overlap is structural, not searched for: the band starts below the
       lowest thing already on the lane, and inside the band each column is
       reserved before it is filled. There is no case where a tile is placed
       "hopefully";
     • a source's images are a vertical stack under that source, so a lot from
       four runs reads as four groups without a legend — and the connector each
       node already draws to its own pill (utils/canvasImageNodes.imageNodeEdges)
       confirms it;
     • it is trivially deterministic — no search order to depend on.

   The cost, stated: a picture is BELOW the tree rather than beside its card, so
   on a deep lineage it is further from its pill than a single hand-pin would
   be. That is the trade the requirement asks for; a pin dropped by hand still
   lands beside its card (defaultImageSpot), which is unchanged.

   ── When there is no room ────────────────────────────────────────────────────

   Structurally, there always is: the band grows down and right, outside
   everything, and the lane grows with it (imageNodeExtent). So the only bound
   is a deliberate one — PIN_BATCH_MAX — and it is REPORTED, never silent. A lot
   over the cap places the cap and says how many it left out and where to get
   them. Nothing is ever stacked quietly. */

import { CARD_W } from './lineageGraph.js';
import { layoutImageNodes, nextGroupId, occupiedBox } from './canvasImageGroups.js';
import { groupBarMaxHeight } from './canvasNodeChrome.js';
import {
  IMG_DEFAULT, IMG_MAX, IMG_MIN, slideBelow, spotBesideCard,
} from './canvasImageNodes.js';

/* How many pictures one click may put down. Not a technical limit — the band
   would happily hold ten times that — but an editorial one: past a few dozen
   tiles the board stops being a comparison and becomes a folder, and undoing a
   click that changed a hundred rows is a worse offer than being told to pick.
   Over the cap, the surplus is named, not dropped in silence. */
export const PIN_BATCH_MAX = 40;

// Air between two tiles, and between the tree and the band under it.
const TILE_GAP = 24;
const BAND_GAP = 56;
// How tall one column may grow before the next image starts a new one. Six
// tiles is roughly one screen at a readable zoom; beyond that a column stops
// being scannable.
const COLUMN_ROWS = 6;

/**
 * How big the tiles of a lot are.
 *
 * A lot is looked at as a WHOLE — that is why it is pinned in one gesture — so
 * the size follows the count: a pair of renders at full size, thirty as a
 * contact sheet you can actually take in. Every value stays inside the bounds
 * the node itself enforces (IMG_MIN/IMG_MAX), and each tile is still resizable
 * afterwards like any other.
 */
export function batchTileSize(count) {
  const n = Number(count) || 0;
  const size = n <= 4 ? IMG_DEFAULT
    : n <= 12 ? 240
      : n <= 24 ? 192
        : 160;
  return Math.min(IMG_MAX, Math.max(IMG_MIN, size));
}

const rectsOverlap = (a, b) => (
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
);

/**
 * Grouping turns several independently placed squares into a WIDE rendered
 * strip whose title bar also occupies space above it. The source-column placer
 * cannot see that future footprint: after its sixth row wraps, a new strip's
 * anchor may sit exactly under the right half of an earlier strip.
 *
 * Keep every automatic group at its preferred x, then move only a conflicting
 * anchor DOWN until its real `occupiedBox` is free. Down is deliberate: the
 * batch band already sits below the lineage tree, so it preserves checkpoint
 * order and cannot cross back into the cards. Existing nodes and fresh singles
 * are reserved too. Only fresh rows are changed; no persisted/manual group is
 * ever reflowed by a later Pin all gesture.
 */
function separateFreshGroupFootprints(nodes, rows, groupIds, graph = null) {
  const targets = groupIds instanceof Set ? groupIds : new Set(groupIds || []);
  if (!targets.size) return rows;
  const freshIds = new Set((rows || []).map((row) => Number(row.imageId)));
  const existing = (nodes || []).filter((node) => node?.visible !== false
    && !freshIds.has(Number(node?.imageId)));
  // Cards/pills are not image nodes and cannot be reconstructed here. The
  // caller therefore passes the lane graph it actually placed against; this is
  // load-bearing for remembered positions, which can each be free while the
  // WIDE strip they are about to form reaches back across a card.
  const reserved = [
    ...boardObstacles(graph, []),
    ...layoutImageNodes(existing).map(occupiedBox),
  ];
  const layout = layoutImageNodes(rows || []);

  // Singles keep the exact spots the placer/remembered geometry chose. Groups
  // move around them, never the other way round.
  for (const row of layout) {
    if (row.kind !== 'group' || !targets.has(row.groupId)) reserved.push(occupiedBox(row));
  }

  const groups = layout.filter((row) => row.kind === 'group' && targets.has(row.groupId))
    .sort((a, b) => a.y - b.y || a.x - b.x
      || (a.members[0]?.node.imageId ?? 0) - (b.members[0]?.node.imageId ?? 0));
  const rowById = new Map((rows || []).map((row) => [Number(row.imageId), row]));
  for (const group of groups) {
    const anchor = rowById.get(Number(group.members[0]?.node.imageId));
    if (!anchor) continue;
    let footprint = occupiedBox(group);
    while (true) {
      const hits = reserved.filter((box) => rectsOverlap(footprint, box));
      if (!hits.length) break;
      const nextTop = Math.max(...hits.map((box) => box.y + box.h)) + TILE_GAP;
      const dy = nextTop - footprint.y;
      anchor.y += dy;
      footprint = { ...footprint, y: footprint.y + dy };
    }
    reserved.push(footprint);
  }
  return rows;
}

/**
 * Everything already occupying the lane, as plain rectangles.
 *
 * A run card and its wrapped checkpoint pills are ONE rectangle: `cellH` is the
 * cell height the layout computed (card + pill block) and the pills never wrap
 * past CARD_W, so the cell box contains them. Publishing two overlapping
 * obstacles for one card would be harmless but would make the tests lie about
 * what they check.
 *
 * CLOSED pictures are not obstacles: they are not on the board.
 */
export function boardObstacles(graph, imageNodes) {
  const out = [];
  for (const n of (graph?.nodes || [])) {
    out.push({ x: n.x, y: n.y, w: CARD_W, h: n.cellH });
  }
  for (const n of (imageNodes || [])) {
    if (n?.visible === false) continue;
    out.push({ x: n.x, y: n.y, w: n.w, h: n.h });
  }
  return out;
}

/** The source checkpoint of an image, as a stable key. One BAND COLUMN each —
 *  which is a question about where a picture came from on the tree, not about
 *  which click produced it. */
const sourceKey = (img) => `${img?.record_id ?? '?'}:${img?.step ?? '?'}`;

/**
 * WHICH GENERATION + PROMPT made a picture — the identity two grids must never
 * share.
 *
 * `lora_test_image.run_id` already groups every cell of one launch (it is what
 * the Test Studio resumes a run from), while `prompt` identifies one grid
 * INSIDE a multi-prompt launch. The checkpoint gallery publishes both. A key
 * made from `run_id` alone turns all prompts selected for one launch into one
 * strip; a key made from `prompt` alone joins separate launches. The pair is
 * the boundary the Canvas actually needs.
 *
 * An image made before that column was backfilled carries no run id and falls
 * back to its checkpoint, so a board that predates this draws exactly what it
 * drew before rather than silently regrouping itself.
 */
const normalPrompt = (value) => String(value ?? '').trim().replace(/\s+/g, ' ');

const runPromptKey = (value) => {
  const image = value?.image || value;
  const runId = image?.run_id;
  if (runId == null || String(runId) === '') return null;
  // JSON, not a delimiter: prompts are free text and may contain any separator
  // we could choose. Normalising whitespace makes the key match what the UI
  // shows as one prompt while preserving meaningful text and case.
  return `run:${JSON.stringify([String(runId), normalPrompt(image?.prompt)])}`;
};

export function imageBatchKey(value) {
  const image = value?.image || value;
  const runKey = runPromptKey(image);
  if (runKey) return runKey;
  if (image?.record_id == null || image?.step == null) return null;
  return `ckpt:${String(image.record_id)}:${String(image.step)}`;
}

/**
 * TRAINING order: the step that made the picture, ascending. The one order a
 * strip of checkpoints is allowed to have — reading epoch 500 next to epoch
 * 2000 is the entire reason the pictures are side by side.
 *
 * It was `${record_id}:${step}` compared as TEXT, which sorts step 1000 before
 * step 500 and put a four-checkpoint lot on the board shuffled. Steps are
 * numbers here, and a missing one sorts last rather than colliding with 0.
 */
export function byTrainingOrder(a, b) {
  const imageA = a?.image || a;
  const imageB = b?.image || b;
  const rawA = Number(imageA?.step);
  const rawB = Number(imageB?.step);
  const stepA = Number.isFinite(rawA) ? rawA : Infinity;
  const stepB = Number.isFinite(rawB) ? rawB : Infinity;
  if (stepA !== stepB) return stepA - stepB;
  const recA = Number(imageA?.record_id) || 0;
  const recB = Number(imageB?.record_id) || 0;
  if (recA !== recB) return recA - recB;
  return (Number(a?.id ?? a?.imageId) || 0) - (Number(b?.id ?? b?.imageId) || 0);
}

/** Turn freshly pinned images into (or append them to) one strip per GENERATION
 * RUN + PROMPT (imageBatchKey) — not per checkpoint. Pinning a picture from a
 * gallery joins the grid of the prompt it belongs to; another prompt in the
 * SAME run starts its own. A picture from a later run does too. Manual mixed
 * groups are never reused. The undo snapshot covers both the new images and any
 * existing member whose membership is rewritten. */
export function groupPinnedBatchBySource({ nodes = [], placed = [], graph = null } = {}) {
  const before = new Map((nodes || []).filter((n) => n?.imageId != null)
    .map((n) => [Number(n.imageId), { ...n }]));
  const working = new Map([...before].map(([id, n]) => [id, { ...n }]));
  const fresh = [...(placed || [])].filter((p) => p?.imageId != null)
    .sort(byTrainingOrder);

  for (const p of fresh) {
    const id = Number(p.imageId);
    const old = before.get(id);
    working.set(id, {
      imageId: id, x: p.x, y: p.y, w: p.w, h: p.h, visible: true,
      groupId: old?.groupId ?? null, groupPos: old?.groupPos ?? null,
      image: p.image || old?.image,
    });
  }

  const affected = new Map();
  const undo = new Map();
  const promptGroupIds = new Set();
  const remember = (node) => {
    const id = Number(node.imageId);
    if (!undo.has(id)) {
      const old = before.get(id);
      undo.set(id, old ? { ...old } : {
        ...node, visible: false, groupId: null, groupPos: null,
      });
    }
    affected.set(id, node);
  };

  const freshByBatch = new Map();
  for (const p of fresh) {
    const node = working.get(Number(p.imageId));
    const key = imageBatchKey(node);
    if (!key) continue;
    if (!freshByBatch.has(key)) freshByBatch.set(key, []);
    freshByBatch.get(key).push(node);
  }

  for (const [key, additions] of [...freshByBatch].sort(([a], [b]) => a.localeCompare(b))) {
    const originalVisible = [...before.values()].filter((n) => n.visible !== false);
    const groupIds = [...new Set(originalVisible.map((n) => n.groupId).filter(Boolean))].sort();
    let members = null;
    let groupId = null;
    for (const gid of groupIds) {
      const list = originalVisible.filter((n) => n.groupId === gid)
        .sort((a, b) => (a.groupPos ?? 0) - (b.groupPos ?? 0) || a.imageId - b.imageId);
      if (list.length >= 2 && list.every((n) => imageBatchKey(n) === key)) {
        members = [...list, ...additions.filter((n) => !list.some((m) => m.imageId === n.imageId))];
        groupId = gid;
        break;
      }
    }
    if (!members) {
      const anchor = originalVisible
        .filter((n) => !n.groupId && imageBatchKey(n) === key)
        .sort(byTrainingOrder)[0];
      members = [...(anchor ? [anchor] : []), ...additions]
        .filter((n, i, all) => all.findIndex((m) => m.imageId === n.imageId) === i);
      if (members.length < 2) continue;
      groupId = nextGroupId([...working.values()], members[0].imageId);
    }
    // A run+prompt key is an automatic Canvas grid. Legacy checkpoint keys are
    // deliberately excluded, as are manual mixed groups (the homogeneous-group
    // check above refuses to reuse those). The whole current membership is in
    // `affected`, so moving its anchor remains one reversible/undoable write.
    if (runPromptKey(additions[0]) === key) promptGroupIds.add(groupId);
    // ALWAYS training order, even when the picture that just arrived belongs
    // earlier than everything already in the strip: pinning epoch 500 after
    // epoch 2000 must not put 500 on the right-hand end. Nothing is lost by
    // re-sorting — an automatic strip's order is never hand-set (dragging a
    // member around INSIDE its strip is a no-op, see canvasImageGroups).
    members.sort(byTrainingOrder);
    members.forEach((member, pos) => {
      const updated = { ...working.get(member.imageId), groupId, groupPos: pos };
      working.set(member.imageId, updated);
      remember(updated);
    });
  }

  for (const p of fresh) remember(working.get(Number(p.imageId)));
  const byId = (a, b) => a.imageId - b.imageId;
  const rows = [...affected.values()].sort(byId);
  separateFreshGroupFootprints(nodes, rows, promptGroupIds, graph);
  return { rows, undoRows: [...undo.values()].sort(byId) };
}

/** Turn one freshly generated/pinned lot into a strip PER GENERATION RUN AND
 * PROMPT, ordered by training step.
 *
 * Different checkpoints of ONE prompt belong together here — that strip is the
 * prompt's comparison grid. Another prompt selected in the SAME 📌 Pin all
 * gesture must get another strip; two runs can never share one either. Existing
 * groups are never reused. Images with no run id (made before the column was
 * backfilled) keep the old whole-gesture strip, which is safe — this function
 * cannot merge into anything that was already on the board. */
export function groupPinnedBatchTogether({ nodes = [], placed = [], graph = null } = {}) {
  const before = new Map((nodes || []).filter((n) => n?.imageId != null)
    .map((n) => [Number(n.imageId), { ...n }]));
  const fresh = [...(placed || [])].filter((p) => p?.imageId != null)
    .sort(byTrainingOrder);

  const lots = new Map();
  for (const p of fresh) {
    // Legacy rows deliberately stay one whole-gesture lot. The unit-pin path's
    // no-run fallback remains per checkpoint in `imageBatchKey`; these are two
    // historical behaviours and neither should be silently rewritten here.
    const key = runPromptKey(p) ?? 'gesture';
    if (!lots.has(key)) lots.set(key, []);
    lots.get(key).push(p);
  }
  const groupOf = new Map();
  const posOf = new Map();
  const promptGroupIds = new Set();
  const taken = [...(nodes || []), ...fresh];
  for (const [lotKey, lot] of lots) {
    if (lot.length < 2) continue;
    const groupId = nextGroupId(taken, lot[0].imageId);
    taken.push({ groupId });
    // Only modern run+prompt groups participate in the new footprint reflow.
    // The `gesture` lot is the pre-run_id fallback and keeps its historical
    // geometry as well as its historical membership semantics.
    if (lotKey !== 'gesture') promptGroupIds.add(groupId);
    lot.forEach((p, pos) => {
      groupOf.set(Number(p.imageId), groupId);
      posOf.set(Number(p.imageId), pos);
    });
  }

  const rows = fresh.map((p) => {
    const id = Number(p.imageId);
    const old = before.get(id);
    const groupId = groupOf.get(id) ?? null;
    return {
      imageId: id, x: p.x, y: p.y, w: p.w, h: p.h, visible: true,
      groupId, groupPos: groupId ? posOf.get(id) : null,
      image: p.image || old?.image,
    };
  });
  separateFreshGroupFootprints(nodes, rows, promptGroupIds, graph);
  const undoRows = rows.map((row) => {
    const old = before.get(row.imageId);
    return old ? { ...old } : {
      ...row, visible: false, groupId: null, groupPos: null,
    };
  });
  return { rows, undoRows };
}

/**
 * Where a source's column WANTS to start, horizontally: the x of the card that
 * produced it. An image whose run is not on the board (deleted, filtered off)
 * anchors past the right edge instead of being refused — losing the picture
 * would be a worse answer than losing the alignment.
 */
function anchorX(graph, recordId) {
  const nodes = graph?.nodes || [];
  const card = nodes.find((n) => n.node?.record_id === recordId);
  if (card) return card.x;
  let maxX = 0;
  for (const n of nodes) maxX = Math.max(maxX, n.x + CARD_W);
  return maxX + TILE_GAP;
}

/**
 * Split a lot into what still has to be pinned and what is already on the board.
 *
 * An image the user can already see is NOT part of the lot: re-placing it would
 * move a picture they positioned themselves, and adding a second node for it
 * would be a duplicate of the same file. A CLOSED one is part of the lot again —
 * closing is "not now", not "never".
 */
export function pinBatchPending(images, laneMap) {
  const map = laneMap || {};
  const pending = [];
  let already = 0;
  for (const img of (Array.isArray(images) ? images : [])) {
    if (img?.id == null) continue;
    if (map[img.id]?.visible) { already += 1; continue; }
    pending.push(img);
  }
  return { pending, already };
}

/**
 * The same question across the WHOLE board: of the lot a run produced, how much
 * is not on the board yet?
 *
 * This is what the button counts, and it is why the button can disappear: a lot
 * whose every picture is already pinned has nothing to offer, and a button that
 * stays lit doing nothing is worse than no button at all.
 *
 * `candidates` are {id, datasetId} (utils/canvasRunResults.runPinCandidates);
 * `lanes` is the board's {datasetId: {imageId: node}} map.
 */
export function pinBatchPendingAcrossLanes(candidates, lanes) {
  const byLane = lanes || {};
  const pending = [];
  let already = 0;
  for (const c of (Array.isArray(candidates) ? candidates : [])) {
    if (c?.id == null) continue;
    if (byLane[c.datasetId]?.[c.id]?.visible) { already += 1; continue; }
    pending.push(c);
  }
  return { pending, already };
}

/**
 * Place a lot of images on one lane.
 *
 * `graph`      — the lane's graph AS DISPLAYED (applyPlacement's output), so the
 *                obstacles are where the user sees them, not where the automatic
 *                tree would have put them.
 * `existing`   — the lane's visible image nodes.
 * `images`     — the lot, each {id, record_id, step, …}. Already-visible ones
 *                must have been filtered out by pinBatchPending first.
 * `remembered` — the lane's full node map, so a picture that was CLOSED can come
 *                back where it was closed (the promise of canvasImageNodes) —
 *                but only when that spot is free. "Nothing overlaps" outranks
 *                "everything remembers": a remembered spot that would land on
 *                something goes into the band like a fresh one.
 *
 * Returns {size, placed, skipped}. `skipped` is never empty in silence — the
 * caller announces it (pinBatchAnnouncement).
 */
export function placeImageBatch({ graph, existing, images, remembered, max } = {}) {
  const cap = Number.isFinite(max) ? Math.max(0, max) : PIN_BATCH_MAX;
  const all = (Array.isArray(images) ? images : []).filter((i) => i?.id != null);

  // Deterministic order FIRST, cap SECOND: which images are refused must not
  // depend on the order the API happened to list the run's cells in. Training
  // order, so an over-cap lot keeps the EARLY epochs and drops the tail —
  // "which checkpoint starts working" is the question the band is read for.
  const ordered = [...all].sort(byTrainingOrder);
  const taking = ordered.slice(0, cap);
  const skipped = ordered.slice(cap).map((image) => ({ image, reason: 'over-cap' }));

  const size = batchTileSize(taking.length);
  const occupied = boardObstacles(graph, existing);
  const placed = [];

  // Pass 1 — the pictures that were closed and remember a free spot. Done first
  // so the band, computed next, is placed below them too.
  const bandBound = new Set();
  for (const image of taking) {
    const known = (remembered || {})[image.id];
    if (!known || known.visible) continue;
    const box = { x: known.x, y: known.y, w: known.w, h: known.h };
    if (![box.x, box.y, box.w, box.h].every((v) => Number.isFinite(v))) continue;
    if (occupied.some((o) => rectsOverlap(box, o))) continue;
    occupied.push(box);
    placed.push({ imageId: image.id, ...box, image });
    bandBound.add(image.id);
  }

  const band = taking.filter((i) => !bandBound.has(i.id));
  if (band.length) {
    // The band starts below EVERYTHING already on the lane. That single line is
    // what makes "no overlap" structural rather than searched for.
    let bandTop = 0;
    for (const o of occupied) bandTop = Math.max(bandTop, o.y + o.h);
    // A current (run_id-bearing) batch will become one or more Canvas groups
    // immediately after this placement. Reserve the TALLEST possible group bar
    // now: it is drawn above the strip, and without this allowance the first
    // prompt grid can cover the lineage card the band was placed below. Legacy
    // no-run lots keep their exact historical geometry.
    const futureGroupBar = band.some((image) => runPromptKey(image) != null)
      ? groupBarMaxHeight(size) : 0;
    bandTop += BAND_GAP + futureGroupBar;

    const colW = size + TILE_GAP;
    // The same reservation between source rows prevents the bar of prompt N+1
    // from climbing into prompt N before the final cross-column reflow above.
    const rowH = size + TILE_GAP + futureGroupBar;

    // One group per source checkpoint, ordered by where that source sits on the
    // board (left to right, then top to bottom) so the band reads in the same
    // order as the tree above it.
    const groups = new Map();
    for (const image of band) {
      const key = sourceKey(image);
      if (!groups.has(key)) {
        groups.set(key, { key, first: image, ax: anchorX(graph, image.record_id), images: [] });
      }
      groups.get(key).images.push(image);
    }
    // Columns read left to right in TRAINING order when two checkpoints anchor
    // at the same place — a tie broken by the alphabet put step 1000 left of
    // step 500.
    const ordering = [...groups.values()].sort((a, b) => (a.ax - b.ax)
      || byTrainingOrder(a.first, b.first));

    // Columns are RESERVED before they are filled: a group takes the first free
    // column at or right of the one its own source sits over. Two runs at the
    // same depth therefore get neighbouring columns instead of the same one.
    const usedColumns = new Set();
    const takeColumn = (preferred) => {
      let c = Math.max(0, preferred);
      while (usedColumns.has(c)) c += 1;
      usedColumns.add(c);
      return c;
    };

    for (const group of ordering) {
      const preferred = Math.max(0, Math.round(group.ax / colW));
      let col = takeColumn(preferred);
      let row = 0;
      for (const image of group.images.sort(byTrainingOrder)) {
        if (row >= COLUMN_ROWS) { col = takeColumn(col + 1); row = 0; }
        const box = { x: col * colW, y: bandTop + row * rowH, w: size, h: size };
        occupied.push(box);
        placed.push({ imageId: image.id, ...box, image });
        row += 1;
      }
    }
  }

  placed.sort(byTrainingOrder);
  return { size, placed, skipped };
}

/**
 * ✦ Tidy up, for the side-by-side STRIPS of one lane: bring each one back
 * beside the run that made it, WITHOUT taking it apart.
 *
 * ── Why strips move at all now ───────────────────────────────────────────────
 * They used to be left exactly where they were, and the argument was sound
 * while a picture could not leave its lane: a strip is something the user
 * assembled on purpose, and re-flowing its members one by one would not tidy it,
 * it would dismantle it. Free placement changes the stakes, not the argument.
 * A strip can now be parked thousands of units above or left of everything, and
 * "the button that rebuilds the board" leaving it out there would mean a whole
 * assembled comparison with no way back short of finding it by hand at 10 %
 * zoom. So the strip comes back — as ONE object, which is the thing the old
 * argument was actually protecting.
 *
 * ── How ──────────────────────────────────────────────────────────────────────
 * Only the ANCHOR's row is written. The strip is DERIVED from the anchor's spot
 * (utils/canvasImageGroups.layoutImageNodes), so moving the anchor moves the
 * band, and every member keeps the geometry it gets back the day it is dragged
 * out. The group fields are deliberately NOT in the returned rows either: a
 * write that only mentions geometry can never dissolve a group, so a half-failed
 * tidy leaves a strip moved, never broken.
 *
 * Placed through the SAME rule a hand-dropped pin uses — beside its own card,
 * sliding down past whatever is already there (canvasImageNodes.spotBesideCard
 * / slideBelow) — measured on `occupiedBox`, so the strip's drag BAR is
 * reserved too and the contact-sheet band placed afterwards cannot land on it.
 *
 * `taken` are extra boxes already spoken for; the lane's own run cards are added
 * here rather than asked for, so a caller cannot forget them and drop a strip on
 * a card. Returns { rows, boxes }: the rows to persist, and the footprints the
 * STRIPS now occupy — exactly the shape `placeImageBatch` wants as `existing`,
 * so the contact-sheet band placed next cannot disagree about what is free.
 * (The cards are not in `boxes`: `placeImageBatch` reads those off the graph
 * itself, and publishing them twice would only make its tests lie.)
 */
export function tidyGroupRows({ graph, layout, taken } = {}) {
  const strips = (layout || []).filter((r) => r?.kind === 'group' && r.members?.length);
  const boxes = [];
  if (!strips.length) return { rows: [], boxes };
  const cards = graph?.nodes || [];
  const busy = [...(taken || [])];
  for (const n of cards) busy.push({ x: n.x, y: n.y, w: CARD_W, h: n.cellH });

  // Deterministic order: where the source card sits on the tree, top to bottom
  // then left to right, tie-broken by the anchor's image id. Two strips must
  // never swap places just because the API listed their rows the other way
  // round — the same reason applyPlacement sorts its arrivals.
  const cardOf = (row) => cards.find(
    (c) => c.node?.record_id === row.members[0].node.image?.record_id);
  const ordered = [...strips].sort((a, b) => {
    const ca = cardOf(a);
    const cb = cardOf(b);
    return ((ca?.y ?? Infinity) - (cb?.y ?? Infinity))
      || ((ca?.x ?? Infinity) - (cb?.x ?? Infinity))
      || (a.members[0].node.imageId - b.members[0].node.imageId);
  });

  const rows = [];
  for (const strip of ordered) {
    const anchor = strip.members[0].node;
    const footprint = occupiedBox(strip);
    // What occupiedBox reserved ABOVE the pictures for the group's drag bar.
    const bar = footprint.h - strip.h;
    const at = spotBesideCard(graph, anchor.image?.record_id);
    const spot = slideBelow({ ...at, w: footprint.w, h: footprint.h }, busy);
    const landed = { x: spot.x, y: spot.y, w: footprint.w, h: footprint.h };
    busy.push(landed);
    boxes.push(landed);
    rows.push({ imageId: anchor.imageId, x: spot.x, y: spot.y + bar,
      w: anchor.w, h: anchor.h });
  }
  return { rows, boxes };
}

/** The button's own words. It must say HOW MANY it is about to put down —
 *  "Pin all" alone gives no idea whether one click adds two pictures or thirty. */
export function pinBatchLabel(count) {
  const n = Number(count) || 0;
  if (n <= 0) return '';
  if (n === 1) return '📌 Pin this image to the board';
  return `📌 Pin all ${n} to the board`;
}

/**
 * What is said out loud once the click has happened (aria-live, and on screen).
 *
 * A bulk action that reports nothing is a bulk action you have to go and audit.
 * The refused images are named WITH the way to get them — a count on its own
 * ("7 left out") is a dead end.
 */
export function pinBatchAnnouncement(result) {
  const placed = result?.placed?.length || 0;
  const left = result?.skipped?.length || 0;
  if (!placed && !left) return 'Nothing to pin — these images are already on the board.';
  const head = placed === 1 ? '1 image pinned' : `${placed} images pinned`;
  if (!left) return `${head} to the board.`;
  return `${head} to the board — ${left} left out (${PIN_BATCH_MAX} at a time). `
    + 'Pin the rest from the checkpoint gallery.';
}
