/* 🖼 Images pinned onto the ◉ LoRA Canvas — the geometry layer.

   The board exists to compare checkpoints, and what is actually being compared
   is the picture. A picture you can only see one at a time, in a modal, is a
   picture you cannot compare — so a generated image can be dropped ONTO the
   board as a node of its own, next to the pill that made it, moved and resized
   like everything else there.

   Same shape as utils/canvasPlacement.js and for the same reason: pure
   functions, no JSX and no DOM, so the parts that decide whether the feature
   ages well — what happens to a pin whose image was deleted, what a lane's
   extent becomes, where a brand-new pin lands — are exercised by `node --test`
   instead of by dragging things in a browser.

   Coordinates are LANE-LOCAL world units, exactly like a card position, so both
   kinds of node live in one coordinate system.

   ⚠️ LANE-LOCAL IS THE ANCHOR, NOT A CAGE. A picture may sit anywhere on the
   board, including ABOVE and LEFT of its own lane's origin — that is what the
   negative half of the reach below is for. What stays lane-local is the
   REFERENCE the number is measured from, and that is deliberate: a lane's world
   position is derived from the board's current filter (which datasets are
   ticked) and from the height of every lane above it, so it moves by hundreds
   of units the moment an unrelated dataset is unticked or gains a run —
   measured on a three-dataset board, 580 units for one untick, 118 for one new
   run. A picture stored in BOARD-absolute units would sit still while its lane
   slid out from under it, and the day a dataset above is filtered off, every
   picture below would be hovering over the wrong lane. Measured from its own
   lane, it travels with the run it is evidence about, which is the only
   relationship on this board that never goes stale.

   ⚠️ THE promise of this module: closing a pinned image must not forget it.
   `visible: false` is a state, not a deletion — the row keeps its x/y/w/h and
   re-opening reads them back (`openGeometry`). A close that dropped the
   geometry would make "put it back where it was" impossible to implement
   anywhere else. */

import { CARD_W, V_GAP, edgePath } from './lineageGraph.js';

// Size bounds, in world units (a run card is CARD_W = 264 wide, for scale).
// The floor keeps a node grabbable at any zoom. The ceiling is the load-bearing
// one: a node blown up to a few thousand pixels grows its lane's extent, and
// ✦ Fit would then collapse the whole board to a scale where nothing else is
// readable. Mirrored server-side — the clamp has to protect the NEXT load too.
export const IMG_MIN = 96;
export const IMG_MAX = 1400;
// What a fresh pin opens at: readable without being a wall, and comfortably
// smaller than the card it sits beside.
export const IMG_DEFAULT = 320;
// Air between a card and the pins beside it, and between two stacked pins.
const PIN_GAP = 48;

/* How far from its lane's origin a picture may be parked, on either axis and in
   either direction. A SAFETY RAIL, not a design limit: it exists so a corrupt
   row (1e9, a hand-edited database, MAX_SAFE_INTEGER) cannot make ✦ Fit collapse
   the whole board to a scale where nothing is readable — the exact failure the
   size ceiling above already guards against, which until now had no equivalent
   on the position axes at all.

   100 000 units is roughly 380 card-widths, and about seven full-screen drags at
   the minimum zoom: far beyond any placement made on purpose, and far beyond any
   made by accident. And a picture that somehow ends up out there is not lost —
   ✦ Tidy up brings every visible picture back beside the run that made it. */
export const IMG_REACH = 100000;

const reach = (v) => Math.min(IMG_REACH, Math.max(-IMG_REACH, v));

const num = (v, fallback) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};

const numOrNull = (v) => {
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

/**
 * Clamp one node's box into a usable SIZE and a reachable POSITION. Unusable
 * numbers degrade to the default rather than to 0/NaN: a node parked at NaN
 * would be unreachable on every future load and there is no UI to fix that.
 *
 * The position is bounded on both sides of zero (±IMG_REACH), not floored at
 * it. The floor was the wall: it made the lane's own origin the top-left corner
 * a picture could never get past, so a render could be dragged down and right
 * but never up or left — never above its lane, never beside the lane above it,
 * never into the free margin left of the board. Nothing about a pinned picture
 * needs that. It is not a step of the lineage, its link to the checkpoint that
 * made it is READ off the image row rather than stored (see `imageNodeEdges`),
 * so no coordinate can make it lie about where it came from, and the lane it
 * belongs to is a fact about the image, not about the pixel it sits on.
 */
export function clampImageBox(box) {
  return {
    x: reach(num(box?.x, 0)),
    y: reach(num(box?.y, 0)),
    w: Math.min(IMG_MAX, Math.max(IMG_MIN, num(box?.w, IMG_DEFAULT))),
    h: Math.min(IMG_MAX, Math.max(IMG_MIN, num(box?.h, IMG_DEFAULT))),
  };
}

/**
 * Normalise the API's rows into {image_id: node}.
 *
 * A row with no id — or with no `image` payload — is DROPPED. The image row is
 * what the node renders and what carries the link back to its checkpoint; a
 * node without one could only draw a broken picture, which is precisely the
 * ghost this feature must not grow. (The server prunes those rows on read; this
 * is the same refusal on the client, so a stale response cannot paint one.)
 */
export function toImageNodeMap(rows) {
  const out = {};
  for (const r of (Array.isArray(rows) ? rows : [])) {
    const id = r?.image_id ?? r?.imageId;
    if (id == null || !r?.image?.url) continue;
    out[id] = {
      imageId: Number(id),
      ...clampImageBox(r),
      visible: r.visible !== false,
      /* 🖼🖼 Group membership, when this picture is part of a side-by-side strip
         (utils/canvasImageGroups). Two ADDITIVE, nullable fields: a lane loaded
         from a database that predates them reads null everywhere and draws the
         board it always drew. Nothing here is renamed — the geometry keys and
         `visible` are stored in every user's database and keep their names. */
      groupId: r.group_id ?? r.groupId ?? null,
      groupPos: numOrNull(r.group_pos ?? r.groupPos),
      image: r.image,
    };
  }
  return out;
}

/** The nodes actually ON the board. */
export function visibleImageNodes(map) {
  return Object.values(map || {}).filter((n) => n.visible);
}

/**
 * Where an image should (re-)open.
 *
 * A remembered node wins over the fallback, always — that IS the feature. The
 * fallback is only for an image that has never been pinned in this lane.
 */
export function openGeometry(map, imageId, fallback) {
  const known = (map || {})[imageId];
  if (known) return { x: known.x, y: known.y, w: known.w, h: known.h };
  return clampImageBox(fallback);
}

/**
 * Where a BRAND-NEW pin lands: to the right of the card that produced it, at
 * the card's own top, sliding DOWN past pins already there.
 *
 * To the right and not below: the horizontal axis of a lineage carries meaning
 * (one column = one generation) for CARDS, and a pinned image is not a
 * generation — parking it in the margin beside its card keeps the tree's
 * columns readable while putting the picture next to what it is evidence for.
 * `taken` are the boxes already placed.
 */
export function defaultImageSpot(graph, recordId, step, taken) {
  const at = spotBesideCard(graph, recordId);
  return clampImageBox(slideBelow({ ...at, w: IMG_DEFAULT, h: IMG_DEFAULT }, taken));
}

/**
 * The top-left corner a pin anchored on `recordId` starts from: just right of
 * that card, at the card's own top.
 *
 * Exported because ✦ Tidy up has to answer the same question for a whole
 * side-by-side STRIP, and two placers would be two chances to disagree about
 * where "beside its run" is.
 *
 * A pin whose card is not on the board still gets a spot — off the right of the
 * lane — rather than being refused. Losing the picture would be a worse answer
 * than losing the card it points at.
 */
export function spotBesideCard(graph, recordId) {
  const nodes = graph?.nodes || [];
  const card = nodes.find((n) => n.node?.record_id === recordId);
  let maxX = 0;
  for (const n of nodes) maxX = Math.max(maxX, n.x + CARD_W);
  return {
    x: (card ? card.x + CARD_W : maxX) + PIN_GAP,
    y: card ? card.y : 0,
  };
}

const overlaps = (a, b) => (a.x < b.x + b.w && b.x < a.x + a.w
  && a.y < b.y + b.h && b.y < a.y + a.h);

/**
 * The first spot at or BELOW `box` that overlaps nothing in `taken`.
 *
 * Down and not sideways, like every other placer on this board: the horizontal
 * axis of a lineage carries meaning (one column = one generation) and vertical
 * space is the free dimension. Bounded by construction — each step jumps below
 * the lowest blocker, so it terminates after at most `taken.length` moves.
 *
 * Size is carried through untouched, so a caller placing something that is not
 * a single picture (a strip, which is as wide as all its members put together)
 * gets an answer measured on the real footprint.
 */
export function slideBelow(box, taken) {
  const list = Array.isArray(taken) ? taken : [];
  let y = num(box?.y, 0);
  for (let guard = 0; guard <= list.length; guard += 1) {
    const hits = list.filter((t) => overlaps({ ...box, y }, t));
    if (!hits.length) break;
    y = Math.max(...hits.map((t) => t.y + t.h)) + V_GAP;
  }
  return { ...box, y };
}

/**
 * The box a lane's pins really occupy, relative to the LANE's origin.
 *
 * `width`/`height` are how far right and down they reach, as before, so
 * `stackLanes` sizes the lane to hold them and ✦ Fit cannot crop a picture off
 * the bottom of the board.
 *
 * `minX`/`minY` are the OVERHANG — how far a picture reaches above or left of
 * its lane's origin — and are never positive, because the lane's own header and
 * tree start at zero and the box has to contain them too. They exist because a
 * picture may now be parked above its lane (see `clampImageBox`): a board that
 * measured its size from the origin down would leave that picture outside the
 * world it fits to, and ✦ Fit would frame a board with a picture floating off
 * the top of it and no way to bring the view back to it.
 */
export function imageNodeExtent(nodes) {
  let minX = 0;
  let minY = 0;
  let width = 0;
  let height = 0;
  for (const n of (nodes || [])) {
    minX = Math.min(minX, num(n?.x, 0));
    minY = Math.min(minY, num(n?.y, 0));
    width = Math.max(width, num(n?.x, 0) + num(n?.w, 0));
    height = Math.max(height, num(n?.y, 0) + num(n?.h, 0));
  }
  return { minX, minY, width, height };
}

/**
 * The link from each pinned image to the checkpoint that produced it, in the
 * shape components/dataset/lineageEdges.jsx already draws.
 *
 * Deliberately the SAME connector as the one joining a continuation to the
 * checkpoint it resumed from — the board already has a grammar for "this came
 * from that" and a second one would just be a second thing to learn. It is the
 * NEUTRAL variant of it (not the trunk, not the amber superseded branch),
 * because a render is evidence about a checkpoint, not a step in the training
 * lineage: claiming the trunk for it would overstate what happened.
 *
 * An image whose checkpoint is not on the board (its run was deleted, its step
 * cleaned off the disk) draws NO edge, and the node stays. A missing link is
 * not a missing picture.
 */
export function imageNodeEdges(nodes, graph) {
  const cards = graph?.nodes || [];
  const edges = [];
  for (const n of (nodes || [])) {
    const recordId = n.image?.record_id;
    const step = n.image?.step;
    if (recordId == null || step == null) continue;
    const card = cards.find((c) => c.node?.record_id === recordId);
    const pill = card?.checkpoints?.find((p) => p.step === step);
    if (!pill) continue;
    const x1 = pill.x + pill.w;
    const y1 = pill.y + pill.h / 2;
    const x2 = n.x;
    const y2 = n.y + n.h / 2;
    edges.push({
      parentId: `ck:${recordId}:${step}`,
      childId: `img:${n.imageId}`,
      x1, y1, x2, y2, d: edgePath(x1, y1, x2, y2),
      onSpine: false, superseded: false,
    });
  }
  return edges;
}

// Keyboard steps. Moving and resizing with a mouse only would put the whole
// feature out of reach of anyone who does not use one; these are the same
// gestures through the arrow keys, with Shift for a coarse step.
const MOVE_STEP = 16;
const MOVE_STEP_FAST = 64;
const SIZE_STEP = 32;
const SIZE_STEP_FAST = 128;

/**
 * One keyboard nudge of a focused node. Returns the new box, or NULL when the
 * key is none of ours — the caller must not swallow keys it does not handle.
 */
export function nudgeImageNode(node, key, shift) {
  const box = clampImageBox(node);
  const move = shift ? MOVE_STEP_FAST : MOVE_STEP;
  const size = shift ? SIZE_STEP_FAST : SIZE_STEP;
  switch (key) {
    case 'ArrowLeft': return clampImageBox({ ...box, x: box.x - move });
    case 'ArrowRight': return clampImageBox({ ...box, x: box.x + move });
    case 'ArrowUp': return clampImageBox({ ...box, y: box.y - move });
    case 'ArrowDown': return clampImageBox({ ...box, y: box.y + move });
    case '+': case '=':
      return clampImageBox({ ...box, w: box.w + size, h: box.h + size });
    case '-': case '_':
      return clampImageBox({ ...box, w: box.w - size, h: box.h - size });
    default: return null;
  }
}
