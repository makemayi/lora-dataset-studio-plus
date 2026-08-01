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
import { nextGroupId } from './canvasImageGroups.js';
import { IMG_DEFAULT, IMG_MAX, IMG_MIN } from './canvasImageNodes.js';

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

/** The source checkpoint of an image, as a stable key. */
const sourceKey = (img) => `${img?.record_id ?? '?'}:${img?.step ?? '?'}`;
const strictSourceKey = (value) => {
  const image = value?.image || value;
  if (image?.record_id == null || image?.step == null) return null;
  return `${String(image.record_id)}:${String(image.step)}`;
};

/** Turn freshly pinned images into (or append them to) one strip per checkpoint.
 * Manual mixed-source groups are never reused. The undo snapshot covers both
 * the new images and any existing member whose membership is rewritten. */
export function groupPinnedBatchBySource({ nodes = [], placed = [] } = {}) {
  const before = new Map((nodes || []).filter((n) => n?.imageId != null)
    .map((n) => [Number(n.imageId), { ...n }]));
  const working = new Map([...before].map(([id, n]) => [id, { ...n }]));
  const fresh = [...(placed || [])].filter((p) => p?.imageId != null)
    .sort((a, b) => Number(a.imageId) - Number(b.imageId));

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

  const freshBySource = new Map();
  for (const p of fresh) {
    const node = working.get(Number(p.imageId));
    const key = strictSourceKey(node);
    if (!key) continue;
    if (!freshBySource.has(key)) freshBySource.set(key, []);
    freshBySource.get(key).push(node);
  }

  for (const [key, additions] of [...freshBySource].sort(([a], [b]) => a.localeCompare(b))) {
    const originalVisible = [...before.values()].filter((n) => n.visible !== false);
    const groupIds = [...new Set(originalVisible.map((n) => n.groupId).filter(Boolean))].sort();
    let members = null;
    let groupId = null;
    for (const gid of groupIds) {
      const list = originalVisible.filter((n) => n.groupId === gid)
        .sort((a, b) => (a.groupPos ?? 0) - (b.groupPos ?? 0) || a.imageId - b.imageId);
      if (list.length >= 2 && list.every((n) => strictSourceKey(n) === key)) {
        members = [...list, ...additions.filter((n) => !list.some((m) => m.imageId === n.imageId))];
        groupId = gid;
        break;
      }
    }
    if (!members) {
      const anchor = originalVisible
        .filter((n) => !n.groupId && strictSourceKey(n) === key)
        .sort((a, b) => a.imageId - b.imageId)[0];
      members = [...(anchor ? [anchor] : []), ...additions]
        .filter((n, i, all) => all.findIndex((m) => m.imageId === n.imageId) === i);
      if (members.length < 2) continue;
      groupId = nextGroupId([...working.values()], members[0].imageId);
    }
    members.forEach((member, pos) => {
      const updated = { ...working.get(member.imageId), groupId, groupPos: pos };
      working.set(member.imageId, updated);
      remember(updated);
    });
  }

  for (const p of fresh) remember(working.get(Number(p.imageId)));
  const byId = (a, b) => a.imageId - b.imageId;
  return { rows: [...affected.values()].sort(byId), undoRows: [...undo.values()].sort(byId) };
}

/** Turn one freshly generated/pinned lot into a single strip.
 * Different checkpoints are intentional here: Pin all represents one
 * generation action. Existing groups are never reused, so separate runs stay
 * as separate strips. */
export function groupPinnedBatchTogether({ nodes = [], placed = [] } = {}) {
  const before = new Map((nodes || []).filter((n) => n?.imageId != null)
    .map((n) => [Number(n.imageId), { ...n }]));
  const fresh = [...(placed || [])].filter((p) => p?.imageId != null)
    .sort((a, b) => Number(a.imageId) - Number(b.imageId));
  const groupId = fresh.length >= 2
    ? nextGroupId([...(nodes || []), ...fresh], fresh[0].imageId)
    : null;
  const rows = fresh.map((p, pos) => {
    const id = Number(p.imageId);
    const old = before.get(id);
    return {
      imageId: id, x: p.x, y: p.y, w: p.w, h: p.h, visible: true,
      groupId, groupPos: groupId ? pos : null,
      image: p.image || old?.image,
    };
  });
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
  // depend on the order the API happened to list the run's cells in.
  const ordered = [...all].sort((a, b) => (
    String(sourceKey(a)).localeCompare(String(sourceKey(b))) || (a.id - b.id)));
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
    bandTop += BAND_GAP;

    const colW = size + TILE_GAP;
    const rowH = size + TILE_GAP;

    // One group per source checkpoint, ordered by where that source sits on the
    // board (left to right, then top to bottom) so the band reads in the same
    // order as the tree above it.
    const groups = new Map();
    for (const image of band) {
      const key = sourceKey(image);
      if (!groups.has(key)) {
        groups.set(key, { key, ax: anchorX(graph, image.record_id), images: [] });
      }
      groups.get(key).images.push(image);
    }
    const ordering = [...groups.values()].sort((a, b) => (a.ax - b.ax)
      || String(a.key).localeCompare(String(b.key)));

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
      for (const image of group.images.sort((a, b) => a.id - b.id)) {
        if (row >= COLUMN_ROWS) { col = takeColumn(col + 1); row = 0; }
        const box = { x: col * colW, y: bandTop + row * rowH, w: size, h: size };
        occupied.push(box);
        placed.push({ imageId: image.id, ...box, image });
        row += 1;
      }
    }
  }

  placed.sort((a, b) => a.imageId - b.imageId);
  return { size, placed, skipped };
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
