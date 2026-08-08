/* 🧬 GENERATION PROVENANCE — the only kind of descent on this board that has
   several parents at once.

   Three notions of "came from" live here and they must not be conflated:

     · TRAINING lineage (`tree.edges`)  — which run continued which run;
     · image → pill                     — which save produced this picture,
                                          one parent, drawn per lane already
                                          (utils/canvasImageNodes.imageNodeEdges);
     · GENERATION PROVENANCE, this file — a 🧬 blend loads N LoRAs into one
                                          image, so that image descends from N
                                          pills at once, and they are routinely
                                          in DIFFERENT lanes: blending across
                                          datasets is the whole point of doing
                                          it from the board.

   That last fact decides everything about this module. A cross-lane edge cannot
   live in a lane's own <svg>, so these edges are computed in WORLD coordinates
   and drawn once, under the lanes. Under, not over: an edge is CONTEXT, not
   content. It must never cover a card or a picture, and — the lesson of the
   group-bar bug, where chrome and content fought over the pointer and a whole
   node became unusable — it must never take a click either. The layer that
   draws them is `pointer-events: none` and sits behind everything.

   ── What is NOT drawn, deliberately ─────────────────────────────────────────
   The HEAD LoRA of a stack keeps the ordinary image → pill edge it already had.
   Drawing it again here would put two connectors on one pair.

   And a source that cannot be placed on the board draws NOTHING. Its run may
   have been deleted, its dataset unticked in the filter, or the blend may
   predate the day members started recording their origin — in which case the
   answer is genuinely unknown. `unresolved` counts those per image so the 🧬
   badge can say "1 of 2 sources isn't on the board". A board whose job is
   showing what came from where must never invent an edge to a plausible
   neighbour; a missing line the user is told about beats a confident wrong one.

   JSX-free so `node --test` can exercise it. */

import { edgePath } from './lineageGraph.js';

/** The stacked members recorded on an image, or [] — never a throw.
 *  `extra_loras` reaches the board as the raw JSON string the cell was written
 *  with, so a hand-edited or truncated value must degrade to "no provenance",
 *  exactly like a blend that predates the field. */
export function stackMembersOf(image) {
  const raw = image?.extra_loras;
  if (!raw) return [];
  let list = raw;
  if (typeof raw === 'string') {
    try { list = JSON.parse(raw); } catch { return []; }
  }
  if (!Array.isArray(list)) return [];
  return list.filter((e) => e && typeof e === 'object' && e.combined);
}

/** Where a (record, step) pill sits in WORLD units, or null when it is not on
 *  the board at all. The lane carries its own offset; a pill's coordinates are
 *  lane-local, exactly as `imageNodeEdges` reads them. */
export function pillWorldBox(lanes, datasetId, recordId, step) {
  if (recordId == null || step == null) return null;
  const lane = (lanes || []).find((l) => Number(l.datasetId) === Number(datasetId));
  if (!lane) return null;
  const card = (lane.graph?.nodes || []).find((c) => c.node?.record_id === recordId);
  const pill = card?.checkpoints?.find((p) => p.step === step);
  if (!pill) return null;
  return { x: (lane.x || 0) + pill.x, y: (lane.graphY || 0) + pill.y, w: pill.w, h: pill.h };
}

/**
 * The provenance edges of every blended picture on the board.
 *
 * `nodes` — the pictures as they are DRAWN, in lane-local units, each carrying
 *           the lane it belongs to: { imageId, x, y, w, h, datasetId, image }.
 * `lanes` — [{ datasetId, x, graphY, graph }], the board's lanes with their
 *           world offsets and their laid-out graphs.
 *
 * Returns { edges, unresolved }:
 *   edges      — one per PLACED source beyond the head, in world units, with the
 *                same `d` connector grammar the rest of the board uses;
 *   unresolved — Map(imageId → { placed, total }), the honest counter the 🧬
 *                badge reads. Only images that actually are blends appear.
 */
export function blendEdgesFor(nodes, lanes) {
  const edges = [];
  const unresolved = new Map();
  for (const n of (nodes || [])) {
    const members = stackMembersOf(n.image);
    if (!members.length) continue;                 // not a blend: nothing to add
    const lane = (lanes || []).find((l) => Number(l.datasetId) === Number(n.datasetId));
    const x2 = (lane?.x || 0) + n.x;
    const y2 = (lane?.graphY || 0) + n.y + n.h / 2;
    let placed = 0;
    for (const m of members) {
      const box = pillWorldBox(lanes, m.dataset_id, m.record_id, m.step);
      if (!box) continue;                          // unknown or off the board
      placed += 1;
      const x1 = box.x + box.w;
      const y1 = box.y + box.h / 2;
      edges.push({
        parentId: `ck:${m.dataset_id}:${m.record_id}:${m.step}`,
        childId: `img:${n.imageId}`,
        x1, y1, x2, y2, d: edgePath(x1, y1, x2, y2),
        // The board's neutral connector, like the image → pill link: a render is
        // evidence about a checkpoint, never a step of the training lineage.
        onSpine: false, superseded: false, blend: true,
      });
    }
    // The HEAD is counted in the total because the user counts it: a stack of
    // two reads "2 sources", not "1 extra source".
    unresolved.set(n.imageId, { placed: placed + 1, total: members.length + 1 });
  }
  return { edges, unresolved };
}

/** What the 🧬 badge says when some sources could not be placed, or null when
 *  they all were. One sentence, in the user's terms — "sources", not
 *  "members", and never a number without its total. */
export function blendSourcesNote(entry) {
  if (!entry) return null;
  const { placed, total } = entry;
  if (!(total > 0) || placed >= total) return null;
  const missing = total - placed;
  return `${missing} of ${total} sources ${missing > 1 ? 'are' : 'is'} not on the board`;
}
