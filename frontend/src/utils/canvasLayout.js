/* Pure geometry for the ◉ LoRA Canvas — the board that puts EVERY dataset's
   lineage on one surface.

   Framework-free (no JSX, no DOM) so `node --test` exercises the real arithmetic
   the screen uses: stacking the datasets into lanes, fitting the board into its
   frame, and the zoom/pan bookkeeping. The renderer lives in
   components/canvas/LineageCanvas.jsx and imports exactly these functions — it
   owns no second copy of the maths.

   Two coordinate systems, and mixing them is the classic canvas bug:
     • WORLD  — the board's own units. A dataset lane is laid out here by
                utils/lineageGraph.js, untouched and unaware of zoom.
     • SCREEN — pixels inside the frame. screen = world * scale + translate.
   Every function below states which one it takes.

   Slice 1 has no per-node placement: lanes stack in the order given, each at
   x = 0. Node dragging and its remembered positions are slice 2's Placement
   layer, which will sit BETWEEN lineageGraph.js and this file — this module
   only ever consumes finished per-lane sizes, so it does not have to change
   when that layer arrives. */

// A lane's title strip (dataset name + run count), above its graph.
export const LANE_HEADER_H = 34;
// Air between one dataset's board and the next. Generous on purpose: the whole
// point of the canvas is telling two datasets' genealogies apart at a glance.
export const LANE_GAP = 56;

// Zoom bounds. The floor is low enough that a library of a dozen trained
// datasets fits in one frame; the ceiling stops a stray pinch from blowing a
// 264-px card up to a wall.
export const MIN_SCALE = 0.1;
// 500% lets a generated image be inspected directly on the board. The former
// 250% ceiling was still too small after Fit had opened a large canvas at 10–50%.
export const MAX_SCALE = 5;

/** Clamp a scale into the usable range. NaN / nonsense degrades to 1 rather
 *  than blanking the board. */
export function clampScale(s) {
  const n = Number(s);
  if (!Number.isFinite(n)) return 1;
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, n));
}

/**
 * Stack per-dataset graphs into vertical lanes.
 *
 * `entries` — [{ datasetId, name, width, height, ... }] in the order they should
 * appear (the caller decides; the canvas keeps the filter's order). A lane with
 * no measurable graph still gets its header, so a dataset that is still loading
 * — or whose runs all vanished — keeps its place on the board instead of making
 * the lanes below it jump when it arrives.
 *
 * Returns { lanes, width, height } in WORLD units, where each lane carries
 * `y` (its header's top) and `graphY` (its graph's top).
 */
export function stackLanes(entries) {
  const list = Array.isArray(entries) ? entries : [];
  const lanes = [];
  let y = 0;
  let width = 0;
  for (const e of list) {
    const w = Math.max(0, Number(e?.width) || 0);
    const h = Math.max(0, Number(e?.height) || 0);
    lanes.push({ ...e, x: 0, y, graphY: y + LANE_HEADER_H, width: w, height: h });
    y += LANE_HEADER_H + h + LANE_GAP;
    width = Math.max(width, w);
  }
  // The trailing gap is not content: without this the board always carried one
  // lane-gap of dead space at the bottom, and "fit" wasted it on every zoom.
  const height = lanes.length ? y - LANE_GAP : 0;
  return { lanes, width, height };
}

/**
 * The view that shows the whole board inside `viewport`, centred.
 * `world` {width,height} and `viewport` {width,height} are both in their own
 * units; the result is { scale, tx, ty } with screen = world * scale + t.
 *
 * Never magnifies past 1 — a two-run lineage blown up to fill a 27" screen
 * looks broken, not helpful. An empty board or an unmeasured frame answers the
 * identity view instead of dividing by zero.
 */
export function fitView(world, viewport, { padding = 16 } = {}) {
  const ww = Math.max(0, Number(world?.width) || 0);
  const wh = Math.max(0, Number(world?.height) || 0);
  const vw = Math.max(0, Number(viewport?.width) || 0);
  const vh = Math.max(0, Number(viewport?.height) || 0);
  if (!ww || !wh || !vw || !vh) return { scale: 1, tx: padding, ty: padding };
  const availW = Math.max(1, vw - padding * 2);
  const availH = Math.max(1, vh - padding * 2);
  const scale = clampScale(Math.min(1, Math.min(availW / ww, availH / wh)));
  return {
    scale,
    tx: (vw - ww * scale) / 2,
    ty: (vh - wh * scale) / 2,
  };
}

// The smallest scale the board is allowed to OPEN at. A true fit is the right
// answer for the Fit button, but not for a first paint on a phone: a board of
// three datasets fits a 400-px frame at ~35 %, where a run card is four pixels
// tall and nothing is readable. Opening a little closer and letting the user
// scroll down beats opening on an unreadable overview.
export const INITIAL_MIN_SCALE = 0.45;

/**
 * The view the board OPENS at: the fit, but never below INITIAL_MIN_SCALE, and
 * ALWAYS top-aligned.
 *
 * Top-aligned even when the board is shorter than its frame. Vertically
 * centring a short board looked tidy on a desktop and was expensive on a phone:
 * the frame is sized from the viewport, not from the content, so a two-lane
 * board opened with a third of the frame as empty sky above the first lane —
 * a third of a 400-px screen spent on nothing before anything is legible.
 * Horizontal centring stays: the lanes are narrower than they are tall and
 * there is no equivalent waste.
 */
export function initialView(world, viewport, { padding = 16 } = {}) {
  const fit = fitView(world, viewport, { padding });
  const ww = Math.max(0, Number(world?.width) || 0);
  const wh = Math.max(0, Number(world?.height) || 0);
  const vw = Math.max(0, Number(viewport?.width) || 0);
  const vh = Math.max(0, Number(viewport?.height) || 0);
  if (!ww || !wh || !vw || !vh) return fit;
  const scale = clampScale(Math.min(1, Math.max(fit.scale, INITIAL_MIN_SCALE)));
  const w = ww * scale;
  return {
    scale,
    tx: w <= vw ? (vw - w) / 2 : padding,
    ty: padding,
  };
}

/** WORLD point under a SCREEN point, for the current view. */
export function toWorld(view, sx, sy) {
  const s = clampScale(view?.scale);
  return { x: (sx - (view?.tx || 0)) / s, y: (sy - (view?.ty || 0)) / s };
}

/**
 * Zoom by `factor` around a SCREEN anchor — the point the user is pointing at
 * stays exactly where it is, which is what makes wheel-zoom and pinch feel like
 * they grab the board rather than shove it. Clamping the scale must not move
 * the anchor either, so the translate is recomputed from the CLAMPED scale.
 */
export function zoomAt(view, factor, anchor) {
  const s0 = clampScale(view?.scale);
  const s1 = clampScale(s0 * (Number(factor) || 1));
  const ax = Number(anchor?.x) || 0;
  const ay = Number(anchor?.y) || 0;
  const w = toWorld({ ...view, scale: s0 }, ax, ay);
  return { scale: s1, tx: ax - w.x * s1, ty: ay - w.y * s1 };
}

/** Pan by a SCREEN delta. */
export function panBy(view, dx, dy) {
  return {
    scale: clampScale(view?.scale),
    tx: (Number(view?.tx) || 0) + (Number(dx) || 0),
    ty: (Number(view?.ty) || 0) + (Number(dy) || 0),
  };
}

/**
 * Keep the board reachable: after any pan or zoom, at least `keep` screen pixels
 * of content must remain inside the frame on each axis. Without this the board
 * can be flung off-screen by one careless swipe and the only way back is a
 * reload — the failure mode every hand-written canvas ships with first.
 */
export function clampView(view, world, viewport, { keep = 80 } = {}) {
  const scale = clampScale(view?.scale);
  const ww = (Math.max(0, Number(world?.width) || 0)) * scale;
  const wh = (Math.max(0, Number(world?.height) || 0)) * scale;
  const vw = Math.max(0, Number(viewport?.width) || 0);
  const vh = Math.max(0, Number(viewport?.height) || 0);
  if (!ww || !wh || !vw || !vh) return { scale, tx: Number(view?.tx) || 0, ty: Number(view?.ty) || 0 };
  const clamp1 = (t, content, frame) => {
    const margin = Math.min(keep, content);
    return Math.min(frame - margin, Math.max(margin - content, t));
  };
  return {
    scale,
    tx: clamp1(Number(view?.tx) || 0, ww, vw),
    ty: clamp1(Number(view?.ty) || 0, wh, vh),
  };
}

/** The CSS transform for a view. `transform-origin: 0 0` is assumed — the order
 *  (translate then scale) is the same one toWorld inverts. */
export function viewTransform(view) {
  const s = clampScale(view?.scale);
  return `translate(${Number(view?.tx) || 0}px, ${Number(view?.ty) || 0}px) scale(${s})`;
}

/** Distance between two pointers, for pinch zoom. */
export function pinchDistance(a, b) {
  const dx = (Number(a?.x) || 0) - (Number(b?.x) || 0);
  const dy = (Number(a?.y) || 0) - (Number(b?.y) || 0);
  return Math.hypot(dx, dy);
}

/** Midpoint of two pointers — the anchor a pinch zooms around. */
export function pinchCenter(a, b) {
  return {
    x: ((Number(a?.x) || 0) + (Number(b?.x) || 0)) / 2,
    y: ((Number(a?.y) || 0) + (Number(b?.y) || 0)) / 2,
  };
}
