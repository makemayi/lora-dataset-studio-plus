/* WHERE THE LIGHTBOX ACTIONS GO — and why it is not a media query.
 *
 * The bar used to be pinned at the bottom unconditionally. On a wide monitor a
 * portrait photo drawn with `object-contain` leaves two thirds of the width
 * black and the buttons still queued on one line under it: the actions pay rent
 * on the ONE axis the image is already short of.
 *
 * The rule is geometric, not a portrait/landscape label, because "portrait" is
 * not what makes a rail free. With `object-contain` the image is drawn at
 * `min(boxW / w, boxH / h)`. When it is HEIGHT-limited — when the leftover
 * width still fits it drawn at full height — taking width away costs the image
 * exactly nothing, while the bottom bar costs it height every time. So:
 *
 *     rail  ⟺  aspect (w/h)  ≤  (viewportWidth − railWidth) / viewportHeight
 *
 * That single inequality gets every case right without special-casing any of
 * them: a 832×1216 portrait on 1440×900 passes with room to spare (0.68 vs
 * 1.30) and gains back the bar's height; a 1216×832 landscape fails (1.46) and
 * keeps the bottom bar, because there is no side space to take; a square passes
 * on a 1440×900 monitor and fails on a 1280×1024 one, which is the honest
 * answer — squareness is not the question, the leftover column is.
 *
 * TWO GUARDS on top of the inequality:
 *  - a floor of `MIN_RAIL_VIEWPORT_PX` (Tailwind `lg`). Below it a 17rem rail
 *    plus a readable image column is a lie however the arithmetic comes out,
 *    and on a phone the bottom is the only option.
 *  - the labels stay words. The rail is sized for "✨ Upscale & improve", not
 *    for a column of mute glyphs: these actions rotate, recrop and spend GPU
 *    time, and an icon you have to hover is not a label.
 *
 * STABILITY is a first-class requirement here — a bar that changes side while
 * you aim at it is worse than the wasted space. Three mechanisms:
 *  - `PLACEMENT_HYSTERESIS`: entering the rail needs the inequality to hold
 *    with 12 % margin, leaving it needs it to fail by 12 %. Dragging a window
 *    edge across the threshold therefore cannot oscillate.
 *  - `RAIL_EXIT_VIEWPORT_PX`: the same dead band on the viewport floor, so
 *    1024 px is not a flip-flop point either.
 *  - `locked`: while a pixel edit is in flight the current placement is
 *    returned verbatim, whatever the geometry says.
 * And the intrinsic size is REMEMBERED per image id (below), so the decision is
 * already made the next time that image opens instead of being re-derived from
 * an `onLoad` the user can watch happen.
 */

/** Rail width in px — must stay in sync with `w-[17rem]` in the lightbox. */
export const RAIL_WIDTH_PX = 272;

/** No rail below this viewport width (Tailwind `lg`). */
export const MIN_RAIL_VIEWPORT_PX = 1024;

/** An existing rail survives down to here — dead band on the floor above. */
export const RAIL_EXIT_VIEWPORT_PX = 960;

/** Margin the inequality must clear to change the answer. */
export const PLACEMENT_HYSTERESIS = 0.12;

const positive = (n) => typeof n === 'number' && Number.isFinite(n) && n > 0;

/**
 * @param {object} input
 * @param {number} input.viewportWidth
 * @param {number} input.viewportHeight
 * @param {number} [input.imageWidth]   intrinsic px — unknown until the image loads
 * @param {number} [input.imageHeight]
 * @param {'rail'|'bottom'} [input.current]  placement in force right now
 * @param {boolean} [input.comparing]  side-by-side mode: both panes want the width
 * @param {boolean} [input.locked]     an action is running — do not move anything
 * @returns {'rail'|'bottom'}
 */
export function decideActionPlacement({
  viewportWidth,
  viewportHeight,
  imageWidth,
  imageHeight,
  current = 'bottom',
  comparing = false,
  locked = false,
} = {}) {
  const held = current === 'rail' ? 'rail' : 'bottom';
  // Comparison first, and it outranks the lock: entering it is a full relayout
  // the user asked for, and its two panes split the width the rail would take.
  if (comparing) return 'bottom';
  if (locked) return held;
  if (!positive(viewportWidth) || !positive(viewportHeight)) return 'bottom';
  // Unknown intrinsic size → the bottom bar, which fits every shape. Never a
  // guess: guessing is what makes the bar jump when the image finally paints.
  if (!positive(imageWidth) || !positive(imageHeight)) return 'bottom';

  const floor = held === 'rail' ? RAIL_EXIT_VIEWPORT_PX : MIN_RAIL_VIEWPORT_PX;
  if (viewportWidth < floor) return 'bottom';

  const aspect = imageWidth / imageHeight;
  const leftover = (viewportWidth - RAIL_WIDTH_PX) / viewportHeight;
  const margin = held === 'rail' ? 1 + PLACEMENT_HYSTERESIS : 1 - PLACEMENT_HYSTERESIS;
  return aspect <= leftover * margin ? 'rail' : 'bottom';
}

/* Intrinsic sizes measured once, keyed by image id. The grid thumbnails and the
 * lightbox request the SAME URL, so the browser usually has the bytes already;
 * remembering the ratio on top of that means reopening an image never replays
 * the bottom→rail commit. Module-level and unbounded on purpose: it is two
 * integers per image of one dataset, and it dies with the tab. */
const ratios = new Map();

/** Exported for tests only. */
export function _resetImageRatios() {
  ratios.clear();
}

/** Record what an <img> reported once it loaded. Zeroes are ignored. */
export function rememberImageRatio(key, imageWidth, imageHeight) {
  if (key === null || key === undefined) return;
  if (!positive(imageWidth) || !positive(imageHeight)) return;
  ratios.set(key, { imageWidth, imageHeight });
}

/** @returns {{imageWidth:number,imageHeight:number}|null} */
export function readImageRatio(key) {
  if (key === null || key === undefined) return null;
  return ratios.get(key) || null;
}
