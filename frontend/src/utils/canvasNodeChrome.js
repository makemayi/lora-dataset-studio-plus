/* The CHROME of a pinned image on the ◉ LoRA Canvas — its ✕, its 🔍 and its
   resize corner — and why those three need arithmetic of their own.

   Everything else on this board is measured in BOARD units, and that is right:
   a card, a pill and a picture all live in one zoomable coordinate system, so
   they keep their relationships at every scale. Controls are the exception, and
   the bug report is the proof. The ✕ of a pinned image "did not work" on a
   phone. The handler was fine — the board has refused to start a drag from a
   node's own button for as long as the feature has existed. What was not fine
   was the TARGET: the header button is ~16 board units, the board is routinely
   read at 24-65 %, and 16 × 0.65 ≈ 10 screen pixels — a quarter of the ~40 px a
   finger actually lands on. The button was not broken, it was unhittable, and
   the 🔍 sitting right beside it meant a near miss opened the full-screen view
   instead of closing the node. That reads exactly like "the cross does nothing".

   So a control is sized in SCREEN space: it is counter-scaled by the board's own
   zoom, which keeps it the same size under the finger whatever the board is
   doing. Two bounds keep that honest:

     • never SMALLER than its board size (zooming in must not shrink the chrome
       into a speck sitting on a huge picture);
     • never so large that it eats the picture it decorates — at 10 % zoom a
       constant-size button would be wider than the thumbnail it sits on, so the
       counter-scale is capped as a fraction of the node's own width.

   JSX-free, because `node --test` cannot parse JSX and this is the part worth
   testing: the component only applies the number. */

const CHROME_BASE = 28;        // one control's own size, in board units
// The whole corner cluster (🔍 + ✕, their gap and their padding) at scale 1…
const CLUSTER_UNITS = 64;
// …and the share of the node it is never allowed to exceed. The cap is on the
// CLUSTER, not on one button: capping each button separately let two of them
// side by side cover almost the entire width of a small tile.
const MAX_CLUSTER_FRACTION = 0.7;

/**
 * The counter-scale to apply to a pinned node's controls at a given board zoom.
 *
 * 1 means "leave it alone". 2 means "draw it twice as big in board units", which
 * at 50 % zoom is exactly its nominal size on screen.
 */
export function chromeScale(boardScale, nodeW) {
  const s = Number(boardScale);
  const w = Number(nodeW);
  if (!Number.isFinite(s) || s <= 0) return 1;
  const wanted = 1 / s;                       // constant size on screen
  const cap = Number.isFinite(w) && w > 0
    ? Math.max(1, (w * MAX_CLUSTER_FRACTION) / CLUSTER_UNITS)
    : Infinity;
  return Math.min(Math.max(1, wanted), cap);
}

/** What that control measures on screen, once counter-scaled — the number the
 *  proof is about ("is it big enough for a finger?"). */
export function chromeScreenSize(boardScale, nodeW) {
  const s = Number(boardScale);
  if (!Number.isFinite(s) || s <= 0) return CHROME_BASE;
  return CHROME_BASE * chromeScale(s, nodeW) * s;
}

/**
 * Is this event target one of a pinned node's own BUTTONS (✕, 🔍)?
 *
 * THE guard, extracted out of the pointer handler so it cannot be lost the next
 * time that handler is rewritten. A pointerdown here must not be captured by the
 * frame and must not arm the long press: a captured pointer retargets the click
 * that follows to the frame, and the button never hears it — which is a second,
 * quieter way for a ✕ to "do nothing", on every pointer type this time.
 *
 * The RESIZE corner is deliberately not one of these. It has no click of its
 * own — it is a gesture — so it WANTS the capture, and only its long-press
 * exemption sets it apart from the picture. See nodePointerIntent.
 *
 * Takes anything with `closest` (a DOM node, or a stub in a test).
 */
export function isNodeControlTarget(target) {
  if (!target || typeof target.closest !== 'function') return false;
  return !!target.closest('[data-canvas-image] button');
}

/**
 * What a pointerdown inside a pinned node means:
 *   'control' — a button: hands off entirely, no capture, no long press;
 *   'resize'  — the corner handle: resize immediately, on every pointer type;
 *   'move'    — mouse/pen on the picture: pick it up;
 *   'press'   — touch on the picture: pan for now, pick it up on a long press.
 *
 * One function instead of a chain of `if`s inside the handler, so the rule
 * "a control is never a gesture" is a thing a test can hold on to.
 */
export function nodePointerIntent(target, pointerType) {
  if (!target || typeof target.closest !== 'function') return 'press';
  if (target.closest('[data-canvas-image] button')) return 'control';
  if (target.closest('[data-canvas-image-resize]')) return 'resize';
  return pointerType === 'touch' ? 'press' : 'move';
}
