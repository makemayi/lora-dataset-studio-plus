/**
 * 🖼🖼 The title bar of a group of pinned images: its grip, its ✕ and its
 * Export grid.
 *
 * ── Why this is not inside CanvasImageGroup any more ────────────────────────
 * The bar is drawn ABOVE the strip's box (`top: -barH`), on board space that
 * belongs to no node. Inside the group component it was an ordinary descendant
 * of one absolutely-positioned sibling among many, and `LaneImages` renders
 * those siblings with NO z-index — so any picture the board happened to place
 * over that strip painted on top of the bar and took its pointer events.
 *
 * That is not cosmetic: the bar carries ALL THREE of a group's affordances, so
 * one picture sitting above a strip left it impossible to move, impossible to
 * export and impossible to close at the same time. Measured on the real DOM: a
 * picture pinned flush above a two-image strip made 5 of 11 points sampled
 * along the bar hand the pointer to the picture.
 *
 * So the bars are drawn as a LAYER, after every picture and every strip — the
 * same answer, and the same `z-` idiom, that the merge hint in `LaneImages`
 * already uses. Chrome wins over content, always; the pictures keep their own
 * paint order among themselves.
 *
 * ── The invariant this file exists to keep ──────────────────────────────────
 * ✕ MUST BE REACHABLE. A node that can be neither moved nor removed is the
 * worst state the board can be in, and closing is the way out of every other
 * one. The render-order contract is asserted in tests/canvas-group-bar.test.mjs.
 *
 * It carries `data-canvas-group` and the same `data-dataset-id`/`data-anchor-id`
 * as the strip, because the frame's pointer handler reads the gesture's target
 * off exactly those — the bar moves the ANCHOR, whose box IS the strip's.
 */
import { groupBarHeight } from '../../utils/canvasNodeChrome';

export default function CanvasGroupBar({ group, datasetId, boardScale = 1,
  onCloseGroup, onExportGrid }) {
  const barH = groupBarHeight(boardScale, group.h);
  const count = group.members.length;
  const anchorId = group.members[0]?.node.imageId;

  return (
    <div
      data-canvas-group=""
      data-canvas-group-bar=""
      data-dataset-id={datasetId}
      data-anchor-id={anchorId}
      data-group-id={group.groupId}
      style={{ position: 'absolute', left: group.x, top: group.y - barH,
        width: group.w, height: barH }}
      title="Drag this bar to move the whole group"
      className="z-10 flex cursor-grab touch-none items-center gap-1 overflow-hidden rounded-t-md border border-b-0 border-indigo-400/40 bg-app/85 pl-1.5 backdrop-blur-sm">
      <span style={{ fontSize: Math.max(9, barH * 0.42) }}
        className="min-w-0 flex-1 truncate font-semibold text-content-muted tabular-nums">
        <span aria-hidden>⠿</span> {count} images
      </span>
      <button type="button"
        onClick={(e) => { e.stopPropagation(); onExportGrid?.(group); }}
        onPointerDown={(e) => e.stopPropagation()}
        data-testid="canvas-group-export-grid"
        style={{ height: barH, fontSize: Math.max(9, barH * 0.36) }}
        title={`Export these ${count} images as a grid`}
        aria-label={`Export grid from these ${count} images`}
        className="flex shrink-0 items-center gap-1 rounded px-1.5 font-semibold text-indigo-200 hover:bg-indigo-500/25 hover:text-white">
        <span aria-hidden>▦</span> Export grid
      </button>
      {/* ✕ on a GROUP closes N pictures at once, so it says N and says what
          happens to them. A destructive action has to be readable before it
          is pressed, not after. */}
      <button type="button"
        onClick={(e) => { e.stopPropagation(); onCloseGroup?.(group); }}
        data-testid="canvas-group-close"
        // The FULL bar height, not a fraction of it: at ✦ Fit on a 400-px
        // screen the bar is 26 px and four fifths of that is 21 — under the
        // floor a finger needs, which is the exact shape of the bug the
        // pinned-image ✕ already had once.
        style={{ width: barH, height: barH, fontSize: Math.max(9, barH * 0.42) }}
        title={`Close all ${count} images — the group is undone and each one goes back `
          + 'to its own size. Re-opening one from its gallery brings just that one back.'}
        aria-label={`Close this group of ${count} images`}
        className="flex shrink-0 items-center justify-center rounded leading-none text-content-subtle hover:bg-red-500/25 hover:text-content">
        ✕{count}
      </button>
    </div>
  );
}
