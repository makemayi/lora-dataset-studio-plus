import { nudgeImageNode } from '../../utils/canvasImageNodes';
import { chromeScale } from '../../utils/canvasNodeChrome';
import { imageFactsLine } from '../../utils/generatedImageFacts';

/* 🖼 One generated image, pinned ON the board.

   The board compares checkpoints and the thing being compared is the picture —
   so a picture that can only be seen one at a time, in a modal, cannot be
   compared at all. Pinned, it becomes a node like any other: draggable,
   resizable, joined by an edge to the checkpoint pill that produced it (that
   edge is drawn by the lane, with the SAME connector a continuation uses — see
   utils/canvasImageNodes.imageNodeEdges).

   Closing is a ✕ on the node and it does NOT forget anything: the geometry is
   kept and re-opening the same image from its gallery puts it back exactly
   here, at exactly this size. That is the feature; the storage side of it is
   `visible: false` on the canvas_image_node row.

   ⚠️ Touch. Dragging a node and panning the board are the same physical
   gesture, and this file inherits the board's answer to that (LineageCanvas):
   with a mouse the press hit-tests onto the node and moves it; with a finger
   the board pans until a LONG PRESS picks the node up. The resize corner is the
   exception — it is hit-tested first on any pointer type, because a finger that
   lands on a 28-px corner handle can only mean one thing.

   ⚠️ …and the size of those controls is NOT a board size. Reported from a phone:
   "the cross does not close the preview". The handler was right — the board has
   always refused to start a gesture from a node's own button — but the button
   itself was ~16 board units on a board read at 65 %, so about ten pixels under
   a finger, with the 🔍 immediately beside it. The controls are therefore
   counter-scaled by the board zoom (utils/canvasNodeChrome.chromeScale) so they
   keep a constant size on screen, and they are laid out as one cluster in the
   corner rather than two glyphs crammed into a 12-px header row.

   ⌨ Keyboard. The node itself is focusable: arrows move it, Shift+arrows move
   it faster, +/− resize it, Esc closes it. Moving and resizing by mouse alone
   would put the whole feature out of reach of anyone who does not use one. The
   arithmetic is nudgeImageNode(), unit-tested; this file only routes keys. */

export default function CanvasImageNode({ node, datasetId, laneName, onGeometry,
  onClose, onOpen, boardScale = 1 }) {
  const img = node.image || {};
  const stepLabel = img.step == null ? 'step unknown' : `step ${img.step}`;
  const facts = imageFactsLine(img);
  // The controls are drawn in SCREEN space: counter-scaled by the board zoom so
  // a finger finds them at 24 % exactly as it does at 100 %.
  const k = chromeScale(boardScale, node.w);
  const chrome = { transform: `scale(${k})`, transformOrigin: 'top right' };

  const onKeyDown = (e) => {
    if (e.key === 'Escape') { e.stopPropagation(); onClose?.(node); return; }
    const next = nudgeImageNode(node, e.key, e.shiftKey);
    if (!next) return;                    // never swallow a key we do not handle
    e.preventDefault();
    e.stopPropagation();
    onGeometry?.(node, next);
  };

  return (
    <div
      // The hit-test handles. The frame's pointer handlers read the dataset and
      // the image off these rather than every node carrying its own listeners —
      // the same arrangement the run cards use (data-canvas-node).
      data-canvas-image=""
      data-dataset-id={datasetId}
      data-image-id={node.imageId}
      role="group"
      tabIndex={0}
      onKeyDown={onKeyDown}
      aria-label={`Pinned image from ${laneName || 'this dataset'}, ${stepLabel}. `
        + 'Arrow keys move it, plus and minus resize it, Escape closes it.'}
      title={`${stepLabel} · ${facts}`}
      style={{ position: 'absolute', left: node.x, top: node.y,
        width: node.w, height: node.h }}
      className="lds-canvas-image group flex flex-col overflow-hidden rounded-lg border border-indigo-400/40 bg-surface-overlay shadow-lg
                 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300">
      <header className="flex shrink-0 items-center gap-1 border-b border-border bg-app/70 px-1.5 py-0.5">
        <span className="min-w-0 flex-1 truncate text-content-muted text-[0.5625rem] font-semibold tabular-nums">
          {stepLabel}
        </span>
      </header>
      {/* The controls, as ONE cluster pinned to the node's corner and drawn at a
          constant SCREEN size. Out of the header's flow on purpose: counter-
          scaling something inside a 12-px row would either clip it or push the
          label off. Each target is 28 units square with air between them — two
          glyphs a pixel apart is how a miss on ✕ opened 🔍 instead. */}
      <div style={chrome}
        data-testid="canvas-image-controls"
        className="absolute right-0 top-0 z-10 flex items-start gap-1 rounded-bl-lg bg-app/85 p-0.5 backdrop-blur-sm">
        {/* Opens the full record — every setting, the prompt, the copy buttons.
            The node is the picture; the facts stay one click away rather than
            being crammed onto a thumbnail. */}
        <button type="button" onClick={(e) => { e.stopPropagation(); onOpen?.(node); }}
          title="Open this image full-screen with all its settings"
          aria-label={`Open ${stepLabel} full-screen`}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded text-content-subtle text-[0.75rem] leading-none hover:bg-app hover:text-content">🔍</button>
        {/* ✕ closes the node and REMEMBERS where it was. Re-pinning the same
            image from its gallery brings it back here, this size. */}
        <button type="button" onClick={(e) => { e.stopPropagation(); onClose?.(node); }}
          data-testid="canvas-image-close"
          title="Close this image — re-opening it from its gallery puts it back here, at this size"
          aria-label={`Close the pinned image at ${stepLabel}`}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded text-content-subtle text-[0.875rem] leading-none hover:bg-red-500/25 hover:text-content">✕</button>
      </div>
      <div className="relative min-h-0 flex-1 bg-black/30">
        <img src={img.url} alt={`Generated at ${stepLabel}`} draggable={false}
          className="h-full w-full select-none object-contain" />
      </div>
      {/* The resize corner. 28 px on purpose — a hairline handle is a desktop-only
          affordance, and this board is used on a phone. Hit-tested BEFORE the
          drag/pan decision, so a finger landing here always resizes. */}
      <span data-canvas-image-resize="" aria-hidden
        title="Drag to resize"
        style={{ position: 'absolute', right: 0, bottom: 0, width: 28, height: 28,
          transform: `scale(${k})`, transformOrigin: 'bottom right' }}
        className="cursor-nwse-resize touch-none rounded-tl-md border-l border-t border-indigo-400/40 bg-app/80 text-content-subtle after:absolute after:bottom-1 after:right-1 after:text-[0.625rem] after:content-['◢']" />
    </div>
  );
}
