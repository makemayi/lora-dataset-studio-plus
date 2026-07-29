import { nudgeImageNode } from '../../utils/canvasImageNodes';
import { CLUSTER_UNITS, chromeScale } from '../../utils/canvasNodeChrome';
import { imageFactsLine } from '../../utils/generatedImageFacts';
import { useImageDownload } from '../../hooks/useImageDownload';

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

   ⬇ …and the cluster now has THREE controls, which is why it wraps. The cap
   that keeps it off the picture is spent on the cluster's WIDTH, so a third
   button drawn beside the others would have shrunk all of them by a third and
   walked the same bug back in. It goes on a second line instead, last, so 🔍
   and ✕ never move. What it downloads keeps its lineage in its NAME — dataset,
   run, step, seed — because this board is the only place that knows all four
   and a file called out_00042_.png in Downloads knows none of them.

   ⌨ Keyboard. The node itself is focusable: arrows move it, Shift+arrows move
   it faster, +/− resize it, Esc closes it. Moving and resizing by mouse alone
   would put the whole feature out of reach of anyone who does not use one. The
   arithmetic is nudgeImageNode(), unit-tested; this file only routes keys. */

export default function CanvasImageNode({ node, datasetId, laneName, onGeometry,
  onClose, onOpen, boardScale = 1, variant = 'node', box = null }) {
  const img = node.image || {};
  const stepLabel = img.step == null ? 'step unknown' : `step ${img.step}`;
  const facts = imageFactsLine(img);
  /* 🖼🖼 A MEMBER of a group draws the same picture with the same actions, in a
     box the strip decides (utils/canvasImageGroups.layoutImageNodes) instead of
     its own. Three things go away and nothing is added:

       • the frame and the rounding — that IS the request ("side by side with no
         border"), and a strip of framed tiles is a contact sheet, not one node;
       • the resize corner — a member has no size of its own to drag; the strip
         is resized as a whole, from its own corner;
       • the permanent header and control cluster — at rest the strip is just
         pictures. They come back on hover/focus of THIS picture and only this
         one, which is also how you can tell which ✕ you are about to press.

     Deliberately the SAME component and the same control cluster rather than a
     second set of buttons drawn by the group: anything the chrome gains next
     lands in a group for free. */
  const member = variant === 'member';
  const geom = box || node;
  // The controls are drawn in SCREEN space: counter-scaled by the board zoom so
  // a finger finds them at 24 % exactly as it does at 100 %.
  // geom.w, not node.w: a member's drawn width is the strip's tile, not the
  // box it carries for the day it leaves.
  const k = chromeScale(boardScale, geom.w);
  // ⚠️ maxWidth, not a guess: the cap chromeScale applies is spent on the
  // cluster's WIDTH, so a third control drawn BESIDE the other two would spend
  // 50 % more of it and shrink every target by a third — 20 px down to 15.7 px
  // at 24 % zoom, which is the unhittable-✕ bug walking back in. It wraps
  // instead, and ⬇ is last so 🔍 and ✕ never move.
  const chrome = { transform: `scale(${k})`, transformOrigin: 'top right',
    maxWidth: CLUSTER_UNITS };
  const dl = useImageDownload();
  // Revealed on hover/focus for a member, always on for a node of its own.
  const reveal = member
    ? ' opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100'
    : '';

  const onKeyDown = (e) => {
    if (e.key === 'Escape') { e.stopPropagation(); onClose?.(node); return; }
    // A member has no geometry of its own to nudge — the strip decides its box.
    // Swallowing the arrows anyway would silently break scrolling the page.
    if (member) return;
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
      aria-label={member
        ? `Pinned image from ${laneName || 'this dataset'}, ${stepLabel}, inside a group. `
          + 'Drag it off the group to take it out. Escape closes it.'
        : `Pinned image from ${laneName || 'this dataset'}, ${stepLabel}. `
          + 'Arrow keys move it, plus and minus resize it, Escape closes it.'}
      title={`${stepLabel} · ${facts}`}
      style={{ position: 'absolute', left: geom.x, top: geom.y,
        width: geom.w, height: geom.h }}
      className={'lds-canvas-image group flex flex-col overflow-hidden bg-surface-overlay '
        + (member
          // No frame and no rounding BETWEEN pictures — that is the request. The
          // inset ring on hover is what replaces it: it lights the picture you
          // are pointing at without ever drawing a rule between two of them.
          ? 'hover:ring-2 hover:ring-inset hover:ring-indigo-300/70 '
          : 'rounded-lg border border-indigo-400/40 shadow-lg ')
        + 'focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-indigo-300'}>
      {/* ⚠️ A member's label is an OVERLAY, not a row. Left in the flex flow it
          still reserved its height while invisible, the picture below it was
          then shorter than its tile, and `object-contain` answered with a dark
          band down each side — which is a border between two images, drawn by
          the very code that exists to remove it. */}
      <header className={(member
        ? 'absolute inset-x-0 top-0 z-10 flex items-center gap-1 bg-app/80 px-1.5 py-0.5 backdrop-blur-sm'
        : 'flex shrink-0 items-center gap-1 border-b border-border bg-app/70 px-1.5 py-0.5')
        + reveal}>
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
        className={'absolute right-0 top-0 z-10 flex flex-wrap items-start justify-end'
          + ' gap-1 rounded-bl-lg bg-app/85 p-0.5 backdrop-blur-sm' + reveal}>
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
        {/* ⬇ Keep this picture. LAST in the cluster, so it wraps onto its own
            line and 🔍 and ✕ stay at the pixel a hand already knows.
            The file lands under a name that still says where it came from —
            dataset, run, step, seed (services/gallery_download.py). That name
            is the ONLY carrier of the lineage: there is no sidecar, and a
            board's whole value is lost the moment two checkpoints' renders
            become two files called out_00042_.png. */}
        <button type="button" disabled={dl.busy}
          onClick={(e) => { e.stopPropagation(); dl.download(node.imageId); }}
          data-testid="canvas-image-download"
          title="Download this image — the file name keeps its dataset, run, step and seed"
          aria-label={`Download the image at ${stepLabel}`}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded text-content-subtle text-[0.75rem] leading-none hover:bg-app hover:text-content disabled:opacity-50">
          {dl.busy ? '…' : '⬇'}
        </button>
      </div>
      {/* A refusal has to be READABLE, and the node is small — so it is a strip
          across the bottom of the picture, counter-scaled like the buttons are,
          rather than a tooltip nobody hovers on a phone. It happens: the board
          lists rows whose file a resume or a trash sweep took off the disk. */}
      {dl.error && (
        <div role="alert" data-testid="canvas-image-download-error"
          onClick={(e) => { e.stopPropagation(); dl.clearError(); }}
          style={{ transform: `scale(${k})`, transformOrigin: 'bottom left' }}
          className="absolute bottom-0 left-0 z-10 max-w-full cursor-pointer rounded-tr-md bg-red-900/90 px-1 py-0.5 text-[0.5rem] leading-tight text-red-50">
          {dl.error}
        </div>
      )}
      <div className="relative min-h-0 flex-1 bg-black/30">
        <img src={img.url} alt={`Generated at ${stepLabel}`} draggable={false}
          className="h-full w-full select-none object-contain" />
      </div>
      {/* The resize corner. 28 px on purpose — a hairline handle is a desktop-only
          affordance, and this board is used on a phone. Hit-tested BEFORE the
          drag/pan decision, so a finger landing here always resizes.
          A group MEMBER has none: its width is its aspect ratio at the strip's
          height, so there is nothing about it to drag. The strip has one. */}
      {!member && (
      <span data-canvas-image-resize="" aria-hidden
        title="Drag to resize"
        style={{ position: 'absolute', right: 0, bottom: 0, width: 28, height: 28,
          transform: `scale(${k})`, transformOrigin: 'bottom right' }}
        className="cursor-nwse-resize touch-none rounded-tl-md border-l border-t border-indigo-400/40 bg-app/80 text-content-subtle after:absolute after:bottom-1 after:right-1 after:text-[0.625rem] after:content-['◢']" />
      )}
    </div>
  );
}
