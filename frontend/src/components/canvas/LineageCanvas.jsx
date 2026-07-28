import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { buildLineageGraph, CARD_W } from '../../utils/lineageGraph';
import {
  LANE_HEADER_H, MAX_SCALE, MIN_SCALE,
  clampScale, clampView, fitView, initialView, panBy, pinchCenter, pinchDistance,
  stackLanes, viewTransform, zoomAt,
} from '../../utils/canvasLayout';
import { applyPlacement, pinSnapshot, toOverrideMap } from '../../utils/canvasPlacement';
import {
  clampImageBox, defaultImageSpot, imageNodeEdges, imageNodeExtent,
  openGeometry, visibleImageNodes,
} from '../../utils/canvasImageNodes';
import { DEPLOY_BAR_CLASS, DEPLOY_LEGEND } from '../../utils/checkpointDeployState';
import { GraphCard, CheckpointPill } from '../dataset/lineageNodes';
import { LineageEdgeDefs, LineageEdges } from '../dataset/lineageEdges';
import { noteBadge, toggleDiffSelection } from '../dataset/lineageDetail.js';
import { lineageImportPayload } from '../dataset/lineagePreview.js';
import { removeRunFromTree } from '../../utils/runDeletable.js';
import {
  canvasCheckpointKey, describeCanvasLaunch, isCanvasCheckpointSelected,
  pruneCanvasSelection, refreshCanvasSelection, toggleCanvasCheckpoint,
} from '../../utils/canvasGeneration';
import { apiFetch, postJson } from '../../api/fetchClient';
import LineageDetailPanel from '../dataset/LineageDetailPanel';
import LineageDiffPanel from '../dataset/LineageDiffPanel';
import CheckpointActionsPopover from '../dataset/CheckpointActionsPopover';
import PreviewLightbox from '../dataset/PreviewLightbox';
import GeneratedImageLightbox from '../shared/GeneratedImageLightbox';
import { clampPopoverToViewport, POPOVER_H, POPOVER_W } from '../dataset/checkpointPopover.js';
import { useCheckpointActions } from '../../hooks/useCheckpointActions';
import { useCanvasRun } from '../../hooks/useCanvasRun';
import { canvasRunDatasetIds, readyImageCount, runPinCandidates } from '../../utils/canvasRunResults';
import { isNodeControlTarget, nodePointerIntent } from '../../utils/canvasNodeChrome';
import {
  pinBatchAnnouncement, pinBatchPendingAcrossLanes, placeImageBatch,
} from '../../utils/canvasPinBatch';
import { cardClickAction, runGalleryTarget } from '../../utils/canvasCardClick';
import { loraFolderLabel } from '../../utils/checkpointBrowser';
import { runIdentityLabel } from '../../utils/runIdentity';
import CanvasGenerationPanel from './CanvasGenerationPanel';
import CanvasRunTracker from './CanvasRunTracker';
import CanvasImageNode from './CanvasImageNode';
import CheckpointGalleryPanel from '../shared/CheckpointGalleryPanel';
import { useToast } from '../common/Toast';
import { HelpBadge } from '../../help/HelpMode';

/* ◉ The LoRA Canvas surface — every selected dataset's genealogy on ONE board,
   with zoom and pan.

   It draws with the SAME card, pill and edge components as the graph embedded in
   a run's card (components/dataset/lineageNodes + lineageEdges); the geometry of
   each dataset's tree is still utils/lineageGraph.js. What is new here is only
   the surface: several trees stacked into lanes, and a viewport you can move.

   Slice 2 added direct manipulation: cards can be dragged, and where they land
   is remembered (utils/canvasPlacement.js + the canvas_node_position table).

   Slice 3 makes the board GENERATE. Ticking a pill adds that checkpoint to a
   run; picks may come from several lanes at once, which is the whole reason the
   board holds every dataset. The settings are not a canvas invention — the
   panel is the Test Studio's own RunSetupPanel on the Test Studio's own hooks
   (see hooks/useCanvasStudio), so the two screens are one implementation. Each
   pill also carries a × N badge opening the gallery of everything that
   checkpoint ever produced.

   Gestures: wheel (or trackpad pinch, which arrives as ctrl+wheel) zooms around
   the pointer; dragging the background pans; two fingers pinch-zoom; dragging a
   card moves it.

   ⚠️ Moving a card and moving the view are the SAME physical gesture. With a
   mouse, hit-testing settles it: the press either landed on a card or it did
   not. On touch there is nothing to hit-test with — a finger on a card could
   mean either — so the finger PANS by default and only starts moving the card
   after a long press, which is the gesture every touch UI already uses for
   "pick this up". Moving before the press completes cancels it, so a flick that
   happens to start on a card still scrolls the board. */

const ZOOM_STEP = 1.25;
// How long a finger must rest on a card before it picks it up. Long enough not
// to fire on a flick that starts on a card, short enough not to feel broken.
const LONG_PRESS_MS = 420;
// Screen pixels of travel before a press counts as a drag. Below it the gesture
// is still a click, so inspecting a run never depends on holding perfectly
// still — and a 2-px twitch must not write a position to the database.
const DRAG_SLOP = 4;

/** One dataset's title strip above its tree. Inside the zoomed world, so it
 *  scales with the board it labels — a lane whose name floated at a constant
 *  size would drift off its tree the moment you zoomed out. */
function LaneHeader({ lane }) {
  return (
    <div style={{ position: 'absolute', left: 0, top: lane.y, height: LANE_HEADER_H,
      width: Math.max(lane.width, CARD_W) }}
      className="flex items-center gap-2 overflow-hidden">
      <span className="truncate text-[0.8125rem] font-semibold text-content" title={lane.name}>
        {lane.name}
      </span>
      <span className="shrink-0 rounded-full border border-border bg-app/60 px-1.5 py-0.5 text-content-muted text-[0.5625rem] font-medium tabular-nums">
        {lane.runs} run{lane.runs === 1 ? '' : 's'}
      </span>
      {lane.status === 'loading' && (
        <span className="shrink-0 animate-pulse text-content-subtle text-[0.625rem]">loading…</span>
      )}
      {lane.status === 'error' && (
        <span className="shrink-0 text-amber-300 text-[0.625rem]" title={lane.error || ''}>
          could not load this dataset
        </span>
      )}
      {lane.status === 'ready' && !lane.height && (
        <span className="shrink-0 text-content-subtle text-[0.625rem]">no runs to draw</span>
      )}
    </div>
  );
}

/** One dataset's tree, drawn exactly as the in-card graph draws it. */
function LaneGraph({ lane, isLit, onHover, onNodeClick, diffRole, noteOf, liftedId,
  isPicked, onTogglePick, onOpenGallery, onOpenActions, onZoomPreview }) {
  const g = lane.graph;
  if (!g || !g.nodes.length) return null;
  return (
    <svg
      style={{ position: 'absolute', left: 0, top: lane.graphY }}
      className="lds-lgraph block overflow-visible"
      width={g.width} height={g.height}
      viewBox={`0 0 ${g.width} ${g.height}`}
      role="img"
      aria-label={`${lane.name}: lineage of ${g.nodes.length} run${g.nodes.length === 1 ? '' : 's'}`}>
      <LineageEdges edges={g.edges} isLit={isLit} />
      <g>
        {g.nodes.map((n) => (
          <foreignObject key={n.node.record_id}
            className="lds-gnode overflow-visible"
            x={n.x} y={n.y} width={CARD_W} height={n.cellH}
            onPointerEnter={() => onHover(n.node.record_id)}
            onPointerLeave={() => onHover(null, n.node.record_id)}>
            {/* data-canvas-node is the hit-test handle: the pointer handlers on
                the frame read the lane + run off it to know WHAT was grabbed,
                without every card needing its own listener. */}
            <div data-canvas-node="" data-dataset-id={lane.datasetId} data-record-id={n.node.record_id}
              style={{ position: 'relative', width: CARD_W, height: n.cellH,
                ...(liftedId === n.node.record_id
                  ? { filter: 'drop-shadow(0 6px 14px rgba(0,0,0,0.55))', opacity: 0.92 }
                  : null) }}
              className={liftedId === n.node.record_id ? 'lds-gnode-lifted' : undefined}>
              <GraphCard node={noteOf(n.node)} lit={isLit(n.node.record_id)}
                annotated={noteBadge(noteOf(n.node))}
                compareRole={diffRole(n.node.record_id)}
                onSelect={onNodeClick} />
              {n.checkpoints.map((p) => (
                <CheckpointPill key={`${p.step}-${p.filename ?? p.x}`}
                  pill={p} offX={p.x - n.x} offY={p.y - n.y}
                  selected={isPicked(lane.datasetId, n.node.record_id, p.step)}
                  preview={p.preview_status || p.preview_url || p.preview_count
                    ? { status: p.preview_status, url: p.preview_url,
                      count: p.preview_count || 0 } : null}
                  // The pill's BODY opens its actions — download, continue,
                  // deploy/undeploy, delete, details — the same popover the
                  // in-card graph has always had. It used to open the detail
                  // drawer instead, which is how the board ended up with a panel
                  // nobody asked for and no actions at all.
                  onOpen={(pill, e) => onOpenActions(lane, n.node, pill, e)}
                  onZoomPreview={onZoomPreview}
                  // A checkpoint still on disk is pickable even when it is not in
                  // ComfyUI yet: the launch button then offers to deploy it first.
                  selectable={p.present !== false}
                  onToggleSelect={() => onTogglePick(lane, n.node, p)}
                  onOpenGallery={() => onOpenGallery(n.node.record_id, p.step)} />
              ))}
            </div>
          </foreignObject>
        ))}
      </g>
    </svg>
  );
}

/** One lane's pinned images, plus the links back to the checkpoints that made
 *  them. The links are drawn with the SAME connector the tree uses for "this
 *  continued from that" (components/dataset/lineageEdges) -- the board already
 *  has a grammar for descent and a second one would only be a second thing to
 *  learn. Its NEUTRAL variant, not the trunk: a render is evidence about a
 *  checkpoint, not a step of the training lineage.
 *
 *  Its own <svg>, sized 1x1 and overflow-visible, because a pinned image may sit
 *  well outside the tree's box and the tree's <svg> is sized to the tree. */
function LaneImages({ lane, nodes, onGeometry, onClose, onOpen, boardScale }) {
  if (!nodes.length) return null;
  const edges = imageNodeEdges(nodes, lane.graph);
  return (
    <div style={{ position: 'absolute', left: 0, top: lane.graphY }}>
      <svg width="1" height="1" className="block overflow-visible" aria-hidden>
        <LineageEdges edges={edges} isLit={() => false} />
      </svg>
      {nodes.map((n) => (
        <CanvasImageNode key={n.imageId} node={n} datasetId={lane.datasetId}
          laneName={lane.name} onGeometry={onGeometry} onClose={onClose}
          onOpen={onOpen} boardScale={boardScale} />
      ))}
    </div>
  );
}

export default function LineageCanvas({ entries, positions, imageNodes, onPinLane,
  onSaveImageNodes, onTidyUp, onRefetchDataset }) {
  const toast = useToast();
  const frameRef = useRef(null);
  const [viewport, setViewport] = useState({ width: 0, height: 0 });
  const [view, setView] = useState({ scale: 1, tx: 0, ty: 0 });
  const [hoverId, setHoverId] = useState(null);
  const [openNode, setOpenNode] = useState(null);
  const [selectedForDiff, setSelectedForDiff] = useState([]);
  const [noteEdits, setNoteEdits] = useState({});
  const [deletedIds, setDeletedIds] = useState([]);

  // A gone run removed from the inspector disappears without a refetch. It is
  // taken out of the TREE and the lane is laid out again — NOT filtered out of
  // the finished graph, which would leave its edges hanging in mid-air pointing
  // at a card that is no longer there. removeRunFromTree is the same helper the
  // in-card graph uses, so children re-root the same way in both.
  const shown = useMemo(() => (entries || []).map((e) => {
    if (!e.tree) return { ...e, graph: null };
    const tree = deletedIds.reduce((t, id) => removeRunFromTree(t, id), e.tree);
    return { ...e, graph: buildLineageGraph(tree) };
  }), [entries, deletedIds]);

  // --- placement: the automatic tree ⊕ what the user moved ------------------
  // A drag is applied as a temporary override on top of the stored map, over a
  // BASELINE captured when the drag started. The baseline matters: dropping a
  // single override onto an otherwise-automatic lane would make that lane
  // "arranged" halfway through the gesture, and applyPlacement would treat every
  // other card as a new arrival to slide out of the way. Freezing the lane at
  // the moment the finger lands keeps the rest of the board perfectly still.
  const [drag, setDrag] = useState(null);   // {datasetId, recordId, x, y, baseline}

  const placed = useMemo(() => shown.map((e) => {
    if (!e.graph) return e;
    let ov = positions?.[e.datasetId] || {};
    if (drag && drag.datasetId === e.datasetId) {
      ov = { ...drag.baseline, [drag.recordId]: { x: drag.x, y: drag.y } };
    }
    return { ...e, graph: applyPlacement(e.graph, ov) };
  }), [shown, positions, drag]);

  // A lane that gained a run since it was arranged reports the new card's spot;
  // persisting it is what makes "a new run moves nothing" survive the NEXT
  // reload too. Guarded by a seen-set so a failed write cannot become a loop.
  const pinned = useRef(new Set());
  useEffect(() => {
    for (const lane of placed) {
      const pins = lane.graph?.pendingPins;
      if (!pins?.length) continue;
      const key = `${lane.datasetId}:${pins.map((p) => p.record_id).join(',')}`;
      if (pinned.current.has(key)) continue;
      pinned.current.add(key);
      onPinLane?.(lane.datasetId, pins);
    }
  }, [placed, onPinLane]);

  /* The images PINNED on the board. Same coordinate system as the cards
     (lane-local world units) and the same storage decision: server-side, next
     to canvas_node_position. A board whose cards follow the dataset from
     machine to machine while its pictures stay stuck in one browser is a board
     that is only half yours -- and you find that out on the day you change
     desk. Geometry mid-gesture is an override on top, exactly like a card drag. */
  const [imgDrag, setImgDrag] = useState(null);   // {datasetId,imageId,x,y,w,h}
  const imagesByLane = useMemo(() => {
    const out = {};
    for (const e of placed) {
      let list = visibleImageNodes(imageNodes?.[e.datasetId] || {});
      if (imgDrag && imgDrag.datasetId === e.datasetId) {
        list = list.map((n) => (n.imageId === imgDrag.imageId
          ? { ...n, x: imgDrag.x, y: imgDrag.y, w: imgDrag.w, h: imgDrag.h } : n));
      }
      out[e.datasetId] = list;
    }
    return out;
  }, [placed, imageNodes, imgDrag]);
  const imagesRef = useRef(imagesByLane);
  useEffect(() => { imagesRef.current = imagesByLane; }, [imagesByLane]);

  // A lane has to be big enough to hold its pinned pictures too, or Fit would
  // crop one off the board with no way back to it.
  const world = useMemo(() => stackLanes(placed.map((e) => {
    const ext = imageNodeExtent(imagesByLane[e.datasetId] || []);
    return {
      ...e,
      width: Math.max(e.graph?.width || 0, ext.width),
      height: Math.max(e.graph?.height || 0, ext.height),
    };
  })), [placed, imagesByLane]);

  // The latest placement, for the pointer handlers (which must not re-bind on
  // every board change just to read a card's current position).
  const placedRef = useRef(placed);
  useEffect(() => { placedRef.current = placed; }, [placed]);

  // Measure the frame. The board is fitted to it, so an unmeasured frame would
  // mean an invisible board on first paint.
  useEffect(() => {
    const el = frameRef.current;
    if (!el) return undefined;
    const measure = () => setViewport({ width: el.clientWidth, height: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Auto-fit until the user takes over. `touched` is what makes the canvas feel
  // like a board and not a slideshow: once you have zoomed or panned, a lane
  // finishing its load must NOT yank your view back to a fit.
  const touched = useRef(false);
  const fitSignature = `${world.width}x${world.height}:${viewport.width}x${viewport.height}`;
  const lastFit = useRef('');
  useEffect(() => {
    if (touched.current || lastFit.current === fitSignature) return;
    if (!viewport.width || !viewport.height) return;
    lastFit.current = fitSignature;
    setView(initialView(world, viewport));
  }, [fitSignature, world, viewport]);

  const applyView = useCallback((next) => {
    touched.current = true;
    setView(clampView(next, world, viewport));
  }, [world, viewport]);

  const fitNow = useCallback(() => {
    touched.current = false;
    lastFit.current = '';
    if (viewport.width && viewport.height) setView(fitView(world, viewport));
  }, [world, viewport]);

  const zoomByButton = useCallback((factor) => {
    const anchor = { x: viewport.width / 2, y: viewport.height / 2 };
    applyView(zoomAt(view, factor, anchor));
  }, [applyView, view, viewport]);

  // The wheel listener is bound once per applyView identity; it reads the live
  // view through a ref so it never zooms from a stale one.
  const viewRef = useRef(view);
  useEffect(() => { viewRef.current = view; }, [view]);

  // Wheel zoom needs a NON-PASSIVE listener: React's onWheel is registered
  // passive, so preventDefault() there is ignored and the page scrolls behind
  // the board. Hence the manual native listener.
  useEffect(() => {
    const el = frameRef.current;
    if (!el) return undefined;
    const onWheel = (e) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const anchor = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      // A trackpad pinch arrives as ctrl+wheel with small deltas; a mouse wheel
      // as large ones. Normalising on the sign keeps both feeling the same.
      const factor = e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
      applyView(zoomAt(viewRef.current, factor, anchor));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [applyView]);

  // --- pointer gestures (pan with one, pinch with two) -----------------------
  const pointers = useRef(new Map());
  const pan = useRef(null);
  const pinch = useRef(null);

  const localPoint = (e) => {
    const rect = frameRef.current?.getBoundingClientRect();
    return { x: e.clientX - (rect?.left || 0), y: e.clientY - (rect?.top || 0) };
  };

  // --- node dragging ---------------------------------------------------------
  const dragRef = useRef(null);      // {datasetId, recordId, sx, sy, ox, oy, moved}
  // The pinned image being moved or resized: {datasetId,imageId,mode,sx,sy,box,cur}
  const imgRef = useRef(null);
  const longPress = useRef(null);    // touch: the pending pick-up
  const suppressClick = useRef(false);
  /* ⚠️ THE press that landed on a card: {datasetId, recordId, at, moved}.
     A card's own onClick NEVER FIRES on this board, which is why clicking a run
     used to do nothing at all. onPointerDown calls setPointerCapture on the
     FRAME (it has to — a drag that leaves the frame must keep receiving moves),
     and a captured pointer retargets the click that follows to the capturing
     element. So the click arrives on the frame, the card never hears it, and no
     amount of handler on the card can change that.

     The gesture therefore answers for itself: the press is remembered here, a
     travel past DRAG_SLOP marks it as a drag, and a release that never travelled
     is the click. Same for touch, where the capture is taken too. */
  const press = useRef(null);
  // When the gesture has just answered, ignore a DOM click that also arrives
  // (keyboard Enter, or a browser that did not retarget) — one press, one open.
  const answeredAt = useRef(0);
  // The live note edits, for the handlers (which must not re-bind on every save).
  const noteEditsRef = useRef(noteEdits);
  useEffect(() => { noteEditsRef.current = noteEdits; }, [noteEdits]);

  /* What a press on a run card DOES. One place for both entry points — the
     gesture (the real one, see `press`) and the DOM click that still arrives for
     a keyboard Enter. ⇧ Shift picks the run for the two-run compare; a plain
     press opens its gallery: every image it made, grouped by step, with its
     notes and the settings it trained with. */
  const runCardGesture = useCallback((node, shiftKey) => {
    if (!node) return;
    if (shiftKey) {
      setSelectedForDiff((sel) => toggleDiffSelection(sel, node.record_id));
      return;
    }
    setOpenCk(null);
    setGallery(runGalleryTarget(noteEditsRef.current[node.record_id] || node));
  }, []);

  const cancelLongPress = () => {
    if (longPress.current) { clearTimeout(longPress.current); longPress.current = null; }
  };

  /** Pick a card up: freeze its lane as it looks RIGHT NOW, then follow the
   *  pointer. `origin` is the screen point the gesture started from. */
  const beginDrag = useCallback((datasetId, recordId, origin) => {
    const lane = placedRef.current.find((l) => l.datasetId === datasetId);
    const n = lane?.graph?.nodes.find((x) => x.node.record_id === recordId);
    if (!n) return;
    dragRef.current = { datasetId, recordId, sx: origin.x, sy: origin.y,
      ox: n.x, oy: n.y, x: n.x, y: n.y, moved: false };
    setDrag({ datasetId, recordId, x: n.x, y: n.y,
      baseline: toOverrideMap(pinSnapshot(lane.graph, recordId, n.x, n.y)) });
    pan.current = null;
  }, []);

  /** Write one pinned image's geometry back. Applied to the screen first by the
   *  page and sent afterwards, like a card position: a picture must follow the
   *  finger at the speed of the finger, and a failed write heals on the next
   *  gesture. `visible: false` is the CLOSE -- the row and its geometry stay,
   *  which is what makes "re-open it exactly where I closed it" possible. */
  const saveImage = useCallback((datasetId, node, box, visible = true) => {
    onSaveImageNodes?.(datasetId, [{
      image_id: node.imageId, ...clampImageBox(box), visible, image: node.image,
    }]);
  }, [onSaveImageNodes]);

  const beginImage = useCallback((datasetId, imageId, mode, origin) => {
    const node = (imagesRef.current[datasetId] || []).find((n) => n.imageId === imageId);
    if (!node) return false;
    const box = { x: node.x, y: node.y, w: node.w, h: node.h };
    // `cur` is the live box, kept on the gesture itself: reading it back off the
    // rendered lane at pointerup would race the last frame of the drag.
    imgRef.current = { datasetId, imageId, mode, sx: origin.x, sy: origin.y,
      box, cur: box, moved: false, node };
    setImgDrag({ datasetId, imageId, ...box });
    pan.current = null;
    return true;
  }, []);

  const onPointerDown = useCallback((e) => {
    suppressClick.current = false;
    press.current = null;
    // A press on a pill is an inspection, never a drag or a pan.
    if (e.target.closest?.('.lds-ckpill-wrap')) return;
    /* A pinned image's own buttons (close, open) answer for themselves.
       Returning WITHOUT capturing the pointer is the point: a captured pointer
       retargets the click that follows to the frame, and the button would never
       hear it -- the same trap the run cards had to work around. The rule lives
       in utils/canvasNodeChrome so it is testable and cannot be lost in a
       rewrite of this handler. */
    if (isNodeControlTarget(e.target)) return;
    const imgEl = e.target.closest?.('[data-canvas-image]');
    if (imgEl) {
      const dsId = Number(imgEl.dataset.datasetId);
      const imageId = Number(imgEl.dataset.imageId);
      const at0 = localPoint(e);
      // The resize corner is hit-tested BEFORE the pan/drag decision, on every
      // pointer type: a finger landing on a 28-px corner handle can only mean
      // one thing, so it must not have to wait out a long press.
      const resizing = nodePointerIntent(e.target, e.pointerType) === 'resize';
      if (resizing || e.pointerType !== 'touch') {
        frameRef.current?.setPointerCapture?.(e.pointerId);
        if (beginImage(dsId, imageId, resizing ? 'resize' : 'move', at0)) return;
      } else {
        // Touch, on the picture itself: the board pans until a LONG PRESS picks
        // the node up -- the board's existing answer to "a finger on a node
        // could mean either".
        pointers.current.set(e.pointerId, at0);
        frameRef.current?.setPointerCapture?.(e.pointerId);
        pan.current = { ...at0, tx: viewRef.current.tx, ty: viewRef.current.ty };
        longPress.current = setTimeout(() => {
          longPress.current = null;
          beginImage(dsId, imageId, 'move', at0);
        }, LONG_PRESS_MS);
        return;
      }
    }
    const card = e.target.closest?.('[data-canvas-node]');
    if (card) {
      press.current = { datasetId: Number(card.dataset.datasetId),
        recordId: Number(card.dataset.recordId), at: localPoint(e), moved: false };
    }
    // A press on the bare board dismisses an open popover. A press on a card
    // does not: its own click decides (open another one, or toggle this one
    // shut), and closing here first would make that click reopen it.
    if (!card) setOpenCk(null);
    const at = localPoint(e);
    if (card && e.pointerType !== 'touch') {
      // Mouse / pen: the press landed on a card, so it IS the card that moves.
      frameRef.current?.setPointerCapture?.(e.pointerId);
      beginDrag(Number(card.dataset.datasetId), Number(card.dataset.recordId), at);
      return;
    }
    pointers.current.set(e.pointerId, at);
    frameRef.current?.setPointerCapture?.(e.pointerId);
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      pinch.current = { dist: pinchDistance(a, b), scale: viewRef.current.scale };
      pan.current = null;
      press.current = null;            // a second finger makes this a pinch
      cancelLongPress();
    } else if (pointers.current.size === 1) {
      pan.current = { ...at, tx: viewRef.current.tx, ty: viewRef.current.ty };
      if (card) {
        // Touch on a card: pan for now, pick the card up if the finger stays.
        const dsId = Number(card.dataset.datasetId);
        const recId = Number(card.dataset.recordId);
        longPress.current = setTimeout(() => {
          longPress.current = null;
          beginDrag(dsId, recId, at);
        }, LONG_PRESS_MS);
      }
    }
    frameRef.current?.classList.add('is-grabbing');
  }, [beginDrag, beginImage]);

  const onPointerMove = useCallback((e) => {
    // A press that travels is a drag or a pan, never a click — whichever of the
    // two branches below ends up handling it.
    if (press.current && !press.current.moved) {
      const p = localPoint(e);
      if (Math.hypot(p.x - press.current.at.x, p.y - press.current.at.y) >= DRAG_SLOP) {
        press.current.moved = true;
      }
    }
    const gi = imgRef.current;
    if (gi) {
      const p = localPoint(e);
      const s = clampScale(viewRef.current.scale);
      const dx = (p.x - gi.sx) / s;
      const dy = (p.y - gi.sy) / s;
      if (!gi.moved && Math.hypot(p.x - gi.sx, p.y - gi.sy) < DRAG_SLOP) return;
      gi.moved = true;
      const box = gi.mode === 'resize'
        ? clampImageBox({ ...gi.box, w: gi.box.w + dx, h: gi.box.h + dy })
        : clampImageBox({ ...gi.box, x: gi.box.x + dx, y: gi.box.y + dy });
      gi.cur = box;
      setImgDrag({ datasetId: gi.datasetId, imageId: gi.imageId, ...box });
      return;
    }
    const d = dragRef.current;
    if (d) {
      const p = localPoint(e);
      const scale = clampScale(viewRef.current.scale);
      const dx = (p.x - d.sx) / scale;
      const dy = (p.y - d.sy) / scale;
      if (!d.moved && Math.hypot(p.x - d.sx, p.y - d.sy) < DRAG_SLOP) return;
      d.moved = true;
      d.x = Math.max(0, d.ox + dx);
      d.y = Math.max(0, d.oy + dy);
      setDrag((cur) => (cur ? { ...cur, x: d.x, y: d.y } : cur));
      return;
    }
    if (!pointers.current.has(e.pointerId)) return;
    const prev = pointers.current.get(e.pointerId);
    pointers.current.set(e.pointerId, localPoint(e));
    // A finger that travels is scrolling the board, not picking a card up.
    if (longPress.current && prev) {
      const p = localPoint(e);
      if (Math.hypot(p.x - prev.x, p.y - prev.y) > DRAG_SLOP) cancelLongPress();
    }
    if (pointers.current.size >= 2 && pinch.current) {
      const [a, b] = [...pointers.current.values()];
      const dist = pinchDistance(a, b);
      if (!pinch.current.dist) return;
      const target = clampScale(pinch.current.scale * (dist / pinch.current.dist));
      applyView(zoomAt(viewRef.current, target / clampScale(viewRef.current.scale),
        pinchCenter(a, b)));
      return;
    }
    if (!pan.current) return;
    const p = localPoint(e);
    applyView(panBy({ ...viewRef.current, tx: pan.current.tx, ty: pan.current.ty },
      p.x - pan.current.x, p.y - pan.current.y));
  }, [applyView]);

  const endPointer = useCallback((e) => {
    cancelLongPress();
    const gi = imgRef.current;
    if (gi) {
      imgRef.current = null;
      // Only a gesture that actually MOVED writes: a tap on a pinned picture
      // must not quietly re-save the same coordinates.
      if (gi.moved) saveImage(gi.datasetId, gi.node, gi.cur);
      setImgDrag(null);
      pointers.current.delete(e.pointerId);
      frameRef.current?.releasePointerCapture?.(e.pointerId);
      if (pointers.current.size === 0) frameRef.current?.classList.remove('is-grabbing');
      return;
    }
    const d = dragRef.current;
    if (d) {
      dragRef.current = null;
      // Only a gesture that actually MOVED writes anything: a plain click on a
      // card must stay a click, and must not turn its lane into an arranged one
      // behind the user's back.
      if (d.moved) {
        suppressClick.current = true;
        const lane = placedRef.current.find((l) => l.datasetId === d.datasetId);
        if (lane?.graph) onPinLane?.(d.datasetId, pinSnapshot(lane.graph, d.recordId, d.x, d.y));
      }
      setDrag(null);
    }
    // A press on a card that never travelled IS the click — see `press` above
    // for why the card's own onClick cannot be relied on here.
    const p = press.current;
    press.current = null;
    if (p && !p.moved && !d?.moved) {
      const node = placedRef.current
        .find((l) => l.datasetId === p.datasetId)?.graph?.nodes
        .find((x) => x.node.record_id === p.recordId)?.node;
      if (node) {
        answeredAt.current = Date.now();
        runCardGesture(node, e.shiftKey);
      }
    }
    pointers.current.delete(e.pointerId);
    frameRef.current?.releasePointerCapture?.(e.pointerId);
    if (pointers.current.size < 2) pinch.current = null;
    if (pointers.current.size === 0) {
      pan.current = null;
      frameRef.current?.classList.remove('is-grabbing');
    }
  }, [onPinLane, runCardGesture, saveImage]);

  // --- inspection / compare (identical rules to the in-card graph) -----------
  const nodeById = useMemo(() => {
    const m = new Map();
    for (const e of shown) for (const n of (e.graph?.nodes || [])) m.set(n.node.record_id, n.node);
    return m;
  }, [shown]);

  const ancestors = useMemo(() => {
    const m = new Map();
    for (const e of shown) {
      for (const [id, set] of (e.graph?.ancestorsOf || new Map())) m.set(id, set);
    }
    return m;
  }, [shown]);

  const litNodes = useMemo(() => {
    const s = new Set();
    if (hoverId != null) {
      s.add(hoverId);
      for (const a of (ancestors.get(hoverId) || [])) s.add(a);
    }
    return s;
  }, [hoverId, ancestors]);
  const isLit = useCallback((id) => litNodes.has(id), [litNodes]);

  const onHover = useCallback((id, leaving) => {
    if (id == null) setHoverId((cur) => (cur === leaving ? null : cur));
    else setHoverId(id);
  }, []);

  // The open actions popover: { lane, node, pill, anchor } | null. `pill` is null
  // when a run CARD was clicked — the same popover, with only its run-level rows.
  const [openCk, setOpenCk] = useState(null);
  // 🖼 The open gallery: {recordId, step} for a checkpoint pill, or
  // {kind:'run', recordId, node} for a whole run card. Declared here because the
  // card-click handler below opens it.
  const [gallery, setGallery] = useState(null);
  const closePopover = useCallback(() => setOpenCk(null), []);
  const [bigPreview, setBigPreview] = useState(null);
  const zoomPreview = useCallback((url, step) => setBigPreview({ url, step }), []);

  const onOpenActions = useCallback((lane, node, pill, e) => {
    const anchor = { x: e?.clientX ?? 0, y: e?.clientY ?? 0 };
    setOpenCk((cur) => (cur && cur.node.record_id === node.record_id
      && (cur.pill?.step ?? null) === (pill?.step ?? null)
      ? null                                   // clicking the same target closes it
      : { lane, node, pill: pill || null, anchor }));
  }, []);

  /* The DOM path into the same gesture — keyboard Enter/Space on a focused card,
     and any browser that does NOT retarget the click to the capturing frame. The
     pointer path (endPointer) is the one that fires in practice, so this guards
     against answering twice; and a click that is only the tail of a drop is
     swallowed, because a rearrangement of the board must never end with a panel
     the user did not ask for. Which of the three it is comes from
     utils/canvasCardClick — testable, unlike this file. */
  const onNodeClick = useCallback((node, e) => {
    const dragged = suppressClick.current;
    if (dragged) suppressClick.current = false;
    if (Date.now() - answeredAt.current < 400) return;   // the gesture answered
    const action = cardClickAction({ dragged, shiftKey: !!(e && e.shiftKey) });
    if (action === 'ignored') return;
    runCardGesture(node, action === 'compare');
  }, [runCardGesture]);

  const diffRole = useCallback((id) => {
    const i = selectedForDiff.indexOf(id);
    return i === 0 ? 'A' : i === 1 ? 'B' : null;
  }, [selectedForDiff]);

  // --- generation: the checkpoints ticked on the board -----------------------
  // A LIST, not a set: the first pick anchors the settings panel's dataset, so
  // the order the user clicked in is meaningful (see utils/canvasGeneration).
  const [picks, setPicks] = useState([]);
  const [panelOpen, setPanelOpen] = useState(false);

  const isPicked = useCallback(
    (dsId, recId, step) => isCanvasCheckpointSelected(picks, dsId, recId, step), [picks]);

  const onTogglePick = useCallback((lane, node, pill) => {
    setPicks((cur) => toggleCanvasCheckpoint(cur, {
      datasetId: lane.datasetId,
      datasetName: lane.name,
      recordId: node.record_id,
      step: pill.step,
      family: node.train_type,
      deployed: pill.testable === true,
      // What create_comparison_run validates against: the name of the COPY in
      // ComfyUI, not the run-folder save.
      filename: pill.deployed_filename || null,
      // Kept with the pick so "deploy, then generate" needs no second lookup —
      // the exact body the flat checkpoint list sends.
      importPayload: lineageImportPayload(node, pill),
    }));
    setPanelOpen(true);
  }, []);

  // Picks whose pill is no longer on the board (its dataset was unticked, its run
  // deleted) are dropped: a launch must never fire on a checkpoint the user can
  // no longer see.
  const liveKeys = useMemo(() => {
    const s = new Set();
    for (const e of placed) {
      for (const n of (e.graph?.nodes || [])) {
        for (const p of n.checkpoints) {
          s.add(canvasCheckpointKey(e.datasetId, n.node.record_id, p.step));
        }
      }
    }
    return s;
  }, [placed]);
  useEffect(() => {
    setPicks((cur) => {
      const next = pruneCanvasSelection(cur, liveKeys);
      return next.length === cur.length ? cur : next;
    });
  }, [liveKeys]);

  /* Deploy the picks that are not in ComfyUI yet, then hand back the SAME picks
     with their fresh deployed state. Announced by the button before it runs
     ("Deploy 2 checkpoints, then generate") — nothing lands on the user's disk
     from a button that did not say so. A failure returns null and the launch is
     abandoned: half a comparison answers a different question than the one asked. */
  const handleDeploy = useCallback(async (needed) => {
    try {
      for (const e of needed) {
        if (!e.importPayload) {
          throw new Error(`Run #${e.recordId} step ${e.step} has no file to deploy`);
        }
        await postJson(`/api/dataset/${e.datasetId}/train/import`, e.importPayload);
      }
      toast.success(`${needed.length} checkpoint(s) deployed to ComfyUI`);
    } catch (err) {
      toast.error(err?.message || 'Deploy failed — nothing was generated');
      return null;
    }
    // Re-read the lanes so the freshly deployed pills come back testable, with
    // the deployed copy's real name.
    const ids = [...new Set(needed.map((e) => e.datasetId))];
    const fresh = new Map();
    for (const id of ids) {
      try {
        const tree = await onRefetchDataset?.(id);
        for (const node of (tree?.nodes || [])) {
          for (const c of (node.checkpoints || [])) {
            fresh.set(canvasCheckpointKey(id, node.record_id, c.step),
              { deployed: c.testable === true, filename: c.deployed_filename || null });
          }
        }
      } catch { /* the import already happened server-side; the pick keeps its state */ }
    }
    const refreshed = refreshCanvasSelection(picks, (e) => fresh.get(
      canvasCheckpointKey(e.datasetId, e.recordId, e.step)) || null);
    // Write it back to the BOARD's selection, not only to the launch in flight.
    // Without this the chips kept reading "to deploy" after a successful deploy
    // and the button kept offering to do it again — a lie about what is on disk.
    setPicks(refreshed);
    return refreshed;
  }, [picks, onRefetchDataset, toast]);

  const launchVerdict = describeCanvasLaunch(picks);

  /* --- checkpoint actions: THE shared ones ---------------------------------
     Deploying and deleting from the board run the exact routes, payloads and
     confirmation the in-card graph runs (hooks/useCheckpointActions). A lane is
     re-read afterwards so the pill stops making a claim about the disk that
     stopped being true — a just-deployed pill flips to ✓ Deployed, a deleted
     save's pill disappears.

     The ★ pin travels WITH the lane (the dataset index publishes
     `best_settings_loras`), so the pin handed to the hook is the one belonging to
     the lane whose popover is open — a board spanning ten datasets must never
     warn with another dataset's pin. Closed popover → null, and the hook simply
     falls back to the plain wording. */
  const onCheckpointChanged = useCallback(
    async (datasetId) => { await onRefetchDataset?.(datasetId); }, [onRefetchDataset]);
  const { importing, deleting, deployCheckpoint, deleteCheckpoint } = useCheckpointActions({
    onChanged: onCheckpointChanged,
    bestSettingsLora: openCk?.lane?.bestSettingsLoras || null,
  });
  const handleDeployCheckpoint = useCallback(async (node, pill) => {
    if (await deployCheckpoint(openCk?.lane?.datasetId ?? null, node, pill)) setOpenCk(null);
  }, [deployCheckpoint, openCk]);
  const handleDeleteCheckpoint = useCallback(async (node, pill) => {
    if (await deleteCheckpoint(openCk?.lane?.datasetId ?? null, node, pill)) setOpenCk(null);
  }, [deleteCheckpoint, openCk]);

  /* --- the generation in flight, owned by the BOARD -------------------------
     Not by the settings panel: closing that panel used to destroy the run id, so
     a launch could only be watched at the moment it was fired. */
  const tracker = useCanvasRun();
  const trackerTargets = tracker.targets;
  // New images = new × N badges and new thumbnails on the pills, which come from
  // the LINEAGE, not from the run. Without this re-read the board looked exactly
  // as it did before the launch until a full reload — the images were there, and
  // nowhere to be seen.
  const seenReady = useRef(0);
  useEffect(() => {
    const n = readyImageCount(tracker.run.data);
    if (n <= seenReady.current) return;
    seenReady.current = n;
    for (const id of canvasRunDatasetIds(trackerTargets)) {
      Promise.resolve(onRefetchDataset?.(id)).catch(() => { /* the poll retries */ });
    }
  }, [tracker.run.data, trackerTargets, onRefetchDataset]);

  const noteOf = useCallback((node) => noteEdits[node.record_id] || node, [noteEdits]);
  const handleNodeChanged = useCallback((updated) => {
    setNoteEdits((m) => ({ ...m, [updated.record_id]: updated }));
    setOpenNode((cur) => (cur && cur.record_id === updated.record_id ? updated : cur));
  }, []);
  const handleNodeDeleted = useCallback((recordId) => {
    setDeletedIds((ids) => (ids.includes(recordId) ? ids : [...ids, recordId]));
    setOpenNode(null);
  }, []);

  /* Pin a generated image onto the board -- from its gallery, or from the
     lightbox that opens over it. An image pinned BEFORE comes back exactly
     where and how big it was left (openGeometry); one that never was lands
     beside the card that produced it, sliding down past the pins already
     there. */
  const [pinnedZoom, setPinnedZoom] = useState(null);
  const handlePinImage = useCallback((img) => {
    const dsId = img?.dataset_id;
    if (dsId == null || img?.id == null) return;
    const map = imageNodes?.[dsId] || {};
    const lane = placedRef.current.find((l) => l.datasetId === dsId);
    const geo = openGeometry(map, img.id,
      defaultImageSpot(lane?.graph, img.record_id, img.step, visibleImageNodes(map)));
    onSaveImageNodes?.(dsId, [{ image_id: img.id, ...geo, visible: true, image: img }]);
  }, [imageNodes, onSaveImageNodes]);

  /* 📌 Pin ALL of a finished run's images, in one click.
     A lot spanning four checkpoints used to mean opening four galleries and
     pinning five pictures one by one — the board's own generation, and the
     board was the last place it landed.

     What each image IS (its prompt, seed, sampler, the checkpoint that made it)
     is read from the gallery route, one request per checkpoint the run was
     launched on, so the node carries exactly the payload every other gallery
     publishes rather than a second shape rebuilt from the run's cells. The
     geometry is decided by utils/canvasPinBatch — the part where "nothing may
     ever overlap anything" is a tested property and not a hope. */
  const [pinAllState, setPinAllState] = useState(null); // {busy}|{said, undo}
  const pinCandidates = useMemo(
    () => runPinCandidates(tracker.run.data), [tracker.run.data]);
  const pinPending = useMemo(
    () => pinBatchPendingAcrossLanes(pinCandidates, imageNodes).pending,
    [pinCandidates, imageNodes]);

  const handlePinAll = useCallback(async () => {
    const wanted = new Set(pinPending.map((c) => c.id));
    if (!wanted.size) return;
    setPinAllState({ busy: true });
    // One read per checkpoint the run was launched on. Failures are per-target:
    // a checkpoint whose gallery cannot be read costs its own pictures, not the
    // whole gesture.
    const byLane = new Map();
    await Promise.all(trackerTargets.map(async (t) => {
      try {
        const d = await apiFetch(`/api/train/checkpoint/${t.recordId}/${t.step}/images`);
        for (const img of (d?.images || [])) {
          if (!wanted.has(img.id)) continue;
          const dsId = img.dataset_id ?? t.datasetId;
          if (!byLane.has(dsId)) byLane.set(dsId, []);
          byLane.get(dsId).push(img);
        }
      } catch { /* reported by the count below: fewer pinned than asked for */ }
    }));

    let placedTotal = 0;
    let skippedTotal = 0;
    const undo = [];
    for (const [dsId, images] of byLane) {
      const lane = placedRef.current.find((l) => l.datasetId === dsId);
      const laneMap = imageNodes?.[dsId] || {};
      const res = placeImageBatch({
        graph: lane?.graph,
        existing: imagesRef.current[dsId] || [],
        images,
        remembered: laneMap,
      });
      if (!res.placed.length) continue;
      placedTotal += res.placed.length;
      skippedTotal += res.skipped.length;
      onSaveImageNodes?.(dsId, res.placed.map((p) => ({
        image_id: p.imageId, x: p.x, y: p.y, w: p.w, h: p.h,
        visible: true, image: p.image,
      })));
      // What Undo has to put back: these rows, closed again, at the geometry
      // they had BEFORE (a picture that had been closed keeps the spot it was
      // closed at, so undoing does not quietly rewrite it).
      undo.push({ datasetId: dsId, rows: res.placed.map((p) => {
        const was = laneMap[p.imageId];
        return { image_id: p.imageId, visible: false, image: p.image,
          x: was?.x ?? p.x, y: was?.y ?? p.y, w: was?.w ?? p.w, h: was?.h ?? p.h };
      }) });
    }
    const missing = wanted.size - placedTotal - skippedTotal;
    setPinAllState({
      said: pinBatchAnnouncement({
        placed: new Array(placedTotal),
        skipped: new Array(skippedTotal + Math.max(0, missing)),
      }),
      undo,
    });
  }, [pinPending, trackerTargets, imageNodes, onSaveImageNodes]);

  /* The way back. Pinning thirty pictures with one tap and offering no way out
     would be the board rearranging itself on the user's behalf; this closes
     exactly the nodes that click opened, and nothing else. It is the ordinary
     close (`visible: false`), so anything pinned by hand before is untouched
     and every geometry promise still holds. */
  const handleUndoPinAll = useCallback(() => {
    for (const { datasetId, rows } of (pinAllState?.undo || [])) {
      onSaveImageNodes?.(datasetId, rows);
    }
    setPinAllState(null);
  }, [pinAllState, onSaveImageNodes]);

  // Closing KEEPS the geometry -- that is the whole promise. Only `visible` flips.
  const handleCloseImage = useCallback((node) => {
    const dsId = node?.image?.dataset_id;
    if (dsId == null) return;
    onSaveImageNodes?.(dsId, [{
      image_id: node.imageId, x: node.x, y: node.y, w: node.w, h: node.h,
      visible: false, image: node.image,
    }]);
  }, [onSaveImageNodes]);

  // The keyboard path into the same write (arrows / +- on a focused node), so
  // moving and resizing are not mouse-only gestures.
  const handleImageGeometry = useCallback((node, box) => {
    const dsId = node?.image?.dataset_id;
    if (dsId == null) return;
    onSaveImageNodes?.(dsId, [{
      image_id: node.imageId, ...clampImageBox(box), visible: true, image: node.image,
    }]);
  }, [onSaveImageNodes]);

  const pct = Math.round(clampScale(view.scale) * 100);
  const empty = !world.lanes.length;
  // Has anything on the visible board been moved? Drives ✦ Tidy up: a button
  // that clears nothing should say so by being disabled, not by doing nothing.
  const arranged = shown.some((e) => Object.keys(positions?.[e.datasetId] || {}).length > 0);

  return (
    <>
      {/* The edge gradients + glow, defined ONCE for the whole page: every lane's
          <svg> references them by id (see lineageEdges.jsx). */}
      <svg width="0" height="0" aria-hidden className="absolute"><LineageEdgeDefs /></svg>

      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <div className="flex items-center gap-1">
          <button type="button" onClick={() => zoomByButton(1 / ZOOM_STEP)}
            disabled={view.scale <= MIN_SCALE + 1e-9}
            title="Zoom out" aria-label="Zoom out"
            className="flex h-9 w-9 items-center justify-center rounded-md border border-border bg-app/60 text-content-muted hover:text-content disabled:opacity-40">−</button>
          <span className="min-w-[3.25rem] text-center text-content-muted text-[0.6875rem] tabular-nums">{pct}%</span>
          <button type="button" onClick={() => zoomByButton(ZOOM_STEP)}
            disabled={view.scale >= MAX_SCALE - 1e-9}
            title="Zoom in" aria-label="Zoom in"
            className="flex h-9 w-9 items-center justify-center rounded-md border border-border bg-app/60 text-content-muted hover:text-content disabled:opacity-40">+</button>
        </div>
        <button type="button" onClick={fitNow}
          title="Fit the whole board in view"
          className="flex h-9 items-center rounded-md border border-border bg-app/60 px-3 text-content-muted text-[0.6875rem] font-semibold hover:text-content">
          Fit
        </button>
        {/* The way out of an arrangement that got away from you. Twenty runs
            later a hand-tidied board can be a knot, and "move them all back by
            hand" is not an answer — this drops every remembered position and
            hands the board to the automatic tree again. */}
        <button type="button" onClick={onTidyUp} disabled={!arranged}
          title={arranged
            ? 'Forget every moved card and rebuild the automatic tree'
            : 'Nothing has been moved yet'}
          className="flex h-9 items-center gap-1 rounded-md border border-border bg-app/60 px-3 text-content-muted text-[0.6875rem] font-semibold hover:text-content disabled:opacity-40">
          <span aria-hidden>✦</span> Tidy up
        </button>
        <HelpBadge topic="canvas-arrange" />
        {/* 🎨 The board's own launch button. It carries the pick count so the
            settings panel can be closed without losing sight of what is queued
            up — at 400 px the panel covers the board, and closing it is normal. */}
        <button type="button" onClick={() => setPanelOpen((v) => !v)}
          aria-pressed={panelOpen}
          title={picks.length
            ? `${picks.length} checkpoint(s) picked — open the run settings`
            : 'Tick checkpoints on the board, then set the run up here'}
          className={'flex h-9 items-center gap-1 rounded-md border px-3 text-[0.6875rem] font-semibold '
            + (picks.length
              ? 'border-indigo-400/60 bg-indigo-500/15 text-indigo-100 '
              : 'border-border bg-app/60 text-content-muted hover:text-content ')}>
          <span aria-hidden>🎨</span> Generate
          {picks.length > 0 && (
            <span className="rounded-full bg-indigo-500/40 px-1.5 tabular-nums">{picks.length}</span>
          )}
        </button>
        {/* The colour key. A colour with no legend is a guess, and this one
            answers the question asked most often on this board: "which of these
            can I generate from RIGHT NOW?". Each state carries a shape as well
            as a colour (filled disc vs hollow ring), because roughly one man in
            twelve reads red and green alike and the theme is dark graphite.
            It renders from utils/checkpointDeployState, the same source the
            pills read, so the key cannot drift from what it explains. */}
        <span data-testid="canvas-deploy-legend"
          className="flex items-center gap-2 text-content-subtle text-[0.625rem]">
          {DEPLOY_LEGEND.map((l) => (
            <span key={l.tone} className="flex items-center gap-1 whitespace-nowrap">
              {/* The swatch is the pill's OWN bar class, so the key is drawn by
                  the thing it explains and cannot drift from it. */}
              <span aria-hidden className={`inline-block h-3 w-0 ${DEPLOY_BAR_CLASS[l.tone]}`} />
              {l.label}
            </span>
          ))}
        </span>
        <span className="ml-auto hidden text-content-subtle text-[0.625rem] lg:inline">
          Drag a run to move it · drag the background to pan · wheel to zoom · click a run for all its images, notes and settings · click a checkpoint for its actions · tick a checkpoint’s <span aria-hidden>✓</span> to generate from it · <span className="font-semibold">⇧ Shift-click</span> two runs to compare - pin an image from its gallery to put it ON the board
        </span>
        {selectedForDiff.length > 0 && (
          <button type="button" onClick={() => setSelectedForDiff([])}
            className="rounded-md border border-amber-400/50 bg-amber-500/10 px-2 py-1 text-amber-100 text-[0.625rem]">
            Clear compare ({selectedForDiff.length})
          </button>
        )}
      </div>

      {/* 🎨 The generation in flight, ON the board. Visible with the settings
          panel closed, and after a reload — which is the whole point: a launch
          you can only watch at the second you fired it is a launch you cannot
          come back to. Its finished state names where the images went. */}
      <CanvasRunTracker
        run={tracker.run.data} targets={trackerTargets}
        onStop={() => tracker.run.cancel?.()}
        onResume={() => tracker.run.resume?.()}
        onOpenPanel={() => setPanelOpen(true)}
        onOpenResult={(t) => setGallery({ recordId: t.recordId, step: t.step })}
        pinCount={pinPending.length}
        pinBusy={!!pinAllState?.busy}
        pinSaid={pinAllState?.said || ''}
        onPinAll={handlePinAll}
        onUndoPinAll={pinAllState?.undo?.length ? handleUndoPinAll : null}
        onDismiss={() => { setPinAllState(null); tracker.forget(); }} />

      <div
        ref={frameRef}
        data-testid="lora-canvas-frame"
        // select-none: shift-click is the compare gesture, and shift-click is ALSO
        // the browser's extend-selection — without this, comparing two runs paints
        // half the board blue.
        className="lds-canvas-frame relative h-[65vh] min-h-[320px] w-full select-none touch-none overflow-hidden rounded-xl border border-border bg-app/40"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endPointer}
        onPointerCancel={endPointer}>
        {empty ? (
          <div className="flex h-full items-center justify-center px-6 text-center text-content-subtle text-[0.8125rem]">
            No dataset selected — pick one in the filter above to put its runs on the board.
          </div>
        ) : (
          <div style={{ position: 'absolute', left: 0, top: 0,
            width: Math.max(world.width, 1), height: Math.max(world.height, 1),
            transform: viewTransform(view), transformOrigin: '0 0' }}>
            {world.lanes.map((lane) => (
              <div key={lane.datasetId}>
                <LaneHeader lane={lane} />
                <LaneImages lane={lane} nodes={imagesByLane[lane.datasetId] || []}
                  onGeometry={handleImageGeometry} onClose={handleCloseImage}
                  onOpen={(n) => setPinnedZoom(n.image)}
                  boardScale={clampScale(view.scale)} />
                <LaneGraph lane={lane} isLit={isLit} onHover={onHover}
                  onNodeClick={onNodeClick} diffRole={diffRole} noteOf={noteOf}
                  liftedId={drag && drag.datasetId === lane.datasetId ? drag.recordId : null}
                  isPicked={isPicked} onTogglePick={onTogglePick}
                  onOpenActions={onOpenActions} onZoomPreview={zoomPreview}
                  onOpenGallery={(recordId, step) => setGallery({ recordId, step })} />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ◉ THE checkpoint actions — the very component the in-card graph draws,
          floated over the board instead of inside an <svg>. Fixed and at a
          CONSTANT size: a popover that scaled with the board would be unreadable
          at the zoom levels the board is actually useful at, and it must never
          be clipped by the frame it hangs over. clampPopoverToViewport keeps it
          inside the window on a 400-px screen, flipping above the click when
          there is no room below and narrowing rather than pushing the page
          sideways. */}
      {openCk && (() => {
        const box = clampPopoverToViewport(openCk.anchor, {
          width: typeof window !== 'undefined' ? window.innerWidth : 400,
          height: typeof window !== 'undefined' ? window.innerHeight : 800,
        }, { width: POPOVER_W, height: POPOVER_H });
        return (
          <div style={{ position: 'fixed', left: box.left, top: box.top, width: box.width, zIndex: 60 }}>
            <CheckpointActionsPopover
              node={openCk.node} pill={openCk.pill}
              runLabel={runIdentityLabel(openCk.node)}
              folderLabel={loraFolderLabel(openCk.node.train_type)}
              // The board has no resume flow of its own: continuing a run is the
              // Runs hub's (cloud) or the dataset panel's (local) gesture, each
              // with its own dialog. Rather than a button that would go nowhere,
              // the row says where the gesture lives.
              continueReason="Continue from here: open this run from the Runs page (cloud) or the dataset’s Checkpoints panel (local)"
              importing={importing} deleting={deleting}
              onDeploy={handleDeployCheckpoint}
              onDelete={handleDeleteCheckpoint}
              onDetails={(node) => setOpenNode(node)}
              onClose={closePopover} />
          </div>
        );
      })()}

      {/* One drawer at a time: two picked runs → the compare diff, otherwise the
          single-run inspector. Both are the EXISTING panels, hosted unchanged —
          and because the board holds several datasets, the compare now works
          across them for free. */}
      {selectedForDiff.length === 2 ? (
        <LineageDiffPanel
          a={nodeById.get(selectedForDiff[0])}
          b={nodeById.get(selectedForDiff[1])}
          onClose={() => setSelectedForDiff([])} />
      ) : (
        <LineageDetailPanel node={openNode} onClose={() => setOpenNode(null)}
          onNodeChanged={handleNodeChanged} onNodeDeleted={handleNodeDeleted} />
      )}

      {/* 🎨 The run settings — the Test Studio's own panel, on the Test Studio's
          own hooks. Only the checkpoints differ: they are the ticked pills. */}
      {panelOpen && (
        <CanvasGenerationPanel
          selection={picks}
          onToggle={(entry) => setPicks((cur) => toggleCanvasCheckpoint(cur, entry))}
          onClear={() => setPicks([])}
          onDeploy={handleDeploy}
          tracker={tracker}
          onClose={() => setPanelOpen(false)} />
      )}

      {/* 🖼 Everything one checkpoint — or one whole run — ever produced. Deleting
          from it re-reads the affected lanes: the pills carry a results COUNT and
          a thumbnail, and without this the board keeps advertising images that no
          longer exist. `onDetails` hands the run over to the drawer that owns
          note EDITING, so the panel can show notes without owning a second
          editor for them. */}
      <CheckpointGalleryPanel target={gallery} onClose={() => setGallery(null)}
        onPin={handlePinImage}
        onDetails={(node) => { setGallery(null); setOpenNode(node); }}
        onDeleted={(ids) => (ids || []).forEach((id) => {
          Promise.resolve(onRefetchDataset?.(id)).catch(() => { /* the poll retries */ });
        })} />

      {/* 🔍 A pill's preview, full-screen. The thumbnail was already clickable on
          the board and did nothing at all — the host passed no handler. */}
      <PreviewLightbox target={bigPreview} onClose={() => setBigPreview(null)} />

      {/* A PINNED image's full record: every setting it was made with, its
          prompt, and the copy buttons. The node on the board is the picture;
          the facts stay one click away rather than crammed onto a thumbnail. */}
      <GeneratedImageLightbox img={pinnedZoom} alt="Pinned generated image"
        onClose={() => setPinnedZoom(null)} />

      {/* An untouched board with picks waiting: say so, because the settings panel
          may be closed and the ✓ boxes are small. Also the only place the
          mixed-family refusal is visible without opening the panel. */}
      {!panelOpen && picks.length > 0 && (
        <p className={'mt-2 rounded-lg border px-3 py-1.5 text-[0.6875rem] '
          + (launchVerdict.blocked
            ? 'border-amber-400/40 bg-amber-500/10 text-amber-100 '
            : 'border-indigo-400/40 bg-indigo-500/10 text-indigo-100 ')}>
          {picks.length} checkpoint{picks.length > 1 ? 's' : ''} picked
          {launchVerdict.reason ? ` — ${launchVerdict.reason}` : ''}
        </p>
      )}
    </>
  );
}
