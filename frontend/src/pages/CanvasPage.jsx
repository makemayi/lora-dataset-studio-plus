import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiFetch, del, putJson } from '../api/fetchClient';
import { buildLineageGraph } from '../utils/lineageGraph';
import {
  readSelection, resolveSelection, toggleSelection, writeSelection,
} from '../utils/canvasSelection';
import { toOverrideMap } from '../utils/canvasPlacement';
import { toImageNodeMap, visibleImageNodes } from '../utils/canvasImageNodes';
import { placeImageBatch } from '../utils/canvasPinBatch';
import CanvasDatasetFilter from '../components/canvas/CanvasDatasetFilter';
import LineageCanvas from '../components/canvas/LineageCanvas';
import { HelpBadge } from '../help/HelpMode';

/* ◉ LoRA Canvas — every dataset's training genealogy on one board.

   Until now a lineage graph was locked inside a single run's card: one dataset
   at a time, in a fixed frame. This page promotes it. Same trees, same cards,
   same edges — but all of them on one surface you can zoom and pan, with a
   filter to decide which datasets are on it.

   Loading is deliberately in TWO steps. The index (which datasets have runs) is
   a couple of grouped queries and arrives instantly, so the filter and the frame
   are usable straight away; each dataset's genealogy — which scans that run's
   saves on disk — is then fetched on its own and its lane fills in when it
   lands. One request for the whole library would have meant an empty screen for
   as long as the slowest disk scan took.

   Fetches are capped to a few at a time. A library of thirty trained datasets
   firing thirty concurrent disk-scanning requests would stall the very server
   that also has to answer whatever training or captioning is running. */

const MAX_PARALLEL_FETCHES = 3;

export default function CanvasPage() {
  const [index, setIndex] = useState({ status: 'loading', datasets: [], error: null });
  const [stored, setStored] = useState(() => readSelection(
    typeof localStorage !== 'undefined' ? localStorage : null));
  // dataset_id -> { status: 'loading'|'ready'|'error', tree, error }
  const [trees, setTrees] = useState({});
  const inflight = useRef(new Set());

  useEffect(() => {
    let alive = true;
    apiFetch('/api/train/canvas/datasets')
      .then((d) => { if (alive) setIndex({ status: 'ready', datasets: d?.datasets || [], error: null }); })
      .catch((e) => { if (alive) setIndex({ status: 'error', datasets: [], error: e?.message || 'Could not load your datasets' }); });
    return () => { alive = false; };
  }, []);

  // Where the user has MOVED cards: {datasetId: {record_id: {x, y}}}. One request
  // for the whole board — the lanes need their overrides before the first paint,
  // and these rows are tiny next to the genealogies they precede.
  const [positions, setPositions] = useState({});
  useEffect(() => {
    let alive = true;
    apiFetch('/api/train/canvas/positions')
      .then((d) => {
        if (!alive) return;
        const next = {};
        for (const [dsId, rows] of Object.entries(d?.positions || {})) next[dsId] = toOverrideMap(rows);
        setPositions(next);
      })
      // A board that opens in its automatic layout is a far better failure than
      // a board that does not open: nothing about the canvas may block it.
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  /* Remember a lane's arrangement. Applied to the screen FIRST and sent
     afterwards, deliberately: a card must follow the finger at the speed of the
     finger, and a position is a display preference — if the write fails the move
     stays on screen and the user is not interrupted by a modal to be told the
     database did not like where they put a rectangle. The next drag re-sends the
     whole lane, so a lost write heals itself. */
  const onPinLane = useCallback((datasetId, rows) => {
    const map = toOverrideMap(rows);
    setPositions((p) => ({ ...p, [datasetId]: { ...(p[datasetId] || {}), ...map } }));
    putJson(`/api/dataset/${datasetId}/canvas/positions`, { positions: rows }).catch(() => {});
  }, []);

  /* The images PINNED on the board, {datasetId: {imageId: node}}.
     Stored SERVER-SIDE, in canvas_image_node, deliberately next to the card
     positions rather than in localStorage: the cards already follow the dataset
     from one machine to the next, and a board whose cards travel while its
     pictures stay behind is an inconsistency you only discover after you have
     changed desk. Its own table, so nothing about the existing card rows
     changes shape -- a user updating into this version finds their cards
     exactly where they left them. */
  const [imageNodes, setImageNodes] = useState({});
  useEffect(() => {
    let alive = true;
    apiFetch('/api/train/canvas/images')
      .then((d) => {
        if (!alive) return;
        const next = {};
        for (const [dsId, rows] of Object.entries(d?.nodes || {})) {
          next[dsId] = toImageNodeMap(rows);
        }
        setImageNodes(next);
      })
      // Same rule as the positions: nothing about the canvas may stop the canvas
      // from opening.
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  /* Pin, move, resize or CLOSE one or more images of a lane. Applied to the
     screen first and sent afterwards, exactly like a card position -- and for
     the same reason: a picture has to follow the finger at the speed of the
     finger, and a lost write heals on the next gesture rather than interrupting
     the user with a modal about a rectangle.

     A closed node keeps its row and its geometry (`visible: false`), which is
     what makes re-opening it land on the same spot at the same size. */
  const onSaveImageNodes = useCallback((datasetId, rows) => {
    setImageNodes((cur) => {
      const lane = { ...(cur[datasetId] || {}) };
      for (const r of rows) {
        const prev = lane[r.image_id];
        lane[r.image_id] = {
          imageId: r.image_id, x: r.x, y: r.y, w: r.w, h: r.h,
          visible: r.visible !== false,
          image: r.image || prev?.image,
        };
      }
      return { ...cur, [datasetId]: lane };
    });
    putJson(`/api/dataset/${datasetId}/canvas/images`, {
      // `image` is the client's own render payload; the server resolves it from
      // the id and must not be handed a copy to trust.
      nodes: rows.map(({ image, ...row }) => row),
    }).catch(() => {});
  }, []);

  const availableIds = useMemo(() => index.datasets.map((d) => d.id), [index.datasets]);
  const selected = useMemo(() => resolveSelection(availableIds, stored), [availableIds, stored]);

  /* ✦ Tidy up: forget every moved card on the VISIBLE board. Scoped to what is
     on screen — a lane the user unticked is not on the board they are looking
     at, and silently flattening its arrangement too would be a surprise. */
  const onTidyUp = useCallback(() => {
    setPositions((p) => {
      const next = { ...p };
      for (const id of selected) delete next[id];
      return next;
    });
    for (const id of selected) del(`/api/dataset/${id}/canvas/positions`).catch(() => {});
    /* And the pinned images RE-FLOW rather than being deleted or left behind.
       Neither of the obvious answers is right: deleting them would make a
       "rebuild the automatic tree" button destroy content the user placed
       deliberately, and leaving them alone would strand every picture at
       coordinates chosen next to a card that has just moved back. So each
       VISIBLE pin is recomputed to its default spot beside its own card, which
       is the tidy version of exactly what it was.

       CLOSED pins are not touched: their remembered geometry is a promise
       ("re-open it where I closed it") and Tidy up is not the place to break
       it. The re-flow needs the laid-out lane, so it is done here against the
       automatic tree the board is about to fall back to.

       It re-flows through the SAME function 📌 Pin all uses
       (utils/canvasPinBatch), on purpose: two placers would be two chances to
       disagree, and the one thing the user is promised on this board is that
       nothing lands on top of anything. Before this, the re-flow only avoided
       other PICTURES — it could park one squarely on a run card. */
    setImageNodes((cur) => {
      const next = { ...cur };
      for (const id of selected) {
        const map = next[id];
        if (!map) continue;
        const tree = trees[id]?.tree;
        const graph = tree ? buildLineageGraph(tree) : null;
        const nodes = visibleImageNodes(map);
        if (!nodes.length) continue;
        const res = placeImageBatch({
          graph,
          existing: [],
          images: nodes.map((n) => ({ id: n.imageId, dataset_id: id,
            record_id: n.image?.record_id, step: n.image?.step })),
          max: nodes.length,
        });
        const lane = { ...map };
        const rows = [];
        for (const p of res.placed) {
          lane[p.imageId] = { ...lane[p.imageId], x: p.x, y: p.y, w: p.w, h: p.h };
          rows.push({ image_id: p.imageId, x: p.x, y: p.y, w: p.w, h: p.h, visible: true });
        }
        next[id] = lane;
        // One write for the lane, not one per picture: a board carrying twenty
        // pins used to fire twenty requests at the server that is probably also
        // training something.
        if (rows.length) putJson(`/api/dataset/${id}/canvas/images`, { nodes: rows }).catch(() => {});
      }
      return next;
    });
  }, [selected, trees]);

  /* Re-read ONE lane from the server and put it back on the board. Used after a
     deploy launched from the canvas: the pills of that dataset have to come back
     `testable`, with the name of the copy that now sits in ComfyUI. Returns the
     fresh tree so the caller can read it directly instead of racing the state
     update it just triggered. */
  const onRefetchDataset = useCallback(async (id) => {
    const tree = await apiFetch(`/api/dataset/${id}/train/lineage`);
    setTrees((t) => ({ ...t, [id]: { status: 'ready', tree, error: null } }));
    return tree;
  }, []);

  const persist = useCallback((ids) => {
    setStored(ids);
    writeSelection(typeof localStorage !== 'undefined' ? localStorage : null, ids);
  }, []);

  const onToggle = useCallback((id) => {
    persist(toggleSelection(selected, id, availableIds));
  }, [persist, selected, availableIds]);

  // Fetch the genealogy of every selected dataset that has none yet, a few at a
  // time. A dataset unticked and re-ticked keeps its cached tree — the board must
  // not re-scan the disk for a filter click.
  useEffect(() => {
    const missing = selected.filter((id) => !trees[id] && !inflight.current.has(id));
    if (!missing.length) return;
    const room = MAX_PARALLEL_FETCHES - inflight.current.size;
    if (room <= 0) return;
    for (const id of missing.slice(0, room)) {
      inflight.current.add(id);
      setTrees((t) => ({ ...t, [id]: { status: 'loading', tree: null, error: null } }));
      apiFetch(`/api/dataset/${id}/train/lineage`)
        .then((tree) => setTrees((t) => ({ ...t, [id]: { status: 'ready', tree, error: null } })))
        .catch((e) => setTrees((t) => ({ ...t,
          [id]: { status: 'error', tree: null, error: e?.message || 'Could not load this lineage' } })))
        .finally(() => {
          inflight.current.delete(id);
          // Nudge the effect so the next batch starts.
          setTrees((t) => ({ ...t }));
        });
    }
  }, [selected, trees]);

  const entries = useMemo(() => selected.map((id) => {
    const row = index.datasets.find((d) => d.id === id);
    const state = trees[id];
    return {
      datasetId: id,
      name: row?.name || `Dataset ${id}`,
      runs: row?.runs || 0,
      families: row?.families || [],
      // The ★ pinned LoRA(s) of this dataset, so a delete from the board can warn
      // that it is about to break the saved winning combo — exactly like the
      // dataset panel does. Travels on the lane, and the canvas reads the lane of
      // whichever checkpoint popover is open.
      bestSettingsLoras: row?.best_settings_loras || [],
      status: state?.status || 'loading',
      error: state?.error || null,
      // The RAW tree, not a laid-out graph: the canvas has to be able to lay it
      // out again when a run is removed from the board (see LineageCanvas).
      tree: state?.tree || null,
    };
  }), [selected, index.datasets, trees]);

  return (
    <div>
      <header className="mb-3">
        <h1 className="flex items-center gap-2 text-lg font-semibold text-content">
          <span aria-hidden>◉</span> LoRA Canvas
          <span className="px-1.5 py-0.5 rounded border border-amber-400/50 bg-amber-500/10 text-amber-300 text-[0.625rem] font-semibold uppercase tracking-wide">Beta</span>
          <HelpBadge topic="page-canvas" />
        </h1>
        <p className="mt-1 text-content-muted text-[0.75rem]">
          Every training run you have made, on one board: each dataset gets a lane, each run a card,
          and a continuation is joined to the exact checkpoint it resumed from.
        </p>
      </header>

      {index.status === 'error' && (
        <p className="mb-3 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-amber-100 text-[0.75rem]">
          {index.error}
        </p>
      )}

      <CanvasDatasetFilter
        datasets={index.datasets}
        selected={selected}
        onToggle={onToggle}
        onAll={() => persist(availableIds)}
        onNone={() => persist([])} />

      {index.status === 'loading'
        ? <p className="text-content-subtle text-[0.75rem]">Loading your datasets…</p>
        : (
          <LineageCanvas entries={entries} positions={positions}
            imageNodes={imageNodes} onSaveImageNodes={onSaveImageNodes}
            onPinLane={onPinLane} onTidyUp={onTidyUp}
            onRefetchDataset={onRefetchDataset} />
        )}
    </div>
  );
}
