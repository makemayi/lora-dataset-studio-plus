/* What a generation launched from the ◉ LoRA Canvas is DOING, and where its
   images went.

   Two things were broken in real use and both come back to the same mistake:
   the run only existed inside the settings panel. Close the panel — or leave the
   page — and the id of the run in flight was gone with it, so reopening showed a
   blank form while ComfyUI was still rendering. And when the images did land,
   nothing on the board said so: they join the checkpoint's gallery (the × N
   badge on the pill), but the board never re-read itself, so the badge did not
   appear until a full reload.

   A generation is a state of the BOARD, not of a panel. That is what this module
   models: the run id and the checkpoints it was launched on are remembered
   (localStorage, so a reload finds them again), and the progress is described
   from the run status the board polls itself.

   JSX-free — `node --test` runs the arithmetic and the wording directly.

   ⚠️ What is persisted is a run id plus DATABASE ids (dataset id, run record id,
   step). Never a name: names are renamed, and a rename must not be able to point
   the tracker at the wrong checkpoint. */

export const CANVAS_RUN_KEY = 'lds.canvasRun';

const asId = (v) => {
  const n = Number(v);
  return Number.isInteger(n) && n > 0 ? n : null;
};

/** The launched checkpoints, cleaned: only entries whose three database ids are
 *  real. A half-formed target would render a button that opens nothing. */
export function normaliseTargets(targets) {
  const out = [];
  for (const t of (Array.isArray(targets) ? targets : [])) {
    const datasetId = asId(t?.datasetId);
    const recordId = asId(t?.recordId);
    const step = Number(t?.step);
    if (datasetId == null || recordId == null || !Number.isFinite(step)) continue;
    out.push({ datasetId, recordId, step, datasetName: t?.datasetName || null });
  }
  return out;
}

/** Read the remembered canvas run. null = none in memory (never an exception:
 *  a corrupt entry must not be able to break the board it decorates). */
export function readCanvasRun(store, key = CANVAS_RUN_KEY) {
  try {
    const raw = store?.getItem(key);
    if (raw == null) return null;
    const parsed = JSON.parse(raw);
    const runId = parsed?.runId;
    if (typeof runId !== 'string' || !runId) return null;
    return { runId, targets: normaliseTargets(parsed.targets) };
  } catch {
    return null;
  }
}

/** Persist (or, with a null value, forget) the remembered run. Best-effort: a
 *  blocked localStorage must never break a launch. */
export function writeCanvasRun(store, value, key = CANVAS_RUN_KEY) {
  try {
    if (!value || !value.runId) store?.removeItem(key);
    else store?.setItem(key, JSON.stringify({
      runId: String(value.runId), targets: normaliseTargets(value.targets),
    }));
    return true;
  } catch {
    return false;
  }
}

/** How many cells of a run status payload have produced a file on disk. This is
 *  the only honest definition of "an image is ready": `created` counts what was
 *  QUEUED, and a cancelled or failed cell never becomes an image. */
export function readyImageCount(run) {
  return (run?.cells || []).filter((c) => !!c?.filename).length;
}

/**
 * The tracker bar, in one object.
 *
 * phase:
 *   'idle'    — nothing to say; the bar is not drawn
 *   'working' — cells are queued or rendering (Stop is offered)
 *   'stopped' — stopped cells remain, resumable with their settings
 *   'done'    — the run finished and produced images (Results are offered)
 *
 * The wording matches the Studio's own bar, deliberately: it is the same engine,
 * and two vocabularies for one run would read as two features.
 */
export function describeCanvasRun(run) {
  const pending = Number(run?.pending) || 0;
  const queued = Number(run?.queued ?? pending) || 0;
  const generating = Number(run?.generating ?? run?.running) || 0;
  const resumable = Number(run?.resumable) || 0;
  const ready = readyImageCount(run);
  if (!run) return { phase: 'idle', generating: 0, queued: 0, resumable: 0, ready: 0, text: '' };
  if (pending > 0) {
    return { phase: 'working', generating, queued, resumable, ready,
      text: `${generating} generating · ${queued} queued` };
  }
  if (resumable > 0) {
    return { phase: 'stopped', generating, queued, resumable, ready,
      text: `${resumable} stopped image${resumable > 1 ? 's' : ''} — resumable with their settings` };
  }
  if (ready > 0) {
    return { phase: 'done', generating, queued, resumable, ready,
      text: `${ready} image${ready > 1 ? 's' : ''} ready` };
  }
  return { phase: 'idle', generating, queued, resumable, ready, text: '' };
}

/**
 * The images a finished run actually produced, as {id, datasetId} — the lot
 * 📌 Pin all offers to drop onto the board.
 *
 * Same definition of "ready" as readyImageCount: a cell with a file on disk.
 * A cancelled or failed cell is not an image and must not be counted into a
 * button that promises to put N pictures down.
 *
 * Only the ids: the FACTS of each image (prompt, seed, sampler, the checkpoint
 * that made it) are read from the gallery route at pin time, so the node on the
 * board carries exactly the payload every other gallery publishes. Rebuilding
 * that payload from the run cells here would be a second shape of the same
 * record, free to drift from the first.
 */
export function runPinCandidates(run) {
  const out = [];
  for (const c of (run?.cells || [])) {
    if (!c?.filename) continue;
    const id = asId(c.id);
    const datasetId = asId(c.dataset_id);
    if (id == null || datasetId == null) continue;
    out.push({ id, datasetId });
  }
  return out;
}

/** The dataset ids a run touched — the lanes that have to be re-read so their
 *  pills gain the new thumbnail and the × N badge. */
export function canvasRunDatasetIds(targets) {
  return [...new Set(normaliseTargets(targets).map((t) => t.datasetId))];
}

/** The label of one result button: short enough for a 400-px row, and naming
 *  the checkpoint by the two ids the user sees on the board. */
export function canvasResultLabel(target) {
  return `#${target.recordId} · step ${target.step}`;
}
