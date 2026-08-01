/* ▶ Continue training FROM THE BOARD — the rules the LoRA Canvas needs to turn a
   checkpoint pill into a real launch, JSX-free so `node --test` exercises the
   ones that matter (LineageCanvas.jsx imports these exact functions).

   WHY this file exists rather than a third dialog: the app already has ONE
   continue form (components/dataset/ContinueDialog.jsx) and three endpoints
   behind it. What the two existing hosts each own is not the form — it is the
   ROUTING (which endpoint a lane maps to) and the LANE RULE (why a lane is
   closed). The board is the third host, so it needs those two answers and
   nothing else. They live here, alone, testable.

   The board is also the first surface that can offer BOTH lanes for BOTH kinds
   of run, because a lineage node carries `dataset_id` on every node — cloud and
   local alike:

     source   lane    endpoint                                         addressed by
     ------   -----   ----------------------------------------------   ------------
     cloud    cloud   POST /api/dataset/train/cloud/continue           run_id
     cloud    local   POST /api/dataset/<id>/train/continue            dataset + base/family/variant
     local    local   POST /api/dataset/<id>/train/continue            dataset + base/family/variant
     local    cloud   POST /api/dataset/<id>/train/cloud/continue-local dataset + base/family/variant

   The Runs hub covers rows 1-2, the dataset panel rows 3-4. Neither covers all
   four, which is exactly why the board's popover used to point elsewhere. */

import { runsHubContinueLanes } from './runsHubContinueLanes.js';

/* The steps this NODE can be resumed from, ascending and distinct.
   `node.checkpoints` is the only resumable list a lineage payload carries
   (`resume_steps` exists on the Runs-hub rows, not here), so the board derives
   it the same way the Runs page does when it opens the dialog from a pill. */
export function canvasContinueSteps(node) {
  const cks = Array.isArray(node?.checkpoints) ? node.checkpoints : [];
  return [...new Set(cks.map((c) => c?.step).filter((s) => Number.isFinite(s) && s > 0))]
    .sort((a, b) => a - b);
}

/* The Runs-hub row for this node, matched on `record_id` — the ONE key both
   payloads share (a local lineage node carries no `run_id` at all). The row is
   worth the lookup for two things the lineage node does not carry: the run's
   own `masked` flag, and its frozen `settings` snapshot. */
export function canvasContinueRow(node, rows) {
  const id = node?.record_id;
  if (id == null || !Array.isArray(rows)) return null;
  return rows.find((r) => r && r.record_id === id) || null;
}

/* What the dialog shows as "the starting point". The lineage node's own
   `config` IS the launch snapshot (it spells the rate as `lr`, not
   `learning_rate` — the Runs hub maps it the same way), and the hub row is
   preferred when we have it because it is the same snapshot re-read live.
   A missing snapshot yields {} and the dialog says so on its own — it never
   invents a rate. */
export function canvasContinueSettings(node, row = null) {
  const s = (row && row.settings) || node?.config || null;
  if (!s || typeof s !== 'object') return {};
  const out = {};
  if (s.optimizer) out.optimizer = s.optimizer;
  // `lr` is the snapshot's name for it; accept `learning_rate` too so a future
  // snapshot that spells it out keeps working.
  const lr = typeof s.lr === 'number' ? s.lr
    : (typeof s.learning_rate === 'number' ? s.learning_rate : null);
  if (lr != null) out.learning_rate = lr;
  if (Number.isFinite(s.save_every)) out.save_every = s.save_every;
  if (Number.isFinite(s.sample_every)) out.sample_every = s.sample_every;
  if (Number.isFinite(s.rank)) out.rank = s.rank;
  if (Number.isFinite(s.alpha)) out.alpha = s.alpha;
  return out;
}

/* WHY the board cannot offer ▶ Continue on this pill at all, or null when it
   can. Only the cases where NEITHER lane could ever work land here; everything
   else is a lane reason (below), because a closed lane that states why is worth
   more than a missing button. */
export function canvasContinueRefusal(node, pill) {
  if (!node) return 'This run is unknown on this machine.';
  if (node.dataset_id == null && !(node.source === 'cloud' && node.run_id != null)) {
    return 'This run’s dataset is unknown on this machine, so it cannot be continued from the board.';
  }
  if (pill && pill.step == null) return 'This save has no step to resume from.';
  if (!canvasContinueSteps(node).length) {
    return 'This run holds no checkpoint to resume from — its saves are gone.';
  }
  return null;
}

/* WHICH lanes the board can offer for this pill.

   The machine-wide / per-dataset guards are NOT re-implemented: they are the
   Runs hub's rule (runsHubContinueLanes), which already answers "is ai-toolkit
   set up, is something training here, is this dataset's family already on a pod,
   is the concurrency limit reached" for a run of ANY dataset. On top of it the
   board adds the two questions only it has to ask:

     • the FILE. The local lane always resumes a file that is on this machine;
       so does the cloud lane for a LOCAL run (continue-local uploads it). A
       cloud run's cloud lane does not — the pod re-seeds from that run's own
       staging by run_id — so a save missing here closes one lane and not the
       other. That asymmetry is stated, not hidden.
     • the ADDRESS. A cloud node with no `run_id` is not linked on this machine,
       so its cloud lane has nothing to continue. */
export function canvasContinueLanes(node, pill, opts = {}) {
  if (!node) return null;
  const lanes = runsHubContinueLanes(
    { run_id: node.run_id, dataset_id: node.dataset_id, train_type: node.train_type },
    opts);
  const cloudRun = node.source === 'cloud' && node.run_id != null;
  // `present === false` never comes from today's lineage payload (a save that is
  // gone simply produces no pill) — this is the guard for a board left open
  // while the disk changed underneath it, and for any payload that starts
  // reporting it. Cheap, and the alternative is a launch that 400s.
  const gone = !!pill && pill.present === false;

  if (!cloudRun && node.source === 'cloud') {
    lanes.cloud = { available: false,
      reason: 'This cloud run is not linked on this machine, so there is no run to relaunch — '
        + 'continue it on this machine instead if its save is here.' };
  } else if (!cloudRun && node.dataset_id == null) {
    lanes.cloud = { available: false,
      reason: 'This run’s dataset is unknown on this machine, so no pod can be seeded from it.' };
  } else if (!cloudRun && gone) {
    lanes.cloud = { available: false,
      reason: 'This save is no longer on this machine, so there is no file to send to a pod — '
        + 'continue from a save that is still here.' };
  }

  if (gone && lanes.local.available) {
    lanes.local = { available: false,
      reason: 'This save is no longer on this machine'
        + (cloudRun
          ? ' — continue in the cloud instead: a fresh pod is seeded from this run’s own staging.'
          : ' — continue from a save that is still here.') };
  }
  return lanes;
}

/* The REQUEST a resolved dialog payload becomes: { url, body }.

   One rule, four cases (the table at the top of this file). Two things it does
   that no caller should have to remember:

     • `from_step` is ALWAYS sent explicitly, even for the newest save. The
       dialog nulls it to mean "resume in place", which is right on a surface
       scoped to one run folder — but the board is not: `/train/continue` with no
       step resumes the newest save OF THE LANE, and a lane's run dir can hold
       several runs' saves. On the board that would silently continue a different
       run than the card that was clicked.
     • `masked` rides the SOURCE run's own flag, not a board-wide default: the
       continuation must train like the checkpoint it resumes. Unknown (no hub
       row) → omitted, and the backend's own default applies — the same
       fallback the Runs hub uses for a legacy row. */
export function canvasContinueRequest(node, payload, { steps = [], masked = null } = {}) {
  if (!node || !payload) return null;
  const list = Array.isArray(steps) ? steps : [];
  const fromStep = payload.fromStep != null
    ? payload.fromStep
    : (list.length ? list[list.length - 1] : null);
  const extra = { extra_steps: payload.extraSteps };
  if (fromStep != null) extra.from_step = fromStep;
  if (payload.overrides) extra.overrides = payload.overrides;
  extra.resume_mode = payload.resumeMode || 'weights_only';
  if (payload.stateBundleId) extra.state_bundle_id = payload.stateBundleId;

  if (payload.lane === 'cloud' && node.source === 'cloud' && node.run_id != null) {
    return { url: '/api/dataset/train/cloud/continue',
      body: { run_id: node.run_id, ...extra } };
  }
  if (node.dataset_id == null) return null;
  const selection = {
    ...(node.base_model != null ? { base_model: node.base_model } : {}),
    ...(node.train_type ? { train_type: node.train_type } : {}),
    ...(node.variant ? { variant: node.variant } : {}),
    ...(masked == null ? {} : { masked: masked !== false }),
  };
  const path = payload.lane === 'cloud'
    ? `/api/dataset/${node.dataset_id}/train/cloud/continue-local`
    : `/api/dataset/${node.dataset_id}/train/continue`;
  return { url: path, body: { ...extra, ...selection } };
}

export default canvasContinueRequest;
