/**
 * useCanvasStudio — the Test-Studio engine, driven from the ◉ LoRA Canvas.
 *
 * This hook does NOT reimplement the Studio. It composes the two hooks the
 * Studio already runs on and swaps exactly one thing:
 *
 *   useLoraTestStudio(anchorDataset)  → the SETTINGS payload `d`
 *                                       (models, aspects, cfg/steps choices,
 *                                        recent prompts, always-on LoRAs)
 *   tracker (useCanvasRun)            → the LIVE state of the launch, owned by
 *                                       the BOARD so it survives this panel
 *                                       being closed and the page reloaded
 *   launch()                          → POST /api/train/canvas/generate
 *                                       instead of the per-dataset route,
 *                                       because a canvas run may span datasets
 *
 * The returned object has the same shape `RunSetupPanel` already consumes
 * ({data, launch, launching, cancel, resume, deletePrompt}), which is what makes
 * the two screens the same code rather than two implementations that agree today.
 *
 * `launch` keeps the Studio's exact signature — checkpoints, strengths, seed,
 * prompt, models, aspects, cfgs, steps, steps2, count, genSettings — so the
 * global generation settings ride along untouched. The `checkpoints` argument is
 * ignored: the canvas already knows which pills were ticked, and it sends their
 * (dataset, run, step) identity rather than a bare filename.
 *
 * Deploy-then-generate lives here too: when picks are not in ComfyUI yet the
 * caller's `onDeploy` runs first, and a failure ABORTS the launch with the real
 * reason — nothing is written to someone's disk by a button that did not say so,
 * and nothing generates on half a selection. That order is what makes 🧬 Blend
 * safe as well: the deploy runs over the WHOLE selection before anything is
 * posted, so a blend never loads a subset of the checkpoints it announced.
 *
 * `blend` / `weights` switch the launch from one pass per checkpoint to ONE
 * generation loading them all, each at its own weight — the engine's `combine`
 * mode (named « Blend » on both screens), the same one the Test Studio's
 * 🧬 toggle drives.
 */
import { useCallback, useState } from 'react';
import { useToast } from '../components/common/Toast';
import { postJson } from './useDataset';
import { useLoraTestStudio } from './useLoraTestStudio';
import { canvasRunSelections, canvasUndeployed } from '../utils/canvasGeneration';

export function useCanvasStudio(selection, family,
  { onDeploy, tracker, blend = false, weights = null, sets = null } = {}) {
  const toast = useToast();
  const anchorId = selection?.[0]?.datasetId ?? null;
  // The settings payload comes from ONE dataset — the first pick. Everything it
  // offers (bases, formats, cfg/steps ladders) is a property of the FAMILY, and
  // the launch is single-family by construction, so it describes every pick.
  const base = useLoraTestStudio(anchorId, family);
  // ⚠️ The run in flight belongs to the BOARD (useCanvasRun), not to this hook.
  // It used to live here, which meant closing the settings panel destroyed the
  // only handle on a generation that was still running — reopening it showed the
  // form again while ComfyUI was busy. The board owns the id, remembers it across
  // a reload, and polls it once for both surfaces.
  const runId = tracker?.runId ?? null;
  const run = tracker?.run ?? { data: null };
  const [launching, setLaunching] = useState(false);

  const launch = useCallback(async (
    _checkpoints, strengths, seed, prompt, zModels, aspects, cfgs,
    stepsList, steps2List, count = 1, genSettings = {},
  ) => {
    setLaunching(true);
    try {
      let picks = selection || [];
      // 1. Deploy what has to be deployed, and stop here if it fails: generating
      //    on half the picks would silently answer a different question than the
      //    one the user asked.
      if (canvasUndeployed(picks).length) {
        const deployed = await onDeploy?.(canvasUndeployed(picks));
        if (!deployed) return { ok: false, error: 'Deploy cancelled — nothing was generated' };
        picks = deployed;
        const stillMissing = canvasUndeployed(picks);
        if (stillMissing.length) {
          const msg = `${stillMissing.length} checkpoint(s) could not be deployed — nothing was generated`;
          toast.error(msg);
          return { ok: false, error: msg };
        }
      }
      // 🧬 Blend needs at least two LoRAs to be one; below that it IS a plain
      //    single-LoRA run, and asking for it would only strip the strength
      //    sweep off a run the user can still legitimately sweep.
      const blending = blend && canvasRunSelections(picks).length > 1;
      const selections = canvasRunSelections(picks, {
        blend: blending, weights: weights || {}, sets: sets || {} });
      if (!selections.length) {
        return { ok: false, error: 'No deployed checkpoint in the selection' };
      }
      // 2. The SAME body the comparison grid posts, plus the exact origin of each
      //    pick — one shared prompt and seed across every dataset on the board.
      //    In a blend every LoRA carries its own weight, so the strength axis is
      //    not sent at all (the engine replaces it with the head LoRA's weight).
      const d = await postJson('/api/train/canvas/generate', {
        selections, ...(blending ? { combine: true } : { strengths }), seed, prompt,
        // Cross-dataset runs sweep ONE base model (the comparison engine's
        // contract); the canvas sends the first pick of the model axis.
        z_model: (zModels || [])[0] ?? null,
        aspects, cfgs, steps: stepsList, steps2: steps2List, count, ...genSettings,
      });
      if (d.ok) {
        // The board adopts the run WITH the checkpoints it was launched on —
        // that pairing is what lets it say, once the images land, which pill's
        // gallery they went into. Without it a finished run ends in silence.
        tracker?.adopt?.(d.run_id, picks.map((e) => ({
          datasetId: e.datasetId, recordId: e.recordId, step: e.step,
          datasetName: e.datasetName || null,
        })));
        toast.success(`${d.created} generation(s) queued (seed ${d.seed}${d.count > 1 ? ` ×${d.count}` : ''})`);
      } else {
        toast.error(d.error || 'Unexpected error');
      }
      return d;
    } finally {
      setLaunching(false);
    }
  }, [selection, onDeploy, toast, tracker, blend, weights, sets]);

  // The payload RunSetupPanel reads: the anchor's SETTINGS, the canvas run's LIVE
  // state, and the ticked pills standing in for the picker's checkpoint list.
  const data = base.data
    ? {
      ...base.data,
      checkpoints: (selection || []).map((e) => ({
        filename: e.filename || `${e.datasetId}:${e.recordId}:${e.step}`,
        label: `step ${e.step}`,
      })),
      pending: run.data?.pending ?? 0,
      queued: run.data?.queued ?? 0,
      generating: run.data?.generating ?? run.data?.running ?? 0,
      resumable: run.data?.resumable ?? 0,
      gpu_busy: run.data?.gpu_busy ?? base.data.gpu_busy,
      comfyui_recovery: run.data?.comfyui_recovery ?? base.data.comfyui_recovery,
      comfyui_recovery_target: run.data?.comfyui_recovery_target ?? base.data.comfyui_recovery_target,
    }
    : null;

  return {
    data, runId, run,
    launch, launching,
    cancel: run.cancel, resume: run.resume,
    refresh: base.refresh, deletePrompt: base.deletePrompt,
    confirmComfyuiRestart: run.data?.comfyui_recovery
      ? run.confirmComfyuiRestart
      : base.confirmComfyuiRestart,
    confirmingComfyuiRestart: run.data?.comfyui_recovery
      ? run.confirmingComfyuiRestart
      : base.confirmingComfyuiRestart,
    rate: run.rate,
  };
}
