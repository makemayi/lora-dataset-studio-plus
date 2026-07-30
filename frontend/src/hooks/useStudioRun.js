/**
 * useStudioRun — data hook of the autonomous multi-LoRA test Studio.
 *
 * Pilots ONE run identified by `runId`. Polls
 * GET /api/studio/run/<runId>/status (3 s while cells are pending — same rhythm
 * as useLoraTestStudio) and exposes the mutations: rate a cell 👍/👎, cancel the
 * run, resume it. When `runId` is null → no poll (blank studio waiting for a run).
 *
 * Contract of the status payload (delivered by the backend):
 *   { run_id, loras:[{dataset_id, lora_label, dataset_name}],
 *     cells:[{id, dataset_id, checkpoint, label, strength, aspect, filename,
 *             rating, seed, run_seed, status, prompt, z_model, cfg, steps}],
 *     lora_ranking:[{dataset_id, lora_label, dataset_name, likes, dislikes,
 *                    voted, net, wilson}],
 *     pending, queued, generating, running, resumable, gpu_busy }
 */
import { useCallback, useEffect, useState } from 'react';
import { useToast } from '../components/common/Toast';
import { postJson } from '../api/fetchClient';

export function useStudioRun(runId) {
  const toast = useToast();
  const [data, setData] = useState(null);

  const [confirmingComfyuiRestart, setConfirmingComfyuiRestart] = useState(false);
  const refresh = useCallback(async () => {
    if (!runId) return;
    try {
      const r = await fetch(`/api/studio/run/${runId}/status`, { credentials: 'include' });
      if (r.ok) setData(await r.json());
    } catch { /* transient network error — the poll retries */ }
  }, [runId]);

  // Vide la grille DÈS que le run change : sinon on garde les cellules du run
  // précédent tant que le refetch n'a pas répondu (et si le fetch échoue ça reste
  // bloqué sur l'ancien run). null = pas de run sélectionné → studio vierge.
  useEffect(() => { setData(null); }, [runId]);

  useEffect(() => { refresh(); }, [refresh]);

  // Poll while generations are in flight (pending cells fill the grid live).
  useEffect(() => {
    if (!data?.pending) return undefined;
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [data, refresh]);

  // Vote sur une image de test — réutilise la route existante lora-test/rate.
  const rate = useCallback(async (imageId, rating) => {
    const d = await postJson(`/api/dataset/lora-test/image/${imageId}/rate`, { rating });
    if (!d.ok) toast.error(d.error);
    await refresh();
  }, [refresh, toast]);

  const cancel = useCallback(async () => {
    if (!runId) return undefined;
    const d = await postJson(`/api/studio/run/${runId}/cancel`);
    if (d.ok) toast.success(`${d.cancelled} generation(s) stopped — resumable`);
    else toast.error(d.error);
    await refresh();
    return d;
  }, [runId, refresh, toast]);

  const resume = useCallback(async () => {
    if (!runId) return undefined;
    const d = await postJson(`/api/studio/run/${runId}/resume`);
    if (d.ok) toast.success(`${d.resumed} cell(s) restarted with their settings`);
    else toast.error(d.error);
    await refresh();
    return d;
  }, [runId, refresh, toast]);

  // A timed-out ComfyUI submit has no safe automatic retry. The backend accepts
  // this only after a fresh ComfyUI probe and the user's explicit restart claim.
  const confirmComfyuiRestart = useCallback(async () => {
    if (!runId || confirmingComfyuiRestart) return undefined;
    setConfirmingComfyuiRestart(true);
    try {
      const d = await postJson(`/api/studio/run/${runId}/confirm-comfyui-restart`, {
        confirmed_comfyui_restart: true,
      });
      toast.success('ComfyUI restart confirmed — the paused cell is ready to resume.');
      await refresh();
      return d;
    } catch (e) {
      toast.error(e.message || 'Could not confirm the ComfyUI restart');
      await refresh();
      return { ok: false, error: e.message || 'Could not confirm the ComfyUI restart' };
    } finally {
      setConfirmingComfyuiRestart(false);
    }
  }, [runId, confirmingComfyuiRestart, refresh, toast]);

  return { data, refresh, rate, cancel, resume, confirmComfyuiRestart, confirmingComfyuiRestart };
}
