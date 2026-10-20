import { useEffect, useState } from 'react';
import { apiFetch } from '../api/fetchClient';

/** Value of `qwen21.scheduler` that swaps the scheduler for the viggle_turbo.py
 *  custom node's ViggleTurboSigmasSelect (qwen21_edit_helper.VIGGLE_SCHEDULER). */
export const VIGGLE_SCHEDULER = 'viggle_turbo';

/** Shapes accepted by qwen21_edit_helper._enum_name — anything else is ignored
 *  server-side, so the field flags it instead of saving a dead value. */
export function isEnumName(value) {
  return /^[a-z0-9_]+$/.test(String(value ?? ''));
}

/** The sampler / scheduler names the connected ComfyUI accepts (GET
 *  /api/comfy/sampler-options). Degrades to empty lists — the fields stay free
 *  text — when ComfyUI is offline or the request fails. */
export function useSamplerOptions() {
  const [state, setState] = useState({
    samplers: [], schedulers: [], viggle: false, reachable: false, loading: true,
  });
  useEffect(() => {
    let cancelled = false;
    apiFetch('/api/comfy/sampler-options')
      .then((d) => {
        if (cancelled) return;
        setState({
          samplers: Array.isArray(d?.samplers) ? d.samplers : [],
          schedulers: Array.isArray(d?.schedulers) ? d.schedulers : [],
          viggle: Boolean(d?.viggle_sigmas),
          reachable: Boolean(d?.reachable),
          loading: false,
        });
      })
      .catch(() => { if (!cancelled) setState((s) => ({ ...s, loading: false })); });
    return () => { cancelled = true; };
  }, []);
  return state;
}

/** Scheduler choices for the Qwen 2.1 fields: ComfyUI's list plus the viggle
 *  sigma mode when its node is loaded (or already selected). */
export function qwen21SchedulerChoices(schedulers, viggle, current) {
  const out = [...schedulers];
  if ((viggle || current === VIGGLE_SCHEDULER) && !out.includes(VIGGLE_SCHEDULER)) {
    out.unshift(VIGGLE_SCHEDULER);
  }
  return out;
}

/** '' (ComfyUI built-in) | 'official' | numeric string — the `qwen21.shift` modes. */
export function shiftMode(value) {
  const v = String(value ?? '').trim().toLowerCase();
  if (v === '' || v === 'default' || v === 'comfy') return 'default';
  if (v === 'official') return 'official';
  return Number.isFinite(Number(v)) ? 'custom' : 'default';
}

/** Released dynamic shift (Qwen-Image 2.1 scheduler config) for a W x H output —
 *  mirrors qwen21_edit_helper.official_shift_mu. */
export function officialShiftMu(width, height) {
  const tokens = Math.floor(width / 16) * Math.floor(height / 16);
  return 0.5 + 0.4 * (tokens - 256) / (8192 - 256);
}
