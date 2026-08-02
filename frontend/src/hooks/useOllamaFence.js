/* The one place a surface refused by the local Ollama fence learns to wait.
 *
 * Wrap the action instead of calling it: `runGuarded(fn)` runs fn, and if the
 * server answers with the fence code, it holds on to fn, starts watching
 * /api/system/ollama-fence, and runs fn AGAIN by itself the moment the other
 * model goes away. The user's click is not lost, so the common case — Ollama's
 * own idle unload a few minutes later — costs nothing at all.
 *
 * Shared deliberately: the fence guards every local Ollama call (Test Studio's
 * ✨ Enhance and 🔎 Describe, captioning), and a per-button copy of this would
 * drift the moment one of them changed.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch, postJson } from '../api/fetchClient';
import { AUTO_RETRY_CAP_MS, isOllamaFenceError, nextPollDelay } from '../utils/ollamaFence';

/**
 * `onError` is how a REPLAY reports a failure that is not the fence.
 *
 * It matters more than it looks: the first attempt fails inside the caller's
 * own try/catch, which toasts. The replay happens minutes later with nobody in
 * that catch — so without this, an Ollama that died while we waited would make
 * the notice quietly disappear and nothing else happen at all. A surface that
 * shows its own errors (the Describe modal writes into its error line) can
 * leave this out.
 */
export default function useOllamaFence({ onError } = {}) {
  const [state, setState] = useState(null);
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;
  const actionRef = useRef(null);
  const timerRef = useRef(null);
  const startedRef = useRef(0);
  const aliveRef = useRef(true);

  const stopTimer = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
  }, []);

  useEffect(() => {
    aliveRef.current = true;
    return () => { aliveRef.current = false; stopTimer(); };
  }, [stopTimer]);

  /* Re-run what was refused. Returns true when the fence took it again — the
     model can be freed and immediately claimed by the other tool, and the
     caller has to put the vigil back on watch or it dies here. */
  const replay = useCallback(async () => {
    const action = actionRef.current;
    if (!action) { setState(null); return false; }
    setState((s) => (s ? { ...s, phase: 'retrying' } : s));
    try {
      await action();
      if (aliveRef.current) setState(null);
      return false;
    } catch (e) {
      if (!aliveRef.current) return false;
      if (isOllamaFenceError(e)) {
        // Still fenced: keep waiting rather than throw the user back to the
        // start, but do not restart the patience clock.
        setState((s) => (s ? { ...s, phase: 'waiting' } : s));
        return true;
      }
      // A different failure, with nobody left holding the original catch.
      setState(null);
      actionRef.current = null;
      onErrorRef.current?.(e);
      return false;
    }
  }, []);

  const tick = useCallback(async () => {
    if (!aliveRef.current) return;
    const elapsedMs = Date.now() - startedRef.current;
    let free = false;
    let models = [];
    try {
      // background: a vigil that may run for ten minutes must not toast every
      // time the server blinks — the offline indicator already owns that story.
      const data = await apiFetch('/api/system/ollama-fence', { background: true });
      free = data?.blocked === false;
      models = data?.models || [];
    } catch {
      // Cannot read the state (old backend, server down): stay put and keep
      // the notice as it is. The user's buttons still work.
      free = false;
    }
    if (!aliveRef.current) return;
    if (free) {
      stopTimer();
      // Freed, then taken again before we could use it: go back on watch
      // instead of leaving a "waiting" notice that nothing is watching.
      if (await replay() && aliveRef.current) {
        timerRef.current = setTimeout(tick, nextPollDelay(elapsedMs));
      }
      return;
    }
    if (elapsedMs >= AUTO_RETRY_CAP_MS) {
      stopTimer();
      setState((s) => (s ? { ...s, phase: 'gave-up', elapsedMs, models } : s));
      return;
    }
    setState((s) => (s && s.phase === 'waiting' ? { ...s, elapsedMs, models } : s));
    timerRef.current = setTimeout(tick, nextPollDelay(elapsedMs));
  }, [replay, stopTimer]);

  const beginWaiting = useCallback((message) => {
    startedRef.current = Date.now();
    setState({ phase: 'waiting', message, models: [], elapsedMs: 0 });
    stopTimer();
    timerRef.current = setTimeout(tick, nextPollDelay(0));
  }, [stopTimer, tick]);

  /**
   * Run `action`. Returns true when it went through, false when the fence took
   * over (the notice is now showing and the action will be retried for you).
   * Every other failure propagates untouched — this hook only knows one story.
   */
  const runGuarded = useCallback(async (action) => {
    actionRef.current = action;
    try {
      await action();
      if (aliveRef.current) setState(null);
      return true;
    } catch (e) {
      if (!isOllamaFenceError(e)) { setState(null); throw e; }
      beginWaiting(e?.message);
      return false;
    }
  }, [beginWaiting]);

  /** The consent click: evict the other model, then resume. */
  const unloadAndRetry = useCallback(async () => {
    setState((s) => (s ? { ...s, phase: 'unloading' } : s));
    stopTimer();
    try {
      await postJson('/api/system/ollama-fence/unload', { confirmed_unload_external: true });
    } catch (e) {
      // The server refuses what it cannot prove and says why (still busy,
      // unreachable). Show that, and go back to waiting rather than dead-end.
      setState((s) => (s ? { ...s, phase: 'waiting', message: e?.message || s.message } : s));
      timerRef.current = setTimeout(tick, nextPollDelay(Date.now() - startedRef.current));
      return false;
    }
    // The other tool can reload a model between our unload and our retry.
    if (await replay() && aliveRef.current) {
      timerRef.current = setTimeout(tick, nextPollDelay(Date.now() - startedRef.current));
    }
    return true;
  }, [replay, stopTimer, tick]);

  /** "Stop waiting" — abandon the queued action without unloading anything. */
  const stopWaiting = useCallback(() => {
    stopTimer();
    actionRef.current = null;
    setState(null);
  }, [stopTimer]);

  return { fence: state, runGuarded, unloadAndRetry, stopWaiting };
}
