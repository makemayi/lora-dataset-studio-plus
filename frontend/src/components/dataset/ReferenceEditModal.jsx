/** ✦ Edit the reference photo with a prompt (+ optional extra reference images),
 * via any engine the install can run — the list is DERIVED from EDIT_ENGINES and
 * from `engineOptions`, never spelled out here.
 *
 * The edit runs as a SERVER background job — slow (1-3 min), and on an API engine
 * PAID, so it must not ride the client's fetch (a backgrounded mobile tab would
 * kill it and lose the paid result). This modal is DRIVEN BY SERVER STATE
 * (`referenceEdit` from the dataset payload): running → spinner, ready →
 * Before/After, failed → error. That makes it restore correctly after a tab
 * sleep, a reload, or a reopen — there is no long client request to lose.
 *
 * TWO LANES, ONE FORM. The local engines (Klein, Krea 2 Edit) render on the
 * user's own ComfyUI: free, private, and therefore the sane way to try five
 * prompts. Three things then differ and every one of them is said BEFORE the
 * click, not after the render: what it costs (editCostNote), which references it
 * consumes (editRefNote / acceptsExtraEditRefs), and whether this install can run
 * it at all (the `blocked` reason on each option).
 *
 * Keep promotes the candidate (atomic on the server); Discard deletes it; the ✕
 * just closes and LEAVES the job running (rediscovered on reopen). Modal idiom
 * mirrors CropModal: role=dialog, Escape closes, initial focus. */
import { useEffect, useRef, useState } from 'react';
import KleinModelSetting from '../shared/KleinModelSetting';
import {
  EDIT_ENGINES, LOCAL_ENGINES, editBlockedReason, batchLiveNote, editPhase,
  editEngineOptions, editCostNote, editKeepNote, editRefNote, acceptsExtraEditRefs, ENGINE_LABELS,
} from './referenceEdit';

const MAX_EDIT_REFS = 3;

export default function ReferenceEditModal({ datasetId, refFilename, nonce = 0,
                                             defaultEngine = 'chatgpt', liveActivity = null,
                                             referenceEdit = null, datasetExtraCount = 0,
                                             comfyuiConfigured = false, engineAvailable = null,
                                             engineReason = null,
                                             onEdit, onRetry = null, canRetry = false,
                                             onKeep, onDiscard, onClose }) {
  const [prompt, setPrompt] = useState('');
  const [engine, setEngine] = useState(EDIT_ENGINES.includes(defaultEngine) ? defaultEngine : 'chatgpt');
  const [editRefs, setEditRefs] = useState([]);            // transient File[]
  const [starting, setStarting] = useState(false);         // bridges POST -> server 'running'
  const [busyAction, setBusyAction] = useState(false);     // keep/discard in flight
  const inpRef = useRef(null);
  const promptRef = useRef(null);

  const serverPhase = editPhase(referenceEdit);            // idle | running | ready | failed
  const phase = starting ? 'running' : serverPhase;
  // Once the server reflects the job (running/ready/failed), drop the local bridge.
  useEffect(() => { if (serverPhase !== 'idle') setStarting(false); },
    [serverPhase, referenceEdit?.started_at]);

  const imgUrl = (fn) => `/api/dataset/${datasetId}/img/${encodeURIComponent(fn)}${nonce ? `?v=${nonce}` : ''}`;
  const beforeUrl = imgUrl(refFilename);
  const afterUrl = referenceEdit?.candidate_filename
    ? `/api/dataset/${datasetId}/img/${encodeURIComponent(referenceEdit.candidate_filename)}`
    : null;
  // This comes from the server-side job record, never the engine pill currently
  // selected in the form. A reopened modal must describe the candidate it shows.
  const resultEngine = referenceEdit?.engine || engine;
  const resultEngineLabel = ENGINE_LABELS[resultEngine] || resultEngine;

  const busy = starting || busyAction;
  // Escape / ✕ close the modal but NEVER discard — a running or ready job is left
  // on the server and rediscovered on reopen.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, busy]);
  useEffect(() => { if (phase === 'idle') promptRef.current?.focus(); }, [phase]);

  // Which engines this install can offer, and why not when it can't. Computed
  // from the SAME capabilities the generation panel reads, so a missing Krea node
  // pack is explained here in the words it is explained there.
  const options = editEngineOptions({
    comfyuiConfigured, available: engineAvailable || {}, reasonFor: engineReason,
  });
  const current = options.find((o) => o.engine === engine) || null;
  // A stored default pointing at an engine this install dropped (ComfyUI removed,
  // Klein was the primary) must not leave the form on a phantom selection.
  useEffect(() => {
    if (phase !== 'idle') return;
    if (!options.some((o) => o.engine === engine)) {
      const fallback = options.find((o) => o.usable) || options[0];
      if (fallback) setEngine(fallback.engine);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine, phase, options.length]);

  const blocked = editBlockedReason(prompt, engine, current?.blocked || null);
  const liveNote = batchLiveNote(liveActivity);
  const refNote = editRefNote(engine, { datasetExtraCount });
  const canAddRefs = acceptsExtraEditRefs(engine);
  // A local engine can't take them: drop what was already staged rather than send
  // a request the server will (rightly) refuse.
  useEffect(() => { if (!canAddRefs) setEditRefs([]); }, [canAddRefs]);

  const addRefs = (files) => {
    const list = Array.from(files || []).filter((f) => f && f.type.startsWith('image/'));
    setEditRefs((cur) => [...cur, ...list].slice(0, MAX_EDIT_REFS));
  };

  const runEdit = async () => {
    if (blocked) return;
    setStarting(true);
    const ok = await onEdit(prompt, engine, editRefs);
    if (!ok) setStarting(false);          // start failed → stay on the form
  };

  const canRetryExact = Boolean(canRetry && typeof onRetry === 'function');
  const retryEdit = async () => {
    if (!canRetryExact) return;
    setStarting(true);
    const ok = await onRetry();
    if (!ok) setStarting(false);
  };

  const keep = async () => {
    setBusyAction(true);
    const ok = await onKeep();
    if (ok) onClose(); else setBusyAction(false);
  };

  const discard = async (close) => {
    setBusyAction(true);
    await onDiscard();
    setBusyAction(false);
    if (close) onClose();
  };

  return (
    <div role="dialog" aria-modal="true" aria-label="Edit reference photo"
      className="fixed inset-0 z-[9995] bg-black/80 backdrop-blur-sm flex flex-col p-3 sm:p-4 overflow-y-auto">
      {/* Opaque card — DO NOT remove. The form has transparent gaps; laid straight
          on the dim overlay, the page reads through. bg-surface is only 4% alpha
          (--surface-alpha) — the opaque modal-panel token is bg-surface-overlay,
          the one every other modal here uses. (Regressed once when this file was
          rewritten; kept explicit so it survives the next rewrite.) */}
      <div className="w-full max-w-3xl mx-auto my-auto flex flex-col gap-3
                      bg-surface-overlay border border-border rounded-2xl shadow-2xl p-4 sm:p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-content text-base font-semibold">✦ Edit reference</h2>
          <button type="button" onClick={onClose} disabled={busy}
            aria-label="Close" className="px-2 py-1 rounded-lg bg-surface text-content text-sm disabled:opacity-40">✕</button>
        </div>

        {liveNote && (
          <p className="text-[0.6875rem] text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg px-2.5 py-1.5">
            {liveNote}
          </p>
        )}

        {phase === 'running' ? (
          <div className="flex flex-col items-center gap-3 py-8">
            <span className="inline-block w-8 h-8 border-2 border-indigo-400/40 border-t-indigo-400 rounded-full animate-spin" aria-hidden />
            <p className="text-content text-sm">
              {LOCAL_ENGINES.includes(referenceEdit?.engine)
                ? 'Editing the reference on your GPU… it queues behind any generation already running.'
                : 'Editing the reference… a “high” render can take 1–3 minutes.'}
            </p>
            <p className="text-content-muted text-[0.6875rem] text-center">
              This runs on the server — you can close this tab and come back; the Before/After will be here.
            </p>
            <button type="button" onClick={() => discard(true)} disabled={busy}
              className="px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
              Cancel edit
            </button>
          </div>
        ) : phase === 'ready' ? (
          <>
            {/* Before / After — side by side on desktop, stacked on mobile. */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <figure className="flex flex-col gap-1">
                <figcaption className="text-content-subtle text-xs">Before</figcaption>
                <img src={beforeUrl} alt="current reference"
                  className="w-full rounded-lg bg-black object-contain max-h-[45vh]" />
              </figure>
              <figure className="flex flex-col gap-1">
                <figcaption className="text-sky-300 text-xs">After (candidate)</figcaption>
                {afterUrl && <img src={afterUrl} alt="edited candidate"
                  className="w-full rounded-lg bg-black object-contain max-h-[45vh]" />}
              </figure>
            </div>
            {/* Worded from the engine that PRODUCED this candidate (the server
                echoes it back), not from the current toggle: the user may have
                closed and reopened the modal while it rendered. */}
            <p className="text-[0.6875rem] text-content-muted">
              {editKeepNote(resultEngine)}
            </p>
            <p className="text-[0.6875rem] text-content-muted" aria-live="polite">
              Engine used for this result: <span className="text-content font-semibold">{resultEngineLabel}</span>
            </p>
            {!canRetryExact && (
              <p className="text-[0.6875rem] text-content-subtle">
                Retry keeps temporary reference files only for this browser session. After reopening
                this page, use Try another prompt to attach them again.
              </p>
            )}
            <div className="flex gap-2 justify-end flex-wrap">
              <button type="button" onClick={() => discard(false)} disabled={busy}
                className="mr-auto px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
                Try another prompt
              </button>
              <button type="button" onClick={retryEdit} disabled={busy || !canRetryExact}
                title={canRetryExact ? undefined : 'The original temporary references are no longer available'}
                className="px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
                ↻ Retry same edit
              </button>
              <button type="button" onClick={() => discard(true)} disabled={busy}
                className="px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
                Discard
              </button>
              <button type="button" onClick={keep} disabled={busy}
                className="px-4 py-2 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                {busyAction ? 'Keeping…' : 'Keep'}
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="flex flex-col gap-1">
              <img src={beforeUrl} alt="current reference"
                className="w-32 h-32 rounded-lg bg-black object-cover self-start" />
            </div>

            <label className="flex flex-col gap-1">
              <span className="text-content-subtle text-xs">What should change?</span>
              <textarea ref={promptRef} value={prompt} onChange={(e) => setPrompt(e.target.value)}
                rows={3} disabled={busy}
                placeholder="e.g. plain studio-grey background, add glasses, warmer lighting"
                className="w-full rounded-lg bg-surface-raised border border-border text-content text-sm p-2 resize-y disabled:opacity-40" />
            </label>

            {/* Engine pills. `flex-wrap` + `min-w-0` is what keeps five of them on
                a 400px screen: they wrap onto a second and third row instead of
                stretching the modal past the viewport. */}
            <div className="flex items-center gap-1.5 flex-wrap min-w-0">
              <span className="text-content-subtle text-xs w-full sm:w-auto">Engine</span>
              {options.map((o) => (
                <button key={o.engine} type="button" onClick={() => setEngine(o.engine)} disabled={busy}
                  aria-pressed={engine === o.engine}
                  title={o.blocked || undefined}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold disabled:opacity-40 ${engine === o.engine
                    ? 'bg-indigo-500 text-white'
                    : o.usable
                      ? 'bg-surface-raised text-content-muted hover:bg-surface'
                      : 'bg-surface-raised text-content-subtle line-through decoration-1'}`}>
                  {o.label}
                </button>
              ))}
            </div>

            {/* An unavailable LOCAL engine is still selectable — picking it is how
                you find out WHY, and the reason names the one action that fixes
                it. Greying it out silently was the failure mode this replaces. */}
            {current?.blocked && (
              <p className="text-[0.6875rem] text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg px-2.5 py-1.5">
                {current.blocked}
              </p>
            )}

            {refNote && (
              <p className="text-[0.6875rem] text-content-muted">{refNote}</p>
            )}

            {/* Optional extra reference images — transient inputs to THIS edit only,
                never saved as the dataset's extra refs. Hidden entirely for the
                local engines: their graphs take file paths, so these bytes would
                be dropped, and an input whose files vanish is worse than none. */}
            {canAddRefs && (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-content-subtle text-xs">Add reference images (optional)</span>
              {editRefs.map((f, i) => (
                <div key={i} className="relative w-12 h-12 rounded-lg overflow-hidden bg-black shrink-0">
                  <img src={URL.createObjectURL(f)} alt="edit reference" className="w-full h-full object-cover" />
                  <button type="button" disabled={busy}
                    onClick={() => setEditRefs((cur) => cur.filter((_, j) => j !== i))}
                    aria-label="Remove this reference image"
                    className="absolute top-0 right-0 w-4 h-4 flex items-center justify-center rounded-bl bg-black/70 text-white text-[0.625rem] leading-none disabled:opacity-40">✕</button>
                </div>
              ))}
              {editRefs.length < MAX_EDIT_REFS && (
                <button type="button" onClick={() => inpRef.current?.click()} disabled={busy}
                  aria-label="Add a reference image for the edit"
                  className="w-12 h-12 rounded-lg border border-dashed border-border-strong text-content-muted text-lg leading-none disabled:opacity-40">+</button>
              )}
              <input ref={inpRef} type="file" accept="image/*" multiple className="hidden" disabled={busy}
                onChange={(e) => { addRefs(e.target.files); e.target.value = ''; }} />
            </div>
            )}

            {phase === 'failed' && referenceEdit?.error && (
              <div className="flex flex-col gap-1 text-[0.6875rem] bg-red-500/10 border border-red-500/30 rounded-lg px-2.5 py-1.5">
                <p className="text-red-300">{referenceEdit.error}</p>
                <p className="text-content-muted">Engine used for this result: {resultEngineLabel}</p>
              </div>
            )}
            <p className="text-[0.6875rem] text-content-muted">{editCostNote(engine)}</p>
            {/* The reference is what the whole dataset is anchored on, and this
                lane runs on the dataset's Klein model like every other one — so
                say which, and let it be changed from here. Klein only: the other
                local engine (Krea) resolves a global base model, and the API
                engines have no local model at all. */}
            {engine === 'klein' && <KleinModelSetting datasetId={datasetId} />}

            <div className="flex gap-2 justify-end">
              <button type="button" onClick={onClose} disabled={busy}
                className="px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">Cancel</button>
              {phase === 'failed' && (
                <button type="button" onClick={retryEdit} disabled={busy || !canRetryExact}
                  title={canRetryExact ? undefined : 'The original temporary references are no longer available'}
                  className="px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
                  ↻ Retry same edit
                </button>
              )}
              <button type="button" onClick={runEdit} disabled={busy || !!blocked}
                title={blocked || undefined}
                className="px-4 py-2 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                {starting ? 'Starting…' : 'Generate edit'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
