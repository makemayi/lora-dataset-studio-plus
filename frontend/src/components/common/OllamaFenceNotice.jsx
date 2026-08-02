/* 🔒 The refusal that carries its own way out.
 *
 * Rendered by any surface that runs a local Ollama call through
 * useOllamaFence. It never appears unless the fence actually blocked
 * something. Wording and state live in utils/ollamaFence.js; this is pixels.
 *
 * Stacks on a 400 px screen, one row from `sm` up — this shows up next to a
 * small button in a modal, so it has to survive being narrow.
 */
import { fenceNoticeModel } from '../../utils/ollamaFence';

export default function OllamaFenceNotice({ fence, onUnload, onStop, className = '' }) {
  const model = fenceNoticeModel(fence);
  if (!model) return null;

  const waiting = model.tone === 'waiting' || model.tone === 'busy';
  return (
    <div role="status" aria-live="polite"
      className={`rounded-lg border px-3 py-2 text-[0.6875rem] ${waiting
        ? 'border-amber-400/40 bg-amber-500/10'
        : 'border-rose-400/40 bg-rose-500/10'} ${className}`}>
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        {model.busy
          ? <span className="inline-block w-3 h-3 border-2 border-amber-400/40 border-t-amber-400 rounded-full animate-spin" aria-hidden />
          : <span aria-hidden>🔒</span>}
        <span className="font-medium text-content">{model.headline}</span>
      </div>
      <p className="mt-1 mb-0 text-content-subtle leading-snug">{model.detail}</p>
      {(model.canUnload || model.canCancel) && (
        <div className="mt-2 flex flex-col gap-2 sm:flex-row sm:items-center">
          {model.canUnload && (
            <button type="button" onClick={onUnload} disabled={model.busy}
              className="rounded-md border border-amber-400/50 bg-amber-500/20 px-3 py-1.5
                         font-medium text-content hover:bg-amber-500/30
                         disabled:cursor-not-allowed disabled:opacity-60">
              {model.unloadLabel}
            </button>
          )}
          {model.canCancel && (
            <button type="button" onClick={onStop}
              className="text-left text-content-subtle underline hover:text-content">
              Stop waiting
            </button>
          )}
        </div>
      )}
    </div>
  );
}
