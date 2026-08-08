import {
  offersSemanticGpuPython, SCORE_STAYS_CLIP_SENTENCE, SEMANTIC_CACHE_SENTENCE,
  SEMANTIC_ENGINE_OPTIONS, semanticDeviceNote, semanticIndexActionLabel,
  semanticPrerequisite, semanticPurposeSentence,
} from './bankSemanticEngine.js'
import { openerLabel } from './scoringPython.js'

/** A deliberately small Bank-local choice. Changing it only selects the
 * persisted cache/group lane; building either remains a second gesture. */
export default function BankSemanticEngine({ state, disabled = false,
  switching = false, live = false, capsLoading = false, gpuPresent = true,
  onChange, onIndex, onPickPython }) {
  const action = semanticIndexActionLabel(state)
  const blocked = semanticPrerequisite(state)
  const ready = state.ready && !(state.engine === 'siglip2' && capsLoading)
  const status = !state.hasStatus
    ? 'reading readiness…'
    : ready
    ? `${state.indexed.toLocaleString()} of ${state.total.toLocaleString()} image(s) ready`
    : state.indexed > 0
      ? `${state.indexed.toLocaleString()} image(s) indexed, but the selected engine is not ready`
      : 'Not ready yet'

  // Which device the index will really use, and — when a card sits unused — the
  // way out. Same offer Score has had, same detector behind it.
  const deviceNote = capsLoading ? null : semanticDeviceNote(state, gpuPresent)
  const offerPython = Boolean(onPickPython) && !capsLoading
    && offersSemanticGpuPython(state, gpuPresent)

  return (
    <fieldset className="rounded-lg border border-indigo-400/30 bg-indigo-500/5 p-3 space-y-2"
      disabled={disabled || switching || live}>
      <legend className="px-1 text-xs font-semibold uppercase tracking-wide text-content-muted">
        Semantic engine
      </legend>
      <div className="flex flex-wrap gap-x-4 gap-y-2">
        {SEMANTIC_ENGINE_OPTIONS.map((option) => (
          <label key={option.id} className="flex items-start gap-2 text-sm text-content">
            <input type="radio" name="bank-semantic-engine" value={option.id}
              checked={state.engine === option.id}
              onChange={() => onChange(option.id)} className="mt-0.5 accent-primary" />
            <span>
              <span className="font-medium">{option.label}</span>
              <span className="block text-[11px] text-content-subtle">{option.hint}</span>
            </span>
          </label>
        ))}
      </div>

      <div className="space-y-1 text-xs text-content-muted">
        <p className="m-0">{semanticPurposeSentence(state.engine)}</p>
        <p className="m-0">{SCORE_STAYS_CLIP_SENTENCE}</p>
        <p className="m-0">{SEMANTIC_CACHE_SENTENCE}</p>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-indigo-400/20 pt-2 text-xs">
        <span className={ready ? 'text-emerald-300' : 'text-amber-300'}>
          {ready ? '✓' : '○'} {state.label}:{' '}
          {state.engine === 'siglip2' && capsLoading ? 'checking the Quality tool…' : status}
        </span>
        {state.engine === 'clip' && (
          <span className="text-content-subtle">Produced by ✨ Score.</span>
        )}
        {state.engine === 'siglip2' && !capsLoading && action && (
          <button type="button" onClick={onIndex} disabled={disabled || switching || live}
            title={state.complete
              ? 'Rebuild only the SigLIP 2 cache. The CLIP/Score cache is preserved.'
              : 'Build or resume the whole-Bank SigLIP 2 semantic index.'}
            className="rounded-md border border-indigo-400/40 bg-indigo-500/10 px-2 py-0.5 font-medium text-indigo-200 hover:bg-indigo-500/20 disabled:opacity-50">
            {action}
          </button>
        )}
        {state.engine === 'siglip2' && !capsLoading && !state.installed && (
          <span className="text-content-subtle">
            {blocked}{' '}
            <a href="#/setup?step=quality"
              className="font-medium text-sky-300 underline underline-offset-2 hover:text-sky-200">
              Open Setup ▸ Quality tools
            </a>
          </span>
        )}
        {switching && <span className="text-content-subtle">Saving choice…</span>}
      </div>

      {deviceNote && (
        <div className="space-y-2 border-t border-indigo-400/20 pt-2">
          <p className={`m-0 text-xs ${deviceNote.tone === 'warn'
            ? 'text-amber-400/90' : 'text-content-subtle'}`}>
            {deviceNote.text}
          </p>
          {offerPython && (
            <button type="button" onClick={onPickPython}
              disabled={disabled || switching || live}
              title="Check the Pythons on this machine and point the SigLIP 2 index at one that reaches your GPU. They are read, never changed."
              className="rounded-md border border-amber-400/50 px-2 py-1 text-xs font-medium text-amber-300 hover:bg-amber-500/10 disabled:opacity-50">
              {openerLabel(gpuPresent)}
            </button>
          )}
        </div>
      )}
    </fieldset>
  )
}
