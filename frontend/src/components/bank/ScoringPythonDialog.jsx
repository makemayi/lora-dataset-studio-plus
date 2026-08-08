import { useCallback, useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import {
  canSelect, DEFAULT_PICKER, detectionFailure, detectionSummary, dialogCopy,
  enteredNote, missingLabels, pickerProfile, sortInterpreters, statusBadge,
} from './scoringPython.js'

/** ⚡ Use a GPU Python you already have — the picker behind the "✨ Score runs on
 * the CPU" warning, and behind the same warning for the SigLIP 2 semantic index.
 *
 * ONE dialog for both, driven by `profile`: the detector, the read-only rule and
 * the refusal are identical, only the dependency list and the config key differ
 * and both live on the server. A twin screen would be a second place to fix
 * every future wording bug — and the two would drift the first time one of them
 * was touched.
 *
 * This app never installs into an environment it did not build. ai-toolkit's venv
 * runs the user's training; ComfyUI's runs their generation. We read them, we
 * report exactly what each one is missing, and we hand over the command — the
 * user decides whether to run it. Anything we could not prove able to run the
 * whole pass is refused by the server, so a pick can never turn an hour of
 * scoring into an import error.
 */
const TONE = {
  ok: 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300',
  warn: 'border-amber-400/50 bg-amber-500/10 text-amber-300',
  off: 'border-border bg-surface text-content-subtle',
}

function Badge({ status }) {
  const { tone, label } = statusBadge(status)
  return (
    <span className={`shrink-0 rounded border px-1.5 py-0.5 text-[0.625rem] font-semibold uppercase tracking-wide ${TONE[tone]}`}>
      {label}
    </span>
  )
}

/** Every dependency as a chip, present or not — the per-package answer is the
 *  whole point: "it has CUDA but no OpenCLIP" is actionable, "no" is not. */
function DepChips({ deps }) {
  if (!deps?.length) return null
  return (
    <ul className="flex flex-wrap gap-1">
      {deps.map((d) => (
        <li key={d.label}
          className={`rounded border px-1.5 py-0.5 text-[0.625rem] ${d.present
            ? 'border-emerald-500/40 text-emerald-300/90'
            : 'border-amber-400/50 text-amber-300'}`}>
          {d.present ? '✓' : '✗'} {d.label}
        </li>
      ))}
    </ul>
  )
}

export default function ScoringPythonDialog({ onClose, onChanged,
  profile = DEFAULT_PICKER }) {
  const picker = pickerProfile(profile)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [typed, setTyped] = useState('')

  const load = useCallback(async (opts = {}) => {
    setLoading(true)
    setError('')
    try {
      const qs = new URLSearchParams()
      if (opts.force) qs.set('force', '1')
      if (opts.path) qs.set('path', opts.path)
      const q = qs.toString()
      setResult(await apiFetch(`${picker.endpoint}${q ? `?${q}` : ''}`))
    } catch (e) {
      setError(e.message || 'Could not look for interpreters.')
    } finally {
      setLoading(false)
    }
  }, [picker.endpoint])

  useEffect(() => { load() }, [load])

  const choose = async (path) => {
    setBusy(path || 'default')
    setError('')
    try {
      await postJson(picker.endpoint, { python: path })
      await load({ force: true })
      onChanged?.()
    } catch (e) {
      setError(e.message || 'That interpreter was refused — nothing changed.')
    } finally {
      setBusy('')
    }
  }

  const rows = sortInterpreters(result?.interpreters)
  const hasOverride = Boolean(result?.selected)
  // Until the probe answers, assume a card: claiming "no NVIDIA card" on a
  // machine that has one is the one wrong thing to flash.
  const nvidia = result ? result.nvidia_present !== false : true
  const copy = dialogCopy(nvidia, picker)
  const entered = loading ? null : enteredNote(result)
  // "the search broke" and "there is nothing here" are NOT the same screen.
  const failure = loading ? null : detectionFailure(result)

  return (
    <div role="dialog" aria-modal="true" aria-label={picker.ariaLabel}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-2 sm:p-4">
      <div className="w-full max-w-2xl max-h-[92vh] overflow-y-auto rounded-xl border border-border bg-surface-overlay p-4 shadow-2xl space-y-4 sm:p-5">
        <div>
          <h2 className="text-base font-bold text-content">{copy.title}</h2>
          <p className="mt-1 text-sm text-content-muted">{copy.intro}</p>
          {picker.extraNote && (
            <p className="mt-2 text-xs text-content-subtle">{picker.extraNote}</p>
          )}
          {!loading && !failure && (
            <p className="mt-2 text-xs text-content-subtle">
              {detectionSummary(rows, nvidia, picker)}
            </p>
          )}
        </div>

        {failure && (
          <div className="rounded-md border border-amber-500/60 bg-amber-500/10 p-3 text-sm text-amber-200 space-y-1">
            <p className="font-semibold">⚠ {failure.title}</p>
            <p className="text-xs text-amber-200/90">{failure.text}</p>
            {failure.detail && (
              <p className="break-all font-mono text-[0.625rem] text-amber-200/70">
                {failure.detail}
              </p>
            )}
          </div>
        )}

        {error && (
          <p className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-xs text-red-300">
            {error}
          </p>
        )}

        {loading ? (
          <p className="text-sm text-content-muted">
            Checking each interpreter — a cold PyTorch import can take a few seconds…
          </p>
        ) : (
          <ul className="space-y-2">
            {rows.map((r) => {
              const missing = missingLabels(r)
              return (
                <li key={r.path}
                  className={`rounded-lg border p-3 space-y-2 ${r.selected
                    ? 'border-emerald-500/50 bg-emerald-500/5' : 'border-border bg-surface'}`}>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold text-content">{r.label}</span>
                    <Badge status={r.status} />
                    {r.selected && (
                      <span className="rounded border border-emerald-500/50 px-1.5 py-0.5 text-[0.625rem] font-semibold uppercase tracking-wide text-emerald-300">
                        In use
                      </span>
                    )}
                  </div>
                  <p className="break-all font-mono text-[0.6875rem] text-content-subtle" title={r.path}>
                    {r.path}
                  </p>
                  <p className="text-xs text-content-muted">
                    {r.detail}
                    {r.python_version && ` · Python ${r.python_version}`}
                    {r.torch_version && ` · torch ${r.torch_version}`}
                  </p>
                  <DepChips deps={r.deps} />
                  {missing.length > 0 && r.install_command && (
                    <div className="space-y-1">
                      <p className="text-xs text-amber-300">
                        We will not install into an environment we did not create. To add
                        {` ${missing.join(', ')}`} yourself, run:
                      </p>
                      <code className="block overflow-x-auto whitespace-pre rounded bg-surface-raised px-2 py-1 font-mono text-[0.6875rem] text-content-muted">
                        {r.install_command}
                      </code>
                    </div>
                  )}
                  <div className="flex flex-wrap gap-2">
                    <button type="button" disabled={!canSelect(r) || !!busy}
                      onClick={() => choose(r.path)}
                      title={canSelect(r) ? `Point ${picker.feature} at this interpreter`
                        : r.selected ? 'Already in use' : 'This one cannot run the pass yet'}
                      className="rounded-md border border-border px-2.5 py-1 text-xs text-content hover:bg-surface-raised disabled:opacity-40 disabled:hover:bg-transparent">
                      {busy === r.path ? 'Checking…' : 'Use this one'}
                    </button>
                  </div>
                </li>
              )
            })}
          </ul>
        )}

        <div className="space-y-2 border-t border-border pt-3">
          <label htmlFor={picker.inputId} className="block text-xs font-medium text-content">
            Not listed? Enter the path to a Python interpreter or its folder
          </label>
          <div className="flex flex-col gap-2 sm:flex-row">
            <input id={picker.inputId} type="text" value={typed} spellCheck={false}
              onChange={(e) => setTyped(e.target.value)}
              placeholder="…/envs/myenv  or  …/envs/myenv/Scripts/python.exe"
              className="min-w-0 flex-1 rounded-md border border-border bg-surface px-2 py-1 font-mono text-xs text-content" />
            <button type="button" disabled={!typed.trim() || loading}
              onClick={() => load({ force: true, path: typed.trim() })}
              className="rounded-md border border-border px-2.5 py-1 text-xs text-content hover:bg-surface-raised disabled:opacity-40">
              Check it
            </button>
          </div>
          {entered ? (
            <p className={`text-[0.6875rem] ${entered.tone === 'warn'
              ? 'text-amber-300' : 'text-content-muted'}`}>
              {entered.text}
            </p>
          ) : (
            <p className="text-[0.6875rem] text-content-subtle">
              An interpreter, or the folder holding it — a venv, a conda env, a portable
              bundle, anywhere on any disk. Checked and added to the list above; it is
              only used once you pick it.
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3">
          <button type="button" onClick={() => load({ force: true })} disabled={loading || !!busy}
            title="Re-check every interpreter — use this right after installing a package"
            className="rounded-md border border-border px-2.5 py-1 text-xs text-content-muted hover:text-content hover:bg-surface-raised disabled:opacity-40">
            ↻ Check again
          </button>
          <div className="flex flex-wrap gap-2">
            {hasOverride && (
              <button type="button" onClick={() => choose('')} disabled={!!busy}
                title="Go back to the environment the app set up for this pass"
                className="rounded-md border border-border px-2.5 py-1 text-xs text-content-muted hover:text-content hover:bg-surface-raised disabled:opacity-40">
                Back to the app default
              </button>
            )}
            <button type="button" onClick={onClose}
              className="rounded-md border border-border px-3 py-1 text-xs text-content hover:bg-surface-raised">
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
