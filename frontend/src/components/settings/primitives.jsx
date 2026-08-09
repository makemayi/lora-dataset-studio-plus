import { useState } from 'react'
import { postJson } from '../../api/fetchClient'

/* `max-w-xl` is the tidy-up: the settings shell went full-width on 2026-08-09
   and every `w-full` control went with it, so a text input stretched ~1400px
   across while the sliders beside it stayed 112px — the raggedness the user
   reported. A control has a comfortable measure regardless of how wide the
   window is; the CARD still fills the pane, only its contents are bounded. */
export const INPUT_CLASS =
  'mt-1 w-full max-w-xl rounded-md border border-border-strong bg-surface-raised px-3 py-2 text-sm text-content ' +
  'placeholder:text-content-subtle focus:border-primary focus:outline-none'

/* Section heading: a small mono "rack tag" eyebrow above the title keeps every
   settings/guide section labeled the same way without shouting. */
export function SectionHeader({ eyebrow, title, description, badge }) {
  return (
    <div>
      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">{eyebrow}</p>
      <h1 className="mt-1 flex items-center gap-2 text-xl font-semibold text-content">
        {title}{badge}
      </h1>
      {description && <p className="mt-1 text-sm text-content-muted">{description}</p>}
    </div>
  )
}

// Status is never color-only: an explicit glyph + text label carries the
// meaning, color is a reinforcing cue on top.
export function StatusBadge({ ok, okLabel = 'Configured', missingLabel = 'Not set' }) {
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium ${ok ? 'text-emerald-400' : 'text-content-subtle'}`}>
      <span aria-hidden="true">{ok ? '✓' : '✗'}</span>
      {ok ? okLabel : missingLabel}
    </span>
  )
}

export function TestResult({ result }) {
  if (!result) return null
  const level = result.severity === 'warning' || result.code === 'broad_access'
    ? 'warning'
    : result.ok ? 'success' : 'error'
  const presentation = {
    success: { glyph: '\u2713', label: 'Success', className: 'text-emerald-400' },
    warning: { glyph: '\u26A0', label: 'Warning', className: 'text-amber-400' },
    error: { glyph: '\u2717', label: 'Error', className: 'text-rose-400' },
  }[level]
  const detail = level === 'warning' ? (result.warning || result.detail) : result.detail
  return (
    <p role={level === 'error' ? 'alert' : 'status'} aria-live="polite"
      className={`text-xs ${presentation.className}`}>
      <span aria-hidden="true">{presentation.glyph}</span>{' '}
      <span className="sr-only">{presentation.label}: </span>{detail}
    </p>
  )
}

export function TestButton({ target, onResult, beforeTest }) {
  const [busy, setBusy] = useState(false)
  const run = async () => {
    setBusy(true)
    try {
      // Secret fields pass beforeTest to persist the value still sitting in the
      // write-only input: the probe reads the SAVED key, so testing an unsaved
      // paste would always answer "key missing".
      if (beforeTest) await beforeTest()
      onResult(await postJson(`/api/settings/test/${target}`, {}))
    } catch (e) {
      onResult({ ok: false, detail: e.message || 'Test failed' })
    } finally {
      setBusy(false)
    }
  }
  return (
    <button
      type="button"
      onClick={run}
      disabled={busy}
      className="shrink-0 rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content hover:bg-surface-raised disabled:opacity-50"
    >
      {busy ? 'Testing…' : 'Test'}
    </button>
  )
}


/* ── Help copy: shown when it is a sentence, folded when it is a paragraph ────
   This page carries ~6,500 characters of explanatory prose and it is the most
   valuable thing about the app — it is why a setting is understandable at all.
   It is also why the page felt heavy: eleven of the thirteen card blurbs run
   263–1006 characters and every one of them was open, forever, competing with
   the controls and with each other.

   So: fold the paragraphs, keep the one-liners. The split in the real copy is
   bimodal (two blurbs at 73 chars, the rest 263+), so the threshold is not a
   fussy judgement call — nothing sits near it.

   A native <details> on purpose, not a React toggle. help/revealTarget.js
   already walks up from a deep-linked field and opens every collapsed <details>
   ancestor, so a Guide link into a folded explanation keeps working with no new
   machinery. It also sidesteps the Chrome-auto-translate crash: the summary and
   the body are both always mounted, so React never removes a text node. */
const HELP_FOLD_OVER = 140

export function HelpText({ children, className = '', summary = 'Why this matters' }) {
  if (!children) return null
  const long = typeof children === 'string' && children.length > HELP_FOLD_OVER
  if (!long) {
    return <p className={`max-w-prose ${className}`}>{children}</p>
  }
  return (
    <details className="group max-w-prose">
      <summary className="inline-flex cursor-pointer list-none items-center gap-1.5 text-xs text-content-subtle hover:text-content-muted [&::-webkit-details-marker]:hidden">
        <span aria-hidden="true"
          className="grid h-3.5 w-3.5 place-items-center rounded-full bg-surface-raised text-[0.5625rem] leading-none">
          ?
        </span>
        {summary}
      </summary>
      <p className={`mt-1.5 ${className}`}>{children}</p>
    </details>
  )
}

/* Chrome's own Settings is the reference here (the user's, 2026-08-09): a card
   is a raised surface with a soft shadow, not a boxed-in outline, and it lifts
   slightly on hover. Two reasons that reads better than what was here:

   * the tokens were ALREADY an elevation system — `surface` is white at 4% and
     `surface-raised` at 9% — so adding `border border-border` on top meant two
     separation mechanisms fighting for the same job, and thirteen of these
     stacked down one page turned into a grid of boxes;
   * on a near-black ground a hairline reads as a hard edge, while a shadow
     reads as depth, which is what the hierarchy actually is.

   The shadow is deliberately heavier than a light-theme card's: at these
   background values a subtle one is invisible. Hover is a lift, not a colour
   change, so it never competes with the accent. */
export function Card({ title, help, children, id }) {
  return (
    <section
      id={id}
      className="scroll-mt-24 rounded-xl bg-surface p-4 shadow-[0_1px_2px_rgba(0,0,0,.35),0_4px_16px_-6px_rgba(0,0,0,.5)]
        transition-shadow duration-200
        hover:shadow-[0_1px_2px_rgba(0,0,0,.4),0_10px_28px_-8px_rgba(0,0,0,.65)]"
    >
      <h2 className="text-[0.9375rem] font-semibold tracking-[-0.01em] text-content">{title}</h2>
      <div className="mt-1">
        <HelpText className="text-[0.8125rem] leading-relaxed text-content-muted">{help}</HelpText>
      </div>
      <div className="mt-3 space-y-3">{children}</div>
    </section>
  )
}

/* `warn` (optional): an amber note UNDER the input, for a value that saves fine but
   will not work — a folder that isn't on disk, say. Distinct from `help` (above the
   input, always-on guidance) so a real problem can't read as documentation.
   `children` renders after it, for a field that needs its own action button. */
export function TextField({ id, label, value, onChange, placeholder, help, warn, children }) {
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium text-content">{label}</label>
      <HelpText className="mb-1 text-xs text-content-muted">{help}</HelpText>
      <input
        id={id}
        type="text"
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className={INPUT_CLASS}
      />
      {warn && (
        <p className="mt-1 break-words text-xs text-amber-400">
          <span aria-hidden="true">⚠</span> {warn}
        </p>
      )}
      {children}
    </div>
  )
}

/* One saved-secret row: write-only password input + presence badge + optional
   Test (persists the pending paste first) + Remove. `field` comes from a
   SECRET_FIELDS-style descriptor: { key, label, testTarget, help, guide? }. */
export function SecretField({
  field, secretsPresence, secretInputs, setSecretInputs,
  testResults, recordTestResult, saveSecretIfPending, handleDeleteSecret,
}) {
  const f = field
  return (
    /* A GRID, not a flex row. Flex sized the input from what was left over, and
       what was left over depended on how many buttons that key happened to have
       — Test + Remove, Test alone, Remove alone — so no two inputs in the card
       ended at the same x. The action column is now a fixed track whether or not
       anything sits in it, and every input lines up. Under `sm` it collapses to
       one column so the buttons drop below instead of crushing the field. */
    <div className="grid items-end gap-x-3 gap-y-2 sm:grid-cols-[minmax(0,1fr)_9.5rem]">
      <div className="min-w-0">
        <div className="flex items-center justify-between">
          <label htmlFor={f.key} className="block text-sm font-medium text-content">{f.label}</label>
          <StatusBadge ok={!!secretsPresence[f.key]} />
        </div>
        <HelpText className="mb-1 text-xs text-content-muted">{f.help}</HelpText>
        {f.guide}
        <input
          id={f.key}
          type="password"
          autoComplete="off"
          value={secretInputs[f.key] ?? ''}
          onChange={(e) => setSecretInputs((prev) => ({ ...prev, [f.key]: e.target.value }))}
          placeholder={secretsPresence[f.key] ? 'Already set — enter a new value to replace it' : 'Not set'}
          className={INPUT_CLASS}
        />
        {f.testTarget && <TestResult result={testResults[f.testTarget]} />}
      </div>
      <div className="flex items-center gap-2">
        {f.testTarget && (
          <TestButton target={f.testTarget} beforeTest={() => saveSecretIfPending(f.key)}
            onResult={(r) => recordTestResult(f.testTarget, r)} />
        )}
        {secretsPresence[f.key] && (
          <button
            type="button"
            onClick={() => handleDeleteSecret(f.key, f.label)}
            title={`Remove the saved ${f.label}`}
            className="shrink-0 rounded-md border border-rose-500/40 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/10"
          >
            Remove
          </button>
        )}
      </div>
    </div>
  )
}
