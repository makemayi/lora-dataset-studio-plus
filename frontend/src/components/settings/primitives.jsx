import { useState } from 'react'
import { postJson } from '../../api/fetchClient'
import { CARD_SURFACE_INTERACTIVE, INPUT_CLASS } from '../common/surfaces'

/* The card surface and the input measure moved to components/common/surfaces.js
   when the redesign left Settings — the bank, and the pages after it, need the
   same ones. Re-exported because ten settings sections import INPUT_CLASS from
   here. */
export { INPUT_CLASS }

/* Section heading: a small mono "rack tag" eyebrow above the title keeps every
   settings/guide section labeled the same way without shouting. */
/* The `key`s are load-bearing, and they are about Chrome auto-translate.
   Translate REPLACES a text node with its own <font> wrapper. React still holds
   the original node, so when the route changes and React updates the heading it
   writes `nodeValue` on a node that is no longer in the document: the new title
   is applied to nothing and the OLD one stays on screen. Observed on
   2026-08-10 — the Server page rendered "Image engines" over Storage's
   description, three pages behind whatever had been visited first.

   Keying on the text makes it a remount instead of a text update: React removes
   the <h1> (an element it does own — translate rewrites what is inside it, not
   the element itself) and mounts a fresh one. No crash, and the heading is
   right. Same reason `hidden` is used instead of a ternary elsewhere: never ask
   React to edit a text node translate may have taken. */
export function SectionHeader({ eyebrow, title, description, badge }) {
  return (
    <div>
      <p key={eyebrow} className="font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">{eyebrow}</p>
      <h1 key={title} className="mt-1 flex items-center gap-2 text-xl font-semibold text-content">
        {title}{badge}
      </h1>
      {description && <p key={description} className="mt-1 text-sm text-content-muted">{description}</p>}
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

/* Most field help is not a bare string — it carries <span className="font-medium">
   emphasis, a <code>, an <a> to the guide. Measuring `children.length` would see
   an array of three and call a 500-character paragraph short, which is exactly
   how the first pass folded thirteen card blurbs and left thirty field blurbs
   open. Walk the tree instead. Interpolated values count as ~1 char each; that
   is close enough for a threshold nothing sits near. */
function plainLength(node) {
  if (node === null || node === undefined || node === false || node === true) return 0
  if (typeof node === 'string') return node.length
  if (typeof node === 'number') return String(node).length
  if (Array.isArray(node)) return node.reduce((n, child) => n + plainLength(child), 0)
  if (node.props) return plainLength(node.props.children)
  return 0
}

/* The caller's className positions the help relative to its input (`mt-0.5`
   under a field, `mt-3` under a card's grid). That margin belongs to whatever
   element is actually in the flow — the <p> when open, the <details> when
   folded — while the rest of the classes style the text. Leaving `mt-3` on the
   inner <p> of a <details> would both indent the body wrongly and fight the
   `mt-1.5` this component wants there (same specificity: Tailwind's own order
   decides, not the order written here). */
const MARGIN_CLASS_RE = /(?:^|\s)-?m[tby]-[^\s]+/g

function splitMargins(className) {
  const margins = (className.match(MARGIN_CLASS_RE) || []).map((c) => c.trim())
  return { outer: margins.join(' '), text: className.replace(MARGIN_CLASS_RE, ' ').trim() }
}

/* A caret, NOT a round "?" — that glyph is taken. `help/HelpMode.jsx` renders a
   round indigo "?" badge beside titles in Help mode, and clicking it jumps to
   the Guide. A second round "?" that expands text in place, sometimes on the
   same row, is two meanings wearing one face. The caret is also just honest:
   this is a disclosure triangle, so it looks like one and rotates like one.

   `summary` defaults to the short form because most callers are FIELDS, where
   the disclosure sits directly under a labelled input and a long label repeated
   five times down one card is the same visual noise the fold was meant to
   remove. `Card` passes the long form: it introduces a whole section, and there
   is only ever one of it. */
export function HelpText({ children, className = '', summary = 'More' }) {
  if (!children) return null
  const { outer, text } = splitMargins(className)
  if (plainLength(children) <= HELP_FOLD_OVER) {
    return <p className={`max-w-prose ${outer} ${text}`}>{children}</p>
  }
  return (
    <details className={`group max-w-prose ${outer}`}>
      <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-xs text-content-subtle hover:text-content-muted [&::-webkit-details-marker]:hidden">
        <span aria-hidden="true"
          className="inline-block text-[0.625rem] leading-none transition-transform duration-150 group-open:rotate-90">
          ▶
        </span>
        {summary}
      </summary>
      <p className={`mt-1.5 ${text}`}>{children}</p>
    </details>
  )
}

/* The surface itself, and why it is a shadow and not an outline, is documented
   in components/common/surfaces.js. */
export function Card({ title, help, children, id }) {
  return (
    <section id={id} className={`scroll-mt-24 p-4 ${CARD_SURFACE_INTERACTIVE}`}>
      <h2 className="text-[0.9375rem] font-semibold tracking-[-0.01em] text-content">{title}</h2>
      <div className="mt-1">
        <HelpText summary="Why this matters"
          className="text-[0.8125rem] leading-relaxed text-content-muted">{help}</HelpText>
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
