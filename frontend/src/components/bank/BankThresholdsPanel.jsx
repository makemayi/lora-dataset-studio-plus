/* 🎚 Filter thresholds — the twelve knobs that decide every Bank flag, edited
   WHERE THE TRIAGE HAPPENS instead of three screens away.

   WHY IT EXISTS. Tightening duplicate detection used to mean leaving the bank
   you were working in, walking to Settings ▸ Captioning & quality, editing a
   number with no idea what it would do, saving, and coming back to find out.
   Everything here is the same setting — the SAME `config.bank.<key>`, written
   through the SAME PUT /api/settings that Settings uses. There is no second
   copy and nothing to keep in sync: the two screens are one value seen twice.
   Editing here therefore changes it for EVERY bank, which the header says in
   plain words, above the fold, before anything is unfolded — that is the one
   real hazard of a shared setting and the only defence is to state it.

   Three things this panel does that a bare list of numbers cannot:

   1. It says WHICH WAY each knob catches more, as visible text. "Stricter" is
      not a direction: `dup_distance` is a DISTANCE (raise it to catch more) and
      `semantic_dup_threshold` is a SIMILARITY (lower it to catch more), and they
      sit next to each other. Getting one backwards looks exactly like the
      feature not working.
   2. It says WHEN a change lands — eight re-sort the bank the instant they are
      saved because the scan persists raw scores and verdicts are recomputed at
      read time; four are baked into stored groups by a pass. Those four carry
      the button that re-runs their own pass, so "applies at the next pass" is a
      gesture rather than an errand.
   3. It shows THE EFFECT: for the eight read-time thresholds it asks the server
      how many images the candidate value WOULD flag, live, before saving —
      "1 240 → 3 019 images flagged". The other four say they cannot, instead of
      showing a number that would describe the previous grouping.

   All wording, grouping, direction and validation live in bankThresholds.js so
   `node --test` can execute them (it parses .js, not .jsx), and every field is
   rendered from that table through a (section, field) pair — nothing here is
   wired to a specific knob, so a future per-bank override layer is a change of
   one resolver, not of twelve fields.

   The per-field reset is the SHARED settings button: one vocabulary for "Reset
   to default" across the app, and its value comes from the server's
   `config_defaults`, never from a number typed into the frontend. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiFetch, postJson, putJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import ResetToDefault from '../settings/ResetToDefault.jsx'
import {
  APPLIES, BANK_SECTION, THRESHOLD_GROUPS,
  coerceValue, customisedCountInGroup, customisedFields, directionHint,
  directionSummary, dirtyFields, effectLine, isValidValue, mergedBank,
  rerunFor, resetAllEdits, thresholdsInGroup,
} from './bankThresholds.js'

// Debounce for the live effect count: long enough that typing "140" is one
// request instead of three, short enough to still feel attached to the keystroke.
const PREVIEW_DEBOUNCE_MS = 350

const INPUT = 'mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 ' +
  'text-sm text-content tabular-nums focus:border-indigo-400 focus:outline-none'

const SMALL_BTN = 'rounded-md border border-border px-2 py-1 text-xs text-content-muted ' +
  'hover:bg-surface-raised hover:text-content disabled:opacity-50'

/** One collapsible group of thresholds. A folded group still advertises how
    many of its fields you have moved off the default — otherwise folding it
    would hide a change you made and forgot. */
function Group({ group, open, onToggle, customised, children }) {
  const panelId = `bank-th-group-${group.id}`
  return (
    <section className="rounded-lg border border-border bg-surface">
      <button type="button" onClick={onToggle} aria-expanded={open} aria-controls={panelId}
        className="flex w-full items-center gap-2 px-3 py-2 text-left">
        <span aria-hidden className="text-sm">{group.emoji}</span>
        <span className="text-sm font-medium text-content">{group.label}</span>
        {customised > 0 && (
          <span className="rounded-full border border-indigo-400/50 bg-indigo-500/10 px-1.5 text-[10px] font-semibold text-indigo-200">
            {customised} customised
          </span>
        )}
        <span aria-hidden className="ml-auto text-xs text-content-subtle">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div id={panelId} className="space-y-3 border-t border-border px-3 py-3">
          <p className="text-xs text-content-subtle">{group.blurb}</p>
          {/* ONE column below ~640 px. Twelve numeric fields two-abreast on a
              400 px phone is precisely the layout that stops being readable. */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">{children}</div>
        </div>
      )}
    </section>
  )
}

export default function BankThresholdsPanel({ bankId, onSaved, onRunPass }) {
  const toast = useToast()
  const [saved, setSaved] = useState(null)          // config.bank as stored
  const [configDefaults, setConfigDefaults] = useState(null)
  const [edits, setEdits] = useState({})            // field -> candidate value
  const [open, setOpen] = useState(() => Object.fromEntries(
    THRESHOLD_GROUPS.map((g) => [g.id, g.defaultOpen])))
  const [busy, setBusy] = useState(false)
  const [loadError, setLoadError] = useState(null)
  const [baseFlags, setBaseFlags] = useState(null)  // counts at the SAVED values
  const [previewFlags, setPreviewFlags] = useState(null)
  const previewTimer = useRef(null)

  // Saved values and shipped defaults come from the SAME payload the Settings
  // page reads, so the two screens cannot disagree about either one.
  const load = useCallback(async () => {
    try {
      const d = await apiFetch('/api/settings')
      setSaved(d.config?.[BANK_SECTION] || {})
      setConfigDefaults(d.config_defaults || null)
      setLoadError(null)
    } catch {
      setLoadError('Could not load the thresholds — Settings ▸ Captioning & quality still edits them.')
    }
  }, [])

  useEffect(() => { load() }, [load])

  // Counts at the SAVED thresholds: the "before" half of every effect line.
  // `background` marks it as a poll the user did not ask for, so an unreachable
  // server does not raise a notification for it. Both preview calls carry it,
  // and both also swallow their own failure — the readout simply does not
  // appear, which is the honest outcome of "we could not count".
  useEffect(() => {
    if (!saved) return undefined
    let alive = true
    postJson(`/api/bank/${bankId}/flag-preview`, {}, { background: true })
      .then((d) => { if (alive && d) setBaseFlags(d.flags) })
      .catch(() => {})
    return () => { alive = false }
  }, [bankId, saved])

  const candidate = useMemo(() => mergedBank(saved, edits), [saved, edits])
  const dirty = useMemo(() => dirtyFields(saved, edits), [saved, edits])
  const customised = useMemo(() => customisedFields(saved, configDefaults),
    [saved, configDefaults])

  // Live effect: ask the server what the candidate numbers WOULD flag. Read-only
  // COUNTs over already-stored scores — no pass runs and nothing is saved. It
  // fires while a value is being nudged, so it is `background`: a momentarily
  // unreachable server must not raise one notification per keystroke.
  useEffect(() => {
    if (!saved || dirty.length === 0) { setPreviewFlags(null); return undefined }
    clearTimeout(previewTimer.current)
    let alive = true
    previewTimer.current = setTimeout(() => {
      postJson(`/api/bank/${bankId}/flag-preview`, { thresholds: candidate },
        { background: true })
        .then((d) => { if (alive && d) setPreviewFlags(d.flags) })
        .catch(() => {})
    }, PREVIEW_DEBOUNCE_MS)
    return () => { alive = false; clearTimeout(previewTimer.current) }
  }, [bankId, candidate, dirty.length, saved])

  // The (section, field) writer the shared ResetToDefault button expects. The
  // section is always ours, so a future per-bank layer changes THIS function
  // and nothing else in the panel.
  const setField = useCallback((_section, field, value) => {
    setEdits((e) => ({ ...e, [field]: value }))
  }, [])

  const save = useCallback(async () => {
    // Only valid, actually-changed fields ride to the server: PUT /api/settings
    // deep-merges a partial, so nothing else in config.json is touched.
    const partial = {}
    for (const f of dirty) partial[f] = Number(edits[f])
    if (Object.keys(partial).length === 0) return
    setBusy(true)
    try {
      const d = await putJson('/api/settings', { config: { [BANK_SECTION]: partial } })
      setSaved(d.config?.[BANK_SECTION] || {})
      setConfigDefaults(d.config_defaults || configDefaults)
      setEdits({})
      setPreviewFlags(null)
      toast.success('Thresholds saved — this bank re-sorts on the read-time ones.')
      onSaved?.()
    } catch {
      toast.error('Could not save the thresholds.')
    } finally {
      setBusy(false)
    }
  }, [dirty, edits, configDefaults, toast, onSaved])

  if (loadError) {
    return <p className="rounded-lg border border-border bg-surface px-3 py-2 text-xs text-amber-300">{loadError}</p>
  }
  if (!saved) {
    return <p className="text-xs text-content-subtle">Loading thresholds…</p>
  }

  const resetAll = resetAllEdits(configDefaults)
  const canResetAll = customised.length > 0 && Object.keys(resetAll).length > 0

  return (
    <div className="space-y-3">
      {/* Scope, in words, before any group is unfolded. */}
      <div className="flex flex-wrap items-start gap-x-3 gap-y-2 rounded-lg border border-border bg-surface-raised px-3 py-2">
        <p className="min-w-[12rem] flex-1 text-xs text-content-muted">
          These apply to <strong className="text-content">every bank</strong> — they are the same
          values as Settings ▸ Captioning &amp; quality. Most re-sort this bank the moment you
          save, with no rescan.
        </p>
        {canResetAll && (
          <button type="button" onClick={() => setEdits(resetAll)}
            className="rounded-md border border-border-strong px-2 py-1 text-xs font-medium text-content hover:bg-surface">
            <span aria-hidden>↺ </span>Reset all to defaults
          </button>
        )}
      </div>

      {THRESHOLD_GROUPS.map((g) => (
        <Group key={g.id} group={g} open={!!open[g.id]}
          customised={customisedCountInGroup(g.id, saved, configDefaults)}
          onToggle={() => setOpen((o) => ({ ...o, [g.id]: !o[g.id] }))}>
          {thresholdsInGroup(g.id).map((t) => {
            const value = edits[t.field] !== undefined ? edits[t.field] : (saved[t.field] ?? '')
            const invalid = edits[t.field] !== undefined && !isValidValue(t, edits[t.field])
            const effect = effectLine(t, previewFlags, baseFlags)
            const rerun = rerunFor(t)
            return (
              <div key={t.field}>
                <label htmlFor={`bank-th-${t.field}`}
                  className="block text-sm font-medium text-content">
                  {t.label}
                  <span className="ml-1 font-normal text-content-subtle">({t.unit})</span>
                </label>
                <input
                  id={`bank-th-${t.field}`}
                  type="number"
                  inputMode="decimal"
                  min={t.min} max={t.max} step={t.step}
                  value={value}
                  // The count is part of the field's DESCRIPTION, not a live
                  // region: it is then read when you land on the field, instead
                  // of being announced again on every nudge of the value.
                  aria-describedby={`bank-th-${t.field}-hint bank-th-${t.field}-effect`}
                  aria-invalid={invalid || undefined}
                  onChange={(e) => setField(BANK_SECTION, t.field, coerceValue(t, e.target.value))}
                  className={`${INPUT} ${invalid ? 'border-rose-400' : ''}`}
                />
                <div id={`bank-th-${t.field}-hint`} className="mt-1 space-y-0.5 text-xs">
                  {/* Direction FIRST and visible — the one fact that keeps the
                      knob from being set backwards. */}
                  <span className="block font-medium text-content-muted">{directionHint(t)}</span>
                  <span className="block text-content-subtle">{t.hint}</span>
                  <span className="block text-content-subtle">{APPLIES[t.applies]}</span>
                  {invalid && (
                    <span className="block text-rose-300">
                      Enter a number{t.min !== undefined && t.max !== undefined
                        ? ` between ${t.min} and ${t.max}` : ''}.
                    </span>
                  )}
                </div>
                {/* The effect readout. Visually a live number; for assistive tech
                    it is a stable sentence reachable from the input, so nudging a
                    value does not fire an announcement per keystroke. */}
                <p className="mt-1 min-h-[1rem] text-xs font-medium text-indigo-200" aria-hidden="true">
                  {effect}
                </p>
                <span id={`bank-th-${t.field}-effect`} className="sr-only">
                  {effect ? `${effect}. ${directionSummary(t)}` : directionSummary(t)}
                </span>
                {rerun && (
                  <div className="mt-1 flex flex-wrap items-center gap-2">
                    <button type="button" className={SMALL_BTN}
                      onClick={() => onRunPass?.(rerun.endpoint)}
                      title={rerun.note}>
                      {rerun.label}
                    </button>
                    <span className="text-[11px] text-content-subtle">{rerun.note}</span>
                  </div>
                )}
                <ResetToDefault
                  label={t.label}
                  section={BANK_SECTION}
                  field={t.field}
                  config={{ [BANK_SECTION]: candidate }}
                  configDefaults={configDefaults}
                  setField={setField}
                  value={value}
                />
              </div>
            )
          })}
        </Group>
      ))}

      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={save} disabled={busy || dirty.length === 0}
          className="rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50">
          {busy ? 'Saving…' : `Save${dirty.length ? ` (${dirty.length})` : ''}`}
        </button>
        {dirty.length > 0 && (
          <button type="button" onClick={() => { setEdits({}); setPreviewFlags(null) }}
            className="rounded-md border border-border px-3 py-1.5 text-sm text-content-muted hover:text-content">
            Discard changes
          </button>
        )}
        {/* A two-state sentence, not a running count: it changes when you start
            and when you stop having unsaved work, never as a value is nudged. */}
        <span aria-live="polite" className="text-xs text-content-subtle">
          {dirty.length === 0 ? 'No unsaved threshold changes.' : 'You have unsaved threshold changes.'}
        </span>
      </div>
    </div>
  )
}
