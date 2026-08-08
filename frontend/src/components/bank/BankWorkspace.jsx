import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiFetch, del, patchJson, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
// "This is configurable, here" — a deep link that lands ON the field, not on a tab.
import SettingsLink from '../common/SettingsLink'
import { useCapabilities } from '../../context/CapabilitiesContext'
import { useConnectionStatus } from '../../hooks/useConnectionStatus'
import DupGroupsPanel from './DupGroupsPanel'
import PromoteDialog from './PromoteDialog'
import DeleteRejectedDialog from './DeleteRejectedDialog'
import LaunchAllDialog from './LaunchAllDialog'
import BankSemanticEngine from './BankSemanticEngine.jsx'
import ScoringPythonDialog from './ScoringPythonDialog'
import PipelineReport from './PipelineReport'
import FolderSyncNote from './FolderSyncNote'
import RelocateBankDialog from './RelocateBankDialog'
import BankReviewLightbox from './BankReviewLightbox'
import BankWatermarkPanel from './BankWatermarkPanel'
import BankOverview from './BankOverview.jsx'
// 👤 "Single person here" — a folder the user declares to hold one person.
import SubfolderPersonPanel from './SubfolderPersonPanel'
import { assertionFor, folderMarker, scanOffer, suggestionFor } from './folderPerson.js'
import PersonPreflightDialog from './PersonPreflightDialog'
import { preflightNeeded, preflightWillSample } from './personPreflight.js'
// 🎚 The twelve triage thresholds, edited here instead of in Settings.
import BankThresholdsPanel from './BankThresholdsPanel.jsx'
// Source-folder re-walk messages (pure/testable).
import { spreadReadout, spreadCoverageNote } from './coverageVisual.js'
// Shared with the dataset coverage panel on purpose: both render the SAME
// caption-lexicon payload, so the row/summary logic lives in one place rather
// than being copied and left to drift.
import { axisRows, axisSummary } from '../dataset/datasetCoverage.js'
import { folderSyncToast } from './bankSync.js'
import { UNDO_HINT, undoBannerText, undoOffer, undoResultMessage } from './bankUndo.js'
// Four progress states, not two — including the honest "I don't know" (pure/testable).
import { progressPresence, PROGRESS_HIDDEN, PROGRESS_UNKNOWN, PROGRESS_STALE } from './progressPresence.js'
import { etaPhrase } from './passEta.js'
// A Stop that answers the click locally, and says the phase's price (pure/testable).
import { jobKey, stopLabel, stopNote, stopRequested } from './passStop.js'
// An occupied bank refuses in OUR words, never in the server's (pure/testable).
import { busyRefusal } from './bankPassRun.js'
import { holdsTheGpu, scoreDeviceNote } from './bankScoreDevice.js'
// Wording that adapts to the machine (a card-less box is never sold CUDA).
import { openerLabel, PICKER_PROFILES } from './scoringPython.js'
// Reuse the dataset's register list so the Bank lane never drifts from it — and the
// same ENGINE list, so "which engine" means the same thing on both surfaces.
import {
  CAPTION_LENGTH_OPTIONS, ENGINE_OPTIONS, OLLAMA_RELEVANT, VOCABULARY_OPTIONS,
} from '../dataset/CaptionOptionsPopover'
// Which pile the caption pass is aimed at, and the number the button quotes (pure).
import {
  captionButtonLabel, captionIncludeAssertedLabel,
  captionRecaptionConfirmation, captionRecaptionDisabledReason, captionRecaptionLabel,
  captionRecaptionNote, captionScopeNote, captionScopeStatuses,
} from './bankCaptionScope.js'
// …and the one thing the captioners are known to do badly, said where the engine is
// still free to change. Pure/testable; stays silent unless the scope is measurably
// the material it was measured on (see the module header for the evidence base).
import { captionNsfwNotice } from './captionNsfwNotice.js'
import { passScopeOption } from './bankPassScope.js'
// 🎛 The launch window every pass now opens, and the two pure modules behind it:
// what each pass is (blocks, offered scopes, refusals) and how big a run is.
import PassDialog from './PassDialog.jsx'
import { BANK_PASSES } from './bankPasses.js'
import {
  semanticEngineLabel, semanticEnginePatchBody, semanticEngineState, semanticPrerequisite,
} from './bankSemanticEngine.js'
// Ordered zone model + the "what's next" accent, both pure/testable.
import { BANK_ZONES, nextBankStep } from './bankGuide.js'
// Provenance wording (effective resolution, origin, black bars) — pure/testable.
import { ORIGIN_CHIPS, PROVENANCE_FLAG_LABEL, detailSummary } from './bankProvenance.js'
// Grid ordering menu (which sorts exist, and which ones have data) — pure/testable.
import { bankSortGroups, loadBankSort, saveBankSort } from '../../utils/gridSort.js'
// WHO wrote the caption on a tile — the per-image half of the same provenance the
// pass reports in aggregate (utils/captionEngines.js).
import { captionOriginTooltipLine } from '../../utils/captionOrigin.js'
// 🏷️ One image's caption → the chips you can filter by, and the same chips over a
// whole SELECTION with how often each was cited (pure/testable).
import {
  captionChips, tagsParam, tagFilterSummary,
  selectionTagCounts, selectionTagsNotes, tagCountLabel,
} from './bankTags.js'
// 🔤 Text search wording — "closest", never "matching" — plus the cold-start and
// CLIP-limitation copy. Pure/testable (node --test cannot parse this JSX).
import {
  PUSH_DOWN_DEFAULT_STRENGTH, PUSH_DOWN_STRENGTHS, pushDownCaveat, pushDownNote,
  limitsSentence, pendingLabel, readinessHint, suggestPushDown, summarize,
  withoutNegation,
} from './bankTextSearch.js'
// ⚖️ Balanced pick — the distribution obtained, in words and numbers. Pure logic
// on purpose: the repartition is what has to be provable (node --test, no JSX).
import {
  BALANCE_AXES, BALANCE_DEFAULT_AXIS, balanceNotes, balanceReadiness,
  balanceRows, summarizeBalance,
} from './bankBalance.js'
// 🎨 Medium + ⤢ Angle — buckets, tooltips and, above all, the LIMITS each row
// has to print. Pure/testable: what matters is that we never claim more than we
// measured (see bankMedium.js for the numbers behind the wording).
import {
  ANGLE_BUCKETS, MEDIUM_BUCKETS, angleBadge, angleReadiness, angleTitle,
  mediumLimits, mediumTitle, shownBuckets,
} from './bankMedium.js'
// 🧹 Auto-reject — the number next to a checkbox must be the number the click
// moves, and a 0 has to say WHICH of its two meanings it carries.
import {
  flagCandidateLabel, flagPrereq, pickedCandidates, unscannedNotice,
} from './autoRejectReadiness.js'
// 🗃️ Chip counters — the number a chip PRINTS (measured under the filters in
// force) is not the number that decides the chip EXISTS (bank-wide).
import DescribeFilterBar from './DescribeFilterBar.jsx'
import { chipCounts, facetDataKey, isFacetFiltered } from './bankFacetCounts.js'

function semanticPayloadMatches(payload, engine, modelKey = null) {
  return payload?.engine === engine
    && (!modelKey || payload?.model_key === modelKey)
}

const PAGE_SIZE = 120
/* How often the bank-wide counts refresh while a pass runs. The banner ticks
   every 2 s off /activity; the dashboard follows on this slower beat — plus
   immediately when the job lands, so the numbers you end on are exact. The
   payload is ~60 full-table aggregates: on a 50 000-image bank, asking for it
   every 2 s is what made the workspace unreadable and its Stop button
   unreachable in the first place.
   ⏱ A WALL-CLOCK deadline, deliberately, not "every Nth tick". The poll effect
   re-subscribes whenever the job's `detail` string changes — which is every
   couple of seconds during a real pass — so a counter living in the effect would
   reset before it ever reached N and the dashboard would never refresh at all. */
const FULL_PAYLOAD_MS = 10000
/* How many off-page captions the 🏷️ row will fetch for a selection.
   Not a taste call: `ids=` travels in the query string, and a few thousand
   integers build a request line the server refuses outright (the same limit the
   "show selected" view already lives with). 500 ids ≈ 3.5 kB, comfortably under
   it. Whatever the cap leaves out is DISCLOSED in the row — a count over 500 of
   3 200 images presenting itself as "your selection" is the very defect this
   surface exists to end. */
const TAG_CAPTION_FETCH_CAP = 500

const FLAG_LABEL = {
  blur: '🌫 Blurry', noise: '📺 Noisy', uniform: '⬜ Flat',
  small: '📐 Small', unreadable: '❌ Unreadable',
  // Provenance pass — same CPU scan, no extras needed.
  ...PROVENANCE_FLAG_LABEL,
  // V2 scoring flags (aesthetic · NSFW · watermark passes).
  low_aesthetic: '💔 Low aesthetic', nsfw: '🔞 NSFW', watermark: '🚩 Watermark',
}
const FLAG_HINT = {
  soft_detail: 'The picture stops before the pixels do — usually an enlargement. '
    + 'A soft or out-of-focus shot reads the same, so check before mass-rejecting.',
  bars: 'Flat black letterbox/pillarbox bars — video screenshots and padded stills. '
    + 'A dark-themed screenshot reads the same, so check before mass-rejecting.',
}
// These two are measurements of PROVENANCE, not quality verdicts, which is why
// the overnight pipeline does not offer them (backend PIPELINE_REJECT_FLAGS) and
// why the standalone button prints their caveat instead of hiding it in a
// tooltip. Here you can see the count, undo, and look at the pile first.
// Quality flags the CPU scan produces vs the ones the ML scoring/watermark
// passes add — auto-reject only offers a flag whose pass has actually run.
const QUALITY_REJECT_FLAGS = ['blur', 'noise', 'uniform', 'small', 'soft_detail', 'bars']
const SCORE_REJECT_FLAGS = ['low_aesthetic', 'nsfw', 'watermark']
// Resolution tiers — ids + labels MUST mirror backend _RES_BUCKETS (order and
// megapixel bands). Rendered as a dedicated chip row so a mixed dump can be
// sliced by resolution and mass-acted one tier at a time.
const RES_BUCKETS = [
  { id: 'res_lt_025', label: '< 0.25 MP' },
  { id: 'res_025_1', label: '0.25–1 MP' },
  { id: 'res_1_2', label: '1–2 MP' },
  { id: 'res_2_4', label: '2–4 MP' },
  { id: 'res_gt_4', label: '≥ 4 MP' },
]
// Framing buckets — ids MUST mirror backend _FRAMING_KEYS. Face/bust/body/back
// are the character composition axes; 'unknown' is a parseable-but-unclassed shot.
// Origin states — ids MUST mirror backend ORIGINS. THREE, never two: 'unknown'
// is the honest answer for any file whose metadata was stripped (which is most
// of them) and it has to be visible and selectable as its own pile.
const ORIGIN_BUCKETS = ORIGIN_CHIPS
const FRAMING_BUCKETS = [
  { id: 'face', label: '😀 Face' },
  { id: 'bust', label: '👤 Bust' },
  { id: 'body', label: '🧍 Body' },
  { id: 'back', label: '🔙 Back' },
  { id: 'unknown', label: '❔ Unknown' },
]
const STATUS_RING = {
  keep: 'ring-2 ring-emerald-400',
  reject: 'ring-2 ring-rose-400 opacity-60',
  pending: '',
}

/** Fetch EVERY image id matching a filter, page by page (used by the
 * cluster/flag "select all" actions — a cluster can exceed one grid page). */
async function fetchAllIds(bankId, params) {
  // ONE request for the ids of the whole filter (`ids_only=1`), in the order the
  // grid is showing. This used to walk the grid 500 rows at a time and keep only
  // `i.id` — 46 sequential round trips and 16 MB of image payloads to end up with
  // 23 000 integers, measured on a 22 940-image bank; with a measure sort active
  // each of those pages also re-ran the COUNT and the ORDER BY over the whole
  // table, which is what put seconds in front of ▶ Review.
  const qs = new URLSearchParams({ ...params, ids_only: '1' })
  const d = await apiFetch(`/api/bank/${bankId}/images?${qs}`)
  return d.ids || []
}

const STEP_SHORT = {
  scan: '🔎 Scan', auto_reject: '🧹 Auto-reject', score: '✨ Score',
  semantic_index: '🧠 Semantic index',
  semantic_dedup: '✂ Crops', watermark: '🚩 Watermarks', faces: '👥 Person',
  framing: '📐 Framing', caption: '🏷️ Caption', medium: '🎨 Medium',
  angles: '⤢ Angles',
}

/* No contact with the server, so no fresh job snapshot. Saying NOTHING here is
   the bug this replaces: a pass that is running perfectly well in the bank's
   server-side thread looked exactly like a pass that had stopped. Plain text,
   NOT a live region — the global ConnectionBanner already announces the outage
   once, and a second live region would double every announcement. */
function ProgressUnknown({ stale }) {
  return (
    <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-sm text-content-muted">
      <span aria-hidden>📡</span>{' '}
      {stale
        ? 'Lost contact — the progress above is the last thing we heard. The pass keeps running on the server.'
        : 'Lost contact — can’t read job progress right now. Anything you started keeps running on the server; this comes back when the connection does.'}
    </p>
  )
}

/** ↩ The net under the bank's biggest gesture.
 *
 * Deliberately NOT a toast: a bar that vanishes after four seconds is unusable
 * for anyone who reads slowly, and this is the one control you reach for after
 * realising you just marked 400 images wrong. It stays until it is used,
 * dismissed, or replaced by a newer action — and it is fed by the bank payload,
 * so it is still there after a reload, unlike an undo that only ever lived in
 * this tab.
 *
 * `role="status"` + `aria-live="polite"` so a screen reader is told the net
 * exists at the moment the bulk action lands, without stealing focus from the
 * grid. Both controls are real buttons, in tab order.
 */
function UndoBar({ offer, busy, onUndo, onDismiss }) {
  if (!offer) return null
  return (
    <div role="status" aria-live="polite"
      className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-lg border border-sky-400/40 bg-sky-500/10 px-3 py-2 text-sm text-content">
      <p className="m-0 min-w-0 grow basis-full sm:basis-auto">
        <span aria-hidden>↩</span>{' '}
        <span className="font-semibold">{undoBannerText(offer)}</span>
        <span className="block text-xs text-content-muted sm:inline sm:before:content-['_—_']">
          {UNDO_HINT}
        </span>
      </p>
      <button type="button" onClick={onUndo} disabled={busy}
        className="rounded border border-sky-400/60 px-2 py-1 text-xs font-semibold hover:bg-white/10 disabled:opacity-50">
        {busy ? 'Undoing…' : '↩ Undo'}
      </button>
      <button type="button" onClick={onDismiss} disabled={busy}
        title="Keep the change and hide this"
        className="rounded border border-border px-2 py-1 text-xs text-content-muted hover:bg-surface-raised hover:text-content disabled:opacity-50">
        Dismiss
      </button>
    </div>
  )
}

// Exported for the render test only: what this bar says during a step with no
// per-image counter is the whole fix for "the pass looks frozen", and a source
// regex cannot see what the renderer produces.
export function ProgressBar({ activity, onCancel, offline = false }) {
  /* Held HERE and not in the parent on purpose: the click has to be answered
     without waiting for the payload that takes 2.7 s to rebuild on a bank this
     size. Keyed by job, so the next pass gets a live button back with no effect
     and no cleanup — see passStop.js for the measurements behind this. */
  const [stopAskedFor, setStopAskedFor] = useState(null)
  const presence = progressPresence(activity, offline)
  if (presence === PROGRESS_HIDDEN) return null
  if (presence === PROGRESS_UNKNOWN) return <ProgressUnknown />
  const stale = presence === PROGRESS_STALE
  const { kind, done, total, detail } = activity
  const pct = total > 0 ? Math.round((100 * done) / total) : null
  // "12939 / 37800" says where the pass is; it never said how long that leaves.
  // On a bank this size the difference between twenty minutes and four hours is
  // the difference between waiting and going to do something else.
  const eta = etaPhrase(activity)
  const asked = stopRequested(activity, stopAskedFor)
  const note = stopNote(activity, asked)
  const pipe = kind === 'pipeline' ? activity.pipeline : null
  return (
    <div className="space-y-2 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-sm">
      {/* flex-wrap: at 400 px the label, the bar and Stop cannot share one row —
          they used to squash the label to a sliver. */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <span aria-hidden>⏳</span>
        <span className={`text-content ${stale ? 'opacity-60' : ''}`}>
          {pipe
            ? `🚀 Launch all — step ${(pipe.index ?? 0) + 1}/${pipe.total_steps} · ${STEP_SHORT[pipe.current] || pipe.current}`
            : ({ scan: 'Quality scan', faces: 'Face pass', score: 'Scoring pass',
              semantic_index: 'Semantic index',
              semantic_dedup: 'Crops & variants', watermark: 'Watermark scan',
              framing: 'Framing pass', caption: 'Captioning', promote: 'Promotion',
              medium: 'Medium pass', angles: 'Measuring head angles',
              bank_promote: 'Copying into the new bank',
              // The one destructive pass: it must NAME itself in the bar, not
              // ride under the anonymous "Job running" fallback.
              delete_rejected: 'Deleting rejected files' }[kind] || 'Job') + ' running'}
          {/* A step with no per-image counter publishes done=total=0 and says
              what it is doing in words (the cache write, the style grouping).
              Printing a bare "0" next to that sentence would read as "0 done"
              on work that is running — so the figure only appears when there
              is one. */}
          {(done || total) ? <>{' — '}{done}{total ? ` / ${total}` : ''}</> : null}
          {eta ? `${(done || total) ? ' · ' : ' — '}${eta}` : ''}
          {detail ? `${(done || total || eta) ? ' · ' : ' — '}${detail}` : ''}
        </span>
        {pct != null && (
          <div className="h-1.5 w-40 overflow-hidden rounded bg-surface-raised" role="progressbar"
            aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-full bg-amber-400" style={{ width: `${pct}%` }} />
          </div>
        )}
        {/* `disabled` is the whole of requirement one: seven POSTs in 20 ms is
            what an enabled button that looks identical after the click buys.
            `title` carries the promise for the pointer that hovers before it
            presses; the line below carries it for everyone else. */}
        <button type="button" disabled={asked} title={note || undefined}
          onClick={() => { setStopAskedFor(jobKey(activity)); onCancel?.() }}
          className="ml-auto rounded-md border border-border px-2 py-0.5 text-xs text-content hover:bg-surface-raised disabled:cursor-default disabled:opacity-60 disabled:hover:bg-transparent">
          {stopLabel(asked)}
        </button>
      </div>
      {/* One line, xs, muted — and it REPLACES the Stop sentence that used to
          ride inside the style-grouping phase label, where everyone read it all
          the time and it wrapped the counter off its row at 400 px.
          aria-live so the answer to the click reaches a screen reader too. */}
      {note && (
        <p className="m-0 pl-6 text-xs text-content-muted" aria-live="polite">{note}</p>
      )}
      {stale && <ProgressUnknown stale />}
      {pipe && Array.isArray(pipe.results) && pipe.results.length > 0 && (
        <ul className="flex flex-wrap gap-1.5 pl-6 text-xs">
          {pipe.results.map((r, i) => (
            <li key={`${r.step}-${i}`}
              className={`rounded px-1.5 py-px ${r.status === 'done' ? 'bg-emerald-500/15 text-emerald-300'
                : r.status === 'error' ? 'bg-rose-500/15 text-rose-300'
                : 'bg-black/20 text-content-subtle'}`}
              title={r.reason || r.detail || ''}>
              {r.status === 'done' ? '✅' : r.status === 'error' ? '⚠️' : '⏭️'} {STEP_SHORT[r.step] || r.step}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function Chip({ active, onClick, children, title }) {
  return (
    <button type="button" onClick={onClick} title={title}
      className={`rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors ${active
        ? 'border-indigo-400/60 bg-indigo-500/20 text-indigo-200'
        : 'border-border bg-surface text-content-muted hover:text-content hover:bg-surface-raised'}`}>
      {children}
    </button>
  )
}

// One header stat — a bold tabular figure with a subtle label. Kept/rejected/
// promoted carry their status colour so the strip reads at a glance.
function Stat({ label, value, tone }) {
  const toneCls = { emerald: 'text-emerald-300', rose: 'text-rose-300', indigo: 'text-indigo-300' }[tone] || 'text-content'
  const n = typeof value === 'number' ? value.toLocaleString() : value
  return (
    <span className="inline-flex items-baseline gap-1">
      <span className={`font-semibold tabular-nums ${toneCls}`}>{n}</span>
      <span className="text-xs text-content-subtle">{label}</span>
    </span>
  )
}

// A small uppercase eyebrow that names a group of controls (a filter facet, the
// passes toolbar) so dense rows read as grouped rather than as a flat wall.
function GroupLabel({ children }) {
  return (
    <span className="text-[10px] font-semibold uppercase tracking-wide text-content-subtle">{children}</span>
  )
}

// A labelled cluster of filter chips — the eyebrow names the facet (Status,
// Quality, Score, Groups); chips wrap together within it on narrow viewports.
function FilterGroup({ label, children }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <GroupLabel>{label}</GroupLabel>
      {children}
    </div>
  )
}

// The selection's caption tags stay visible while the gallery changes. The
// component is mounted once per responsive position: in the filter zone below
// xl, and as a sticky inspector beside the gallery on desktop.
function SelectionTagsPanel({ tagRow, tagPicked, onToggle, onClear }) {
  return (
    <div className="space-y-1.5 rounded-lg border border-emerald-400/30 bg-emerald-500/5 px-2.5 py-2">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <GroupLabel>
          {tagRow.kind === 'image' ? `🏷️ Tags of ${tagRow.name}`
            : tagRow.size === 1 ? '🏷️ Tags of the selected image'
              : `🏷️ Tags across ${tagRow.size} selected images`}
        </GroupLabel>
        <span className="text-[11px] text-content-subtle">
          attributes you pick — unlike 🎯 Similar, which matches the look
        </span>
        {tagRow.frozen && (
          <span className="rounded border border-border px-1.5 text-[11px] text-content-subtle">
            held from the selection you filtered on
          </span>
        )}
        <button type="button" onClick={onClear}
          className="ml-auto rounded border border-border px-2 py-0.5 text-[11px] text-content-muted hover:text-content">
          ✕ Close
        </button>
      </div>
      {tagRow.rows.length === 0 ? (
        <p className="m-0 text-xs text-content-muted">
          {tagRow.uncaptioned > 0 && tagRow.counted === 0
            ? (tagRow.size === 1
              ? 'This image has no caption yet — run 🏷️ Caption and its tags appear here.'
              : `None of these ${tagRow.size} images has a caption yet — run 🏷️ Caption on them and their tags appear here.`)
            : 'No caption here has a word worth filtering on.'}
        </p>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {tagRow.rows.map(({ tag, count }) => {
            const n = tagCountLabel(count, tagRow.counted)
            return (
              <Chip key={tag} active={tagPicked.has(tag)}
                onClick={() => onToggle(tag)}
                title={tagPicked.has(tag)
                  ? `Stop requiring “${tag}”`
                  : `Show only images whose caption mentions “${tag}”`
                    + (n ? ` — cited by ${count} of the ${tagRow.counted} captioned images you selected` : '')}>
                {tag}
                {n && <span className="ml-1 text-[10px] text-content-subtle">{n}</span>}
              </Chip>
            )
          })}
        </div>
      )}
      {selectionTagsNotes(tagRow, tagRow.unread).map((note) => (
        <p key={note} className="m-0 text-[11px] leading-snug text-content-subtle">{note}</p>
      ))}
      {tagPicked.size > 0 && (
        <p className="m-0 text-[11px] text-content-muted">
          {tagFilterSummary(tagPicked)} Matched as whole words, in captions and file names.
        </p>
      )}
    </div>
  )
}

// The individual analysis passes share one quiet, uniform button so they read
// as a secondary group next to the prominent Launch all / Promote actions.
function PassButton({ onClick, disabled, title, children }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} title={title}
      className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content transition-colors hover:bg-surface disabled:opacity-50 disabled:hover:bg-surface-raised">
      {children}
    </button>
  )
}

// One numbered workflow zone: a labelled, collapsible section grouping the
// controls of one step (① Analyze … ④ Promote). `accented` draws a discreet
// amber ring + "Next step" pill on the ONE zone nextBankStep recommends — purely
// advisory: every zone stays open and clickable. Default expanded (nothing hidden).
function ZoneSection({ zone, accented, children }) {
  const [open, setOpen] = useState(true)
  return (
    <section className={`rounded-xl border bg-surface ${accented
      ? 'border-amber-400/60 ring-1 ring-amber-400/40' : 'border-border'}`}>
      <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
        className="flex w-full items-center gap-2 px-4 py-2 text-left">
        <span aria-hidden className="text-base tabular-nums">{zone.emoji}</span>
        <span className="text-sm font-semibold text-content">{zone.label}</span>
        {accented && (
          <span className="rounded-full border border-amber-400/50 bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-300">
            Next step
          </span>
        )}
        <span aria-hidden className="ml-auto text-xs text-content-subtle">{open ? '▲' : '▼'}</span>
      </button>
      {open && <div className="space-y-3 px-4 pb-3">{children}</div>}
    </section>
  )
}

// 📊 Coverage advice (idea by @antonp) — a read-only, collapsible panel. Reads
// what the passes already computed (framing, person/style clusters, resolution)
// and says, in plain sentences, what the kept set leans on and what's thin for a
// good LoRA. Never selects or rejects; warnings first, then gentler notes.
function FramingBar({ framing }) {
  const total = FRAMING_BUCKETS.reduce((a, b) => a + (framing[b.id] || 0), 0)
  if (!total) return null
  const tone = { face: 'bg-teal-400', bust: 'bg-sky-400', body: 'bg-indigo-400',
    back: 'bg-fuchsia-400', unknown: 'bg-content-subtle' }
  return (
    <div className="space-y-1">
      <div className="flex h-2 overflow-hidden rounded bg-surface-raised" role="img"
        aria-label="Framing distribution of the set">
        {FRAMING_BUCKETS.map((b) => (framing[b.id] || 0) > 0 && (
          <div key={b.id} className={tone[b.id]} style={{ width: `${(100 * framing[b.id]) / total}%` }}
            title={`${b.label}: ${framing[b.id]}`} />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-content-subtle">
        {FRAMING_BUCKETS.map((b) => (framing[b.id] || 0) > 0 && (
          <span key={b.id}>{b.label} {framing[b.id]}</span>
        ))}
      </div>
    </div>
  )
}

// 👁 What the labels cannot see: how alike the pool actually LOOKS, measured on
// the currently selected semantic cache. A Bank with no usable index shows "Not
// measured" rather than a reassuring colour — the whole point is that silence
// must never read as variety.
function VisualSpread({ visual, total, semanticEngine }) {
  const r = spreadReadout(visual, semanticEngine)
  if (!r) return null
  const tone = r.tone === 'warn' ? 'border-amber-400/50 bg-amber-400/10 text-amber-200'
    : r.tone === 'ok' ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
      : 'border-border bg-surface-raised text-content-subtle'
  const note = spreadCoverageNote(visual, total, semanticEngine)
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] uppercase tracking-wide text-content-muted">Visual spread</span>
        <span className={`rounded-full border px-2 py-0.5 text-[11px] ${tone}`}>{r.label}</span>
      </div>
      <p className="m-0 text-[11px] text-content-subtle">{r.detail}</p>
      {note && <p className="m-0 text-[11px] text-content-subtle">{note}</p>}
    </div>
  )
}

// The caption-derived axes, rendered from the SAME pure helpers the dataset panel
// uses (imported, not copied) so the two surfaces cannot drift apart.
function VarietyAxes({ variety }) {
  if (!variety || !variety.captioned || !(variety.axes || []).length) return null
  const chip = { ok: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
    thin: 'border-amber-400/50 bg-amber-400/10 text-amber-300',
    gap: 'border-rose-400/50 bg-rose-400/10 text-rose-300',
    none: 'border-border bg-surface-raised text-content-subtle' }
  return (
    <div className="flex flex-col gap-2 border-t border-border pt-2">
      {variety.axes.map((axis) => (
        <div key={axis.id} className="flex flex-col gap-1">
          <div className="flex flex-wrap items-baseline gap-x-2">
            <span className="text-[11px] uppercase tracking-wide text-content-muted">{axis.label}</span>
            {axis.hint && <span className="text-[11px] text-content-subtle">{axis.hint}</span>}
          </div>
          {/* The chips decorate a sentence a screen reader can read out; the
              sentence is the carrier, never the colour. */}
          <span className="sr-only">{axisSummary(axis)}</span>
          <div aria-hidden className="flex flex-wrap gap-1">
            {axisRows(axis).map((r) => (
              <span key={r.id}
                title={r.count ? `${r.count} caption${r.count === 1 ? '' : 's'} mention this`
                  : (r.state === 'gap' ? 'No caption mentions this' : 'Not mentioned (optional)')}
                className={`rounded-full border px-2 py-0.5 text-[11px] ${chip[r.state]}`}>
                {r.label}<span className="opacity-60"> {r.count}</span>
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

function CoveragePanel({ coverage, semanticEngine, semanticLabel,
  onClose, onBalance = null, balanceReason = '' }) {
  if (!coverage) {
    return <p className="text-sm text-content-subtle">Reading coverage…</p>
  }
  const poolWord = coverage.pool === 'kept' ? 'kept' : 'candidate (nothing kept yet)'
  return (
    <div className="space-y-3 rounded-lg border border-indigo-400/40 bg-indigo-500/5 px-3 py-2.5">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-content">📊 Coverage advice</span>
        <span className="text-xs text-content-subtle">
          {coverage.total.toLocaleString()} {poolWord} image{coverage.total === 1 ? '' : 's'}
        </span>
        <span className="ml-auto rounded border border-indigo-400/40 px-1.5 py-px text-[10px] uppercase tracking-wide text-indigo-300"
          title="Community idea by @antonp">idea by @antonp</span>
        <button type="button" onClick={onClose} aria-label="Hide coverage advice"
          className="rounded-md border border-border px-1.5 py-0.5 text-xs text-content-subtle hover:text-content">✕</button>
      </div>
      {coverage.framing_available && <FramingBar framing={coverage.framing} />}
      <VisualSpread visual={coverage.visual} total={coverage.total}
        semanticEngine={semanticEngine} />
      <ul className="space-y-1 text-sm">
        {coverage.advice.map((a, i) => (
          <li key={i} className="flex items-start gap-2">
            <span aria-hidden>{a.tone === 'warn' ? '⚠️' : '💡'}</span>
            <span className={a.tone === 'warn' ? 'text-amber-200' : 'text-content-muted'}>{a.text}</span>
          </li>
        ))}
      </ul>
      {/* The advice said what leans; this is the gesture that acts on it. Still
          only a SELECTION — the panel itself never keeps or rejects anything. */}
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={onBalance || undefined} disabled={!onBalance}
          title={onBalance
            ? 'Select a set spread evenly over the framings, instead of the top of one ranking'
            : balanceReason}
          className="rounded-md border border-emerald-400/40 bg-emerald-500/10 px-2 py-0.5 text-xs text-content disabled:opacity-50 hover:bg-emerald-500/20">
          ⚖️ Pick a balanced set…
        </button>
        {!onBalance && balanceReason && (
          <span className="text-[11px] text-content-subtle">{balanceReason}</span>
        )}
      </div>
      <VarietyAxes variety={coverage.variety} />
      <p className="text-[11px] text-content-subtle">
        Advice only — nothing is kept or rejected. Based on what the passes already computed:
        the labels, your captions (words, not pixels — a shot the captioner never described is
        invisible here, and “not smiling” still counts as a smile) and the {semanticLabel} semantic index.
        Judged as a character source, like the framing target above.
      </p>
    </div>
  )
}

function Tile({ img, bankId, selected, onToggle, onReview, onTags, size }) {
  // `key` matters only for the flags list below (the one mapped array) — it was
  // missing and logged a React warning on every bank grid render.
  // The chips this image would actually offer. Computed HERE rather than asked of
  // the caption's mere existence: a caption of nothing but stop words ("a photo of
  // her") yields zero chips, and `img.caption && …` would have offered the button
  // anyway. The test for "can this button do its job" is the job's own output.
  const tagChips = captionChips(img.caption)
  const badge = (txt, cls, key) => (
    <span key={key} className={`rounded px-1 py-px text-[10px] font-semibold leading-none ${cls}`}>{txt}</span>
  )
  return (
    <li className={`relative overflow-hidden rounded-lg border border-border bg-surface ${STATUS_RING[img.status] || ''}`}>
      <button type="button" onClick={onToggle}
        title={`${img.name} — ${img.width || '?'}×${img.height || '?'}`
          + (img.blur_score != null ? ` · sharpness ${Math.round(img.blur_score)}` : '')
          + (img.aesthetic_score != null ? ` · aesthetic ${img.aesthetic_score.toFixed(1)}` : '')
          + (img.nsfw_score != null ? ` · NSFW ${Math.round(img.nsfw_score * 100)}%` : '')
          + (img.face_cluster ? ` · person #${img.face_cluster}`
            // A declaration is not a measurement — the tooltip says which it is.
            + (img.face_cluster_origin === 'asserted' ? ' (your folder assertion)' : '') : '')
          + (img.framing ? ` · ${img.framing}` : '')
          + (img.medium ? ` · ${img.medium}` : '')
          + (img.face_yaw != null ? ` · head turned ${Math.round(Math.abs(img.face_yaw))}°` : '')
          + (detailSummary(img)?.soft ? ` · only ~${detailSummary(img).real} px of real detail` : '')
          + (img.origin && img.origin !== 'unknown' ? ` · ${img.origin}` : '')
          + (img.style_cluster ? ` · style #${img.style_cluster}` : '')
          + (img.semantic_dup_group ? ` · same shot #${img.semantic_dup_group}` : '')
          // The caption, and WHO WROTE IT in the same breath. The 'auto' backend
          // chains two engines inside one run, so a bank holds both and the
          // sentence alone cannot say which half produced it — exactly the reading
          // this tooltip was missing.
          + (img.caption
            ? `\n${captionOriginTooltipLine(img.caption, img.caption_origin)}: ${img.caption}`
            : '')}
        className="block w-full">
        {/* ?r= is a cache buster, not a parameter the server reads: the thumb
            route answers with max-age=3600, so a turned image would keep showing
            its old orientation for an hour and read as "the button did nothing". */}
        <img src={`/api/bank/${bankId}/thumb/${img.id}${img.rotation ? `?r=${img.rotation}` : ''}`}
          alt={img.rotation ? `${img.name} (rotated ${img.rotation}°)` : img.name} loading="lazy"
          className={`w-full object-cover ${size === 'S' ? 'h-24' : 'h-36'}`} />
      </button>
      {selected && (
        <span aria-hidden className="absolute inset-0 bg-indigo-500/30 ring-2 ring-indigo-400 rounded-lg pointer-events-none" />
      )}
      <span className="absolute left-1 top-1 flex flex-wrap gap-0.5 max-w-[85%]">
        {img.status === 'keep' && badge('✓', 'bg-emerald-500/80 text-white')}
        {img.status === 'reject' && badge(`✕ ${img.reject_reason || ''}`.trim(), 'bg-rose-500/80 text-white')}
        {/* ⬆ = this image left for somewhere: a dataset, another bank, or both.
            One badge for both destinations — the tile says THAT it went, the
            tooltip and the review lightbox say where. */}
        {(img.promoted_dataset_id != null || img.promoted_bank_id != null)
          && badge('⬆', 'bg-indigo-500/80 text-white')}
        {img.flags.map((f) => badge(FLAG_LABEL[f]?.slice(0, 2) || f, 'bg-black/60 text-amber-200', f))}
        {img.face_cluster != null && badge(`👤${img.face_cluster}`, 'bg-black/60 text-sky-200')}
        {img.framing && badge(`📐${img.framing}`, 'bg-black/60 text-teal-200')}
        {/* A medium badge is stamped only when the classifier actually COMMITTED
            to one — 'unsure' is a real verdict but it is not a label to write on
            a thumbnail, and NULL means the pass never reached this image. */}
        {img.medium && img.medium !== 'unsure'
          && badge(`🎨${img.medium}`, 'bg-black/60 text-lime-200')}
        {angleBadge(img) && badge(angleBadge(img).text, 'bg-black/60 text-cyan-200')}
        {/* Only the PROVEN states get a badge. Stamping ❔ on the 80% of files
            whose metadata was stripped would be noise, not information. */}
        {img.origin === 'ai' && badge('🤖', 'bg-black/60 text-violet-200')}
        {img.origin === 'camera' && badge('📷', 'bg-black/60 text-emerald-200')}
        {img.style_cluster != null && badge(`🎨${img.style_cluster}`, 'bg-black/60 text-fuchsia-200')}
        {img.dup_group != null && badge(`≈${img.dup_group}`, 'bg-black/60 text-fuchsia-200')}
        {img.semantic_dup_group != null && badge(`✂${img.semantic_dup_group}`, 'bg-black/60 text-orange-200')}
      </span>
      {/* ▶ starts the fast-triage lightbox AT this image. It's a separate hit
          target on purpose: the tile's own click still (de)selects for the bulk
          ✓/✕/⬆ bar, so neither use loses its gesture. */}
      {/* 🏷️ MOVED HERE, out of the badge cluster in the top-left corner.
          Everything up there is a STATE READOUT — ✓, ✕, ⬆, the flags, the person
          and framing chips — and none of it is clickable. A button dropped in the
          middle of that row does not read as an action; in the maintainer's own
          words, "otherwise it gets drowned at the top with the icons that aren't
          clickable". The tile's two real actions live in this bottom-right group,
          so the third one joins them, in the same clothes.

          AND IT ONLY APPEARS WHEN IT CAN DELIVER. The chips ARE the words of the
          caption, so on an image whose caption yields none the picker could only
          say "This caption has no word worth filtering on." — a button promising
          something it cannot do. That silence has already cost once, the other way
          round: the feature had shipped for two days and was read as absent,
          because the bank simply had no captions. So the button says WHY it is
          not there, exactly the way a semantic action names its missing index
          on the pass row — a shipped feature that says nothing is indistinguishable
          from one that does not exist. */}
      {tagChips.length > 0 ? (
        <button type="button" onClick={onTags}
          title={`Filter the bank by this image's tags — ${img.caption}`}
          aria-label={`Use the tags of ${img.name} as a filter`}
          className="absolute bottom-1 right-11 rounded bg-black/60 px-1 text-[11px] text-emerald-200 hover:bg-black/80">🏷️</button>
      ) : (
        <span
          title={img.caption
            ? '🏷️ Tags — this caption has no word worth filtering on (the chips are the caption\'s own words)'
            : '🏷️ Tags (needs a caption) — run 🏷️ Caption on this bank and the chips appear here'}
          aria-label={img.caption
            ? 'Tags unavailable: this caption has no word worth filtering on'
            : 'Tags unavailable: this image has no caption yet'}
          className="absolute bottom-1 right-11 rounded bg-black/40 px-1 text-[11px] text-white/35">🏷️</span>
      )}
      <button type="button" onClick={onReview}
        title="Review from this image — full size, one at a time, with Keep/Reject/Skip"
        aria-label={`Review from ${img.name}`}
        className="absolute bottom-1 right-6 rounded bg-black/60 px-1 text-[11px] text-white hover:bg-black/80">▶</button>
      <a href={`/api/bank/${bankId}/file/${img.id}`} target="_blank" rel="noreferrer"
        title="Open the original file" aria-label={`Open ${img.name} full size`}
        className="absolute bottom-1 right-1 rounded bg-black/60 px-1 text-[11px] text-white no-underline hover:bg-black/80">⛶</a>
    </li>
  )
}

export default function BankWorkspace({ bankId, onBack, onGone }) {
  const toast = useToast()
  const { caps, loading: capsLoading, refresh: refreshCaps } = useCapabilities()
  // Updated on every render once the payload has been normalised. The unmount
  // cleanup must release the engine selected NOW, not the one from first mount.
  const semanticEngineRef = useRef('clip')
  const semanticModelKeyRef = useRef(null)
  // Every coverage response is tied to the request that produced it. Switching
  // engines invalidates an older request even if that request finishes last.
  const coverageRequestRef = useRef(0)
  const textStatusRequestRef = useRef(0)
  const [payload, setPayload] = useState(null)
  // The chip counters measured under the ACTIVE filter (null = nothing filtered,
  // so the payload's bank-wide numbers are the honest answer). See
  // bankFacetCounts.js for why these are two maps and not one.
  const [facets, setFacets] = useState(null)
  // `sort` opens on whatever order this bank was last reviewed in (per bank, not
  // global — see gridSort.bankSortStorageKey). Every other facet starts empty on
  // purpose: an order is a habit, a filter is a question you asked once.
  const [filter, setFilter] = useState(() => ({ status: null, flag: null, cluster: null,
    style: null, subfolder: null, search: null, exclude: null, tags: null,
    sort: loadBankSort(bankId), resBucket: null,
    origin: null,
    framing: null,
    // Dedicated keys, never folded into `flag` or the text lane.
    medium: null,
    angle: null }))
  const [searchText, setSearchText] = useState('')
  // 🚫 The inverse of the search box: hide what already carries a word. Session
  // state, deliberately NOT remembered like the sort — an order you can see in a
  // menu is a habit, images missing from a grid for a reason you set last week
  // reads as data loss.
  const [excludeText, setExcludeText] = useState('')
  // 🏷️ Tag chips lifted off ONE image's caption. `tagSource` is the image the
  // chips were read from (kept so the row can say WHOSE tags these are — chips
  // with no provenance are just mystery words), `tagPicked` the ticked subset.
  // Both are session state: this is a question you ask about one image now, not
  // a standing preference like the sort order.
  const [tagSource, setTagSource] = useState(null)
  const [tagPicked, setTagPicked] = useState(() => new Set())
  // 🏷️ …and the SELECTION's tags, which need no click at all. `tagFreeze` is the
  // snapshot taken the instant a chip is ticked: applying a filter clears the
  // selection (setF does, and must — the ids no longer match what is on screen),
  // so without this the row would compute the answer and then delete the question.
  const [tagFreeze, setTagFreeze] = useState(null)
  const [subfolders, setSubfolders] = useState([])
  // 👤 Folder-level person assertions ("this subfolder is one person").
  const [folderPersons, setFolderPersons] = useState([])
  // The whole folder-person payload: assertions PLUS the suggestions the app
  // probed by itself, and what a scan would cost.
  const [folderPersonInfo, setFolderPersonInfo] = useState(null)
  const [folderPersonBusy, setFolderPersonBusy] = useState(false)
  // 👤 The preflight of the person pass: { plan, probing, run } — `run` is the
  // pass (or the whole 🚀 Launch all) the user actually asked for, held until
  // they have answered the folder question.
  const [preflight, setPreflight] = useState(null)
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState({ images: [], total: 0 })
  const [selected, setSelected] = useState(() => new Set())
  const [promoteOpen, setPromoteOpen] = useState(false)
  const [deleteRejectedOpen, setDeleteRejectedOpen] = useState(false)
  // ↩ one step back over the last bulk decision. `undoDismissedAt` remembers the
  // offer the user waved away by its timestamp, so the NEXT bulk action shows a
  // bar again instead of inheriting the dismissal.
  const [undoBusy, setUndoBusy] = useState(false)
  const [undoDismissedAt, setUndoDismissedAt] = useState(0)
  const [launchOpen, setLaunchOpen] = useState(false)
  const [semanticSwitching, setSemanticSwitching] = useState(false)
  /* 🎛 Which pass's launch window is open (a BANK_PASSES id, or null).
     ONE piece of state for every launch window: a pass button no longer fires, it opens
     the window that shows where the run applies, what the calculation reads and
     what is NOT decided there — then launches from the bottom of it. */
  const [passOpen, setPassOpen] = useState(null)
  // ✨ Score's interpreter picker — reuse a CUDA Python this machine already has
  // instead of downloading another torch. Opened from the CPU warning.
  // Which interpreter picker is open, if any: a PICKER_PROFILES key ('scoring'
  // for ✨ Score, 'semantic' for the SigLIP 2 index) or '' for none. One dialog
  // serves both — the profile decides the endpoint and the wording.
  const [pythonPickerFor, setPythonPickerFor] = useState('')
  const [dismissedReportAt, setDismissedReportAt] = useState(null)
  const [relocating, setRelocating] = useState(false)
  const [openingSourceFolder, setOpeningSourceFolder] = useState(false)
  const [rejectFlags, setRejectFlags] = useState(() => new Set(['blur', 'uniform']))
  const [showAutoReject, setShowAutoReject] = useState(false)
  // 🎚 The threshold editor folds away: the chips are the daily gesture, the
  // numbers behind them are the occasional one.
  const [thresholdsOpen, setThresholdsOpen] = useState(false)
  // Curation popovers ('diverse' | 'similar' | null) and their target counts.
  const [curateOpen, setCurateOpen] = useState(null)
  const [diverseN, setDiverseN] = useState(60)
  // Typicality guard for 🎨 Pick diverse. Pure farthest-point sampling maximises
  // the distance to what is already picked — mathematically the criterion that
  // prefers ISOLATED images, so the first picks used to be the memes and the
  // stray photos of someone else. 0 = the historical behaviour, on purpose still
  // reachable; 0.5 = the default (see BANK_TYPICALITY_DEFAULT rationale in the
  // service docstring).
  const [diverseTypicality, setDiverseTypicality] = useState(0.5)
  const [diverseBusy, setDiverseBusy] = useState(false)
  // ⚖️ Balanced pick — the OTHER question ("does my set cover the framings?").
  // Axis ids are persisted keys, never renamed (see bankBalance.js).
  const [balanceN, setBalanceN] = useState(60)
  const [balanceAxis, setBalanceAxis] = useState(BALANCE_DEFAULT_AXIS)
  const [balanceBusy, setBalanceBusy] = useState(false)
  const [balanceResult, setBalanceResult] = useState(null)
  const [similarN, setSimilarN] = useState(60)
  const [similarBusy, setSimilarBusy] = useState(false)
  // 🔤 Text search. `textStatus` is the BEFORE-the-click truth (available? model
  // already warm? would it download?), `textResult` the AFTER-the-click one that
  // keeps the ranking legible once the grid has switched to it.
  const [textQuery, setTextQuery] = useState('')
  const [textN, setTextN] = useState(60)
  // 🔤 what to push DOWN the ranking. Not a filter — see bankTextSearch.js.
  const [textExclude, setTextExclude] = useState('')
  const [textExcludeW, setTextExcludeW] = useState(PUSH_DOWN_DEFAULT_STRENGTH)
  const [textStatus, setTextStatus] = useState(null)
  const [textPending, setTextPending] = useState(false)
  const [textResult, setTextResult] = useState(null)
  const semanticOperationBusy = diverseBusy || balanceBusy || similarBusy || textPending
  // "Show selected" VIEW: render ONLY the selected ids, in a chosen order.
  // showSelected flips the grid from the facet page to the selection; selectedOrder
  // holds the order to render them in — the similarity/diversity ranking after a
  // curate action (reference first, closest→farthest), else insertion order. It's a
  // VIEW, not a status: Keep/Reject/Promote still act on the selection itself.
  const [showSelected, setShowSelected] = useState(false)
  const [selectedOrder, setSelectedOrder] = useState(null)
  const [tileSize, setTileSize] = useState('M')
  // Caption register for the 🏷️ Caption pass ('' = model's own wording). Explicit is
  // the NSFW lane — same registers as the dataset caption, passed per-run.
  const [captionVocab, setCaptionVocab] = useState('')
  // Caption LENGTH preset, per RUN like the vocabulary register above (a bank has no
  // caption_options row to persist to). '' = standard: nothing appended to the prompt.
  const [captionLength, setCaptionLength] = useState('')
  // WHICH ENGINE and WHICH VISION MODEL write this run's captions. Per RUN, like every
  // other dial on this row: the global Settings stay the default and are never written
  // from here, so a user can try a different captioner on one pass without changing what
  // every dataset does afterwards. '' on either = follow the setting, and the key is then
  // left OUT of the request — a run that picks nothing is byte-identical to before.
  const [captionEngine, setCaptionEngine] = useState('')
  const [captionModel, setCaptionModel] = useState('')
  // The pulled Ollama models, for the picker. Not in `caps` (which carries only the
  // configured vision model), so it is its own always-200 fetch — an unreachable Ollama
  // is an empty list, never an error.
  const [ollamaModels, setOllamaModels] = useState([])
  /* WHICH PILE each pass runs on, and whether it re-does rows that already have a
     result — kept HERE, not inside the windows, so closing one does not silently
     undo a choice. Keyed by pass id; '' is the historical scope (kept + undecided,
     and the request omits `statuses` entirely).

     🏷️ Caption reads its entry under the name it has always used, because the
     re-caption arithmetic beside it is written against that name. */
  const [passScopes, setPassScopes] = useState({})
  const [passRedo, setPassRedo] = useState({})
  const setPassScope = (id, v) => setPassScopes((p) => ({ ...p, [id]: v }))
  const setPassRedoFor = (id, v) => setPassRedo((p) => ({ ...p, [id]: v }))
  const captionScope = passScopes.caption || ''
  /* The ESCAPE HATCH, and the reason it is a piece of state and not a request key: it has
     to be visible, deliberate and re-read in the confirmation. Never persisted, so it
     resets with the panel — an opt-out of a protection is not a preference. */
  const [captionIncludeAsserted, setCaptionIncludeAsserted] = useState(false)
  // Coverage advice (idea by @antonp) — a collapsible read-only panel, fetched
  // on demand (and refreshed whenever it's open and the bank changes).
  const [coverageOpen, setCoverageOpen] = useState(false)
  const [coverage, setCoverage] = useState(null)
  // ▶ Review — the fast-triage lightbox. `review` holds the SNAPSHOT of ids it
  // walks ({ids, startId}); null when closed. Snapshotting at open is the whole
  // point: a decision drops the image out of the current filter, so a live list
  // would reorder under the cursor and make the run skip or loop.
  const [review, setReview] = useState(null)
  const [reviewLoading, setReviewLoading] = useState(false)
  const activityWasLive = useRef(false)
  // When the dashboard was last pulled in full. A ref, not effect-local state:
  // the poll effect re-subscribes on every change of the job's `detail`.
  const fullPayloadAt = useRef(0)
  // 📡 Drives the "we lost contact" note in the progress zone: a failed poll
  // must never render as "no job running".
  const connection = useConnectionStatus()

  const loadCoverage = useCallback(async () => {
    const requestId = ++coverageRequestRef.current
    const expectedEngine = semanticEngineRef.current
    const expectedModelKey = semanticModelKeyRef.current
    try {
      const next = await apiFetch(`/api/bank/${bankId}/coverage`)
      if (requestId !== coverageRequestRef.current
          || expectedEngine !== semanticEngineRef.current
          || !semanticPayloadMatches(next, expectedEngine, expectedModelKey)) return
      setCoverage(next)
    } catch { /* transient — the panel keeps its last read */ }
  }, [bankId])

  // The grid refresher, held in a ref: the folder walk below can add images at
  // any poll, and refreshImages is defined after refreshPayload.
  const refreshImagesRef = useRef(null)

  const refreshPayload = useCallback(async (opts = {}) => {
    try {
      // ?refresh=1 forces the source-folder re-walk (the bank was just opened);
      // a plain poll lets the server's cooldown decide, so the 2 s job poll
      // doesn't hammer the disk.
      const d = await apiFetch(`/api/bank/${bankId}${opts.force ? '?refresh=1' : ''}`,
        { background: !!opts.background })
      setPayload(d)
      // Images dropped in the folder show up on their own — say it, and pull
      // them into the grid, so the counters never move without a reason.
      const note = folderSyncToast(d.folder_sync)
      if (note) toast[note.type](note.text)
      if ((d.folder_sync?.added || 0) > 0) refreshImagesRef.current?.()
      return d
    } catch (e) {
      if (String(e?.message || '').includes('not found')) { onGone?.(); return null }
      return null
    }
  }, [bankId, onGone, toast])

  const filterParams = useCallback((f) => {
    const params = {}
    if (f.status) params.status = f.status
    if (f.flag) params.flag = f.flag
    if (f.cluster != null) params.cluster = String(f.cluster)
    if (f.style != null) params.style = String(f.style)
    // subfolder is a string facet where '' is meaningful (bank root) — send it
    // whenever it isn't null, empty string included.
    if (f.subfolder != null) params.subfolder = f.subfolder
    if (f.search) params.search = f.search
    // The exclude terms travel with the search on every surface that reads a
    // filter — the grid, "Select all in filter", ▶ Review and the curation picks.
    if (f.exclude) params.exclude = f.exclude
    // 🏷️ ticked chips — its own key, matched as WORDS and ANDed server-side.
    if (f.tags) params.tags = f.tags
    // Grid sort (any measured quantity, each way) — sent to the grid
    // AND to fetchAllIds so "Select all in filter" and > Review walk the SAME
    // order the user is looking at. 'default' keeps the server's flag order.
    if (f.sort && f.sort !== 'default') params.sort = f.sort
    // Resolution tier — a facet like the flags; also flows to fetchAllIds so
    // "Select all in filter" stays scoped to the active tier.
    if (f.resBucket) params.res_bucket = f.resBucket
    // Framing bucket (face/bust/body/back/unknown) — a facet like the flags.
    if (f.framing) params.framing = f.framing
    // Origin state (ai/camera/unknown) — a facet like the flags.
    if (f.origin) params.origin = f.origin
    // 🎨 what the picture is made of, and ⤢ where the head points. Their OWN
    // query keys, so they compose with search/exclude/sort instead of fighting
    // them for one.
    if (f.medium) params.medium = f.medium
    if (f.angle) params.angle = f.angle
    return params
  }, [])

  const refreshImages = useCallback(async (f = filter, off = offset, view) => {
    // `view` (optional) lets a caller drive the fetch with values it just set,
    // dodging state-closure lag; otherwise read the current selection view.
    const on = view ? view.on : showSelected
    const order = view ? view.order : selectedOrder
    const params = on
      ? { ids: (order || []).join(','), offset: String(off), limit: String(PAGE_SIZE) }
      : { ...filterParams(f), offset: String(off), limit: String(PAGE_SIZE) }
    try {
      const d = await apiFetch(`/api/bank/${bankId}/images?${new URLSearchParams(params)}`)
      setPage(d)
    } catch { /* transient — next poll retries */ }
  }, [bankId, filter, offset, filterParams, showSelected, selectedOrder])

  useEffect(() => { refreshImagesRef.current = refreshImages }, [refreshImages])

  // The chip counters, re-measured whenever the filter — or the data under it —
  // moves. Skipped entirely while nothing is filtered: there the payload's
  // bank-wide numbers ARE the answer, so the unfiltered bank costs exactly what
  // it did before. `sort` is stripped: reordering the same rows cannot change a
  // count, and leaving it in would re-fetch every time the user changes the
  // grid's order. A failed fetch keeps the last read rather than falling back to
  // the bank-wide totals — flashing 4 043 for one tick is the very bug this
  // replaces.
  const facetDeps = facetDataKey(payload?.counts, payload?.thresholds)
  useEffect(() => {
    if (!isFacetFiltered(filter)) { setFacets(null); return undefined }
    let alive = true
    const params = filterParams(filter)
    delete params.sort
    apiFetch(`/api/bank/${bankId}/facets?${new URLSearchParams(params)}`)
      .then((d) => { if (alive) setFacets(d) })
      .catch(() => { /* transient — the chips keep their last read */ })
    return () => { alive = false }
  }, [bankId, filter, filterParams, facetDeps])

  useEffect(() => {
    // Opening ANOTHER bank without unmounting (the workspace is not keyed by id)
    // has to pick up THAT bank's remembered order, not keep the previous one —
    // and the first fetch must already use it, hence the explicit filter here
    // instead of a setFilter that the fetch below would race.
    const f = { ...filter, sort: loadBankSort(bankId) }
    if (f.sort !== filter.sort) setFilter(f)
    refreshPayload({ force: true }); refreshImages(f)
    apiFetch(`/api/bank/${bankId}/subfolders`)
      .then((d) => setSubfolders(d.subfolders || []))
      .catch(() => setSubfolders([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bankId])

  // Poll while a job runs; refresh the grid once when it lands.
  const live = payload?.activity && !payload.activity.finished
  useEffect(() => {
    if (!live) {
      if (activityWasLive.current) {
        activityWasLive.current = false
        refreshImages()
        if (payload?.activity?.error) toast.error(`Job failed — ${payload.activity.error}`)
        else if (payload?.activity?.cancelled && payload?.activity?.detail)
          toast.info(payload.activity.detail)   // Stopped — N cached (M remaining)…
        else if (payload?.activity?.detail) toast.success(payload.activity.detail)
      }
      return undefined
    }
    activityWasLive.current = true
    // ⏱ The 2 s tick asks for the JOB, not the dashboard. It used to re-fetch the
    // whole bank payload — ~60 bank-wide aggregates plus a per-image path walk —
    // which took 12.5 s at rest on a 50 397-image bank and 28.9 s while a pass
    // ran. Requests issued every 2 s that take 12 s stack up, and that payload is
    // what carries this banner: the bank went blank and Stop stopped answering,
    // at the one moment both were needed. /activity reads one indexed row.
    // The dashboard still refreshes, on its own slower beat, and IMMEDIATELY when
    // the job lands so the final counts are never a poll behind.
    let dropped = false
    const tick = async () => {
      let next = null
      try {
        // background: this tick is the one that stacked ten "Connection lost"
        // banners over the whole app when the phone's connection dropped.
        next = await apiFetch(`/api/bank/${bankId}/activity`, { background: true })
      } catch { return }               // transient — the next tick retries
      if (dropped) return
      setPayload((p) => (p ? { ...p, activity: next.activity } : p))
      const landed = !next.activity || next.activity.finished
      if (landed || Date.now() - fullPayloadAt.current >= FULL_PAYLOAD_MS) {
        fullPayloadAt.current = Date.now()
        refreshPayload({ background: true })
      }
    }
    const t = setInterval(tick, 2000)
    return () => { dropped = true; clearInterval(t) }
  }, [live, bankId, refreshPayload, refreshImages, toast, payload?.activity?.error,
      payload?.activity?.cancelled, payload?.activity?.detail])

  // Keep the coverage panel current: refetch when it opens, and whenever the kept
  // set or the framing classification changes (a keep/reject or the framing pass).
  useEffect(() => {
    if (coverageOpen) loadCoverage()
  }, [coverageOpen, loadCoverage, payload?.counts?.keep,
      payload?.counts?.framing_classified,
      // The panel now also reads captions and embeddings, so it must refresh
      // when those passes land — otherwise it keeps showing "no captions yet"
      // after the 🏷️ pass finished.
      payload?.counts?.captioned, payload?.counts?.scored,
      payload?.counts?.semantic_indexed, payload?.semantic?.engine])

  // 👤 "Single person here" — the folder-level person assertions. Reloaded when
  // a job LANDS too: the sample check writes its verdict from the background.
  const loadFolderPersons = useCallback(() => {
    apiFetch(`/api/bank/${bankId}/folder-persons`)
      .then((d) => { setFolderPersons(d.assertions || []); setFolderPersonInfo(d) })
      .catch(() => { setFolderPersons([]); setFolderPersonInfo(null) })
  }, [bankId])

  useEffect(() => { loadFolderPersons() }, [loadFolderPersons, live])

  // 🏷️ The pulled Ollama models, for the per-run caption model picker. Fetched ONCE per
  // mount and never blocking: the endpoint always answers 200, and an unreachable Ollama
  // is an empty list — the picker then offers only "Use the configured model", which is
  // exactly the truth on that machine.
  useEffect(() => {
    let alive = true
    apiFetch('/api/ollama/models').catch(() => ({ models: [] }))
      .then((d) => { if (alive) setOllamaModels(d?.models || []) })
    return () => { alive = false }
  }, [])

  const openSourceFolder = async () => {
    if (openingSourceFolder) return
    setOpeningSourceFolder(true)
    try {
      await postJson(`/api/bank/${bankId}/open-source-folder`, {})
    } catch (e) {
      toast.error(e?.message || 'Could not open the bank source folder')
    } finally {
      setOpeningSourceFolder(false)
    }
  }

  const runFolderPerson = async (call, success) => {
    setFolderPersonBusy(true)
    try {
      const d = await call()
      if (success) toast.success(success(d))
      // The payload too, not only the grid: an assertion creates (or dissolves)
      // a person cluster, and the PEOPLE row above would otherwise keep showing
      // a group that no longer exists until the next poll.
      loadFolderPersons(); refreshImages(); refreshPayload({ force: true })
    } catch (e) {
      toast.error(e?.message || 'That did not work')
    } finally { setFolderPersonBusy(false) }
  }

  const assertFolderPerson = () => runFolderPerson(
    () => postJson(`/api/bank/${bankId}/folder-person`, { subfolder: filter.subfolder }),
    (d) => `${d.images} image(s) grouped as person #${d.cluster_id} — the face pass `
      + 'will skip them',
  )

  const revokeFolderPerson = () => runFolderPerson(
    () => del(`/api/bank/${bankId}/folder-person`
      + `?subfolder=${encodeURIComponent(filter.subfolder ?? '')}`),
    (d) => `${d.cleared} image(s) back to normal clustering`,
  )

  const checkFolderPerson = () => runFolderPerson(
    () => postJson(`/api/bank/${bankId}/folder-person/check`,
      { subfolder: filter.subfolder }),
    (d) => `Checking ${d.sample_size} images of this folder…`,
  )

  const scanFolderPersons = () => runFolderPerson(
    () => postJson(`/api/bank/${bankId}/folder-scan`, {}),
    () => 'Sampling the folders — nothing is grouped until you confirm',
  )

  // Leaving the selection view: back to the facet grid.
  const exitSelectionView = () => { setShowSelected(false); setSelectedOrder(null) }

  // The "Show selected (N)" / "Show all" toggle. Entering keeps any curate ranking
  // (selectedOrder); a plain manual selection shows in insertion order.
  const toggleSelectionView = () => {
    if (showSelected) {
      exitSelectionView(); setOffset(0); refreshImages(filter, 0, { on: false })
    } else {
      const order = (selectedOrder && selectedOrder.length) ? selectedOrder : [...selected]
      setSelectedOrder(order); setShowSelected(true); setOffset(0)
      refreshImages(filter, 0, { on: true, order })
    }
  }

  const setF = (patch) => {
    const f = { ...filter, ...patch }
    setFilter(f); setOffset(0); setSelected(new Set()); exitSelectionView()
    refreshImages(f, 0, { on: false })
  }

  // Sort only reorders — the same rows match, so the selection (a set of ids)
  // is kept; just jump back to page 1 to read the new order top-down. A facet
  // sort has no meaning inside the selection view, so it drops back to the grid.
  const setSort = (sort) => {
    const f = { ...filter, sort }
    saveBankSort(bankId, sort)
    setFilter(f); setOffset(0); exitSelectionView()
    refreshImages(f, 0, { on: false })
  }

  // 🏷️ Open the chip row on an image, with nothing ticked yet: reading the tags
  // is not the same act as filtering by them, and auto-applying all of them would
  // usually return that one image alone.
  //
  // It CLEARS the selection, because the selection now drives the same row: with
  // one live, this button would set a source the row never reads and read as dead.
  const openTagPicker = (img) => {
    setTagSource(img)
    setTagPicked(new Set())
    setTagFreeze(null)
    setSelected(new Set())
    if (filter.tags) setF({ tags: null })
  }

  // Tick / untick one chip and re-filter immediately — the grid IS the feedback,
  // so an Apply button would only add a click between the question and its answer.
  //
  // `basis` is the row the chip was clicked in. When it is the LIVE selection it
  // is frozen first: setF empties the selection on the next line, and a row that
  // vanished the moment you used it would read as a crash.
  const toggleTag = (tag, basis = null) => {
    const next = new Set(tagPicked)
    if (next.has(tag)) next.delete(tag); else next.add(tag)
    setTagPicked(next)
    if (basis && basis.kind === 'selection' && !basis.frozen) {
      setTagFreeze({ ...basis, frozen: true })
    }
    setF({ tags: tagsParam(next) })
  }

  const clearTags = () => {
    setTagSource(null)
    setTagFreeze(null)
    setTagPicked(new Set())
    if (filter.tags) setF({ tags: null })
  }

  // --- 🏷️ The captions the tag row counts over ------------------------------
  // Every image the grid has ever rendered leaves its caption here, so ticking
  // tiles on the page costs ZERO requests. Only ids the grid never showed — what
  // "Select all in filter" produces — are fetched, and only those.
  const captionCache = useRef(null)
  if (captionCache.current === null) captionCache.current = new Map()
  const [captionsSeen, setCaptionsSeen] = useState(0)
  useEffect(() => {
    let added = false
    for (const im of page.images || []) {
      if (!captionCache.current.has(im.id)) added = true
      captionCache.current.set(im.id, im.caption || '')
    }
    if (added) setCaptionsSeen((n) => n + 1)
  }, [page])

  // The ids we never rendered. Fetched in ONE request and capped: `ids=` rides in
  // the QUERY STRING, and a selection of a few thousand integers builds a request
  // line the server rejects outright. What the cap leaves out is stated in the row
  // rather than folded silently into the denominator.
  useEffect(() => {
    const ids = [...selected]
    const missing = ids.filter((id) => !captionCache.current.has(id))
    if (!missing.length) return undefined
    const batch = missing.slice(0, TAG_CAPTION_FETCH_CAP)
    let cancelled = false
    const t = setTimeout(async () => {
      try {
        const qs = new URLSearchParams({
          ids: batch.join(','), limit: String(batch.length),
        })
        const d = await apiFetch(`/api/bank/${bankId}/images?${qs}`, { background: true })
        if (cancelled) return
        for (const im of d.images || []) captionCache.current.set(im.id, im.caption || '')
        setCaptionsSeen((n) => n + 1)
      } catch { /* the row then counts what it has and says how many it could not read */ }
    }, 300)
    return () => { cancelled = true; clearTimeout(t) }
  }, [selected, bankId])

  // A new selection replaces a frozen one: a snapshot of last question's images
  // sitting above a fresh selection is a row that describes something else.
  useEffect(() => { if (selected.size) setTagFreeze(null) }, [selected])

  const selectionTags = useMemo(() => {
    if (!selected.size) return null
    const ids = [...selected]
    const known = ids.filter((id) => captionCache.current.has(id))
    return {
      kind: 'selection',
      ...selectionTagCounts(known.map((id) => captionCache.current.get(id))),
      unread: ids.length - known.length,
      size: ids.length,
    }
    // captionsSeen is the dependency that matters — the cache is a ref, so React
    // cannot see it change on its own.
  }, [selected, captionsSeen])

  /* WHICH row is on screen, in priority order. A frozen selection outranks a live
     one (it IS the live one, held still while its filter runs); a selection
     outranks the 🏷️ button, because it is the more recent gesture. */
  const tagRow = tagFreeze || selectionTags || (tagSource ? {
    kind: 'image',
    name: tagSource.name,
    ...selectionTagCounts([tagSource.caption]),
    unread: 0,
    size: 1,
  } : null)

  // Debounce the search box, then apply it as a filter (page 1, selection cleared).
  useEffect(() => {
    const term = searchText.trim()
    if ((filter.search || '') === term) return undefined
    const t = setTimeout(() => setF({ search: term || null }), 300)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchText])

  // Same debounce for the exclude box. It is a FILTER like any other, so it goes
  // through setF: page 1, selection cleared, and it rides to the curation
  // endpoints too (filterParams) — hiding an image in the grid must also keep it
  // out of "pick 60 diverse".
  useEffect(() => {
    const term = excludeText.trim()
    if ((filter.exclude || '') === term) return undefined
    const t = setTimeout(() => setF({ exclude: term || null }), 300)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [excludeText])
  const goto = (off) => { setOffset(off); refreshImages(filter, off) }

  /* `onRefusal` is for a caller that OWNS a surface for the refusal — today the
     🚀 Launch all dialog, which draws it next to the checkboxes it is about and
     stays open so they survive. When it is given, the message goes THERE instead
     of to a toast: two copies of the same sentence is not twice as clear. Every
     other button keeps the toast, because it has nowhere else to put it. */
  const act = async (fn, okMsg, { onRefusal } = {}) => {
    try {
      const d = await fn()
      if (okMsg) toast.success(okMsg)
      await refreshPayload(); await refreshImages()
      return d
    } catch (e) {
      // ONE bank, ONE background job: every action here can be refused because
      // another pass owns the bank. That refusal used to reach the user as the
      // server's own sentence — "a scan job is already running on this bank" —
      // which names no progress and no way out. The route now labels the 409
      // with `busy_kind`, so it is rewritten HERE, once, for every button that
      // goes through act(): the ✨ Analyze, the ↻ re-runs in the threshold
      // panel, 🗑 Delete rejected, ⬆ Promote, 🚀 Launch all. Anything else keeps
      // its own message — only a refusal that identified itself is reworded.
      const kind = e?.body?.busy_kind
      const message = e?.status === 409 && kind
        ? busyRefusal({ kind, activity: payload?.activity })
        : (e?.message || 'Action failed.')
      if (onRefusal) onRefusal(message)
      else toast.error(message)
      return null
    }
  }

  const runFacesPass = () => act(() => postJson(`/api/bank/${bankId}/faces`, {}), null)

  /* 👤 THE PREFLIGHT GATE.
     The person pass is the bank's most expensive one, and on scraped material
     most of it is spent rediscovering what the folder names already said. The
     app knew how to sample folders and offer the obvious ones — but only from a
     panel the user who presses 🚀 Launch all never opens. So the sampling now
     runs HERE, in front of the pass, whichever button started it.

     `gate` returns:
       false      — nothing worth asking (no subfolder, or all declared already);
                    the caller runs its pass immediately, exactly as before.
       'refused'  — the bank is busy. The refusal is raised at THIS point on
                    purpose: it is the same 409 the pass itself would produce, so
                    the Launch all dialog still gets it while its checkboxes are
                    alive, instead of losing them to a preflight that dies later.
       true       — the dialog is up and owns the decision. */
  const gate = async (run, onRefusal) => {
    let plan = null
    try { plan = await apiFetch(`/api/bank/${bankId}/person-preflight`) } catch { plan = null }
    if (!preflightNeeded(plan)) return false
    if (preflightWillSample(plan)) {
      try {
        await postJson(`/api/bank/${bankId}/person-preflight`, {})
      } catch (e) {
        const kind = e?.body?.busy_kind
        const message = e?.status === 409 && kind
          ? busyRefusal({ kind, activity: payload?.activity })
          : (e?.message || 'Could not check the folders.')
        if (onRefusal) onRefusal(message); else toast.error(message)
        return 'refused'
      }
      // The 2 s poll only ticks while a job is live, and the payload we hold
      // predates the one we just started — refresh so the dialog can watch it.
      refreshPayload({ force: true })
    }
    setPreflight({ plan, probing: preflightWillSample(plan), run })
    return true
  }


  /* The answer. Accepted folders become ORDINARY assertions (same endpoint
     family, same revoke) and only then does the pass the user asked for run. */
  const proceedPreflight = async ({ accept }) => {
    const run = preflight?.run
    setPreflight(null)
    if (accept && accept.length) {
      const d = await act(
        () => postJson(`/api/bank/${bankId}/folder-persons/accept`, { subfolders: accept }),
        null)
      if (d) {
        const missed = (d.failed || []).length
        toast.success(`${d.accepted.length} folder(s) · ${d.images} image(s) grouped `
          + 'as one person each — the pass will skip them'
          + (missed ? ` · ${missed} could not be grouped` : ''))
      }
      loadFolderPersons()
    }
    if (run) await run()
  }
  // ✨ Score always covers the WHOLE bank and skips what is already computed —
  // `rescore` is the explicit "recompute it all" lane (new model, scores you no
  // longer trust), exactly like the quality pass's "Rescan all".
  /* 🎛 ONE launcher for every pass window.
     The window hands back WHERE the run applies ({statuses} for a pile, the
     'selection' marker for the images the user ticked, `redo` for the "do it
     again on rows that already have a result" line). Everything is spread-if-set,
     so a run that changes nothing posts the SAME body the button posted before
     the windows existed — the byte-identical contract the caption options already
     followed, now the rule for all of them.

     It answers {ok,error} rather than toasting: the window owns its own refusal
     surface and stays open with the choices intact (utils/submitOutcome.js). */
  const passBody = (passId, { statuses, imageIds, redo } = {}, extra = {}) => {
    const spec = BANK_PASSES[passId]
    return {
      ...(spec?.redo?.explicit
        ? { [spec.redo.key]: !!redo }
        : (spec?.redo && redo ? { [spec.redo.key]: true } : {})),
      ...(statuses ? { statuses } : {}),
      ...(imageIds === 'selection' && selected.size ? { image_ids: [...selected] } : {}),
      ...extra,
    }
  }

  const runPass = async (passId, run, extra = {}) => {
    const spec = BANK_PASSES[passId]
    if (!spec) return { ok: false, error: 'Unknown pass.' }
    let error = null
    const d = await act(
      () => postJson(`/api/bank/${bankId}/${spec.endpoint}`, passBody(passId, run, extra)),
      null, { onRefusal: (m) => { error = m } })
    return d ? { ok: true } : { ok: false, error }
  }

  /* 👥 is the one pass whose launch is not a POST: the folder preflight can take
     ownership of it. When it does, this window has nothing left to do and closes
     on a success — the run is now the preflight dialog's to start. */
  const launchFacesFromDialog = async () => {
    let error = null
    const gated = await gate(runFacesPass, (m) => { error = m })
    if (gated === 'refused') return { ok: false, error }
    if (gated === true) return { ok: true }
    return runPass('faces', {})
  }
  /* Every option is spread-if-set, so a run that changes nothing posts the SAME body it
     posted before any of these controls existed — the contract the vocabulary/length
     pair set and the two new dials join.

     `statuses` is deliberately omitted while a selection is live: the server INTERSECTS
     the two, so "kept only" plus a selection of undecided images would caption fewer
     than the button says. The selection wins, the scope select goes inert, and the label
     switches to the selection count. */
  const captionRunOptions = () => ({
    ...(captionVocab ? { vocabulary: captionVocab } : {}),
    ...(captionLength ? { length: captionLength } : {}),
    ...(captionEngine ? { backend: captionEngine } : {}),
    ...(captionModel ? { ollama_model: captionModel } : {}),
  })
  const startCaption = (run) => runPass('caption', run, captionRunOptions())
  const cancelJob = () => act(() => postJson(`/api/bank/${bankId}/cancel`, {}), null)

  const changeSemanticEngine = async (engine) => {
    if (engine === semanticState.engine || semanticSwitching || semanticOperationBusy || live) return
    const previousEngine = semanticState.engine
    setSemanticSwitching(true)
    try {
      const d = await act(
        () => patchJson(`/api/bank/${bankId}/semantic-engine`, semanticEnginePatchBody(engine)),
        `Semantic engine changed to ${engine === 'siglip2' ? 'SigLIP 2' : 'CLIP'} — both caches were kept.`,
      )
      if (d) {
        // A text search may have left the previous engine warm. It no longer
        // belongs to this Bank view after a successful switch.
        releaseTextEncoder(previousEngine)
        // Close engine-specific explanatory overlays; the user's selection,
        // images and both caches stay exactly where they are.
        setCurateOpen(null)
        setTextStatus(null)
        setTextResult(null)
        setBalanceResult(null)
        // A curated order and coverage response belong to one vector space. Keep
        // the selection itself, but return to the ordinary grid and invalidate
        // every old-engine coverage request/result before showing the new engine.
        exitSelectionView()
        setOffset(0)
        refreshImages(filter, 0, { on: false })
        coverageRequestRef.current += 1
        textStatusRequestRef.current += 1
        setCoverage(null)
      }
    } finally {
      setSemanticSwitching(false)
    }
  }
  /* 🔄 THE DESTRUCTIVE TWIN. Same endpoint, same options, plus `force:true` — which
     drops the server's "no caption yet" filter and rewrites the whole pile. It exists
     because 🏷️ Caption greys out at zero uncaptioned rows and takes the engine/model
     selects down with it, leaving a fully captioned bank with no way to redo its
     captions with a better model.

     THREE RULES, all of them about not lying:
     - it ASKS FIRST, with the count of captions it will destroy, in the Dataset's own
       wording (dataset/captionCategory.js) so the app has one way of asking this;
     - it never carries `image_ids`. A selection can span pages that were never loaded,
       so how many selected rows already have a caption is unknowable client-side, and
       this button does not run on a number it cannot state (see
       captionRecaptionDisabledReason). 🏷️ Caption still honours selections;
     - `statuses` rides WITHOUT the `!selected.size` guard the normal pass needs,
       precisely because a selection makes this button inert instead. */
  const startRecaption = async () => {
    if (captionRecaptionDisabledReason(selected.size, live, counts, captionScope,
                                       captionIncludeAsserted)) {
      return { ok: false, error: null }
    }
    if (!window.confirm(captionRecaptionConfirmation(counts, captionScope,
                                                     captionIncludeAsserted))) {
      // A declined confirmation is not a failure: the window stays as it is and
      // says nothing (the question they just answered IS the explanation).
      return { ok: false, error: null }
    }
    let error = null
    const d = await act(() => postJson(`/api/bank/${bankId}/caption`, {
      force: true,
      // Sent ONLY when ticked. Omitting the key is the protected reading, on this
      // side as on the server's — the destructive one is never the default shape
      // of the request.
      ...(captionIncludeAsserted ? { include_asserted: true } : {}),
      ...captionRunOptions(),
      ...(captionScopeStatuses(captionScope)
        ? { statuses: captionScopeStatuses(captionScope) } : {}),
    }), null, { onRefusal: (m) => { error = m } })
    return d ? { ok: true } : { ok: false, error }
  }
  /* Posts with the dialog still OPEN and answers {ok,error}: a refused launch —
     "a scan job is already running on this bank" is the usual one — used to close
     the dialog first and reset all seven pass checkboxes and the reject flags to
     their defaults. The dialog decides what to do with the answer.

     🚀 Launch all is also the FIRST thing a new user presses, so it is the path
     the folder check has to be on — otherwise that saving only ever reaches the
     people who went looking for it. The question is asked BEFORE the run starts,
     which is the only honest place for it: the point of Launch all is walking
     away, and a dialog that woke the user three passes in would defeat that.
     The gate posts too, so a busy bank is refused HERE, while the checkboxes
     are still alive — the rule this whole contract is about. */
  const startPipeline = async (config) => {
    if ((config?.steps || []).includes('faces')) {
      let error = null
      const gated = await gate(() => runPipeline(config), (m) => { error = m })
      if (gated === 'refused') return { ok: false, error }
      // The preflight dialog owns the launch now; the Launch all card can close.
      if (gated === true) { setLaunchOpen(false); return { ok: true } }
    }
    return runPipeline(config)
  }

  const runPipeline = async (config) => {
    let error = null
    const d = await act(() => postJson(`/api/bank/${bankId}/pipeline`, config),
      '🚀 Launch all started — you can walk away; Stop any time.',
      { onRefusal: (m) => { error = m } })
    if (!d) return { ok: false, error }
    setLaunchOpen(false)
    return { ok: true }
  }

  const batchStatus = async (ids, status) => {
    if (!ids.length) return
    await act(() => postJson(`/api/bank/${bankId}/images/status`, { ids, status }),
      `${ids.length} image(s) → ${status}`)
    setSelected(new Set())
    // The selection is gone, so its view has nothing left to show — return to the grid.
    if (showSelected) { exitSelectionView(); setOffset(0); refreshImages(filter, 0, { on: false }) }
  }

  // ↩ Put the last bulk decision back. The reply is a ledger, not an "ok": it
  // is reported verbatim, so a restore that only got 340 of 400 rows back says
  // which ones it could not take and why.
  const undoLast = async () => {
    setUndoBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/undo`, {})
      const msg = undoResultMessage(d)
      toast[msg.type === 'error' ? 'error' : msg.type](msg.text)
      setSelected(new Set())
      if (showSelected) exitSelectionView()
      await refreshPayload(); await refreshImages()
    } catch (e) {
      toast.error(e?.message || 'Undo failed — nothing was changed.')
      await refreshPayload()
    } finally {
      setUndoBusy(false)
    }
  }

  // 🔄 Quarter turns on the selection (idea by 1Tomber, GitHub #17). A whole
  // scraped folder can come out sideways, so this is a BULK action here rather
  // than a per-tile button — the tile already carries the selection gesture, the
  // status badges and two hit targets.
  const rotateSelection = async (degrees) => {
    const ids = [...selected]
    if (!ids.length) return
    const d = await act(() => postJson(`/api/bank/${bankId}/rotate`, { ids, degrees }),
      null)
    if (d?.rotated) {
      toast.success(`${d.rotated} image(s) rotated 90° ${degrees === 90 ? 'right' : 'left'}`
        + ' — your own files are untouched.')
      await refreshImages()
    }
  }

  const applyAutoReject = async () => {
    setShowAutoReject(false)
    const flags = [...rejectFlags]
    const d = await act(() => postJson(`/api/bank/${bankId}/apply-flags`, { flags }), null)
    if (d?.rejected) {
      const n = Object.values(d.rejected).reduce((a, b) => a + b, 0)
      // Zero is not a success. It used to be announced in green as "0 image(s)
      // rejected", which on a second pass is the exact shape of a broken
      // feature — the pass had nothing left to do because the first one had
      // already rejected them all, and nothing said so.
      if (n === 0) {
        toast.info(`Auto-reject: nothing to do (${flags.join(', ')}) — every image carrying`
          + ' these flags has already been decided. Manual ✓/✕ are never re-flipped.')
      } else {
        toast.success(`Auto-reject: ${n} image(s) rejected (${flags.join(', ')}). Manual ✓/✕ untouched.`)
      }
    }
  }

  // Open ▶ Review over what the user is actually looking at: the whole current
  // filter (all pages, current sort), or the selection when the "Show selected"
  // view is on. `startId` (the ▶ on a tile) opens on that image.
  const openReview = async (startId = null) => {
    setReviewLoading(true)
    try {
      const ids = showSelected
        ? ((selectedOrder && selectedOrder.length) ? selectedOrder : [...selected])
        : await fetchAllIds(bankId, filterParams(filter))
      if (!ids.length) {
        toast.info('Nothing to review — no image matches the current filter.')
        return
      }
      setReview({ ids, startId })
    } catch (e) {
      toast.error(e?.message || 'Could not build the review list.')
    } finally {
      setReviewLoading(false)
    }
  }

  // One decision landed in the lightbox — refresh the header counters so
  // kept/rejected/undecided track the run live. The grid is refreshed once, on
  // close, so its tiles don't shuffle around behind the lightbox.
  const onReviewDecided = () => { refreshPayload() }
  // A turn made in ▶ Review must already be right on the tile behind it: the
  // grid is only refetched on close, and a tile still lying sideways would read
  // as "it didn't take".
  const onReviewRotated = (imageId, rotation) => setPage((prev) => ({
    ...prev,
    images: prev.images.map((im) => (im.id === imageId
      ? { ...im, rotation, width: im.height, height: im.width }
      : im)),
  }))
  const closeReview = () => { setReview(null); refreshPayload(); refreshImages() }

  const selectAllCurrent = async () => {
    try {
      const ids = await fetchAllIds(bankId, filterParams(filter))
      setSelected(new Set(ids))
      toast.info(`${ids.length} image(s) selected (whole filter, all pages).`)
    } catch (e) {
      toast.error(e?.message || 'Selection failed.')
    }
  }

  // --- Curation selectors (read the selected semantic index) -----------------
  // Both build a SELECTION the user then reviews with the existing ✓/✕/Promote
  // bar — nothing is auto-kept or deleted. The candidate pool is the current
  // filter (composable), so "60 most diverse of this subfolder" just works.
  // Curate a selection AND switch to the "show selected" view so the result is
  // actually visible (60 ids scattered across a 24k-image bank are invisible as
  // mere checkmarks). `order` is the ids in the order the grid should render them.
  const showCuratedSelection = (order) => {
    setSelected(new Set(order))
    setSelectedOrder(order)
    setShowSelected(true)
    setOffset(0)
    refreshImages(filter, 0, { on: true, order })
  }

  // The typicality guard reads the whole pool's neighbourhood before sampling, so
  // on a big bank this click is no longer instant — say so instead of looking dead.
  const pickDiverse = async () => {
    const requestEngine = semanticState.engine
    const requestModelKey = semanticState.modelKey
    setCurateOpen(null)
    setDiverseBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/select-diverse`,
        { n: diverseN, typicality: diverseTypicality, ...filterParams(filter) })
      if (semanticEngineRef.current !== requestEngine
          || !semanticPayloadMatches(d, requestEngine, requestModelKey)) return
      if (!d.image_ids?.length) {
        toast.info(`Nothing to sample — no ${semanticState.label}-indexed images match the current filter.`)
        return
      }
      showCuratedSelection(d.image_ids)
      toast.info(`Showing the ${d.image_ids.length} most diverse of ${d.pool}. Review, then ✓ Keep or ⬆ Promote — or “Show all” to leave this view.`)
    } catch (e) {
      toast.error(e?.message || 'Diversity sampling failed.')
    } finally {
      setDiverseBusy(false)
    }
  }

  // ⚖️ Balanced pick — spread over the framings instead of taking the top of one
  // ranking. Same embeddings and same typicality guard as 🎨 Pick diverse, applied
  // INSIDE each bucket. The result is only useful if the user can see its shape,
  // so the distribution is kept on screen (numbers, aria-live) after the click.
  const pickBalanced = async () => {
    const requestEngine = semanticState.engine
    const requestModelKey = semanticState.modelKey
    setCurateOpen(null)
    setBalanceBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/select-balanced`,
        { n: balanceN, axis: balanceAxis, typicality: diverseTypicality,
          ...filterParams(filter) })
      if (semanticEngineRef.current !== requestEngine
          || !semanticPayloadMatches(d, requestEngine, requestModelKey)) return
      if (!d.image_ids?.length) {
        toast.info('Nothing to balance — no labelled images match the current filter.')
        return
      }
      setBalanceResult(d)
      showCuratedSelection(d.image_ids)
      toast.info(summarizeBalance(d))
    } catch (e) {
      // A missing pass is the DEFAULT state of a fresh bank, not a failure: the
      // backend names the pass, so show that sentence rather than "failed".
      toast.error(e?.message || 'Balanced selection failed.')
    } finally {
      setBalanceBusy(false)
    }
  }

  const findSimilar = async () => {
    const requestEngine = semanticState.engine
    const requestModelKey = semanticState.modelKey
    setCurateOpen(null)
    const ref = [...selected][0]
    if (ref == null) return
    setSimilarBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/select-similar`,
        { ref_id: ref, n: similarN, ...filterParams(filter) })
      if (semanticEngineRef.current !== requestEngine
          || !semanticPayloadMatches(d, requestEngine, requestModelKey)) return
      if (!d.image_ids?.length) {
        toast.info(`No matches — no ${semanticState.label}-indexed images match the current filter.`)
        return
      }
      // Backend returns the ids ranked by similarity (reference first); keep that
      // order so the view reads closest→farthest instead of by id.
      showCuratedSelection(d.image_ids)
      toast.info(`Showing the ${d.image_ids.length} most similar to the reference (of ${d.pool}), closest first. Review, then ✓ Keep or ⬆ Promote — or “Show all” to leave this view.`)
    } catch (e) {
      toast.error(e?.message || 'Similarity search failed.')
    } finally {
      setSimilarBusy(false)
    }
  }

  // 🔤 Text search — same engine as 🎯 Similar, with the reference vector coming
  // from words instead of a picture. Opening the panel asks the backend what it
  // is about to cost (model warm? weights present?) so a slow FIRST search is
  // announced before the click rather than felt as a freeze after it.
  const openTextSearch = async () => {
    const next = curateOpen === 'text' ? null : 'text'
    setCurateOpen(next)
    if (next !== 'text') {
      textStatusRequestRef.current += 1
      releaseTextEncoder()
      return
    }
    const requestId = ++textStatusRequestRef.current
    const expectedEngine = semanticState.engine
    if (semanticState.text) setTextStatus(semanticState.text)
    try {
      const status = await apiFetch('/api/bank/text-search/status'
        + `?engine=${encodeURIComponent(expectedEngine)}`)
      if (requestId === textStatusRequestRef.current
          && expectedEngine === semanticEngineRef.current
          && semanticPayloadMatches(status, expectedEngine)) setTextStatus(status)
    } catch {
      // The Bank payload already carries an engine-aware status. Keep it when
      // the optional warm/cold probe cannot be read.
      if (!semanticState.text) setTextStatus(null)
    }
  }

  // Hand the selected text encoder's memory back as soon as the panel closes.
  // Best effort by design — the backend idle timer remains the guarantee for a
  // tab that simply vanished.
  const releaseTextEncoder = (engine = semanticState.engine) => {
    postJson('/api/bank/text-search/release', semanticEnginePatchBody(engine)).catch(() => {})
  }

  // Leaving the Bank entirely is the same signal as closing the panel: give the
  // memory back. The backend idle timer still covers a browser that just died.
  useEffect(() => () => {
    postJson('/api/bank/text-search/release',
      semanticEnginePatchBody(semanticEngineRef.current)).catch(() => {})
  }, [])

  const runTextSearch = async () => {
    const q = textQuery.trim()
    if (!q) return
    const requestEngine = semanticState.engine
    const requestModelKey = semanticState.modelKey
    setTextPending(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/search-text`,
        { query: q, n: textN, push_down: textExclude.trim() || null,
          push_down_weight: textExcludeW, ...filterParams(filter) })
      if (semanticEngineRef.current !== requestEngine
          || !semanticPayloadMatches(d, requestEngine, requestModelKey)) return
      setTextResult(d)
      setCurateOpen(null)
      if (!d.image_ids?.length) {
        // NOT a silent empty grid: say why nothing could be ranked.
        toast.info(summarize(d, semanticState.engine))
        return
      }
      showCuratedSelection(d.image_ids)
      // Refresh the warm flag so the panel now promises "instant" truthfully.
      apiFetch('/api/bank/text-search/status'
        + `?engine=${encodeURIComponent(semanticState.engine)}`)
        .then(setTextStatus).catch(() => {})
    } catch (e) {
      // 503 = this install cannot do it at all; 400 = do something first. Both
      // arrive as a message written for a human — show it as-is.
      toast.error(e?.message || 'Text search failed.')
    } finally {
      setTextPending(false)
    }
  }

  const counts = payload?.counts
  const semanticState = semanticEngineState(payload, capsLoading ? null : caps)
  semanticEngineRef.current = semanticState.engine
  semanticModelKeyRef.current = semanticState.modelKey
  const semanticReady = semanticState.ready
    && !(semanticState.engine === 'siglip2' && capsLoading)
  const semanticIndexed = semanticState.indexed
  const semanticBlocked = semanticState.engine === 'siglip2' && capsLoading
    ? 'Checking whether the SigLIP 2 Quality tool is installed…'
    : semanticPrerequisite(semanticState)
  // The Sort menu greys an entry out when its pass has measured NOTHING. Face
  // confidence is the one whose progress the payload reports outside `counts`
  // (faces_scanned, a sibling key), so it is folded in here rather than by
  // changing the payload shape every other reader depends on.
  const sortGroups = bankSortGroups(
    counts ? { ...counts, faces: payload?.faces_scanned } : counts)
  // ↩ the live offer, minus the one the user already waved away.
  const offer = undoOffer(payload)
  const undoBar = offer && offer.at !== undoDismissedAt ? offer : null
  // `print` = what each chip SHOWS, measured under the filters in force; `wide` =
  // the bank-wide truth, which is what decides a chip is offered at all. Keeping
  // visibility on `wide` is deliberate: chips that disappeared as a filter
  // emptied them would leave no way back to the values you just excluded.
  const { print: chipPrint, wide: chipWide, filtered: chipsFiltered } =
    chipCounts(payload, facets)
  const flags = chipPrint.flags
  // A THIRD question, kept as its own map and deliberately NOT filtered:
  // `flags` is "images this chip would show" and `flags_actionable` is "what
  // 🧹 Auto-reject would flip" — a pass that runs over the whole bank, not over
  // the current view, so narrowing its number to the filter would under-promise
  // exactly as badly as the chips used to over-promise.
  const flagsActionable = payload?.flags_actionable || {}
  const resBuckets = chipPrint.resBuckets
  // Only surface tiers that actually hold scanned images (plus the active one,
  // so a tier you're filtering on never vanishes mid-review).
  const shownResBuckets = RES_BUCKETS.filter(
    (b) => (chipWide.resBuckets[b.id] || 0) > 0 || filter.resBucket === b.id)
  const clusters = payload?.clusters || []
  const styleClusters = payload?.style_clusters || []
  const framingCounts = chipPrint.framing
  const framingClassified = counts?.framing_classified || 0
  // Only surface framing chips once the pass has classified something (plus the
  // active one, so a chip you're filtering on never vanishes mid-review).
  const shownFramings = FRAMING_BUCKETS.filter(
    (b) => (chipWide.framing[b.id] || 0) > 0 || filter.framing === b.id)
  // Origin chips appear as soon as the quality scan has measured anything. All
  // three are shown together once any is non-zero: hiding 'ai' at 0 would read as
  // "no AI images here", when what it means is "none that still carry metadata".
  const originCounts = chipPrint.origins
  const originMeasured = ORIGIN_BUCKETS.reduce(
    (n, b) => n + (chipWide.origins[b.id] || 0), 0)
  // 🎨 Medium / ⤢ Angle — same "only show what holds something, plus the active
  // one" rule as the framing and resolution rows.
  const mediumCounts = chipPrint.mediums
  const shownMediums = shownBuckets(MEDIUM_BUCKETS, chipWide.mediums, filter.medium)
  // The limits sentence is about the MEASUREMENT, not about the current view: it
  // weighs the buckets against how many rows the pass classified bank-wide, so
  // it reads the wide map or it would announce a fake blind spot on every filter.
  const mediumNote = mediumLimits(chipWide.mediums, counts?.medium_classified)
  const angleCounts = chipPrint.angles
  const shownAngles = shownBuckets(ANGLE_BUCKETS, chipWide.angles, filter.angle)
  const angleState = angleReadiness(payload)
  const visionReady = !!caps.ollama?.vision_model_ready
  // The explicit lane only spells acts out with an uncensored (abliterated) vision
  // model. We can't prove abliteration, but the common builds name themselves — a soft
  // heuristic drives an honest "may soften" hint (never a hard block: a differently
  // named abliterated model still works).
  // It reads the EFFECTIVE model — this run's override if one was picked, else the
  // configured one. Warning about the global model while the run uses another is worse
  // than not warning at all.
  const visionModel = captionModel || caps.ollama?.vision_model || ''
  const visionModelLooksUncensored = /abliterat|uncensor|huihui|nsfw/i.test(visionModel)
  // The Ollama model choice only bites when the resolved engine can reach Ollama.
  const ollamaPicksApply = OLLAMA_RELEVANT.has(captionEngine)
  // A model pulled elsewhere (or configured in Settings) stays selectable even when the
  // live list doesn't carry it — silently dropping the user's choice is worse than
  // offering a name we can't confirm.
  const captionModelChoices = captionModel && !ollamaModels.includes(captionModel)
    ? [captionModel, ...ollamaModels] : ollamaModels
  // 🔄 Re-caption: inert (and why), plus the sentence that names what it destroys.
  const includeAssertedLabel = captionIncludeAssertedLabel(counts, captionScope)
  const recaptionInert = captionRecaptionDisabledReason(
    selected.size, live, counts, captionScope, captionIncludeAsserted)
  const recaptionNote = captionRecaptionNote(selected.size, live, counts, captionScope,
                                             captionIncludeAsserted)

  /* 🏷️ THE FIVE CAPTION OPTIONS, now INSIDE the caption window — literally the
     maintainer's example of what these windows are for ("this way we can gather
     all the caption options"). Nothing was dropped in the move: engine, model,
     register, length, the pile (which became the window's THIS RUN block) and
     🔄 Re-caption with both its figures and its confirmation.

     The state stays out here, in the workspace, so closing the window does not
     reset a choice the user made. Each select keeps the width rules measured at a
     400 px viewport: only the MODEL one is capped at 11rem (Ollama refs run long
     and overflow on their own); the rest are max-w-full with a 16rem ceiling from
     sm up, because capping them by symmetry truncated them into nonsense
     ("Standard — the prompt as⌄"). */
  const captionSelectClass = 'mt-0.5 w-full rounded-lg border border-border bg-app/60 '
    + 'px-2 py-1 text-xs text-content disabled:opacity-40'
  /* Rendered UNDER the engine picker, because it is about the engine and the picker
     is the lever it names. It re-computes as the engine changes, so ticking
     JoyCaption turns the warning into its own confirmation instead of leaving an
     alarm on screen about a half that will no longer run. */
  const captionNsfw = captionNsfwNotice({
    payload,
    scopeId: captionScope,
    piles: passScopeOption(captionScope).piles,
    engineId: captionEngine,
  })
  const captionRunControls = (
    <div className="space-y-2 rounded-md border border-border bg-surface-raised p-2">
      <p className="m-0 text-[11px] font-semibold uppercase tracking-wide text-content-muted">
        Options for this run
      </p>
      <p className="m-0 text-[11px] leading-snug text-content-subtle">
        These override your Settings for this run only — the global values are never
        written from here.
      </p>
      <div className="grid gap-2 sm:grid-cols-2">
        <label className="block text-[11px] text-content-subtle">
          Engine
          <select value={captionEngine} onChange={(e) => setCaptionEngine(e.target.value)}
            disabled={live} aria-label="Caption engine"
            title="Which engine writes this run's captions, without changing your Settings. Auto is a CHAIN, not a choice between two: JoyCaption drafts, then Ollama covers whatever it missed."
            className={`${captionSelectClass} sm:max-w-[16rem]`}>
            {/* 'none' is dropped on purpose: "caption with nothing" is not a pass. */}
            {ENGINE_OPTIONS.filter((o) => o.id !== 'none')
              .map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
        </label>
        <label className="block text-[11px] text-content-subtle">
          Vision model
          <select value={captionModel} onChange={(e) => setCaptionModel(e.target.value)}
            disabled={live || !ollamaPicksApply} aria-label="Caption vision model"
            title={ollamaPicksApply
              ? 'Which pulled Ollama vision model writes this run. Your Settings model stays the default and is not changed. Which model writes a caption is not a matter of taste: one that describes things in evasive terms produces captions that are about something slightly other than the images.'
              : 'Only used when the engine can reach Ollama.'}
            className={`${captionSelectClass} sm:max-w-[11rem]`}>
            <option value="">Configured model</option>
            {captionModelChoices.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          {!ollamaPicksApply && (
            <span className="mt-0.5 block text-[11px] leading-snug text-amber-300/90">
              The engine you picked does not reach Ollama, so this choice would change
              nothing — disabled rather than quietly ignored.
            </span>
          )}
        </label>
        <label className="block text-[11px] text-content-subtle">
          Register
          <select value={captionVocab} onChange={(e) => setCaptionVocab(e.target.value)}
            disabled={live} aria-label="Caption vocabulary register"
            title="How captions name nude or sexual content. Explicit needs an uncensored (abliterated) Ollama vision model. Richer, more explicit captions also make the 🔍 search find more."
            className={`${captionSelectClass} sm:max-w-[16rem]`}>
            {VOCABULARY_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
        </label>
        <label className="block text-[11px] text-content-subtle">
          Length
          <select value={captionLength} onChange={(e) => setCaptionLength(e.target.value)}
            disabled={live} aria-label="Caption length"
            title="How much the captioner writes. Concise aims for one short sentence, Detailed for several - a target the model follows loosely, not a hard cap. Standard leaves the prompt untouched. Longer captions give the search more to match on."
            className={`${captionSelectClass} sm:max-w-[16rem]`}>
            {CAPTION_LENGTH_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
        </label>
      </div>
      {captionNsfw && (
        <div role="note" aria-label="What this run is about to caption"
          className={`space-y-1 rounded-md border px-2 py-1.5 ${
            captionNsfw.tone === 'warn'
              ? 'border-amber-400/40 bg-amber-500/10' : 'border-sky-400/40 bg-sky-500/10'}`}>
          <p className={`m-0 text-[11px] font-semibold leading-snug ${
            captionNsfw.tone === 'warn' ? 'text-amber-200' : 'text-sky-200'}`}>
            {captionNsfw.tone === 'warn' ? '⚠ ' : 'ℹ ' }{captionNsfw.heading}
          </p>
          {captionNsfw.paragraphs.map((line) => (
            <p key={line} className={`m-0 text-[11px] leading-snug ${
              captionNsfw.tone === 'warn' ? 'text-amber-100/90' : 'text-sky-100/90'}`}>
              {line}
            </p>
          ))}
        </div>
      )}
      <p className="m-0 text-[11px] leading-snug text-content-subtle">
        {captionScopeNote(selected.size, counts, captionScope)}
      </p>
    </div>
  )

  /* 🔄 THE DESTRUCTIVE TWIN, in the window's footer beside the normal launch —
     a SECOND launch button, not a pass of its own. It keeps everything it had:
     the number it rewrites, the number it spares, its amber warning and its
     window.confirm. And it keeps greying out WITH ITS REASON rather than
     disappearing: on a bank whose only captions are ones you wrote by hand, that
     disabled reason is the only place the protection is visible at all. */
  const captionSecondary = (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={startRecaption} disabled={!!recaptionInert}
          aria-label="Re-caption"
          title={recaptionInert || recaptionNote}
          className="rounded-md border border-amber-400/40 bg-amber-500/10 px-3 py-1.5 text-xs font-semibold text-amber-200 disabled:opacity-40">
          {captionRecaptionLabel(counts, captionScope, recaptionInert,
                                 captionIncludeAsserted)}
        </button>
        {includeAssertedLabel && (
          <label className="flex items-center gap-1 text-[11px] text-amber-400/90">
            <input type="checkbox" checked={captionIncludeAsserted} disabled={live}
              onChange={(e) => setCaptionIncludeAsserted(e.target.checked)}
              aria-label="Also re-caption the captions I wrote by hand"
              className="accent-amber-500" />
            {includeAssertedLabel}
          </label>
        )}
      </div>
      {recaptionInert && (
        <p className="m-0 text-[11px] leading-snug text-amber-300/90">{recaptionInert}</p>
      )}
      {recaptionNote && (
        <p className="m-0 text-[11px] leading-snug text-amber-400/90">{recaptionNote}</p>
      )}
    </div>
  )

  const scored = counts?.scored || 0
  // ⚖️ Can a balanced pick even run? Answered BEFORE the click when we already
  // know (semantic index missing; coverage says nothing is classified) — otherwise the
  // backend answers it with the exact pass and the numbers.
  const balanceReady = balanceReadiness({
    semanticReady, coverage, engineLabel: semanticState.label,
    prerequisite: semanticBlocked,
  })
  // What ✨ Score will really run on — the pass no longer holds the GPU when it
  // computes on the CPU, and the UI must say which of the two is happening.
  const scoreDevice = payload?.score_device
  const scoreNote = scoreDeviceNote(scoreDevice, Boolean(caps.bank_scoring))
  // Does this machine have an NVIDIA card AT ALL? Reported by the same
  // score_device probe, and only meaningful while the pass would run on the
  // CPU (a GPU pass answers gpu:true and stops looking). Undefined until the
  // payload lands — assume a card, so we never flash "no NVIDIA card" at
  // someone who has one.
  const scoreGpuPresent = scoreDevice ? scoreDevice.gpu || !!scoreDevice.gpu_present : true
  const watermarkScanned = counts?.watermark_scanned || 0
  // Score flags only make sense once their pass ran; watermark is its own pass.
  const availableScoreFlags = SCORE_REJECT_FLAGS.filter(
    (f) => (f === 'watermark' ? watermarkScanned : scored) > 0)
  // 🧹 Auto-reject readiness. The never-scanned pile is unreachable by EVERY
  // quality flag (they are all gated on quality_state=='ok'), so the popover
  // says so and names the gesture; the per-flag numbers below come from
  // flags_actionable, which is what the click actually flips.
  const autoRejectNotice = unscannedNotice(counts)
  const autoRejectPicked = pickedCandidates(rejectFlags, flagsActionable)
  const canPromote = (counts?.keep || 0) > 0 || selected.size > 0
  // Is any facet narrowing the grid? Drives the "N shown of TOTAL" readout AND
  // the chip counters, which is why it is one shared predicate: two lists of
  // facet keys drift, and the one here used to be missing 🔎 Origin, 🚫 exclude
  // and the 🏷️ tag chips — so a grid narrowed by those said it was showing
  // everything.
  const isFiltered = isFacetFiltered(filter)

  // The ONE recommended next step, from the counters the header strip already
  // reads. Advisory only — draws an amber "Next step" accent on that zone.
  const activeStep = nextBankStep({
    scanned: counts?.scanned || 0,
    scored: scored || 0,
    keep: counts?.keep || 0,
    scoringAvailable: !!caps?.bank_scoring,
  })
  const analyzeZone = BANK_ZONES.find((z) => z.id === 'analyze')
  const triageZone = BANK_ZONES.find((z) => z.id === 'triage')
  const curateZone = BANK_ZONES.find((z) => z.id === 'curate')
  const promoteZone = BANK_ZONES.find((z) => z.id === 'promote')

  return (
    <div className="space-y-4">
      <header className="space-y-2 rounded-xl border border-border bg-surface px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={onBack}
            className="rounded-md border border-border px-2 py-1 text-xs text-content-muted hover:text-content hover:bg-surface-raised">
            ← Banks
          </button>
          <h1 className="text-lg font-bold text-content">🗃️ {payload?.name || `Bank #${bankId}`}</h1>
        </div>
        {payload?.source_path && (
          <div className="flex min-w-0 items-center gap-2">
            <p className="min-w-0 grow truncate font-mono text-xs text-content-subtle"
              title={payload.source_path}>
              {payload.source_path}
            </p>
            <button type="button" onClick={openSourceFolder}
              disabled={openingSourceFolder} aria-busy={openingSourceFolder}
              title="Open this Bank's source folder in the system file explorer."
              className="shrink-0 rounded border border-border px-2 py-0.5 text-xs text-content-muted hover:bg-surface-raised hover:text-content disabled:cursor-wait disabled:opacity-60">
              {openingSourceFolder ? 'Opening…' : '📂 Open folder'}
            </button>
            {/* Cold path. The folder-sync note below offers this too, but only once
                the folder is already gone — and the real move is PLANNED: you look
                for the option before you drag 30 000 files to another drive, not
                after breaking the bank to discover it could have been repaired. */}
            <button type="button" onClick={() => setRelocating(true)}
              title="Moving this folder to another disk? Point the bank at its new location."
              className="shrink-0 rounded border border-border px-2 py-0.5 text-xs text-content-muted hover:bg-surface-raised hover:text-content">
              📦 Move folder…
            </button>
          </div>
        )}
        {/* A bank created before this was refused can still point at a dataset's
            own image folder. Nothing is repaired behind the user's back — the
            bank stays fully readable — but the fact is stated every time it is
            opened, because the click that hurts (🗑 Delete rejected) is right
            here on this page. */}
        {payload?.dataset_conflict && (
          <div role="alert"
            className="rounded-md border border-rose-500/70 bg-rose-500/15 p-3 text-sm text-rose-100 space-y-1">
            <p className="font-semibold">⛔ This bank sits on a dataset’s image folder</p>
            <p className="text-rose-100/90">{payload.dataset_conflict.message}</p>
            <p className="text-rose-100/90">
              🗑 Delete rejected is disabled here. Use 📦 Move folder… to point this
              bank at a folder of its own, or remove the bank — removing a bank never
              touches files.
            </p>
          </div>
        )}
        <FolderSyncNote sync={payload?.folder_sync}
          onRelocate={() => setRelocating(true)} />
        {counts && (
          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-t border-border pt-2 text-sm">
            <Stat label="images" value={counts.total} />
            <Stat label="scanned" value={counts.scanned} />
            {scored > 0 && <Stat label="scored" value={scored} />}
            {semanticState.hasStatus && (
              <Stat label={`${semanticState.label} semantic-ready`}
                value={semanticState.total > 0
                  ? `${semanticIndexed.toLocaleString()}/${semanticState.total.toLocaleString()}`
                  : semanticIndexed}
                tone={semanticReady ? 'emerald' : undefined} />
            )}
            {watermarkScanned > 0 && <Stat label="watermark-checked" value={watermarkScanned} />}
            <Stat label="undecided" value={counts.pending} />
            <Stat label="kept" value={counts.keep} tone="emerald" />
            <Stat label="rejected" value={counts.reject} tone="rose" />
            <Stat label="promoted" value={counts.promoted} tone="indigo" />
          </div>
        )}
      </header>

      <ProgressBar activity={payload?.activity} onCancel={cancelJob} offline={!connection.online} />

      <UndoBar offer={undoBar} busy={undoBusy} onUndo={undoLast}
        onDismiss={() => setUndoDismissedAt(undoBar?.at || 0)} />

      {!live && payload?.pipeline_report
        && payload.pipeline_report.finished_at !== dismissedReportAt && (
        <PipelineReport report={payload.pipeline_report}
          onDismiss={() => setDismissedReportAt(payload.pipeline_report.finished_at)} />
      )}

      {/* At xl the workflow uses the available width: the action-heavy Analyze
          zone and the read-only, bank-wide overview sit side by side. Below xl
          this remains the same simple vertical flow. */}
      <div className="grid gap-4 xl:grid-cols-12 xl:items-start">
      <div className="min-w-0 xl:col-span-7">
      {/* ① Analyze — run the analysis passes (or 🚀 Launch all) on the dump.
          Grouping + accent only; every pass keeps its own endpoint/behaviour. */}
      <ZoneSection zone={analyzeZone} accented={activeStep === 'analyze'}>
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => setLaunchOpen(true)} disabled={live || !(counts?.total > 0)}
          title={`Run the whole triage in one go — scan, auto-reject, Score${semanticState.engine === 'siglip2' ? ', SigLIP 2 semantic index' : ''}, crops/variants, watermarks, group by person and (optionally) caption. Start it and walk away. If the person pass is in, it checks your folders first and asks once, before the run.`}
          className="rounded-md bg-gradient-primary px-4 py-2 text-sm font-bold text-white shadow disabled:opacity-50">
          🚀 Launch all…
        </button>
        <p className="hidden text-xs text-content-subtle md:block">
          One-click funnel — or step through the passes below.
        </p>
      </div>

      <BankSemanticEngine state={semanticState} capsLoading={capsLoading}
        switching={semanticSwitching} disabled={semanticOperationBusy} live={live}
        gpuPresent={scoreGpuPresent}
        onPickPython={() => setPythonPickerFor('semantic')}
        onChange={changeSemanticEngine}
        onIndex={() => {
          // The visible “Reindex” promise and the request body agree on first
          // open; incomplete indexes resume with rescan:false.
          setPassRedoFor('semantic_index', semanticState.complete)
          setPassOpen('semantic_index')
        }} />

      {/* Analysis passes — individual, quieter than the primary actions. */}
      <div className="space-y-1.5">
        <GroupLabel>Analysis passes</GroupLabel>
        <div className="flex flex-wrap items-center gap-1.5">
          {/* Every button OPENS ITS WINDOW instead of firing. The window is where
              the scope, the counts, the settings the calculation really reads and
              the things it does NOT decide finally have room to be said — and
              where the launch button quotes the exact number it will move.

              “Rescan all” is gone as a button on purpose: it was never a second
              pass, it was a SCOPE wearing a button's clothes. It is now the last
              line of 🔎 Scan's THIS RUN block, next to the piles it belongs with.
              “Rescore all” went the same way, into ✨ Score's window. */}
          <PassButton onClick={() => setPassOpen('scan')} disabled={live}
            title="Measure sharpness, noise, flatness, size and detail, hash every image and group the exact duplicates — CPU only. Opens the launch window.">
            🔎 Scan quality…
          </PassButton>
          <PassButton onClick={() => setPassOpen('faces')} disabled={live || !caps.face_scoring}
            title={caps.face_scoring
              ? 'Detect the dominant face of every non-rejected image and cluster the bank by person (no reference needed). CPU, can take a while on thousands of images. It samples your subfolders first and offers the ones that look like a single person, so you can skip them.'
              : 'Install the Quality tools (Setup) to sort by person'}>
            👥 Group by person…
          </PassButton>
          <PassButton onClick={() => setPassOpen('score')} disabled={live || !caps.bank_scoring}
            title={caps.bank_scoring
              ? `Rate every non-rejected image for aesthetics (1–10), flag NSFW, and group by visual style — one CLIP pass. Powers a smarter "keep best". Already-scored images are reused, so stopping and relaunching costs only what is left. Runs in the background${
                holdsTheGpu(scoreDevice) ? ', and holds the GPU (ComfyUI is unloaded and training cannot start) for its duration' : ' on the CPU, leaving the GPU free'}.`
              : 'Install the Bank scoring extra (Setup ▸ Quality tools) to score aesthetics / NSFW / style'}>
            ✨ Score…{!caps.bank_scoring && ' (needs setup)'}
          </PassButton>
          <PassButton onClick={() => setPassOpen('medium')} disabled={live || !caps.bank_scoring}
            title={caps.bank_scoring
              ? 'Sort every scored image into photograph / anime / 3D render / illustration — read off the CLIP embeddings ✨ Score already computed, so no image is looked at again and the GPU stays free. It answers “unsure” rather than guessing: measured on a real 23 500-image bank, it named 2 anime drawings and no wrong verdict.'
              : 'Install the Bank scoring extra (Setup ▸ Quality tools) — 🎨 Medium reads the embeddings ✨ Score produces'}>
            🎨 Classify medium…{!caps.bank_scoring && ' (needs setup)'}
          </PassButton>
          <PassButton onClick={() => setPassOpen('framing')} disabled={live || !visionReady}
            title={visionReady
              ? 'Classify every non-rejected image by shot type — face close-up, bust, full body, back view — with the same Qwen3-VL classifier the datasets use. Powers the 📐 Framing filter and the coverage advice. GPU vision pass.'
              : 'Pull the vision model (Settings ▸ Local tools) to classify framing'}>
            📐 Classify framing…{!visionReady && ' (needs setup)'}
          </PassButton>
          <PassButton onClick={() => setPassOpen('semantic_dedup')} disabled={live || !semanticReady}
            title={semanticReady
              ? `Group crops and re-compressed variants of the SAME shot the exact-duplicate hash misses from the ${semanticState.label} semantic index. Review them under the ✂ Same shot chip.`
              : semanticBlocked}>
            ✂ Find crops &amp; variants…{!semanticReady
              && ` (needs ${semanticState.engine === 'clip' ? 'Score' : 'SigLIP 2 index'})`}
          </PassButton>
          {/* The label QUOTES THE NUMBER IT WILL MOVE — the scope's uncaptioned rows,
              or the selection when there is one. "Caption all" was the older, vaguer
              promise, and a button that announces one figure and acts on another is the
              misunderstanding this whole row exists to end.

              IT NO LONGER GREYS OUT AT ZERO. It used to, and it took the engine, the
              model, the register and the length pickers down with it — on exactly the
              bank whose captions you wanted redone with a better model. The window is
              always reachable; the LAUNCH button inside it is what refuses a run of 0,
              and it says why. */}
          <PassButton onClick={() => setPassOpen('caption')} disabled={live}
            title={selected.size
              ? `Caption the ${selected.size} selected image(s). Opens the window with the engine, model, register, length and scope.`
              : `${captionScopeNote(selected.size, counts, captionScope)} Opens the window with the engine, model, register, length and scope.`}>
            {captionButtonLabel(selected.size, counts, captionScope)}…
          </PassButton>
          {/* ⤢ the opt-in angle backfill. Its own button and its own window, never
              folded into 👥: it is hours of work on a big bank and nobody must pay
              it by accident. Shown only when there is something to measure. */}
          {(counts?.angle_backfillable || 0) > 0 && (
            <PassButton onClick={() => setPassOpen('angles')} disabled={live}
              title="Measure the head angle of the images a previous build face-scanned without keeping it. Writes the angle and nothing else.">
              ⤢ Measure head angles…
            </PassButton>
          )}
        </div>
        {/* Watermark CLEANING — the two manual levels (crop, then inpaint), with
            their own per-level progress. Lives in its own component so the
            "which level can run, and why not" logic stays unit-tested. */}
        <BankWatermarkPanel bankId={bankId} live={live}
          onFind={() => setPassOpen('watermark')}
          payload={payload} selectedIds={[...selected]}
          onChanged={async () => { await refreshPayload(); await refreshImages() }} />
        {scoreNote && (
          <p className={`text-xs ${scoreNote.tone === 'warn'
            ? 'text-amber-400/90' : 'text-content-subtle'}`}>
            {scoreNote.text}
          </p>
        )}
        {!capsLoading && !caps.bank_scoring && (
          <p className="text-xs text-content-muted">
            ✨ Score needs its own packages (Setup ▸ Quality tools) — or an interpreter
            that already has them{scoreGpuPresent
              ? '. If you train LoRAs or run ComfyUI, this machine probably has one.'
              : ', which saves installing them twice.'}
          </p>
        )}
        {/* The interpreter picker, offered where it can actually help: the pass is
            about to crawl on the CPU of a machine that HAS a card, or the scoring
            packages are missing and another Python here may already carry them
            (true with or without a card — it saves an install either way). The
            LABEL adapts: a machine with no NVIDIA card must never be promised "a
            GPU Python", and the note it gets alongside is "this is how it is",
            not a fix to chase. */}
        {!capsLoading && (scoreNote?.tone === 'warn' || !caps.bank_scoring) && (
          <div>
            <button type="button" onClick={() => setPythonPickerFor('scoring')}
              className={`rounded-md border px-2 py-1 text-xs font-medium ${scoreGpuPresent
                ? 'border-amber-400/50 text-amber-300 hover:bg-amber-500/10'
                : 'border-border text-content-muted hover:bg-surface-raised hover:text-content'}`}>
              {openerLabel(scoreGpuPresent)}
            </button>
          </div>
        )}
        {captionVocab === 'explicit' && !visionModelLooksUncensored && (
          /* The old wording sent people to Settings ▸ Captioning & quality, which holds
             the ENGINE selector — the vision model lives in Settings ▸ Local tools. Now
             that the model is pickable right here, the sentence points at the control
             on screen instead, and only mentions Settings for the case the picker
             cannot solve: no uncensored model pulled on this machine at all. */
          <p className="text-xs text-amber-400/90">
            ⚠️ Explicit captions need an uncensored (abliterated) Ollama vision model
            {visionModel ? ` — “${visionModel}” may refuse or soften explicit terms` : ''}.
            Pick another one in <b>Caption vision model</b> above, or pull one from{' '}
            <SettingsLink section="local-tools" focus="ollama-vision-model" tone="warning">
              Settings ▸ Local tools
            </SettingsLink>. Richer captions also feed the 🔍 search.
          </p>
        )}
      </div>
      </ZoneSection>
      </div>
      <div className="min-w-0 xl:col-span-5 xl:sticky xl:top-20">
        <BankOverview payload={payload} />
      </div>
      </div>

      {/* ② Triage — browse by facet/cluster and Keep/Reject to decide what stays.
          Stays fully visible; density was never the complaint. */}
      <ZoneSection zone={triageZone} accented={activeStep === 'triage'}>

      {/* Say it in words; the app sets its own chips and the counters below —
          measured, not the model — say what that lands on. Placed at the TOP of
          triage because it is a shortcut to the controls underneath, not a
          separate way of selecting. */}
      <DescribeFilterBar bankId={bankId} onApply={setF} />

      {/* Person clusters (after the face pass) */}
      {clusters.length > 0 && (
        <div className="space-y-1">
          <GroupLabel>
            People ({clusters.length} cluster{clusters.length > 1 ? 's' : ''} — biggest first)
          </GroupLabel>
          <ul className="flex gap-2 overflow-x-auto pb-1">
            {clusters.map((c) => (
              <li key={c.id} className="shrink-0">
                <button type="button" onClick={() => setF({ cluster: filter.cluster === c.id ? null : c.id, flag: null })}
                  title={`Show person #${c.id} (${c.size} image(s))`}
                  className={`relative block overflow-hidden rounded-lg border ${filter.cluster === c.id
                    ? 'border-indigo-400 ring-2 ring-indigo-400' : 'border-border'}`}>
                  {c.cover_image_id != null && (
                    <img src={`/api/bank/${bankId}/thumb/${c.cover_image_id}`} alt={`Person ${c.id}`}
                      loading="lazy" className="h-16 w-16 object-cover" />
                  )}
                  <span className="absolute bottom-0 inset-x-0 bg-black/60 text-center text-[10px] font-semibold text-white">
                    #{c.id} · {c.size}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Style clusters (after the scoring pass) — group screenshots/memes vs photoreal */}
      {styleClusters.length > 0 && (
        <div className="space-y-1">
          <GroupLabel>
            Styles ({styleClusters.length} group{styleClusters.length > 1 ? 's' : ''} — biggest first)
          </GroupLabel>
          <ul className="flex gap-2 overflow-x-auto pb-1">
            {styleClusters.map((c) => (
              <li key={c.id} className="shrink-0">
                <button type="button" onClick={() => setF({ style: filter.style === c.id ? null : c.id, flag: null, cluster: null })}
                  title={`Show style group #${c.id} (${c.size} image(s))`}
                  className={`relative block overflow-hidden rounded-lg border ${filter.style === c.id
                    ? 'border-fuchsia-400 ring-2 ring-fuchsia-400' : 'border-border'}`}>
                  {c.cover_image_id != null && (
                    <img src={`/api/bank/${bankId}/thumb/${c.cover_image_id}`} alt={`Style ${c.id}`}
                      loading="lazy" className="h-16 w-16 object-cover" />
                  )}
                  <span className="absolute bottom-0 inset-x-0 bg-black/60 text-center text-[10px] font-semibold text-white">
                    🎨{c.id} · {c.size}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Search, subfolder scoping, and the grouped flag filters. */}
      <div className="space-y-2.5 rounded-lg border border-border bg-surface px-3 py-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-[12rem] max-w-md flex-1">
            <span aria-hidden className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-content-subtle">🔍</span>
            <input type="search" value={searchText} onChange={(e) => setSearchText(e.target.value)}
              placeholder="Search captions and file names… (e.g. red dress)"
              aria-label="Search the bank by caption or file name"
              className="w-full rounded-md border border-border bg-surface py-1.5 pl-8 pr-8 text-sm text-content placeholder:text-content-subtle" />
            {searchText && (
              <button type="button" onClick={() => setSearchText('')} aria-label="Clear search"
                className="absolute right-2 top-1/2 -translate-y-1/2 text-content-subtle hover:text-content">✕</button>
            )}
          </div>
          {/* 🚫 Exclude — the search bar read backwards. Captioning a big bank
              turns it into a checklist ("which ones have I not tagged with X
              yet?"), and that question has no answer while the only text tool
              can just narrow TO a word. Same fields as the search (caption +
              file name), same debounce, composes with every other facet.
              Comma-separated: hiding 'logo, watermark' in one pass is the
              normal case. Its own min-width so the pair wraps to two rows —
              not two half-unusable boxes — inside a 400 px toolbar. */}
          <div className="relative min-w-[12rem] max-w-md flex-1">
            <span aria-hidden className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-content-subtle">🚫</span>
            <input type="search" value={excludeText} onChange={(e) => setExcludeText(e.target.value)}
              placeholder="Exclude words… (e.g. logo, watermark)"
              aria-label="Hide images whose caption or file name contains these words"
              title="Hides every image whose caption or file name contains one of these words (comma-separated). Matches anywhere in the text, so 'car' also hides 'scarf'. Images with no caption are never hidden."
              className="w-full rounded-md border border-border bg-surface py-1.5 pl-8 pr-8 text-sm text-content placeholder:text-content-subtle" />
            {excludeText && (
              <button type="button" onClick={() => setExcludeText('')} aria-label="Clear the exclude filter"
                className="absolute right-2 top-1/2 -translate-y-1/2 text-content-subtle hover:text-content">✕</button>
            )}
          </div>
          {/* Subfolder scoping (a Telegram export nests one folder per chat/date).
              Scoping to ONE folder is also the moment the user knows whose it is,
              so the 👤 "Single person here" assertion lives right under it — and
              only then, which is why the group takes the full row only when a
              folder is actually scoped (otherwise it stays inline as before). */}
          {subfolders.length > 1 && (
            <div className={`flex flex-col gap-1.5 ${filter.subfolder != null ? 'w-full' : ''}`}>
              <div className="flex items-center gap-1.5">
                <GroupLabel>Subfolder</GroupLabel>
                <select value={filter.subfolder ?? '__all__'}
                  onChange={(e) => setF({ subfolder: e.target.value === '__all__' ? null : e.target.value })}
                  className="rounded-md border border-border bg-surface px-2 py-1 text-xs text-content">
                  <option value="__all__">All subfolders</option>
                  {subfolders.map((s) => (
                    <option key={s.name || '__root__'} value={s.name}>
                      {s.name === '' ? '(bank root)' : s.name} · {s.count}
                      {/* '👤?' = the app sampled this folder and it looked like
                          one person. A hint that something is worth opening,
                          never a claim — the panel below says the numbers. */}
                      {folderMarker(folderPersonInfo?.suggestions, s.name)}
                    </option>
                  ))}
                </select>
              </div>
              <SubfolderPersonPanel
                subfolder={filter.subfolder}
                entry={assertionFor(folderPersons, filter.subfolder)}
                suggestion={suggestionFor(folderPersonInfo?.suggestions, filter.subfolder)}
                offer={scanOffer(folderPersonInfo)}
                busy={folderPersonBusy}
                onAssert={assertFolderPerson}
                onRevoke={revokeFolderPerson}
                onCheck={checkFolderPerson}
                onScan={scanFolderPersons} />
            </div>
          )}
        </div>

        {/* Phones keep the established placement inside the filter zone. The
            desktop copy below is display:none here, so only one inspector is
            visible (and reachable) at any viewport size. */}
        {tagRow && (
          <div className="xl:hidden">
            <SelectionTagsPanel tagRow={tagRow} tagPicked={tagPicked}
              onToggle={(tag) => toggleTag(tag, tagRow)} onClear={clearTags} />
          </div>
        )}

        {/* Filters — grouped by facet so the chips read as a system, not a wall.
            While anything is filtered the numbers describe the FILTERED bank, and
            the row says so: a count that silently changed meaning would be worse
            than the bank-wide one it replaces. */}
        {chipsFiltered && (
          <p className="m-0 text-[11px] leading-snug text-content-subtle">
            Counts below follow the active filters. Each chip is counted with the
            others applied and its own value lifted, so you can always switch to
            a neighbour.
          </p>
        )}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <FilterGroup label="Status">
            <Chip active={!filter.status && !filter.flag && filter.cluster == null && filter.style == null}
              onClick={() => setF({ status: null, flag: null, cluster: null, style: null })}>All</Chip>
            <Chip active={filter.status === 'pending'} onClick={() => setF({ status: filter.status === 'pending' ? null : 'pending' })}>Undecided</Chip>
            <Chip active={filter.status === 'keep'} onClick={() => setF({ status: filter.status === 'keep' ? null : 'keep' })}>✓ Kept</Chip>
            <Chip active={filter.status === 'reject'} onClick={() => setF({ status: filter.status === 'reject' ? null : 'reject' })}>✕ Rejected</Chip>
          </FilterGroup>

          <FilterGroup label="Quality">
            {['blur', 'noise', 'uniform', 'small', 'soft_detail', 'bars', 'unreadable'].map((f) => (
              <Chip key={f} active={filter.flag === f}
                onClick={() => setF({ flag: filter.flag === f ? null : f })}
                title={FLAG_HINT[f] || 'Sorted worst-first'}>
                {FLAG_LABEL[f]} {flags[f] ?? 0}
              </Chip>
            ))}
            <Chip active={filter.flag === 'clean'} onClick={() => setF({ flag: filter.flag === 'clean' ? null : 'clean' })}>✨ Clean</Chip>
          </FilterGroup>

          {/* Score-derived flags — only surfaced once their pass has produced data.
              These used to clear the person/style cluster on click, and the ≈
              Duplicates chips cleared the cluster too. That silently undid a
              filter the user had set — and now that each chip PRINTS the size of
              the page it opens, it would also have made that number wrong.
              A chip toggles its own facet and nothing else. */}
          {availableScoreFlags.length > 0 && (
            <FilterGroup label="Score">
              {availableScoreFlags.map((f) => (
                <Chip key={f} active={filter.flag === f}
                  onClick={() => setF({ flag: filter.flag === f ? null : f })}
                  title={f === 'watermark' ? 'Overlaid watermark detected' : 'Sorted worst-first'}>
                  {FLAG_LABEL[f]} {flags[f] ?? 0}
                </Chip>
              ))}
            </FilterGroup>
          )}

          <FilterGroup label="Groups">
            <Chip active={filter.flag === 'dups'} onClick={() => setF({ flag: filter.flag === 'dups' ? null : 'dups' })}
              title="Exact / resized duplicate groups (perceptual hash) with their resolution panel">
              ≈ Duplicates {payload?.dup?.unresolved ?? 0}
            </Chip>
            {(payload?.semantic_dup?.groups ?? 0) > 0 && (
              <Chip active={filter.flag === 'semantic_dups'}
                onClick={() => setF({ flag: filter.flag === 'semantic_dups' ? null : 'semantic_dups' })}
                title="Semantic near-duplicates — same shot, different crop/compression — with their resolution panel">
                ✂ Same shot {payload?.semantic_dup?.unresolved ?? 0}
              </Chip>
            )}
            {payload?.faces_scanned > 0 && (
              <Chip active={filter.flag === 'no_face'} onClick={() => setF({ flag: filter.flag === 'no_face' ? null : 'no_face' })}>
                🚫👤 No face
              </Chip>
            )}
          </FilterGroup>

          {/* Resolution tiers — one active at a time; re-click clears.
              Composes with every filter and with the Sort menu below. */}
          {shownResBuckets.length > 0 && (
            <FilterGroup label="📐 Resolution">
              {shownResBuckets.map((b) => (
                <Chip key={b.id} active={filter.resBucket === b.id}
                  onClick={() => setF({ resBucket: filter.resBucket === b.id ? null : b.id })}
                  title="Filter the grid to this resolution tier (megapixels = width×height)">
                  {b.label} {resBuckets[b.id] ?? 0}
                </Chip>
              ))}
            </FilterGroup>
          )}

          {/* Origin — ai / camera / unknown, read off the file's own metadata by
              the quality scan. 'Unknown' is the usual answer and is deliberately
              shown next to the other two: silence is not evidence of anything. */}
          {originMeasured > 0 && (
            <FilterGroup label="🔎 Origin">
              {ORIGIN_BUCKETS.map((b) => (
                <Chip key={b.id} active={filter.origin === b.id}
                  onClick={() => setF({ origin: filter.origin === b.id ? null : b.id })}
                  title={b.id === 'unknown'
                    ? 'No metadata left in the file — the normal case for anything '
                      + 'scraped or sent through a chat app. NOT evidence that it is '
                      + 'a real photo.'
                    : (b.id === 'ai'
                      ? 'The file still carries generation metadata (ComfyUI workflow, '
                        + 'A1111 parameters, or a C2PA "generated" marker).'
                      : 'The file still carries camera EXIF (make, model or exposure).')}>
                  {b.label} {originCounts[b.id] ?? 0}
                </Chip>
              ))}
            </FilterGroup>
          )}

          {/* Framing tiers — face/bust/body/back (+ unknown), from the 📐 Framing
              pass. One active at a time; re-click clears. Composes with everything. */}
          {shownFramings.length > 0 && (
            <FilterGroup label="📐 Framing">
              {shownFramings.map((b) => (
                <Chip key={b.id} active={filter.framing === b.id}
                  onClick={() => setF({ framing: filter.framing === b.id ? null : b.id })}
                  title="Filter the grid to this shot type (from the Framing pass)">
                  {b.label} {framingCounts[b.id] ?? 0}
                </Chip>
              ))}
            </FilterGroup>
          )}

          {/* 🎨 Medium — what the picture is MADE of, from the ✨ Score
              embeddings. A DIFFERENT question from 🔎 Origin above, which reads
              the file's metadata: a photorealistic AI render is 🤖 AI and 📷
              Photo at once. The limits sentence is part of the row, not a
              tooltip, because "unsure" being the second-biggest pile is the
              single most important thing to know about this measurement. */}
          {shownMediums.length > 0 && (
            <div className="flex flex-col gap-1">
              <FilterGroup label="🎨 Medium">
                {shownMediums.map((b) => (
                  <Chip key={b.id} active={filter.medium === b.id}
                    onClick={() => setF({ medium: filter.medium === b.id ? null : b.id })}
                    title={mediumTitle(b.id)}>
                    {b.label} {mediumCounts[b.id] ?? 0}
                  </Chip>
                ))}
              </FilterGroup>
              {mediumNote && (
                <p className="m-0 pl-1 text-[11px] leading-snug text-content-subtle">{mediumNote}</p>
              )}
            </div>
          )}

          {/* ⤢ Angle — measured in the pixels by the 🎭 Faces pass. The backfill
              offer lives HERE rather than with the other passes: it only makes
              sense next to the counts that explain why it exists, and it must be
              priced before it is clicked. */}
          {(shownAngles.length > 0 || angleState.offer) && (
            <div className="flex flex-col gap-1">
              <FilterGroup label="⤢ Angle">
                {shownAngles.map((b) => (
                  <Chip key={b.id} active={filter.angle === b.id}
                    onClick={() => setF({ angle: filter.angle === b.id ? null : b.id })}
                    title={angleTitle(b.id)}>
                    {b.label} {angleCounts[b.id] ?? 0}
                  </Chip>
                ))}
              </FilterGroup>
              {angleState.note && (
                <p className="m-0 pl-1 text-[11px] leading-snug text-content-subtle">{angleState.note}</p>
              )}
              {angleState.offer && (
                <p className="m-0 flex flex-wrap items-center gap-2 pl-1 text-[11px] leading-snug text-content-subtle">
                  <span>{angleState.offer.why}</span>
                  <button type="button" onClick={() => setPassOpen('angles')} disabled={!!live}
                    title={angleState.offer.why}
                    className="rounded-md border border-border bg-surface-raised px-2 py-0.5 text-[11px] text-content transition-colors hover:bg-surface disabled:opacity-50">
                    ⤢ {angleState.offer.label}
                  </button>
                </p>
              )}
            </div>
          )}
        </div>

        {/* 🎚 The numbers BEHIND those chips. Folded by default — the chips are
            the everyday gesture and the thresholds are the occasional one — but
            present, because tuning them used to mean leaving the bank for
            Settings and coming back blind. Same config keys, same save. */}
        <div className="border-t border-border pt-2">
          <button type="button" onClick={() => setThresholdsOpen((v) => !v)}
            aria-expanded={thresholdsOpen} aria-controls="bank-thresholds-panel"
            className="flex w-full items-center gap-1.5 text-left text-xs text-content-muted hover:text-content">
            <span aria-hidden>🎚</span>
            <span className="font-medium">Filter thresholds</span>
            {/* The gloss is the first thing to go on a phone: the label already
                says what it opens, and a wrapped subtitle costs two lines. */}
            <span className="hidden text-content-subtle sm:inline">
              — what the chips above count as blurry, small, duplicate…
            </span>
            <span aria-hidden className="ml-auto text-content-subtle">{thresholdsOpen ? '▲' : '▼'}</span>
          </button>
          {thresholdsOpen && (
            <div id="bank-thresholds-panel" className="mt-2">
              {/* The panel's ↻ re-run buttons start the SAME one-job-per-bank
                  passes as the toolbar above, so they need the same two facts
                  the progress bar reads: whether a job holds the bank (to refuse
                  the click before it is made, saying which pass and where it is)
                  and the duplicate counts (to report what the pass produced).
                  Both are already in this payload — no extra poll, no second
                  progress mechanism. */}
              <BankThresholdsPanel bankId={bankId}
                activity={payload?.activity} offline={!connection.online}
                dupSummary={payload?.dup} semanticDupSummary={payload?.semantic_dup}
                onSaved={() => { refreshPayload(); refreshImages() }}
                onRunPass={(endpoint, body) => act(
                  () => postJson(`/api/bank/${bankId}/${endpoint}`, body || {}),
                  null)} />
            </div>
          )}
        </div>

        {/* View controls — order and tile size, off to the right on their own line */}
        <div className="flex flex-wrap items-center gap-2 border-t border-border pt-2">
          <GroupLabel>View</GroupLabel>
          <label className="flex items-center gap-1 text-xs text-content-muted">
            Sort
            {/* Order the grid on ANY quantity the passes measured — resolution,
                file size, aesthetic rating, NSFW likelihood, sharpness, noise,
                contrast, detail, bars, JPEG quality, face confidence — each way,
                so a review opens on what it is looking for. Grouped by the pass
                that produces them: eleven measures is a long flat list, and the
                grouping doubles as "run THIS pass to unlock these".
                Images the matching pass never reached sink to the end (never the
                top), and an entry whose pass has produced nothing yet is greyed
                out saying which pass to run. The value rides to the server, which
                sorts in SQL: it applies to the WHOLE filter, not this page, so
                "Select all in filter" and ▶ Review walk the same order — and it
                is remembered per bank, so a dump you review by sharpness opens
                that way tomorrow. max-w keeps the control inside a 400 px toolbar. */}
            <select value={filter.sort} onChange={(e) => setSort(e.target.value)}
              title="Order the grid by anything the passes measured — resolution, size, aesthetic, NSFW, sharpness, noise, contrast, detail, bars, JPEG quality, face confidence. Images a pass never reached sink to the end. Remembered for this bank."
              aria-label="Sort the grid"
              className="max-w-[11rem] rounded-md border border-border bg-surface px-2 py-0.5 text-xs text-content">
              {sortGroups.map((g) => (g.group ? (
                <optgroup key={g.group} label={g.group}>
                  {g.options.map((o) => (
                    <option key={o.id} value={o.id} disabled={o.disabled} title={o.title}>
                      {o.label}
                    </option>
                  ))}
                </optgroup>
              ) : g.options.map((o) => (
                <option key={o.id} value={o.id} disabled={o.disabled} title={o.title}>
                  {o.label}
                </option>
              ))))}
            </select>
          </label>
          <span className="ml-auto" />
          <button type="button" onClick={() => setTileSize((s) => (s === 'M' ? 'S' : 'M'))}
            className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">
            {tileSize === 'M' ? 'Small tiles' : 'Medium tiles'}
          </button>
        </div>
      </div>

      {/* Results readout + selection actions */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-content-muted">
          <span className="font-semibold tabular-nums text-content">{(page.total ?? 0).toLocaleString()}</span> shown
          {isFiltered && (
            <span className="text-content-subtle"> of {(counts?.total ?? 0).toLocaleString()}</span>
          )}
        </span>
        <span aria-hidden className="h-4 w-px bg-border" />
        {/* The fast lane: full-size, one image at a time, Keep/Reject/Skip with
            K/R/S. Deliberately a button of its own rather than a tile gesture —
            the tile click stays the bulk-selection gesture it has always been. */}
        <button type="button" onClick={() => openReview(null)} disabled={reviewLoading}
          title="Review the images of this filter one at a time, full size: ✓ Keep / ✕ Reject / ⏭ Skip (K/R/S) each move to the next. Optional random order."
          className="rounded-md border border-indigo-400/60 bg-indigo-500/20 px-2.5 py-0.5 text-xs font-semibold text-indigo-200 disabled:opacity-50 hover:bg-indigo-500/30">
          {reviewLoading ? '▶ Preparing…' : '▶ Review one by one'}
        </button>
        <span aria-hidden className="h-4 w-px bg-border" />
        <span className="text-content-muted">{selected.size} selected</span>
        <button type="button" onClick={selectAllCurrent}
          className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content hover:bg-surface-raised">
          Select all in filter
        </button>
        {/* Bulk-reject undecided images by quality flag — a triage shortcut that
            leaves your manual ✓/✕ untouched and deletes nothing off disk. */}
        <div className="relative">
          <button type="button" onClick={() => setShowAutoReject((v) => !v)} disabled={live}
            aria-expanded={showAutoReject}
            title="Bulk-reject the still-undecided images carrying the chosen quality flags"
            className="rounded-md border border-border bg-surface-raised px-2 py-0.5 text-xs text-content disabled:opacity-50 hover:bg-surface">
            🧹 Auto-reject…
          </button>
          {showAutoReject && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setShowAutoReject(false)} aria-hidden />
              {/* Anchored under the button on a desktop, a bottom sheet on a
                  phone. It used to be `absolute left-0` at every width: the
                  button sits mid-toolbar, so at 400 px a 288-px panel ran off
                  the right edge and clipped its own text — including, now, the
                  counts this panel exists to show. Capped height + internal
                  scroll so "Reject them" is reachable however long the caveats
                  get. */}
              <div className="fixed inset-x-3 bottom-3 z-50 max-h-[70vh] overflow-y-auto rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2 sm:absolute sm:inset-x-auto sm:bottom-auto sm:left-0 sm:mt-1 sm:w-72">
                <p className="text-xs text-content-muted">
                  Rejects the UNDECIDED images with these flags. Your manual ✓/✕ are never changed;
                  everything stays reversible (nothing is deleted from disk).
                </p>
                {/* The pile no flag can reach. Every quality flag is gated on a
                    completed scan, so a never-scanned image is invisible to all
                    of them — and "0 blurry" would otherwise read as good news
                    when it means nothing was ever measured. */}
                {autoRejectNotice && (
                  <p className="m-0 rounded-md border border-amber-400/40 bg-amber-500/10 px-2 py-1 text-[0.6875rem] leading-snug text-amber-200">
                    ⚠ {autoRejectNotice.text} {autoRejectNotice.action}
                    {autoRejectNotice.caveat ? ` ${autoRejectNotice.caveat}` : ''}
                  </p>
                )}
                {/* The caveat is printed, not left in a title= tooltip: soft_detail
                    and bars are provenance HINTS, not verdicts, and this is the one
                    screen that offers to act on them in bulk. A tooltip is invisible
                    on a phone and to anyone who does not hover. */}
                {[...QUALITY_REJECT_FLAGS, ...availableScoreFlags].map((f) => {
                  // A 0 has two opposite meanings and used to render identically:
                  // "nothing left to reject" (clean) vs "this flag's pass never
                  // ran, so it cannot catch anything" (a missing prerequisite).
                  const prereq = flagPrereq(f, counts, originMeasured)
                  return (
                    <label key={f} className="block text-sm text-content">
                      <span className="flex items-center gap-2">
                        <input type="checkbox" checked={rejectFlags.has(f)}
                          onChange={(e) => setRejectFlags((prev) => {
                            const next = new Set(prev)
                            if (e.target.checked) next.add(f); else next.delete(f)
                            return next
                          })} />
                        {FLAG_LABEL[f]}{' '}
                        <span className="text-content-subtle">({flagCandidateLabel(f, flagsActionable)})</span>
                      </span>
                      {prereq && (
                        <span className="mt-0.5 block pl-6 text-[0.6875rem] leading-snug text-sky-200/90">
                          ⓘ {prereq}
                        </span>
                      )}
                      {FLAG_HINT[f] && (
                        <span className="mt-0.5 block pl-6 text-[0.6875rem] leading-snug text-amber-200/80">
                          ⚠ {FLAG_HINT[f]}
                        </span>
                      )}
                    </label>
                  )
                })}
                {/* What the button is about to do, before it is pressed — the
                    whole point of the fix. Amber when the answer is zero, so
                    "nothing will happen" is visible rather than discovered. */}
                {autoRejectPicked && (
                  <p className={`m-0 text-[0.6875rem] leading-snug ${
                    autoRejectPicked.sum ? 'text-content-muted' : 'text-amber-200'}`}>
                    {autoRejectPicked.text}
                  </p>
                )}
                <button type="button" onClick={applyAutoReject} disabled={!rejectFlags.size}
                  className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-50">
                  Reject them
                </button>
              </div>
            </>
          )}
        </div>
        {(selected.size > 0 || showSelected) && (
          <button type="button" onClick={toggleSelectionView}
            aria-pressed={showSelected}
            title={showSelected
              ? 'Back to the full grid with its filters'
              : 'Show only the selected images (as their own view), so a scattered curation/similarity result is visible in one place'}
            className={`rounded-md border px-2 py-0.5 text-xs font-medium ${showSelected
              ? 'border-indigo-400/60 bg-indigo-500/20 text-indigo-200'
              : 'border-border text-content-muted hover:text-content hover:bg-surface-raised'}`}>
            {showSelected ? '↩ Show all' : `🔎 Show selected (${selected.size})`}
          </button>
        )}
        {selected.size > 0 && (
          <>
            <button type="button" onClick={() => { setSelected(new Set()); if (showSelected) { exitSelectionView(); setOffset(0); refreshImages(filter, 0, { on: false }) } }}
              className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">Clear</button>
            <button type="button" onClick={() => batchStatus([...selected], 'keep')}
              className="rounded-md border border-emerald-400/50 bg-emerald-500/10 px-2 py-0.5 text-xs font-semibold text-emerald-200 hover:bg-emerald-500/20">✓ Keep</button>
            <button type="button" onClick={() => batchStatus([...selected], 'reject')}
              className="rounded-md border border-rose-400/50 bg-rose-500/10 px-2 py-0.5 text-xs font-semibold text-rose-200 hover:bg-rose-500/20">✕ Reject</button>
            <button type="button" onClick={() => batchStatus([...selected], 'pending')}
              className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">↺ Undecided</button>
            {/* 🔄 Rotate the selection. Your own files are never rewritten: the
                angle is stored on the row and applied to what the app shows and
                to what it promotes — so four turns cost the original nothing. */}
            <button type="button" onClick={() => rotateSelection(-90)}
              aria-label={`Rotate the ${selected.size} selected image(s) 90 degrees left`}
              title="Rotate 90° left (counter-clockwise). Your own files are never modified — the turn is stored and applied to what you see and to what gets promoted."
              className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:bg-surface-raised hover:text-content">
              <span aria-hidden="true">↺</span> Rotate left
            </button>
            <button type="button" onClick={() => rotateSelection(90)}
              aria-label={`Rotate the ${selected.size} selected image(s) 90 degrees right`}
              title="Rotate 90° right (clockwise). Your own files are never modified — the turn is stored and applied to what you see and to what gets promoted."
              className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:bg-surface-raised hover:text-content">
              <span aria-hidden="true">↻</span> Rotate right
            </button>
          </>
        )}
      </div>
      </ZoneSection>

      {/* The two short finishing zones share a row on wide screens. Promote
          stays visible while a long Curate result is reviewed; phones retain
          the natural Curate → Promote stack. */}
      <div className="grid gap-4 xl:grid-cols-12 xl:items-start">
      <div className="min-w-0 xl:col-span-8">
      {/* ③ Curate — optional refinement (diverse/similar/coverage). Always
          accessible, but never the accented "next step". */}
      <ZoneSection zone={curateZone} accented={false}>

      {/* Curation — build a good LoRA subset out of a big dump from this Bank's
          selected semantic index. Diversity coverage + reference similarity, both
          producing a SELECTION the user reviews above. */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-xs font-semibold uppercase tracking-wide text-content-subtle">Curate</span>
        <div className="relative">
          <button type="button" disabled={live || !semanticReady || diverseBusy}
            onClick={() => setCurateOpen((v) => (v === 'diverse' ? null : 'diverse'))}
            aria-expanded={curateOpen === 'diverse'}
            title={semanticReady
              ? `Pick the N images that best COVER the visual variety of the current filter (varied angles/outfits/scenes) using the ${semanticState.label} semantic index.`
              : semanticBlocked}
            className="rounded-md border border-border bg-surface-raised px-2.5 py-0.5 text-xs text-content disabled:opacity-50 hover:bg-surface">
            🎨 Pick diverse…{!semanticReady && ` (needs ${semanticState.label})`}{diverseBusy && ' (sampling…)'}
          </button>
          {curateOpen === 'diverse' && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setCurateOpen(null)} aria-hidden />
              <div className="absolute z-50 mt-1 w-72 rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2">
                <p className="text-xs text-content-muted">
                  Selects the most <strong>varied</strong> images of the current filter — the best
                  coverage of the visual space, not N look-alikes. Reviews as a normal selection
                  (nothing is kept or deleted yet).
                </p>
                <label className="flex items-center gap-2 text-sm text-content">
                  How many
                  <input type="number" min={1} max={2000} value={diverseN}
                    onChange={(e) => setDiverseN(Math.max(1, Math.min(2000, Number(e.target.value) || 1)))}
                    className="w-20 rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content" />
                </label>
                {/* Typicality guard — "most diverse" used to mean "most isolated",
                    which is how a meme or a photo of someone else won the first
                    picks. Exposed rather than silently changed: 0 is the old
                    behaviour, to the pick. */}
                <label className="block space-y-1 text-sm text-content">
                  <span className="flex items-center justify-between gap-2">
                    <span>Skip the odd ones out</span>
                    <span className="tabular-nums text-xs text-content-muted">
                      {diverseTypicality === 0 ? 'off' : `${Math.round(diverseTypicality * 100)}%`}
                    </span>
                  </span>
                  <input type="range" min={0} max={1} step={0.05} value={diverseTypicality}
                    onChange={(e) => setDiverseTypicality(Number(e.target.value))}
                    className="w-full accent-primary" />
                </label>
                <p className="text-[11px] leading-snug text-content-muted">
                  {diverseTypicality === 0
                    ? 'Off — pure coverage, exactly like before. The most isolated images win the first picks, so memes, wrong-person shots and botched frames tend to come up first.'
                    : 'Images that look like nothing else in the bank (memes, screenshots, someone else) stop winning on isolation alone. Variety inside your subject is untouched.'}
                </p>
                <button type="button" onClick={pickDiverse} disabled={diverseBusy}
                  className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-60">
                  {diverseBusy ? 'Sampling…' : `Select ${diverseN} most diverse`}
                </button>
              </div>
            </>
          )}
        </div>
        {/* ⚖️ Balanced pick — a DIFFERENT question from 🎨 Pick diverse: not "is
            my set varied?" but "does it cover the framings I want to generate?".
            Kept as its own button rather than a mode of the other one, because a
            bank with no 📐 Framing pass can still use diversity. */}
        <div className="relative">
          <button type="button" disabled={live || balanceBusy || !balanceReady.ready}
            onClick={() => setCurateOpen((v) => (v === 'balanced' ? null : 'balanced'))}
            aria-expanded={curateOpen === 'balanced'}
            title={balanceReady.ready
              ? `Select N images SPREAD OVER the framings (face / bust / body / back), then diversify each bucket with the ${semanticState.label} semantic index.`
              : balanceReady.reason}
            className="rounded-md border border-border bg-surface-raised px-2.5 py-0.5 text-xs text-content disabled:opacity-50 hover:bg-surface">
            ⚖️ Balanced pick…{balanceBusy && ' (sampling…)'}
          </button>
          {curateOpen === 'balanced' && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setCurateOpen(null)} aria-hidden />
              {/* Bottom sheet below sm (measured at 400 px, an anchored w-80 panel
                  pushes the page sideways), normal popover from sm up. */}
              <div className="fixed inset-x-4 bottom-4 z-50 max-h-[75vh] overflow-y-auto rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2 sm:absolute sm:inset-x-auto sm:bottom-auto sm:left-0 sm:mt-1 sm:max-h-none sm:w-80 sm:overflow-visible">
                <p className="text-xs text-content-muted">
                  Splits your pick <strong>evenly across the framings</strong> — “20 face, 20 bust,
                  20 body” — and fills each bucket with the same most-varied sampling.
                  Nothing is kept or deleted; you get a selection to review.
                </p>
                <label className="flex items-center gap-2 text-sm text-content">
                  How many
                  <input type="number" min={1} max={2000} value={balanceN}
                    onChange={(e) => setBalanceN(Math.max(1, Math.min(2000, Number(e.target.value) || 1)))}
                    className="w-20 rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content" />
                </label>
                <fieldset className="space-y-1">
                  <legend className="text-sm text-content">Balance on</legend>
                  {BALANCE_AXES.map((a) => (
                    <label key={a.id} className="flex items-start gap-2 text-xs text-content-muted">
                      <input type="radio" name="bank-balance-axis" value={a.id}
                        checked={balanceAxis === a.id}
                        onChange={() => setBalanceAxis(a.id)}
                        className="mt-0.5 accent-primary" />
                      <span><span className="text-content">{a.label}</span> — {a.hint}</span>
                    </label>
                  ))}
                </fieldset>
                <p className="text-[11px] leading-snug text-content-muted">
                  Framing is the reliable axis on a one-subject bank: person groups there tend to be
                  few, sparse and arbitrary. It uses the same “Skip the odd ones out” setting as
                  🎨 Pick diverse ({diverseTypicality === 0 ? 'off' : `${Math.round(diverseTypicality * 100)}%`}).
                </p>
                <button type="button" onClick={pickBalanced} disabled={balanceBusy}
                  className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-60">
                  {balanceBusy ? 'Sampling…' : `Select ${balanceN}, balanced`}
                </button>
              </div>
            </>
          )}
        </div>
        <div className="relative">
          <button type="button"
            disabled={live || !semanticReady || selected.size !== 1 || similarBusy}
            onClick={() => setCurateOpen((v) => (v === 'similar' ? null : 'similar'))}
            aria-expanded={curateOpen === 'similar'}
            title={!semanticReady
              ? semanticBlocked
              : selected.size === 1
                ? `Rank the current filter against the ONE selected image with the ${semanticState.label} semantic index and select the closest N.`
                : 'Select exactly one image to use as the reference'}
            className="rounded-md border border-border bg-surface-raised px-2.5 py-0.5 text-xs text-content disabled:opacity-50 hover:bg-surface">
            🎯 Similar to selected…{similarBusy && ' (ranking…)'}
          </button>
          {curateOpen === 'similar' && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setCurateOpen(null)} aria-hidden />
              <div className="absolute z-50 mt-1 w-72 rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2">
                <p className="text-xs text-content-muted">
                  Ranks the current filter by {semanticState.label} similarity to your one selected image and selects
                  the closest — a fast way to extract one person or look. The reference is kept in
                  the selection.
                </p>
                <label className="flex items-center gap-2 text-sm text-content">
                  How many
                  <input type="number" min={1} max={2000} value={similarN}
                    onChange={(e) => setSimilarN(Math.max(1, Math.min(2000, Number(e.target.value) || 1)))}
                    className="w-20 rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content" />
                </label>
                <button type="button" onClick={findSimilar} disabled={similarBusy}
                  className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-60">
                  {similarBusy ? 'Ranking…' : `Select ${similarN} most similar`}
                </button>
              </div>
            </>
          )}
        </div>
        <div className="relative">
          <button type="button" disabled={live || !semanticReady}
            onClick={openTextSearch}
            aria-expanded={curateOpen === 'text'}
            title={semanticReady
              ? `Describe what you are looking for in words ("brunette outdoors, wide shot") and rank the current filter with ${semanticState.label}.`
              : semanticBlocked}
            className="rounded-md border border-border bg-surface-raised px-2.5 py-0.5 text-xs text-content disabled:opacity-50 hover:bg-surface">
            🔤 Find by text…{!semanticReady && ` (needs ${semanticState.label})`}
          </button>
          {curateOpen === 'text' && (
            <>
              <div className="fixed inset-0 z-40"
                onClick={() => { setCurateOpen(null); releaseTextEncoder() }} aria-hidden />
              {/* Below sm this is a BOTTOM SHEET, not a dropdown. Anchored to its
                  trigger it is a w-80 panel hanging off a button that sits
                  mid-row: measured on a 400 px viewport it reached x=517 and made
                  the whole page scroll sideways. Pinning only the horizontal
                  gutters was not enough either — a fixed box with `top: auto`
                  resolves to its static position and lands off-screen once the
                  page is scrolled. So the vertical anchor is explicit, and the
                  sheet scrolls internally when the copy is long. From sm up it
                  behaves exactly like its two sibling popovers. */}
              <div className="fixed inset-x-4 bottom-4 z-50 max-h-[75vh] overflow-y-auto rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2 sm:absolute sm:inset-x-auto sm:bottom-auto sm:left-0 sm:mt-1 sm:max-h-none sm:w-80 sm:overflow-visible">
                <p className="text-xs text-content-muted">
                  Ranks the <strong>current filter</strong> by how close each image is to your
                  words. It refines what the grid is showing — it does not search the whole bank.
                </p>
                <label htmlFor="bank-text-search" className="block text-sm text-content">
                  What are you looking for?
                </label>
                <input id="bank-text-search" type="search" value={textQuery}
                  placeholder="brunette outdoors, wide shot"
                  onChange={(e) => setTextQuery(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !textPending) runTextSearch() }}
                  className="w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-content" />
                {/* The pedagogy that matters most, because the failure it
                    prevents is INVISIBLE: "without a hat" comes back full of
                    hats, confidently, with no signal. Offered, never applied on
                    its own — a wrong guess acted on silently would be the same
                    class of bug. */}
                {suggestPushDown(textQuery) && !textExclude.trim() && (
                  <p className="text-xs text-amber-300/90">
                    “without” is ignored by the search.{' '}
                    <button type="button"
                      onClick={() => {
                        setTextExclude(suggestPushDown(textQuery))
                        setTextQuery(withoutNegation(textQuery))
                      }}
                      className="underline underline-offset-2 hover:text-amber-200">
                      Push “{suggestPushDown(textQuery)}” down instead?
                    </button>
                  </p>
                )}
                <label htmlFor="bank-text-exclude" className="block text-sm text-content">
                  Push down (optional)
                </label>
                <input id="bank-text-exclude" type="search" value={textExclude}
                  placeholder="hat, sunglasses"
                  onChange={(e) => setTextExclude(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !textPending) runTextSearch() }}
                  className="w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-content" />
                <p className="text-xs text-content-subtle">{pushDownCaveat()}</p>
                {textExclude.trim() && (
                  <label className="flex flex-wrap items-center gap-2 text-sm text-content">
                    How hard
                    <select value={textExcludeW}
                      onChange={(e) => setTextExcludeW(Number(e.target.value))}
                      className="rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content">
                      {PUSH_DOWN_STRENGTHS.map((s) => (
                        <option key={s.value} value={s.value}>{s.label}</option>
                      ))}
                    </select>
                    <span className="w-full text-xs text-content-subtle">
                      {PUSH_DOWN_STRENGTHS.find((s) => s.value === textExcludeW)?.hint}
                    </span>
                  </label>
                )}
                <label className="flex items-center gap-2 text-sm text-content">
                  How many
                  <input type="number" min={1} max={2000} value={textN}
                    onChange={(e) => setTextN(Math.max(1, Math.min(2000, Number(e.target.value) || 1)))}
                    className="w-20 rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content" />
                </label>
                {/* The cost, BEFORE the click — a cold CLIP load is ~10 s and an
                    unexplained wait is exactly how this reads as a hang. */}
                <p className="text-xs text-content-subtle">
                  {readinessHint(textStatus, semanticState.engine)}
                </p>
                <p className="text-xs text-amber-300/80">
                  {limitsSentence(semanticState.engine)}
                </p>
                <button type="button" onClick={runTextSearch}
                  disabled={textPending || !textQuery.trim()}
                  className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-50">
                  {textPending ? pendingLabel(textStatus) : `Rank the closest ${textN}`}
                </button>
              </div>
            </>
          )}
        </div>
        {!semanticReady && (
          <span className="text-xs text-content-subtle">{semanticBlocked}</span>
        )}
        <button type="button" onClick={() => setCoverageOpen((v) => !v)}
          aria-expanded={coverageOpen}
          title="See what your kept set leans on and what's thin for a good LoRA — advice only, nothing is kept or rejected."
          className="rounded-md border border-border bg-surface-raised px-2.5 py-0.5 text-xs text-content disabled:opacity-50 hover:bg-surface">
          📊 Coverage advice{coverageOpen ? ' ▲' : ' ▼'}
        </button>
      </div>

      {/* 🔤 What the grid is currently showing, in words. This is the whole
          honesty of the feature: the grid alone would read as "these match",
          when it is a RANKING in which everything scores something. The range,
          the strength wording and the unsearchable count live here. aria-live so
          a screen reader hears the outcome — the grid change is silent. */}
      <div aria-live="polite">
        {textResult && (
          <div className="mt-2 space-y-1 rounded-lg border border-indigo-400/40 bg-indigo-500/10 px-3 py-2 text-xs text-content">
            <div className="flex flex-wrap items-start gap-x-2 gap-y-1">
              <span aria-hidden>🔤</span>
              <span className="min-w-0 flex-1">
                {summarize(textResult, semanticState.engine)}
              </span>
              <button type="button"
                onClick={() => { setTextResult(null); setShowSelected(false); refreshImages(filter, 0, { on: false }) }}
                className="shrink-0 rounded-md border border-border px-2 py-0.5 text-xs text-content hover:bg-surface-raised">
                Clear search
              </button>
            </div>
            {/* What the push-down did HERE, measured on this bank for this
                pair of phrases — including "it changed nothing", which is the
                outcome the user would otherwise never detect. */}
            {pushDownNote(textResult) && (
              <p className="text-content-muted">{pushDownNote(textResult)}</p>
            )}
            {textResult.cached === false && (
              <p className="text-content-subtle">
                This phrase is now cached — searching it again is instant, even after a restart.
              </p>
            )}
            <p className="text-amber-300/80">{limitsSentence(semanticState.engine)}</p>
          </div>
        )}
      </div>

      {/* ⚖️ What the balanced pick actually GAVE you. A repartition the user
          cannot see is indistinguishable from an unbalanced one, so this is
          numbers first — the bar is decoration over a list that reads out. */}
      <div aria-live="polite">
        {balanceResult && (
          <div className="mt-2 space-y-2 rounded-lg border border-emerald-400/40 bg-emerald-500/10 px-3 py-2 text-xs text-content">
            <div className="flex flex-wrap items-start gap-x-2 gap-y-1">
              <span aria-hidden>⚖️</span>
              <span className="min-w-0 flex-1">{summarizeBalance(balanceResult)}</span>
              <button type="button" onClick={() => setBalanceResult(null)}
                className="shrink-0 rounded-md border border-border px-2 py-0.5 text-xs text-content hover:bg-surface-raised">
                Dismiss
              </button>
            </div>
            <ul className="flex flex-wrap gap-x-4 gap-y-1 tabular-nums">
              {balanceRows(balanceResult).map((r) => (
                <li key={r.key} className={r.short ? 'text-amber-200' : 'text-content-muted'}>
                  <span className="text-content">{r.selected}</span> {r.label}
                  <span className="text-content-subtle"> of {r.available}</span>
                  {r.short && <span> · wanted {r.fairShare}</span>}
                </li>
              ))}
            </ul>
            {balanceNotes(balanceResult).map((note, i) => (
              <p key={i} className={note.tone === 'warn' ? 'text-amber-300/90' : 'text-content-subtle'}>
                {note.tone === 'warn' ? '⚠️ ' : '💡 '}{note.text}
              </p>
            ))}
          </div>
        )}
      </div>

      {coverageOpen && (
        <CoveragePanel coverage={coverage} semanticEngine={coverage?.engine || semanticState.engine}
          semanticLabel={semanticEngineLabel(coverage?.engine || semanticState.engine)}
          onClose={() => setCoverageOpen(false)}
          onBalance={balanceReady.ready ? () => setCurateOpen('balanced') : null}
          balanceReason={balanceReady.reason} />
      )}
      </ZoneSection>
      </div>

      {/* ④ Promote — ship the kept set into a dataset, or clear rejects off
          disk. Same actions/handlers as before — just grouped as the last step. */}
      <div className="min-w-0 xl:col-span-4 xl:sticky xl:top-20">
      <ZoneSection zone={promoteZone} accented={activeStep === 'promote'}>
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => setPromoteOpen(true)} disabled={live || !canPromote}
          title={canPromote
            ? 'Copy the kept selection into a dataset — or into a brand-new bank, to keep working on a shortlist apart'
            : 'Keep some images first'}
          className="rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
          ⬆ Promote…
        </button>
        {/* Disabled outright when this bank's folder belongs to a dataset: the
            banner above says it is, and a button that still opened a dialog only
            to be refused there would make that sentence a lie. */}
        <button type="button" onClick={() => setDeleteRejectedOpen(true)}
          disabled={live || !(counts?.reject > 0) || !!payload?.dataset_conflict}
          title={payload?.dataset_conflict
            ? 'This bank sits on a dataset’s image folder — deleting these files would delete the dataset’s images.'
            : (counts?.reject > 0)
              ? 'Delete the rejected images from your disk (OS trash when available). Irreversible — asks you to type DELETE first. Kept images are untouched.'
              : 'No rejected images to delete'}
          className="rounded-md border border-rose-500/50 px-3 py-1.5 text-sm text-rose-300 disabled:opacity-40 hover:bg-rose-500/10">
          🗑 Delete rejected from disk{(counts?.reject > 0) ? ` (${counts.reject})` : ''}
        </button>
      </div>
      </ZoneSection>
      </div>
      </div>

      {/* The selected-image tags are an inspector for the desktop gallery, not
          another full-width filter row. This responsive mount is hidden below
          xl; phones keep their established copy in the filter zone above. */}
      <div className={`grid gap-4 ${tagRow
        ? 'xl:grid-cols-[minmax(0,1fr)_20rem] xl:items-start'
        : 'grid-cols-1'}`}>
        {tagRow && (
          <aside aria-label="Image tags"
            className="hidden xl:col-start-2 xl:row-start-1 xl:block xl:sticky xl:top-20 xl:max-h-[calc(100vh-6rem)] xl:overflow-y-auto">
            <SelectionTagsPanel tagRow={tagRow} tagPicked={tagPicked}
              onToggle={(tag) => toggleTag(tag, tagRow)} onClear={clearTags} />
          </aside>
        )}
        <div className="min-w-0 xl:col-start-1 xl:row-start-1">
        {filter.flag === 'dups' ? (
          <DupGroupsPanel bankId={bankId} live={live} kind="exact"
            onChanged={async () => { await refreshPayload(); await refreshImages() }} />
        ) : filter.flag === 'semantic_dups' ? (
          <DupGroupsPanel bankId={bankId} live={live} kind="semantic"
            semanticLabel={semanticState.label}
            onChanged={async () => { await refreshPayload(); await refreshImages() }} />
        ) : (
          <>
          <ul className={`grid gap-2 ${tileSize === 'S'
            ? 'grid-cols-4 sm:grid-cols-6 lg:grid-cols-8'
            : 'grid-cols-3 sm:grid-cols-4 lg:grid-cols-6'}`}>
            {page.images.map((img) => (
              <Tile key={img.id} img={img} bankId={bankId} size={tileSize}
                selected={selected.has(img.id)}
                onReview={() => openReview(img.id)}
                onTags={() => openTagPicker(img)}
                onToggle={() => setSelected((prev) => {
                  const next = new Set(prev)
                  if (next.has(img.id)) next.delete(img.id); else next.add(img.id)
                  return next
                })} />
            ))}
          </ul>
          {page.total === 0 && (
            <p className="text-sm text-content-muted">Nothing matches this filter.</p>
          )}
          {page.total > PAGE_SIZE && (
            <nav className="flex items-center gap-3 text-sm" aria-label="Grid pages">
              <button type="button" disabled={offset === 0} onClick={() => goto(Math.max(0, offset - PAGE_SIZE))}
                className="rounded-md border border-border px-2 py-1 text-content disabled:opacity-40">← Prev</button>
              <span className="text-content-muted">
                {offset + 1}–{Math.min(offset + PAGE_SIZE, page.total)} of {page.total}
              </span>
              <button type="button" disabled={offset + PAGE_SIZE >= page.total}
                onClick={() => goto(offset + PAGE_SIZE)}
                className="rounded-md border border-border px-2 py-1 text-content disabled:opacity-40">Next →</button>
            </nav>
          )}
          </>
        )}
        </div>
      </div>

      {promoteOpen && (
        <PromoteDialog bankId={bankId}
          selectedIds={[...selected]}
          onClose={() => setPromoteOpen(false)}
          // refreshImages too: a promotion marks the rows it carried, and the ⬆
          // badge lives on the TILES — refreshing only the counters left the
          // header saying "3 promoted" over a grid showing none.
          onStarted={() => { setPromoteOpen(false); refreshPayload(); refreshImages() }} />
      )}

      {deleteRejectedOpen && (
        <DeleteRejectedDialog bankId={bankId} count={counts?.reject || 0}
          sourcePath={payload?.source_path}
          onClose={() => setDeleteRejectedOpen(false)}
          onDone={() => { setDeleteRejectedOpen(false); setSelected(new Set()); refreshPayload(); refreshImages() }} />
      )}

      {/* 🎛 THE PASS LAUNCH WINDOW — one component for every pass. 👥 routes through
          the folder preflight, 🏷️ Caption carries its five options and its
          destructive twin; everything else is the shared three blocks. */}
      {passOpen && (
        <PassDialog passId={passOpen} payload={payload} live={live}
          semanticEngine={semanticState.engine}
          selectionSize={selected.size}
          detectorReady={!!caps.watermark_detect}
          scope={passScopes[passOpen] || ''}
          onScope={(v) => setPassScope(passOpen, v)}
          redo={!!passRedo[passOpen]}
          onRedo={(v) => setPassRedoFor(passOpen, v)}
          onClose={() => setPassOpen(null)}
          onLaunch={(run) => (passOpen === 'faces'
            ? launchFacesFromDialog()
            : (passOpen === 'caption'
              ? startCaption(run)
              : runPass(passOpen, run)))}
          secondary={passOpen === 'caption' ? captionSecondary : null}>
          {passOpen === 'caption' ? captionRunControls : null}
        </PassDialog>
      )}

      {launchOpen && (
        <LaunchAllDialog caps={caps} visionReady={visionReady}
          semanticEngine={semanticState.engine}
          counts={counts} flagsActionable={flagsActionable}
          onClose={() => setLaunchOpen(false)} onLaunch={startPipeline} />
      )}

      {preflight && (
        <PersonPreflightDialog plan={preflight.plan} probing={preflight.probing}
          activity={payload?.activity}
          reload={() => apiFetch(`/api/bank/${bankId}/person-preflight`)}
          onProceed={proceedPreflight}
          onStopProbe={() => postJson(`/api/bank/${bankId}/cancel`, {})}
          onCancel={() => setPreflight(null)} />
      )}

      {pythonPickerFor && (
        <ScoringPythonDialog profile={PICKER_PROFILES[pythonPickerFor]}
          onClose={() => setPythonPickerFor('')}
          onChanged={async () => {
            // The passes read bank_scoring_gpu_available() /
            // bank_siglip2_gpu_available(); force both probes so the CPU warning
            // and the "holds the GPU" tooltip agree at once.
            await refreshCaps(true)
            await refreshPayload()
          }} />
      )}

      {review && (
        <BankReviewLightbox bankId={bankId} ids={review.ids} startId={review.startId}
          seedImages={page.images} onDecided={onReviewDecided}
          onRotated={onReviewRotated} onClose={closeReview} />
      )}

      {relocating && (
        <RelocateBankDialog bankId={bankId} bankName={payload?.name || `Bank #${bankId}`}
          sourcePath={payload?.source_path} onClose={() => setRelocating(false)}
          onDone={() => { refreshPayload({ force: true }); refreshImages() }} />
      )}
    </div>
  )
}
