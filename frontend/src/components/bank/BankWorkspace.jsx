import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { useCapabilities } from '../../context/CapabilitiesContext'
import { useConnectionStatus } from '../../hooks/useConnectionStatus'
import DupGroupsPanel from './DupGroupsPanel'
import PromoteDialog from './PromoteDialog'
import DeleteRejectedDialog from './DeleteRejectedDialog'
import LaunchAllDialog from './LaunchAllDialog'
import ScoringPythonDialog from './ScoringPythonDialog'
import PipelineReport from './PipelineReport'
import FolderSyncNote from './FolderSyncNote'
import RelocateBankDialog from './RelocateBankDialog'
import BankReviewLightbox from './BankReviewLightbox'
import BankWatermarkPanel from './BankWatermarkPanel'
// 🎚 The twelve triage thresholds, edited here instead of in Settings.
import BankThresholdsPanel from './BankThresholdsPanel.jsx'
// Source-folder re-walk messages (pure/testable).
import { folderSyncToast } from './bankSync.js'
import { UNDO_HINT, undoBannerText, undoOffer, undoResultMessage } from './bankUndo.js'
// Four progress states, not two — including the honest "I don't know" (pure/testable).
import { progressPresence, PROGRESS_HIDDEN, PROGRESS_UNKNOWN, PROGRESS_STALE } from './progressPresence.js'
// An occupied bank refuses in OUR words, never in the server's (pure/testable).
import { busyRefusal } from './bankPassRun.js'
import { holdsTheGpu, scoreDeviceNote } from './bankScoreDevice.js'
// Wording that adapts to the machine (a card-less box is never sold CUDA).
import { openerLabel } from './scoringPython.js'
// Reuse the dataset's register list so the Bank lane never drifts from it.
import { VOCABULARY_OPTIONS } from '../dataset/CaptionOptionsPopover'
// Ordered zone model + the "what's next" accent, both pure/testable.
import { BANK_ZONES, nextBankStep } from './bankGuide.js'
// Provenance wording (effective resolution, origin, black bars) — pure/testable.
import { ORIGIN_CHIPS, PROVENANCE_FLAG_LABEL, detailSummary } from './bankProvenance.js'
// Grid ordering menu (which sorts exist, and which ones have data) — pure/testable.
import { bankSortOptions } from '../../utils/gridSort.js'
// 🔤 Text search wording — "closest", never "matching" — plus the cold-start and
// CLIP-limitation copy. Pure/testable (node --test cannot parse this JSX).
import {
  limitsSentence, pendingLabel, readinessHint, summarize,
} from './bankTextSearch.js'
// ⚖️ Balanced pick — the distribution obtained, in words and numbers. Pure logic
// on purpose: the repartition is what has to be provable (node --test, no JSX).
import {
  BALANCE_AXES, BALANCE_DEFAULT_AXIS, balanceNotes, balanceReadiness,
  balanceRows, summarizeBalance,
} from './bankBalance.js'

const PAGE_SIZE = 120

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
  { id: 'res_gt_4', label: '> 4 MP' },
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
  const ids = []
  let offset = 0
  for (;;) {
    const qs = new URLSearchParams({ ...params, offset: String(offset), limit: '500' })
    const d = await apiFetch(`/api/bank/${bankId}/images?${qs}`)
    ids.push(...d.images.map((i) => i.id))
    offset += d.images.length
    if (offset >= d.total || d.images.length === 0) break
  }
  return ids
}

const STEP_SHORT = {
  scan: '🔎 Scan', auto_reject: '🧹 Auto-reject', score: '✨ Score',
  semantic_dedup: '✂ Crops', watermark: '🚩 Watermarks', faces: '👥 Person',
  framing: '📐 Framing', caption: '🏷️ Caption',
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

function ProgressBar({ activity, onCancel, offline = false }) {
  const presence = progressPresence(activity, offline)
  if (presence === PROGRESS_HIDDEN) return null
  if (presence === PROGRESS_UNKNOWN) return <ProgressUnknown />
  const stale = presence === PROGRESS_STALE
  const { kind, done, total, detail } = activity
  const pct = total > 0 ? Math.round((100 * done) / total) : null
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
              semantic_dedup: 'Crops & variants', watermark: 'Watermark scan',
              framing: 'Framing pass', caption: 'Captioning', promote: 'Promotion',
              bank_promote: 'Copying into the new bank' }[kind] || 'Job') + ' running'}
          {' — '}{done}{total ? ` / ${total}` : ''}{detail ? ` · ${detail}` : ''}
        </span>
        {pct != null && (
          <div className="h-1.5 w-40 overflow-hidden rounded bg-surface-raised" role="progressbar"
            aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-full bg-amber-400" style={{ width: `${pct}%` }} />
          </div>
        )}
        <button type="button" onClick={onCancel}
          className="ml-auto rounded-md border border-border px-2 py-0.5 text-xs text-content hover:bg-surface-raised">
          Stop
        </button>
      </div>
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

function CoveragePanel({ coverage, onClose, onBalance = null, balanceReason = '' }) {
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
      <p className="text-[11px] text-content-subtle">
        Advice only — nothing is kept or rejected. Based on what the passes already computed.
      </p>
    </div>
  )
}

function Tile({ img, bankId, selected, onToggle, onReview, size }) {
  // `key` matters only for the flags list below (the one mapped array) — it was
  // missing and logged a React warning on every bank grid render.
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
          + (img.face_cluster ? ` · person #${img.face_cluster}` : '')
          + (img.framing ? ` · ${img.framing}` : '')
          + (detailSummary(img)?.soft ? ` · only ~${detailSummary(img).real} px of real detail` : '')
          + (img.origin && img.origin !== 'unknown' ? ` · ${img.origin}` : '')
          + (img.style_cluster ? ` · style #${img.style_cluster}` : '')
          + (img.semantic_dup_group ? ` · same shot #${img.semantic_dup_group}` : '')
          + (img.caption ? `\n${img.caption}` : '')}
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
        {/* Only the PROVEN states get a badge. Stamping ❔ on the 80% of files
            whose metadata was stripped would be noise, not information. */}
        {img.origin === 'ai' && badge('🤖', 'bg-black/60 text-violet-200')}
        {img.origin === 'camera' && badge('📷', 'bg-black/60 text-emerald-200')}
        {img.style_cluster != null && badge(`🎨${img.style_cluster}`, 'bg-black/60 text-fuchsia-200')}
        {img.dup_group != null && badge(`≈${img.dup_group}`, 'bg-black/60 text-fuchsia-200')}
        {img.semantic_dup_group != null && badge(`✂${img.semantic_dup_group}`, 'bg-black/60 text-orange-200')}
        {img.caption && badge('🏷️', 'bg-black/60 text-emerald-200')}
      </span>
      {/* ▶ starts the fast-triage lightbox AT this image. It's a separate hit
          target on purpose: the tile's own click still (de)selects for the bulk
          ✓/✕/⬆ bar, so neither use loses its gesture. */}
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
  const [payload, setPayload] = useState(null)
  const [filter, setFilter] = useState({ status: null, flag: null, cluster: null,
    style: null, subfolder: null, search: null, sort: 'default', resBucket: null,
    origin: null,
    framing: null })
  const [searchText, setSearchText] = useState('')
  const [subfolders, setSubfolders] = useState([])
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
  // ✨ Score's interpreter picker — reuse a CUDA Python this machine already has
  // instead of downloading another torch. Opened from the CPU warning.
  const [scoringPythonOpen, setScoringPythonOpen] = useState(false)
  const [dismissedReportAt, setDismissedReportAt] = useState(null)
  const [relocating, setRelocating] = useState(false)
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
  // 🔤 Text search. `textStatus` is the BEFORE-the-click truth (available? model
  // already warm? would it download?), `textResult` the AFTER-the-click one that
  // keeps the ranking legible once the grid has switched to it.
  const [textQuery, setTextQuery] = useState('')
  const [textN, setTextN] = useState(60)
  const [textStatus, setTextStatus] = useState(null)
  const [textPending, setTextPending] = useState(false)
  const [textResult, setTextResult] = useState(null)
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
  // 📡 Drives the "we lost contact" note in the progress zone: a failed poll
  // must never render as "no job running".
  const connection = useConnectionStatus()

  const loadCoverage = useCallback(async () => {
    try {
      setCoverage(await apiFetch(`/api/bank/${bankId}/coverage`))
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
    // Grid sort (resolution / aesthetic / sharpness, each way) — sent to the grid
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

  useEffect(() => {
    refreshPayload({ force: true }); refreshImages()
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
    // background: this 2 s tick is the one that stacked ten "Connection lost"
    // banners over the whole app when the phone's connection dropped.
    const t = setInterval(() => refreshPayload({ background: true }), 2000)
    return () => clearInterval(t)
  }, [live, refreshPayload, refreshImages, toast, payload?.activity?.error,
      payload?.activity?.cancelled, payload?.activity?.detail])

  // Keep the coverage panel current: refetch when it opens, and whenever the kept
  // set or the framing classification changes (a keep/reject or the framing pass).
  useEffect(() => {
    if (coverageOpen) loadCoverage()
  }, [coverageOpen, loadCoverage, payload?.counts?.keep,
      payload?.counts?.framing_classified])

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
    setFilter(f); setOffset(0); exitSelectionView()
    refreshImages(f, 0, { on: false })
  }

  // Debounce the search box, then apply it as a filter (page 1, selection cleared).
  useEffect(() => {
    const term = searchText.trim()
    if ((filter.search || '') === term) return undefined
    const t = setTimeout(() => setF({ search: term || null }), 300)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchText])
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

  const startScan = (rescan) => act(
    () => postJson(`/api/bank/${bankId}/scan`, { rescan: !!rescan }), null)
  const startFaces = () => act(() => postJson(`/api/bank/${bankId}/faces`, {}), null)
  const startScore = () => act(() => postJson(`/api/bank/${bankId}/score`, {}), null)
  const startSemanticDedup = () => act(
    () => postJson(`/api/bank/${bankId}/semantic-dedup`, {}), null)
  const startFraming = () => act(() => postJson(`/api/bank/${bankId}/framing`, {}), null)
  const startCaption = () => act(
    () => postJson(`/api/bank/${bankId}/caption`, {
      ...(selected.size ? { image_ids: [...selected] } : {}),
      ...(captionVocab ? { vocabulary: captionVocab } : {}),
    }), null)
  const cancelJob = () => act(() => postJson(`/api/bank/${bankId}/cancel`, {}), null)
  /* Posts with the dialog still OPEN and answers {ok,error}: a refused launch —
     "a scan job is already running on this bank" is the usual one — used to close
     the dialog first and reset all seven pass checkboxes and the reject flags to
     their defaults. The dialog decides what to do with the answer. */
  const startPipeline = async (config) => {
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
      toast.success(`Auto-reject: ${n} image(s) rejected (${flags.join(', ')}). Manual ✓/✕ untouched.`)
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

  // --- Curation selectors (reuse the ✨ Score embeddings — no GPU) ------------
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
    setCurateOpen(null)
    setDiverseBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/select-diverse`,
        { n: diverseN, typicality: diverseTypicality, ...filterParams(filter) })
      if (!d.image_ids?.length) {   // scored, but the current filter holds nothing
        toast.info('Nothing to sample — no scored images match the current filter.')
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
    setCurateOpen(null)
    setBalanceBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/select-balanced`,
        { n: balanceN, axis: balanceAxis, typicality: diverseTypicality,
          ...filterParams(filter) })
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
    setCurateOpen(null)
    const ref = [...selected][0]
    if (ref == null) return
    try {
      const d = await postJson(`/api/bank/${bankId}/select-similar`,
        { ref_id: ref, n: similarN, ...filterParams(filter) })
      if (!d.image_ids?.length) {
        toast.info('No matches — no scored images match the current filter.')
        return
      }
      // Backend returns the ids ranked by similarity (reference first); keep that
      // order so the view reads closest→farthest instead of by id.
      showCuratedSelection(d.image_ids)
      toast.info(`Showing the ${d.image_ids.length} most similar to the reference (of ${d.pool}), closest first. Review, then ✓ Keep or ⬆ Promote — or “Show all” to leave this view.`)
    } catch (e) {
      toast.error(e?.message || 'Similarity search failed.')
    }
  }

  // 🔤 Text search — same engine as 🎯 Similar, with the reference vector coming
  // from words instead of a picture. Opening the panel asks the backend what it
  // is about to cost (model warm? weights present?) so a slow FIRST search is
  // announced before the click rather than felt as a freeze after it.
  const openTextSearch = async () => {
    const next = curateOpen === 'text' ? null : 'text'
    setCurateOpen(next)
    if (next !== 'text') { releaseTextEncoder(); return }
    try {
      setTextStatus(await apiFetch('/api/bank/text-search/status'))
    } catch {
      setTextStatus(null)      // a status we couldn't read never blocks the field
    }
  }

  // Hand the ~2.4 GB back as soon as the panel closes. Best effort by design —
  // the backend's idle timer is the real guarantee for a tab that just vanished.
  const releaseTextEncoder = () => {
    postJson('/api/bank/text-search/release', {}).catch(() => {})
  }

  // Leaving the Bank entirely is the same signal as closing the panel: give the
  // memory back. The backend idle timer still covers a browser that just died.
  useEffect(() => () => { releaseTextEncoder() }, [])   // eslint-disable-line react-hooks/exhaustive-deps

  const runTextSearch = async () => {
    const q = textQuery.trim()
    if (!q) return
    setTextPending(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/search-text`,
        { query: q, n: textN, ...filterParams(filter) })
      setTextResult(d)
      setCurateOpen(null)
      if (!d.image_ids?.length) {
        // NOT a silent empty grid: say why nothing could be ranked.
        toast.info(summarize(d))
        return
      }
      showCuratedSelection(d.image_ids)
      // Refresh the warm flag so the panel now promises "instant" truthfully.
      apiFetch('/api/bank/text-search/status').then(setTextStatus).catch(() => {})
    } catch (e) {
      // 503 = this install cannot do it at all; 400 = do something first. Both
      // arrive as a message written for a human — show it as-is.
      toast.error(e?.message || 'Text search failed.')
    } finally {
      setTextPending(false)
    }
  }

  const counts = payload?.counts
  // ↩ the live offer, minus the one the user already waved away.
  const offer = undoOffer(payload)
  const undoBar = offer && offer.at !== undoDismissedAt ? offer : null
  const flags = payload?.flags || {}
  const resBuckets = payload?.res_buckets || {}
  // Only surface tiers that actually hold scanned images (plus the active one,
  // so a tier you're filtering on never vanishes mid-review).
  const shownResBuckets = RES_BUCKETS.filter(
    (b) => (resBuckets[b.id] || 0) > 0 || filter.resBucket === b.id)
  const clusters = payload?.clusters || []
  const styleClusters = payload?.style_clusters || []
  const framingCounts = payload?.framing || {}
  const framingClassified = counts?.framing_classified || 0
  // Only surface framing chips once the pass has classified something (plus the
  // active one, so a chip you're filtering on never vanishes mid-review).
  const shownFramings = FRAMING_BUCKETS.filter(
    (b) => (framingCounts[b.id] || 0) > 0 || filter.framing === b.id)
  // Origin chips appear as soon as the quality scan has measured anything. All
  // three are shown together once any is non-zero: hiding 'ai' at 0 would read as
  // "no AI images here", when what it means is "none that still carry metadata".
  const originCounts = payload?.origins || {}
  const originMeasured = ORIGIN_BUCKETS.reduce((n, b) => n + (originCounts[b.id] || 0), 0)
  const visionReady = !!caps.ollama?.vision_model_ready
  // The explicit lane only spells acts out with an uncensored (abliterated) vision
  // model. We can't prove abliteration, but the common builds name themselves — a soft
  // heuristic drives an honest "may soften" hint (never a hard block: a differently
  // named abliterated model still works).
  const visionModel = caps.ollama?.vision_model || ''
  const visionModelLooksUncensored = /abliterat|uncensor|huihui|nsfw/i.test(visionModel)
  const scored = counts?.scored || 0
  // ⚖️ Can a balanced pick even run? Answered BEFORE the click when we already
  // know (Score missing; coverage says nothing is classified) — otherwise the
  // backend answers it with the exact pass and the numbers.
  const balanceReady = balanceReadiness({ scored, coverage })
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
  const canPromote = (counts?.keep || 0) > 0 || selected.size > 0
  // Is any facet narrowing the grid? Drives the "N shown of TOTAL" readout.
  const isFiltered = !!(filter.status || filter.flag || filter.cluster != null
    || filter.style != null || filter.subfolder != null || filter.search
    || filter.resBucket || filter.framing)

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

      {/* ① Analyze — run the analysis passes (or 🚀 Launch all) on the dump.
          Grouping + accent only; every pass keeps its own endpoint/behaviour. */}
      <ZoneSection zone={analyzeZone} accented={activeStep === 'analyze'}>
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => setLaunchOpen(true)} disabled={live || !(counts?.total > 0)}
          title="Run the whole triage in one go — scan, auto-reject, score, watermarks, group by person and (optionally) caption. Start it and walk away."
          className="rounded-md bg-gradient-primary px-4 py-2 text-sm font-bold text-white shadow disabled:opacity-50">
          🚀 Launch all…
        </button>
        <p className="hidden text-xs text-content-subtle md:block">
          One-click funnel — or step through the passes below.
        </p>
      </div>

      {/* Analysis passes — individual, quieter than the primary actions. */}
      <div className="space-y-1.5">
        <GroupLabel>Analysis passes</GroupLabel>
        <div className="flex flex-wrap items-center gap-1.5">
          <PassButton onClick={() => startScan(false)} disabled={live}
            title="Score every unscanned image (sharpness/noise/flat/size), hash it and group near-duplicates — CPU only, runs in the background">
            🔎 Scan quality
          </PassButton>
          {(counts?.scanned || 0) > 0 && (
            <PassButton onClick={() => startScan(true)} disabled={live}
              title="Re-score everything (e.g. after files changed on disk)">
              Rescan all
            </PassButton>
          )}
          <PassButton onClick={startFaces} disabled={live || !caps.face_scoring}
            title={caps.face_scoring
              ? 'Detect the dominant face of every non-rejected image and cluster the bank by person (no reference needed). CPU, can take a while on thousands of images.'
              : 'Install the Quality tools (Setup) to sort by person'}>
            👥 Group by person
          </PassButton>
          <PassButton onClick={startScore} disabled={live || !caps.bank_scoring}
            title={caps.bank_scoring
              ? `Rate every non-rejected image for aesthetics (1–10), flag NSFW, and group by visual style — one CLIP pass. Powers a smarter "keep best". Runs in the background${
                holdsTheGpu(scoreDevice) ? ', and holds the GPU (ComfyUI is unloaded and training cannot start) for its duration' : ' on the CPU, leaving the GPU free'}.`
              : 'Install the Bank scoring extra (Setup ▸ Quality tools) to score aesthetics / NSFW / style'}>
            ✨ Score{!caps.bank_scoring && ' (needs setup)'}
          </PassButton>
          <PassButton onClick={startFraming} disabled={live || !visionReady}
            title={visionReady
              ? 'Classify every non-rejected image by shot type — face close-up, bust, full body, back view — with the same Qwen3-VL classifier the datasets use. Powers the 📐 Framing filter and the coverage advice. GPU vision pass.'
              : 'Pull the vision model (Settings ▸ Captioning & quality) to classify framing'}>
            📐 Classify framing{!visionReady && ' (needs setup)'}
          </PassButton>
          <PassButton onClick={startSemanticDedup} disabled={live || scored === 0}
            title={scored > 0
              ? 'Group crops and re-compressed variants of the SAME shot the exact-duplicate hash misses — reuses the ✨ Score embeddings, so it costs no extra GPU time. Review them under the ✂ Same shot chip.'
              : 'Run ✨ Score first — semantic near-duplicates reuse its embeddings'}>
            ✂ Find crops &amp; variants{scored === 0 && ' (needs Score)'}
          </PassButton>
          <PassButton onClick={startCaption} disabled={live}
            title={selected.size
              ? `Caption the ${selected.size} selected image(s) with your caption engine (Settings ▸ Captioning & quality). Captions become searchable and follow the images when you promote them to a dataset.`
              : 'Caption every not-yet-captioned image (skips rejected) with your caption engine. Captions become searchable tags and follow the images when you promote them to a dataset. Select images first to caption just those.'}>
            🏷️ Caption{selected.size ? ` ${selected.size} selected` : ' all'}
          </PassButton>
          <label className="flex items-center gap-1 text-xs text-content-subtle">
            <span className="sr-only">Caption vocabulary register</span>
            <select value={captionVocab} onChange={(e) => setCaptionVocab(e.target.value)}
              disabled={live} aria-label="Caption vocabulary register"
              title="How captions name nude or sexual content. Explicit needs an uncensored (abliterated) Ollama vision model. Richer, more explicit captions also make the 🔍 search find more."
              className="px-2 py-1 rounded-lg bg-app/60 border border-border text-content text-xs disabled:opacity-40">
              {VOCABULARY_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
            </select>
          </label>
        </div>
        {/* Watermark CLEANING — the two manual levels (crop, then inpaint), with
            their own per-level progress. Lives in its own component so the
            "which level can run, and why not" logic stays unit-tested. */}
        <BankWatermarkPanel bankId={bankId} live={live}
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
            <button type="button" onClick={() => setScoringPythonOpen(true)}
              className={`rounded-md border px-2 py-1 text-xs font-medium ${scoreGpuPresent
                ? 'border-amber-400/50 text-amber-300 hover:bg-amber-500/10'
                : 'border-border text-content-muted hover:bg-surface-raised hover:text-content'}`}>
              {openerLabel(scoreGpuPresent)}
            </button>
          </div>
        )}
        {captionVocab === 'explicit' && !visionModelLooksUncensored && (
          <p className="text-xs text-amber-400/90">
            ⚠️ Explicit captions need an uncensored (abliterated) Ollama vision model
            {visionModel ? ` — “${visionModel}” may refuse or soften explicit terms` : ''}.
            Pull one in Settings ▸ Captioning &amp; quality. Richer captions also feed the 🔍 search.
          </p>
        )}
      </div>
      </ZoneSection>

      {/* ② Triage — browse by facet/cluster and Keep/Reject to decide what stays.
          Stays fully visible; density was never the complaint. */}
      <ZoneSection zone={triageZone} accented={activeStep === 'triage'}>

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
          {/* Subfolder scoping (a Telegram export nests one folder per chat/date) */}
          {subfolders.length > 1 && (
            <div className="flex items-center gap-1.5">
              <GroupLabel>Subfolder</GroupLabel>
              <select value={filter.subfolder ?? '__all__'}
                onChange={(e) => setF({ subfolder: e.target.value === '__all__' ? null : e.target.value })}
                className="rounded-md border border-border bg-surface px-2 py-1 text-xs text-content">
                <option value="__all__">All subfolders</option>
                {subfolders.map((s) => (
                  <option key={s.name || '__root__'} value={s.name}>
                    {s.name === '' ? '(bank root)' : s.name} · {s.count}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* Filters — grouped by facet so the chips read as a system, not a wall */}
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

          {/* Score-derived flags — only surfaced once their pass has produced data. */}
          {availableScoreFlags.length > 0 && (
            <FilterGroup label="Score">
              {availableScoreFlags.map((f) => (
                <Chip key={f} active={filter.flag === f}
                  onClick={() => setF({ flag: filter.flag === f ? null : f, cluster: null, style: null })}
                  title={f === 'watermark' ? 'Overlaid watermark detected' : 'Sorted worst-first'}>
                  {FLAG_LABEL[f]} {flags[f] ?? 0}
                </Chip>
              ))}
            </FilterGroup>
          )}

          <FilterGroup label="Groups">
            <Chip active={filter.flag === 'dups'} onClick={() => setF({ flag: filter.flag === 'dups' ? null : 'dups', cluster: null })}
              title="Exact / resized duplicate groups (perceptual hash) with their resolution panel">
              ≈ Duplicates {payload?.dup?.unresolved ?? 0}
            </Chip>
            {(payload?.semantic_dup?.groups ?? 0) > 0 && (
              <Chip active={filter.flag === 'semantic_dups'}
                onClick={() => setF({ flag: filter.flag === 'semantic_dups' ? null : 'semantic_dups', cluster: null })}
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
                onRunPass={(endpoint) => act(
                  () => postJson(`/api/bank/${bankId}/${endpoint}`, {}), null)} />
            </div>
          )}
        </div>

        {/* View controls — order and tile size, off to the right on their own line */}
        <div className="flex flex-wrap items-center gap-2 border-t border-border pt-2">
          <GroupLabel>View</GroupLabel>
          <label className="flex items-center gap-1 text-xs text-content-muted">
            Sort
            {/* Order the grid on what the passes MEASURED — resolution, aesthetic
                rating, sharpness — so a review opens on what it is looking for.
                Images the matching pass never reached sink to the end (never the
                top), and an entry whose pass has produced nothing yet is greyed
                out saying which pass to run. The value rides to the server, which
                sorts in SQL: it applies to the WHOLE filter, not this page, so
                "Select all in filter" and ▶ Review walk the same order.
                max-w keeps the control inside a 400 px toolbar. */}
            <select value={filter.sort} onChange={(e) => setSort(e.target.value)}
              title="Order the grid by resolution, aesthetic rating or sharpness. Images a pass never reached sink to the end."
              aria-label="Sort the grid"
              className="max-w-[11rem] rounded-md border border-border bg-surface px-2 py-0.5 text-xs text-content">
              {bankSortOptions(counts).map((o) => (
                <option key={o.id} value={o.id} disabled={o.disabled} title={o.title}>
                  {o.label}
                </option>
              ))}
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
              <div className="absolute left-0 z-50 mt-1 w-72 rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2">
                <p className="text-xs text-content-muted">
                  Rejects the UNDECIDED images with these flags. Your manual ✓/✕ are never changed;
                  everything stays reversible (nothing is deleted from disk).
                </p>
                {/* The caveat is printed, not left in a title= tooltip: soft_detail
                    and bars are provenance HINTS, not verdicts, and this is the one
                    screen that offers to act on them in bulk. A tooltip is invisible
                    on a phone and to anyone who does not hover. */}
                {[...QUALITY_REJECT_FLAGS, ...availableScoreFlags].map((f) => (
                  <label key={f} className="block text-sm text-content">
                    <span className="flex items-center gap-2">
                      <input type="checkbox" checked={rejectFlags.has(f)}
                        onChange={(e) => setRejectFlags((prev) => {
                          const next = new Set(prev)
                          if (e.target.checked) next.add(f); else next.delete(f)
                          return next
                        })} />
                      {FLAG_LABEL[f]} <span className="text-content-subtle">({flags[f] ?? 0} flagged)</span>
                    </span>
                    {FLAG_HINT[f] && (
                      <span className="mt-0.5 block pl-6 text-[0.6875rem] leading-snug text-amber-200/80">
                        ⚠ {FLAG_HINT[f]}
                      </span>
                    )}
                  </label>
                ))}
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

      {/* ③ Curate — optional refinement (diverse/similar/coverage). Always
          accessible, but never the accented "next step". */}
      <ZoneSection zone={curateZone} accented={false}>

      {/* Curation — build a good LoRA subset out of a big dump (reuses ✨ Score
          embeddings, no GPU). Diversity coverage + reference similarity, both
          producing a SELECTION the user reviews above. */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-xs font-semibold uppercase tracking-wide text-content-subtle">Curate</span>
        <div className="relative">
          <button type="button" disabled={live || scored === 0 || diverseBusy}
            onClick={() => setCurateOpen((v) => (v === 'diverse' ? null : 'diverse'))}
            aria-expanded={curateOpen === 'diverse'}
            title={scored > 0
              ? 'Pick the N images that best COVER the visual variety of the current filter (varied angles/outfits/scenes) — the fix for a dump of near-identical shots. Reuses the ✨ Score embeddings, no GPU.'
              : 'Run ✨ Score first — diversity sampling reuses its embeddings'}
            className="rounded-md border border-border bg-surface-raised px-2.5 py-0.5 text-xs text-content disabled:opacity-50 hover:bg-surface">
            🎨 Pick diverse…{scored === 0 && ' (needs Score)'}{diverseBusy && ' (sampling…)'}
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
              ? 'Select N images SPREAD OVER the framings (face / bust / body / back) instead of the top of one ranking — so a LoRA does not learn one shot type and fail the rest. Reuses the ✨ Score embeddings, no GPU.'
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
          <button type="button" disabled={live || scored === 0 || selected.size !== 1}
            onClick={() => setCurateOpen((v) => (v === 'similar' ? null : 'similar'))}
            aria-expanded={curateOpen === 'similar'}
            title={scored === 0
              ? 'Run ✨ Score first — reference similarity reuses its embeddings'
              : selected.size === 1
                ? 'Rank the current filter by how much it looks like the ONE selected image, and select the closest N — pull a person/look out of a mixed dump. Reuses the ✨ Score embeddings, no GPU.'
                : 'Select exactly one image to use as the reference'}
            className="rounded-md border border-border bg-surface-raised px-2.5 py-0.5 text-xs text-content disabled:opacity-50 hover:bg-surface">
            🎯 Similar to selected…
          </button>
          {curateOpen === 'similar' && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setCurateOpen(null)} aria-hidden />
              <div className="absolute z-50 mt-1 w-72 rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2">
                <p className="text-xs text-content-muted">
                  Ranks the current filter by CLIP similarity to your one selected image and selects
                  the closest — a fast way to extract one person or look. The reference is kept in
                  the selection.
                </p>
                <label className="flex items-center gap-2 text-sm text-content">
                  How many
                  <input type="number" min={1} max={2000} value={similarN}
                    onChange={(e) => setSimilarN(Math.max(1, Math.min(2000, Number(e.target.value) || 1)))}
                    className="w-20 rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content" />
                </label>
                <button type="button" onClick={findSimilar}
                  className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white">
                  Select {similarN} most similar
                </button>
              </div>
            </>
          )}
        </div>
        <div className="relative">
          <button type="button" disabled={live || scored === 0}
            onClick={openTextSearch}
            aria-expanded={curateOpen === 'text'}
            title={scored > 0
              ? 'Describe what you are looking for in words ("brunette outdoors, wide shot") and rank the current filter by how close each image is. Reuses the ✨ Score embeddings, no GPU.'
              : 'Run ✨ Score first — text search ranks the embeddings it computes'}
            className="rounded-md border border-border bg-surface-raised px-2.5 py-0.5 text-xs text-content disabled:opacity-50 hover:bg-surface">
            🔤 Find by text…{scored === 0 && ' (needs Score)'}
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
                <label className="flex items-center gap-2 text-sm text-content">
                  How many
                  <input type="number" min={1} max={2000} value={textN}
                    onChange={(e) => setTextN(Math.max(1, Math.min(2000, Number(e.target.value) || 1)))}
                    className="w-20 rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content" />
                </label>
                {/* The cost, BEFORE the click — a cold CLIP load is ~10 s and an
                    unexplained wait is exactly how this reads as a hang. */}
                <p className="text-xs text-content-subtle">{readinessHint(textStatus)}</p>
                <p className="text-xs text-amber-300/80">{limitsSentence()}</p>
                <button type="button" onClick={runTextSearch}
                  disabled={textPending || !textQuery.trim()}
                  className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-50">
                  {textPending ? pendingLabel(textStatus) : `Rank the closest ${textN}`}
                </button>
              </div>
            </>
          )}
        </div>
        {scored === 0 && (
          <span className="text-xs text-content-subtle">Run ✨ Score to unlock curation.</span>
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
              <span className="min-w-0 flex-1">{summarize(textResult)}</span>
              <button type="button"
                onClick={() => { setTextResult(null); setShowSelected(false); refreshImages(filter, 0, { on: false }) }}
                className="shrink-0 rounded-md border border-border px-2 py-0.5 text-xs text-content hover:bg-surface-raised">
                Clear search
              </button>
            </div>
            {textResult.cached === false && (
              <p className="text-content-subtle">
                This phrase is now cached — searching it again is instant, even after a restart.
              </p>
            )}
            <p className="text-amber-300/80">{limitsSentence()}</p>
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
        <CoveragePanel coverage={coverage} onClose={() => setCoverageOpen(false)}
          onBalance={balanceReady.ready ? () => setCurateOpen('balanced') : null}
          balanceReason={balanceReady.reason} />
      )}
      </ZoneSection>

      {/* ④ Promote — ship the kept set into a dataset, or clear rejects off
          disk. Same actions/handlers as before — just grouped as the last step. */}
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

      {filter.flag === 'dups' ? (
        <DupGroupsPanel bankId={bankId} live={live} kind="exact"
          onChanged={() => { refreshPayload(); refreshImages() }} />
      ) : filter.flag === 'semantic_dups' ? (
        <DupGroupsPanel bankId={bankId} live={live} kind="semantic"
          onChanged={() => { refreshPayload(); refreshImages() }} />
      ) : (
        <>
          <ul className={`grid gap-2 ${tileSize === 'S'
            ? 'grid-cols-4 sm:grid-cols-6 lg:grid-cols-8'
            : 'grid-cols-3 sm:grid-cols-4 lg:grid-cols-6'}`}>
            {page.images.map((img) => (
              <Tile key={img.id} img={img} bankId={bankId} size={tileSize}
                selected={selected.has(img.id)}
                onReview={() => openReview(img.id)}
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

      {launchOpen && (
        <LaunchAllDialog caps={caps} visionReady={visionReady}
          onClose={() => setLaunchOpen(false)} onLaunch={startPipeline} />
      )}

      {scoringPythonOpen && (
        <ScoringPythonDialog onClose={() => setScoringPythonOpen(false)}
          onChanged={async () => {
            // The pass reads bank_scoring_gpu_available(); force both probes so
            // the CPU warning and the "holds the GPU" tooltip agree at once.
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
