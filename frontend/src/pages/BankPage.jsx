import { useCallback, useEffect, useState } from 'react'
import { FLOAT_SHADOW } from '../components/common/surfaces'
import { apiFetch, del, postJson } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { HelpBadge } from '../help/HelpMode'
import BankWorkspace from '../components/bank/BankWorkspace'
import FolderPickerField from '../components/common/FolderPicker'
import { hiddenCount, previewSlots } from '../components/bank/bankPreview'
import { bankListSyncToast } from '../components/bank/bankSync'
import { overlapNotice } from '../components/bank/bankOverlap'
import { datasetFolderNotice } from '../utils/pathRelation'
import FolderSyncNote from '../components/bank/FolderSyncNote'
import FolderCheckLine from '../components/bank/FolderCheckLine'
import RelocateBankDialog from '../components/bank/RelocateBankDialog'
import BankScrapePanel from '../components/bank/BankScrapePanel'
import BankLaneTabs from '../components/videobank/BankLaneTabs'
import { bankListOverview } from '../components/bank/bankOverview.js'
import {
  CARD_SURFACE, CARD_SURFACE_INTERACTIVE, INPUT_CLASS, PRIMARY_BUTTON, QUIET_BUTTON,
} from '../components/common/surfaces'
import {
  MoveIcon, CloseIcon, PlusIcon, ArrowRightIcon, SpinnerIcon, BankIcon,
  ICON_BUTTON_QUIET,
} from '../components/common/icons'
import EmptyState from '../components/common/EmptyState'
import { HelpText } from '../components/common/HelpText'
import PageHeader from '../components/common/PageHeader'

const CURRENT_KEY = 'bankCurrentId'

/** The card's thumbnail strip: the bank's first few images, so a list of banks
 * reads at a glance instead of as a wall of folder paths. Clicking a thumbnail
 * opens the bank, like the title and the Open button. Thumbnails are served by
 * the same route the workspace grid uses (generated on demand when the bank was
 * never scanned) and load lazily, so an off-screen card costs nothing. */
function BankPreviewStrip({ bank, onOpen }) {
  if (!bank.preview_ids?.length) return null
  const extra = hiddenCount(bank.total, bank.preview_ids)
  return (
    <div className="relative grid grid-cols-5 gap-1">
      {previewSlots(bank.preview_ids).map((id, i) => (
        <div key={id ?? `empty-${i}`}
          className="aspect-square overflow-hidden rounded-md bg-surface-raised">
          {id != null && (
            <button type="button" onClick={onOpen} tabIndex={-1} aria-hidden="true"
              className="block h-full w-full">
              <img src={`/api/bank/${bank.id}/thumb/${id}`} alt="" loading="lazy"
                onError={(e) => { e.currentTarget.style.visibility = 'hidden' }}
                className="h-full w-full object-cover" />
            </button>
          )}
        </div>
      ))}
      {extra > 0 && (
        <span className="pointer-events-none absolute bottom-1 right-1 rounded bg-black/60 px-1 text-[0.625rem] font-semibold text-white">
          +{extra}
        </span>
      )}
    </div>
  )
}

const BANK_STATUS_TONE = {
  keep: 'bg-emerald-400', pending: 'bg-amber-300', reject: 'bg-rose-400',
}

function BankListSummary({ bank }) {
  const summary = bankListOverview(bank)
  return (
    <div className="space-y-1.5">
      {summary.total > 0 ? (
        <div className="flex h-2 overflow-hidden rounded-full bg-surface-raised" role="img"
          aria-label={summary.status.map((row) => `${row.label}: ${row.value}, ${row.percent}%`).join('; ')}>
          {summary.status.filter((row) => row.value > 0).map((row) => (
            <span key={row.id} className={BANK_STATUS_TONE[row.id]}
              style={{ width: `${row.widthPercent}%`, minWidth: '1px' }} />
          ))}
        </div>
      ) : summary.total === 0
        ? <p className="text-[11px] text-content-subtle">No images.</p>
        : <p className="text-[11px] text-amber-700/90">Curation totals unavailable.</p>}
      <ul className="flex flex-wrap gap-x-2 gap-y-0.5 text-[11px] text-content-muted">
        {summary.status.map((row) => (
          <li key={row.id} className="tabular-nums">{row.label} <span className="text-content">{row.value ?? '—'}</span>
            {row.percent != null && <span className="text-content-subtle"> · {row.percent}%</span>}</li>
        ))}
      </ul>
      <div className="flex items-center gap-2 text-[11px]">
        <span className="text-content-muted">Quality</span>
        <span className={summary.scanPercent == null ? 'text-amber-700/90' : 'text-content-subtle'}>
          {summary.scanText}
        </span>
      </div>
    </div>
  )
}

/** One bank in the list. Exported so a test can render a card without the page
 *  having to reach the server for the list first — the page renders "Loading…"
 *  until /api/banks answers, so nothing below it was ever executed by a test. */
export function BankCard({ bank, onOpen, onRelocate, onRemove }) {
  const b = bank
  return (
    <li className={`flex min-w-0 flex-col gap-2 p-3.5 ${CARD_SURFACE_INTERACTIVE}`}>
      <div className="flex min-w-0 items-center gap-1">
        <button type="button" onClick={onOpen}
          className="min-w-0 truncate text-left text-base font-semibold text-content hover:underline">
          {b.name}
        </button>
        {b.activity && !b.activity.finished && (
          <span className="inline-flex shrink-0 items-center gap-1 text-xs text-amber-700">
            <SpinnerIcon className="h-3.5 w-3.5 shrink-0" />{b.activity.kind}…
          </span>
        )}
        <button type="button" onClick={onRelocate}
          aria-label={`Move the folder of bank ${b.name}`}
          title="Moved this folder to another disk? Point the bank at its new location."
          className={`ml-auto grid h-7 w-7 shrink-0 place-items-center rounded-full transition-colors ${ICON_BUTTON_QUIET}`}>
          <MoveIcon />
        </button>
        <button type="button" onClick={onRemove} aria-label={`Remove bank ${b.name}`}
          className={`grid h-7 w-7 shrink-0 place-items-center rounded-full text-content-subtle ${FLOAT_SHADOW} transition-[box-shadow,transform,background-color] duration-200 hover:bg-surface hover:-translate-y-0.5 hover:text-rose-700`}>
          <CloseIcon />
        </button>
      </div>
      <p className="truncate font-mono text-xs text-content-subtle" title={b.source_path}>
        {b.source_path}
      </p>
      <BankPreviewStrip bank={b} onOpen={onOpen} />
      <BankListSummary bank={b} />
      <FolderSyncNote sync={b.folder_sync} onRelocate={onRelocate} />
      <button type="button" onClick={onOpen} className={`self-start ${QUIET_BUTTON}`}>
        Open <ArrowRightIcon className="h-3.5 w-3.5 shrink-0" />
      </button>
    </li>
  )
}

/** 🗃️ Image bank — triage a big unsorted folder BEFORE it becomes datasets.
 * List view (create/open/delete banks) + per-bank workspace. The bank
 * references the folder in place: nothing is copied until promotion, and the
 * source files are never modified. */
export default function BankPage() {
  const toast = useToast()
  const [banks, setBanks] = useState(null)
  const [currentId, setCurrentId] = useState(() => {
    try { return Number(localStorage.getItem(CURRENT_KEY)) || null } catch { return null }
  })
  const [name, setName] = useState('')
  const [folder, setFolder] = useState('')
  const [creating, setCreating] = useState(false)
  const [relocating, setRelocating] = useState(null)   // the bank being repointed
  // Dataset storage folders, so a folder that belongs to a dataset can be named
  // as such WHILE it is typed. The server refuses it either way — this only
  // spares the round-trip and the "why not?" (see utils/pathRelation.js).
  const [datasets, setDatasets] = useState([])

  // ⚠️ Plain loads do NOT re-walk the source folders any more: doing that cost a
  // full disk inventory of the whole library on every navigation to this page
  // (690-1 190 ms on a real 8-bank / 86 493-image library). `rescan` is the 🔄
  // button, and it is the only caller that asks the server to walk.
  const refresh = useCallback(async ({ rescan = false } = {}) => {
    try {
      const d = await apiFetch(`/api/banks${rescan ? '?rescan=1' : ''}`)
      setBanks(d.banks || [])
      if (!rescan) return
      // A walk just happened: say what it found, so the counters never move
      // without an explanation — and say so even when it found nothing, because
      // silence after a click reads as a broken button.
      const note = bankListSyncToast(d.banks)
      if (note) toast[note.type](note.text)
      else toast.success('Source folders checked — no new image found.')
    } catch (e) {
      toast.error(e?.message || 'Could not load the banks.')
      if (!rescan) setBanks([])
    }
  }, [toast])

  const [rescanning, setRescanning] = useState(false)
  const rescan = async () => {
    if (rescanning) return
    setRescanning(true)
    try { await refresh({ rescan: true }) } finally { setRescanning(false) }
  }

  useEffect(() => { if (currentId == null) refresh() }, [currentId, refresh])

  // Best effort: a failed list just means no live hint, never a broken form.
  useEffect(() => {
    if (currentId != null) return undefined
    let alive = true
    apiFetch('/api/dataset/list')
      .then((d) => { if (alive) setDatasets(d.datasets || []) })
      .catch(() => { if (alive) setDatasets([]) })
    return () => { alive = false }
  }, [currentId])

  const folderNotice = datasetFolderNotice(folder, datasets)

  const open = (id) => {
    try { localStorage.setItem(CURRENT_KEY, String(id)) } catch { /* ignore */ }
    setCurrentId(id)
  }
  const close = () => {
    try { localStorage.removeItem(CURRENT_KEY) } catch { /* ignore */ }
    setCurrentId(null)
  }

  const create = async (e) => {
    e.preventDefault()
    // A bank over a dataset's folder would share the dataset's LIVE files; the
    // server refuses it, and so does the form (the notice says what to do).
    if (creating || folderNotice) return
    setCreating(true)
    try {
      const d = await postJson('/api/bank/create', { name, folder })
      toast.success(`Bank created — ${d.added} image(s) inventoried.`)
      // Nested folders mean two banks over the same files. Harmless while
      // triaging, destructive at 🗑 Delete rejected — said once, up front.
      const overlap = overlapNotice(d.overlaps)
      if (overlap) toast.warning(overlap, 12000)
      setName(''); setFolder('')
      open(d.id)
    } catch (err) {
      toast.error(err?.message || 'Could not create the bank.')
    } finally {
      setCreating(false)
    }
  }

  const remove = async (bank) => {
    // eslint-disable-next-line no-alert
    if (!window.confirm(`Remove the bank “${bank.name}”?\n\nOnly the triage data (decisions, scores, thumbnails) is deleted — the source folder and its images are NOT touched.`)) return
    try {
      await del(`/api/bank/${bank.id}`)
      toast.success('Bank removed — source folder untouched.')
      refresh()
    } catch (e) {
      toast.error(e?.message || 'Could not remove the bank.')
    }
  }

  if (currentId != null) {
    return <BankWorkspace bankId={currentId} onBack={close} onGone={close} />
  }

  return (
    <div className="space-y-5">
      <PageHeader eyebrow="bank" title="Image bank" badge={<HelpBadge topic="page-bank" />}
        /* The kind of bank you are making, said WHERE you make one. Until now a
           .mp4 dropped in this folder was skipped in silence — this is the only
           place someone with a folder of rushes would ever have looked. */
        actions={<BankLaneTabs />}
        description={(
          /* Folded: it explains the page once, and after the first visit it is
             300 characters between the title and the thing you came to click. */
          <HelpText summary="What this page is for" className="text-sm text-content-muted">
            Point the app at a big unsorted folder (a Telegram export, a scrape dump…) and triage it
            into dataset-ready selections: a quality pass flags blur/noise/flat/small shots and groups
            near-duplicates, the face pass sorts the dump by person — then you promote the keepers
            into a dataset. The folder itself is never modified.
          </HelpText>
        )} />

      <form onSubmit={create}
        className={`flex flex-wrap items-end gap-3 p-3.5 ${CARD_SURFACE}`}>
        <div className="grow min-w-40">
          <label htmlFor="bank-name" className="block text-sm font-medium text-content">Name</label>
          <input id="bank-name" value={name} onChange={(e) => setName(e.target.value)}
            placeholder="Telegram export 07/2026" required className={INPUT_CLASS} />
        </div>
        <div className="grow-[3] min-w-64">
          <FolderPickerField id="bank-folder" label="Folder on this computer"
            value={folder} onChange={setFolder} required
            placeholder="C:\path\to\unsorted-images (subfolders included)" />
        </div>
        {/* Both labels mounted, one hidden: a ternary on the button's text is
            the shape Chrome auto-translate turns into a removeChild crash. */}
        <button type="submit" disabled={creating || !!folderNotice}
          title={folderNotice ? 'That folder belongs to a dataset' : undefined}
          className={PRIMARY_BUTTON}>
          <span hidden={!creating}>Inventorying…</span>
          <span hidden={!!creating} className="inline-flex items-center gap-1.5">
            <PlusIcon /> Create bank
          </span>
        </button>
        {/* basis-full: its own row inside the wrapping flex form, so the sentence
            never squeezes the fields — including at 400 px. */}
        {folderNotice && (
          <p role="alert"
            className="basis-full rounded-md border border-rose-500/70 bg-rose-500/15 p-3 text-sm text-rose-700">
            ⛔ {folderNotice.text}
          </p>
        )}
      </form>

      {/* Second way in: the scraper's own destination. A bank no longer needs a
          folder you prepared by hand — you can fill one straight from the web. */}
      <BankScrapePanel banks={banks} onDone={() => refresh()} />

      <FolderCheckLine banks={banks} busy={rescanning} onRescan={rescan} />

      {banks == null ? (
        <p className="text-sm text-content-muted">Loading…</p>
      ) : banks.length === 0 ? (
        <EmptyState icon={<BankIcon className="h-5 w-5" />} title="No bank yet"
          action={(
            <button type="button" className={PRIMARY_BUTTON}
              onClick={() => document.getElementById('bank-name')?.focus()}>
              <PlusIcon /> Create your first bank
            </button>
          )}>
          A bank points at a folder you already have — nothing is copied, and the
          folder is never modified. Name it and pick the folder above.
        </EmptyState>
      ) : (
        // grid-cols-1 (= minmax(0,1fr)), NOT the implicit auto column: an auto
        // column is sized on max-content, so the unbreakable source PATH inside
        // a card stretched it past the viewport and scrolled the whole page
        // sideways on a phone — with `truncate` never getting a chance to fire.
        <ul className="grid gap-3 grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {banks.map((b) => (
            <BankCard key={b.id} bank={b} onOpen={() => open(b.id)}
              onRelocate={() => setRelocating(b)} onRemove={() => remove(b)} />
          ))}
        </ul>
      )}

      {relocating && (
        <RelocateBankDialog bankId={relocating.id} bankName={relocating.name}
          sourcePath={relocating.source_path}
          onClose={() => setRelocating(null)} onDone={() => refresh()} />
      )}
    </div>
  )
}
