import { useEffect, useState } from 'react'
import { FLOAT_SHADOW, INPUT_CLASS } from '../common/surfaces'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import {
  canStartPromote, promoteButtonLabel, promoteSummary, weightNotice,
  splitTotal, rebalance, BUILD_FRAMINGS,
} from './bankPromote.js'

/** ⬆ Promote: copy the selection somewhere it can be worked on. TWO
 * destinations.
 *
 * • A DATASET — the original door. Goes through the normal import path (webp
 *   normalization + perceptual dedup vs the dataset).
 * • A NEW BANK — for isolating candidates out of a big dump (200 out of 9 000)
 *   and continuing to triage them apart, without committing them to a
 *   training container yet.
 *
 * Either way the bank KEEPS its images and marks them promoted; promotion
 * copies, and two banks never share a file. Which is why the new-bank door
 * states the measured weight before the click — images are a footnote, video is
 * not. */
export default function PromoteDialog({ bankId, selectedIds, onClose, onStarted }) {
  const toast = useToast()
  const [destination, setDestination] = useState('dataset')
  const [datasets, setDatasets] = useState(null)
  const [datasetId, setDatasetId] = useState('')
  const [bankName, setBankName] = useState('')
  const [promotable, setPromotable] = useState(null)
  const [size, setSize] = useState(null)
  const [busy, setBusy] = useState(false)
  // The build view of the framing decision: a TOTAL the operator gives, a split
  // the sliders nudge, and a plan the server computes before anything runs.
  // Reuse is conditional — while the pool covers the request, one picture one
  // framing; ask for more images than pictures and the best give a second and
  // third, never the same framing twice.
  const [total, setTotal] = useState(90)
  const [counts, setCounts] = useState(() => splitTotal(90))
  const [plan, setPlan] = useState(null)
  const [upscale, setUpscale] = useState(true)
  const useSelection = selectedIds.length > 0

  useEffect(() => {
    apiFetch('/api/dataset/list')
      .then((d) => setDatasets(d.datasets || []))
      .catch(() => setDatasets([]))
  }, [])

  // The kept-but-not-yet-on-THIS-dataset count is per-target (an image promoted
  // to another dataset still counts), so it can only be known once a target is
  // chosen. Fetch it then, so the copy line reflects what the server will do.
  useEffect(() => {
    if (useSelection || !datasetId) { setPromotable(null); return }
    let live = true
    setPromotable(null)
    apiFetch(`/api/bank/${bankId}/promotable?dataset_id=${Number(datasetId)}`)
      .then((d) => { if (live) setPromotable(d.count) })
      .catch(() => { if (live) setPromotable(null) })
    return () => { live = false }
  }, [bankId, datasetId, useSelection])

  // The PLAN the server computed, 300 ms after a slider settles. Never
  // recomputed here — two implementations of the quota rule would drift, and
  // the number on screen is a promise. The endpoint reads the whole promotable
  // set, so it is debounced.
  useEffect(() => {
    if (destination !== 'dataset' || !datasetId) { setPlan(null); return }
    let live = true
    const t = setTimeout(() => {
      postJson(`/api/bank/${bankId}/promote/plan`, {
        dataset_id: Number(datasetId), quotas: counts,
        ...(useSelection ? { image_ids: selectedIds } : {}),
      }).then((d) => { if (live) setPlan(d) })
        .catch(() => { if (live) setPlan(null) })
    }, 300)
    return () => { live = false; clearTimeout(t) }
  }, [bankId, datasetId, destination, counts, useSelection, selectedIds])

  // What the selection WEIGHS. Asked once, for the exact set the server would
  // copy — never estimated from an average, because the day a bank holds video
  // that average is wrong by three orders of magnitude.
  useEffect(() => {
    let live = true
    const qs = useSelection ? `?ids=${selectedIds.join(',')}` : ''
    apiFetch(`/api/bank/${bankId}/selection-size${qs}`)
      .then((d) => { if (live) setSize(d) })
      .catch(() => { if (live) setSize(null) })
    return () => { live = false }
  }, [bankId, useSelection, selectedIds.join(',')])

  const start = async () => {
    if (!canStartPromote({ destination, datasetId, bankName, busy })) return
    setBusy(true)
    try {
      if (destination === 'bank') {
        await postJson(`/api/bank/${bankId}/promote-to-bank`, {
          name: bankName.trim(),
          image_ids: useSelection ? selectedIds : [],
        })
        // The job runs on THIS bank (its rows are the ones being marked), so the
        // progress bar is here — say where the new bank turns up rather than
        // yanking the user off the page that is reporting the work.
        toast.success(`Copying into “${bankName.trim()}” — follow the progress bar. `
          + 'The new bank is in ← Banks once it finishes.', 9000)
      } else {
        await postJson(`/api/bank/${bankId}/build`, {
          dataset_id: Number(datasetId),
          quotas: counts,
          ...(upscale ? { upscale_below: 1536 } : {}),
          ...(useSelection ? { image_ids: selectedIds } : {}),
        })
        toast.success('Build started — follow the progress bar.')
      }
      onStarted?.()
    } catch (e) {
      toast.error(e?.message || 'Promotion failed to start.')
      setBusy(false)
    }
  }

  const toBank = destination === 'bank'
  const weight = weightNotice({ destination, size })
  const tab = (id, label) => (
    <button type="button" key={id} onClick={() => setDestination(id)}
      aria-pressed={destination === id}
      className={`flex-1 rounded-md border px-3 py-2 text-sm ${destination === id
        ? 'border-indigo-400 bg-indigo-100 font-semibold text-content'
        : 'border-border text-content-muted hover:bg-surface-raised'}`}>
      {label}
    </button>
  )

  return (
    <div role="dialog" aria-modal="true" aria-label="Promote the selection"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-3 sm:p-4">
      <div className="w-full max-w-md max-h-full overflow-y-auto rounded-xl bg-surface-overlay/85 backdrop-blur-md p-4 sm:p-5 shadow-2xl space-y-4">
        <h2 className="text-base font-bold text-content">⬆ Promote the selection</h2>

        <div>
          <p className="text-sm font-medium text-content">Send it to…</p>
          <div className="mt-1 flex flex-col gap-2 sm:flex-row">
            {tab('dataset', '📁 An existing dataset')}
            {tab('bank', '🗃 A new image bank')}
          </div>
        </div>

        <p className="text-sm text-content-muted">
          {promoteSummary({
            destination, useSelection, selectedCount: selectedIds.length,
            promotable, size, datasetChosen: !!datasetId,
          })}
        </p>

        {weight && (
          <p className="rounded-md bg-surface-raised px-3 py-2 text-xs text-content-muted">
            💾 {weight}
          </p>
        )}

        {!toBank && (
          <div className="flex flex-col gap-2 rounded-md bg-surface-raised px-3 py-2">
            <span className="block text-sm font-medium text-content">Framings</span>
            <p className="mt-1 text-xs text-content-muted">
              Tell it the TOTAL and how to split it. Each framing is cut from
              its own measured face box; pictures without a usable face go to
              the full frame. While your Bank has enough pictures each one is
              used once — ask for more images than you have pictures and the
              best ones give a second and third framing, never the same one
              twice.
            </p>

            <label className="mt-1 block text-xs font-medium text-content" htmlFor="build-total">
              Total images
            </label>
            <div className="flex items-center gap-2">
              <input id="build-total" type="range"
                min="0" max={(plan?.usable || 0) * 3 || 300} value={total}
                onChange={(e) => {
                  const n = Number(e.target.value)
                  setTotal(n); setCounts(splitTotal(n))
                }}
                className="flex-1 accent-primary" />
              <span className="w-10 text-right tabular-nums">{total}</span>
            </div>

            {BUILD_FRAMINGS.map((f) => (
              <label key={f} className="flex items-center gap-2 text-xs text-content-muted">
                <span className="w-12 capitalize">{f}</span>
                <input type="range" min="0" max={total} value={counts[f]}
                  onChange={(e) => setCounts(rebalance(counts, f, Number(e.target.value)))}
                  className="flex-1 accent-primary" />
                <span className="w-8 text-right tabular-nums">{counts[f]}</span>
              </label>
            ))}

            <label className="mt-1 flex items-center gap-2 text-xs text-content-muted">
              <input type="checkbox" checked={upscale}
                onChange={(e) => setUpscale(e.target.checked)} className="accent-primary" />
              Upscale what is too small (short edge &lt; 1536 px)
            </label>

            <BuildPlanPanel plan={plan} />
          </div>
        )}

        {toBank ? (
          <div>
            <label htmlFor="promote-bank-name" className="block text-sm font-medium text-content">
              Name of the new bank
            </label>
            <input id="promote-bank-name" type="text" value={bankName} autoFocus
              onChange={(e) => setBankName(e.target.value)}
              placeholder="Candidates"
              className="mt-1 w-full rounded-lg bg-surface-raised px-3 py-1.5 text-sm text-content focus:outline-none focus:ring-1 focus:ring-primary" />
            <p className="mt-1 text-xs text-content-subtle">
              The copies get a folder of their own inside the app's data, so this bank and the
              new one can be curated independently — neither ever touches the other's files, nor
              your original folder.
            </p>
          </div>
        ) : (
          <div>
            <label htmlFor="promote-dataset" className="block text-sm font-medium text-content">
              Target dataset
            </label>
            <select id="promote-dataset" value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              className="mt-1 w-full rounded-lg bg-surface-raised px-3 py-1.5 text-sm text-content focus:outline-none focus:ring-1 focus:ring-primary">
              <option value="">{datasets == null ? 'Loading…' : 'Choose a dataset…'}</option>
              {(datasets || []).map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.kind}, {d.images_total} image{d.images_total === 1 ? '' : 's'})
                </option>
              ))}
            </select>
            {datasets != null && datasets.length === 0 && (
              <p className="mt-1 text-xs text-amber-700">
                No dataset yet — create one on the Datasets page first, or send this selection to a
                new image bank instead.
              </p>
            )}
          </div>
        )}

        <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <button type="button" onClick={onClose}
            className={`rounded-full bg-surface-raised px-3 py-1.5 text-sm text-content ${FLOAT_SHADOW} transition-[box-shadow,transform,background-color] duration-200 hover:bg-surface hover:-translate-y-0.5`}>
            Cancel
          </button>
          <button type="button" onClick={start}
            disabled={!canStartPromote({ destination, datasetId, bankName, busy })}
            className="rounded-md bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
            {promoteButtonLabel({ destination, busy })}
          </button>
        </div>
      </div>
    </div>
  )
}


/* The plan the SERVER computed. Never recomputed here: two implementations of
   the quota rule would drift, and the number on screen is a promise. Both plan
   states stay mounted and flip with `hidden` — Chrome auto-translate rewrites
   text nodes, and a ternary that unmounts one is what becomes a removeChild
   crash. */
export function BuildPlanPanel({ plan }) {
  if (!plan) return null
  const short = Object.entries(plan.shortfall || {})
  return (
    <div className="flex flex-col gap-0.5 rounded-lg bg-surface px-3 py-2 text-[0.6875rem] text-content-muted">
      <span>{plan.usable} pictures usable ({plan.with_face_box} with a face)</span>
      <span className="text-content font-semibold tabular-nums">
        {plan.total} images — face {plan.counts.face || 0} · half {plan.counts.half || 0} · full {plan.counts.full || 0}
      </span>
      <span hidden={!plan.reused}>
        {plan.reused} pictures used more than once
      </span>
      <span hidden={!!plan.reused}>
        every image comes from its own picture
      </span>
      <span hidden={!short.length} className="text-amber-700">
        short: {short.map(([f, n]) => `${f} ${n}`).join(' · ')}
      </span>
    </div>
  )
}
