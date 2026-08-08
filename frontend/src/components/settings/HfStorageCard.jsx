import { useState } from 'react'
import { INPUT_CLASS, Card } from './primitives'
import { defaultValueAt } from './settingDefaults.js'
import { apiFetch, del, postJson } from '../../api/fetchClient'
import { deletionSafety, formatBytes, lastRunLabel, sortedCaches, storageSummary } from './hfStorage.js'
import { useToast } from '../common/Toast'

/* Hugging Face private storage — the wall run #146 hit at step 2750/3000, after
   hours of paid GPU, on a 403 "private repository storage limit reached". The
   space was taken by lds-base-* custom-base caches that nothing in the app ever
   listed. This card is where they become visible and deletable.

   It never fetches on mount: every button here talks to the Hub, and a Settings
   page that silently calls huggingface.co when you open it is not the deal. */
export default function HfStorageCard({ config, setField, configDefaults }) {
  // cloud.full_transformer is a NESTED block, which defaultValueAt (flat
  // section/key) cannot address — read it off the same server payload directly.
  // Still no literal default in this file: that rule is about the SOURCE of the
  // value, not the shape of the lookup.
  const dflt = (key) => ((defaultValueAt(configDefaults, 'cloud', 'full_transformer') || {})[key])
  const toast = useToast()
  const [data, setData] = useState(null)
  const [busy, setBusy] = useState(false)
  const [pending, setPending] = useState('')

  const load = async () => {
    setBusy(true)
    try {
      setData(await apiFetch('/api/cloud/hf-storage'))
    } catch (e) {
      setData({ ok: false, error: e.message || 'Could not reach Hugging Face' })
    } finally {
      setBusy(false)
    }
  }

  const removeOne = async (cache) => {
    const safety = deletionSafety(cache)
    if (!window.confirm(`Delete ${cache.name} (${formatBytes(cache.used_bytes)}) from Hugging Face?\n\n${safety.text}`)) return
    setPending(cache.name)
    try {
      await del(`/api/cloud/hf-storage/base/${encodeURIComponent(cache.name)}`)
      toast.success(`${cache.name} deleted — ${formatBytes(cache.used_bytes)} freed.`)
      await load()
    } catch (e) {
      toast.error(e.message || 'Delete failed')
    } finally {
      setPending('')
    }
  }

  const removeAll = async () => {
    const caches = sortedCaches(data)
    const risky = caches.filter((c) => !c.local_available)
    const total = caches.reduce((n, c) => n + (c.used_bytes || 0), 0)
    if (!caches.length) return
    if (!window.confirm(
      `Delete all ${caches.length} custom-base cache repo(s) (${formatBytes(total)})?\n\n`
      + (risky.length
        ? `${risky.length} of them have no local source left on this machine — for those, this is the last copy.`
        : 'Every one of them still has its local file here, so each is one re-upload away.'),
    )) return
    setBusy(true)
    try {
      const res = await postJson('/api/cloud/hf-storage/base/delete-all', {})
      if (res.failed?.length) toast.warning(`${res.deleted.length} deleted, ${res.failed.length} failed.`)
      else toast.success(`${res.deleted.length} cache repo(s) deleted — ${formatBytes(res.freed_bytes)} freed.`)
      await load()
    } catch (e) {
      toast.error(e.message || 'Delete failed')
    } finally {
      setBusy(false)
    }
  }

  const summary = data ? storageSummary(data) : null
  const caches = sortedCaches(data)
  const toneClass = { ok: 'text-emerald-300', blocked: 'text-rose-300', unknown: 'text-amber-300' }
  return (
    <Card
      title="Hugging Face storage"
      help="A full-model (dense) run now lands on this computer first and only backs its ~26 GB master up to a private Hugging Face repo afterwards, so a full allowance can no longer end a training — it costs the backup, and with it the ability to continue that model later. Custom training bases are cached here too. Check the space before a long run, and delete the caches you no longer need."
    >
      <div className="flex flex-wrap items-center gap-2">
        <button
          id="hf-storage-check"
          type="button"
          onClick={load}
          disabled={busy}
          className="rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content hover:bg-surface-raised disabled:opacity-50"
        >
          {busy ? 'Checking…' : data ? 'Re-check storage' : 'Check storage'}
        </button>
        <a
          href="https://huggingface.co/settings/storage"
          target="_blank"
          rel="noreferrer"
          className="text-xs font-medium text-sky-300 underline hover:text-sky-200"
        >
          Open Hugging Face storage ↗
        </a>
      </div>

      {summary && (
        <div className={`space-y-1 rounded-lg border border-border bg-surface-raised p-3 text-xs ${toneClass[summary.tone] || 'text-content-muted'}`}>
          {summary.lines.map((line) => (
            <p key={line} className="break-words">{line}</p>
          ))}
        </div>
      )}

      {data?.ok && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm font-medium text-content">
              Custom-base caches ({caches.length}
              {caches.length ? ` — ${formatBytes(data.cache_bytes)}` : ''})
            </p>
            {caches.length > 1 && (
              <button
                type="button"
                onClick={removeAll}
                disabled={busy}
                className="rounded-md border border-rose-500/40 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/10 disabled:opacity-50"
              >
                Delete all
              </button>
            )}
          </div>
          {!caches.length && (
            <p className="text-xs text-content-muted">
              No <code>lds-base-*</code> repo on this account. Custom bases are pushed there once
              and reused by every cloud run on the same weights.
            </p>
          )}
          {caches.map((cache) => {
            const safety = deletionSafety(cache)
            return (
              <div key={cache.name} className="rounded-lg border border-border bg-surface-raised p-3">
                {/* Column on a phone, row from sm up: the repo hash + a size + a
                    button never fit on 400 px in one line. */}
                <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                  <div className="min-w-0">
                    <p className="break-all text-sm font-medium text-content">{cache.name}</p>
                    <p className="mt-0.5 text-xs text-content-muted">
                      {formatBytes(cache.used_bytes)}
                      {cache.family ? ` · ${cache.family}` : ''}
                      {' · '}
                      <span className={safety.level === 'safe' ? 'text-emerald-300' : 'text-amber-300'}>
                        {safety.label}
                      </span>
                    </p>
                    {cache.local_path && (
                      <p className="mt-0.5 break-all text-[0.6875rem] text-content-subtle">{cache.local_path}</p>
                    )}
                    <p className="mt-0.5 text-[0.6875rem] text-content-subtle">{lastRunLabel(cache)}</p>
                    <p className="mt-0.5 text-[0.6875rem] text-content-subtle">{safety.text}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeOne(cache)}
                    disabled={busy || pending === cache.name}
                    className="shrink-0 self-start rounded-md border border-rose-500/40 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/10 disabled:opacity-50"
                  >
                    {pending === cache.name ? 'Deleting…' : 'Delete'}
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      <div>
        <label htmlFor="cloud-full-model-delivery" className="block text-sm font-medium text-content">
          Full-model delivery
        </label>
        <select
          id="cloud-full-model-delivery"
          value={config.cloud?.full_transformer?.delivery ?? dflt('delivery')}
          onChange={(e) => setField('cloud', 'full_transformer', {
            ...(config.cloud?.full_transformer || {}),
            delivery: e.target.value,
          })}
          className={INPUT_CLASS}
        >
          <option value="both">This computer, then a Hugging Face backup (recommended)</option>
          <option value="local">This computer only</option>
          <option value="hub">Hugging Face only (previous behaviour)</option>
        </select>
        <p className="mt-1 text-xs text-content-muted">
          Where a finished full model goes. The default downloads it here first and only
          then backs the 26 GB master up to the private repository — so a full Hugging Face
          quota can no longer end a training the way it did at step 2750 of 3000. Nothing is
          pushed while the run trains.
          {' '}
          <strong>The Hugging Face copy is what makes a run resumable</strong>: continuing a
          full model means putting its checkpoint on a fresh pod, and a 26 GB file can only
          get there from the Hub. “This computer only” saves your quota and gives that up.
          Checkpoints land in the folder set by Settings ▸ Storage ▸ Checkpoints, and a launch
          refuses (confirmably) when that drive plainly has no room.
        </p>
      </div>

      <div>
        <label htmlFor="cloud-private-storage-limit" className="block text-sm font-medium text-content">
          Private storage allowance (GB, 0 = infer from plan)
        </label>
        <input
          id="cloud-private-storage-limit"
          type="number"
          min="0"
          step="10"
          value={config.cloud?.full_transformer?.private_storage_limit_gb ?? dflt('private_storage_limit_gb')}
          onChange={(e) => setField('cloud', 'full_transformer', {
            ...(config.cloud?.full_transformer || {}),
            private_storage_limit_gb: parseInt(e.target.value, 10) || 0,
          })}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-xs text-content-muted">
          What the pre-check compares against before renting a pod for a full-model run.
          Hugging Face publishes no quota endpoint, so at 0 this falls back to the documented
          plan figure (100 GB free, 1 TB PRO) — which is a guess: the refusal that prompted
          this feature arrived well below it. Put your real ceiling here to make the check exact.
          The refusal is confirmable either way — “Train anyway” always exists.
        </p>
      </div>
    </Card>
  )
}
