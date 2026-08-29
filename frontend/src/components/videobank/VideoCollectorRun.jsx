import { useEffect, useState } from 'react'
import { postJson } from '../../api/fetchClient'
import { HelpText } from '../common/HelpText'
import { CARD_SURFACE } from '../common/surfaces'

/**
 * 🧲 Run a local video collector — for accounts no scan can reach.
 *
 * The video lane's twin of the image bank's collector panel, with the smaller
 * contract the video architecture allows: the collector DOWNLOADS ITS OWN
 * files (the command reaches this bank's source folder through `{folder}`),
 * so the job is "run it, then refresh" — the refresh IS the import, and this
 * panel never sees a preview grid. What lands shows up as sources when the
 * bank re-inventories.
 *
 * NOTHING IS CONFIGURED BY DEFAULT, and that is the shipped state rather than a
 * missing step. A collector is pinned to one site's markup and breaks when that
 * site reskins, so the app ships the socket and no plug. With none configured
 * this says so plainly instead of offering a button that cannot work — the
 * limit is visible, which is the whole point of showing the block at all.
 *
 * The run is the bank's ONE job: a collector walking a whole account takes
 * minutes, the request returns as soon as it is launched, and the passes are
 * busy until it ends. You can close the page.
 *
 * `collectors` is injectable so a render test can mount the configured branch
 * without a server; production leaves it unset and the component asks.
 */
export default function VideoCollectorRun({ bankId, busy, collectors = null }) {
  const [list, setList] = useState(collectors)   // null = still asking
  const [picked, setPicked] = useState(() => collectors?.[0] || '')
  const [url, setUrl] = useState('')
  const [starting, setStarting] = useState(false)
  const [started, setStarted] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (collectors) return undefined             // injected: nothing to ask
    let alive = true
    fetch('/api/video-bank/collectors')
      .then((r) => (r.ok ? r.json() : { collectors: [] }))
      .then((d) => {
        if (!alive) return
        const got = Array.isArray(d.collectors) ? d.collectors : []
        setList(got)
        setPicked((p) => p || got[0] || '')
      })
      .catch(() => alive && setList([]))
    return () => { alive = false }
  }, [collectors])

  const configured = (list || []).length > 0
  const ready = configured && !!picked && /^https?:\/\/\S+$/i.test(url.trim())
    && !busy && !starting

  const run = async () => {
    if (!ready) return
    setError('')
    setStarting(true)
    try {
      await postJson(`/api/video-bank/${bankId}/collect`,
        { collector: picked, url: url.trim() })
      setUrl('')
      setStarted(true)
    } catch (e) {
      setError(String(e?.message || e) || 'Could not start the collector.')
    } finally {
      setStarting(false)
    }
  }

  return (
    <div className={`flex flex-col gap-2 p-3 ${CARD_SURFACE}`} id="video-collector-run">
      <span className="text-[0.6875rem] font-semibold uppercase tracking-wide text-content-subtle">
        Run a local collector
      </span>

      {/* Both states stay mounted and flip `hidden`: Chrome auto-translate
          rewrites text nodes, and swapping them with a ternary is what throws
          NotFoundError and takes the panel down (CLAUDE.md ▸ UI changes). */}
      <div hidden={configured}>
        <p className="text-xs text-content-muted">
          No collector is configured, so there is nothing to run here yet.
        </p>
        <HelpText className="mt-1 text-[0.6875rem] leading-relaxed text-content-subtle">
          A collector is a command on this machine that walks an account
          yourself — its own browser, your own signed-in session — and saves
          the videos it finds into a folder. The app ships none: each one is
          tied to a single site&apos;s markup and stops working when that site
          changes, so it belongs with you rather than in the app. Add one under
          <code className="mx-1 rounded bg-surface px-1">video_collectors</code>
          in your config — <code className="rounded bg-surface px-1">{'{url}'}</code>
          becomes the address you type below and
          <code className="mx-1 rounded bg-surface px-1">{'{folder}'}</code>
          this bank&apos;s folder — then reload.
        </HelpText>
      </div>

      <div hidden={!configured} className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <select value={picked} onChange={(e) => setPicked(e.target.value)}
            aria-label="Collector to run"
            className="rounded-md border border-border-strong bg-surface px-2 py-1.5 text-sm text-content focus:border-primary focus:outline-none">
            {(list || []).map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            aria-label="Account URL to collect from"
            placeholder="https://…  (the account or page to collect from)"
            className="min-w-0 flex-1 rounded-md border border-border-strong bg-surface px-3 py-1.5 text-sm text-content placeholder:text-content-subtle focus:border-primary focus:outline-none"
          />
          <button type="button" onClick={run} disabled={!ready}
            className="rounded-md bg-surface-raised px-3 py-1.5 text-sm font-semibold text-content transition-colors hover:bg-surface disabled:cursor-not-allowed disabled:opacity-50">
            <span hidden={starting}>Run</span>
            <span hidden={!starting}>Starting…</span>
          </button>
        </div>

        <p className="text-xs text-red-500" aria-live="polite" hidden={!error}>{error}</p>
        <p className="text-xs text-content-muted" aria-live="polite" hidden={!started}>
          Collector started — its progress shows above, and what it downloads
          appears here as sources as the bank re-inventories.
        </p>

        <HelpText className="text-[0.6875rem] leading-relaxed text-content-subtle">
          The collector may take several minutes — it is opening pages and
          saving videos, not calling an API. It runs as this bank&apos;s one
          job, so the passes are busy until it ends, and you can close this
          page. Nothing is filtered on the way in: what lands in the folder is
          what the bank holds.
        </HelpText>
      </div>
    </div>
  )
}
