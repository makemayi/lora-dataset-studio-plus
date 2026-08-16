import { useEffect, useState } from 'react'
import { HelpText } from '../common/HelpText'

/**
 * 🧲 Run a local collector — for galleries no scan can reach.
 *
 * The scan sources enumerate a page from this machine's server. Some sites
 * cannot be enumerated that way at all: they draw their listings in JavaScript
 * behind a signed API, so the image links exist in the clear only inside a
 * browser that is already logged in and looking at the page. A program on YOUR
 * machine can reach them; this runs it and imports what it reports.
 *
 * NOTHING IS CONFIGURED BY DEFAULT, and that is the shipped state rather than a
 * missing step. A collector is pinned to one site's markup and breaks when that
 * site reskins, so the app ships the socket and no plug. With none configured
 * this says so plainly instead of offering a button that cannot work — the
 * limit is visible, which is the whole point of showing the block at all.
 *
 * The run is a BANK JOB: a collector walking a whole account takes minutes, so
 * the request returns as soon as it is launched and the bank's own progress
 * carries the rest. You can close the page.
 */
export default function BankCollectorRun({ destinationOf, post, onDone, busy }) {
  const [collectors, setCollectors] = useState(null)   // null = still asking
  const [picked, setPicked] = useState('')
  const [url, setUrl] = useState('')
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let alive = true
    fetch('/api/bank/collectors')
      .then((r) => (r.ok ? r.json() : { collectors: [] }))
      .then((d) => {
        if (!alive) return
        const list = Array.isArray(d.collectors) ? d.collectors : []
        setCollectors(list)
        setPicked((p) => p || list[0] || '')
      })
      .catch(() => alive && setCollectors([]))
    return () => { alive = false }
  }, [])

  const configured = (collectors || []).length > 0
  const ready = configured && !!picked && /^https?:\/\/\S+$/i.test(url.trim())
    && !busy && !starting

  const run = async () => {
    if (!ready) return
    setError('')
    const destination = destinationOf?.()
    if (!destination) {
      setError('Choose or name the bank that will receive the images.')
      return
    }
    setStarting(true)
    try {
      const res = await post('/api/bank/collect',
        { collector: picked, url: url.trim(), ...destination })
      if (res?.error) { setError(res.error); return }
      setUrl('')
      await onDone?.(res)
    } catch (e) {
      setError(String(e?.message || e) || 'Could not start the collector.')
    } finally {
      setStarting(false)
    }
  }

  return (
    <div className="flex flex-col gap-2 rounded-lg bg-surface-raised p-3" id="bank-collector-run">
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
          A collector is a command on this machine that visits a page and prints
          the image links it finds. The app ships none: each one is tied to a
          single site&apos;s markup and stops working when that site changes, so
          it belongs with you rather than in the app. Add one under
          <code className="mx-1 rounded bg-surface px-1">bank.collectors</code>
          in your config, then reload.
        </HelpText>
      </div>

      <div hidden={!configured} className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <select value={picked} onChange={(e) => setPicked(e.target.value)}
            aria-label="Collector to run"
            className="rounded-md border border-border-strong bg-surface px-2 py-1.5 text-sm text-content focus:border-primary focus:outline-none">
            {(collectors || []).map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            aria-label="Page or account URL to collect from"
            placeholder="https://…  (the page or account to collect from)"
            className="min-w-0 flex-1 rounded-md border border-border-strong bg-surface px-3 py-1.5 text-sm text-content placeholder:text-content-subtle focus:border-primary focus:outline-none"
          />
          <button type="button" onClick={run} disabled={!ready}
            className="rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50">
            <span hidden={starting}>Run</span>
            <span hidden={!starting}>Starting…</span>
          </button>
        </div>

        <p className="text-xs text-red-500" aria-live="polite" hidden={!error}>{error}</p>

        <HelpText className="text-[0.6875rem] leading-relaxed text-content-subtle">
          The collector may take several minutes — it is opening pages, not
          calling an API. It runs as a bank job, so progress shows on the bank
          itself and you can close this page. What it collects is imported
          exactly like a scan: nothing is filtered on the way in.
        </HelpText>
      </div>
    </div>
  )
}
