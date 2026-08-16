import { useEffect, useMemo, useState } from 'react'
import { HelpText } from '../common/HelpText'
import ScrapePickGrid from '../shared/ScrapePickGrid.jsx'
// Explicit .js: Vite resolves an extensionless import, Node's ESM loader (which
// the render tests run under) does not.
import { parsePastedItems, PASTE_IMPORT_MAX } from './bankPasteParse.js'

/**
 * Paste a list of image links, SEE them, then download the ones you want.
 *
 * The scan sources above enumerate a page from the server. Some sites cannot be
 * enumerated that way at all — their listings are drawn by JavaScript behind a
 * signed API, so the only place the image links exist in the clear is a browser
 * that is already logged in and already looking at them. For those, you collect
 * the links in your own browser and drop the result here.
 *
 * It shows the same pick-grid a scan does, and for the same reason: a list of
 * URLs tells you nothing about what you are about to download, and a link whose
 * signature has expired is indistinguishable from a good one until something
 * tries to fetch it. A thumbnail answers both — the picture, and whether it is
 * still reachable.
 *
 * It is NOT a crawler and the copy says so: it imports exactly what you tick.
 *
 * Batching, the destination and every toast are the panel's, not this
 * component's — it hands `[{url, title}]` to the same `onImport` the scan
 * sources use, so a pasted import and a scanned one land identically.
 */
export default function BankPasteImport({ onImport, busy }) {
  const [text, setText] = useState('')
  const [sending, setSending] = useState(false)
  const [selected, setSelected] = useState(() => new Set())
  const [broken, setBroken] = useState(() => new Set())

  // Parsed on every keystroke so the preview is live: the paste is the step
  // most likely to be wrong, and finding out after a 700-image queue has
  // started is too late.
  const parsed = useMemo(() => parsePastedItems(text), [text])

  // A new paste is a new pile: everything ticked, nothing remembered as dead.
  // Keyed on the URLs rather than the array identity, which useMemo makes new
  // on every keystroke — without that this would clear the selection while the
  // user is still typing.
  const key = parsed.items.map((i) => i.url).join('\n')
  useEffect(() => {
    setSelected(new Set(parsed.items.map((i) => i.url)))
    setBroken(new Set())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  const liveItems = parsed.items.filter((it) => !broken.has(it.url))
  const deadCount = parsed.items.length - liveItems.length
  const picked = liveItems.filter((it) => selected.has(it.url))
  const ready = !!picked.length && !busy && !sending

  const toggle = (u) => setSelected((prev) => {
    const next = new Set(prev)
    if (next.has(u)) next.delete(u); else next.add(u)
    return next
  })

  // A thumbnail that fails to load is a link that will fail to download too, so
  // it leaves the grid AND the selection — importing it would only produce a
  // skip the user then has to explain to themselves.
  const markBroken = (u) => {
    setBroken((prev) => new Set(prev).add(u))
    setSelected((prev) => {
      if (!prev.has(u)) return prev
      const next = new Set(prev); next.delete(u); return next
    })
  }

  const submit = async () => {
    if (!ready) return
    setSending(true)
    try {
      const res = await onImport?.(picked)
      // Clear only on success: a failed import leaves the list in place so it
      // can be retried without going back to the browser to collect it again.
      if (res?.ok) setText('')
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="flex flex-col gap-2 rounded-lg bg-surface-raised p-3" id="bank-paste-import">
      <span className="text-[0.6875rem] font-semibold uppercase tracking-wide text-content-subtle">
        Paste image links
      </span>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        aria-label="Image links to import into the bank"
        rows={3}
        spellCheck={false}
        placeholder={'One URL per line, or the JSON a collector snippet produced.'}
        className="w-full resize-y rounded-md border border-border-strong bg-surface px-3 py-2 font-mono text-xs text-content placeholder:text-content-subtle focus:border-primary focus:outline-none"
      />

      {/* Both states stay mounted and flip `hidden`: Chrome auto-translate
          rewrites text nodes, and swapping them with a ternary is what throws
          NotFoundError and takes the panel down (CLAUDE.md ▸ UI changes). */}
      <p className="text-xs text-content-muted" aria-live="polite">
        <span hidden={!parsed.error}>{parsed.error}</span>
        <span hidden={!!parsed.error}>
          {picked.length} of {liveItems.length} picked
          {parsed.account ? ` from ${parsed.account}` : ''}
          {parsed.dropped ? ` — ${parsed.dropped} duplicate or unusable line(s) ignored` : ''}
          {deadCount ? ` — ${deadCount} link(s) would not load and were dropped` : ''}
        </span>
      </p>

      <div hidden={!liveItems.length} className="flex flex-col gap-2">
        <div className="flex items-center gap-3 text-xs">
          <button type="button"
            onClick={() => setSelected(new Set(liveItems.map((i) => i.url)))}
            className="text-content-muted transition-colors hover:text-content">
            Select all
          </button>
          <button type="button" onClick={() => setSelected(new Set())}
            className="text-content-muted transition-colors hover:text-content">
            Select none
          </button>
        </div>
        <ScrapePickGrid
          items={liveItems}
          selected={selected}
          onToggle={toggle}
          onBroken={markBroken}
          ariaLabel="Pasted images"
        />
      </div>

      <div className="flex items-center gap-3">
        <button type="button" onClick={submit} disabled={!ready}
          className="rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50">
          <span hidden={sending}>Import {picked.length || ''} into the bank</span>
          <span hidden={!sending}>Importing…</span>
        </button>
        <button type="button" onClick={() => setText('')} disabled={!text || sending}
          className="text-xs text-content-subtle transition-colors hover:text-content disabled:opacity-40">
          Clear
        </button>
      </div>

      <HelpText className="text-[0.6875rem] leading-relaxed text-content-subtle">
        This imports the links you tick and nothing else — it does not follow an
        account or fetch more later. Links from a signed CDN carry their signature
        in the query string; paste them whole, because trimming it makes them
        unreadable to the downloader. A thumbnail that will not load is a link
        that has expired or was never an image, and it is dropped rather than
        queued. Up to {PASTE_IMPORT_MAX} per paste, sent in batches.
      </HelpText>
    </div>
  )
}
