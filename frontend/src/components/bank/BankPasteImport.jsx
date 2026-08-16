import { useMemo, useState } from 'react'
import { HelpText } from '../common/HelpText'
// Explicit .js: Vite resolves an extensionless import, Node's ESM loader (which
// the render tests run under) does not.
import { parsePastedItems, PASTE_IMPORT_MAX } from './bankPasteParse.js'

/**
 * Paste a list of image links straight into a bank.
 *
 * The scan sources above enumerate a page from the server. Some sites cannot be
 * enumerated that way at all — their listings are drawn by JavaScript behind a
 * signed API, so the only place the image URLs exist in the clear is a browser
 * that is already logged in and already looking at them. For those, you collect
 * the links in your own browser and drop the result here.
 *
 * It is NOT a crawler and the copy says so: it imports exactly what you paste.
 * Hiding that would set up the wrong expectation the first time someone waits
 * for it to "keep going".
 *
 * Batching, the destination and every toast are the panel's, not this
 * component's — it hands `[{url, title}]` to the same `onImport` the scan
 * sources use, so a pasted import and a scanned one land identically.
 */
export default function BankPasteImport({ onImport, busy }) {
  const [text, setText] = useState('')
  const [sending, setSending] = useState(false)

  // Parsed on every keystroke so the count below is a live preview: the paste is
  // the step most likely to be wrong, and finding out after a 700-image queue
  // has started is too late.
  const parsed = useMemo(() => parsePastedItems(text), [text])
  const ready = !!parsed.items.length && !busy && !sending

  const submit = async () => {
    if (!ready) return
    setSending(true)
    try {
      const res = await onImport?.(parsed.items)
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
        rows={4}
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
          {parsed.items.length} image{parsed.items.length === 1 ? '' : 's'} ready
          {parsed.account ? ` from ${parsed.account}` : ''}
          {parsed.dropped ? ` — ${parsed.dropped} duplicate or unusable line(s) ignored` : ''}
        </span>
      </p>

      <div className="flex items-center gap-3">
        <button type="button" onClick={submit} disabled={!ready}
          className="rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50">
          <span hidden={sending}>Import into the bank</span>
          <span hidden={!sending}>Importing…</span>
        </button>
        <button type="button" onClick={() => setText('')} disabled={!text || sending}
          className="text-xs text-content-subtle transition-colors hover:text-content disabled:opacity-40">
          Clear
        </button>
      </div>

      <HelpText className="text-[0.6875rem] leading-relaxed text-content-subtle">
        This imports the links you paste and nothing else — it does not follow an
        account or fetch more later. Links from a signed CDN carry their signature
        in the query string; paste them whole, because trimming it makes them
        unreadable to the downloader. Up to {PASTE_IMPORT_MAX} per paste, sent in
        batches.
      </HelpText>
    </div>
  )
}
