/* Parsing for "paste a list of images" — the scraper's manual intake.
 *
 * WHY THIS EXISTS
 * ---------------
 * The scan sources in ConceptSourcesPanel enumerate a page for you. Some sites
 * cannot be enumerated from a server at all: their listings are rendered by
 * JavaScript behind a signed API, and the only place the image URLs exist in the
 * clear is a browser that is already logged in and already looking at them.
 * Douyin is the case that prompted this (its endpoints are signed, the signature
 * rotates, and every published library implementing it is long unmaintained).
 *
 * So the collecting happens in YOUR browser and the result arrives here as text.
 * This module owns only the text → items step, and it is deliberately a pure
 * function: the shape of what people paste is the part most likely to be wrong,
 * and it is the part worth testing without a DOM.
 *
 * FORGIVING ON PURPOSE
 * --------------------
 * Three shapes are accepted, because all three are what people actually end up
 * with and rejecting two of them teaches nothing:
 *   1. {items:[{url,title}]}   — what the helper snippets produce
 *   2. [{url,title}] or ["https://…"]  — a bare array
 *   3. one URL per line       — hand-assembled, or copied out of devtools
 * Anything that is not an http(s) URL is dropped rather than failing the paste:
 * a stray blank line or a trailing comma should not cost you a 700-image list.
 */

/** Upper bound on ONE paste. Not a server limit — `runBankScrapeImport` already
 *  batches to `BANK_SCRAPE_BATCH` — but a guard against pasting the wrong buffer
 *  entirely and queuing hours of downloads by accident. */
export const PASTE_IMPORT_MAX = 2000

const isHttp = (u) => typeof u === 'string' && /^https?:\/\/\S+$/i.test(u.trim())

function coerce(entry, index) {
  if (isHttp(entry)) return { url: entry.trim(), title: `Pasted ${index + 1}` }
  if (entry && typeof entry === 'object' && isHttp(entry.url)) {
    const title = typeof entry.title === 'string' && entry.title.trim()
      ? entry.title.trim().slice(0, 200)
      : `Pasted ${index + 1}`
    return { url: entry.url.trim(), title }
  }
  return null
}

/**
 * Parse pasted text into `[{url, title}]`.
 *
 * Returns `{items, error, dropped, account}`. `error` is set only when NOTHING
 * usable came out — a partially usable paste succeeds and reports `dropped`,
 * because the alternative is refusing 699 good URLs over one bad one.
 */
export function parsePastedItems(text) {
  const raw = typeof text === 'string' ? text.trim() : ''
  if (!raw) return { items: [], error: 'Nothing pasted yet.', dropped: 0, account: '' }

  let source = null
  let account = ''
  if (raw.startsWith('{') || raw.startsWith('[')) {
    let parsed
    try {
      parsed = JSON.parse(raw)
    } catch {
      return {
        items: [], dropped: 0, account: '',
        error: 'That looks like JSON but it will not parse — copy the whole output, including the outer braces.',
      }
    }
    if (Array.isArray(parsed)) {
      source = parsed
    } else if (parsed && Array.isArray(parsed.items)) {
      source = parsed.items
      account = typeof parsed.account === 'string' ? parsed.account.trim() : ''
    } else {
      return {
        items: [], dropped: 0, account: '',
        error: 'JSON parsed, but there is no list of images in it (expected an array, or an object with `items`).',
      }
    }
  } else {
    // Plain text: one URL per line. Commas are tolerated because a copied JS
    // array minus its brackets is a very common way to arrive here.
    source = raw.split(/[\r\n,]+/)
  }

  const seen = new Set()
  const items = []
  let dropped = 0
  for (let i = 0; i < source.length; i++) {
    const item = coerce(source[i], items.length)
    if (!item) { dropped++; continue }
    // De-dupe on the URL PATH: signed CDN links carry per-render query strings,
    // so the same image arrives several times under different signatures and a
    // full-URL comparison keeps every copy.
    let key = item.url
    try { const u = new URL(item.url); key = u.host + u.pathname } catch { /* keep raw */ }
    if (seen.has(key)) { dropped++; continue }
    seen.add(key)
    items.push(item)
    if (items.length >= PASTE_IMPORT_MAX) break
  }

  if (!items.length) {
    return { items: [], dropped, account, error: 'No http(s) image links found in what you pasted.' }
  }
  return { items, dropped, account, error: null }
}
