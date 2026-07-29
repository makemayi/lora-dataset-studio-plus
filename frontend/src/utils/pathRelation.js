/* Do two folder paths point at the same files? — the browser-side twin of
 * `backend/app/services/path_guard.py`.
 *
 * A dataset and a bank must never share bytes on disk: images only ever TRANSIT
 * between them, by copy. The server is what enforces that (it can call
 * realpath, so it also sees through symlinks and NTFS junctions). This module
 * exists so the 🗃 bank creation form can say it WHILE the folder is being
 * typed, instead of letting the user press the button and take a 400.
 *
 * It is deliberately LEXICAL — a browser has no filesystem. So it is a hint,
 * never a guarantee: it catches the paste that actually happens (the path the
 * dataset itself displayed, in some spelling), and it stays silent on the ones
 * only the server can see. It must therefore never be the only check anywhere.
 *
 * Kept free of JSX so `node --test` can run it.
 */

/** One canonical spelling: quotes stripped (Windows «Copy as path» pastes them),
 *  separators unified, `.`/`..` resolved, trailing separators dropped.
 *  `caseInsensitive` folds case — true for Windows-style paths, where the
 *  filesystem folds it too. Returns '' when there is nothing to compare. */
export function normalizePath(input, { caseInsensitive } = {}) {
  let raw = String(input ?? '').trim()
  if (raw.length >= 2 && ((raw[0] === '"' && raw.at(-1) === '"')
    || (raw[0] === "'" && raw.at(-1) === "'"))) raw = raw.slice(1, -1).trim()
  if (!raw) return ''
  const windows = caseInsensitive ?? looksWindows(raw)
  let s = windows ? raw.replace(/\\/g, '/') : raw
  const rooted = s.startsWith('/')
  const out = []
  for (const part of s.split('/')) {
    if (!part || part === '.') continue
    // `..` climbs, but never past the root — `/..` is `/`, not the parent of it.
    if (part === '..') {
      if (out.length && out.at(-1) !== '..') out.pop()
      else if (!rooted) out.push('..')
      continue
    }
    out.push(part)
  }
  s = (rooted ? '/' : '') + out.join('/')
  s = s.length > 1 ? s.replace(/\/+$/, '') : s
  return windows ? s.toLowerCase() : s
}

/** A drive letter or a UNC share is the only reliable tell that a string is a
 *  Windows path — a backslash alone is a legal character in a POSIX filename. */
export function looksWindows(p) {
  const s = String(p ?? '')
  return /^[a-zA-Z]:[\\/]/.test(s) || s.startsWith('\\\\')
}

/** How folder `a` stands to folder `b`: 'same', 'inside' (a is under b),
 *  'contains' (a holds b), or null when the two are disjoint.
 *
 *  Both directions matter: a bank INSIDE a dataset's folder lists its files, and
 *  a bank that CONTAINS it walks recursively and lists them too. */
export function pathRelation(a, b, opts) {
  const caseInsensitive = opts?.caseInsensitive ?? (looksWindows(a) || looksWindows(b))
  const na = normalizePath(a, { caseInsensitive })
  const nb = normalizePath(b, { caseInsensitive })
  if (!na || !nb) return null
  if (na === nb) return 'same'
  // On a separator boundary only: `…/data2` is NOT inside `…/data`.
  if (na.startsWith(`${nb}/`)) return 'inside'
  if (nb.startsWith(`${na}/`)) return 'contains'
  return null
}

/** The live notice under the bank's folder field: is the typed folder one a
 *  dataset owns? `datasets` is [{id, name, storage_path}] (as served by
 *  /api/dataset/list). Returns {datasetId, name, relation, text} or null.
 *
 *  The text names the alternative on purpose. A refusal that only says no turns
 *  a trap into a wall — and "🗃 Import to bank" already does the right thing:
 *  it copies. */
export function datasetFolderNotice(folder, datasets) {
  if (!String(folder ?? '').trim()) return null
  for (const d of datasets || []) {
    if (!d?.storage_path) continue
    const rel = pathRelation(folder, d.storage_path)
    if (!rel) continue
    const who = d.name ? `“${d.name}”` : `#${d.id}`
    const what = rel === 'contains'
      ? `That folder contains the image folder of dataset ${who}.`
      : `That folder belongs to dataset ${who} — it is where the app stores its images.`
    return {
      datasetId: d.id ?? null,
      name: d.name || '',
      relation: rel,
      text: `${what} A bank and a dataset must never share files: a bank points at a `
        + 'LIVE folder, so 🗑 Delete rejected here would delete the dataset’s images. '
        + 'Open the dataset and use 🗃 Import to bank instead — it copies them into a '
        + 'bank of their own.',
    }
  }
  return null
}
