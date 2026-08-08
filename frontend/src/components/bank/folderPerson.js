// 👤 "Single person here" — the wording of a folder-level person assertion.
//
// Pure on purpose (node --test cannot parse JSX): what this feature says is the
// part that has to be provable. Two sentences have to stay honest whatever the
// data does:
//   * an assertion is the USER'S WORD, never a measurement — the panel never
//     phrases it as something the app found out;
//   * a sample check looked at ~15 images, so it may only ever speak about the
//     sample. "sample consistent" is not "this folder is clean", and the copy
//     must not let one be read as the other.

export const SAMPLE_SIZE = 15

/** The assertion covering one subfolder, or null. '' (the bank root) is a real
 *  subfolder here as everywhere else, so the lookup compares against null. */
export function assertionFor(assertions, subfolder) {
  if (subfolder == null || !Array.isArray(assertions)) return null
  return assertions.find((a) => a.subfolder === subfolder) || null
}

export function folderLabel(subfolder) {
  return subfolder === '' ? 'the bank root' : subfolder
}

/** What the assertion IS, in one line. Always attributes it to the user. */
export function assertionSummary(entry) {
  if (!entry) return ''
  const n = entry.images || 0
  return `Person #${entry.cluster_id} · ${n} image${n === 1 ? '' : 's'} grouped by you, `
    + 'with no face pass'
}

/** 'ok' | 'warn' | 'muted' — the tone of a sample verdict. Unknown verdicts read
 *  as muted rather than reassuring: a shape we do not recognise is not good news. */
export function verdictTone(sample) {
  if (!sample) return 'muted'
  if (sample.verdict === 'consistent' || sample.verdict === 'partial') return 'ok'
  if (sample.verdict === 'mixed') return 'warn'
  return 'muted'
}

/** The verdict sentence, scoped to the sample it came from. Never states a fact
 *  about the folder — the check only ever saw ~15 of its images. */
export function verdictLine(sample) {
  if (!sample) return null
  const note = sample.note || ''
  if (sample.verdict === 'consistent' || sample.verdict === 'partial') return `✓ ${note}`
  if (sample.verdict === 'mixed') return `⚠ ${note}`
  return note ? `· ${note}` : null
}

/** What a sample check will cost, said before it is paid — the whole point of
 *  the feature is the number it is compared against. */
export function checkCostNote(entry) {
  const total = (entry && entry.images) || 0
  if (total <= SAMPLE_SIZE) {
    return `Embeds the ${total} image${total === 1 ? '' : 's'} of this folder.`
  }
  return `Embeds ${SAMPLE_SIZE} images spread across the folder — not the `
    + `${total} a full 👤 Group by person pass would.`
}

/** The "could not read a face here" list. It is a HEADS-UP, never an exclusion:
 *  those images stay in the group, and the sentence has to say so. */
export function toCheckNote(entry) {
  const n = (entry && entry.to_check && entry.to_check.length) || 0
  if (!n) return null
  return `${n} image${n === 1 ? '' : 's'} here showed no usable face `
    + '(no face, too small, or unreadable). Still in the group — worth a look.'
}

export function revokeNote(subfolder) {
  return `${folderLabel(subfolder)} goes back to normal clustering, and the `
    + 'group disappears. Nothing is deleted.'
}

// ── Automatic suggestions ───────────────────────────────────────────────────
// The app samples folders on its own and offers the ones that look like a
// single person. Everything below is wording for an OFFER: it must never read
// as something the app already did, and it must never be more certain than
// fifteen images allow. A stale probe is dropped rather than shown, because a
// suggestion about a folder that has since changed is worse than none.

/** The probe covering one subfolder, or null. Stale probes never surface. */
export function suggestionFor(suggestions, subfolder) {
  if (subfolder == null || !Array.isArray(suggestions)) return null
  const got = suggestions.find((s) => s.subfolder === subfolder)
  return got && !got.stale ? got : null
}

/** The subfolders currently being offered as "looks like one person". */
export function suggestedFolders(suggestions) {
  if (!Array.isArray(suggestions)) return []
  return suggestions
    .filter((s) => !s.stale && (s.verdict === 'consistent' || s.verdict === 'partial'))
    .map((s) => s.subfolder)
}

/** The offer, phrased as a question the user answers — never as a fact. */
export function suggestionLine(entry) {
  if (!entry) return null
  if (entry.verdict === 'consistent') {
    return `Looks like one person (${entry.largest}/${entry.scorable} of the `
      + `${entry.sample} sampled) — assert?`
  }
  if (entry.verdict === 'partial') {
    // Offered like the one above, but never dressed up as it: the number of
    // usable faces IS the caveat, so it leads.
    return `Looks like one person, on thin evidence — only ${entry.scorable} `
      + `usable face${entry.scorable === 1 ? '' : 's'} in ${entry.sample} images `
      + 'tried — assert?'
  }
  if (entry.verdict === 'mixed') {
    return `Sampled ${entry.faces} different faces here — probably not one person.`
  }
  // Not "the check could not tell": the folder has almost nothing a face
  // detector can read, and the full pass uses the same detector on the same
  // images. Saying otherwise would send the user to buy an answer that is not
  // for sale.
  return `Almost no readable face here — ${entry.scorable} in ${entry.sample} `
    + 'images tried. Face grouping has little to work with in this folder.'
}

export function suggestionTone(entry) {
  if (!entry) return 'muted'
  return (entry.verdict === 'consistent' || entry.verdict === 'partial') ? 'ok' : 'muted'
}

/** Marker appended to a folder's name in the picker, so the offer is visible
 *  without scoping every folder one by one. Deliberately a hint, not a claim. */
export function folderMarker(suggestions, subfolder) {
  const got = suggestionFor(suggestions, subfolder)
  const offered = got && (got.verdict === 'consistent' || got.verdict === 'partial')
  return offered ? ' · 👤?' : ''
}

/** What the "scan folders" button should say, and whether it should be there.
 *  It always states the cost BEFORE it is paid, and never promises to cover
 *  more folders than the server will.
 *
 *  It is now a SECONDARY path: 👤 Group by person (and 🚀 Launch all) run this
 *  same check by themselves at launch time and ask there, which is where the
 *  user actually is. The note says so, so this button reads as "ask me now"
 *  rather than as a step you were supposed to know about. */
export function scanOffer(payload) {
  const pending = (payload && payload.scannable) || 0
  if (!pending) return null
  const limit = (payload && payload.scan_limit) || pending
  const covered = Math.min(pending, limit)
  const size = (payload && payload.sample_size) || SAMPLE_SIZE
  // The ceiling belongs in the same sentence as the typical cost: a folder whose
  // draw comes back unreadable gets more images drawn to replace them, and a
  // number that only ever describes the easy case is not a cost estimate.
  const max = (payload && payload.sample_max) || 0
  const each = max > size ? `~${size} images each, up to ${max} in the hard ones`
    : `~${size} images each`
  let note = `Samples ${covered} folder${covered === 1 ? '' : 's'} `
    + `(${each}) and tells you which look like a single person.`
  if (pending > limit) {
    note += ` ${pending} are waiting — the biggest go first, run it again for the rest.`
  }
  note += ' It only suggests; you confirm. 👤 Group by person runs this check by '
    + 'itself when you launch it — this is for asking beforehand.'
  return { label: `🔎 Scan ${covered} folder${covered === 1 ? '' : 's'}`, note, pending }
}
