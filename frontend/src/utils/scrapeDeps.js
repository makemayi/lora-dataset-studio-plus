/** Which scraper packages this machine is actually missing.
 *
 *  WHY THIS EXISTS
 *  ---------------
 *  The Concept Sources banner used to recite a hand-written list — "curl_cffi,
 *  gallery-dl, cloudscraper…" — while the backend probe watches seven modules
 *  (`probe_scrape_deps` in capabilities.py). Two of them, `ddgs` (the keyless
 *  web image search) and `yt_dlp` (the video path, launched as `python -m`),
 *  were added to the probe after the banner was written. A user flagged BECAUSE
 *  of those two read a warning naming neither, so the message could not explain
 *  why it was asking for a reinstall.
 *
 *  The fix is to stop keeping a second copy of the list: the probe already
 *  reports `detail` = "missing: a, b, c", and it is now published as
 *  `scrape_deps_detail`. This module parses it. Names are shown as the probe
 *  reports them (import names, e.g. `yt_dlp`) — the reinstall is a button, not
 *  a pip line the user retypes, so renaming them here would only re-introduce
 *  a list to maintain.
 *
 *  An older backend (or a caps payload that predates the field) sends nothing.
 *  In that case the banner names NO package rather than an out-of-date list:
 *  saying less is honest, saying the wrong three is not.
 */

const PREFIX = 'missing:'

/** The module names the probe says are absent, or [] when it did not say. */
export function missingScrapeDeps(detail) {
  if (typeof detail !== 'string') return []
  const i = detail.indexOf(PREFIX)
  if (i < 0) return []
  return detail
    .slice(i + PREFIX.length)
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
}

/** The banner's first sentence — with the real list when the backend gave one. */
export function scrapeDepsBanner(detail) {
  const missing = missingScrapeDeps(detail)
  return missing.length
    ? `⚠ Some optional scraper packages are missing on this machine: ${missing.join(', ')}.`
    : '⚠ The optional scraper packages are not installed.'
}
