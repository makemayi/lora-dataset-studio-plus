/* Pure helpers for the mode-aware Updates card. Kept out of the component so the
   mode/label/progress logic is unit-testable without a DOM (see updateStatus.test.js).

   Install modes:
   - 'pinokio'     launched by the Pinokio launcher: the tree is a git checkout
                   and updates the same way, but the RESTART belongs to Pinokio.
                   Updating here would relaunch the server detached from the
                   launcher, which would then show the app as stopped while an
                   untracked one held the port.
   - 'git'         a git checkout: "Update & restart" fast-forwards (unchanged).
   - 'zip'         a packaged install whose latest release ships a ZIP asset: the
                   button downloads + swaps the release, with a progress bar.
   - 'unavailable' non-git and no downloadable release: don't promise an update
                   the app can't perform — link out to the releases page instead. */

// Pinokio's own three clicks. Not shell commands — the user never opens a
// terminal in this install shape, so naming `git pull` here would send them
// looking for one. Pinokio's Update tab runs the same `git pull --ff-only`.
export const PINOKIO_UPDATE_STEPS = Object.freeze([
  'Stop the app in Pinokio',
  'Click the Update tab',
  'Click Start',
])

export const PINOKIO_UPDATE_GUIDE_URL =
  'https://github.com/perfectgf/lora-dataset-studio#option-5--pinokio-one-click-any-os'

export function isPinokioInstall(s) {
  return s?.install_mode === 'pinokio'
}

export function formatMB(bytes) {
  if (!bytes || bytes <= 0) return ''
  const mb = bytes / 1e6
  return `${mb >= 100 ? Math.round(mb) : mb.toFixed(1)} MB`
}

export function installMode(s) {
  if (!s) return 'unknown'
  // Before the is_git branch ON PURPOSE: a Pinokio install IS a git checkout,
  // and answering 'git' would put back the button that strands the launcher.
  if (isPinokioInstall(s)) return 'pinokio'
  if (s.ok === false) return 'unknown'
  if (s.is_git) return 'git'
  if (s.can_apply) return 'zip'
  return 'unavailable'
}

/* Headline for a ZIP-mode update, e.g. "Update to v2026.07.19 (download ~42 MB)".
   The size hint is omitted when the release didn't report an asset size. */
export function zipUpdateHeadline(s) {
  const v = s && s.latest ? `Update to v${s.latest}` : 'Update available'
  const size = s && s.zip_size ? formatMB(s.zip_size) : ''
  return size ? `${v} (download ~${size})` : v
}

/* Percent complete for the download phase, or null when the total is unknown
   (server sent no Content-Length) — the UI then shows an indeterminate bar. */
export function progressPercent(p) {
  if (!p || !p.total || p.total <= 0) return null
  return Math.max(0, Math.min(100, Math.round((p.downloaded || 0) / p.total * 100)))
}

/* Human phase line for the progress area. Returns null for phases the card
   renders elsewhere (idle/done) so the caller can branch on that. */
export function progressLabel(p) {
  const phase = p && p.phase
  if (phase === 'downloading') {
    const pct = progressPercent(p)
    const dl = formatMB(p.downloaded)
    if (p.total) return `⬇ Downloading… ${pct == null ? '' : `${pct}% `}(${dl || '0 MB'} / ${formatMB(p.total)})`
    return `⬇ Downloading… ${dl || ''}`.trim()
  }
  if (phase === 'extracting') return '📦 Extracting the update…'
  if (phase === 'installing') return '🔧 Installing the new files…'
  if (phase === 'restarting') return '↻ Restarting the app…'
  return null
}
