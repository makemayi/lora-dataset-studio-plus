// "Is it downloading, or is it frozen?" — the question a rented pod could not
// answer. While a run fetches its base weights (26 GB for Krea) the card showed
// one fixed sentence, `running:   - fetching transformer weights`, for as long
// as it took; two people waited hours because a stuck pod and a healthy one
// looked identical. The pod's own log has always carried the answer
// (`raw.safetensors: 7%|▋| 1.95G/26.3G [15:30<2:37:06, 2.58MB/s]`) and the
// backend now forwards it as `download`.
//
// The numbers are passed through as strings, exactly as the log printed them —
// no unit maths here, because '1.95G' is a rendering choice of the producer and
// re-deriving bytes from it would invent a figure nobody measured.
//
// Nothing here is required: `download` is null whenever no byte bar could be
// parsed (a phase without one, or a tqdm format that changed), and the caller
// keeps showing the phase sentence it always showed.

const MODEL_FILE = /\.(safetensors|gguf|bin|ckpt|pth|pt)$/i;

/* A tqdm bar is labelled with whatever the producing library passed as `desc`:
   a filename for huggingface_hub, a sentence for ai-toolkit's own bars. Turn it
   into something a user can read, and never let an unbounded third-party string
   run across the card. */
export function downloadLabel(raw) {
  const label = String(raw || '').trim();
  if (!label) return 'Downloading';
  if (MODEL_FILE.test(label)) return 'Fetching model weights';
  const clean = label.replace(/\s+/g, ' ').replace(/[.…\s]+$/, '');
  if (/^download/i.test(clean)) return clean.slice(0, 48);
  return `Downloading ${clean}`.slice(0, 48);
}

/* Backend `download` payload -> what the card renders, or null.
   `aria` is quantised to whole percent on purpose: a progress bar that
   re-announces itself on every byte is unusable with a screen reader, so the
   element carries a stable label plus aria-valuenow, and never aria-live. */
export function formatDownloadProgress(download) {
  if (!download || typeof download !== 'object') return null;
  const { done, total } = download;
  if (!done || !total) return null;
  const percent = Number.isFinite(Number(download.percent))
    ? Math.max(0, Math.min(100, Math.round(Number(download.percent))))
    : null;
  const label = downloadLabel(download.label);
  const size = `${done} of ${total}`;
  const headline = percent == null ? `${label} — ${size}` : `${label} — ${size} (${percent}%)`;
  const detail = [
    download.speed || null,
    download.eta ? `ETA ${download.eta}` : null,
    download.elapsed ? `${download.elapsed} elapsed` : null,
  ].filter(Boolean).join(' · ');
  return {
    label,
    percent,
    size,
    headline,
    // Empty right after the bar appears (tqdm prints '?' for both rate and ETA
    // until it has an estimate) — the caller drops the line rather than showing
    // a row of placeholders.
    detail,
    aria: percent == null ? `${label}, ${size}` : `${label}, ${percent}%, ${size}`,
  };
}
