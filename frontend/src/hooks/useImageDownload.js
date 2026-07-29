import { useCallback, useEffect, useRef, useState } from 'react';
import { imageDownloadUrl, nameFromDisposition } from '../utils/galleryDownload';

/* ⬇ Save ONE generated image, and say so when it cannot be saved.
 *
 * Deliberately NOT `<a href={img.url} download>`: that is one line and it lies.
 * The gallery routinely lists rows whose file has been cleaned off the disk (a
 * resume sets old saves aside, the Test Studio's trash sweeps), and an anchor
 * pointed at a missing file saves the 404 PAGE under a .png name — the user
 * finds out days later, by opening it. Fetching means the failure is a
 * sentence on screen instead of a corrupt file in Downloads.
 *
 * The name comes back on Content-Disposition. It is NOT rebuilt here: the
 * lineage scheme lives once, in services/gallery_download.py, where pytest can
 * throw hostile dataset names at it.
 *
 * No toast dependency on purpose — this hook is used from the canvas node and
 * from a viewer that also opens outside the board's toast provider, so the
 * error is returned and each host places it.
 */

export function useImageDownload() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const alive = useRef(true);
  // ⚠️ The `alive.current = true` is NOT redundant, and leaving it out cost an
  // afternoon. React's StrictMode mounts, unmounts and re-mounts every effect in
  // development, so the cleanup below runs once on a component that is very much
  // still there. Without the re-arm, `alive` is false for the rest of the
  // component's life and every `setBusy(false)` / `setError(...)` below is
  // skipped: the button says "Downloading…" forever and a refusal is never
  // shown — the exact silence this hook exists to prevent.
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  const download = useCallback(async (imageId) => {
    const url = imageDownloadUrl(imageId);
    if (!url || busy) return false;
    setBusy(true);
    setError('');
    try {
      const res = await fetch(url, { credentials: 'same-origin' });
      if (!res.ok) {
        let msg = 'That image could not be downloaded.';
        try { msg = (await res.json())?.error || msg; } catch { /* not JSON */ }
        throw new Error(msg);
      }
      const name = nameFromDisposition(res.headers.get('Content-Disposition'));
      const blob = await res.blob();
      const href = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = href;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      // Revoked on the next tick: revoking synchronously races the browser's
      // own read of the blob in Safari and the download silently produces
      // nothing at all.
      setTimeout(() => URL.revokeObjectURL(href), 0);
      return true;
    } catch (e) {
      if (alive.current) setError(e?.message || 'That image could not be downloaded.');
      return false;
    } finally {
      if (alive.current) setBusy(false);
    }
  }, [busy]);

  return { download, busy, error, clearError: () => setError('') };
}
