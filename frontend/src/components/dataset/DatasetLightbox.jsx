/**
 * Full-screen inspection lightbox (F3): toggle fit ↔ 100 % (native pixels) to
 * hunt skin/eyes artefacts before keeping an image. Esc, ✕ or a click on the
 * backdrop close it; a click on the image toggles the zoom mode.
 */
import { useEffect, useRef, useState } from 'react';
import SettingsLink from '../common/SettingsLink';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { displayLabel } from '../../utils/labels';
import PexelsAttribution from './PexelsAttribution';

const IMPROVE_HELP = 'Klein creates a new 2 MP version to validate and leaves the original intact.';
const COMPARE_HELP = 'Show the original this image was made from, next to it, at the same scale.';

/**
 * One half of the comparison. The two panes are cells of the SAME grid, so they
 * get identical boxes; `object-contain` then renders both images at the same
 * scale and the same framing whatever their pixel size — the improve pass
 * rescales to a megapixel budget and keeps the aspect ratio, so this is the only
 * reading where "it looks better" means something. Each side is named in text,
 * never by colour alone.
 */
function ComparePane({ label, url, alt, accent }) {
  return (
    <figure className="m-0 flex min-h-0 min-w-0 flex-col overflow-hidden rounded-lg border border-white/15">
      <figcaption className={`shrink-0 border-b border-white/10 bg-black/70 px-2 py-1 text-[11px] font-semibold ${
        accent ? 'text-indigo-200' : 'text-white/80'}`}>
        {label}
      </figcaption>
      <div className="min-h-0 flex-1 p-1">
        {/* h-full w-full, NOT max-h/max-w: an <img> left at its intrinsic size
            is capped by max-* but never scaled UP, so a 0.4 MP original
            rendered small next to a 2 MP result that filled its pane — the two
            were shown at different scales, which is precisely the comparison
            this mode must not produce. Filling the box and letting
            object-contain letterbox makes both fit the SAME box. */}
        <img src={url} alt={`${label} — ${alt}`}
          className="h-full w-full select-none object-contain" />
      </div>
    </figure>
  );
}

export default function DatasetLightbox({
  img,
  datasetId,
  nonce = 0,
  compare = null,
  parentNonce = 0,
  onClose,
  onCrop,
  onMirror,
  onRotate,
  onImprove,
  busy = false,
  mirrorBusy = false,
  improvePending = false,
  improveReady = false,
  kleinAvailable = false,
}) {
  const [full, setFull] = useState(false); // false = fit screen, true = 100 %
  const [comparing, setComparing] = useState(false);
  const [improving, setImproving] = useState(false);
  const dialogRef = useRef(null);
  const closeRef = useRef(null);

  // Focus trap keeps Tab inside the dialog (P2-7).
  useFocusTrap(dialogRef, !!(img && img.filename));

  // Keyboard support: Escape closes, initial focus on the close button.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  useEffect(() => { closeRef.current?.focus(); }, []);
  // Moving to another image must not leave the previous comparison open on a
  // parent that has nothing to do with it.
  useEffect(() => { setComparing(false); }, [img?.id]);

  if (!img || !img.filename) return null;
  const fileUrl = (filename, v) =>
    `/api/dataset/${datasetId}/img/${encodeURIComponent(filename)}${v ? `?v=${v}` : ''}`;
  const url = fileUrl(img.filename, nonce);
  const alt = displayLabel(img.variation_label) || 'dataset image';
  // A comparison is only ever entered when the original is actually renderable;
  // an unavailable one degrades to a stated reason, never a dead button.
  const canCompare = !!(compare && compare.available && compare.parent?.filename);
  const inCompare = canCompare && comparing;
  const improvementActive = improving || improvePending;
  const improveDisabled = busy || improvementActive || improveReady || !kleinAvailable;
  const improveTitle = !kleinAvailable
    ? `Klein is not available in this setup. ${IMPROVE_HELP}`
    : improveReady
      ? `A new version is waiting for validation. ${IMPROVE_HELP}`
    : improvePending
      ? `An improvement is already running for this image. ${IMPROVE_HELP}`
      : IMPROVE_HELP;

  const improve = async (event) => {
    event.stopPropagation();
    if (!onImprove || improveDisabled) return;
    setImproving(true);
    try {
      await onImprove(img.id);
    } finally {
      setImproving(false);
    }
  };

  const mirror = async (event) => {
    event.stopPropagation();
    if (!onMirror || busy || mirrorBusy) return;
    await onMirror(img.id);
  };

  // 🔄 Quarter turns (idea by 1Tomber, GitHub #17). `mirrorBusy` is the shared
  // "a pixel edit is running on this image" flag — both actions rewrite the same
  // file, so neither may start while the other is in flight.
  const rotate = (degrees) => async (event) => {
    event.stopPropagation();
    if (!onRotate || busy || mirrorBusy) return;
    await onRotate(img.id, degrees);
  };

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label={`Inspect — ${alt}`}
      className="fixed inset-0 z-[9996] bg-black/95 flex flex-col" onClick={onClose}>
      <button type="button" ref={closeRef}
        onClick={(e) => { e.stopPropagation(); onClose(); }}
        title="Close (Esc)" aria-label="Close inspection"
        className="absolute top-3 right-3 z-10 w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 text-white text-lg leading-none">✕</button>

      {inCompare ? (
        /* Side by side once there is room for it (≥640 px), stacked below —
           two 190 px-wide thumbnails on a phone would prove nothing, and a
           stacked pair keeps each image at full width where width is the scarce
           axis. Equal grid cells on both layouts = equal display scale. */
        <div onClick={(e) => e.stopPropagation()}
          className="flex-1 min-h-0 grid grid-rows-2 grid-cols-1 sm:grid-rows-1 sm:grid-cols-2 gap-2 p-2 sm:p-4">
          <ComparePane label={compare.beforeLabel} alt={alt}
            url={fileUrl(compare.parent.filename, parentNonce)} />
          <ComparePane label={compare.afterLabel} alt={alt} url={url} accent />
        </div>
      ) : full ? (
        <div className="flex-1 min-h-0 overflow-auto">
          <img src={url} alt={alt}
            onClick={(e) => { e.stopPropagation(); setFull(false); }}
            className="max-w-none cursor-zoom-out select-none" />
        </div>
      ) : (
        <div className="flex-1 min-h-0 flex items-center justify-center p-4">
          <img src={url} alt={alt}
            onClick={(e) => { e.stopPropagation(); setFull(true); }}
            className="max-h-full max-w-full object-contain cursor-zoom-in select-none" />
        </div>
      )}

      <div onClick={(e) => e.stopPropagation()}
        className="shrink-0 flex flex-wrap items-center justify-center gap-2 px-4 py-2.5 bg-black/60">
        <span className="text-white text-sm">{alt}</span>
        <span className="px-1.5 py-0.5 rounded text-[10px] bg-white/10 text-white/80">
          {img.source === 'import' ? 'real' : 'generated'}{img.framing ? ` · ${img.framing}` : ''}
        </span>
        <PexelsAttribution metadata={img.source_metadata}
          className="text-[11px] text-white/70" />
        <span className="text-white/50 text-[11px]">
          {inCompare
            /* Zoom is OFF here, and says so. At 100 % a 2 MP result and a 0.5 MP
               original cover different parts of the subject, which is exactly
               the dishonest comparison this mode exists to avoid. */
            ? 'same scale — exit comparison to zoom to 100 %'
            : full ? '100 % — click image to fit' : 'fitted — click image for 100 %'}
        </span>
        {canCompare && (
          <button type="button" aria-pressed={comparing}
            onClick={(e) => { e.stopPropagation(); setFull(false); setComparing((v) => !v); }}
            aria-label={comparing
              ? `Hide the original next to ${alt}`
              : `Show the original next to ${alt}`}
            title={COMPARE_HELP}
            className="min-h-9 w-full sm:w-auto px-3 py-1.5 rounded-lg border border-indigo-400/50 bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-100 text-xs font-semibold">
            {comparing ? '⊟ Exit comparison' : '⧉ Compare with original'}
          </button>
        )}
        {compare && !compare.available && (
          /* No affordance rather than a dead one — and it says why, because
             "there is no compare button here" is otherwise indistinguishable
             from a bug. */
          <span role="note"
            className="max-w-full break-words rounded-lg border border-amber-400/40 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-100">
            <span aria-hidden>⚠ </span>{compare.reason}
          </span>
        )}
        {onCrop && (
          <button type="button" onClick={() => onCrop(img)}
            title="Open the crop editor for this image (stretchable box, any ratio)"
            className="px-3 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-semibold">
            ✂ Crop
          </button>
        )}
        {onMirror && (
          <button type="button" onClick={mirror} disabled={busy || mirrorBusy}
            aria-busy={mirrorBusy}
            aria-label={mirrorBusy ? `Mirroring ${alt} horizontally` : `Mirror ${alt} horizontally`}
            title={mirrorBusy ? 'Mirroring horizontally…' : 'Mirror horizontally (flip left and right)'}
            className="min-h-9 w-full sm:w-auto px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-45">
            {mirrorBusy ? '⇆ Mirroring…' : '⇆ Mirror horizontally'}
          </button>
        )}
        {onRotate && (
          /* The pair shares ONE row even on a 400 px screen: two full-width
             rows for two halves of the same gesture would push everything else
             below the fold. Emoji stay aria-hidden — the label is the text. */
          <div className="flex w-full items-stretch gap-2 sm:w-auto">
            <button type="button" onClick={rotate(270)} disabled={busy || mirrorBusy}
              aria-busy={mirrorBusy} aria-label={`Rotate ${alt} 90 degrees left`}
              title="Rotate 90° left (counter-clockwise) — keeps the file's format; four turns come back round"
              className="min-h-9 flex-1 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-semibold text-white hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-45 sm:flex-none">
              <span aria-hidden="true">↺</span> Rotate left
            </button>
            <button type="button" onClick={rotate(90)} disabled={busy || mirrorBusy}
              aria-busy={mirrorBusy} aria-label={`Rotate ${alt} 90 degrees right`}
              title="Rotate 90° right (clockwise) — keeps the file's format; four turns come back round"
              className="min-h-9 flex-1 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-semibold text-white hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-45 sm:flex-none">
              <span aria-hidden="true">↻</span> Rotate right
            </button>
          </div>
        )}
        {onImprove && (
          <button type="button" onClick={improve} disabled={improveDisabled}
            aria-busy={improvementActive} title={improveTitle}
            className="min-h-9 w-full sm:w-auto px-3 py-1.5 rounded-lg border border-indigo-400/50 bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-100 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-45">
            {improveReady
              ? '✓ Review improvement first'
              : improvementActive ? '✨ Improving…' : '✨ Upscale & improve'}
          </button>
        )}
        {/* Its strength, step count and instruction are all editable, and nothing
            here said so — the reported case for making settings discoverable from
            where the action happens. `focus` lands on the strength block itself:
            a label promising one setting must not leave the reader hunting for it
            down a long section. */}
        {onImprove && !improvementActive && (
          <SettingsLink section="engines" focus="klein-improve-strength" className="self-center">
            Adjust improve strength
          </SettingsLink>
        )}
      </div>
    </div>
  );
}
