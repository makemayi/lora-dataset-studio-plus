import { useRef, useState, useEffect } from 'react';
import IdentityPromptModal from './IdentityPromptModal';
import PoseSlotPanel from './PoseSlotPanel';
// Engine names come from the derived edit list — spelling them out here is how
// this tooltip ended up naming two engines while a third could already edit.
import { editEngineNames } from './referenceEdit';
import { imageFromClipboard } from './clipboardImage';

// Cap identique à MAX_EXTRA_REFS côté backend (face_dataset_service).
const MAX_EXTRA_REFS = 3;

export default function ReferencePanel({ refFilename, datasetId, onSetRef, onCropRef, onEditRef, busy,
                                         importBusy = busy, visionBusy = false, nonce = 0,
                                         extraRefs = [], onAddExtraRef, onRemoveExtraRef,
                                         onCropExtraRef, subjectType = 'human',
                                         poseSlots = {}, onSetPoseSlot, onCropPoseSlot,
                                         onMirrorPoseSlot, onTogglePoseSlotEnabled,
                                         onRemovePoseSlot }) {
  const inp = useRef(null);
  const inpExtra = useRef(null);
  // Auto head-crop = OPT-IN (vision pass, pauses ComfyUI). Default OFF: upload is
  // instant (centered square) and ✂ Crop adjusts manually — faster in practice.
  const [autoCrop, setAutoCrop] = useState(false);
  // ✎ next to the Extra-refs "+": the identity instruction those extra photos
  // ride on is a GLOBAL setting buried in Settings ▸ Image engines — reachable
  // here, in the one place where the user is thinking about identity locking.
  const [promptModal, setPromptModal] = useState(false);
  const imgUrl = (fn) => `/api/dataset/${datasetId}/img/${encodeURIComponent(fn)}${nonce ? `?v=${nonce}` : ''}`;

  // Paste-to-upload: hover the reference tile, Ctrl+V an image instead of
  // always opening the file picker. Hover state lives in a ref (not React
  // state) since it only needs to be read inside the paste handler, never
  // rendered. `refHover`/`extraHover` are mutually exclusive — the mouse can
  // only be over one tile at a time — so a single document listener with two
  // hover flags never double-fires.
  const refHover = useRef(false);
  const extraHover = useRef(false);
  useEffect(() => {
    const onPaste = (e) => {
      if (!refHover.current && !extraHover.current) return;
      const file = imageFromClipboard(e);
      if (!file) return;
      e.preventDefault();
      if (refHover.current) {
        if (!importBusy) onSetRef(file, { autoCrop: autoCrop && !visionBusy });
      } else if (extraHover.current && extraRefs.length < MAX_EXTRA_REFS && !importBusy) {
        onAddExtraRef?.(file);
      }
    };
    document.addEventListener('paste', onPaste);
    return () => document.removeEventListener('paste', onPaste);
  }, [onSetRef, onAddExtraRef, autoCrop, visionBusy, importBusy, extraRefs.length]);

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-3">
      <div className="flex items-center gap-3">
        <div className="w-20 h-20 rounded-lg bg-black overflow-hidden shrink-0 flex items-center justify-center"
          onMouseEnter={() => { refHover.current = true; }} onMouseLeave={() => { refHover.current = false; }}
          title="Hover and press Ctrl+V to paste an image here">
          {refFilename
            ? <img src={imgUrl(refFilename)} alt="ref" className="w-full h-full object-cover" />
            : <span className="text-content-subtle text-xs">none</span>}
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-content text-sm font-medium">Reference photo</span>
          <span className="text-content-subtle text-[0.6875rem]">source of Klein variations — crop with ✂ after upload</span>
          <div className="flex gap-1.5 items-center flex-wrap">
            <button type="button" onClick={() => inp.current?.click()} disabled={importBusy}
              className="px-2.5 py-1 rounded-lg bg-surface-raised text-content text-xs disabled:opacity-40">
              {refFilename ? 'Change' : 'Set'} reference
            </button>
            {refFilename && (
              <button type="button" onClick={onCropRef} disabled={busy}
                className="px-2.5 py-1 rounded-lg bg-surface-raised text-content text-xs disabled:opacity-40">✂ Crop</button>
            )}
            {refFilename && onEditRef && (
              <button type="button" onClick={onEditRef} disabled={busy}
                title={`Edit the reference with a prompt (${editEngineNames()}) — compare before/after, then Keep or Discard`}
                className="px-2.5 py-1 rounded-lg bg-surface-raised text-content text-xs disabled:opacity-40">✦ Edit</button>
            )}
            <label className="flex items-center gap-1 text-[0.625rem] text-content-muted cursor-pointer"
              title={visionBusy ? 'Auto head-crop is unavailable during local generation; the reference imports with a centered crop.' : 'ON: a vision pass finds the head and crops around it (slower, pauses ComfyUI). OFF (default): instant centered square — adjust with ✂ Crop, usually faster.'}>
              <input type="checkbox" checked={autoCrop} disabled={visionBusy} onChange={(e) => setAutoCrop(e.target.checked)}
                className="accent-indigo-500 w-3 h-3" />
              ✂ Auto head-crop{visionBusy ? ' — unavailable during local generation' : ''}
            </label>
          </div>
          <input ref={inp} type="file" accept="image/*" className="hidden" disabled={importBusy}
            onChange={(e) => { if (e.target.files[0]) onSetRef(e.target.files[0], { autoCrop: autoCrop && !visionBusy }); e.target.value = ''; }} />
        </div>
      </div>

      {refFilename && (
        <PoseSlotPanel datasetId={datasetId} poseSlots={poseSlots} busy={busy}
          importBusy={importBusy} nonce={nonce} onSetPoseSlot={onSetPoseSlot}
          onCropPoseSlot={onCropPoseSlot} onMirrorPoseSlot={onMirrorPoseSlot}
          onTogglePoseSlotEnabled={onTogglePoseSlotEnabled} onRemovePoseSlot={onRemovePoseSlot} />
      )}
      {/* Références additionnelles — identité multi-angles, consommées par TOUS
          les moteurs : Nano Banana & ChatGPT (jointes à l'appel API) et Klein
          (chaînées en ReferenceLatent natifs). Recadrables une par une (✂ sur la
          vignette) ; le scoring reste sur la principale. */}
      {refFilename && (
        <div className="flex items-center gap-2 flex-wrap border-t border-border pt-2"
          onMouseEnter={() => { extraHover.current = true; }} onMouseLeave={() => { extraHover.current = false; }}
          title="Hover and press Ctrl+V to paste a new extra reference">
          <span className="text-content-subtle text-[0.6875rem]">
            Extra refs <span className="opacity-70">(all engines — stronger identity lock)</span>
          </span>
          {extraRefs.map((fn) => (
            <div key={fn} className="relative w-12 h-12 rounded-lg overflow-hidden bg-black shrink-0">
              <img src={imgUrl(fn)} alt="extra reference" className="w-full h-full object-cover" />
              <button type="button" onClick={() => onRemoveExtraRef?.(fn)} disabled={busy}
                aria-label="Remove this extra reference"
                title="Remove this extra reference"
                className="absolute top-0 right-0 w-4 h-4 flex items-center justify-center rounded-bl bg-black/70 text-white text-[0.625rem] leading-none disabled:opacity-40">
                ✕
              </button>
              {/* ✂ in the OPPOSITE corner of ✕: the tile is 48 px, two 16 px targets
                  diagonally apart never overlap and stay reachable. */}
              <button type="button" onClick={() => onCropExtraRef?.(fn)} disabled={busy}
                aria-label="Crop this extra reference"
                title="Crop this extra reference — the full frame stays kept, so you can widen it back out later"
                className="absolute bottom-0 left-0 w-4 h-4 flex items-center justify-center rounded-tr bg-black/70 text-white text-[0.625rem] leading-none disabled:opacity-40">
                ✂
              </button>
            </div>
          ))}
          {extraRefs.length < MAX_EXTRA_REFS && (
            <button type="button" onClick={() => inpExtra.current?.click()} disabled={importBusy}
              aria-label="Add an extra reference photo (other angles of the same face)"
              title="Add an extra reference photo — every engine (Nano Banana, ChatGPT, Klein) uses them together to lock the identity"
              className="w-12 h-12 rounded-lg border border-dashed border-border-strong text-content-muted text-lg leading-none disabled:opacity-40">
              +
            </button>
          )}
          <button type="button" onClick={() => setPromptModal(true)}
            aria-label="Edit the identity instruction used with multiple references"
            title="Edit the identity instruction sent with multiple references — one box per engine family, for this dataset's subject type"
            className="w-6 h-6 rounded-lg border border-border-strong text-content-muted text-xs leading-none hover:bg-surface-raised">
            ✎
          </button>
          <input ref={inpExtra} type="file" accept="image/*" className="hidden" disabled={importBusy}
            onChange={(e) => { if (e.target.files[0]) onAddExtraRef?.(e.target.files[0]); e.target.value = ''; }} />
        </div>
      )}
      {/* The modal edits the prompts of THIS dataset's subject — a human lock
          shown on an Animal dataset is what made an animal-tuned text leak into
          human generations. */}
      {promptModal && <IdentityPromptModal subjectType={subjectType} onClose={() => setPromptModal(false)} />}
    </div>
  );
}
