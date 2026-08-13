import { useRef, useState, useEffect } from 'react';
import IdentityPromptModal from './IdentityPromptModal';
import PoseSlotPanel from './PoseSlotPanel';
// Engine names come from the derived edit list — spelling them out here is how
// this tooltip ended up naming two engines while a third could already edit.
import { editEngineNames, pendingEditNote } from './referenceEdit';
import { imageFromClipboard } from './clipboardImage';

// Cap identique à MAX_EXTRA_REFS côté backend (face_dataset_service).
const MAX_EXTRA_REFS = 3;

/* The small action pill under a reference card. Large radius, soft shadow, and
   a hair of lift on hover — the same "filled pill" voice the rest of the rail
   uses, just tightened down for a 112px card. */
const CARD_BUTTON =
  'w-full rounded-full bg-surface-raised px-2 py-1 text-[0.6875rem] font-medium text-content ' +
  'shadow-[0_1px_2px_rgba(0,0,0,0.14),0_4px_8px_rgba(0,0,0,0.12)] transition-[box-shadow,background-color,transform] duration-200 ' +
  'hover:bg-surface hover:shadow-[0_2px_4px_rgba(0,0,0,0.18),0_8px_16px_rgba(0,0,0,0.14)] hover:-translate-y-px ' +
  'disabled:cursor-not-allowed disabled:opacity-40';

/* The photo frame: a large rounded square that gains a deeper shadow on hover. */
const CARD_FRAME =
  'relative w-28 h-28 rounded-2xl bg-black overflow-hidden shrink-0 ' +
  'shadow-[0_2px_4px_rgba(0,0,0,0.16),0_8px_16px_rgba(0,0,0,0.12)] transition-shadow duration-200 ' +
  'hover:shadow-[0_6px_14px_rgba(0,0,0,0.24),0_16px_28px_rgba(0,0,0,0.16)]';

export default function ReferencePanel({ refFilename, datasetId, onSetRef, onCropRef, onEditRef, onRecropAuto,
                                         busy, importBusy = busy, visionBusy = false, nonce = 0,
                                         extraRefs = [], onAddExtraRef, onRemoveExtraRef,
                                         onCropExtraRef, subjectType = 'human',
                                         poseSlots = {}, onSetPoseSlot, onCropPoseSlot,
                                         onMirrorPoseSlot, onTogglePoseSlotEnabled,
                                         onRemovePoseSlot, referenceEdit = null }) {
  const inp = useRef(null);
  const inpExtra = useRef(null);
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
        if (!importBusy) onSetRef(file);
      } else if (extraHover.current && extraRefs.length < MAX_EXTRA_REFS && !importBusy) {
        onAddExtraRef?.(file);
      }
    };
    document.addEventListener('paste', onPaste);
    return () => document.removeEventListener('paste', onPaste);
  }, [onSetRef, onAddExtraRef, importBusy, extraRefs.length]);

  // An edit that already landed and is waiting for Keep or Discard. It matters
  // most after a restart: the modal only opens on a click, so without this the
  // paid result would sit unannounced until the TTL deleted it.
  const waiting = onEditRef ? pendingEditNote(referenceEdit) : null;

  return (
    <div className="flex flex-col gap-3 rounded-2xl bg-surface p-3">
      {/* Main reference + angle references share ONE row. */}
      <div className="flex items-start gap-3 overflow-x-auto pb-1">
        <div className="flex w-28 shrink-0 flex-col items-center gap-2">
          <span className="text-center text-xs font-medium text-content">Reference photo</span>
          <div className={CARD_FRAME}
            onMouseEnter={() => { refHover.current = true; }} onMouseLeave={() => { refHover.current = false; }}
            title="Hover and press Ctrl+V to paste an image here">
            {refFilename
              ? <img src={imgUrl(refFilename)} alt="ref" className="w-full h-full object-cover" />
              : <span className="text-content-subtle text-xs">none</span>}
            {waiting && (
              /* A button, not a label: the whole point is to lead somewhere, and
                 the modal it opens is where Keep and Discard live. */
              <button type="button" onClick={onEditRef} disabled={busy}
                title="Open the edit to compare it with the current reference, then Keep or Discard"
                className="absolute inset-x-1 bottom-1 rounded-full bg-amber-400/90 px-1.5 py-0.5 text-[0.5625rem] font-semibold text-black disabled:opacity-40">
                ✦ {waiting} →
              </button>
            )}
          </div>
          <div className="flex w-full flex-col gap-1">
            <button type="button" onClick={() => inp.current?.click()} disabled={importBusy}
              className={CARD_BUTTON}>
              {refFilename ? 'Change' : 'Set'} reference
            </button>
            {refFilename && onEditRef && (
              <button type="button" onClick={onEditRef} disabled={busy}
                title={`Edit the reference with a prompt (${editEngineNames()}) — compare before/after, then Keep or Discard`}
                className={CARD_BUTTON}>✦ Edit</button>
            )}
            {refFilename && (
              <button type="button" onClick={onCropRef} disabled={busy}
                className={CARD_BUTTON}>✂ Crop</button>
            )}
            {refFilename && onRecropAuto && (
              <button type="button" onClick={onRecropAuto} disabled={busy || visionBusy}
                title={visionBusy ? 'Auto head-crop is unavailable during local generation.' : 'Re-crop around the head automatically — a vision pass on the kept original'}
                className={CARD_BUTTON}>✂ Auto head-crop</button>
            )}
          </div>
          <input ref={inp} type="file" accept="image/*" className="hidden" disabled={importBusy}
            onChange={(e) => { if (e.target.files[0]) onSetRef(e.target.files[0]); e.target.value = ''; }} />
        </div>

        {refFilename && (
          <PoseSlotPanel datasetId={datasetId} poseSlots={poseSlots} busy={busy}
            importBusy={importBusy} nonce={nonce} onSetPoseSlot={onSetPoseSlot}
            onCropPoseSlot={onCropPoseSlot} onMirrorPoseSlot={onMirrorPoseSlot}
            onTogglePoseSlotEnabled={onTogglePoseSlotEnabled} onRemovePoseSlot={onRemovePoseSlot} />
        )}
      </div>

      {/* Références additionnelles — identité multi-angles : Nano Banana &
          ChatGPT (jointes à l'appel API) et Klein (chaînées en ReferenceLatent
          natifs). PAS Krea 2 Edit : son unique slot secondaire a été entraîné
          pour un sujet DIFFÉRENT, donc il lit une image ajoutée dans la modale
          ✦ Edit reference, jamais ce vivier-ci (cf. LOCAL_EDIT_REF_SUPPORT).
          Ne pas réécrire « tous les moteurs » ici sans vérifier cette table.
          Recadrables une par une (✂ sur la vignette) ; le scoring reste sur la
          principale. */}
      {refFilename && (
        <div className="flex items-center gap-2 flex-wrap border-t border-border pt-2"
          onMouseEnter={() => { extraHover.current = true; }} onMouseLeave={() => { extraHover.current = false; }}
          title="Hover and press Ctrl+V to paste a new extra reference">
          <span className="text-content-subtle text-[0.6875rem]">额外</span>
          {extraRefs.map((fn) => (
            <div key={fn} className="relative w-12 h-12 rounded-xl overflow-hidden bg-black shrink-0 shadow-[0_1px_2px_rgba(0,0,0,0.16),0_4px_8px_rgba(0,0,0,0.12)] transition-shadow hover:shadow-[0_2px_4px_rgba(0,0,0,0.22),0_8px_16px_rgba(0,0,0,0.14)]">
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
              title="Add an extra reference photo — Klein and the API engines use these together to lock the identity, on every generation. Krea 2 Edit does not read them: its second image is added inside the ✦ Edit reference dialog, and is meant to be a different subject."
              className="w-12 h-12 rounded-xl border border-dashed border-border-strong text-content-muted text-lg leading-none transition-shadow hover:shadow-[0_2px_4px_rgba(0,0,0,0.16),0_8px_16px_rgba(0,0,0,0.12)] disabled:opacity-40">
              +
            </button>
          )}
          <button type="button" onClick={() => setPromptModal(true)}
            aria-label="Edit the identity instruction used with multiple references"
            title="Edit the identity instruction sent with multiple references — one box per engine family, for this dataset's subject type"
            className="w-6 h-6 rounded-full bg-surface-raised text-content-muted text-xs leading-none hover:bg-surface hover:text-content">
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
