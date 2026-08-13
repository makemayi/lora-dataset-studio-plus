import { useRef, useEffect } from 'react';
import { imageFromClipboard } from './clipboardImage';

// All five, wired end to end since 2026-08-09. The order is the order they are
// drawn in: the two three-quarter slots first because they are the ones a shot
// catalog asks for most, then the profiles, then the back.
const ACTIVE_POSE_KEYS = ['left45', 'right45', 'left90', 'right90', 'back'];

const POSE_LABELS = {
  left45: 'Left 45°', right45: 'Right 45°',
  back: 'Back', left90: 'Left 90°', right90: 'Right 90°',
};

// What each slot answers, so the card says which shots it will be used for
// rather than leaving the degree to be guessed from its name.
const POSE_HINTS = {
  left45: 'three-quarter left',
  right45: 'three-quarter right',
  left90: 'strict left profile',
  right90: 'strict right profile',
  back: 'from behind',
};

/* Same pill + frame recipes as the main reference card in ReferencePanel.jsx —
   one voice across every card in the row. */
const CARD_BUTTON =
  'w-full rounded-full bg-surface-raised px-2 py-1 text-[0.6875rem] font-medium text-content ' +
  'shadow-[0_1px_2px_rgba(0,0,0,0.14),0_4px_8px_rgba(0,0,0,0.12)] transition-[box-shadow,background-color,transform] duration-200 ' +
  'hover:bg-surface hover:shadow-[0_2px_4px_rgba(0,0,0,0.18),0_8px_16px_rgba(0,0,0,0.14)] hover:-translate-y-px ' +
  'disabled:cursor-not-allowed disabled:opacity-40';

const CARD_FRAME =
  'relative w-28 h-28 rounded-2xl bg-black overflow-hidden shrink-0 ' +
  'shadow-[0_2px_4px_rgba(0,0,0,0.16),0_8px_16px_rgba(0,0,0,0.12)] transition-shadow duration-200 ' +
  'hover:shadow-[0_6px_14px_rgba(0,0,0,0.24),0_16px_28px_rgba(0,0,0,0.16)]';

export default function PoseSlotPanel({ datasetId, poseSlots = {}, busy, importBusy = busy,
                                       nonce = 0, onSetPoseSlot, onCropPoseSlot,
                                       onMirrorPoseSlot, onTogglePoseSlotEnabled, onRemovePoseSlot }) {
  const inputs = useRef({});
  const imgUrl = (fn) => `/api/dataset/${datasetId}/img/${encodeURIComponent(fn)}${nonce ? `?v=${nonce}` : ''}`;

  // Paste-to-upload: hover a card, Ctrl+V an image instead of opening the
  // file picker. See ReferencePanel.jsx for the same pattern on the primary
  // ref / extra refs.
  const hoverKey = useRef(null);
  useEffect(() => {
    const onPaste = (e) => {
      const poseKey = hoverKey.current;
      if (!poseKey || importBusy) return;
      const file = imageFromClipboard(e);
      if (!file) return;
      e.preventDefault();
      onSetPoseSlot?.(poseKey, file);
    };
    document.addEventListener('paste', onPaste);
    return () => document.removeEventListener('paste', onPaste);
  }, [onSetPoseSlot, importBusy]);

  return (
    <>
      {ACTIVE_POSE_KEYS.map((poseKey) => {
        const slot = poseSlots[poseKey] || { filename: null, enabled: false };
        return (
          <div key={poseKey} className="flex w-28 shrink-0 flex-col items-center gap-2">
            <div className="flex w-full items-center justify-center gap-1.5">
              <span className="truncate text-xs text-content-muted"
                title={`Used for shots asking for ${POSE_HINTS[poseKey]}`}>
                {POSE_LABELS[poseKey]}
              </span>
              {slot.filename && (
                <label className="flex items-center gap-0.5 text-[0.5625rem] text-content-subtle cursor-pointer"
                  title="Enable this angle for generation">
                  <input type="checkbox" checked={!!slot.enabled} disabled={busy}
                    onChange={(e) => onTogglePoseSlotEnabled?.(poseKey, e.target.checked)}
                    className="accent-indigo-500 h-3 w-3" />
                  On
                </label>
              )}
            </div>
            <div className={CARD_FRAME}
              onMouseEnter={() => { hoverKey.current = poseKey; }} onMouseLeave={() => { hoverKey.current = null; }}
              title={`Hover and press Ctrl+V to paste the ${POSE_LABELS[poseKey]} reference`}>
              {slot.filename
                ? <img src={imgUrl(slot.filename)} alt={`${POSE_LABELS[poseKey]} reference`}
                    className="w-full h-full object-cover" />
                : <button type="button" onClick={() => inputs.current[poseKey]?.click()}
                    disabled={importBusy}
                    aria-label={`Add a ${POSE_LABELS[poseKey]} reference photo`}
                    className="w-full h-full border border-dashed border-border-strong text-content-muted text-lg leading-none disabled:opacity-40">
                    +
                  </button>}
              {slot.filename && (
                <>
                  <button type="button" onClick={() => onRemovePoseSlot?.(poseKey)} disabled={busy}
                    aria-label={`Remove the ${POSE_LABELS[poseKey]} reference`}
                    title="Remove this angle reference"
                    className="absolute top-0 right-0 w-4 h-4 flex items-center justify-center rounded-bl bg-black/70 text-white text-[0.625rem] leading-none disabled:opacity-40">
                    ✕
                  </button>
                  <button type="button" onClick={() => onMirrorPoseSlot?.(poseKey)} disabled={busy}
                    aria-label={`Flip the ${POSE_LABELS[poseKey]} reference 180 degrees horizontally`}
                    title="Mirror — flip this photo 180° horizontally, in place"
                    className="absolute bottom-0 right-0 w-4 h-4 flex items-center justify-center rounded-tl bg-black/70 text-white text-[0.625rem] leading-none disabled:opacity-40">
                    ⇋
                  </button>
                </>
              )}
            </div>
            <div className="flex w-full flex-col gap-1">
              <button type="button" onClick={() => inputs.current[poseKey]?.click()} disabled={importBusy}
                className={CARD_BUTTON}>
                {slot.filename ? 'Change' : 'Set'}
              </button>
              <button type="button" onClick={() => onCropPoseSlot?.(poseKey)} disabled={busy || !slot.filename}
                title="Crop this angle reference — the full frame stays kept"
                className={CARD_BUTTON}>✂ Crop</button>
            </div>
            <input ref={(el) => { inputs.current[poseKey] = el; }} type="file" accept="image/*"
              className="hidden" disabled={importBusy}
              onChange={(e) => {
                if (e.target.files[0]) onSetPoseSlot?.(poseKey, e.target.files[0]);
                e.target.value = '';
              }} />
          </div>
        );
      })}
    </>
  );
}
