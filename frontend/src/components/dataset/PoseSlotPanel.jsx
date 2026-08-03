import { useRef } from 'react';

// v1 wires only these two through the backend detector (krea_pose_direction).
// The other three (back/left90/right90) exist in the API/schema already — this
// list is the ONLY change needed to open one later.
const ACTIVE_POSE_KEYS = ['left45', 'right45'];
const RESERVED_POSE_KEYS = ['back', 'left90', 'right90'];

const POSE_LABELS = {
  left45: 'Left 45°', right45: 'Right 45°',
  back: 'Back', left90: 'Left 90°', right90: 'Right 90°',
};

export default function PoseSlotPanel({ datasetId, poseSlots = {}, busy, importBusy = busy,
                                       nonce = 0, onSetPoseSlot, onCropPoseSlot,
                                       onMirrorPoseSlot, onToggleEnabled, onRemovePoseSlot }) {
  const inputs = useRef({});
  const imgUrl = (fn) => `/api/dataset/${datasetId}/img/${encodeURIComponent(fn)}${nonce ? `?v=${nonce}` : ''}`;

  return (
    <div className="flex items-center gap-2 flex-wrap border-t border-border pt-2">
      <span className="text-content-subtle text-[0.6875rem]">
        Angle references <span className="opacity-70">(Krea 2 Edit — matched to the shot's angle)</span>
      </span>
      {ACTIVE_POSE_KEYS.map((poseKey) => {
        const slot = poseSlots[poseKey] || { filename: null, enabled: false };
        return (
          <div key={poseKey} className="flex flex-col items-center gap-1 w-16">
            <div className="relative w-16 h-16 rounded-lg overflow-hidden bg-black shrink-0">
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
                  <button type="button" onClick={() => onCropPoseSlot?.(poseKey)} disabled={busy}
                    aria-label={`Crop the ${POSE_LABELS[poseKey]} reference`}
                    title="Crop this angle reference — the full frame stays kept"
                    className="absolute bottom-0 left-0 w-4 h-4 flex items-center justify-center rounded-tr bg-black/70 text-white text-[0.625rem] leading-none disabled:opacity-40">
                    ✂
                  </button>
                  <button type="button" onClick={() => onMirrorPoseSlot?.(poseKey)} disabled={busy}
                    aria-label={`Flip the ${POSE_LABELS[poseKey]} reference 180 degrees horizontally`}
                    title="Mirror — flip this photo 180° horizontally, in place"
                    className="absolute bottom-0 right-0 w-4 h-4 flex items-center justify-center rounded-tl bg-black/70 text-white text-[0.625rem] leading-none disabled:opacity-40">
                    ⇋
                  </button>
                  <button type="button" onClick={() => onRemovePoseSlot?.(poseKey)} disabled={busy}
                    aria-label={`Remove the ${POSE_LABELS[poseKey]} reference`}
                    className="absolute top-0 right-0 w-4 h-4 flex items-center justify-center rounded-bl bg-black/70 text-white text-[0.625rem] leading-none disabled:opacity-40">
                    ✕
                  </button>
                </>
              )}
            </div>
            <span className="text-content-subtle text-[0.625rem]">{POSE_LABELS[poseKey]}</span>
            {slot.filename && (
              <label className="flex items-center gap-1 text-[0.625rem] text-content-muted cursor-pointer">
                <input type="checkbox" checked={!!slot.enabled} disabled={busy}
                  onChange={(e) => onToggleEnabled?.(poseKey, e.target.checked)}
                  className="accent-indigo-500 w-3 h-3" />
                On
              </label>
            )}
            <input ref={(el) => { inputs.current[poseKey] = el; }} type="file" accept="image/*"
              className="hidden" disabled={importBusy}
              onChange={(e) => {
                if (e.target.files[0]) onSetPoseSlot?.(poseKey, e.target.files[0]);
                e.target.value = '';
              }} />
          </div>
        );
      })}
      {RESERVED_POSE_KEYS.map((poseKey) => (
        <div key={poseKey} className="flex flex-col items-center gap-1 w-16 opacity-40">
          <div className="w-16 h-16 rounded-lg border border-dashed border-border-strong flex items-center justify-center text-content-subtle text-[0.625rem] text-center px-1">
            Coming soon
          </div>
          <span className="text-content-subtle text-[0.625rem]">{POSE_LABELS[poseKey]}</span>
        </div>
      ))}
    </div>
  );
}
