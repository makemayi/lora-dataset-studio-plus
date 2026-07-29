/** Drop / pick real photos to import into the dataset.
 *
 * The dropzone used to say "normalized to 1024" and nothing else — no why, no
 * way out (reported by Qeeyana on Reddit: "why? Let me choose not to"). It now
 * states what will be stored, why that number, and links to the setting that
 * changes it. The value comes from capabilities, never from a copy of the
 * default kept here: a hint that can go stale is worse than no hint.
 */
import { useRef, useState } from 'react';
import { useCapabilities } from '../../context/CapabilitiesContext';
import SettingsLink from '../common/SettingsLink';
import { importPolicyLine } from './importPolicy.js';

export default function ImportDropzone({ onImport, busy, visionBusy = false, cropOption = false, defaultCrop = true }) {
  const { caps } = useCapabilities();
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);
  // Auto head-crop (square, vision pass). OFF keeps the original framing —
  // a bust/body photo stays a bust/body photo (aspect kept, no padding).
  // Body-fidelity datasets pass defaultCrop=false: full frames are the point.
  const [crop, setCrop] = useState(defaultCrop);

  const handle = (files) => {
    if (busy) return; // drop events bypass pointer-events-none — guard here too (I2)
    if (files && files.length) onImport(files, { crop: cropOption ? crop && !visionBusy : false });
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); handle(e.dataTransfer.files); }}
      onClick={() => inputRef.current?.click()}
      className={`flex flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed p-4 cursor-pointer text-center
        ${over ? 'border-primary bg-primary/10' : 'border-border bg-surface'} ${busy ? 'opacity-50 pointer-events-none' : ''}`}
    >
      <span className="text-xl">📥</span>
      <span className="text-content text-xs font-medium">Import real photos</span>
      <span className="text-content-subtle text-[0.625rem]">
        drag and drop or click — kept, and {importPolicyLine(caps.dataset_import)}
      </span>
      <span onClick={(e) => e.stopPropagation()}
        className="flex flex-wrap items-center justify-center gap-x-1.5 gap-y-0.5 text-[0.625rem] text-content-subtle">
        <span>trainers only ever downscale, so extra pixels cost disk, not quality.</span>
        <SettingsLink section="captioning" focus="dataset-import-max-side">Change this</SettingsLink>
      </span>
      {cropOption && (
        <label onClick={(e) => e.stopPropagation()}
          className="flex items-center gap-1.5 text-[0.625rem] text-content-muted cursor-pointer"
          title={visionBusy ? 'Auto head-crop is unavailable during local generation; photos import full-frame.' : 'ON: each photo is auto-cropped to a square head shot (vision pass, pauses ComfyUI). OFF: the photo keeps its original framing — use for bust/body shots.'}>
          <input type="checkbox" checked={crop} disabled={visionBusy} onChange={(e) => setCrop(e.target.checked)}
            className="accent-indigo-500 w-3 h-3" />
          ✂ Auto head-crop (square){visionBusy ? ' — unavailable during local generation' : ''}
        </label>
      )}
      <input ref={inputRef} type="file" accept="image/*" multiple className="hidden"
        onChange={(e) => { handle(e.target.files); e.target.value = ''; }} />
    </div>
  );
}
