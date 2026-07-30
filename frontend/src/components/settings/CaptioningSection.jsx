import { INPUT_CLASS, Card } from './primitives'
import ResetToDefault from './ResetToDefault'
import { defaultValueAt } from './settingDefaults.js'
import { importInputLimitLine } from '../dataset/importPolicy.js'

const CAPTIONING_OPTIONS = [
  { id: 'auto', label: 'Auto (best available)' },
  { id: 'joycaption', label: 'JoyCaption' },
  { id: 'ollama', label: 'Ollama vision' },
  { id: 'none', label: 'None' },
]

export default function CaptioningSection({ config, setField, configDefaults }) {
  // The bank thresholds below were tuned on a real 36 000-image bank and are
  // re-tuned between releases: the numbers shown when a key is missing, like the
  // ones "Reset to default" writes, come from the server (config_defaults).
  const bankDefault = (key) => defaultValueAt(configDefaults, 'bank', key)
  const importDefault = (key) => defaultValueAt(configDefaults, 'dataset_import', key)
  const importEncoding = String(config.dataset_import?.encoding ?? importDefault('encoding') ?? 'preserve')
  const preservesOriginals = importEncoding === 'preserve'
  const inputLimit = importInputLimitLine()
  return (
    <div className="space-y-6">
      <Card
        title="Dataset import"
        help={`What happens to a photo the moment it enters a dataset. Preserve originals keeps the master file; training makes a temporary working copy only when a run starts. Every import must fit within ${inputLimit}; convert or resize a larger source before importing.`}
      >
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label htmlFor="dataset-import-max-side" className="block text-sm font-medium text-content">
              Stored resolution for WebP modes
            </label>
            <select id="dataset-import-max-side"
              value={String(config.dataset_import?.max_side ?? importDefault('max_side'))}
              onChange={(e) => setField('dataset_import', 'max_side', Number(e.target.value))}
              className={INPUT_CLASS}>
              <option value="1024">1024 px long side</option>
              <option value="1536">1536 px long side</option>
              <option value="2048">2048 px long side</option>
              <option value="4096">4096 px long side</option>
              <option value="0">Original size — no downscale</option>
            </select>
            <p className="mt-0.5 text-xs text-content-muted">
              {preservesOriginals
                ? 'Preserve originals ignores this setting: no resize is performed. It becomes active only if you choose a WebP mode below.'
                : 'Longest side kept; the aspect ratio is always preserved and an image is never enlarged. WebP normalization starts only after the source passes the import safety limit below.'}
            </p>
            <ResetToDefault label="Stored resolution" section="dataset_import" field="max_side"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="dataset-import-encoding" className="block text-sm font-medium text-content">
              Stored encoding
            </label>
            <select id="dataset-import-encoding"
              value={importEncoding}
              onChange={(e) => setField('dataset_import', 'encoding', e.target.value)}
              className={INPUT_CLASS}>
              <option value="preserve">Preserve originals — no resize or conversion (default)</option>
              <option value="standard">Standard — resize + WebP quality 92</option>
              <option value="high">High — WebP quality 100</option>
              <option value="lossless">Lossless — pixel-identical</option>
            </select>
            <p className="mt-0.5 text-xs text-content-muted">
              {preservesOriginals
                ? 'Non-cropped JPEG, PNG, WebP and BMP files keep their exact bytes with the matching extension. ai-toolkit receives a disposable PNG copy only when training starts.'
                : 'WebP modes are opt-in normalization: they resize and re-encode each new import. Lossless keeps every pixel and costs about 5× the disk space (measured on a noisy photo: 158 KB → 797 KB).'}
            </p>
            <ResetToDefault label="Stored encoding" section="dataset_import" field="encoding"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
        </div>
        <p className="mt-3 text-xs text-content-muted">
          <span className="font-medium">Import safety limit:</span> every source, including
          WebP modes and Auto head-crop, must be no larger than {inputLimit}. A larger file is
          rejected; convert or resize it before importing.
        </p>
        <p className="mt-3 text-xs text-content-muted">
          Applies to images imported <span className="font-medium">from now on</span>: changing
          it mid-way can leave a dataset holding mixed formats or sizes. That is harmless for
          training (every trainer buckets and downscales on its own) but it does mean the folder
          is no longer uniform. Auto head-crop intentionally creates a derived WebP; existing
          WebPs cannot be turned back into their original files. Generated images and copies sent
          to an image API keep their own fixed sizes. Thanks to Qeeyana (Reddit) for asking why
          this was decided for you.
        </p>
      </Card>

      <Card
        title="Captioning"
        help="Who writes the captions. Auto prefers JoyCaption (via ai-toolkit) and falls back to the Ollama vision model."
      >
        <div>
          <label htmlFor="captioning-backend" className="block text-sm font-medium text-content">Captioning backend</label>
          <select
            id="captioning-backend"
            value={config.captioning.backend}
            onChange={(e) => setField('captioning', 'backend', e.target.value)}
            className={INPUT_CLASS}
          >
            {CAPTIONING_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
          <ResetToDefault label="Captioning backend" section="captioning" field="backend"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
      </Card>

      <Card
        title="Watermark inpainting"
        help="Choose where LaMa removes small off-center watermarks. Auto uses CUDA when the configured ML Python supports it and otherwise falls back to CPU."
      >
        <div>
          <label htmlFor="watermark-device" className="block text-sm font-medium text-content">Processing device</label>
          <select id="watermark-device" value={config.watermark?.device || defaultValueAt(configDefaults, 'watermark', 'device')}
            onChange={(e) => setField('watermark', 'device', e.target.value)} className={INPUT_CLASS}>
            <option value="auto">Auto (GPU when available, otherwise CPU)</option>
            <option value="cuda">GPU (CUDA required; pauses ComfyUI while cleaning)</option>
            <option value="cpu">CPU (keeps the GPU free)</option>
          </select>
          <ResetToDefault label="Processing device" section="watermark" field="device"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
        <label className="mt-3 flex items-start gap-2 text-sm text-content">
          <input id="watermark-allow-crop" type="checkbox"
            checked={config.watermark?.allow_crop !== false}
            onChange={(e) => setField('watermark', 'allow_crop', e.target.checked)}
            className="mt-0.5" />
          <span>
            <span className="font-medium">Allow automatic crop</span>
            <span className="block text-xs text-content-muted">
              On: a watermark sitting in a border is cropped off (no invented pixels). Off:
              border marks are repainted instead (LaMa/Klein). You can still override this per
              image in the watermark review. Also toggleable from the Clean bar.
            </span>
          </span>
        </label>
      </Card>

      <Card
        title="Face similarity"
        help="Every image is scored against the reference face (InsightFace). These thresholds set where the badges flip."
      >
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label htmlFor="face-threshold-green" className="block text-sm font-medium text-content">
              Face score — green threshold
            </label>
            <input
              id="face-threshold-green"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={config.face_scoring.green}
              onChange={(e) => setField('face_scoring', 'green', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS}
            />
            <ResetToDefault label="Face score — green threshold" section="face_scoring" field="green"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="face-threshold-orange" className="block text-sm font-medium text-content">
              Face score — orange threshold
            </label>
            <input
              id="face-threshold-orange"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={config.face_scoring.orange}
              onChange={(e) => setField('face_scoring', 'orange', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS}
            />
            <ResetToDefault label="Face score — orange threshold" section="face_scoring" field="orange"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
        </div>
        <p className="text-xs text-content-muted">
          Green marks a strong match to the reference face; orange is borderline — review it before keeping.
          Anything below orange is likely a different person and worth rejecting.
        </p>
      </Card>

      <Card
        title="Image bank triage"
        help="Thresholds for the 🗃️ Bank quality flags. Raw scores are stored per image, so changing a threshold re-sorts an already-scanned bank instantly — no rescan."
      >
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <div>
            <label htmlFor="bank-sharpness-min" className="block text-sm font-medium text-content">
              Sharpness minimum
            </label>
            <input id="bank-sharpness-min" type="number" min="0" step="10"
              value={config.bank?.sharpness_min ?? bankDefault('sharpness_min')}
              onChange={(e) => setField('bank', 'sharpness_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Laplacian variance under this = 🌫 blurry.</p>
            <ResetToDefault label="Sharpness minimum" section="bank" field="sharpness_min"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-noise-max" className="block text-sm font-medium text-content">
              Noise maximum
            </label>
            <input id="bank-noise-max" type="number" min="0" step="1"
              value={config.bank?.noise_max ?? bankDefault('noise_max')}
              onChange={(e) => setField('bank', 'noise_max', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Residual grain over this = 📺 noisy.</p>
            <ResetToDefault label="Noise maximum" section="bank" field="noise_max"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-uniformity-min" className="block text-sm font-medium text-content">
              Uniformity minimum
            </label>
            <input id="bank-uniformity-min" type="number" min="0" step="1"
              value={config.bank?.uniformity_min ?? bankDefault('uniformity_min')}
              onChange={(e) => setField('bank', 'uniformity_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Grayscale spread under this = ⬜ flat frame.</p>
            <ResetToDefault label="Uniformity minimum" section="bank" field="uniformity_min"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-min-side" className="block text-sm font-medium text-content">
              Minimum side (px)
            </label>
            <input id="bank-min-side" type="number" min="0" step="64"
              value={config.bank?.min_side ?? bankDefault('min_side')}
              onChange={(e) => setField('bank', 'min_side', parseInt(e.target.value, 10) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Smaller side under this = 📐 small (trainers only downscale).</p>
            <ResetToDefault label="Minimum side" section="bank" field="min_side"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-detail-min" className="block text-sm font-medium text-content">
              Real-detail minimum
            </label>
            <input id="bank-detail-min" type="number" min="0" max="1" step="0.02"
              value={config.bank?.detail_min ?? bankDefault('detail_min')}
              onChange={(e) => setField('bank', 'detail_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">
              Share of the stored size that still carries real picture, under which an image is flagged 🫧 soft detail — the usual cause is an enlargement. 0.72 picks the softest few percent. A soft or out-of-focus photo reads the same way, so treat it as a score, not proof.
            </p>
            <ResetToDefault label="Real-detail minimum" section="bank" field="detail_min"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-bars-max" className="block text-sm font-medium text-content">
              Black-bar maximum
            </label>
            <input id="bank-bars-max" type="number" min="0" max="1" step="0.01"
              value={config.bank?.bars_max ?? bankDefault('bars_max')}
              onChange={(e) => setField('bank', 'bars_max', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Share of the frame that may be flat black letterbox before an image is flagged 🎞 black bars (video screenshots, padded stills).</p>
            <ResetToDefault label="Black-bar maximum" section="bank" field="bars_max"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-dup-distance" className="block text-sm font-medium text-content">
              Duplicate distance
            </label>
            <input id="bank-dup-distance" type="number" min="0" max="16" step="1"
              value={config.bank?.dup_distance ?? bankDefault('dup_distance')}
              onChange={(e) => setField('bank', 'dup_distance', parseInt(e.target.value, 10) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">dHash bits (of 64) two images may differ by and still group as ≈ duplicates. Applies at the next scan.</p>
            <ResetToDefault label="Duplicate distance" section="bank" field="dup_distance"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-face-threshold" className="block text-sm font-medium text-content">
              Same-person similarity
            </label>
            <input id="bank-face-threshold" type="number" min="0" max="1" step="0.01"
              value={config.bank?.face_threshold ?? bankDefault('face_threshold')}
              onChange={(e) => setField('bank', 'face_threshold', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Cosine similarity for the 👥 person clustering. Applies at the next face pass.</p>
            <ResetToDefault label="Same-person similarity" section="bank" field="face_threshold"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-aesthetic-min" className="block text-sm font-medium text-content">
              Aesthetic minimum
            </label>
            <input id="bank-aesthetic-min" type="number" min="0" max="10" step="0.5"
              value={config.bank?.aesthetic_min ?? bankDefault('aesthetic_min')}
              onChange={(e) => setField('bank', 'aesthetic_min', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">LAION score (~1–10) under which an image is flagged 💔 low aesthetic. Set by the ✨ Score pass.</p>
            <ResetToDefault label="Aesthetic minimum" section="bank" field="aesthetic_min"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-nsfw-max" className="block text-sm font-medium text-content">
              NSFW maximum
            </label>
            <input id="bank-nsfw-max" type="number" min="0" max="1" step="0.05"
              value={config.bank?.nsfw_max ?? bankDefault('nsfw_max')}
              onChange={(e) => setField('bank', 'nsfw_max', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">NSFW probability (0–1) over which an image is flagged 🔞 NSFW. Set by the ✨ Score pass.</p>
            <ResetToDefault label="NSFW maximum" section="bank" field="nsfw_max"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-style-threshold" className="block text-sm font-medium text-content">
              Same-style similarity
            </label>
            <input id="bank-style-threshold" type="number" min="0" max="1" step="0.01"
              value={config.bank?.style_threshold ?? bankDefault('style_threshold')}
              onChange={(e) => setField('bank', 'style_threshold', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Cosine similarity for the 🎨 style clustering. Applies at the next scoring pass.</p>
            <ResetToDefault label="Same-style similarity" section="bank" field="style_threshold"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
          <div>
            <label htmlFor="bank-semantic-dup-threshold" className="block text-sm font-medium text-content">
              Semantic duplicate similarity
            </label>
            <input id="bank-semantic-dup-threshold" type="number" min="0" max="1" step="0.01"
              value={config.bank?.semantic_dup_threshold ?? bankDefault('semantic_dup_threshold')}
              onChange={(e) => setField('bank', 'semantic_dup_threshold', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS} />
            <p className="mt-0.5 text-xs text-content-muted">Cosine similarity at or above which two scored images are a ✂ semantic near-duplicate (crop/variant of the same shot). Re-runs instantly from cached embeddings — no re-scan.</p>
            <ResetToDefault label="Semantic duplicate similarity" section="bank" field="semantic_dup_threshold"
              config={config} configDefaults={configDefaults} setField={setField} />
          </div>
        </div>
      </Card>
    </div>
  )
}
