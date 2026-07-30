/* ComfyUI folder overrides — the four keys under `comfyui` that redirect where the
   app reads from and writes to when ComfyUI is not laid out under its install
   directory (a ComfyUI started with --input-directory / --output-directory).

   The keys have always existed in config.json and always worked; what was missing was
   any field to type them in, so from the app they looked like they did not exist at
   all (reported on Discord by vykas22). Pure helpers, no JSX — the Settings section
   renders them, these functions decide WHAT each field says. */

export const COMFY_FOLDER_FIELDS = [
  {
    key: 'output_dir', id: 'comfyui-output-dir', label: 'Output folder override',
    derived: 'output',
    help: 'Where finished images are read back from. Set this when ComfyUI runs with --output-directory.',
  },
  {
    key: 'input_dir', id: 'comfyui-input-dir', label: 'Input folder override',
    derived: 'input',
    help: 'Where source images are dropped for ComfyUI to pick up. Set this when ComfyUI runs with --input-directory.',
  },
  {
    key: 'models_dir', id: 'comfyui-models-dir', label: 'Models folder override',
    derived: 'models',
    help: 'Scanned for checkpoints and training bases. extra_model_paths.yaml is still read on top of this.',
  },
  {
    key: 'loras_dir', id: 'comfyui-loras-dir', label: 'LoRAs folder override',
    derived: 'models/loras',
    help: 'Where trained LoRAs are installed so ComfyUI can load them. Leave empty to follow extra_model_paths.yaml, which wins over the default folder when a profile is marked is_default.',
  },
]

/** Field metadata by config key. The DOM ids are written out literally at the JSX call
 *  sites (the help-registry contract discovers Settings ids by scanning for id="…"),
 *  so `id` here is the value those call sites must use — the test cross-checks both. */
export function comfyFolderField(key) {
  const f = COMFY_FOLDER_FIELDS.find((x) => x.key === key)
  if (!f) throw new Error(`unknown ComfyUI folder field: ${key}`)
  return f
}

/** Placeholder for one override field: the SHAPE of the folder it falls back to.
 *  Deliberately short — an input clips its placeholder with no ellipsis and no way to
 *  scroll it, so a real path put here is unreadable on a narrow screen (checked at
 *  400px). The actual computed path goes to `folderEffective`, on its own wrapping
 *  line where it can be read in full. */
export function folderPlaceholder(field) {
  return `Empty = <ComfyUI>/${field.derived}`
}

/** The path the app will actually use while the field is empty, or null when the
 *  field is filled (the value is right there) or nothing can be computed yet.
 *  Rendered as a wrapping line under the input: making the effective folder visible
 *  without having to work it out is the entire point of this block.
 *  `extra_paths` = the folder comes from the ComfyUI install's
 *  extra_model_paths.yaml rather than from the install layout — it must show here
 *  too, or the panel would name a folder deploys no longer use (GitHub #25). */
export function folderEffective(info) {
  if (!info || !info.resolved) return null
  if (info.source !== 'derived' && info.source !== 'extra_paths') return null
  return info.resolved
}

/** Where an effective path came from, for the label next to it. Only the yaml case
 *  needs saying: "derived" is already implied by an empty field. */
export function folderEffectiveNote(info) {
  return info && info.source === 'extra_paths' ? 'from extra_model_paths.yaml' : null
}

/** Warning under one override field, or null when there is nothing to say.
 *  A path that is not on disk must be named as such: silently accepting it would
 *  reproduce the original bug in a new place (the app looking somewhere the user
 *  never intended, with nothing on screen saying so). */
export function folderWarning(info) {
  if (!info || !info.resolved) return null
  if (!info.exists) {
    if (info.source === 'override') {
      return `Not found on disk: ${info.resolved} — the folder is used as typed, so generation will fail until it exists.`
    }
    return `Not found on disk: ${info.resolved} — check the ComfyUI install directory above, or set an override here.`
  }
  // Present but not usable FROM HERE — the half the old check certified without
  // testing. The app hands ComfyUI its source images through the input folder, so a
  // ComfyUI in another container (folder not shared, or mounted read-only) passed
  // every green check and then failed at the first generation with a bare 500
  // (reported on Discord by nofaceman). The backend's sentence already names the
  // folder, the cause and the shared-volume requirement, and is path-redacted.
  if (info.usable === false && info.problem) return info.problem
  return null
}

/** True when the running ComfyUI reported a folder that differs from what is typed,
 *  so the field can offer a one-click "Use detected". Detection comes from ComfyUI's
 *  own command line; when it reported nothing, nothing is offered. */
export function detectedSuggestion(field, detected, currentValue) {
  const found = (detected && detected[field.key]) || ''
  if (!found) return null
  if (norm(found) === norm(currentValue)) return null
  return found
}

const norm = (p) => String(p || '').trim().replace(/[\\/]+$/, '').replace(/\\/g, '/').toLowerCase()

/** Query string for GET /api/setup/comfyui-folders from the live ComfyUI form values. */
export function foldersQuery(comfy, { detect = false } = {}) {
  const c = comfy || {}
  const params = new URLSearchParams()
  params.set('base_dir', c.base_dir || '')
  for (const f of COMFY_FOLDER_FIELDS) params.set(f.key, c[f.key] || '')
  if (detect) params.set('detect', '1')
  return params.toString()
}

/** True when any override field is filled — the <details> block opens on its own so a
 *  user who already configured these by hand in config.json sees them straight away
 *  instead of behind a closed "Advanced" summary. */
export function hasAnyOverride(comfy) {
  return COMFY_FOLDER_FIELDS.some((f) => String((comfy || {})[f.key] || '').trim() !== '')
}
