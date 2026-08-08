/** Optional generation-LoRA PRESETS (Idea by @waltm — Discord feature request).
 *
 * The user defines named combinations in Settings — each preset is an ORDERED
 * list of {file, strength} rows (list order = chain order after the
 * consistency LoRA on the local Klein edit graph; files are loras-relative
 * names the user points, never hardcoded). Per run the workspace just PICKS a
 * preset — it opens on klein.default_generation_lora_preset ("None" until one
 * is set), and picking something else there applies to that run only, without
 * rewriting the setting. No per-LoRA toggles, no automatic gating: the chosen
 * preset carries the intent. The request only ever sends the preset NAME; the
 * backend resolves files/strengths/order from config (fail-closed, unknown
 * names degrade to no extra LoRAs).
 */

export const LORA_STRENGTH_MAX = 1.5;

/** Hard caps — mirror the backend's klein_edit_helper.MAX_GENERATION_LORAS /
 *  MAX_GENERATION_LORA_PRESETS (shown in the Settings card). */
export const MAX_GENERATION_LORAS = 8;
export const MAX_GENERATION_LORA_PRESETS = 12;

/** Clamp a slider/user value into the [0, 1.5] strength range the backend
 *  enforces too (NaN and negatives collapse to 0). */
export function clampLoraStrength(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 0;
  return Math.min(LORA_STRENGTH_MAX, n);
}

/** Sanitize a config-shaped preset list (from /api/settings or a Settings
 *  edit): drop blank/duplicate names and blank/malformed rows, normalize
 *  strengths (junk -> 0.6), cap rows per preset and the preset count. Order
 *  is preserved everywhere — row order IS the chain order. */
export function sanitizeGenerationLoraPresets(list) {
  const out = [];
  const seen = new Set();
  for (const preset of Array.isArray(list) ? list : []) {
    if (!preset || typeof preset !== 'object') continue;
    const name = typeof preset.name === 'string' ? preset.name.trim() : '';
    if (!name || seen.has(name)) continue;
    const rows = [];
    for (const row of Array.isArray(preset.loras) ? preset.loras : []) {
      if (!row || typeof row !== 'object') continue;
      const file = typeof row.file === 'string' ? row.file.trim() : '';
      if (!file) continue;
      const n = Number(row.strength);
      rows.push({
        file,
        strength: Number.isFinite(n) ? Math.min(LORA_STRENGTH_MAX, Math.max(0, n)) : 0.6,
      });
      if (rows.length >= MAX_GENERATION_LORAS) break;
    }
    seen.add(name);
    out.push({ name, loras: rows });
    if (out.length >= MAX_GENERATION_LORA_PRESETS) break;
  }
  return out;
}

/** Which preset the run panel STARTS on, from `klein.default_generation_lora_preset`.
 *
 *  The panel used to open on "None" on every visit, so a configured preset only
 *  ever applied when the user remembered to re-pick it — and a run that forgot
 *  showed no LoRA anywhere in its PNG metadata, which reads as the app ignoring
 *  its own settings. This is a STARTING POINT, never a lock: the picker still
 *  offers None and every other preset for that run, and the choice made there is
 *  not written back to the setting.
 *
 *  Fail-closed, like every other step of the preset chain: a name matching no
 *  configured preset (renamed, deleted, typed by hand into config.json) resolves
 *  to '' — "no preset" — rather than blocking or guessing a neighbour. */
export function resolveDefaultPresetName(defaultName, presets = []) {
  const name = typeof defaultName === 'string' ? defaultName.trim() : '';
  if (!name) return '';
  return sanitizeGenerationLoraPresets(presets).some((p) => p.name === name) ? name : '';
}

/** Body fragment for /generate (and /regenerate): the picked preset's NAME as
 *  { generation_lora_preset: name } — the backend resolves the chain from its
 *  own config (fail-closed). Empty fragment ({}) when:
 *   - the engine is not Klein (API engines never see these knobs);
 *   - no preset is picked (the "None" default);
 *   - the picked name matches no configured preset, or the preset has no rows
 *    (nothing would chain — don't send a dead name). */
export function generationLoraPresetPayload({ isKlein = false, presetName = '', presets = [] } = {}) {
  if (!isKlein) return {};
  const name = typeof presetName === 'string' ? presetName.trim() : '';
  if (!name) return {};
  const preset = sanitizeGenerationLoraPresets(presets).find((p) => p.name === name);
  if (!preset || preset.loras.length === 0) return {};
  return { generation_lora_preset: name };
}
