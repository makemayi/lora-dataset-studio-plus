/* What belongs to the SELECTED training family — the base list it may offer,
 * and whether the cloud lane serves it at all.
 *
 * Both answers used to be spelled inline in TrainingPanel.jsx as ladders of
 * family names, and both ladders forgot Anima when it was added. The base one
 * failed loudly in the wrong direction: `bases_by_type[family] || bases` falls
 * back to the WHOLE-response `bases` key, which is the Z-Image list, so the
 * panel showed `MODEL FAMILY = Anima` next to `BASE = Official - Z-Image-Turbo
 * (recommended)` — and offered this install's Z-Image merges as Anima bases.
 * A missing family must degrade to "nothing to choose here", never to another
 * architecture's catalogue.
 *
 * Plain .js on purpose: node --test does not parse JSX, so the logic worth
 * testing lives outside TrainingPanel.jsx (same reason as preflightLane.js).
 */

/** The bases `family` may be trained on, straight from /train/base-info.
 *
 * `bases` (the flat legacy key) is Z-Image's list and is returned ONLY for
 * Z-Image. Any family the server did not enumerate gets `[]`, which is what
 * makes the panel fall through to its own family-aware "Official — <family>"
 * placeholder instead of impersonating a Z-Image selector. */
export function basesForFamily(baseInfo, family) {
  const listed = baseInfo?.bases_by_type?.[family];
  if (Array.isArray(listed)) return listed;
  return family === 'zimage' && Array.isArray(baseInfo?.bases) ? baseInfo.bases : [];
}

/* An ABSOLUTE base value used to mean one thing only: "the user typed a path
 * into « Custom weights… »". Since the Krea 2 selector lists the checkpoints
 * installed on this machine, and the trainer addresses those by absolute path
 * (a RELATIVE name on Krea is read as another family's base and silently
 * ignored), absoluteness alone can no longer decide. The catalog does: a value
 * the server OFFERED is a catalog pick, anything else absolute was typed.
 *
 * Getting this wrong is visible — the panel would reopen in custom-weights mode
 * with the path in the free-text field every time the dataset is reloaded, and
 * the dropdown would show nothing selected. */
const ABSOLUTE_PATH = /^(?:[A-Za-z]:[\\/]|\\\\|\/)/;

export function looksAbsoluteBase(value) {
  return ABSOLUTE_PATH.test(String(value || ''));
}

export function isCustomWeightsBase(base, bases) {
  const value = String(base || '');
  if (!looksAbsoluteBase(value)) return false;
  return !(bases || []).some((b) => String(b?.value ?? '') === value);
}

/** What the panel must say about the base currently selected, or null.
 *
 * The server annotates each listed checkpoint once (it already read the header
 * to build the list), so switching entries in the dropdown says its piece with
 * no round trip. `level: 'error'` is a base the trainer cannot load — the same
 * refusal the save and the launch raise, shown BEFORE either is attempted;
 * `level: 'warning'` is a base that trains from degraded weights. */
export function baseSelectionNote(bases, base, typed = null) {
  const value = String(base || '');
  if (!value) return null;
  const hit = (bases || []).find((b) => String(b?.value ?? '') === value);
  if (hit) return hit.note ? { level: hit.trainable === false ? 'error' : 'warning', text: hit.note } : null;
  return typedBaseNote(typed, value);
}

/** The same verdict for a path TYPED into « Custom weights… », or null.
 *
 * `typed.for` is the path the server was ASKED about, and it is compared here
 * rather than trusted. A field like this fires a request per keystroke-pause:
 * without that comparison a slow answer for `C:\a.safetensors` lands while the
 * box already reads `C:\b.safetensors`, and the panel confidently refuses a file
 * nobody ever asked it about. The freshest answer for THIS path, or nothing. */
export function typedBaseNote(typed, base) {
  const value = String(base || '');
  if (!typed || !value || String(typed.for || '') !== value) return null;
  if (!typed.note) return null;
  return { level: typed.trainable === false ? 'error' : 'warning', text: typed.note };
}

/** The short tag appended to a base option, so the dropdown itself says which
 * entries are compromised — the long explanation is `baseSelectionNote`, shown
 * once for the SELECTED entry rather than six times inside a 230 px select. */
export function baseOptionSuffix(entry) {
  if (!entry || !entry.value) return '';
  if (entry.trainable === false) return ' · packed export';
  if (entry.quantization === 'bare_cast') return ' · fp8 cast';
  return '';
}

/** Families the cloud lane does not serve → the sentence saying so, else null.
 *
 * Mirrors the server's pre-reservation refusals (cloud_training._assert_*): the
 * button must state the refusal instead of enabling itself and spending the
 * round trip to be told no. Anima's is a "not yet", the other two a "not here". */
const CLOUD_UNSUPPORTED = {
  sdxl: 'SDXL trains locally only — the cloud lane covers Z-Image, Krea 2 and FLUX.2 Klein',
  flux: 'FLUX.1 trains locally only — the cloud lane covers Z-Image, Krea 2 and FLUX.2 Klein',
  anima: 'Anima cloud training is coming once the pod image is verified — train it locally for now',
};

export function cloudUnsupportedFamilyReason(family) {
  return CLOUD_UNSUPPORTED[family] || null;
}
