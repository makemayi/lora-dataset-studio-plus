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
