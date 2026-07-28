/**
 * Seed the Studio's CFG / steps axes from the SELECTED base model, not from one
 * family-wide constant.
 *
 * Why (bobba84, GitHub #18): picking “Z-Image Base” in the Test Studio landed on
 * CFG 1 and 8 steps — Turbo's numbers. Turbo is guidance-distilled, so CFG 1 is
 * right there and means “no guidance at all” on Base, which renders mush. The
 * backend now publishes `model_defaults` keyed by the same `value` as `z_models`;
 * this picks the entry for whichever base is selected.
 *
 * Two rules the callers depend on:
 *  - a user's OWN selection always wins. These helpers are only consulted when the
 *    corresponding axis has never been touched (its persisted value is null), so a
 *    saved CFG/steps choice is never rewritten underneath anyone.
 *  - unknown / absent entry -> the family fallback (`default_cfg` / `default_steps`),
 *    so an older backend, or a base the backend does not differentiate, behaves
 *    exactly as before.
 *
 * Multi-base runs (the base axis is a sweep): when the selected bases disagree on
 * a default we take the FIRST selected base's value rather than inventing a
 * compromise — the grid shows one CFG axis, and silently averaging two models'
 * recommended guidance would be right for neither.
 */

/** The defaults entry for one base model value, or null. */
export function modelDefaultsFor(payload, modelValue) {
  const table = payload?.model_defaults;
  if (!table || typeof table !== 'object') return null;
  const entry = table[modelValue];
  return entry && typeof entry === 'object' ? entry : null;
}

/**
 * Default CFG for the current selection: the first selected base that the backend
 * differentiates, else the family fallback, else 1.0.
 */
export function defaultCfgFor(payload, selectedModels) {
  for (const m of selectedModels || []) {
    const e = modelDefaultsFor(payload, m);
    if (e && typeof e.cfg === 'number') return e.cfg;
  }
  return payload?.default_cfg != null ? payload.default_cfg : 1.0;
}

/** Default step count for the current selection (same rule as defaultCfgFor). */
export function defaultStepsFor(payload, selectedModels) {
  for (const m of selectedModels || []) {
    const e = modelDefaultsFor(payload, m);
    if (e && typeof e.steps === 'number') return e.steps;
  }
  return payload?.default_steps != null ? payload.default_steps : 8;
}

/**
 * True when the selected bases do NOT all share the same defaults — the UI says so
 * rather than pretending one CFG/steps pair fits a Turbo and a Base in the same run.
 */
export function mixedModelDefaults(payload, selectedModels) {
  const seen = new Set();
  for (const m of selectedModels || []) {
    const e = modelDefaultsFor(payload, m);
    seen.add(e ? `${e.cfg}/${e.steps}` : 'fallback');
  }
  return seen.size > 1;
}
