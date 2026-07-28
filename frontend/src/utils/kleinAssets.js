/* The Klein REQUIRED-weights vocabulary — which assets gate the engine and what
   a sentence calls each one.

   PURE, IMPORT-FREE LEAF on purpose. It used to live in hooks/useSetupSteps.js,
   which utils/localEngineReason.js imported. Now that the Setup wizard reads the
   engine's readiness verdict from localEngineReason.js (one answer for "is Klein
   ready", instead of Setup re-deriving its own and drifting), that import has to
   go the other way — so the shared vocabulary moved down here and nobody has a
   cycle. useSetupSteps.js re-exports these names, so existing importers are
   unaffected. */

// The consistency LoRA is only RECOMMENDED, so it never gates readiness.
export const KLEIN_REQUIRED_ASSETS = ['klein_model', 'klein_text_encoder', 'klein_vae'];

// setup_installer action name -> the short human word used in Setup/picker hints.
export const KLEIN_ASSET_LABELS = {
  klein_model: 'model',
  klein_text_encoder: 'text encoder',
  klein_vae: 'VAE',
};

/** Human names of the REQUIRED Klein weights still missing (recommended LoRA
 *  excluded), in a stable canonical order — so both the Setup step header and the
 *  picker's Klein hint can say exactly what to download. Empty => the trio is on disk. */
export function kleinMissingLabels(kleinMissing) {
  const m = Array.isArray(kleinMissing) ? kleinMissing : [];
  return KLEIN_REQUIRED_ASSETS.filter((a) => m.includes(a)).map((a) => KLEIN_ASSET_LABELS[a]);
}
