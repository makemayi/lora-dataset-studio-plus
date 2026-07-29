// Which training families can actually honour dual (long + short) captions.
//
// Krea 2 and Anima pre-cache their text embeddings and unload the text encoder to fit
// their DiT in VRAM. ai-toolkit caches exactly ONE embedding per image (the long
// caption), and once the encoder is gone the train loop reads those cached embeddings
// instead of the prompt strings — so a short caption has nowhere to be encoded. Emitting
// the pair anyway crashed the run at the first step (GitHub issue #22, reported by
// 1Tomber). The backend now refuses the combination when it builds the ai-toolkit config;
// this mirror exists so the toggle can say so BEFORE the launch instead of letting the
// user believe both wordings are training.
//
// Kept in a plain .js module (not the JSX panel) so `node --test` can cover it.
export const DUAL_CAPTION_UNSUPPORTED_FAMILIES = ['krea', 'anima'];

const FAMILY_LABEL = {
  zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL',
  flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein', anima: 'Anima',
};

/**
 * @param {string} family training family id (`train_type`)
 * @returns {{supported: boolean, note: string}} `note` is empty when supported.
 */
export function dualCaptionsSupport(family) {
  if (!DUAL_CAPTION_UNSUPPORTED_FAMILIES.includes(family)) return { supported: true, note: '' };
  const label = FAMILY_LABEL[family] || family;
  return {
    supported: false,
    note: `${label} caches its text embeddings and unloads the text encoder, so the short `
      + 'caption cannot be encoded — this run trains on the long caption alone.',
  };
}
