/* Qwen-Image 2.1 — the pure, testable half of the engine's UI.
   PURE JS (no JSX) so `node --test` can import and exercise it directly, the
   same split as engineSelection.js, kreaEngine.js and minimaxH3Engine.js.

   WHY THIS FILE EXISTS
   --------------------
   Same reason as the other two: an engine that greys out with "not available"
   is an inert button. 2.1 needs THREE weight files and a recent ComfyUI (the
   edit node is CORE — there is no pack to install), so its failure modes are
   "weights missing", "weights broken" and "ComfyUI too old". Each gets its own
   sentence, because each has a different fix. */
import { comfyuiDownReason } from './comfyuiStatus.js';

/** Capabilities asset keys -> the words a sentence uses. Mirrors
 *  qwen21_edit_helper.QWEN21_ASSETS; an unknown key falls back to itself rather
 *  than disappearing from the message. */
export const QWEN21_ASSET_LABELS = {
  qwen21_model: 'Qwen 2.1 base model',
  qwen21_text_encoder: 'Qwen3-VL 8B text encoder',
  qwen21_vae: 'Qwen Image 2.1 VAE',
};

/** Stable-order words for the missing assets, so the sentence reads the same way
 *  every time regardless of how the server ordered the list. */
export function qwen21MissingLabels(missing) {
  const set = new Set(Array.isArray(missing) ? missing : []);
  return Object.keys(QWEN21_ASSET_LABELS)
    .filter((k) => set.has(k))
    .map((k) => QWEN21_ASSET_LABELS[k]);
}

/** Why the Qwen 2.1 engine can't be picked, or null when it can. Ordered by
 *  what the user has to do FIRST: a disabled engine and an unreachable ComfyUI
 *  both make the asset list meaningless, and the core-node check comes before
 *  the weights because without it nothing can run even with every file in
 *  place.
 *
 *  There is no auto-install for 2.1's weights and no pack to clone: the node
 *  ships in ComfyUI's core, so a missing class is an UPDATE, and the weights
 *  come from wherever the user got their build. Naming the gap honestly beats
 *  a made-up download button. */
export function qwen21UnavailableReason({
  enabledInSettings = true, comfyuiReachable = true,
  missingAssets = [], missingNodes = [], comfyui = null,
} = {}) {
  if (!enabledInSettings) return '⚠ Qwen 2.1 is disabled in Settings (engines)';
  if (!comfyuiReachable) return comfyuiDownReason(comfyui || { reachable: false });
  if (Array.isArray(missingNodes) && missingNodes.length) {
    return '⚠ Update ComfyUI — Qwen-Image 2.1 needs its core edit node (restart after updating)';
  }
  const words = qwen21MissingLabels(missingAssets);
  if (words.length) {
    return `⚠ Qwen 2.1 ${words.join(' + ')} missing — see the Guide for where each file goes`;
  }
  return null;
}
