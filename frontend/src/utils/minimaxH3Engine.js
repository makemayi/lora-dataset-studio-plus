/* MiniMax H3 — the pure, testable half of the engine's UI.
   PURE JS (no JSX) so `node --test` can import and exercise it directly, the
   same split as engineSelection.js and kreaEngine.js.

   WHY THIS FILE EXISTS
   --------------------
   Same reason as kreaEngine.js: an engine that greys out with "not available"
   is an inert button. H3 needs FIVE weight files and one community node pack,
   and it has a sixth failure mode the other engines do not have — it can be
   completely installed and still run six times slower than it should, because
   ComfyUI was launched without --disable-dynamic-vram. That one is not a
   blocker, so it is a separate sentence rather than a reason the card is dark:
   collapsing "cannot run" and "will crawl" into one message would make the
   engine look broken when it merely needs a flag. */
import { comfyuiDownReason } from './comfyuiStatus.js';

/** Capabilities asset keys -> the words a sentence uses. Mirrors
 *  minimax_h3_helper.H3_ASSETS; an unknown key falls back to itself rather than
 *  disappearing from the message. */
export const H3_ASSET_LABELS = {
  h3_unet: 'Ref2VA model',
  h3_text_encoder: 'text encoder',
  h3_video_vae: 'video VAE',
  h3_audio_vae: 'audio VAE',
  h3_clip_vision: 'CLIP-Vision tower',
};

/** Stable-order words for the missing assets, so the sentence reads the same way
 *  every time regardless of how the server ordered the list. */
export function h3MissingLabels(missing) {
  const set = new Set(Array.isArray(missing) ? missing : []);
  return Object.keys(H3_ASSET_LABELS)
    .filter((k) => set.has(k))
    .map((k) => H3_ASSET_LABELS[k]);
}

export const H3_NODE_PACK_URL = 'https://github.com/Merserk/MinimaxH3-Image';

/** Why the MiniMax H3 engine can't be picked, or null when it can.
 *  Ordered by what the user has to do FIRST: a disabled engine and an
 *  unreachable ComfyUI both make the asset list meaningless, and the node pack
 *  comes before the weights because without it nothing can run even with every
 *  file in place.
 *
 *  Unlike Klein and Krea there is no "Setup can download them for you": H3's
 *  weights are ~40 GB spread over five files, several of them community
 *  re-quantisations, and none is wired into the installer. Saying otherwise
 *  would be the message lying, which is worse than the message being short. */
export function minimaxH3UnavailableReason({
  enabledInSettings = true, comfyuiReachable = true,
  missingAssets = [], missingNodes = [], comfyui = null,
} = {}) {
  if (!enabledInSettings) return '⚠ MiniMax H3 is disabled in Settings (engines)';
  if (!comfyuiReachable) return comfyuiDownReason(comfyui || { reachable: false });
  if (Array.isArray(missingNodes) && missingNodes.length) {
    return '⚠ Install the MinimaxH3-Image node pack in ComfyUI, then restart it';
  }
  const words = h3MissingLabels(missingAssets);
  if (words.length) {
    return `⚠ H3 ${words.join(' + ')} missing — see the Guide for where each file goes`;
  }
  return null;
}

/** The one advisory that is NOT a reason the card is dark: ComfyUI running
 *  without --disable-dynamic-vram. Returned separately so a usable engine can
 *  still carry a warning.
 *
 *  Measured on a 24 GB card: the same prompt change took 397 s without the flag
 *  and 77.5 s with it, and identical runs varied by 6x. Nothing errors — H3
 *  simply loads more weights than the card holds, so the driver pages VRAM.
 *  The server publishes the sentence (it is the side that can see ComfyUI's
 *  launch command); this just decides whether to show it. */
export function minimaxH3SpeedWarning(caps = null) {
  const warning = caps && caps.minimax_h3 && caps.minimax_h3.vram_warning;
  return typeof warning === 'string' && warning ? warning : null;
}

/** What the reference-size dial currently means, in one short phrase. The value
 *  alone tells nobody anything; this is the whole point of exposing it. */
export function refImageSizeDescription(mode) {
  return String(mode) === 'max'
    ? 'max · 2048px reference, best likeness, several times slower'
    : 'match · reference scaled to the output size (measured default)';
}

/** What a packet length costs and buys. `length` is the number of frames H3
 *  samples before ONE is kept, so it multiplies the sampling cost while only
 *  ever adding candidates for the selector to choose between. */
export function packetLengthDescription(length) {
  const n = Number(length);
  if (!Number.isFinite(n) || n <= 1) {
    return '1 frame · nothing sampled that is then thrown away (default, needs the ComfyUI patch)';
  }
  if (n <= 5) return '5 frames · the stock node floor, four of them discarded';
  return `${n} frames · more candidates to pick from, proportionally slower`;
}
