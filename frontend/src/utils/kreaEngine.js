/* Krea 2 Identity Edit — the pure, testable half of the engine's UI.
   PURE JS (no JSX) so `node --test` can import and exercise it directly, the
   same split as engineSelection.js.

   WHY THIS FILE EXISTS
   --------------------
   "Other people don't have my install." Krea needs FOUR weight files AND a
   community custom-node pack, and each of those can be missing for a different
   reason with a different fix. A card that just greys out with "not available"
   is an inert button; the user has no idea whether to install a node pack, drop
   a LoRA somewhere, or fix their ComfyUI URL. So the reason is computed here,
   from the capabilities payload, and it is unit-tested — one branch per real
   failure mode, in the order the user must fix them. */
import { brokenAssetReason, blockingInvalid } from './modelIntegrityWords.js';

/** Setup/capabilities asset keys -> the words a sentence uses. Mirrors
 *  krea_edit_helper.KREA_ASSETS; an unknown key falls back to itself rather
 *  than disappearing from the message. */
export const KREA_ASSET_LABELS = {
  krea_model: 'base model',
  krea_identity_lora: 'identity edit LoRA',
  krea_text_encoder: 'text encoder',
  krea_vae: 'VAE',
};

/** Stable-order words for the missing assets, so the sentence reads the same
 *  way every time regardless of how the server ordered the list. */
export function kreaMissingLabels(missing) {
  const set = new Set(Array.isArray(missing) ? missing : []);
  return Object.keys(KREA_ASSET_LABELS)
    .filter((k) => set.has(k))
    .map((k) => KREA_ASSET_LABELS[k]);
}

export const KREA_NODE_PACK_URL = 'https://github.com/lbouaraba/comfyui-krea2edit';

/** Why the Krea engine can't be picked, or null when it can.
 *  Ordered by what the user has to do FIRST: a disabled engine and an
 *  unreachable ComfyUI both make the asset lists meaningless, and the node pack
 *  comes before the weights because without it nothing can run even with every
 *  file in place. */
export function kreaUnavailableReason({
  enabledInSettings = true, comfyuiReachable = true,
  missingAssets = [], missingNodes = [], invalidAssets = [],
  nodePackInstalled = false,
} = {}) {
  if (!enabledInSettings) return '⚠ Krea 2 Edit is disabled in Settings (engines)';
  if (!comfyuiReachable) return '⚠ Configure ComfyUI in Settings';
  if (Array.isArray(missingNodes) && missingNodes.length) {
    // The pack is ON DISK but ComfyUI hasn't loaded it: ComfyUI registers custom
    // nodes at STARTUP only. Now that the app installs the pack itself, this is
    // the common state right after the install — and telling someone to install
    // what they just watched install is how a working feature reads as broken.
    if (nodePackInstalled) {
      return '⚠ The comfyui-krea2edit node pack is installed but ComfyUI has not loaded '
        + 'it yet — restart ComfyUI';
    }
    return '⚠ Install the comfyui-krea2edit node pack in ComfyUI, then restart it';
  }
  const words = kreaMissingLabels(missingAssets);
  // This line has now been wrong in BOTH directions, which is why the test next
  // to it pins the reason rather than the wording. It first said "see Setup"
  // while Setup covered Klein only and never said the word Krea -- a pointer to
  // a page about something else. It was then changed to name the Guide, correct
  // at the time: reading was all a user could do. Since the app installs these
  // files itself, the Guide is no longer the best answer -- Setup is, because it
  // now ACTS instead of explaining. The invariant is not "say Setup" or "say
  // Guide": it is that this message names a place that both exists and covers
  // Krea. Re-check that before rewording it again.
  if (words.length) {
    return `⚠ Krea ${words.join(' + ')} missing — Setup can download them for you`;
  }
  // Present but NOT weights: an interrupted, proxied or error-page download saves
  // HTML or a half file as .safetensors. The file exists, which is why "missing"
  // says nothing — and without this the only symptom was ComfyUI's raw
  // "Expecting value: line 1 column 1 (char 0)" at generate time.
  // Worded by the SHARED helper: Klein hit the same class of failure (a truncated
  // 9.5 GB UNET reported as "missing", zigzag4794 on Discord) and one corrupted
  // file must not be described two different ways two screens apart.
  const broken = blockingInvalid(invalidAssets);
  if (broken.length) return brokenAssetReason('Krea', broken[0], KREA_ASSET_LABELS);
  return null;
}

/* ── Reference shape vs framing (MEASURED, 2026-07-25) ──────────────────────
   Krea reproduces the REFERENCE's aspect ratio, capped at 2 MP
   (krea_edit_helper.fit_output_size): the edit LoRA was trained on same-size
   pairs, so the shot's own `aspect` hint is deliberately ignored — for this
   engine ONLY. Klein and the API engines still honour the shot.

   Consequence, same shot / same seed / same graph:
     square 1024x1024 reference  -> `body_stand_front` came back a BUST
     portrait 835x1024 reference -> full figure down to the calves
   Nothing is broken and no prompt fixes it: a standing figure does not fit in a
   square, so the model resolves the conflict by moving in. The human catalog is
   17 `body` + 1 `back` shots (plus the 🔞 ones), so a square reference quietly
   squeezes almost every wide shot a character dataset needs.

   Hence: say it when the user picks the engine or the shots, not after twenty
   generations — and offer the gesture (crop the reference to a portrait ratio)
   rather than the observation. */

/** Framings that need vertical room. Mirrors face_variations' `framing` enum;
 *  `face`/`bust` are unaffected — a square is fine, even ideal, for those. */
export const KREA_TIGHT_FRAMINGS = ['body', 'back'];

/** width/height above which a reference stops leaving room for a full figure.
 *  0.9 sits between the two MEASURED cases (0.815 portrait = worked, 1.0 square
 *  = cropped to a bust) and treats "nearly square" like square, which is what it
 *  behaves like. */
export const KREA_PORTRAIT_MAX_RATIO = 0.9;

/** What we offer to crop to. 3:4 is the catalog's own body/back aspect
 *  (face_variations.ASPECT_BY_FRAMING) — asking for the shape the shot was
 *  written for, not an arbitrary taller one. */
export const KREA_SUGGESTED_ASPECT = 3 / 4;
export const KREA_SUGGESTED_ASPECT_LABEL = '3:4';

/** 'portrait' | 'square' | 'landscape' for a w/h ratio, or null if unmeasurable.
 *  'square' covers the near-square band because that is how it renders. */
export function refOrientation(width, height) {
  const w = Number(width); const h = Number(height);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return null;
  const r = w / h;
  if (r <= KREA_PORTRAIT_MAX_RATIO) return 'portrait';
  if (r < 1.1) return 'square';
  return 'landscape';
}

/** Advisory for the generation panel, or null when there is nothing honest to
 *  say. Pure: hand it the reference's pixel size and the framings of the shots
 *  currently ticked.
 *
 *  Returns null when the reference cannot be measured (an unknown shape is not a
 *  reason to alarm anyone), when it is already portrait, or when no body/back
 *  shot is selected — the warning must not follow a user who only wants faces.
 *
 *  Deliberately NOT worded as a blocker: a square reference still produces body
 *  shots, they just land tighter. "Impossible" would be a lie. */
export function kreaFramingAdvisory({ width, height, framings } = {}) {
  const orientation = refOrientation(width, height);
  if (!orientation || orientation === 'portrait') return null;
  const list = Array.isArray(framings) ? framings : [];
  const tight = list.filter((f) => KREA_TIGHT_FRAMINGS.includes(f)).length;
  if (!tight) return null;
  const w = Math.round(Number(width)); const h = Math.round(Number(height));
  return {
    tight,
    total: list.length,
    orientation,
    width: w,
    height: h,
    sizeLabel: `${w}×${h}`,
    suggestAspect: KREA_SUGGESTED_ASPECT,
    suggestLabel: KREA_SUGGESTED_ASPECT_LABEL,
    headline: `${tight} of your ${list.length} selected shot${list.length === 1 ? '' : 's'} `
      + `${tight === 1 ? 'is a' : 'are'} body or back framing${tight === 1 ? '' : 's'}`,
    detail: `Krea keeps your reference's shape (${w}×${h}, ${orientation}), so those `
      + 'will come out cropped tighter than asked — a full-body shot lands around the '
      + 'waist. They still generate; they are just closer in. A portrait reference '
      + `(${KREA_SUGGESTED_ASPECT_LABEL} or taller) leaves room for the whole figure. `
      + 'Only Krea works this way — the other engines follow each shot.',
  };
}

/** What the grounding dial currently means, in one short phrase. The number
 *  alone tells nobody anything; this is the whole point of exposing it. */
export function groundingDescription(px) {
  const n = Number(px);
  if (!Number.isFinite(n) || n <= 0) return 'default (1024)';
  if (n <= 640) return `${n}px · follows the prompt, looser likeness`;
  if (n < 1024) return `${n}px · leans towards the prompt`;
  if (n === 1024) return `${n}px · balanced (recommended for people)`;
  if (n <= 1280) return `${n}px · leans towards the reference`;
  return `${n}px · sticks to the reference, may ignore the shot`;
}
