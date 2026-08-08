/* The list behind every "which model file?" picker (Klein model files, Krea 2
   Edit base model and identity LoRA). PURE JS (no JSX) so `node --test` can
   exercise it directly, the same split as kreaEngine.js / kreaDials.js.

   WHY THE LIST IS NOT JUST `scanned`
   ----------------------------------
   These fields used to be free text: you typed a filename and hoped. A dropdown
   of the files actually on disk is the fix, but the interesting case is the one
   a plain dropdown gets WRONG — a value pinned in the config that is NOT on
   disk.

   The tempting behaviour is to drop it and fall back to auto-detection. We do
   the opposite, and the reason is a real incident, not a preference: a base
   model was elected in silence on a shared ComfyUI and a whole training ran on
   a third-party finetune before anybody noticed. A setting that quietly resolves
   to something OTHER than what it displays is the worst kind of failure,
   because it is invisible until after the expensive part. So:

     * the pinned-but-absent value stays in the list, FIRST, flagged `missing`;
     * it stays SELECTED — nothing is reset behind the user's back;
     * and the engine that would use it refuses to start, naming the file
       (backend side: krea_edit_helper.krea_pin_gaps / klein_edit_helper.
       klein_pin_gaps, surfaced through capabilities).

   The value written is the SAME string the free-text field wrote — the config
   keys (`klein.unet`, `krea.base_model`, …) are untouched, so an install that
   typed a name by hand finds it selected here, not blanked. Free text also
   stays available: absolute paths from outside ComfyUI's roots are a supported
   input for the Klein slots and no scan can enumerate them. */

/** Case/separator-insensitive comparison of two loader names. ComfyUI emits the
 *  OS separator; users paste the `/` form; Windows filesystems are
 *  case-insensitive. All three name the same file and must not read as a miss. */
export function sameModelRef(a, b) {
  const norm = (s) => String(s || '').replace(/\\/g, '/').trim().toLowerCase();
  return norm(a) !== '' && norm(a) === norm(b);
}

const baseName = (s) => String(s || '').replace(/\\/g, '/').split('/').pop();

/** Is `value` one of the scanned files?
 *
 *  Exact name OR bare BASENAME. The basename half is not laxness, it is the
 *  install this feature must not break: `krea.base_model` has always been
 *  matched on its basename by the resolver ("a value copied from a listing still
 *  resolves"), so someone who typed `krea2_raw_fp8.safetensors` for a file the
 *  scan reports as `Krea\krea2_raw_fp8.safetensors` has a working setting. A
 *  badge accusing THAT of being missing would be the picker crying wolf at a
 *  perfectly good install on the day it shipped.
 *
 *  The asymmetry it buys, said out loud: Klein's slots do NOT basename-match in
 *  the resolver, so a bare filename naming a file that only exists in a
 *  subfolder is a genuine gap this badge will stay quiet about. It is not
 *  hidden — the backend computes the gap itself (klein_pin_gaps) and the engine
 *  card refuses by name. Between under-badging a case the engine still catches
 *  and over-badging a case that works, the badge takes the quiet side. */
export function isScanned(value, files) {
  return (files || []).some((f) => sameModelRef(f, value)
    || sameModelRef(baseName(f), baseName(value)));
}

/**
 * The options a picker shows, for a given scan and current value.
 *
 * Returns `{ options, pinnedMissing }` where each option is
 * `{ name, missing }`. `pinnedMissing` is true when the current value names a
 * file the scan did not find — the caller uses it to refuse-and-explain rather
 * than to blank the field.
 *
 * `scanned` empty is AMBIGUOUS on purpose: it happens both when the folder is
 * genuinely empty and when ComfyUI is unconfigured/unreachable/slow, and those
 * need different words. So this never reports `pinnedMissing` off an empty
 * scan — crying "not found" at someone whose ComfyUI is simply down would send
 * them to re-download a file they already have. `loading` does the same for the
 * moment before the answer arrives.
 */
export function buildModelOptions(value, scanned, { loading = false } = {}) {
  const files = Array.isArray(scanned) ? scanned : [];
  const v = String(value || '').trim();
  const known = isScanned(v, files);
  const pinnedMissing = !!v && !known && !loading && files.length > 0;
  const options = files.map((name) => ({ name, missing: false }));
  // First, not last: it is the selected value, and a list that opens on
  // something else invites picking it by accident.
  if (pinnedMissing) options.unshift({ name: v, missing: true });
  return { options, pinnedMissing };
}

/** Substring filter, applied only once the user types something that is not
 *  simply the current value (opening the dropdown on a filled field must show
 *  the whole folder, not the one file already selected). */
export function filterModelOptions(options, query, currentValue) {
  const q = String(query || '').trim().toLowerCase();
  if (!q || sameModelRef(q, currentValue)) return options;
  return options.filter((o) => o.name.toLowerCase().includes(q)
    || sameModelRef(o.name, currentValue));
}

/** What the dropdown says when it has nothing to offer. Never a mute empty box:
 *  each state names the next action, and they are DIFFERENT actions. */
export function emptyScanMessage({ loading, error, count, folderHint }) {
  if (loading) return 'Scanning ComfyUI’s model folders…';
  if (error) {
    return 'Couldn’t reach ComfyUI to list the files — type the name or path by hand.';
  }
  if (count === 0) {
    return `No model file found in ${folderHint}. Put the file there (or set ComfyUI’s `
      + 'folder in Setup ▸ ComfyUI), then ↻ to rescan — you can also type a full path.';
  }
  return null;
}

/** capabilities pin-gap slot -> the words a sentence uses. Mirrors
 *  klein_edit_helper.KLEIN_OVERRIDE_KEYS + krea_edit_helper.KREA_PIN_KEYS; an
 *  unknown slot falls back to itself rather than vanishing from the message. */
export const PIN_SLOT_LABELS = {
  unet: 'diffusion model',
  text_encoder: 'text encoder',
  vae: 'VAE',
  consistency_lora: 'consistency LoRA',
  base_model: 'base model',
  identity_lora: 'identity edit LoRA',
};

/** Why an engine refuses because of a pinned file, or null when nothing is
 *  pinned-and-absent.
 *
 *  It names the FILE, not just the slot: "base model missing" would send someone
 *  to download a model they already have. The whole point of this branch is that
 *  the gap is a value they typed, and the fix is in the same field. */
export function pinnedModelGapReason(engineLabel, gaps) {
  const list = (Array.isArray(gaps) ? gaps : []).filter((g) => g && g.slot);
  if (!list.length) return null;
  const words = list
    .map((g) => `${PIN_SLOT_LABELS[g.slot] || g.slot} “${g.configured}”`)
    .join(' + ');
  return `⚠ ${engineLabel}: the ${words} you pinned in Settings is not on disk — `
    + 'pick a file that exists, or clear the field to go back to auto-detection. '
    + 'Nothing else will be loaded in its place.';
}

/** The badge beside a value that is pinned but absent. Wording is deliberate:
 *  it says what WILL happen, because that is the part a badge saying "not
 *  found" leaves the user to guess. */
export const PINNED_MISSING_BADGE = 'not found';
export const PINNED_MISSING_TITLE =
  'No file with this name was found in ComfyUI’s model folders. It is kept as you '
  + 'typed it and the engine will refuse to run rather than quietly load a different '
  + 'file — clear the field to go back to auto-detection.';
