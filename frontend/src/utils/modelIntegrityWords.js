/* How the app TALKS about a model file that is on disk but cannot be loaded.
   PURE JS (no JSX, no imports) so `node --test` drives it, and a LEAF module on
   purpose: both local engines (Klein, Krea) and the Setup wizard read it, and a
   leaf is the only shape that lets them without an import cycle.

   WHY THIS FILE EXISTS
   --------------------
   Reported by zigzag4794 (Discord): a 9.5 GB Klein UNET sitting in the right
   folder, Setup showing "✓ Installed" for every required model, ComfyUI tested
   OK — and the Generate page refusing with "⚠ Klein model missing — download it
   in the Setup step". The file was there. `/api/capabilities` reported it under
   `klein_invalid` with verdict `truncated_or_garbage`: a .safetensors declares
   its JSON header length in its first 8 bytes, and his file was shorter than it
   claimed (an interrupted or corrupted download).

   Two OPPOSITE problems — absent vs corrupted — were being collapsed into one
   sentence, and their fixes are not the same:
     absent    -> download it;
     corrupted -> DELETE it, then download it again. Re-downloading alone was a
                  no-op (setup_installer._run_model_download returned "already
                  present" on any existing file), so the one action the message
                  suggested could not work. That is fixed on the backend; this
                  module is the half that says the right words.

   The verdicts come from backend/app/services/model_integrity.py — the validator
   already existed and worked; nothing on the Setup screen was asking it. */

/** Plain-English cause for one blocking integrity verdict. `html_or_text` is the
 *  licence/login gate page saved as .safetensors; `truncated_or_garbage` is the
 *  interrupted or corrupted download. Unknown verdicts get a truthful, vague
 *  fallback rather than a confident wrong guess. */
export function integrityCause(verdict) {
  if (verdict === 'html_or_text') {
    return 'it is a web page, not weights — the download skipped the licence/login step';
  }
  if (verdict === 'truncated_or_garbage') {
    return 'the download was cut short or corrupted — the file is shorter than its own header says';
  }
  return 'the file is not readable as model weights';
}

/** The ONE sentence every surface uses for a present-but-unloadable asset.
 *  Deliberately never contains the word "missing": the file IS on disk, and
 *  sending someone to look for a file they are staring at is the bug this
 *  replaces. It names the file, the cause, and the two-step fix. */
export function brokenAssetReason(engineWord, item, labels = {}) {
  const i = item || {};
  const what = labels[i.asset] || i.asset || 'model file';
  const name = i.filename ? ` (${i.filename})` : '';
  return `⚠ ${engineWord} ${what}${name} is on disk but cannot be loaded: `
    + `${integrityCause(i.verdict)}. Delete it and download it again.`;
}

/** The blocking ones only, in payload order. `too_small` is advisory — the file
 *  loads, it is just suspiciously small — and must never be dressed up as a
 *  broken install. `only` restricts to the assets that actually gate an engine
 *  (Klein's recommended LoRA is not one of them). */
export function blockingInvalid(invalidAssets, only = null) {
  return (Array.isArray(invalidAssets) ? invalidAssets : []).filter(
    (i) => i && i.blocking && (!only || only.includes(i.asset)));
}
