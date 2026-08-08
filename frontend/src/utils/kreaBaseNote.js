/** The sentence naming the Krea 2 base an install ACTUALLY loads.
 *
 * `krea.base_model` blank means "elect one", and until now nothing on screen
 * said which file won: a ComfyUI folder holding the canonical Turbo build next
 * to a community finetune whose filename ALSO reads as "turbo" puts both in the
 * same sampling-regime tier, and the tie-break silently hands every run to one
 * of them. The only way to notice was to open a finished PNG and read its
 * metadata — so every judgement about quality made in between was about a model
 * the user never chose.
 *
 * `resolved` comes from caps.comfyui.krea_base_resolved, i.e. the very
 * `resolve_krea_unet()` the generation path calls. NOTHING is ranked here: model
 * resolution is a server decision, and a second copy of the ranking in the
 * browser would eventually disagree with the first — which is a worse lie than
 * saying nothing.
 */

const tail = (s) => s.replace(/\\/g, '/').split('/').filter(Boolean).pop() || s;

/** {tone, text} for the line under the "Base model file" field.
 *  - blank pin + a resolved file  → names it (neutral: this is just information);
 *  - blank pin + nothing on disk  → says so, and points at the engine card;
 *  - a pin that resolved to itself→ confirms it (positive);
 *  - a pin that did NOT resolve   → says the engine is held until it is fixed
 *    (warning), which is the case a typo used to hide completely.
 *
 * That last branch used to promise a SUBSTITUTE ("runs load X instead"), which was
 * true when it was written and stopped being true in the same wave: a pin that is
 * not on disk now gates the engine (capabilities.krea_pin_gaps → krea_ready false)
 * rather than quietly electing another build. resolve_krea_unet still falls back,
 * so `resolved` is populated and this branch still fires — but no run consumes it.
 * Naming a file that will never be loaded is the exact silence this note exists to
 * end, pointed the other way. */
export function kreaBaseNote(pinned, resolved) {
  const pin = typeof pinned === 'string' ? pinned.trim() : '';
  const got = typeof resolved === 'string' ? resolved.trim() : '';
  if (!pin) {
    return got
      ? { tone: 'neutral', text: `Currently loading: ${got}` }
      : { tone: 'warn',
          text: 'No compatible Krea 2 base found in your ComfyUI yet — the engine card '
            + 'in the workspace says which files are missing.' };
  }
  if (!got) {
    return { tone: 'warn',
             text: `“${pin}” was not found under any krea folder, and no other compatible `
               + 'Krea 2 base is on disk either.' };
  }
  if (tail(got).toLowerCase() !== tail(pin).toLowerCase()) {
    return { tone: 'warn',
             text: `“${pin}” was not found under any krea folder — Krea 2 Edit will not run `
               + 'until you fix the name or clear the field to pick a base automatically.' };
  }
  return { tone: 'ok', text: `Currently loading: ${got}` };
}

/** Tailwind colour per tone, kept beside the wording so a new tone cannot ship
 *  without one. */
export const KREA_BASE_NOTE_CLASS = {
  neutral: 'text-content-subtle',
  ok: 'text-emerald-400',
  warn: 'text-amber-400',
};
