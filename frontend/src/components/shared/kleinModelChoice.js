/* Which Klein model runs, said out loud — and chosen where the work happens.
   PURE JS (JSX-free) so `node --test` can import it.

   THE REPORTED PROBLEM. "I have no option anywhere to choose the Klein model
   used for improve and upgrade." Exactly right: ✨ Upscale & improve sent no
   model at all, so the server resolved one on its own. Generation DID have a
   picker — but it hid itself below two models, it saved to localStorage (a
   browser preference describing what a dataset contains), and improve never read
   it.

   Two questions that look like one and are not:
     * "should we offer a CHOICE?" — no, when there is only one model. A select
       with a single option is furniture.
     * "should we say WHICH model runs?" — always. Silence is what made the
       question unanswerable from the screen; it costs one line of text and it is
       the only part that works on a one-model install.
   Hence: `modelLine` is unconditional, the <select> is not. */

/** The per-browser key the generation picker used before the choice moved onto
    the dataset. Never renamed, never deleted behind the user's back: it is still
    read as a SUGGESTION when a dataset has not chosen, and the notice below asks
    once before adopting it. */
export const LEGACY_STORAGE_KEY = 'editPage_flux2KleinModel_v1';

/** The legacy browser pick, or '' — safe in private mode / disabled storage. */
export function readLegacyPick(storage) {
  try {
    return (storage && storage.getItem(LEGACY_STORAGE_KEY)) || '';
  } catch {
    return '';
  }
}

export const MODEL_LINE_UNKNOWN = 'Runs on the Klein model Studio detects.';
export const MODEL_LINE_NONE =
  'No Klein model detected — improve cannot run until one is installed.';

/**
 * The sentence naming the model this dataset's Klein work will run on.
 * @param {{loaded?: boolean, stored?: string|null, effective?: string|null,
 *          missing?: string|null}} state — the /klein-model payload.
 * @returns {{ text: string, tone: 'muted'|'warn' }}
 */
export function modelLine({ loaded = false, stored = null, effective = null,
                            missing = null } = {}) {
  if (!loaded) return { text: MODEL_LINE_UNKNOWN, tone: 'muted' };
  // A chosen file that has left the disk: the run refuses BY NAME rather than
  // swapping in a neighbour, so the screen must say it before the click and not
  // after it.
  if (missing) {
    return {
      text: `The model chosen for this dataset is missing: ${missing}. `
        + 'Pick another one, or put the file back.',
      tone: 'warn',
    };
  }
  if (!effective) return { text: MODEL_LINE_NONE, tone: 'warn' };
  return {
    text: stored ? `Runs on ${effective}.` : `Runs on ${effective} (auto-detected).`,
    tone: 'muted',
  };
}

export const LEGACY_NOTICE =
  'This browser has its own model choice from an earlier version. Save it to the '
  + 'dataset so ✨ Upscale & improve uses it too.';

/**
 * The one-time carry-over notice, or null.
 * Shown only when there is a real divergence to disclose: the browser holds a
 * pick, the dataset holds none, and the pick still exists on this machine. It is
 * never adopted silently — a dataset that never chose keeps resolving exactly the
 * model it resolved before, and that is the whole anti-regression contract.
 */
export function legacyNotice({ stored = null, legacy = '', choices = [],
                               effective = null } = {}) {
  if (stored || !legacy) return null;
  if (!choices.includes(legacy)) return null;
  if (legacy === effective) return null;   // nothing would change — say nothing
  return { text: LEGACY_NOTICE, value: legacy };
}

/** Where the /klein-model payload for a screen comes from: the dataset's own
    state when there is a dataset, else the global read-only one (the bank's
    watermark inpaint has no dataset to inherit a pick from). Naming the model and
    choosing it are two different questions — the second stays a dataset gesture,
    which is why there is no POST counterpart to the global route. */
export function modelEndpoint(datasetId) {
  return datasetId ? `/api/dataset/${datasetId}/klein-model` : '/api/klein-model';
}

/**
 * Whether to show the <select> at all. A dropdown with a single option is
 * furniture; a dropdown with nowhere to save is worse — it would imply a second
 * place to choose a Klein model, and there is exactly one (the dataset).
 */
export function canChooseModel({ datasetId = null, choices = [] } = {}) {
  return Boolean(datasetId) && (choices || []).length >= 2;
}

/**
 * The value a <select> should show: the dataset's pick when it is still one of
 * the detected models, else '' (the "Auto" row). A stored-but-missing model is
 * NOT silently replaced by the first option — it keeps the field empty so the
 * warning line above is the only thing speaking about it.
 */
export function selectValue({ stored = null, choices = [] } = {}) {
  return stored && choices.includes(stored) ? stored : '';
}
