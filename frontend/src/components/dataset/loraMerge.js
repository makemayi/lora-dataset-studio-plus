/** The grammar of "bake a LoRA into a base", with no React in it.
 *
 * Split out for the same reason `denseModels.js` is: `node --test` can import
 * this and assert the sentences, the row bookkeeping and the guards without a
 * DOM. What is left in the component is markup.
 *
 * The wording here is load-bearing, not decoration. On the model sites a
 * checkpoint made exactly this way is routinely published as a "finetune" — by
 * authors who describe the merge themselves a sentence later. LDS produces the
 * same object and refuses the same vocabulary: what comes out of this is a base
 * with LoRAs folded into it, it is not a model that was trained, and both the
 * screen and the file's own header say so.
 */

export const MERGE_RUNNING_STATES = ['running'];

export const fmtGB = (bytes) => (
  typeof bytes === 'number' && bytes > 0 ? `${(bytes / 1e9).toFixed(1)} GB` : '—'
);

/** "about 4 minutes" — never a false precision on something disk-bound. */
export function fmtDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value <= 0) return '';
  if (value < 90) return 'under a minute';
  const minutes = Math.round(value / 60);
  if (minutes < 60) return `about ${minutes} minute${minutes > 1 ? 's' : ''}`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return `about ${hours} h${rest ? ` ${rest} min` : ''}`;
}

export const pct = (done, total) => (
  total > 0 ? Math.min(100, Math.max(0, Math.round((done / total) * 100))) : 0
);

/** A blank LoRA row. 1.0 = the LoRA exactly as it was trained. */
let nextRowId = 0;
export function newLoraRow(path = '', weight = 1) {
  nextRowId += 1;
  return { id: `lora-${nextRowId}`, path, weight };
}

/** Only rows that actually name a file reach the server. */
export function loraPayload(rows) {
  return (rows || [])
    .filter((row) => String(row?.path || '').trim())
    .map((row) => ({ path: String(row.path).trim(), weight: Number(row.weight) }));
}

/** Can we even ask for a plan? Cheap client-side gate — the SERVER owns every
 *  real refusal, so this only stops a request that carries nothing. */
export function canAskPlan(base, rows) {
  return Boolean(String(base || '').trim()) && loraPayload(rows).length > 0;
}

/** The one-line summary of what the merge would write. */
export function planHeadline(plan) {
  if (!plan?.ok) return '';
  const count = plan.loras?.length || 0;
  return `Folds ${count} LoRA${count > 1 ? 's' : ''} into `
    + `${plan.base_name}, and writes ${plan.destination_name} (${fmtGB(plan.output_bytes)}) `
    + `into ${plan.destination_dir}.`;
}

/** Tensors the merge will copy through without understanding them.
 *
 * This exists because one real community Krea 2 checkpoint carries ~75 MB of an
 * image in two tensors named `last.down.weight` / `last.up.weight`, hiding under
 * a legitimate prefix. We do not refuse the merge over them and we do not
 * silently drop somebody's bytes — we carry them and say so, with their names.
 */
export function carriedOverNote(plan) {
  const rows = plan?.carried_over || [];
  if (!rows.length) return '';
  const names = rows.slice(0, 4).map((row) => row.name).join(', ');
  const more = rows.length > 4 ? `, and ${rows.length - 4} more` : '';
  return `${rows.length} tensor${rows.length > 1 ? 's are' : ' is'} not part of the `
    + `${plan.family_label || 'model'} layout (${names}${more}, ${fmtGB(plan.carried_over_bytes)}). `
    + 'They are copied over unchanged — nothing is dropped, but they are not weights we merge into.';
}

/** Said on the screen, and written into the file's own header. */
export const HONESTY_NOTE = 'What this produces is a base with LoRAs folded into its '
  + 'weights — not a model that was trained as a whole. The file records that in its '
  + 'metadata, so it stays true after the file is renamed or re-uploaded.';

/** The reserve we owe anyone reading about the Turbo transplant. */
export const TURBO_NOTE = 'Merging a re-distillation LoRA (the one Krea publishes for '
  + 'Turbo) at 0.8-1.0 into a model trained on Raw is the published route to getting '
  + 'few-step speed back. We have not tested it ourselves, and it is an approximation '
  + 'rather than an identity — expect to compare before you publish.';

/** Merging into an already-quantized file is refused, and this says why. */
export const PRECISION_NOTE = 'Merge into the full-precision (bf16) model, then quantize '
  + 'the result — quantizing first and merging after loses precision twice, and the loss '
  + 'compounds every time.';

/** Weight bounds mirrored from the server, so the field can guide before it refuses. */
export const WEIGHT_MIN = -2;
export const WEIGHT_MAX = 2;

export function weightHint(weight) {
  const value = Number(weight);
  if (!Number.isFinite(value)) return '';
  if (value === 0) return 'A weight of 0 contributes nothing.';
  if (value > 1) return 'Above 1 applies the LoRA harder than it was trained.';
  if (value < 0) return 'A negative weight subtracts the LoRA.';
  return '';
}

/* ── The draft, so a resize stops eating what somebody typed ──────────────────
 *
 * The tool in Checkpoints & LoRAs renders inside `TrainingPanel`'s
 * `CheckpointPortal`, whose host node appears and disappears with the layout.
 * React alternates between a portal and a plain subtree there, and that swap
 * UNMOUNTS everything inside it. So turning a phone to landscape threw away the
 * checkpoint path and the LoRA rows just typed, and closed the disclosure on
 * top of it — `open` on a <details> is DOM state, which a remount also loses.
 *
 * The portal is what puts the manager in the right section, so the fix is not
 * to remove it: the two things worth keeping simply outlive the remount in
 * localStorage. Both keys are versioned and must NEVER be renamed without an
 * alias — they live on user machines, not in this repo.
 */
export const MERGE_DRAFT_KEY = 'loraMergeDraft_v1';
export const MERGE_OPEN_KEY = 'loraMergeOpen_v1';

/* Ceiling on rows read back: a corrupt — or hand-edited — entry must not be
   able to render an unbounded column of fields. Nobody stacks 24 LoRAs. */
const MAX_DRAFT_ROWS = 24;

const storeFor = (store) => {
  if (store !== undefined) return store;
  try { return globalThis.localStorage || null; } catch { return null; }
};

export const emptyMergeDraft = () => ({ base: '', rows: [newLoraRow()] });

/** Read the draft. Nothing in here may throw: no localStorage (SSR, a locked-
 *  down browser), a read that raises, invalid JSON or a shape from another era
 *  all degrade to a blank form. A lost draft is an annoyance; an exception
 *  during state initialisation takes the whole training panel down with it. */
export function loadMergeDraft(store) {
  try {
    const raw = storeFor(store)?.getItem(MERGE_DRAFT_KEY);
    if (!raw) return emptyMergeDraft();
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return emptyMergeDraft();
    }
    const rows = (Array.isArray(parsed.rows) ? parsed.rows : [])
      .filter((row) => row && typeof row === 'object' && typeof row.path === 'string')
      .slice(0, MAX_DRAFT_ROWS)
      // Ids are re-minted rather than trusted: a stored id could collide with a
      // row added later in the session, and `dropRow` would then remove both.
      .map((row) => newLoraRow(
        row.path,
        typeof row.weight === 'number' || typeof row.weight === 'string' ? row.weight : 1,
      ));
    return {
      base: typeof parsed.base === 'string' ? parsed.base : '',
      rows: rows.length ? rows : [newLoraRow()],
    };
  } catch { return emptyMergeDraft(); }
}

/** Remember the draft. A full quota must not break the keystroke that otherwise
 *  worked, so a failed write just means a draft that is not kept. */
export function saveMergeDraft(draft, store) {
  try {
    storeFor(store)?.setItem(MERGE_DRAFT_KEY, JSON.stringify({
      base: String(draft?.base || ''),
      rows: (draft?.rows || []).map((row) => ({
        path: String(row?.path || ''), weight: row?.weight,
      })),
    }));
  } catch { /* not kept */ }
}

/** Forget it. Called once a merge really starts: a form that was submitted must
 *  not come back on the next visit looking like work that never left. */
export function clearMergeDraft(store) {
  try { storeFor(store)?.removeItem(MERGE_DRAFT_KEY); } catch { /* ignore */ }
}

/** Which base the field starts on. A base handed in by a full-model card ALWAYS
 *  wins over the draft: that instance is scoped to one model, and showing it a
 *  path left over from elsewhere would merge into the wrong checkpoint. */
export function initialMergeBase(base, draft) {
  return String(base || '') || String(draft?.base || '');
}

/** Was the merge disclosure left open? Anything unreadable means closed — the
 *  panel's default, and the state that hides the least. */
export function loadMergeOpen(store) {
  try { return storeFor(store)?.getItem(MERGE_OPEN_KEY) === '1'; } catch { return false; }
}

export function saveMergeOpen(open, store) {
  try { storeFor(store)?.setItem(MERGE_OPEN_KEY, open ? '1' : '0'); } catch { /* ignore */ }
}
