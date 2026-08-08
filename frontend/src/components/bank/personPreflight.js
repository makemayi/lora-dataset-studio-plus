// 👤 The person-pass PREFLIGHT — deciding, and saying, what the sampling found
// before the expensive pass runs.
//
// WHY IT MOVED HERE. The folder sampling used to live behind 🔎 Scan folders in
// the Subfolder panel. The first thing anyone does with a fresh bank is press
// 🚀 Launch all, so they never opened that panel and paid a full face pass over
// forty folders that each held one person. A saving the default path walks past
// is not a saving.
//
// So the same probe now runs as the PREAMBLE of 👤 Group by person (standalone
// or inside Launch all) and the answer is presented where the user already is,
// with the consistent folders PRE-TICKED and one click to accept them all.
//
// Two rules this wording may never break:
//   * it still never asserts by itself. Pre-ticked is not decided — the list is
//     visible, every box can be unticked, and "Analyze everything anyway" is a
//     first-class way out. A silent wrong grouping is one nobody would think to
//     look for, and that is still true when the app is being helpful.
//   * it may only ever speak about the SAMPLE. Fifteen images cannot say a
//     folder is clean; they can say fifteen images looked one way.
//
// Pure .js (no JSX) so `node --test` executes all of it.

export const SAMPLE_SIZE = 15;

/* The verdicts the probe writes, and what each one MEANS for the pass:
   'consistent'  → offered, pre-ticked, its folder can be skipped;
   'partial'     → the same answer on fewer usable faces than asked for, because
                   the folder ran out of re-draw budget before it found fifteen.
                   Offered and pre-ticked TOO, and the row says how thin it is;
   'mixed'       → several faces in the sample, so it gets the full analysis;
   'inconclusive'→ almost nothing a face detector can read. Not a failed check —
                   a fact about the folder, said as one.
   Only the first two are ever pre-ticked. */
const CONSISTENT = 'consistent';
const PARTIAL = 'partial';
/* WHY 'partial' IS PRE-TICKED, since it is the one decision here that looks
   arguable. The bar for offering a folder has always been TWO agreeing faces —
   a folder answered "2/2 of 15" is pre-ticked today. A partial verdict rests on
   at least as many faces as that and usually far more (six in forty, say), drawn
   from more of the folder. Pre-ticking the weaker evidence and not the stronger
   would be incoherent; what the weaker one owes the user is a row that SAYS how
   thin it is, which is what changed. Every box is still untickable, and
   "Analyze everything anyway" is still one click. */
const OFFERED = [CONSISTENT, PARTIAL];

/** Is there anything worth asking the user before the pass?
 *  No candidate to sample AND no verdict already on file = no question, so the
 *  pass starts immediately. Never showing an empty dialog is half the point. */
export function preflightNeeded(plan) {
  if (!plan || !plan.available) return false;
  const known = Array.isArray(plan.known) ? plan.known : [];
  return (plan.candidates || 0) > 0 || known.length > 0;
}

/** Does the preflight still have folders to SAMPLE, or is it only showing
 *  verdicts it already had? (The second case costs nothing and runs no job.) */
export function preflightWillSample(plan) {
  return !!plan && (plan.candidates || 0) > 0;
}

function n(x) {
  return Number(x) || 0;
}

function plural(count, one, many) {
  return count === 1 ? one : many;
}

/** What the sampling costs, said BEFORE it is paid and always next to the number
 *  it is being compared against. That comparison IS the feature. */
export function preflightCostLine(plan) {
  if (!preflightWillSample(plan)) return null;
  const covered = n(plan.covered);
  const size = n(plan.sample_size) || SAMPLE_SIZE;
  const cost = n(plan.sample_cost) || covered * size;
  const full = n(plan.full_cost);
  // The CEILING, whenever the re-draw can push the bill past the typical case.
  // A folder whose first images have no readable face gets more drawn to replace
  // them, so "45 in all" would be the cost of the lucky day only — and the whole
  // offer rests on the user trusting this number against the one below it.
  const max = n(plan.sample_cost_max);
  let line = `Checking ${covered} ${plural(covered, 'folder', 'folders')} `
    + `(~${size} images each — ${cost} in all`;
  line += max > cost
    ? `, up to ${max} where faces are hard to find)`
    : ')';
  if (full > 0) line += `, against the ${full} this pass would embed.`;
  else line += '.';
  return line;
}

/** The ceiling, never muted: a preflight that covered 200 of 500 folders and
 *  said nothing would read as "the other 300 are not one person". */
export function notReachedLine(plan) {
  const left = n(plan && plan.left);
  if (!left) return null;
  return `${left} ${plural(left, 'folder was', 'folders were')} not checked `
    + '(biggest first) — they get the full analysis.';
}

/** One line per folder, in the wording of a checkbox row. `preselect` is true
 *  only for a consistent verdict. */
export function preflightRows(plan) {
  const known = (plan && Array.isArray(plan.known)) ? plan.known : [];
  return known.map((s) => {
    const sample = n(s.sample);
    const scorable = n(s.scorable);
    if (s.verdict === CONSISTENT) {
      return {
        subfolder: s.subfolder,
        verdict: CONSISTENT,
        tone: 'ok',
        preselect: true,
        images: n(s.images),
        line: `${n(s.largest)}/${scorable} of ${sample} sampled images `
          + 'look like the same person',
      };
    }
    if (s.verdict === PARTIAL) {
      // The honest middle. It says what the answer rests on, in the numbers the
      // user can weigh — "6 usable faces in 40 images tried" — rather than
      // rounding a weak verdict up to a confident one or down to silence.
      return {
        subfolder: s.subfolder,
        verdict: PARTIAL,
        tone: 'ok',
        preselect: true,
        images: n(s.images),
        line: `looks like one person, on thin evidence — only ${scorable} `
          + `usable ${plural(scorable, 'face', 'faces')} in ${sample} images tried`,
      };
    }
    if (s.verdict === 'mixed') {
      return {
        subfolder: s.subfolder,
        verdict: 'mixed',
        tone: 'warn',
        preselect: false,
        images: n(s.images),
        line: `${n(s.faces)} different faces in the sample — analyzed in full`,
      };
    }
    /* Almost nothing readable. This used to say "only 0 of 15 sampled images had
       a usable face — analyzed in full", which was two wrongs: it read as the
       check having failed, and it pointed at a full pass as the thing that would
       do better. It will not. The preflight, the folder check and the pass all
       drive ONE script (backend/infer/face_embed_infer.py) with the same
       detector and the same gates, and the probe's answers land in the pass's
       own embedding cache — the pass re-reads them rather than re-detecting. So
       the row reports what was found about the FOLDER instead. */
    return {
      subfolder: s.subfolder,
      verdict: s.verdict || 'inconclusive',
      tone: 'muted',
      preselect: false,
      images: n(s.images),
      line: `${scorable ? `only ${scorable} readable face` : 'no readable face'} `
        + `in ${sample} images tried across the folder — crops, backs or blur. `
        + 'The full pass reads them the same way, so much of this folder will '
        + 'not group by face.',
    };
  });
}

/** The boxes ticked when the dialog opens: every consistent folder, nothing
 *  else. Returns subfolder names ('' — the bank root — is a real one). */
export function defaultPicked(rows) {
  return (rows || []).filter((r) => r.preselect).map((r) => r.subfolder);
}

/** The headline over the list, in plain terms: what was found and what
 *  accepting it does. Null when the sampling found no single-person folder —
 *  then the list is only reporting, and there is nothing to pre-tick. */
export function preflightHeadline(rows) {
  const good = (rows || []).filter((r) => OFFERED.includes(r.verdict));
  if (!good.length) return null;
  const nf = good.length;
  return `${nf} ${plural(nf, 'folder looks', 'folders look')} like a single `
    + `person — treat ${plural(nf, 'it', 'each of them')} as one person and `
    + `skip ${plural(nf, 'its', 'their')} full analysis.`;
}

/** What ticking these boxes actually saves, from the counts the probe carries.
 *  Silent when nothing is ticked — a "saves 0 images" line is noise. */
export function savingLine(rows, picked) {
  const set = new Set(picked || []);
  const chosen = (rows || []).filter((r) => set.has(r.subfolder));
  if (!chosen.length) return null;
  const images = chosen.reduce((sum, r) => sum + n(r.images), 0);
  if (!images) return null;
  return `${images} ${plural(images, 'image is', 'images are')} grouped `
    + 'instantly and skipped by the pass.';
}

/** The primary button. It always says what will happen to BOTH halves, so
 *  "accept" can never read as "and the rest is left alone". */
export function acceptLabel(picked) {
  const k = (picked || []).length;
  if (!k) return '👥 Analyze everything';
  return `👤 Group ${k} ${plural(k, 'folder', 'folders')} & analyze the rest`;
}

/** The escape hatch, worded as a choice and not as a cancel. */
export const SKIP_LABEL = '👥 Analyze everything anyway';

/** What the escape hatch costs, so choosing it is informed too. */
export function skipNote(plan) {
  const full = n(plan && plan.full_cost);
  if (!full) return 'Every folder gets the full face pass.';
  return `Every folder gets the full face pass — ${full} `
    + `${plural(full, 'image', 'images')} embedded.`;
}

/** Toggling one row. Kept here (rather than inline in the dialog) so the
 *  "'' is a real subfolder" rule is exercised by a test and not by a click. */
export function togglePicked(picked, subfolder) {
  const set = new Set(picked || []);
  if (set.has(subfolder)) set.delete(subfolder);
  else set.add(subfolder);
  return [...set];
}

/** '' is the bank root everywhere else in the bank, so it is here too. */
export function folderLabel(subfolder) {
  return subfolder === '' ? 'the bank root' : subfolder;
}

/** What the app says when the sampling is over and it found nothing to offer —
 *  never silence, which would read as "it did not run". */
export function nothingFoundLine(rows) {
  if (preflightHeadline(rows)) return null;
  if (!rows || !rows.length) {
    return 'Nothing to check here — every folder gets the full analysis.';
  }
  return `None of the ${rows.length} checked folders looked like a single `
    + 'person, so all of them get the full analysis.';
}
