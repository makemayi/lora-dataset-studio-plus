/**
 * 🎨 Medium and ⤢ Angle — the words, the buckets and the LIMITS of the two
 * measurements the Bank added last. Pure on purpose (node --test cannot parse
 * JSX), because what has to be provable here is not a layout: it is that the app
 * never states more than it measured.
 *
 * Two facets, two very different confidences, and the UI must not flatten them:
 *
 *  • MEDIUM answers "what is this picture MADE of" from the CLIP embedding the
 *    ✨ Score pass already cached — no new image inference, ever. It is NOT
 *    `origin`, which reads the file's metadata: an AI-generated photorealistic
 *    portrait is origin 🤖 AI and medium 📷 photograph at the same time, and
 *    presenting either as evidence for the other would be wrong twice.
 *    Measured on a 23 532-image real bank against 167 hand-labelled images:
 *    photograph verdicts were right 90/90 times, the two real anime drawings
 *    were both found, and every 3D render and illustration in the sample came
 *    back 'unsure'. That last part is not a bug to hide — see MEDIUM_LIMITS.
 *
 *  • ANGLE answers "where is the head pointing" from the yaw the 🎭 Faces pass
 *    measures in the pixels. Frontal/three-quarter/profile are cuts on that
 *    number; 'behind' is not a yaw at all but the crossing of "no face found"
 *    with "the 📐 Framing pass called it a back view".
 *
 * `id`s are stored (query string, and for medium a DB column) — never rename one
 * without an alias, per the repo rule on stored identifiers.
 */

/** Medium buckets, in chip order. 'unsure' is LAST and deliberately present: it
 *  is a real, measured verdict and the only way to work through that pile is to
 *  be able to select it. */
export const MEDIUM_BUCKETS = [
  { id: 'photo', label: '📷 Photo' },
  { id: 'anime', label: '🅰 Anime' },
  { id: 'render3d', label: '🧊 3D render' },
  { id: 'illustration', label: '🖌 Illustration' },
  { id: 'unsure', label: '❔ Unsure' },
];

const MEDIUM_TITLE = {
  photo: 'A photograph — a real camera image (or something that looks exactly '
    + 'like one, including a photorealistic AI render: this reads the picture, '
    + 'not the file).',
  anime: 'An anime or manga DRAWING. A photo of somebody cosplaying an anime '
    + 'character is a photograph, and that confusion is why this verdict needs '
    + 'a high bar to be given at all.',
  render3d: '3D computer graphics — a render or a game capture.',
  illustration: 'A drawing or a painting that is not anime-styled.',
  unsure: 'The classifier could not call it. Banners, screenshots, collages and '
    + 'anything between two mediums land here — it is an answer, not a gap.',
};

export function mediumTitle(id) {
  return MEDIUM_TITLE[id] || null;
}

/** Angle buckets, in chip order. */
export const ANGLE_BUCKETS = [
  { id: 'frontal', label: '😐 Frontal' },
  { id: 'three_quarter', label: '◑ Three-quarter' },
  { id: 'profile', label: '👤 Profile' },
  { id: 'behind', label: '🔙 From behind' },
];

/** Degrees, mirroring backend ANGLE_FRONTAL_MAX / ANGLE_PROFILE_MIN. Kept here
 *  only to WRITE the tooltips — the buckets themselves are computed server-side
 *  from the same two numbers, so the chips and the grid cannot drift apart. */
export const ANGLE_FRONTAL_MAX = 20;
export const ANGLE_PROFILE_MIN = 60;

const ANGLE_TITLE = {
  frontal: `Head turned less than ${ANGLE_FRONTAL_MAX}° — facing the camera.`,
  three_quarter: `Head turned ${ANGLE_FRONTAL_MAX}–${ANGLE_PROFILE_MIN}° — the `
    + 'three-quarter view most portrait sets are short of.',
  profile: `Head turned more than ${ANGLE_PROFILE_MIN}°. Under-counted on `
    + 'purpose-built sets: a head turned that far often defeats the face '
    + 'detector outright, and an image with no detected face cannot be measured.',
  behind: 'No face found AND the 📐 Framing pass called it a back view. Needs '
    + 'BOTH passes — without the framing pass this stays empty rather than '
    + 'guessing that every faceless picture has somebody in it.',
};

export function angleTitle(id) {
  return ANGLE_TITLE[id] || null;
}

/** Which chips to show. Same rule as the framing/resolution rows: a bucket that
 *  holds nothing is hidden, EXCEPT the one currently being filtered on — a chip
 *  must never vanish under the cursor mid-review. */
export function shownBuckets(buckets, counts, active) {
  const c = counts || {};
  return buckets.filter((b) => (c[b.id] || 0) > 0 || active === b.id);
}

/** The MEDIUM row's honest footnote, or null when there is nothing to warn
 *  about. Written from THIS bank's own numbers, never from a constant: the
 *  reference measurement is a fact about one corpus, the sentence below is a
 *  fact about yours. */
export function mediumLimits(counts, classified) {
  const c = counts || {};
  const total = MEDIUM_BUCKETS.reduce((n, b) => n + (c[b.id] || 0), 0);
  if (!total) return null;
  const unsure = c.unsure || 0;
  const notes = [];
  if (unsure) {
    notes.push(`${unsure} of ${total} came back “unsure” — the classifier `
      + 'refuses a verdict rather than guessing one.');
  }
  if ((c.anime || 0) > 0) {
    notes.push('“Anime” is the hardest call: a photo of somebody cosplaying an '
      + 'anime character looks like anime to the model, so check before you act '
      + 'on that pile.');
  }
  if (typeof classified === 'number' && classified > total) {
    notes.push(`${classified - total} image(s) have no ✨ Score embedding yet and `
      + 'cannot be classified at all.');
  }
  return notes.length ? notes.join(' ') : null;
}

/** The ⤢ row's footnote + the backfill offer, both from this bank's numbers.
 *  Returns {note, offer} where `offer` is null unless there is really something
 *  to re-measure. `minutes` is the server's own estimate — shown BEFORE the
 *  click, because an action that costs hours has to be priced first. */
export function angleReadiness(payload) {
  const counts = payload?.counts || {};
  const measured = counts.angle_measured || 0;
  const backfillable = counts.angle_backfillable || 0;
  const facesScanned = payload?.faces_scanned || 0;
  const buckets = payload?.angles || {};
  const behind = buckets.behind || 0;
  const framed = counts.framing_classified || 0;
  const notes = [];
  if (!measured && !backfillable) {
    notes.push(facesScanned
      ? `${facesScanned} image${facesScanned === 1 ? ' was' : 's were'} face-checked, `
        + 'but no measurable head angle was found — there is nothing to backfill.'
      : 'Run 🎭 Person groups to measure head angles.');
  }
  if (measured && !behind && !framed) {
    notes.push('“From behind” also needs the 📐 Framing pass — without it a '
      + 'faceless picture cannot be told from a back view.');
  }
  const offer = backfillable ? {
    count: backfillable,
    minutes: payload?.angle_backfill_minutes || null,
    label: `Measure ${backfillable} missing angle${backfillable === 1 ? '' : 's'}`,
    // Why the work exists at all, in one sentence, because "re-run something you
    // already ran" is otherwise an insulting button.
    why: `${backfillable} image${backfillable === 1 ? ' was' : 's were'} face-scanned `
      + 'by a build that measured the head angle and did not keep it. Reading it '
      + 'back means running the face detector on those images again'
      + (payload?.angle_backfill_minutes
        ? ` — about ${payload.angle_backfill_minutes} minute(s) on this machine.`
        : '.'),
  } : null;
  return { note: notes.length ? notes.join(' ') : null, offer };
}

/** The bucket a stored yaw falls in, for the tile badge. Mirrors the server's
 *  _angle_case for the three yaw buckets ONLY: 'behind' is not a yaw and is
 *  never derived here. null = not measured, which is never shown as 'frontal'. */
export function angleOfYaw(yaw) {
  if (typeof yaw !== 'number' || !Number.isFinite(yaw)) return null;
  const a = Math.abs(yaw);
  if (a < ANGLE_FRONTAL_MAX) return 'frontal';
  if (a < ANGLE_PROFILE_MIN) return 'three_quarter';
  return 'profile';
}

/** Short tile badge for an image's angle, or null. */
export function angleBadge(img) {
  const id = angleOfYaw(img?.face_yaw);
  if (!id) return null;
  return { id, text: { frontal: '⤢0', three_quarter: '⤢¾', profile: '⤢90' }[id] };
}
