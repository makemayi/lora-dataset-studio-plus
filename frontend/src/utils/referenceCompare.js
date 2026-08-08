/**
 * Any dataset image → the dataset's REFERENCE photo.
 *
 * `derivedCompare.js` answers "is this candidate better than the shot it was
 * made from". That question only exists for the two derivation flows, so a
 * plainly generated variation — the bulk of a character dataset — had no
 * comparison at all, while the question people actually ask of it is a
 * different one: *is this still the same person?* The answer is the reference
 * photo, which is one click away in another panel and therefore never on
 * screen next to the image being judged.
 *
 * Same descriptor shape as `describeDerivedComparison` on purpose: the lightbox
 * renders both through the SAME pane component and the SAME availability guard,
 * so a second comparison mode is a second descriptor, not a second renderer.
 *
 * WHAT THIS MODE IS NOT. The derived comparison shows two framings of the same
 * shot and its whole value is that both panes are at the same scale. Here the
 * two images are unrelated crops — a square head reference against a full body
 * plan — so each pane fits its own image and the scales legitimately differ.
 * The lightbox says which of the two readings is on screen; do not carry the
 * "same scale" promise over to this one.
 *
 * Pure JS (node --test cannot parse JSX): the decision is unit-tested here, the
 * lightbox only renders it.
 */

export const REFERENCE_COMPARISON = {
  beforeLabel: 'Reference',
  afterLabel: 'This image',
};

const unavailable = (reason) => ({
  ...REFERENCE_COMPARISON, parent: null, available: false, reason,
});

/**
 * @param img          the image currently inspected (a dataset payload row)
 * @param refFilename  the dataset's `ref_filename` — served by the very same
 *                     `/api/dataset/<id>/img/<name>` endpoint as the images, so
 *                     the caller needs no second URL builder.
 * @returns null when this image must offer NOTHING (no button, and no note
 *          either): a dataset with no reference yet already says so, loudly, in
 *          its reference panel — repeating it inside the lightbox would be
 *          noise on a screen that cannot act on it. Otherwise the descriptor
 *          `{ beforeLabel, afterLabel, parent, available, reason }`, with
 *          `available` alone deciding whether the mode can be entered.
 */
export function describeReferenceComparison(img, refFilename) {
  if (!img || !img.filename) return null;

  // No reference recorded at all — silent, on purpose (see above).
  if (refFilename === null || refFilename === undefined || refFilename === '') return null;

  // Recorded, but not something that can be fetched: a legacy or half-restored
  // row whose value survived while the file behind it did not. That IS worth a
  // sentence, because "the compare button is missing here and present on the
  // next image" is otherwise indistinguishable from a bug.
  if (typeof refFilename !== 'string' || !refFilename.trim()) {
    return unavailable('The reference photo file is missing — nothing to compare.');
  }

  // The reference inspected against itself teaches nothing.
  if (img.filename === refFilename) return null;

  return { ...REFERENCE_COMPARISON, parent: { filename: refFilename }, available: true, reason: '' };
}
