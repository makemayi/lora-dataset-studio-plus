/* Reasons a frame was rejected during video→image promotion, in the voice the
   summary dialog uses. Pure so node --test can hold every label to account. */
export const REASON_LABELS = {
  too_blurry: 'Blurry',
  face_blurry: 'Blurry face',
  no_face: 'No person',
  low_det: 'Face too uncertain',
  face_too_small: 'Face too small',
  extreme_profile: 'Face turned away',
  face_px_unknown: 'Face size unknown',
  face_too_few_pixels: 'Face too few pixels',
  wrong_person: 'Not the reference person',
  exposure: 'Bad exposure',
  unmeasured: 'Not measured',
  too_close: 'Too close in time',
  duplicate: 'Duplicate',
  over_limit: 'Over the frame cap',
  limit_is_zero: 'No budget',
}

export const reasonLabel = (r) => REASON_LABELS[r] || r

/* Sorted by count desc, then alphabetically — the reason that ate the most
   frames leads the line. */
export function summarizeRejections(rejected) {
  return Object.entries(rejected || {})
    .map(([r, n]) => ({ reason: r, label: reasonLabel(r), count: Number(n) || 0 }))
    .filter((x) => x.count > 0)
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
}
