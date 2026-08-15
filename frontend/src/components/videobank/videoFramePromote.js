/** 🖼 Promoting kept clips into an IMAGE dataset of extracted frames.
 *
 *  The pure half of the second promotion target: what a request has to carry,
 *  what the server will refuse, and how to say out loud the two things about the
 *  result that are otherwise invisible.
 *
 *  WHY THE CEILING IS SAID TWICE — before and after. `frames per clip` reads
 *  like a promise and is not one: a clip whose every frame is over-exposed, or
 *  that never shows a usable face, contributes nothing, and the job never pads
 *  to reach a number. Someone who asked for 3 × 40 clips and received 61 images
 *  has to be able to see that this is the feature working, not a bug.
 */

export const FRAMES_PER_CLIP_MAX = 20   // mirrors the service's own ceiling
export const FRAMES_PER_CLIP_DEFAULT = 3

/** Why the request cannot be sent yet, or null. */
export function framePromoteProblem({ name, framesPerClip, requireFace,
  refDatasetId }) {
  if (!(name || '').trim()) return 'Name the dataset first.'
  const n = Number(framesPerClip)
  if (!Number.isInteger(n) || n <= 0) {
    return 'Frames per clip is a whole number, at least 1.'
  }
  if (n > FRAMES_PER_CLIP_MAX) {
    return `${FRAMES_PER_CLIP_MAX} frames per clip is the ceiling — past that `
      + 'the clip is a video, not a source of stills.'
  }
  // Refused here rather than server-side alone, because the server's refusal
  // arrives after the dialog has already been dismissed on some paths.
  //
  // A dataset ID, never a path: the request names a dataset the user owns and
  // the server builds the paths. A body that could name arbitrary absolute
  // paths would hand the face subprocess a file-read primitive.
  if (requireFace && !refDatasetId) {
    return 'Face filtering needs a dataset whose reference photo shows the person.'
  }
  return null
}

/** The POST body for /video-bank/<id>/promote-frames. */
export function framePromotePayload({ name, framesPerClip, totalLimit, ids,
  maxPerSource, requireFace, refDatasetId, triggerWord }) {
  const body = {
    name: (name || '').trim(),
    frames_per_clip: Number(framesPerClip) || FRAMES_PER_CLIP_DEFAULT,
    require_face: !!requireFace,
  }
  // Omitted, never sent as null: the server reads absent as "no cap" and a
  // null would have to mean the same thing in a second place.
  if (Number(totalLimit) > 0) body.total_limit = Number(totalLimit)
  if (Number(maxPerSource) > 0) body.max_per_source = Number(maxPerSource)
  if ((ids || []).length) body.ids = ids
  if (requireFace && refDatasetId) body.ref_dataset_id = Number(refDatasetId)
  if ((triggerWord || '').trim()) body.trigger_word = triggerWord.trim()
  return body
}

/** “X clips selected” / “every kept clip”, matching the clip promotion's voice. */
export function frameScopeLabel(selectedCount, keepCount) {
  if (selectedCount > 0) {
    return `${selectedCount} selected clip${selectedCount === 1 ? '' : 's'}`
  }
  const n = Number(keepCount) || 0
  return `all ${n} kept clip${n === 1 ? '' : 's'}`
}

/** What the ceiling means, said BEFORE the job runs. */
export function frameCeilingHint(composition) {
  const ceiling = Number(composition?.frames_ceiling) || 0
  if (!ceiling) return ''
  return `Up to ${ceiling} image${ceiling === 1 ? '' : 's'} — a ceiling, not a `
    + 'promise: clips with nothing sharp, well-exposed and (with the face '
    + 'filter on) showing a usable face contribute fewer, or none.'
}

/** What the face filter was actually doing, said AFTER the request lands. */
export function frameFaceNote(composition) {
  if (!composition) return ''
  return composition.face_filtered
    ? ''
    : 'The face filter is OFF for this run — frames were picked on sharpness '
      + 'and exposure alone, so some may not show the subject at all.'
}
