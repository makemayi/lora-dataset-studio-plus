/* Which SOURCE tiles have an improvement waiting for them.

   An improve candidate is a separate row: the source keeps its pixels and its
   keep/reject state until you review the new version. That is the whole safety
   of the pass — and it is also how a result goes unnoticed. The candidate lands
   somewhere else in the grid (its own tile, its own position), so from the
   source you were looking at, nothing happened. People re-ran the pass on
   images that already had a result waiting, which costs GPU time and produces a
   second indistinguishable candidate.

   So the SOURCE says it: a tile with a finished candidate carries a badge, and
   one still rendering says that instead.

   Pure module — `node --test` cannot parse JSX, and this is the part with the
   cases worth pinning. */

/* The stored derivation kind of every improve candidate, whichever engine ran.
   It is a legacy name: the value predates the second engine and is written in
   user databases, so it stays `klein_image_improve` for a SeedVR2 result too
   (renaming it would strand every existing row). The ENGINE is carried
   separately, by the candidate's own label. */
export const IMPROVE_DERIVATION = 'klein_image_improve'

/** `{parentId: 'ready' | 'generating'}` for every source with a live candidate.

    'ready' wins over 'generating' when a source somehow has both: a file you can
    look at is more actionable than one still cooking, and it is the state that
    should stop you re-running the pass. */
export function improvementStateByParent(images) {
  const out = new Map()
  for (const image of (Array.isArray(images) ? images : [])) {
    if (!image || image.derivation_kind !== IMPROVE_DERIVATION) continue
    if (image.status !== 'pending') continue          // reviewed already
    const parentId = image.parent_image_id
    if (parentId === null || parentId === undefined) continue
    const state = image.filename ? 'ready' : 'generating'
    if (state === 'ready' || !out.has(parentId)) out.set(parentId, state)
  }
  return out
}

/** The badge for ONE source tile, or null when it has nothing pending.
    `{ text, title, tone }` — `tone` picks the styling, never the meaning. */
export function improvementBadge(state) {
  if (state === 'ready') {
    return {
      text: '✨ result to review',
      tone: 'ready',
      title: 'An upscale of this image is finished and waiting for you to keep or'
        + ' reject it. Until you do, this original is untouched — and running the'
        + ' pass again would just make a second copy of the same thing.',
    }
  }
  if (state === 'generating') {
    return {
      text: '✨ upscaling…',
      tone: 'generating',
      title: 'An upscale of this image is being generated. The original stays as'
        + ' it is; the result will arrive as its own tile.',
    }
  }
  return null
}
