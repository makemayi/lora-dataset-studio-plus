/** 🚩 Bank watermark MASK — the state around the hand-drawn correction zones
 * (pure and JSX-free, so `node --test` covers it directly).
 *
 * Reported by Qeeyana (Reddit): the mask could only be edited from a dataset.
 * The drawing itself is NOT reimplemented here — the Bank mounts the dataset's
 * WatermarkRegionEditor and its utils/watermarkRegions geometry, so a zone drawn
 * in a bank and a zone drawn in a dataset are the same object down to the
 * rounding. What lives here is only what the Bank does differently: which images
 * can be masked, and what each cleaning LEVEL will do with the mask.
 *
 * The one rule worth stating out loud: an EMPTY mask is a value, not a blank.
 * "I deleted every zone" means "repaint nothing here" — it must never read as
 * "no mask, use the detected box", which would clean pixels the user chose to
 * keep.
 */
import { cloneWatermarkRegions, serializeWatermarkRegions } from '../../utils/watermarkRegions.js'

/** Only a still-flagged image can be masked: on a cleaned/dismissed row the mask
 * is out of both cleaning levels' pool, so editing it would change nothing. */
export function canEditMask(img) {
  return img?.watermark_state === 'detected'
}

/** What the editor opens on: the boxes to draw, and whether they are the user's
 * own (`manual`) or the detector's. `effective_watermark_regions` is the
 * server's answer to that question — it already resolves the override, so an
 * emptied mask arrives as [] and stays []. */
export function initialMask(img) {
  const manual = Array.isArray(img?.watermark_regions)
  const effective = Array.isArray(img?.effective_watermark_regions)
    ? img.effective_watermark_regions
    : (Array.isArray(img?.watermark_bbox) && img.watermark_bbox.length === 4
      ? [img.watermark_bbox]
      : [])
  return { regions: cloneWatermarkRegions(effective), manual }
}

/** The sentence under the editor. It names the LEVEL that will act, because the
 * mask alone tells the user nothing: hand zones are repainted by 🧽 Inpaint and
 * deliberately skipped by ✂ Auto-crop (a crop cannot express several zones, nor
 * a zone on the subject). */
export function maskStatus({ regions, manual } = {}) {
  const count = Array.isArray(regions) ? regions.length : 0
  if (manual && count === 0) {
    return { tone: 'warn',
      text: 'Empty mask — cleaning will repaint nothing on this image. Draw a zone, '
        + 'reset it to the detected box, or mark it “not a watermark”.' }
  }
  if (manual) {
    return { tone: 'ok',
      text: `${count} hand-drawn zone${count === 1 ? '' : 's'} — 🧽 Inpaint repaints `
        + 'exactly these; ✂ Auto-crop skips this image.' }
  }
  if (count) {
    return { tone: 'info',
      text: 'Detected zone — drag or resize it if the box is off, or add zones. '
        + 'Your edits become the mask both cleaning levels use.' }
  }
  return { tone: 'warn',
    text: 'No zone recorded for this image — draw the watermark yourself to make it '
      + 'cleanable.' }
}

/** The PUT body. `null` drops the override (back to the detected box); a list —
 * including the empty one — is stored as the user's explicit mask. */
export function maskPayload(regionsOrNull) {
  return { regions: serializeWatermarkRegions(
    regionsOrNull === null ? null : cloneWatermarkRegions(regionsOrNull)) }
}

/** Fold a successful save back into the image row, so a reopen (or the next
 * navigation) shows what the SERVER stored rather than what we hoped it did.
 * Anything that isn't a success answer leaves the row untouched — a failed save
 * must not look like an applied one. */
export function applyMaskResponse(img, response) {
  if (!response || response.ok === false) return img
  if (!Object.prototype.hasOwnProperty.call(response, 'watermark_regions')) return img
  return {
    ...img,
    watermark_regions: Array.isArray(response.watermark_regions)
      ? cloneWatermarkRegions(response.watermark_regions)
      : null,
    effective_watermark_regions: Array.isArray(response.effective_watermark_regions)
      ? cloneWatermarkRegions(response.effective_watermark_regions)
      : [],
  }
}
