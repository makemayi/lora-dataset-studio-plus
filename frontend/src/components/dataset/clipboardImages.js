/** Images out of a paste event, and the names they get.
 *
 * Kept PURE and DOM-free so both halves are testable without a browser: the
 * component below only wraps what comes back into `File` objects.
 *
 * A clipboard image has no filename — Chrome hands over `image.png` for every
 * screenshot, every time. Importing several of those in a row is how a dataset
 * ends up with a pile of files whose names say nothing and collide with each
 * other, so they are named here, from the paste time and the position in the
 * paste. The extension follows the MIME type because the server accepts four
 * formats and rejects the rest by extension.
 */

// What the import route accepts (see importPolicy.js / IMPORT_IMAGE_ACCEPT).
// A clipboard can carry SVG, TIFF, HEIC and application/* — none of which
// survive the server's own check, so they are dropped here where the user is
// still looking at the screen rather than at a "0 imported" toast.
export const PASTEABLE_TYPES = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
  'image/jpg': 'jpg',
  'image/webp': 'webp',
  'image/bmp': 'bmp',
};

export function extensionFor(type) {
  return PASTEABLE_TYPES[String(type || '').toLowerCase().split(';')[0].trim()] || null;
}

/** `{blob, type}` for every image on a clipboard payload, in paste order.
 *
 * Reads `items` first and `files` second, because the two disagree on Windows:
 * a screenshot arrives as an item of kind "file" with an empty `files` list in
 * some Chrome builds, and as both in others — taking either one alone loses
 * pastes on somebody's machine. De-duplicated by identity, so a payload that
 * exposes the same blob twice imports it once.
 */
export function clipboardImageBlobs(clipboardData) {
  if (!clipboardData) return [];
  const out = [];
  const seen = new Set();
  const push = (blob) => {
    if (!blob || seen.has(blob)) return;
    if (!extensionFor(blob.type)) return;
    seen.add(blob);
    out.push({ blob, type: String(blob.type || '').toLowerCase() });
  };
  for (const item of Array.from(clipboardData.items || [])) {
    if (item && item.kind === 'file' && typeof item.getAsFile === 'function') {
      push(item.getAsFile());
    }
  }
  for (const file of Array.from(clipboardData.files || [])) push(file);
  return out;
}

/** A stable, readable, collision-free name for one pasted image. */
export function pastedFileName(type, index = 0, date = new Date()) {
  const pad = (n, width = 2) => String(n).padStart(width, '0');
  const stamp = `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}`
    + `-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`
    + `-${pad(date.getMilliseconds(), 3)}`;
  const suffix = index > 0 ? `-${index + 1}` : '';
  return `pasted-${stamp}${suffix}.${extensionFor(type) || 'png'}`;
}

/** True when a paste should be left alone because the user is typing into
 * something. A caption box is the case that matters: pasting there is a text
 * gesture, and hijacking it to import an image would be a surprise in the one
 * place the app asks people to write. */
export function isEditableTarget(target) {
  if (!target || typeof target !== 'object') return false;
  const tag = String(target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
  return Boolean(target.isContentEditable);
}
