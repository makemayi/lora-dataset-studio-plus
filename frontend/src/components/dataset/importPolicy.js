/** The sentence the import surfaces say about what they are about to store.
 *
 * Its own module, not a helper inside ImportDropzone.jsx: `node --test` parses
 * .js and not .jsx, and this string is the whole point of the change — the app
 * used to claim "normalized to 1024" as a fact of nature, with no why and no
 * way out (reported by Qeeyana on Reddit). A wrong or stale sentence here is
 * exactly the bug we are fixing, so it gets locked by a test.
 *
 * The policy comes from GET /api/capabilities (`dataset_import`), never from a
 * copy of the default kept in the front — that copy is how a hint goes stale.
 */
export const IMPORT_ENCODING_LABEL = {
  preserve: 'original file and format',
  standard: 'WebP q92',
  high: 'WebP q100',
  lossless: 'WebP lossless',
};

export const IMPORT_FALLBACK_MAX_SIDE = 1024;
export const IMPORT_FALLBACK_CEILING = 8192;
export const IMPORT_FALLBACK_ENCODING = 'preserve';
export const IMPORT_FALLBACK_INPUT_MAX_SIDE = 8192;
export const IMPORT_FALLBACK_INPUT_MAX_PIXELS = 16 * 1024 * 1024;
export const IMPORT_IMAGE_ACCEPT = 'image/jpeg,image/png,image/webp,image/bmp';
export const IMPORT_IMAGE_FORMATS = 'JPEG, PNG, WebP and BMP';

function firstPositiveInteger(values, fallback) {
  for (const value of values) {
    const parsed = Number(value);
    if (Number.isSafeInteger(parsed) && parsed > 0) return parsed;
  }
  return fallback;
}

/** The admission limit is checked before preserve, crop, or WebP normalization. */
export function importInputLimitLine(policy) {
  const maxSide = firstPositiveInteger(
    [policy?.input_max_side, policy?.preserve_max_side],
    IMPORT_FALLBACK_INPUT_MAX_SIDE,
  );
  const maxPixels = firstPositiveInteger(
    [policy?.input_max_pixels, policy?.preserve_max_pixels],
    IMPORT_FALLBACK_INPUT_MAX_PIXELS,
  );
  const mebiPixels = maxPixels / (1024 * 1024);
  const displayPixels = Number.isInteger(mebiPixels)
    ? String(mebiPixels)
    : mebiPixels.toFixed(1).replace(/\.0$/, '');
  return `${displayPixels} Mi-pixels and ${maxSide} px per side`;
}

export function preservesOriginalFiles(policy) {
  const encoding = policy?.encoding;
  return policy?.preserve === true || !IMPORT_ENCODING_LABEL[encoding] || encoding === 'preserve';
}

export function importPolicyLine(policy) {
  const p = policy || {};
  const encoding = preservesOriginalFiles(p) ? 'preserve' : p.encoding;
  const enc = IMPORT_ENCODING_LABEL[encoding];
  const inputLimit = importInputLimitLine(p);
  if (encoding === 'preserve') {
    return `stored byte-for-byte in the original file and format (input limit: ${inputLimit})`;
  }
  if (p.max_side === 0) {
    return `stored as ${enc} at original size (input limit: ${inputLimit})`;
  }
  const side = p.max_side || IMPORT_FALLBACK_MAX_SIDE;
  return `stored as ${enc}, resized to ${side} px on the long side, ratio kept (input limit: ${inputLimit})`;
}
