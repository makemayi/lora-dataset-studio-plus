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
  standard: 'WebP q92',
  high: 'WebP q100',
  lossless: 'WebP lossless',
};

export const IMPORT_FALLBACK_MAX_SIDE = 1024;
export const IMPORT_FALLBACK_CEILING = 8192;

export function importPolicyLine(policy) {
  const p = policy || {};
  const enc = IMPORT_ENCODING_LABEL[p.encoding] || IMPORT_ENCODING_LABEL.standard;
  if (p.max_side === 0) {
    return `kept at their original size, up to ${p.ceiling || IMPORT_FALLBACK_CEILING} px (${enc})`;
  }
  const side = p.max_side || IMPORT_FALLBACK_MAX_SIDE;
  return `resized to ${side} px on the long side, ratio kept (${enc})`;
}
