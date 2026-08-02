/** Split a caption into plain/leak segments for inline highlighting.
 *
 * `terms` are the exact leaking words the backend detector matched
 * (face_variations.identity_leak_terms / caption_concept_leaks) — lower-cased,
 * deduped. Matching here is case-insensitive so the highlight lands on the
 * caption's own casing; word-boundary matching avoids highlighting a term
 * that is merely a substring of an unrelated word.
 *
 * Returns an array of {text, leak} runs covering the whole caption in order,
 * so a caller can render each run as plain text or a <mark>. Empty terms (or
 * a caption with none of them present) returns the caption as one clean run.
 */
export function splitCaptionByLeakTerms(caption, terms) {
  const text = caption || '';
  const list = (terms || []).filter((t) => t && t.trim());
  if (!text || list.length === 0) return [{ text, leak: false }];

  const escaped = list
    .map((t) => t.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .sort((a, b) => b.length - a.length); // longest-first avoids a short term
                                          // pre-empting a longer one it's a substring of
  const re = new RegExp(`\\b(?:${escaped.join('|')})\\b`, 'gi');

  const runs = [];
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) runs.push({ text: text.slice(last, m.index), leak: false });
    runs.push({ text: m[0], leak: true });
    last = m.index + m[0].length;
    if (m.index === re.lastIndex) re.lastIndex += 1; // zero-width guard
  }
  if (last < text.length) runs.push({ text: text.slice(last), leak: false });
  return runs;
}
