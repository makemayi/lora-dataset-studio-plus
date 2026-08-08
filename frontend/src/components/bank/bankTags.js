/**
 * Turning ONE image's caption into the chips you can filter the bank by.
 *
 * Asked for by the maintainer: "let me open an image, see its tags, tick the
 * ones I care about, and show me the other images that have them". It is the
 * complement of 🎯 Similar to selected, not a rival — and the difference has to
 * stay legible in the UI, because the two answer different questions:
 *
 *   🎯 Similar to selected  — visual similarity, from CLIP embeddings. A black
 *                             box: it works on images nobody captioned, and it
 *                             cannot tell you WHY two images landed together.
 *   🏷️ Tags from this image — attributes YOU picked, matched as words. Fully
 *                             legible (you can read the chips you ticked), and
 *                             it can only ever see what a caption put in words.
 *
 * WHAT A CHIP IS. Captions in this app come in two shapes, so the split does too
 * — the same distinction `utils/tagFilter.js` draws for the dataset grid:
 *
 *  • BOORU ("a woman, red dress, balcony, golden hour") — commas ARE the tag
 *    separator, so each segment is a chip, whole and unsplit. This is detected,
 *    not configured: several commas and short segments between them.
 *  • PROSE ("a woman in a red dress standing on a balcony") — there are no tag
 *    boundaries to read, so words are the best available unit. Grammar words
 *    carry no visual meaning and would produce a wall of useless chips, so a
 *    stop list drops them.
 *
 * The honest limits, stated here and repeated in the UI:
 *  • A prose caption yields WORDS, not concepts: "golden hour" becomes two chips
 *    ("golden", "hour"), and ticking both means "captions mentioning both words",
 *    not "captions about golden hour".
 *  • A chip can only find what a captioner wrote down. An attribute nobody
 *    captioned is invisible to this, however plainly it is in the picture.
 */

/** Words that survive every caption and mean nothing visually. Deliberately dull
 *  and short: a stop list that tries to be clever starts eating real subjects
 *  ("light", "young", "long" are NOT here on purpose). */
const STOP = new Set([
  'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'with', 'from',
  'for', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'being', 'been', 'it',
  'its', 'this', 'that', 'these', 'those', 'there', 'here', 'her', 'his',
  'their', 'them', 'they', 'he', 'she', 'him', 'you', 'your', 'we', 'our',
  'has', 'have', 'had', 'while', 'into', 'onto', 'over', 'under', 'out', 'up',
  'down', 'off', 'about', 'above', 'below', 'very', 'some', 'both', 'than',
  'then', 'so', 'but', 'if', 'not', 'no', 'photo', 'image', 'picture', 'shot',
]);

/** Longest chip we keep. A whole sentence that slipped through as one booru
 *  segment is not a tag, and a 90-character chip breaks every layout. */
const MAX_CHIP_LEN = 32;
/** Enough chips to choose from, few enough to read at 400 px. */
export const MAX_CHIPS = 24;

const clean = (s) => (s || '').trim().toLowerCase();

/**
 * Booru or prose? Commas alone are not the test — a prose caption uses them too
 * ("a woman, seen from behind, on a balcony"). What separates the two is that
 * booru segments are SHORT: the tag list has many commas and few words between
 * them. Falling back to prose is the safe direction: prose splitting always
 * yields usable chips, while treating a sentence as one booru tag yields one
 * unusable chip.
 */
export function captionStyle(caption) {
  const parts = (caption || '').split(',').map((s) => s.trim()).filter(Boolean);
  if (parts.length < 3) return 'prose';
  const words = parts.map((p) => p.split(/\s+/).length);
  const avg = words.reduce((a, b) => a + b, 0) / words.length;
  return avg <= 3.5 ? 'booru' : 'prose';
}

/**
 * The chips for one caption, in the order they appear (a reader scanning the
 * caption finds them where they expect), de-duplicated, capped.
 * Returns [] for an empty/absent caption — the caller then says "this image has
 * no caption yet" rather than showing an empty tag row.
 */
export function captionChips(caption) {
  const style = captionStyle(caption);
  const raw = style === 'booru'
    ? (caption || '').split(',')
    // Prose: cut on everything that is not a letter, a digit or an inner
    // hyphen/apostrophe, so "t-shirt" and "woman's" survive as single words.
    : (caption || '').split(/[^\p{L}\p{N}'’-]+/u);
  const out = [];
  const seen = new Set();
  for (const piece of raw) {
    const tag = clean(piece).replace(/^[-'’]+|[-'’]+$/g, '');
    if (!tag || tag.length > MAX_CHIP_LEN) continue;
    // Single letters and bare numbers are never a useful filter.
    if (tag.length < 2 || /^\d+$/.test(tag)) continue;
    if (style === 'prose' && STOP.has(tag)) continue;
    if (seen.has(tag)) continue;
    seen.add(tag);
    out.push(tag);
    if (out.length >= MAX_CHIPS) break;
  }
  return out;
}

/**
 * The tags OF A SELECTION, each with how often it was cited.
 *
 * Asked for in these words: "when the captions are already done and you select an
 * image, show the tags in every case. When several images are selected, show the
 * tags in common with the number of times it was cited."
 *
 * WHY A FREQUENCY COUNT AND NOT AN INTERSECTION. "In common" reads like "keep the
 * tags every image has", but that set answers with N next to every surviving tag —
 * the number says nothing, and on a real selection it is usually empty (twelve
 * captions rarely share one word beyond "woman"). What carries the information is
 * that a tag was cited 7 times out of 12: that is a tag describing over half the
 * selection, which is exactly the judgement the maintainer is making when he looks
 * at this. So: every tag, sorted by how many images cite it, most-cited first.
 *
 * WHAT COUNTS AS AN IMAGE HERE. Three populations, kept apart because collapsing
 * them is how a denominator starts lying:
 *   · `counted`     — contributed at least one chip. THE DENOMINATOR: "7 / 12"
 *                     means 7 of the 12 images that had something to say.
 *   · `uncaptioned` — no caption at all. Nothing to count, and the caller says so
 *                     out loud rather than rendering an empty box.
 *   · `wordless`    — captioned, but every word was a stop word ("a photo of her").
 *                     A different fact from "not captioned", and the fix is a
 *                     different one, so it is not folded into the line above.
 *
 * Ties break alphabetically, not by encounter order: with counts on screen the eye
 * reads down the numbers, and two 4s in caption order look like a sort that broke.
 * ONE captioned image is the exception and keeps CAPTION ORDER, because there are
 * no counts to read down and the caption is right there in the tooltip — a reader
 * scanning it expects to find the chips where he read the words.
 *
 * @param captions array of caption strings (null/'' allowed), one per image
 * @param limit    how many rows to return, most-cited first
 */
export function selectionTagCounts(captions, limit = MAX_CHIPS) {
  const list = captions || [];
  const counts = new Map();
  let counted = 0;
  let uncaptioned = 0;
  let wordless = 0;
  for (const caption of list) {
    if (!clean(caption)) { uncaptioned += 1; continue; }
    const chips = captionChips(caption);
    if (!chips.length) { wordless += 1; continue; }
    counted += 1;
    for (const tag of chips) counts.set(tag, (counts.get(tag) || 0) + 1);
  }
  const all = [...counts.entries()].map(([tag, count]) => ({ tag, count }));
  if (counted > 1) all.sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag));
  return {
    rows: all.slice(0, limit),
    total: all.length,
    counted,
    uncaptioned,
    wordless,
    size: list.length,
  };
}

/** What one chip's number reads as. '' for a single image — "1 / 1" next to every
 *  chip is noise, and the single-image row never had a count to begin with. */
export function tagCountLabel(count, counted) {
  if (!counted || counted < 2) return '';
  return `${count} / ${counted}`;
}

/**
 * The sentences under a selection's chip row: what was counted, and what was NOT.
 *
 * Returned as a list because a selection can be short on more than one count at
 * once, and because an empty list is the caller's signal that there is nothing to
 * disclose. Every omission gets a line — a tag row computed over 9 of 12 images
 * while silently claiming to describe 12 is the same defect as a launch button
 * quoting a number the pass does not walk.
 *
 * @param stats  the return of selectionTagCounts
 * @param unread images in the selection whose caption was not fetched at all
 */
export function selectionTagsNotes(stats, unread = 0) {
  const out = [];
  if (!stats) return out;
  if (stats.counted >= 2) {
    out.push(`Each number is how many of the ${stats.counted} captioned image(s) `
      + 'mention that word.');
  }
  if (stats.uncaptioned > 0) {
    out.push(`${stats.uncaptioned} selected image(s) have no caption yet — they `
      + 'count for nothing here. Run 🏷️ Caption on them and they join in.');
  }
  if (stats.wordless > 0) {
    out.push(`${stats.wordless} selected image(s) have a caption with no word `
      + 'worth filtering on.');
  }
  if (stats.total > stats.rows.length) {
    out.push(`Showing the ${stats.rows.length} most-cited of ${stats.total} `
      + 'distinct tags.');
  }
  if (unread > 0) {
    out.push(`${unread} more selected image(s) are not counted: this row reads the `
      + 'captions it can fetch in one request, and a bigger selection would send a '
      + 'query the server refuses. Narrow the selection for a count that covers it.');
  }
  return out;
}

/** The query value for the `tags` parameter: the ticked chips, comma-joined.
 *  Empty selection ⇒ null, i.e. "no tag filter", never an empty-string filter
 *  that would match everything or nothing depending on the server's mood. */
export function tagsParam(selected) {
  const list = [...(selected || [])].map(clean).filter(Boolean);
  return list.length ? list.join(',') : null;
}

/** How the active tag filter reads above the grid. AND is stated in words, not
 *  left to be inferred from the chips: "red + dress" showing images that have
 *  only one of the two would be a silent lie. */
export function tagFilterSummary(selected) {
  const list = [...(selected || [])];
  if (!list.length) return '';
  if (list.length === 1) return `Showing images whose caption mentions “${list[0]}”.`;
  return `Showing images whose caption mentions ALL of: ${list.map((t) => `“${t}”`).join(' + ')}.`;
}
