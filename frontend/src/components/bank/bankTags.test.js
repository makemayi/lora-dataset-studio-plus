import test from 'node:test';
import assert from 'node:assert/strict';

import {
  captionChips, captionStyle, tagsParam, tagFilterSummary, MAX_CHIPS,
  selectionTagCounts, selectionTagsNotes, tagCountLabel,
} from './bankTags.js';

// ---- which shape is this caption? ------------------------------------------

test('booru captions are detected by SHORT comma segments, not by commas alone', () => {
  assert.equal(captionStyle('a woman, red dress, balcony, golden hour'), 'booru');
  // Prose uses commas too — treating this as four tags would produce four
  // sentence-long chips nobody can click.
  assert.equal(captionStyle(
    'a woman in a red dress, seen from behind, standing on a stone balcony'), 'prose');
  assert.equal(captionStyle('a woman in a red dress on a balcony'), 'prose');
  // Too few segments to tell: prose is the safe fallback (it always yields
  // usable chips; booru on a sentence yields one unusable one).
  assert.equal(captionStyle('a woman, a balcony'), 'prose');
  assert.equal(captionStyle(''), 'prose');
});

// ---- the chips themselves ---------------------------------------------------

test('a booru caption keeps its tags WHOLE, in order', () => {
  assert.deepEqual(captionChips('a woman, red dress, golden hour, balcony'),
    ['a woman', 'red dress', 'golden hour', 'balcony']);
});

test('a prose caption yields meaningful words, without the grammar', () => {
  const chips = captionChips('A woman in a red dress standing on the balcony');
  assert.deepEqual(chips, ['woman', 'red', 'dress', 'standing', 'balcony']);
  // The stop list must not eat real subjects — these are attributes someone
  // would genuinely filter on.
  for (const kept of ['woman', 'red', 'dress', 'balcony']) {
    assert.ok(chips.includes(kept), kept);
  }
});

test('words that survive every caption are dropped, real ones are not', () => {
  const chips = captionChips('the photo of a young woman with long hair');
  for (const dropped of ['the', 'of', 'a', 'with', 'photo']) {
    assert.ok(!chips.includes(dropped), `${dropped} should be dropped`);
  }
  // 'young' and 'long' are deliberately NOT in the stop list.
  assert.deepEqual(chips, ['young', 'woman', 'long', 'hair']);
});

test('hyphens and apostrophes survive inside a word', () => {
  assert.deepEqual(captionChips('a woman in a t-shirt holding her friend’s hand'),
    ['woman', 't-shirt', 'holding', 'friend’s', 'hand']);
});

test('no empty, duplicate, single-letter or numeric chips', () => {
  const chips = captionChips('a dress, dress,  , 2, x, DRESS, red dress');
  assert.deepEqual(chips, ['a dress', 'dress', 'red dress']);
  assert.equal(new Set(chips).size, chips.length);
});

test('an absent caption yields nothing at all (the UI then says so)', () => {
  assert.deepEqual(captionChips(''), []);
  assert.deepEqual(captionChips(null), []);
  assert.deepEqual(captionChips(undefined), []);
});

test('a caption dump cannot flood the row', () => {
  const long = Array.from({ length: 80 }, (_, i) => `tag${String.fromCharCode(97 + i % 26)}${i}`).join(' ');
  assert.equal(captionChips(long).length, MAX_CHIPS);
  // A whole sentence that slipped through as one booru segment is not a tag.
  assert.deepEqual(captionChips(`${'x'.repeat(40)}, red dress, y, z`), ['red dress']);
});

// ---- what gets sent, and what the user is told -----------------------------

test('the query value is comma-joined, and an empty pick is NO filter', () => {
  assert.equal(tagsParam(new Set(['red dress', 'balcony'])), 'red dress,balcony');
  // null, not '' — an empty-string filter is a filter, and would be a bug the
  // server has to guess about.
  assert.equal(tagsParam(new Set()), null);
  assert.equal(tagsParam(null), null);
});

test('the summary SPELLS OUT that several chips mean AND', () => {
  assert.match(tagFilterSummary(new Set(['red'])), /mentions “red”/);
  const two = tagFilterSummary(new Set(['red', 'dress']));
  assert.match(two, /ALL of/, 'AND must be stated, not inferred from the chips');
  assert.match(two, /“red” \+ “dress”/);
  assert.equal(tagFilterSummary(new Set()), '');
});

// ---- the tags of a SELECTION, with how often each was cited ------------------

test('a selection counts every tag, most-cited first, not the intersection', () => {
  const stats = selectionTagCounts([
    'a woman, red dress, balcony',
    'a woman, red dress, kitchen',
    'a woman, blue dress, balcony',
  ]);
  // "a woman" is in all three; "red dress" and "balcony" in two. An INTERSECTION
  // would keep only "a woman" and print 3 next to it — the count would carry no
  // information at all, and the two-thirds tags would vanish.
  assert.deepEqual(stats.rows, [
    { tag: 'a woman', count: 3 },
    { tag: 'balcony', count: 2 },
    { tag: 'red dress', count: 2 },
    { tag: 'blue dress', count: 1 },
    { tag: 'kitchen', count: 1 },
  ]);
  assert.equal(stats.counted, 3);
  assert.equal(stats.total, 5);
});

test('a tag cited twice in ONE caption still counts as one image', () => {
  // The number means "how many images mention it", not "how many times the word
  // appears" — a captioner repeating itself must not out-rank a real majority.
  const stats = selectionTagCounts([
    'red dress, red dress, red dress, balcony',
    'red dress, kitchen, evening',
  ]);
  assert.equal(stats.rows.find((r) => r.tag === 'red dress').count, 2);
});

test('a single image gets its tags with NO count next to them', () => {
  // ONE captioned image keeps CAPTION order: there are no counts to read down,
  // and the reader expects the chips where he read the words.
  const stats = selectionTagCounts(['a woman, red dress, balcony']);
  assert.deepEqual(stats.rows.map((r) => r.tag), ['a woman', 'red dress', 'balcony']);
  assert.equal(stats.counted, 1);
  // "1 / 1" on every chip is noise; the single-image row never had a count.
  assert.equal(tagCountLabel(1, 1), '');
  assert.equal(tagCountLabel(7, 12), '7 / 12');
});

test('uncaptioned and word-less images are counted apart, never as each other', () => {
  const stats = selectionTagCounts([
    'a woman, red dress',   // counts
    '',                     // no caption at all
    null,                   // idem
    'a photo of her',       // captioned, but every word is a stop word
  ]);
  assert.equal(stats.counted, 1);
  assert.equal(stats.uncaptioned, 2);
  assert.equal(stats.wordless, 1);
  assert.equal(stats.size, 4);
  // The two shortfalls have different fixes, so they get different sentences.
  const notes = selectionTagsNotes(stats).join(' ');
  assert.match(notes, /2 selected image\(s\) have no caption yet/);
  assert.match(notes, /1 selected image\(s\) have a caption with no word/);
});

test('a selection with no caption anywhere says so instead of rendering nothing', () => {
  const stats = selectionTagCounts(['', null, undefined]);
  assert.deepEqual(stats.rows, []);
  assert.equal(stats.counted, 0);
  assert.match(selectionTagsNotes(stats).join(' '), /no caption yet/);
});

test('the row is capped, and says how many tags it is NOT showing', () => {
  // MAX_CHIPS is a per-CAPTION cap too, so 40 distinct tags need more than one
  // caption to exist at all — which is exactly the case this cap is for.
  const captions = [
    Array.from({ length: 20 }, (_, i) => `taga${i}`).join(', '),
    Array.from({ length: 20 }, (_, i) => `tagb${i}`).join(', '),
  ];
  const stats = selectionTagCounts(captions);
  assert.equal(stats.rows.length, MAX_CHIPS);
  assert.equal(stats.total, 40);
  assert.match(selectionTagsNotes(stats).join(' '),
    new RegExp(`Showing the ${MAX_CHIPS} most-cited of 40`));
});

test('images whose caption was never fetched are disclosed, not silently dropped', () => {
  // The count is honest about its own reach: a row computed over 500 of 3 200
  // selected images while presenting itself as "your selection" is the same lie
  // as a launch button quoting a number the pass does not walk.
  const stats = selectionTagCounts(['a woman, red dress', 'a woman, kitchen']);
  assert.deepEqual(selectionTagsNotes(stats, 0).filter((n) => /not counted/.test(n)), []);
  assert.match(selectionTagsNotes(stats, 2700).join(' '),
    /2700 more selected image\(s\) are not counted/);
});

test('the denominator counts the images that SPOKE, not the ones selected', () => {
  // 7 / 12 must mean "7 of the 12 that had tags". Folding the silent ones into
  // the denominator would quietly deflate every tag on a half-captioned bank.
  const stats = selectionTagCounts([
    'red dress', 'red dress', 'balcony', '', '', null,
  ]);
  assert.equal(stats.counted, 3);
  assert.equal(tagCountLabel(2, stats.counted), '2 / 3');
});
