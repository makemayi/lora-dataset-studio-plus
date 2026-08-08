import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

/* THE WIRING of the 🏷️ row, which node --test cannot mount (JSX). The counting
 * itself is unit-tested in bankTags.test.js; what is pinned here is that the row
 * is driven by the SELECTION and not by a second click, and that the three ways
 * it can be short of the truth are all said out loud.
 *
 * Asked for in these words: "when the captions are already done and you select an
 * image, show the tags in every case. When several images are selected, show the
 * tags in common with the number of times it was cited."
 */
const ws = readFileSync(new URL('./BankWorkspace.jsx', import.meta.url), 'utf8')
  .replace(/\r\n/g, '\n');

test('the tag row opens on the SELECTION, with no second click', () => {
  // Not `tagSource &&` any more: a live selection produces the row by itself.
  assert.match(ws, /const selectionTags = useMemo\(/);
  assert.match(ws, /\{tagRow && \(/);
  // Priority: a frozen row, then the live selection, then the tile's 🏷️ button.
  assert.match(ws, /const tagRow = tagFreeze \|\| selectionTags \|\| \(tagSource \?/);
});

test('one selected image reads as one image, several read as several', () => {
  assert.match(ws, /🏷️ Tags of the selected image/);
  assert.match(ws, /🏷️ Tags across \$\{tagRow\.size\} selected images/);
  // The tile button keeps its own wording — it is a different gesture.
  assert.match(ws, /🏷️ Tags of \$\{tagRow\.name\}/);
});

test('the tag inspector sits to the right of the Bank gallery on desktop', () => {
  assert.match(ws, /xl:grid-cols-\[minmax\(0,1fr\)_20rem\]/);
  assert.match(ws, /<aside aria-label="Image tags"/);
  assert.match(ws, /hidden xl:col-start-2 xl:row-start-1 xl:block xl:sticky xl:top-20/);
  assert.match(ws, /xl:max-h-\[calc\(100vh-6rem\)\] xl:overflow-y-auto/);
  assert.match(ws, /className="min-w-0 xl:col-start-1 xl:row-start-1"/);
  assert.ok(ws.indexOf('<aside aria-label="Image tags"')
    < ws.indexOf("{filter.flag === 'dups' ? ("));
});

test('mobile keeps the tag inspector in the filter zone', () => {
  assert.match(ws, /<div className="xl:hidden">\s*<SelectionTagsPanel/);
  assert.match(ws, /onToggle=\{\(tag\) => toggleTag\(tag, tagRow\)\}/);
  const mobile = ws.indexOf('<div className="xl:hidden">');
  const filters = ws.indexOf('{/* Filters — grouped by facet');
  const gallery = ws.indexOf("{filter.flag === 'dups' ? (");
  assert.ok(mobile < filters && filters < gallery);
});

test('every chip carries a FRACTION, never a bare count', () => {
  // "7" alone cannot be read; "7 / 12" is the judgement. tagCountLabel returns ''
  // for a single image, so the one-image row is unchanged.
  assert.match(ws, /const n = tagCountLabel\(count, tagRow\.counted\)/);
  assert.match(ws, /cited by \$\{count\} of the \$\{tagRow\.counted\} captioned images you selected/);
});

test('an empty row SAYS why instead of rendering an empty box', () => {
  assert.match(ws, /has no caption yet — run 🏷️ Caption/);
  assert.match(ws, /No caption here has a word worth filtering on/);
});

test('the shortfalls are all disclosed, none folded into the denominator', () => {
  assert.match(ws, /selectionTagsNotes\(tagRow, tagRow\.unread\)/);
});

test('ticking a chip FREEZES the row it was ticked in', () => {
  // setF clears the selection (it must — the ids no longer match the grid). A row
  // that vanished the moment you used it would read as a crash, so the snapshot
  // is taken before the filter is applied.
  assert.match(ws, /const toggleTag = \(tag, basis = null\) =>/);
  assert.match(ws, /if \(basis && basis\.kind === 'selection' && !basis\.frozen\) \{\s*\n\s*setTagFreeze\(\{ \.\.\.basis, frozen: true \}\)/);
  // …and a NEW selection replaces it, so a stale snapshot never sits over fresh ids.
  assert.match(ws, /useEffect\(\(\) => \{ if \(selected\.size\) setTagFreeze\(null\) \}, \[selected\]\)/);
});

test('off-page captions are fetched once, capped, and the cap is disclosed', () => {
  // `ids=` rides in the query string: a few thousand integers build a request line
  // the server refuses. The cap is a constant with its reason, not a magic number.
  assert.match(ws, /const TAG_CAPTION_FETCH_CAP = 500/);
  assert.match(ws, /missing\.slice\(0, TAG_CAPTION_FETCH_CAP\)/);
  // Only ids the grid never rendered are fetched — clicking tiles costs nothing.
  assert.match(ws, /ids\.filter\(\(id\) => !captionCache\.current\.has\(id\)\)/);
});

test('the tile 🏷️ button clears the selection, so it can never read as dead', () => {
  const fn = ws.slice(ws.indexOf('const openTagPicker'), ws.indexOf('const toggleTag'));
  assert.match(fn, /setSelected\(new Set\(\)\)/);
  assert.match(fn, /setTagFreeze\(null\)/);
});
