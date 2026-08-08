import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MAX_MOUNTED_PLAYERS, clipFragmentSrc, clipLabel, playerBudgetWarning,
  shouldRemountPlayer,
} from './videoClipFragment.js';

// --- the media fragment ------------------------------------------------------

test('a clip plays a range of its SOURCE, never a file of its own', () => {
  // The bank writes no clip files at all: cutting means re-encoding, and we only
  // pay that at promotion, for the clips actually kept. So playback has to point
  // at the source with a time range.
  const src = clipFragmentSrc('/api/video-banks/3/source/12/stream', 41.2, 46.3);
  assert.match(src, /#t=41\.2,46\.3$/);
  assert.ok(src.startsWith('/api/video-banks/3/source/12/stream'));
});

test('bounds keep their sub-second precision in the fragment', () => {
  // Rounding to whole seconds moves a quarter of a two-second clip.
  assert.match(clipFragmentSrc('/s', 41.24, 46.31), /#t=41\.24,46\.31$/);
});

test('a trailing fragment on the base url is replaced, not appended', () => {
  // Re-deriving a src from an already-fragmented one must not produce
  // "#t=1,2#t=3,4", which browsers parse as neither.
  assert.match(clipFragmentSrc('/s#t=1,2', 5, 9), /^\/s#t=5,9$/);
});

test('negative or inverted bounds do not produce a fragment a browser ignores', () => {
  // A malformed fragment does not error — the browser silently plays the WHOLE
  // file, which on a two-hour rush is the worst possible failure.
  assert.equal(clipFragmentSrc('/s', -3, 5), '/s#t=0,5');
  assert.equal(clipFragmentSrc('/s', 9, 4), null);
});

// --- keeping exactly one player alive ----------------------------------------

test('the mounted-player budget stays far under the browser ceiling', () => {
  // Chrome caps WebMediaPlayers at ~60 in total, leaving ~40 usable <video>
  // elements per page; past that, new elements simply never load, with no error.
  // The grid shows JPEG thumbnails only and playback happens in a single
  // lightbox, so this budget is not a workaround — it is the design.
  assert.equal(MAX_MOUNTED_PLAYERS, 1);
});

test('a warning is produced only if something ever mounts more than one player', () => {
  assert.equal(playerBudgetWarning(1), null);
  assert.match(playerBudgetWarning(2), /one/i);
});

test('switching clips within one source remounts the player', () => {
  // Assigning a new #t to a live <video> is not reliable across browsers: some
  // ignore the fragment once the resource is loaded, and the viewer silently
  // watches the previous clip's range.
  assert.equal(shouldRemountPlayer({ sourceId: 7, start: 1 },
                                   { sourceId: 7, start: 30 }), true);
});

test('re-rendering the same clip does not remount the player', () => {
  // Remounting on every render restarts playback from the head, which makes the
  // lightbox unusable.
  assert.equal(shouldRemountPlayer({ sourceId: 7, start: 1 },
                                   { sourceId: 7, start: 1 }), false);
});

test('closing the lightbox unmounts the player', () => {
  assert.equal(shouldRemountPlayer({ sourceId: 7, start: 1 }, null), true);
});

// --- what the user reads ------------------------------------------------------

test('a clip is labelled by its position in the source, not by an opaque id', () => {
  assert.equal(clipLabel(41.2, 46.3), '0:41 – 0:46 (5.1s)');
});

test('a clip past the hour mark is still readable', () => {
  assert.equal(clipLabel(3725.0, 3730.0), '1:02:05 – 1:02:10 (5.0s)');
});
