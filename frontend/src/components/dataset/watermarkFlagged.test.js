import test from 'node:test';
import assert from 'node:assert/strict';
import {
  summarizeFlagged, rejectableFlagged, hasWatermarkPosition,
  rejectFlaggedConfirmText, flaggedSourceNote,
} from './watermarkFlagged.js';

// One list carrying every shape the pile can really hold.
const IMAGES = [
  { id: 1, status: 'keep', watermark_state: 'detected', watermark_bbox: [0, 0, 0.2, 0.1], watermark_source: 'detector' },
  { id: 2, status: 'keep', watermark_state: 'detected', watermark_bbox: null, watermark_source: 'detector' },
  { id: 3, status: 'keep', watermark_state: 'detected', watermark_bbox: [0, 0, 0.2, 0.1], watermark_source: 'vision' },
  { id: 4, status: 'keep', watermark_state: 'detected', watermark_bbox: [0, 0, 0.2, 0.1] },      // legacy row
  { id: 5, status: 'failed', watermark_state: 'detected', watermark_bbox: [0, 0, 0.2, 0.1] },    // server skips it
  { id: 6, status: 'keep', watermark_state: 'detected', watermark_bbox: [0, 0, 0.2, 0.1], derivation_kind: 'small_image_source' },
  { id: 7, status: 'keep', watermark_state: 'none' },
  { id: 8, status: 'keep', watermark_state: 'dismissed' },
  { id: 9, status: 'reject', watermark_state: null },
];

test('the rejectable set excludes exactly what the server would not reject', () => {
  // 5 is skipped by the batch loop, 6 makes the WHOLE request 400 — both would
  // turn an announced count into a promise the click cannot keep.
  assert.deepEqual(rejectableFlagged(IMAGES).map((i) => i.id), [1, 2, 3, 4]);
});

test('the summary counts the pile without conflating its parts', () => {
  const s = summarizeFlagged(IMAGES);
  assert.equal(s.flagged, 6);
  assert.equal(s.rejectable, 4);
  assert.deepEqual(s.rejectableIds, [1, 2, 3, 4]);
  assert.equal(s.heldBack, 2);
  assert.equal(s.unlocated, 1);          // only id 2 has no position at all
  assert.equal(s.dismissed, 1);
  assert.deepEqual(s.bySource, { detector: 2, vision: 1, unknown: 3 });
});

test('an empty or undefined list is a zeroed summary, not a crash', () => {
  assert.deepEqual(summarizeFlagged(undefined).rejectableIds, []);
  assert.equal(summarizeFlagged([]).flagged, 0);
  assert.equal(flaggedSourceNote(summarizeFlagged([])), '');
});

test('hand-drawn zones count as a position; an empty override does not', () => {
  assert.equal(hasWatermarkPosition({ watermark_regions: [[0, 0, 0.1, 0.1]] }), true);
  assert.equal(hasWatermarkPosition({ watermark_regions: [], watermark_bbox: [0, 0, 1, 1] }), false);
  assert.equal(hasWatermarkPosition({ watermark_bbox: [0, 0, 1, 1] }), true);
  assert.equal(hasWatermarkPosition({}), false);
});

test('the confirmation states the exact number, the way back and what it destroys', () => {
  const text = rejectFlaggedConfirmText(summarizeFlagged(IMAGES));
  assert.match(text, /Reject 4 flagged image\(s\)\?/);
  assert.match(text, /Show ▸ Rejected/);
  assert.match(text, /✓ Keep/);
  assert.match(text, /clears their watermark flags/);
  // The two that cannot be included are named rather than silently dropped.
  assert.match(text, /2 more flagged image\(s\) are NOT included/);
});

test('the confirmation stays quiet when nothing is held back', () => {
  const clean = IMAGES.filter((i) => ![5, 6].includes(i.id));
  const text = rejectFlaggedConfirmText(summarizeFlagged(clean));
  assert.match(text, /Reject 4 flagged image\(s\)\?/);
  assert.doesNotMatch(text, /NOT included/);
});

test('the provenance line appears only when there is something to disambiguate', () => {
  const oneSourceAllLocated = [
    { id: 1, status: 'keep', watermark_state: 'detected', watermark_bbox: [0, 0, 1, 1], watermark_source: 'vision' },
  ];
  assert.equal(flaggedSourceNote(summarizeFlagged(oneSourceAllLocated)), '');
  const note = flaggedSourceNote(summarizeFlagged(IMAGES));
  assert.match(note, /2 by the watermark detector/);
  assert.match(note, /1 by the vision model/);
  assert.match(note, /3 before the source was recorded/);
  assert.match(note, /1 flagged without a position/);
});

test('a single-source pile still says when a flag has no position', () => {
  const blind = [
    { id: 1, status: 'keep', watermark_state: 'detected', watermark_bbox: null, watermark_source: 'detector' },
  ];
  const note = flaggedSourceNote(summarizeFlagged(blind));
  assert.match(note, /1 flagged without a position/);
  assert.doesNotMatch(note, /Judged/);
});
