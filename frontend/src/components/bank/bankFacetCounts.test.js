import { test } from 'node:test';
import assert from 'node:assert/strict';

import { chipCounts, facetDataKey, isFacetFiltered } from './bankFacetCounts.js';

const PAYLOAD = {
  counts: { total: 100, pending: 60, keep: 30, reject: 10, scanned: 100 },
  flags: { blur: 4043, noise: 900 },
  res_buckets: { res_lt_025: 12, res_1_2: 88 },
  framing: { face: 40 },
  origins: { unknown: 100 },
  mediums: { photo: 70 },
  angles: { frontal: 55 },
};

const FILTERED = {
  counts: { total: 12, pending: 0, keep: 0, reject: 12 },
  flags: { blur: 12, noise: 3 },
  res_buckets: { res_lt_025: 2, res_1_2: 10 },
  framing: { face: 5 },
  origins: { unknown: 12 },
  mediums: { photo: 8 },
  angles: { frontal: 6 },
};

test('nothing filtered is not a filter', () => {
  assert.equal(isFacetFiltered(null), false);
  assert.equal(isFacetFiltered({}), false);
  // A sort reorders the same rows — it can never change a count.
  assert.equal(isFacetFiltered({ sort: 'noise_desc' }), false);
});

test('the facets that are meaningful at 0 or empty still count as filters', () => {
  // cluster #0 IS a cluster and '' IS the bank root — a truthiness test here is
  // how "showing everything" gets printed over a narrowed grid.
  assert.equal(isFacetFiltered({ cluster: 0 }), true);
  assert.equal(isFacetFiltered({ style: 0 }), true);
  assert.equal(isFacetFiltered({ subfolder: '' }), true);
});

test('every chip facet is recognised as a filter', () => {
  for (const key of ['status', 'flag', 'search', 'exclude', 'tags', 'resBucket',
    'framing', 'origin', 'medium', 'angle']) {
    assert.equal(isFacetFiltered({ [key]: 'x' }), true, key);
  }
});

test('with no filtered answer the chips keep the bank-wide numbers', () => {
  const { print, wide, filtered } = chipCounts(PAYLOAD, null);
  assert.equal(filtered, false);
  assert.equal(print.flags.blur, 4043);
  assert.equal(wide.flags.blur, 4043);
});

test('the printed count follows the filter, the visibility one does not', () => {
  // The reported defect, in two numbers: the chip must PRINT 12 (what it opens)
  // while still being OFFERED because the bank holds 4 043 of them.
  const { print, wide, filtered } = chipCounts(PAYLOAD, FILTERED);
  assert.equal(filtered, true);
  assert.equal(print.flags.blur, 12);
  assert.equal(wide.flags.blur, 4043);
  for (const key of ['resBuckets', 'framing', 'origins', 'mediums', 'angles']) {
    assert.notDeepEqual(print[key], wide[key], key);
  }
});

test('a chip whose filtered count is 0 is still offered', () => {
  // The way back. A row that hid its empty values would strand the user in the
  // filter they just set.
  const { print, wide } = chipCounts(PAYLOAD, { ...FILTERED, flags: { blur: 0, noise: 0 } });
  assert.equal(print.flags.blur, 0);
  assert.ok(wide.flags.blur > 0, 'the chip is still offered');
});

test('a missing map degrades to empty, never to undefined', () => {
  const { print, wide } = chipCounts(null, null);
  assert.deepEqual(print.flags, {});
  assert.deepEqual(wide.angles, {});
});

test('the data key moves when the data moves and only then', () => {
  const a = facetDataKey(PAYLOAD.counts);
  assert.equal(facetDataKey({ ...PAYLOAD.counts }), a, 'same numbers, same key');
  // A caption written, or any counter the chips do not read, must NOT re-measure.
  assert.equal(facetDataKey({ ...PAYLOAD.counts, caption_todo_keep: 7 }), a);
  // A decision must.
  assert.notEqual(facetDataKey({ ...PAYLOAD.counts, reject: 11 }), a);
  // ...and so must a pass landing.
  assert.notEqual(facetDataKey({ ...PAYLOAD.counts, medium_classified: 5 }), a);
  assert.equal(facetDataKey(null), '');
});

test('re-tuning a threshold re-measures, even though no counter moved', () => {
  // Flag verdicts are recomputed from stored scores, so 🎚 Sharpness min re-cuts
  // 🌫 Blurry without changing a single number in `counts`. Keyed on counts
  // alone, the chips would keep the old thresholds' numbers.
  const th = { sharpness_min: 40, noise_max: 12 };
  const a = facetDataKey(PAYLOAD.counts, th);
  assert.equal(facetDataKey(PAYLOAD.counts, { noise_max: 12, sharpness_min: 40 }), a,
    'key order must not matter');
  assert.notEqual(facetDataKey(PAYLOAD.counts, { ...th, sharpness_min: 55 }), a);
});
