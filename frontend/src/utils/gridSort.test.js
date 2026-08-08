import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BANK_SORTS, bankSortOptions, bankSortGroups, bankSortStorageKey,
  loadBankSort, saveBankSort, normalizeBankSort,
  DATASET_SORTS, datasetSortOptions, sortDatasetImages, normalizeDatasetSort,
} from './gridSort.js';
import { filterImagesByStatus } from './gridStatusFilter.js';

// id → the dataset row shape the grid actually receives.
const img = (id, face_score, status = 'keep') => ({
  id, face_score, status, filename: `${id}.png`,
});

// Deliberately unordered scores + two rows the face pass never reached.
const IMAGES = [
  img(5, 0.41), img(4, null), img(3, 0.88), img(2, 0.62), img(1, undefined),
];
const order = (list) => list.map((i) => i.id);

// ---- dataset: the central assertion ---------------------------------------

test('face similarity sorts both ways and sinks the unscored to the END', () => {
  // Closest first: 0.88 > 0.62 > 0.41, then the two unscored — never at the top.
  assert.deepEqual(order(sortDatasetImages(IMAGES, 'face_desc')), [3, 2, 5, 4, 1]);
  // Least alike first: the unscored are STILL last, not first — a "worst first"
  // sort that opened on the un-measured pile would be worse than no sort at all.
  assert.deepEqual(order(sortDatasetImages(IMAGES, 'face_asc')), [5, 2, 3, 4, 1]);
  // Ties (both unscored) fall back to the default newest-first order.
  assert.deepEqual(order(sortDatasetImages([img(1, null), img(9, null)], 'face_desc')),
    [9, 1]);
});

test('default (and an unknown id) is a no-op that keeps the same array', () => {
  assert.equal(sortDatasetImages(IMAGES, 'default'), IMAGES);
  assert.equal(sortDatasetImages(IMAGES, 'bogus-from-an-old-localStorage'), IMAGES);
  assert.equal(normalizeDatasetSort('bogus'), 'default');
  assert.equal(normalizeDatasetSort('face_asc'), 'face_asc');
});

test('sorting never mutates the input', () => {
  const before = order(IMAGES);
  sortDatasetImages(IMAGES, 'face_desc');
  assert.deepEqual(order(IMAGES), before);
});

// ---- dataset: composition with the existing filters ------------------------

test('sort composes with the decision filter — no image lost, none invented', () => {
  const mixed = [
    img(5, 0.41, 'keep'), img(4, 0.95, 'reject'), img(3, 0.88, 'keep'),
    img(2, null, 'keep'), img(1, 0.62, 'reject'),
  ];
  const kept = filterImagesByStatus(mixed, 'kept');
  // Filter → sort and sort → filter agree, and both keep exactly the kept set.
  const filterThenSort = sortDatasetImages(kept, 'face_desc');
  const sortThenFilter = filterImagesByStatus(sortDatasetImages(mixed, 'face_desc'), 'kept');
  assert.deepEqual(order(filterThenSort), [3, 5, 2]);
  assert.deepEqual(order(sortThenFilter), order(filterThenSort));
  assert.deepEqual(new Set(order(filterThenSort)), new Set(order(kept)));
  // The rejected rows stay out even though one of them has the best score.
  assert.ok(!order(filterThenSort).includes(4));
});

test('"select all" follows the active sort — it takes the list the grid shows', () => {
  // DatasetGrid derives BOTH its tiles and its select-all from the same `images`
  // prop, so whatever the workspace sorted+filtered IS the selection universe.
  const shown = sortDatasetImages(filterImagesByStatus(IMAGES, 'kept'), 'face_asc');
  const selectAll = shown.filter((i) => i.filename).map((i) => i.id);
  assert.deepEqual(selectAll, order(shown));
  assert.deepEqual(selectAll, [5, 2, 3, 4, 1]);
});

// ---- menus: an option whose data does not exist is disabled WITH the reason -

test('dataset menu greys out similarity until something is scored', () => {
  const none = datasetSortOptions([img(1, null), img(2, undefined)]);
  assert.deepEqual(none.map((o) => o.disabled), [false, true, true]);
  for (const o of none.slice(1)) {
    assert.match(o.label, /Analyze faces/, 'the reason must be IN the label');
    assert.match(o.title, /No image has a face score yet/);
  }
  // One scored image is enough to make the sort meaningful.
  const some = datasetSortOptions(IMAGES);
  assert.deepEqual(some.map((o) => o.disabled), [false, false, false]);
  assert.deepEqual(some.map((o) => o.id), DATASET_SORTS.map((s) => s.id));
});

test('bank menu greys out what the passes have not produced yet', () => {
  const fresh = bankSortOptions({ scanned: 0, scored: 0 });
  const byId = Object.fromEntries(fresh.map((o) => [o.id, o]));
  assert.equal(byId.default.disabled, false);
  for (const id of ['res_desc', 'res_asc', 'sharp_desc', 'sharp_asc']) {
    assert.equal(byId[id].disabled, true, `${id} should be disabled`);
    assert.match(byId[id].label, /Scan quality/);
  }
  for (const id of ['aesthetic_desc', 'aesthetic_asc']) {
    assert.equal(byId[id].disabled, true, `${id} should be disabled`);
    assert.match(byId[id].label, /Score/);
  }
  // Scanned but not scored: everything the scan measures is usable, everything
  // the ✨ Score pass measures is not — one pass, one whole section of the menu.
  const scanned = Object.fromEntries(
    bankSortOptions({ scanned: 120, scored: 0, faces: 0 }).map((o) => [o.id, o.disabled]));
  for (const id of ['res_desc', 'size_asc', 'sharp_desc', 'noise_asc', 'flat_desc',
    'detail_asc', 'bars_desc', 'jpeg_asc']) {
    assert.equal(scanned[id], false, `${id} should be usable after a scan`);
  }
  for (const id of ['aesthetic_desc', 'nsfw_asc', 'face_desc']) {
    assert.equal(scanned[id], true, `${id} needs its own pass`);
  }
  // Everything measured: nothing greyed out.
  assert.ok(bankSortOptions({ scanned: 120, scored: 120, faces: 120 })
    .every((o) => !o.disabled));
  // Payload not loaded yet — "I don't know" is never rendered as "you can't".
  assert.ok(bankSortOptions(null).every((o) => !o.disabled));
  assert.ok(bankSortOptions(undefined).every((o) => !o.disabled));
});

test('bank sort ids are the ones the server accepts (never rename: they are stored)', () => {
  // Mirrors image_bank_service._SORT_KEYS × (desc, asc), in menu order. A rename
  // here silently breaks a remembered order in someone's localStorage AND stops
  // matching the server, which would drop back to the default without a word.
  assert.deepEqual(BANK_SORTS.map((s) => s.id), [
    'default',
    'res_desc', 'res_asc', 'size_desc', 'size_asc',
    'aesthetic_desc', 'aesthetic_asc', 'nsfw_desc', 'nsfw_asc',
    'sharp_desc', 'sharp_asc', 'noise_desc', 'noise_asc',
    'flat_desc', 'flat_asc', 'detail_desc', 'detail_asc',
    'bars_desc', 'bars_asc', 'jpeg_desc', 'jpeg_asc',
    'face_desc', 'face_asc',
    'yaw_desc', 'yaw_asc',
    'medium_conf_desc', 'medium_conf_asc',
  ]);
  // Every measure is offered BOTH ways — the point of the widening. A one-way
  // sort is the gap the chips already had.
  const ids = new Set(BANK_SORTS.map((s) => s.id));
  for (const id of ids) {
    if (id === 'default') continue;
    const [, key, dir] = id.match(/^(.+)_(desc|asc)$/);
    assert.ok(ids.has(`${key}_${dir === 'asc' ? 'desc' : 'asc'}`), `${id} has no opposite`);
  }
  // Every entry says WHY you would click it, and no title is a restatement of
  // the arrow — the menu is now long enough that labels alone stop helping.
  for (const s of BANK_SORTS) assert.ok(s.title && s.title.length > 12, s.id);
});

// ---- the menu is grouped by the pass that measures it ----------------------

test('bank menu groups by pass, keeps every option, and heads with Default', () => {
  const groups = bankSortGroups({ scanned: 1, scored: 1, faces: 1 });
  assert.deepEqual(groups[0], {
    group: '',
    options: [{ id: 'default', group: '', label: 'Default', disabled: false,
      title: groups[0].options[0].title }],
  });
  assert.deepEqual(groups.map((g) => g.group),
    ['', '📁 File', '✨ Score', '🔎 Scan quality', '🎭 Faces', '🎨 Medium']);
  // Grouping is a RE-ARRANGEMENT: no option invented, none lost, none duplicated.
  const flat = groups.flatMap((g) => g.options.map((o) => o.id));
  assert.deepEqual(flat.slice().sort(), BANK_SORTS.map((s) => s.id).slice().sort());
  assert.equal(new Set(flat).size, flat.length);
});

// ---- the chosen order is remembered PER BANK -------------------------------

/** The two localStorage methods the helpers touch, over a plain Map. */
const fakeStore = (init = {}) => {
  const map = new Map(Object.entries(init));
  return {
    map,
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, v),
    removeItem: (k) => map.delete(k),
  };
};

test('a bank remembers its own order, and one bank never speaks for another', () => {
  const store = fakeStore();
  assert.equal(loadBankSort(7, store), 'default');
  saveBankSort(7, 'sharp_asc', store);
  assert.equal(loadBankSort(7, store), 'sharp_asc');
  // Bank 8 is a different bank: the order that suits a 9 000-image dump is not
  // the one that suits a hand-picked set.
  assert.equal(loadBankSort(8, store), 'default');
  assert.equal(bankSortStorageKey(7), 'lds.bank.7.sort');
});

test('going back to Default FORGETS the preference instead of storing it', () => {
  const store = fakeStore();
  saveBankSort(3, 'nsfw_desc', store);
  saveBankSort(3, 'default', store);
  assert.equal(store.map.size, 0, 'no key left behind for the default order');
  assert.equal(loadBankSort(3, store), 'default');
});

test('a stored order from another build degrades to Default, never to a 500', () => {
  // The value outlives the build that wrote it: a sort this version dropped, a
  // hand-edited key, a corrupted profile. All of them read as "no preference".
  for (const bogus of ['face_score_desc', 'sharp', '', 'null', '{"a":1}']) {
    assert.equal(loadBankSort(1, fakeStore({ 'lds.bank.1.sort': bogus })), 'default');
  }
  assert.equal(normalizeBankSort('sharp_asc'), 'sharp_asc');
  assert.equal(normalizeBankSort(undefined), 'default');
  // A storage that throws on every access (private mode) is not a crash.
  const hostile = { getItem() { throw new Error('denied') },
    setItem() { throw new Error('denied') }, removeItem() { throw new Error('denied') } };
  assert.equal(loadBankSort(1, hostile), 'default');
  assert.equal(saveBankSort(1, 'res_asc', hostile), 'res_asc');
});
