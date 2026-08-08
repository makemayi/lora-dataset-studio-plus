import test from 'node:test';
import assert from 'node:assert/strict';
import { GRID_PAGE_SIZE, clampPage, pageCount, pageSlice } from './gridPaging.js';

const list = (n) => Array.from({ length: n }, (_, i) => ({ id: i + 1 }));

test('an empty list is one empty page, not zero pages', () => {
  assert.equal(pageCount(0), 1);
  const view = pageSlice([], 0);
  assert.deepEqual(view.items, []);
  assert.equal(view.pages, 1);
  assert.equal(view.from, 0);
  assert.equal(view.to, 0);
  assert.equal(view.paged, false);
});

test('a list that fits on one page hides the pager', () => {
  const view = pageSlice(list(GRID_PAGE_SIZE), 0);
  assert.equal(view.paged, false);
  assert.equal(view.items.length, GRID_PAGE_SIZE);
  assert.equal(view.pages, 1);
});

test('one image over the page size starts paging', () => {
  const view = pageSlice(list(GRID_PAGE_SIZE + 1), 0);
  assert.equal(view.paged, true);
  assert.equal(view.pages, 2);
});

test('pages slice the list in order, with a 1-based inclusive label', () => {
  const images = list(6211);
  const first = pageSlice(images, 0, 500);
  assert.equal(first.items.length, 500);
  assert.equal(first.items[0].id, 1);
  assert.equal(first.items[499].id, 500);
  assert.equal(first.from, 1);
  assert.equal(first.to, 500);

  const third = pageSlice(images, 2, 500);
  assert.equal(third.items[0].id, 1001);
  assert.equal(third.from, 1001);
  assert.equal(third.to, 1500);

  const last = pageSlice(images, 12, 500);
  assert.equal(last.items.length, 211);
  assert.equal(last.from, 6001);
  assert.equal(last.to, 6211);
  assert.equal(last.pages, 13);
});

test('every image appears exactly once across the pages', () => {
  const images = list(1234);
  const seen = [];
  for (let p = 0; p < pageCount(1234, 100); p++) {
    seen.push(...pageSlice(images, p, 100).items.map((i) => i.id));
  }
  assert.equal(seen.length, 1234);
  assert.deepEqual(seen, images.map((i) => i.id));
});

test('a page past the end lands on the last real page, never on a blank grid', () => {
  // A filter or a delete shrinks the list under the page the user stands on.
  const view = pageSlice(list(120), 9, 100);
  assert.equal(view.page, 1);
  assert.equal(view.items.length, 20);
  assert.equal(clampPage(9, 120, 100), 1);
  assert.equal(clampPage(-3, 120, 100), 0);
});

test('clampPage survives the values a shrinking list produces', () => {
  assert.equal(clampPage(0, 0), 0);
  assert.equal(clampPage(5, 0), 0);
  assert.equal(clampPage(NaN, 6211), 0);
  assert.equal(clampPage(undefined, 6211), 0);
  assert.equal(clampPage(2.7, 6211, 500), 2);
});

test('a non-array list degrades to an empty page instead of throwing', () => {
  const view = pageSlice(undefined, 3);
  assert.deepEqual(view.items, []);
  assert.equal(view.total, 0);
  assert.equal(view.page, 0);
});

test('the page size is the one the grid measured as smooth', () => {
  // Changing this is a perf decision, not a cosmetic one: 6 211 tiles rendered
  // at once measured ~41 ms/frame while a 300-tile grid measured 6 ms.
  assert.equal(GRID_PAGE_SIZE, 500);
});
