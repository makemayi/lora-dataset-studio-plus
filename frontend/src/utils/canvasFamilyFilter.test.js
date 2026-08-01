import assert from 'node:assert/strict';
import test from 'node:test';
import {
  CANVAS_FAMILY_SELECTION_KEY, availableModelFamilies,
  availableStatusCategories, filterDatasetIdsByFamilies, filterLineageTree,
  filterLineageTreeByFamilies, readCanvasExtraFilters, readFamilySelection,
  resolveFamilySelection, runStatusCategory,
  toggleFamilySelection, writeFamilySelection,
} from './canvasFamilyFilter.js';

const memoryStore = () => {
  const data = {};
  return {
    getItem: (key) => (key in data ? data[key] : null),
    setItem: (key, value) => { data[key] = String(value); },
  };
};

test('all represented model families are selected by default', () => {
  assert.deepEqual(resolveFamilySelection(['zimage', 'krea'], null), ['zimage', 'krea']);
});

test('available families use product order and de-duplicate values', () => {
  const datasets = [{ families: ['anima', 'future'] },
    { families: ['krea', 'zimage', 'anima'] }];
  assert.deepEqual(availableModelFamilies(datasets), ['zimage', 'krea', 'anima', 'future']);
});

test('saved subsets survive reload while stale values are discarded', () => {
  const store = memoryStore();
  writeFamilySelection(store, ['krea', 'gone']);
  assert.deepEqual(resolveFamilySelection(['zimage', 'krea'], readFamilySelection(store)), ['krea']);
});

test('empty and corrupt selections have distinct meanings', () => {
  const store = memoryStore();
  writeFamilySelection(store, []);
  assert.deepEqual(resolveFamilySelection(['zimage'], readFamilySelection(store)), []);
  store.setItem(CANVAS_FAMILY_SELECTION_KEY, 'bad json');
  assert.equal(readFamilySelection(store), null);
  assert.deepEqual(resolveFamilySelection(['zimage'], readFamilySelection(store)), ['zimage']);
});

test('family toggles keep the stable available order', () => {
  const all = ['zimage', 'krea', 'sdxl'];
  assert.deepEqual(toggleFamilySelection(all, 'krea', all), ['zimage', 'sdxl']);
  assert.deepEqual(toggleFamilySelection(['sdxl'], 'zimage', all), ['zimage', 'sdxl']);
});

test('filtering uses family intersection without changing dataset selection', () => {
  const datasets = [{ id: 3, families: ['zimage', 'krea'] },
    { id: 2, families: ['sdxl'] }, { id: 1, families: ['krea'] }];
  const selected = [3, 2];
  assert.deepEqual(filterDatasetIdsByFamilies(datasets, selected, ['krea']), [3]);
  assert.deepEqual(selected, [3, 2]);
  assert.deepEqual(filterDatasetIdsByFamilies(datasets, selected, []), []);
});

test('mixed datasets retain only runs and edges from selected model families', () => {
  const tree = {
    root_id: 1, current_id: 3,
    nodes: [
      { record_id: 1, train_type: 'zimage' },
      { record_id: 2, train_type: 'krea' },
      { record_id: 3, train_type: 'krea' },
    ],
    edges: [{ parent: 1, child: 2 }, { parent: 2, child: 3 }],
  };
  const filtered = filterLineageTreeByFamilies(tree, ['krea']);
  assert.deepEqual(filtered.nodes.map((node) => node.record_id), [2, 3]);
  assert.deepEqual(filtered.edges, [{ parent: 2, child: 3 }]);
  assert.equal(filtered.root_id, 2);
  assert.equal(filtered.current_id, 3);
  assert.equal(tree.nodes.length, 3, 'the cached raw tree is untouched');
});

test('run statuses collapse backend states into useful canvas categories', () => {
  assert.equal(runStatusCategory({ status: 'training' }), 'active');
  assert.equal(runStatusCategory({ status: 'done' }), 'completed');
  assert.equal(runStatusCategory({ status: null, checkpoint_ready: true }), 'completed');
  assert.equal(runStatusCategory({ status: null, source: 'local', checkpoints: [] }), 'completed');
  assert.equal(runStatusCategory({ status: 'error_pod_kept' }), 'error');
  assert.equal(runStatusCategory({ status: null }), 'unknown');
  assert.deepEqual(availableStatusCategories({ one: { tree: { nodes: [
    { status: 'done' }, { status: 'training' }, { status: 'error' },
  ] } } }), ['active', 'completed', 'error']);
});

test('combined run filters search fields and prune dangling edges', () => {
  const tree = { root_id: 1, current_id: 2,
    nodes: [
      { record_id: 1, train_type: 'krea', variant: 'photo', status: 'done' },
      { record_id: 2, train_type: 'zimage', status: 'error' },
    ], edges: [{ parent: 1, child: 2 }] };
  const filtered = filterLineageTree(tree, { families: ['krea'], statuses: ['completed'],
    query: 'alice', datasetName: 'Alice portraits' });
  assert.deepEqual(filtered.nodes.map((node) => node.record_id), [1]);
  assert.deepEqual(filtered.edges, []);
  assert.equal(tree.nodes.length, 2, 'the cached raw tree is untouched');
});

test('extra filters keep an explicit empty status choice and safe defaults', () => {
  const store = memoryStore();
  assert.deepEqual(readCanvasExtraFilters(store), { statuses: null, showPinned: true });
  store.setItem('lds.canvasExtraFilters', JSON.stringify({ statuses: [], showPinned: false }));
  assert.deepEqual(readCanvasExtraFilters(store), { statuses: [], showPinned: false });
});
