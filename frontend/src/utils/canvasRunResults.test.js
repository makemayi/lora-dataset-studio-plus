import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CANVAS_RUN_KEY, canvasResultLabel, canvasRunDatasetIds, describeCanvasRun,
  normaliseTargets, readCanvasRun, readyImageCount, runPinCandidates, writeCanvasRun,
} from './canvasRunResults.js';

const fakeStore = (initial = {}) => {
  const m = new Map(Object.entries(initial));
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => m.set(k, String(v)),
    removeItem: (k) => m.delete(k),
    _map: m,
  };
};

test('a launched run is remembered WITH its checkpoints, and survives a reload', () => {
  // The pairing is the point: the run id alone tells you something is running;
  // it is the checkpoints that let the board say where the images landed.
  const store = fakeStore();
  writeCanvasRun(store, { runId: 'abc-123',
    targets: [{ datasetId: 4, recordId: 12, step: 2000, datasetName: 'a' }] });
  const back = readCanvasRun(store);
  assert.equal(back.runId, 'abc-123');
  assert.deepEqual(back.targets,
    [{ datasetId: 4, recordId: 12, step: 2000, datasetName: 'a' }]);
  assert.ok(store._map.has(CANVAS_RUN_KEY));
});

test('forgetting a run clears the entry, and a corrupt one reads as none', () => {
  const store = fakeStore({ [CANVAS_RUN_KEY]: '{"runId":"x"}' });
  writeCanvasRun(store, null);
  assert.equal(readCanvasRun(store), null);
  assert.equal(readCanvasRun(fakeStore({ [CANVAS_RUN_KEY]: 'not json' })), null);
  assert.equal(readCanvasRun(fakeStore({ [CANVAS_RUN_KEY]: '{"runId":7}' })), null);
  assert.equal(readCanvasRun(null), null);
});

test('half-formed targets are dropped rather than rendered as buttons that open nothing', () => {
  assert.deepEqual(normaliseTargets([
    { datasetId: 4, recordId: 12, step: 2000 },
    { datasetId: 0, recordId: 12, step: 2000 },      // not a database id
    { datasetId: 4, recordId: null, step: 2000 },
    { datasetId: 4, recordId: 12, step: 'final' },   // not a step
    null,
  ]), [{ datasetId: 4, recordId: 12, step: 2000, datasetName: null }]);
  assert.deepEqual(normaliseTargets(undefined), []);
});

test('"ready" counts images that EXIST, not cells that were queued', () => {
  // A cancelled or failed cell never becomes a file, and telling the user three
  // images are waiting for them when one is would be worse than saying nothing.
  assert.equal(readyImageCount({ cells: [{ filename: 'a.png' }, { filename: null }, {}] }), 1);
  assert.equal(readyImageCount(null), 0);
});

test('the tracker says what the run is doing, in the Studio’s own words', () => {
  const working = describeCanvasRun({ pending: 3, generating: 1, queued: 2, cells: [] });
  assert.equal(working.phase, 'working');
  assert.equal(working.text, '1 generating · 2 queued');

  const stopped = describeCanvasRun({ pending: 0, resumable: 2, cells: [] });
  assert.equal(stopped.phase, 'stopped');
  assert.match(stopped.text, /2 stopped images — resumable/);

  const done = describeCanvasRun({ pending: 0, resumable: 0,
    cells: [{ filename: 'a.png' }, { filename: 'b.png' }] });
  assert.equal(done.phase, 'done');
  assert.equal(done.text, '2 images ready');
  assert.equal(done.ready, 2);
});

test('nothing to report draws no bar at all', () => {
  assert.equal(describeCanvasRun(null).phase, 'idle');
  assert.equal(describeCanvasRun({ pending: 0, resumable: 0, cells: [] }).phase, 'idle');
});

test('a finished run knows which lanes to re-read, once each', () => {
  // Without this re-read the board looks exactly as it did before the launch:
  // the × N badge and the new thumbnail come from the lineage, not from the run.
  assert.deepEqual(canvasRunDatasetIds([
    { datasetId: 4, recordId: 1, step: 100 },
    { datasetId: 4, recordId: 2, step: 200 },
    { datasetId: 9, recordId: 3, step: 300 },
  ]), [4, 9]);
});

test('a result button names the checkpoint by the ids shown on the board', () => {
  assert.equal(canvasResultLabel({ recordId: 12, step: 2000 }), '#12 · step 2000');
});

test('the pinnable lot is the cells that produced a FILE, with their lane', () => {
  const run = { cells: [
    { id: 11, dataset_id: 3, filename: 'a.png', status: 'done' },
    { id: 12, dataset_id: 3, filename: null, status: 'cancelled' },
    { id: 13, dataset_id: 8, filename: 'c.png', status: 'done' },
    { id: 14, dataset_id: null, filename: 'd.png' },
  ] };
  assert.deepEqual(runPinCandidates(run),
    [{ id: 11, datasetId: 3 }, { id: 13, datasetId: 8 }]);
  assert.deepEqual(runPinCandidates(null), []);
  // A cell with a file but no lane is ready (it exists on disk) and NOT
  // pinnable (there is no lane to put it in) — the two counts are allowed to
  // disagree, and the button must follow the pinnable one.
  assert.equal(readyImageCount(run), 3);
});
