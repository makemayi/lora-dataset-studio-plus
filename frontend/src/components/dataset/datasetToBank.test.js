import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  canStartDatasetToBank, datasetToBankRequest, datasetToBankUrl,
} from './datasetToBank.js';

test('dataset → bank sends an explicit preservation choice, defaulting to keep it', () => {
  assert.equal(datasetToBankUrl(), '/api/bank/from-dataset');
  assert.deepEqual(datasetToBankRequest('42', '  Review later  '), {
    dataset_id: 42,
    name: 'Review later',
    preserve_analysis: true,
  });
  assert.deepEqual(datasetToBankRequest(42, 'Fresh review', false), {
    dataset_id: 42,
    name: 'Fresh review',
    preserve_analysis: false,
  });
});

test('the dialog only starts with a nonblank name and keeps an accessible fresh-start choice', () => {
  assert.equal(canStartDatasetToBank({ name: 'Bank' }), true);
  assert.equal(canStartDatasetToBank({ name: '   ' }), false);
  assert.equal(canStartDatasetToBank({ name: 'Bank', busy: true }), false);

  const dialog = readFileSync(new URL('./DatasetToBankDialog.jsx', import.meta.url), 'utf8');
  assert.match(dialog, /useFocusTrap\(dialogRef, true\)/);
  assert.match(dialog, /role="dialog"/);
  assert.match(dialog, /aria-modal="true"/);
  assert.match(dialog, /Both choices keep Dataset-owned metadata/);
  assert.match(dialog, /Reuse compatible final-file analysis/);
  assert.match(dialog, /Start fresh analysis/);
  assert.match(dialog, /skip reuse of prior analysis/);
  assert.doesNotMatch(dialog, /starts unanalysed/);
  assert.match(dialog, /role="alert"/);
});

test('the workspace opens the dialog instead of using a native prompt for this action', () => {
  const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');
  const start = workspace.indexOf('const importToBank');
  const end = workspace.indexOf('const exportZipGuarded', start);
  const action = workspace.slice(start, end);
  assert.ok(start >= 0 && end > start, 'import-to-bank action must remain identifiable');
  assert.doesNotMatch(action, /window\.prompt/);
  assert.match(workspace, /<DatasetToBankDialog/);
  assert.match(workspace, /Both choices keep Dataset-owned captions/);
  assert.match(workspace, /Start fresh skips only reuse of prior analysis/);
  assert.doesNotMatch(workspace, /start unanalysed/);
});
