import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const control = readFileSync(new URL('./DatasetCaptionControl.jsx', import.meta.url), 'utf8');
const promptField = readFileSync(new URL('./PromptField.jsx', import.meta.url), 'utf8');
const comparisonSetup = readFileSync(new URL('./StudioRunSetup.jsx', import.meta.url), 'utf8');

test('caption control lists datasets, persists a lock, and asks the random-caption endpoint', () => {
  assert.match(control, /apiFetch\('\/api\/dataset\/list'\)/);
  assert.match(control, /localStorage\.setItem\(STORAGE_KEY, JSON\.stringify\(dataset\)\)/);
  assert.match(control, /postJson\('\/api\/studio\/random-caption', \{ dataset_id: target\.id \}\)/);
  assert.match(control, /setLockedDataset\(choice\)/);
  assert.match(control, /drawCaption\(choice\)/);
});

test('caption control keeps recovery actionable and accessible', () => {
  assert.match(control, /role="dialog" aria-modal="true"/);
  assert.match(control, /useFocusTrap\(dialogRef, open\)/);
  assert.match(control, /Choose or change the caption dataset/);
  assert.match(control, /Caption dataset locked:/);
  assert.match(control, /err\?\.status === 422/);
  assert.match(control, /Choose a dataset/);
});

test('both Studio prompt surfaces use the shared control and protect typed prompts', () => {
  for (const source of [promptField, comparisonSetup]) {
    assert.match(source, /import DatasetCaptionControl from '\.\/DatasetCaptionControl'/);
    assert.match(source, /<DatasetCaptionControl onCaption=\{applyCaption\} \/>/);
    assert.match(source, /Replace the current (test )?prompt with a random dataset caption\?/);
  }
});
