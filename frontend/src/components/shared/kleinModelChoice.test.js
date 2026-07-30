import assert from 'node:assert/strict';
import test from 'node:test';

import {
  LEGACY_NOTICE, LEGACY_STORAGE_KEY, MODEL_LINE_NONE, MODEL_LINE_UNKNOWN,
  canChooseModel, legacyNotice, modelEndpoint, modelLine, readLegacyPick, selectValue,
} from './kleinModelChoice.js';

test('before the payload arrives, the line promises nothing it cannot name', () => {
  assert.equal(modelLine().text, MODEL_LINE_UNKNOWN);
  assert.equal(modelLine({ loaded: false, effective: 'a.safetensors' }).text,
    MODEL_LINE_UNKNOWN);
});

test('a one-model install still says WHICH model runs', () => {
  // The whole point: hiding a pointless <select> is fine, staying silent is not.
  const line = modelLine({ loaded: true, stored: null, effective: 'flux-2-klein-9b.safetensors' });
  assert.match(line.text, /flux-2-klein-9b\.safetensors/);
  assert.match(line.text, /auto-detected/);
  assert.equal(line.tone, 'muted');
});

test('an explicit choice is named without the auto caveat', () => {
  const line = modelLine({ loaded: true, stored: 'heavy.safetensors', effective: 'heavy.safetensors' });
  assert.equal(line.text, 'Runs on heavy.safetensors.');
  assert.doesNotMatch(line.text, /auto/);
});

test('a chosen model that left the disk is named, and warns', () => {
  const line = modelLine({ loaded: true, stored: 'gone.safetensors', missing: 'gone.safetensors',
    effective: null });
  assert.match(line.text, /gone\.safetensors/);
  assert.equal(line.tone, 'warn');
});

test('no model at all is a warning, not a blank', () => {
  const line = modelLine({ loaded: true, stored: null, effective: null });
  assert.equal(line.text, MODEL_LINE_NONE);
  assert.equal(line.tone, 'warn');
});

test('the legacy browser key is never renamed', () => {
  assert.equal(LEGACY_STORAGE_KEY, 'editPage_flux2KleinModel_v1');
});

test('the legacy pick survives a hostile localStorage', () => {
  assert.equal(readLegacyPick(null), '');
  assert.equal(readLegacyPick({ getItem() { throw new Error('private mode'); } }), '');
  assert.equal(readLegacyPick({ getItem: () => 'x.safetensors' }), 'x.safetensors');
});

test('the carry-over is offered, never applied silently', () => {
  const n = legacyNotice({ stored: null, legacy: 'b.safetensors',
    choices: ['a.safetensors', 'b.safetensors'], effective: 'a.safetensors' });
  assert.equal(n.value, 'b.safetensors');
  assert.equal(n.text, LEGACY_NOTICE);
});

test('nothing is disclosed when there is nothing to disclose', () => {
  const choices = ['a.safetensors', 'b.safetensors'];
  // Already chosen on the dataset.
  assert.equal(legacyNotice({ stored: 'a.safetensors', legacy: 'b.safetensors', choices }), null);
  // No browser value.
  assert.equal(legacyNotice({ stored: null, legacy: '', choices }), null);
  // The browser value is a model this machine no longer has.
  assert.equal(legacyNotice({ stored: null, legacy: 'gone.safetensors', choices }), null);
  // Adopting it would change nothing — auto already resolves to it.
  assert.equal(legacyNotice({ stored: null, legacy: 'a.safetensors', choices,
    effective: 'a.safetensors' }), null);
});

test('a screen with no dataset reads the global state, not a dataset one', () => {
  // The bank's watermark inpaint: no dataset to inherit a pick from, so it reads
  // the model that WILL run and nothing else.
  assert.equal(modelEndpoint(7), '/api/dataset/7/klein-model');
  assert.equal(modelEndpoint(null), '/api/klein-model');
  assert.equal(modelEndpoint(undefined), '/api/klein-model');
});

test('there is exactly one place to CHOOSE a Klein model: the dataset', () => {
  const choices = ['a.safetensors', 'b.safetensors'];
  assert.equal(canChooseModel({ datasetId: 3, choices }), true);
  // A single option is furniture…
  assert.equal(canChooseModel({ datasetId: 3, choices: ['a.safetensors'] }), false);
  // …and a picker with nowhere to save would imply a second authority for the
  // same UNETLoader. Naming stays; choosing does not.
  assert.equal(canChooseModel({ datasetId: null, choices }), false);
});

test('the select falls back to Auto rather than to a neighbour', () => {
  const choices = ['a.safetensors', 'b.safetensors'];
  assert.equal(selectValue({ stored: 'b.safetensors', choices }), 'b.safetensors');
  assert.equal(selectValue({ stored: null, choices }), '');
  // A stored-but-vanished model must NOT show as the first option: that would
  // read as "your choice is fine" while the warning line says the opposite.
  assert.equal(selectValue({ stored: 'gone.safetensors', choices }), '');
});
