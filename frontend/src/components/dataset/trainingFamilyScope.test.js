import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { basesForFamily, cloudUnsupportedFamilyReason } from './trainingFamilyScope.js';

const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');

// The families the panel's own selector offers — the list these helpers must
// stay exhaustive against.
const FAMILIES = ['zimage', 'sdxl', 'krea', 'flux', 'flux2klein', 'anima'];

// A base-info shaped like the server's, with a Z-Image merge in the flat list
// so an accidental fallback is visible rather than merely wrong-labelled.
const BASE_INFO = {
  bases: [
    { value: '', label: 'Official - Z-Image-Turbo (recommended)' },
    { value: 'z image\\bigLove_zt3.safetensors', label: 'bigLove_zt3' },
  ],
  bases_by_type: {
    zimage: [
      { value: '', label: 'Official - Z-Image-Turbo (recommended)' },
      { value: 'z image\\bigLove_zt3.safetensors', label: 'bigLove_zt3' },
    ],
    sdxl: [{ value: 'sdxlBase.safetensors', label: 'sdxlBase' }],
    krea: [{ value: '', label: 'Official - Krea 2' }],
    flux: [{ value: '', label: 'Official - FLUX.1-dev' }],
    flux2klein: [{ value: '', label: 'Official - FLUX.2 Klein' }],
    anima: [{ value: '', label: 'Official - Anima-Base' }],
  },
};

// --- the base list is family-scoped -------------------------------------------

test('each family gets its own list, never another architecture s', () => {
  for (const family of FAMILIES) {
    assert.deepEqual(basesForFamily(BASE_INFO, family),
      BASE_INFO.bases_by_type[family], `${family} was handed the wrong list`);
  }
});

test('a family the server did not enumerate gets nothing, not the Z-Image list', () => {
  // The reported bug, reduced: base-info listed five families and the panel
  // asked for the sixth. `|| bases` answered with Z-Image's catalogue, so the
  // Anima selector read "Official - Z-Image-Turbo (recommended)" and offered
  // this install's Z-Image merges as things an Anima run could load.
  const oldServer = { ...BASE_INFO, bases_by_type: { ...BASE_INFO.bases_by_type } };
  delete oldServer.bases_by_type.anima;
  assert.deepEqual(basesForFamily(oldServer, 'anima'), []);
  // …and a family nobody has heard of behaves identically.
  assert.deepEqual(basesForFamily(oldServer, 'not-a-family'), []);
});

test('Z-Image still reads the flat legacy key when the map is absent', () => {
  // Older payloads carried only `bases`. That key IS the Z-Image list, so it
  // keeps serving Z-Image and no one else.
  const legacy = { bases: BASE_INFO.bases };
  assert.deepEqual(basesForFamily(legacy, 'zimage'), BASE_INFO.bases);
  assert.deepEqual(basesForFamily(legacy, 'anima'), []);
  assert.deepEqual(basesForFamily(legacy, 'krea'), []);
});

test('no base-info at all is an empty list, not a crash', () => {
  assert.deepEqual(basesForFamily(null, 'anima'), []);
  assert.deepEqual(basesForFamily({}, 'zimage'), []);
});

// --- the cloud lane names the families it does not serve ----------------------

test('the three local-only families each state their own refusal', () => {
  assert.match(cloudUnsupportedFamilyReason('sdxl'), /SDXL trains locally only/);
  assert.match(cloudUnsupportedFamilyReason('flux'), /FLUX\.1 trains locally only/);
  // Anima was missing from the ladder: with enough kept images the cloud button
  // enabled itself, and the server refused only after the click.
  assert.match(cloudUnsupportedFamilyReason('anima'), /Anima cloud training is coming/);
});

test('the cloud-served families are not blocked by family', () => {
  for (const family of ['zimage', 'krea', 'flux2klein']) {
    assert.equal(cloudUnsupportedFamilyReason(family), null);
  }
});

// --- the panel actually uses the helpers --------------------------------------

test('TrainingPanel reads its base list and cloud block through this module', () => {
  // Guards the fix itself: an inline `bases_by_type[...] || baseInfo.bases`
  // anywhere in the panel is the bug coming back with a different family.
  // (booleans, not assert.match: a failed match would dump the whole panel)
  assert.ok(panel.includes('basesForFamily('), 'the panel builds its base list inline');
  assert.ok(panel.includes('cloudUnsupportedFamilyReason('),
    'the panel spells its local-only families inline');
  assert.equal(/bases_by_type\?\.\[[^\]]+\]\s*\|\|\s*baseInfo\?\.bases\b/.test(panel), false,
    'the panel still falls back to the Z-Image list for an unlisted family');
});
