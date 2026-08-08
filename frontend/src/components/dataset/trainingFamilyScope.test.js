import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
  baseOptionSuffix, baseSelectionNote, basesForFamily,
  cloudUnsupportedFamilyReason, isCustomWeightsBase, looksAbsoluteBase,
  typedBaseNote,
} from './trainingFamilyScope.js';

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

// --- absolute no longer means « Custom weights… » on its own -------------------
// The Krea 2 selector lists the checkpoints installed on this machine, and the
// trainer addresses those by ABSOLUTE path (a relative name on Krea is read as
// another family's base and silently ignored). So the panel can no longer decide
// "the user typed this" from absoluteness alone — the catalog decides.

const KREA_CATALOG = [
  { value: '', label: 'Official - Krea 2' },
  { value: 'D:\\ComfyUI\\models\\unet\\Krea\\my_merge.safetensors', label: 'my_merge',
    trainable: true, quantization: '', note: null },
  { value: 'D:\\ComfyUI\\models\\unet\\Krea\\krea2_turbo_fp8.safetensors',
    label: 'krea2_turbo_fp8', trainable: true, quantization: 'bare_cast',
    note: 'krea2_turbo_fp8.safetensors is a quantized cast: 266 of its 432 tensors…' },
  { value: 'D:\\ComfyUI\\models\\unet\\Krea\\packed_fp8.safetensors',
    label: 'packed_fp8', trainable: false, quantization: 'structured',
    note: 'This is a packed inference export…' },
];

test('a base the catalog offers is a dropdown pick, not custom weights', () => {
  // RED before the fix: the panel reopened in « Custom weights… » mode with the
  // path in the free-text field on every reload, and the dropdown showed nothing.
  for (const entry of KREA_CATALOG) {
    assert.equal(isCustomWeightsBase(entry.value, KREA_CATALOG), false, entry.label);
  }
});

test('an absolute path the catalog does NOT offer is custom weights', () => {
  assert.equal(isCustomWeightsBase('D:\\downloads\\some_krea.safetensors', KREA_CATALOG), true);
  assert.equal(isCustomWeightsBase('/opt/models/krea.safetensors', []), true);
  assert.equal(isCustomWeightsBase('\\\\nas\\models\\krea.safetensors', []), true);
});

test('a relative base name is never custom weights, catalog or not', () => {
  // Z-Image merges and SDXL basenames keep their historical meaning.
  assert.equal(isCustomWeightsBase('z image\\bigLove_zt3.safetensors', []), false);
  assert.equal(isCustomWeightsBase('sdxlBase.safetensors', []), false);
  assert.equal(isCustomWeightsBase('', KREA_CATALOG), false);
  assert.equal(isCustomWeightsBase(null, null), false);
  assert.equal(looksAbsoluteBase('C:/models/x.safetensors'), true);
  assert.equal(looksAbsoluteBase('Krea/x.safetensors'), false);
});

// --- what the panel says about the selected base ------------------------------

test('a packed export is an error, an fp8 cast a warning, a clean file nothing', () => {
  assert.equal(baseSelectionNote(KREA_CATALOG, KREA_CATALOG[3].value).level, 'error');
  assert.equal(baseSelectionNote(KREA_CATALOG, KREA_CATALOG[2].value).level, 'warning');
  assert.equal(baseSelectionNote(KREA_CATALOG, KREA_CATALOG[1].value), null);
  assert.equal(baseSelectionNote(KREA_CATALOG, ''), null);
  // A typed path the server never annotated says nothing rather than guessing.
  assert.equal(baseSelectionNote(KREA_CATALOG, 'D:\\downloads\\x.safetensors'), null);
  assert.equal(baseSelectionNote(null, 'anything'), null);
});

// --- a path TYPED into « Custom weights… » gets the same verdict --------------
// It is the one base that cannot be picked from a list — a checkpoint downloaded
// five minutes ago — and it was the only one whose "the trainer cannot load
// this" arrived after the dataset export and, on the cloud lane, after a GPU had
// been rented.

const TYPED = 'D:\\downloads\\fresh_krea.safetensors';
const PACKED_ANSWER = {
  for: TYPED, status: 'ok', filename: 'fresh_krea.safetensors',
  trainable: false, level: 'error', quantization: 'structured',
  note: 'This is a packed inference export…',
};

test('a typed path shows the same verdict a listed one would', () => {
  const note = baseSelectionNote(KREA_CATALOG, TYPED, PACKED_ANSWER);
  assert.equal(note.level, 'error');
  assert.equal(note.text, PACKED_ANSWER.note);
  // …and a clean typed file still says nothing, like a clean listed one.
  assert.equal(baseSelectionNote(KREA_CATALOG, TYPED,
    { ...PACKED_ANSWER, trainable: true, level: '', note: null }), null);
});

test('an answer about ANOTHER path never annotates this one', () => {
  // The mechanism, asserted rather than assumed: the field fires one request per
  // typing pause, so a slow answer for the previous path lands while the box
  // already reads the next one. Without the `for` comparison the panel refuses a
  // file nobody asked it about — and `baseBlocksTrain` disables the Train button
  // on the strength of it.
  const stale = { ...PACKED_ANSWER, for: 'D:\\downloads\\OTHER.safetensors' };
  assert.equal(baseSelectionNote(KREA_CATALOG, TYPED, stale), null);
  assert.equal(typedBaseNote(stale, TYPED), null);
  assert.equal(typedBaseNote(PACKED_ANSWER, TYPED).level, 'error');
  assert.equal(typedBaseNote(null, TYPED), null);
  assert.equal(typedBaseNote(PACKED_ANSWER, ''), null);
});

test('a listed base is never overruled by a typed answer', () => {
  // The catalog is authoritative for what it lists; the typed lookup only fills
  // the gap. An answer that happens to carry a listed value must not turn a
  // clean catalog entry into a refusal.
  const clean = KREA_CATALOG[1].value;
  const hostile = { for: clean, trainable: false, note: 'nonsense' };
  assert.equal(baseSelectionNote(KREA_CATALOG, clean, hostile), null);
});

test('the note carries the server sentence, not a client-side paraphrase', () => {
  // The numbers come from the file header; restating them here would let the two
  // drift, and the whole point of the warning is that it is checkable.
  assert.equal(baseSelectionNote(KREA_CATALOG, KREA_CATALOG[2].value).text,
    KREA_CATALOG[2].note);
});

test('the dropdown tags compromised entries, and only those', () => {
  assert.equal(baseOptionSuffix(KREA_CATALOG[0]), '');   // official, never tagged
  assert.equal(baseOptionSuffix(KREA_CATALOG[1]), '');
  assert.equal(baseOptionSuffix(KREA_CATALOG[2]), ' · fp8 cast');
  assert.equal(baseOptionSuffix(KREA_CATALOG[3]), ' · packed export');
  // Families whose server never sends the annotations (Z-Image, SDXL) stay bare.
  assert.equal(baseOptionSuffix({ value: 'sdxlBase.safetensors', label: 'x' }), '');
  assert.equal(baseOptionSuffix(null), '');
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

test('TrainingPanel decides custom-weights mode through the catalog, not a regex', () => {
  // Six call sites set that mode. A single surviving `looksAbsolute(x)` inside a
  // setCustomBase(...) is enough to bring the bug back on one code path only —
  // the hardest kind to notice, because five reloads out of six behave.
  assert.equal(/setCustomBase\(\s*looksAbsolute\(/.test(panel), false,
    'a setCustomBase site still decides on absoluteness alone');
  assert.ok(panel.includes('isCustomWeightsBase('), 'the panel spells the rule inline');
  assert.ok(panel.includes('baseSelectionNote('), 'the panel builds the base note inline');
  assert.ok(panel.includes('baseOptionSuffix('), 'the panel tags its options inline');
});
