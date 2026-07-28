/* Contract: every editable SCALAR setting can be put back to its shipped value,
   and the value it goes back to comes from the SERVER.

   The bug behind this file: "Upscale & improve ▸ Steps" was set to 43 and there
   was no way back to the shipped 4 without knowing the number. The prompt boxes
   had had "Reset to default" since they shipped — because the API sends them
   their default text. The scalars had none, because it didn't send theirs.

   The regression these tests exist to prevent is NOT a missing button; it is a
   button that lies. Copy a default into the JSX, move it server-side six weeks
   later, and "Reset to default" quietly restores a value that is no longer the
   default, while telling the user it is. So: the helpers are unit tested, and
   the section sources are scanned for hardcoded defaults.

   node --test parses .js, not .jsx — hence the pure helpers here and the
   source-text assertions for the components (same approach as
   identityPromptDefaults.test.js). */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  defaultValueAt, isAtDefault, describeDefault, resetAriaLabel, RESET_TO_DEFAULT_TEXT,
} from './settingDefaults.js';

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const SECTION_FILES = ['EnginesSection.jsx', 'CaptioningSection.jsx', 'TrainingSection.jsx',
  'LocalToolsSection.jsx', 'ServerSection.jsx', 'ScrapingSection.jsx', 'MaintenanceSection.jsx'];
const sources = Object.fromEntries(SECTION_FILES.map((f) => [f, read(`./${f}`)]));
const button = read('./ResetToDefault.jsx');
const settingsPage = read('../../pages/SettingsPage.jsx');

// --- the value offered comes from the server -------------------------------

test('the default is looked up in the server payload, never assumed', () => {
  const payload = { klein: { improve_steps: 4 }, engines: { nanobanana_model: '' } };
  assert.equal(defaultValueAt(payload, 'klein', 'improve_steps'), 4);
  // A key the payload does not carry has NO default we can honour — undefined,
  // so the button hides rather than writing a guess.
  assert.equal(defaultValueAt(payload, 'klein', 'nope'), undefined);
  assert.equal(defaultValueAt(payload, 'nope', 'improve_steps'), undefined);
  assert.equal(defaultValueAt(undefined, 'klein', 'improve_steps'), undefined);
  // ...and a falsy default is still a default: 0 and '' must not read as absent.
  assert.equal(defaultValueAt({ klein: { improve_base_lora_strength: 0 } },
    'klein', 'improve_base_lora_strength'), 0);
  assert.equal(defaultValueAt(payload, 'engines', 'nanobanana_model'), '');
});

test('a default that MOVES server-side moves the offered value with it', () => {
  // The whole point. Same lookup, new payload, new answer — nothing to edit here
  // and nothing to edit in the JSX.
  assert.equal(defaultValueAt({ klein: { improve_steps: 4 } }, 'klein', 'improve_steps'), 4);
  assert.equal(defaultValueAt({ klein: { improve_steps: 6 } }, 'klein', 'improve_steps'), 6);
});

test('SettingsPage loads config_defaults and threads it to the sections', () => {
  assert.match(settingsPage, /setConfigDefaults\(data\.config_defaults \|\| \{\}\)/);
  assert.match(settingsPage, /configDefaults,/);          // present in sectionProps
});

// --- the button exists only when it does something -------------------------

test('a field sitting on its default offers no reset', () => {
  assert.equal(isAtDefault(4, 4), true);
  assert.equal(isAtDefault(43, 4), false);
  assert.equal(isAtDefault('', ''), true);
  assert.equal(isAtDefault('auto', 'auto'), true);
  assert.equal(isAtDefault('api', 'auto'), false);
  assert.equal(isAtDefault(true, true), true);
  assert.equal(isAtDefault(false, true), false);
  // A hand-edited config.json may hold the number as a string — same value to
  // the backend, so it must not read as "customised".
  assert.equal(isAtDefault('4', 4), true);
  assert.equal(isAtDefault(2.0, 2), true);
  assert.equal(isAtDefault(0, 0), true);
  // 0 vs '' is a REAL difference (a number field vs a blank-means-auto field).
  assert.equal(isAtDefault(0, ''), false);
  assert.equal(isAtDefault(undefined, 4), false);
});

test('engines.enabled compares as a selection, not as a sequence', () => {
  const shipped = ['nanobanana', 'chatgpt', 'openrouter', 'klein', 'krea'];
  assert.equal(isAtDefault([...shipped].reverse(), shipped), true);   // re-ticked, same set
  assert.equal(isAtDefault(shipped.slice(1), shipped), false);        // one unticked
  assert.equal(isAtDefault([], shipped), false);
});

test('the button renders nothing when the value is the default, or unknown', () => {
  assert.match(button, /if \(def === undefined\) return null/);
  assert.match(button, /if \(isAtDefault\(current, def\)\) return null/);
});

// --- "blank" stays blank ---------------------------------------------------

test('resetting a blank-means-default field writes the blank back, not a number', () => {
  // engines.nanobanana_model, engines.chatgpt_image_model, krea.base_model,
  // paths.dataset_images_root, klein.small_image_prompt: '' means "work it out
  // yourself" (and, for the two model slugs, "let my NANOBANANA_MODEL env var
  // decide"). The reset writes DEFAULTS' own value, which for these IS '', so
  // the field goes back to implicit — it never gets pinned to whatever the app
  // resolves today, and the user keeps following future changes.
  const payload = { engines: { nanobanana_model: '' }, krea: { base_model: '' },
    paths: { dataset_images_root: '' }, klein: { small_image_prompt: '' } };
  for (const [section, field] of [['engines', 'nanobanana_model'], ['krea', 'base_model'],
    ['paths', 'dataset_images_root'], ['klein', 'small_image_prompt']]) {
    const def = defaultValueAt(payload, section, field);
    assert.equal(def, '', `${section}.${field} default must stay blank`);
    // ...and a blank field shows no button at all, because it is already there.
    assert.equal(isAtDefault('', def), true);
  }
});

test('the button writes the served default verbatim — no local massaging', () => {
  assert.match(button, /setField\(section, field, detach\(def\)\)/);
  // detach = deep copy, so writing a list default cannot alias the payload
  assert.match(button, /JSON\.parse\(JSON\.stringify\(v\)\)/);
});

// --- no default may live in the frontend -----------------------------------

test('no Settings section hardcodes a config default as a display fallback', () => {
  // `value={config.<section>?.<key> ?? 4}` is the exact shape that used to hold
  // a second copy of DEFAULTS (klein improve_* , krea, bank, cloud, ollama) —
  // and one of them, improve_consistency_strength, had ALREADY drifted (0 in the
  // JSX, 1.0 in config.py). Read the server payload instead.
  const offenders = [];
  for (const [name, src] of Object.entries(sources)) {
    for (const m of src.matchAll(/value=\{[^}]*\?\?\s*-?\d[\d.]*\s*\}/g)) {
      offenders.push(`${name}: ${m[0]}`);
    }
  }
  assert.deepEqual(offenders, []);
});

test('the sources read the defaults through the shared lookup', () => {
  for (const name of ['EnginesSection.jsx', 'CaptioningSection.jsx', 'TrainingSection.jsx',
    'LocalToolsSection.jsx']) {
    assert.match(sources[name], /import \{ defaultValueAt \} from '\.\/settingDefaults\.js'/,
      `${name} must read defaults from the payload`);
  }
});

// --- coverage: the fields that had no way back to their default ------------

// section, config key, and the file that must offer its reset.
const COVERED = [
  // Image engines — the reported gap. "Upscale & improve ▸ Steps" is the last one.
  ['EnginesSection.jsx', 'engines', 'default'],
  ['EnginesSection.jsx', 'engines', 'enabled'],
  ['EnginesSection.jsx', 'engines', 'chatgpt_auth'],
  ['EnginesSection.jsx', 'engines', 'nanobanana_model'],
  ['EnginesSection.jsx', 'engines', 'chatgpt_image_model'],
  ['EnginesSection.jsx', 'engines', 'openrouter_model'],
  ['EnginesSection.jsx', 'klein', 'generation_steps'],
  ['EnginesSection.jsx', 'klein', 'improve_megapixels'],
  ['EnginesSection.jsx', 'klein', 'improve_base_lora_strength'],
  ['EnginesSection.jsx', 'klein', 'improve_consistency_strength'],
  ['EnginesSection.jsx', 'klein', 'improve_steps'],
  ['EnginesSection.jsx', 'krea', 'grounding_px'],
  ['EnginesSection.jsx', 'krea', 'steps'],
  ['EnginesSection.jsx', 'krea', 'base_model'],
  ['EnginesSection.jsx', 'krea', 'identity_lora'],
  // the same gap in the other sections
  ['CaptioningSection.jsx', 'captioning', 'backend'],
  ['CaptioningSection.jsx', 'watermark', 'device'],
  ['CaptioningSection.jsx', 'face_scoring', 'green'],
  ['CaptioningSection.jsx', 'face_scoring', 'orange'],
  ...['sharpness_min', 'noise_max', 'uniformity_min', 'min_side', 'detail_min', 'bars_max',
    'dup_distance', 'face_threshold', 'aesthetic_min', 'nsfw_max', 'style_threshold',
    'semantic_dup_threshold'].map((k) => ['CaptioningSection.jsx', 'bank', k]),
  ...['max_concurrent_runs', 'max_price_per_hour', 'monthly_budget_usd', 'stall_timeout_minutes',
    'freeze_watchdog_minutes', 'unreachable_grace_minutes', 'min_reliability']
    .map((k) => ['TrainingSection.jsx', 'cloud', k]),
  // Concept face masking (issue #15) — both knobs are user-tunable, so both must
  // have a way back to the shipped value.
  ['TrainingSection.jsx', 'face_mask', 'expand'],
  ['TrainingSection.jsx', 'face_mask', 'min_weight'],
  ['LocalToolsSection.jsx', 'comfyui', 'object_info_timeout_s'],
  ['LocalToolsSection.jsx', 'ollama', 'vision_concurrency'],
  ['LocalToolsSection.jsx', 'ollama', 'vision_keep_warm_seconds'],
  ['ServerSection.jsx', 'server', 'port'],
  ['ScrapingSection.jsx', 'klein', 'small_image_prompt'],
  ['MaintenanceSection.jsx', 'paths', 'dataset_images_root'],
];

/* Which `section.field` pairs a file actually offers a reset for.

   Most call sites spell the key out literally, which the repo prefers (it keeps
   a config key greppable). Two render one reset for a LIST of fields — the three
   engine model slugs through ModelField, the four "Upscale & improve" knobs
   through IMPROVE_KNOBS — so their key arrives as a prop; those are resolved
   back to the literal keys the call sites / the list provide. An unrecognised
   dynamic form THROWS rather than silently passing: a reset this test cannot
   read is a reset it cannot guarantee. */
const resetPairs = (src) => {
  const pairs = new Set();
  const re = /<ResetToDefault[^>]*section="(\w+)" field=(?:"(\w+)"|\{([\w.]+)\})/g;
  for (const [, section, literal, expr] of src.matchAll(re)) {
    if (literal) { pairs.add(`${section}.${literal}`); continue; }
    if (expr === 'configKey') {
      for (const m of src.matchAll(/configKey="(\w+)"/g)) pairs.add(`${section}.${m[1]}`);
    } else if (expr === 'k.key') {
      for (const m of src.matchAll(/\{ key: '(\w+)', label:/g)) pairs.add(`${section}.${m[1]}`);
    } else {
      throw new Error(`unresolvable ResetToDefault field expression: {${expr}}`);
    }
  }
  return pairs;
};
const PAIRS = Object.fromEntries(Object.entries(sources).map(([f, s]) => [f, resetPairs(s)]));

for (const [file, section, field] of COVERED) {
  test(`${section}.${field} can be put back to its default`, () => {
    assert.ok(PAIRS[file].has(`${section}.${field}`),
      `${file} offers no reset for ${section}.${field} — ${[...PAIRS[file]].join(', ')}`);
  });
}

test('every reset carries a label, for the accessible name', () => {
  for (const [name, src] of Object.entries(sources)) {
    for (const m of src.matchAll(/<ResetToDefault[^>]*>/g)) {
      assert.match(m[0], /label=/, `${name}: a reset button without a label`);
    }
  }
});

// --- accessibility ---------------------------------------------------------

test('the accessible name names the field AND the value it would write', () => {
  assert.equal(resetAriaLabel('Steps', 4), 'Reset to default: Steps, 4');
  assert.equal(resetAriaLabel('Nano Banana (Gemini) model', ''),
    'Reset to default: Nano Banana (Gemini) model, blank');
  // It STARTS with the visible text verbatim (WCAG 2.5.3 label-in-name: a voice
  // user says what they can read), then disambiguates among a dozen identical
  // buttons on one page.
  for (const [label, def] of [['Steps', 4], ['Port', 5050], ['Enabled engines', ['a', 'b']]]) {
    assert.ok(resetAriaLabel(label, def).startsWith(RESET_TO_DEFAULT_TEXT));
  }
});

test('defaults are spelled out, never left as a raw literal', () => {
  assert.equal(describeDefault(''), 'blank');
  assert.equal(describeDefault(null), 'blank');
  assert.equal(describeDefault(true), 'on');
  assert.equal(describeDefault(false), 'off');
  assert.equal(describeDefault(0), '0');
  assert.equal(describeDefault(0.72), '0.72');
  assert.equal(describeDefault(['a', 'b']), 'a, b');
  assert.equal(describeDefault([]), 'nothing selected');
});

test('it is a real button, and its state is not carried by colour', () => {
  assert.match(button, /<button\s/);
  assert.match(button, /type="button"/);
  assert.match(button, /aria-label=\{resetAriaLabel\(label, def\)\}/);
  // presence + words carry the meaning; the glyph is decorative
  assert.match(button, /aria-hidden="true"/);
  assert.match(button, /RESET_TO_DEFAULT_TEXT/);
});

test('it uses the same words as the prompt boxes, which shipped first', () => {
  const promptField = read('../common/PromptOverrideField.jsx');
  assert.ok(promptField.includes(RESET_TO_DEFAULT_TEXT),
    'the prompt field owns the wording — do not grow a second vocabulary');
});
