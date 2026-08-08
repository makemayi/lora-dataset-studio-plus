/** Several features are driven by a setting that is discoverable only by knowing it
 * exists — the manual Upscale & improve being the reported case: its strength and
 * instruction are editable, and nothing on the button said so. These links live
 * where the user is ACTING, not on the Settings page they are not on. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const read = (p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const link = read('./SettingsLink.jsx');
const registry = read('../settings/registry.js');

test('every section a link points at really exists in the settings registry', () => {
  const known = new Set([...registry.matchAll(/id: '([a-z0-9-]+)'/g)].map((m) => m[1]));
  assert.ok(known.size >= 8, 'settings registry did not parse');
  const files = [
    '../dataset/CaptionToolsBar.jsx',
    '../dataset/TrainingPanel.jsx', '../dataset/ConceptSourcesPanel.jsx',
  ];
  let found = 0;
  for (const f of files) {
    for (const m of read(f).matchAll(/<SettingsLink section="([a-z0-9-]+)"/g)) {
      found += 1;
      assert.ok(known.has(m[1]), `${f} links to unknown settings section "${m[1]}"`);
    }
  }
  assert.ok(found >= 4, `expected links in every surveyed surface, found ${found}`);
});

test('a settings link never triggers the surface it sits on', () => {
  // These sit on top of tiles and lightboxes whose parents open viewers or start
  // jobs; without this a click would do both.
  assert.match(link, /onClick=\{\(e\) => e\.stopPropagation\(\)\}/);
});

test('the improve button still exists on both improve surfaces', () => {
  // The in-context settings links (KleinImproveNote) were removed when the improve
  // pass's 'klein' engine stopped running Klein at all — Krea2 Ostris Edit +
  // SeedVR2 now, see krea_hq_helper. The action itself is unaffected.
  const lightbox = read('../dataset/DatasetLightbox.jsx');
  assert.match(lightbox, /lightboxImproveButtons\(/);
  assert.match(lightbox, /<KleinImproveNote\b/);
  // ...but under KLEIN's button only: the note is about Klein's INSTRUCTION
  // pulling drawn skin towards realism, and SeedVR2 sends no instruction — a
  // drawn dataset is exactly where SeedVR2 is the right click, so warning
  // under it would argue against the pass that fixes the complaint.
  // hidden while it runs — a settings trip mid-job is not the offer being made
  // Two branches now, one per placement: between the buttons in the rail
  // (a column, where that is what attaches it to Klein) and under the whole
  // group in the bottom bar (a row, where a full-width paragraph dropped
  // mid-row stranded the second button on its own line).
  assert.match(lightbox, /\{rail && btn\.showKleinNote && !improvementActive && \(/);
  assert.match(lightbox, /\{!rail && improveButtons\.some\(\(b\) => b\.showKleinNote\)/);
  // The BULK improve is the same action at scale, and it went targetless for a
  // long time: the instruction that spoils one tile spoils the whole selection.
  assert.match(read('../dataset/DatasetGrid.jsx'), /<KleinImproveNote\b/);
});

test('the note offers BOTH improve levers — the words and the amount', () => {
  // "Adjust improve strength" was the only pointer for a long time, and it aims
  // at the knobs. The reported complaint (anime turned realistic, Qeeyana on
  // Reddit) is caused by the INSTRUCTION, which had no pointer at all.
  const note = read('../dataset/KleinImproveNote.jsx');
  assert.match(note, /focus="identity-prompt-klein-improve"/);
  assert.match(note, /focus="klein-improve-strength"/);
});

test('the cloud banner lands on the section holding the key, not the landing page', () => {
  const page = read('../../pages/CloudRunsPage.jsx');
  assert.match(page, /navigate\('\/settings\/training'\)/);
  assert.doesNotMatch(page, /navigate\('\/settings'\)/);
});
