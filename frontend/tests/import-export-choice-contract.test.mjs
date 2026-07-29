/* Two refusals the app made on the user's behalf without saying why
   (reported by Qeeyana on Reddit):
     - every imported image was resampled to 1024 px, with no setting anywhere;
     - a Style dataset could not export its ZIP until every image was captioned,
       with no way past it — which blocked the legitimate "let me caption these
       somewhere else" trip.

   node --test parses .js, not .jsx, so the component halves are asserted on
   source text; the sentence itself is a real module and is imported. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { importPolicyLine } from '../src/components/dataset/importPolicy.js';
import { helpTopics } from '../src/help/helpRegistry.js';

const here = dirname(fileURLToPath(import.meta.url));
const src = (p) => readFileSync(join(here, '..', 'src', p), 'utf8');

const dropzone = src('components/dataset/ImportDropzone.jsx');
const workspace = src('components/dataset/DatasetWorkspace.jsx');
const settings = src('components/settings/CaptioningSection.jsx');

// --- the sentence at the point of import ------------------------------------

test('the import line states the resolution actually configured', () => {
  assert.equal(importPolicyLine({ max_side: 1024, encoding: 'standard' }),
    'resized to 1024 px on the long side, ratio kept (WebP q92)');
  assert.equal(importPolicyLine({ max_side: 2048, encoding: 'lossless' }),
    'resized to 2048 px on the long side, ratio kept (WebP lossless)');
});

test('"no downscale" says so, and names the ceiling it still has', () => {
  const line = importPolicyLine({ max_side: 0, encoding: 'high', ceiling: 8192 });
  assert.match(line, /original size/);
  assert.match(line, /8192 px/);
  assert.match(line, /WebP q100/);
});

test('a missing policy degrades to the shipped default, never to nonsense', () => {
  assert.equal(importPolicyLine(undefined),
    'resized to 1024 px on the long side, ratio kept (WebP q92)');
});

test('the dropzone reads the live policy instead of asserting 1024', () => {
  assert.ok(!dropzone.includes('(normalized to 1024, kept)'),
    'the hardcoded "normalized to 1024" claim must be gone from the UI text');
  assert.ok(dropzone.includes('importPolicyLine(caps.dataset_import)'),
    'the line must be built from the live policy');
  // and it points at the setting that changes it
  assert.ok(dropzone.includes('focus="dataset-import-max-side"'),
    'the hint must link to the setting it describes');
});

// --- the setting exists, is resettable, is documented -----------------------

test('both import knobs are rendered and registered for help', () => {
  for (const id of ['dataset-import-max-side', 'dataset-import-encoding']) {
    assert.ok(settings.includes(`id="${id}"`), `${id} must be rendered`);
    assert.ok(helpTopics.some((t) => t.app?.focus === id),
      `${id} needs a help topic`);
  }
  assert.ok(settings.includes('section="dataset_import" field="max_side"'));
  assert.ok(settings.includes('section="dataset_import" field="encoding"'));
});

test('the resolution choice warns that changing it mid-way is not retroactive', () => {
  assert.ok(settings.includes('from now on'),
    'the setting must say it is not retroactive');
});

// --- the export refusal is a refusal, not a wall ----------------------------

test('a Style export without captions can be confirmed past', () => {
  const guard = workspace.slice(workspace.indexOf('const exportZipGuarded'),
    workspace.indexOf('const importFolderPrompt'));
  assert.ok(guard.includes('isStyle && keptUncaptioned'), 'the guard must still exist');
  assert.ok(!/toast\.error/.test(guard),
    'the Style branch must refuse with a confirm, not a dead-end error toast');
  const style = guard.slice(guard.indexOf('isStyle && keptUncaptioned'));
  assert.match(style, /window\.confirm/);
  // cancelling still walks the user to the captions — the guard-rail stays useful
  assert.match(style, /jumpTo\(\{ targetId: 'gf-captions' \}\)/);
  // and it says WHY, and where the images can come back in
  assert.match(style, /Import dataset/);
});

test('the caption-elsewhere round trip is stated where the ZIP buttons are', () => {
  const at = workspace.indexOf('id="ds-caption-elsewhere"');
  assert.ok(at > 0, 'the round trip must be named next to the import/export buttons');
  const hint = workspace.slice(at, at + 900);
  assert.match(hint, /another tool/);
  assert.match(hint, /never overwritten/);
});
