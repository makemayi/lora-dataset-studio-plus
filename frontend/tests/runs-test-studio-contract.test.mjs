/* Node cannot parse JSX, so this text contract protects the four Runs surfaces
   that open the selected dataset in Test Studio. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { getHelpTopic } from '../src/help/helpRegistry.js';
import { WHATS_NEW } from '../src/whatsNew.js';

const source = fs.readFileSync(
  new URL('../src/pages/CloudRunsPage.jsx', import.meta.url),
  'utf8',
);
const guide = fs.readFileSync(
  new URL('../../docs/guide/using-the-app.md', import.meta.url),
  'utf8',
);

test('Runs uses one dataset-aware helper for every Test Studio surface', () => {
  assert.match(source,
    /const openTestStudio = \(id\) => \{\s*if \(id == null\) return;\s*navigate\(`\/dataset\/studio\/\$\{id\}`\);/);
  assert.equal((source.match(/onClick=\{\(\) => openTestStudio\(/g) || []).length, 4,
    'history cards, active local/cloud runs, and folded recent groups stay covered');
  // Text-labelled, and since the 2026-08-10 restyle the glyph beside the label
  // is the app's drawn Studio icon rather than a 🧪 — the LABEL is what this
  // contract is about, so it counts labels and requires the icon to travel
  // with each one.
  assert.equal((source.match(/Test in Studio/g) || []).length, 4,
    'each Runs surface keeps a visible, text-labelled Studio action');
  assert.equal((source.match(/<StudioIcon [^>]*\/> Test in Studio/g) || []).length, 4,
    'every Studio action carries the icon set glyph, none kept an emoji');
  assert.match(source, /data\.local_active\.current\.dataset_id != null/);
  assert.match(source, /group\.datasetId != null/);
});

test('Runs-to-Studio is discoverable in help, the guide, and What’s New', () => {
  const topic = getHelpTopic('runs-test-in-studio');
  assert.equal(topic?.app.route, '/cloud');
  assert.deepEqual(topic?.guide, {
    chapter: 'using-the-app',
    anchor: 'test-a-run-straight-from-runs',
  });
  assert.match(guide, /^## Test a run straight from Runs$/m);
  assert.match(guide, /🧪 Test in Studio/);

  const news = WHATS_NEW.find((entry) => entry.id === '2026-07-30-runs-test-in-studio');
  assert.equal(news?.to, '/cloud');
  assert.match(news?.blurb || '', /🧪 Test in Studio/);
});

test('the live local card never renders the crash payload object as a child', () => {
  // React error #31 took the whole Runs page to the error boundary on
  // 2026-08-29: `local_active.error` is `training_status().error`, which is the
  // crash PAYLOAD ({rc, excerpt, log_tail, dataset_id}), not a sentence.
  assert.doesNotMatch(source, /\{data\.local_active\.error\}/,
    'render the payload through failureChip(), never straight');
  assert.match(source, /failureChip\(data\.local_active\.error\)/);
  assert.match(source, /import \{ failureChip \} from '\.\.\/components\/dataset\/trainingFailure'/);
});

test('the Stop confirm says which of the two endings you are asking for', () => {
  // OneTrainer saves the LoRA when it is cancelled; ai-toolkit is terminated.
  // A confirm that promises the same thing for both is wrong for one of them.
  assert.match(source, /const graceful = local\.trainer === 'onetrainer'/);
  assert.match(source, /asked to stop rather than killed/);
  assert.match(source, /The training process is terminated/,
    'the ai-toolkit wording stays for the lane it is true of');
  // 'stopping' is accepted-not-finished: the card must survive it, because the
  // run is still on the GPU writing its LoRA.
  assert.match(source, /if \(d\.stopping\) \{/);
  const stopping = source.slice(source.indexOf('if (d.stopping) {'));
  const clearsCard = stopping.slice(0, stopping.indexOf('}')).includes('local_active: null');
  assert.equal(clearsCard, false, 'a run that is still saving must keep its card');
});
