import test from 'node:test';
import assert from 'node:assert/strict';
import { downloadLabel, formatDownloadProgress } from './downloadProgress.js';

// Exactly what the backend parsed out of the run-121 pod log on 2026-07-28,
// the run that sat two hours behind one fixed sentence.
const RUN_121 = {
  label: 'raw.safetensors', percent: 7, done: '1.95G', total: '26.3G',
  elapsed: '15:30', eta: '2:37:06', speed: '2.58MB/s',
};

test('the real run-121 bar becomes a decidable line', () => {
  const f = formatDownloadProgress(RUN_121);
  assert.equal(f.headline, 'Fetching model weights — 1.95G of 26.3G (7%)');
  assert.equal(f.detail, '2.58MB/s · ETA 2:37:06 · 15:30 elapsed');
  assert.equal(f.percent, 7);
});

test('the announced label is quantised, so it cannot re-announce per byte', () => {
  assert.equal(formatDownloadProgress(RUN_121).aria,
    'Fetching model weights, 7%, 1.95G of 26.3G');
  // Same whole percent, different byte counts -> same announcement.
  const a = formatDownloadProgress({ ...RUN_121, percent: 7.2 }).percent;
  const b = formatDownloadProgress({ ...RUN_121, percent: 7.4 }).percent;
  assert.equal(a, b);
});

test('a bar with no estimate yet drops the detail line instead of showing ?', () => {
  const f = formatDownloadProgress({
    label: 'raw.safetensors', percent: 0, done: '0.00', total: '26.3G',
    elapsed: '00:00', eta: null, speed: null,
  });
  assert.equal(f.headline, 'Fetching model weights — 0.00 of 26.3G (0%)');
  assert.equal(f.detail, '00:00 elapsed');
});

test('nothing parsable degrades to null — the card keeps its phase sentence', () => {
  assert.equal(formatDownloadProgress(null), null);
  assert.equal(formatDownloadProgress(undefined), null);
  assert.equal(formatDownloadProgress({}), null);
  assert.equal(formatDownloadProgress({ percent: 40 }), null);      // no counters
  assert.equal(formatDownloadProgress('7%'), null);
});

test('a third-party label is made readable and bounded', () => {
  assert.equal(downloadLabel('raw.safetensors'), 'Fetching model weights');
  assert.equal(downloadLabel('Downloading (incomplete total...)'),
    'Downloading (incomplete total...)');
  assert.equal(downloadLabel('shards'), 'Downloading shards');
  assert.equal(downloadLabel(''), 'Downloading');
  assert.ok(downloadLabel('x'.repeat(400)).length <= 48);
});
