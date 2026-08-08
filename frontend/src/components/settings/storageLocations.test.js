import test from 'node:test';
import assert from 'node:assert/strict';

import {
  formatSize, volumeLabel, movePercent, moveLabel, relocationChoices, locationRows,
} from './storageLocations.js';

test('an unmeasured folder never reads as empty', () => {
  assert.equal(formatSize(null), '—');
  assert.equal(formatSize(undefined), '—');
  assert.equal(formatSize(0), 'empty');
  assert.equal(formatSize(4096), '4 KB');
  assert.equal(formatSize(5e8), '500 MB');
  assert.equal(formatSize(127e9), '127.0 GB');
});

test('volumeLabel stays silent when the volume could not be read', () => {
  assert.equal(volumeLabel(null), '');
  assert.equal(volumeLabel({ free_bytes: 1e9 }), '');
  assert.equal(volumeLabel({ free_bytes: 2e9, total_bytes: 1e12 }),
    '2.0 GB free of 1000.0 GB');
});

test('move progress is null until the job knows how much it has to carry', () => {
  assert.equal(movePercent({ phase: 'scanning' }), null);
  assert.equal(movePercent({ phase: 'copying', bytes: 500, bytes_total: 1000 }), 50);
  assert.equal(movePercent({ phase: 'copying', bytes: 9e9, bytes_total: 1e9 }), 100);
  assert.match(moveLabel({ phase: 'scanning' }), /Looking at/);
  assert.equal(moveLabel({ phase: 'error', error: 'disk full' }), 'disk full');
  assert.match(moveLabel({ phase: 'copying', files: 2, files_total: 4, bytes: 1, bytes_total: 2 }),
    /2 \/ 4 files — 50%/);
});

test('a rejected target offers no choice at all', () => {
  assert.deepEqual(relocationChoices({ validation: { ok: false } }), []);
  assert.deepEqual(relocationChoices({}), []);
});

test('both choices are offered, and neither happens implicitly', () => {
  const choices = relocationChoices({
    validation: { ok: true, free_bytes: 5e11 }, currentSize: 4e9,
  });
  assert.deepEqual(choices.map((c) => c.id), ['move', 'adopt']);
  assert.match(choices[0].label, /Move the 4\.0 GB/);
  assert.equal(choices[0].disabled, false);
  // "adopt" must say out loud that the old files are neither moved nor deleted
  assert.match(choices[1].detail, /keeps its files/);
});

test('a move that would not fit is offered but disabled, with the numbers', () => {
  const choices = relocationChoices({
    validation: { ok: true, free_bytes: 1e9 }, currentSize: 100e9,
  });
  assert.equal(choices[0].disabled, true);
  assert.match(choices[0].detail, /1\.0 GB free/);
  assert.match(choices[0].detail, /100\.0 GB to move/);
  // adopting stays available: on a full disk it is often the only answer
  assert.equal(choices[1].disabled, undefined);
});

test('going back to the default is a single, explicit choice', () => {
  const choices = relocationChoices({ validation: { ok: true, default: true } });
  assert.deepEqual(choices.map((c) => c.id), ['adopt']);
  assert.match(choices[0].detail, /stay where they are/);
});

test('the map distinguishes "measured 0" from "not measured"', () => {
  const rows = locationRows(
    [{ key: 'checkpoints', label: 'Checkpoint store', volume: { free_bytes: 1e9, total_bytes: 2e9 } },
      { key: 'trash', label: 'Trash' }],
    { checkpoints: 0 },
  );
  assert.equal(rows[0].sizeLabel, 'empty');
  assert.equal(rows[0].sizeBytes, 0);
  assert.equal(rows[0].volumeLabel, '1.0 GB free of 2.0 GB');
  assert.equal(rows[1].sizeLabel, 'not measured');
  assert.equal(rows[1].sizeBytes, null);
  assert.equal(rows[1].volumeLabel, '');
});
