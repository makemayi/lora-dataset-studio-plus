import assert from 'node:assert/strict';
import test from 'node:test';
import {
  CONFIRM_ABOVE_SECONDS, DEFAULT_SECONDS_PER_IMAGE,
  durationLabel, heavyRunConfirm, heavyRunNotice, runCost,
} from './runCost.js';

test('the pace is the measured one when the machine has a history', () => {
  const cost = runCost(10, 30);
  assert.equal(cost.secondsPerImage, 30);
  assert.equal(cost.seconds, 300);
  assert.equal(cost.measured, true);
});

test('no history means the default, and the UI is told it is a guess', () => {
  for (const missing of [null, undefined, 0, -5, 'x']) {
    const cost = runCost(10, missing);
    assert.equal(cost.secondsPerImage, DEFAULT_SECONDS_PER_IMAGE);
    assert.equal(cost.measured, false, `${missing} must not read as measured`);
  }
});

test('durations are spoken the way a human says them', () => {
  assert.equal(durationLabel(45), '45 s');
  assert.equal(durationLabel(60), '1 min');
  assert.equal(durationLabel(1500), '25 min');
  assert.equal(durationLabel(3600), '1 h');
  assert.equal(durationLabel(6000), '1 h 40');
  assert.equal(durationLabel(0), '0 s');
});

test('a long run is flagged, a short one is not — and nothing is ever refused', () => {
  // 33 prompts on one checkpoint at a measured 30 s: 16 min. This is the exact
  // run the first version REFUSED. It must sail through without even a warning.
  const thirtyThree = runCost(33, 30);
  assert.equal(thirtyThree.cells, 33);
  assert.equal(thirtyThree.heavy, false);
  assert.equal(thirtyThree.label, '17 min');

  // The same 33 prompts across 8 checkpoints IS a long run — and the old cap let
  // that one through untouched, which is what made it the wrong axis to govern.
  const wide = runCost(33 * 8, 30);
  assert.equal(wide.heavy, true);
  assert.ok(wide.seconds > CONFIRM_ABOVE_SECONDS);
});

test('a slower machine asks sooner — that is the point of measuring', () => {
  // 200 generations: fine on a fast card, an evening on a slow one.
  assert.equal(runCost(200, 10).heavy, false);
  assert.equal(runCost(200, 60).heavy, true);
});

test('the warning states the count, the duration, and the way out', () => {
  const notice = heavyRunNotice(runCost(400, 30));
  assert.match(notice, /400 generations/);
  assert.match(notice, /3 h 20/);
  assert.match(notice, /at your current pace/);
  assert.match(notice, /stop it at any time/);
  // Without a measured pace it must not claim to know the machine.
  assert.doesNotMatch(heavyRunNotice(runCost(400, null)), /at your current pace/);
});

test('the confirmation asks a question and names what it will queue', () => {
  const msg = heavyRunConfirm(runCost(400, 30));
  assert.match(msg, /will queue 400 generations/);
  assert.match(msg, /3 h 20/);
  assert.match(msg, /Start it\?/);
  // It promises the images already made are kept — a Stop must not read as a loss.
  assert.match(msg, /already generated are kept/);
});
