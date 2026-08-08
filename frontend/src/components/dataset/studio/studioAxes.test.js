import assert from 'node:assert/strict';
import test from 'node:test';
import { axisPayload, axisTotal, effectiveAxis, toggleAxisValue } from './studioAxes.js';

test('an untouched axis is the family default, alone', () => {
  assert.deepEqual(effectiveAxis(null, 8), [8]);
  assert.deepEqual(effectiveAxis([], 8), [8]);
  assert.deepEqual(effectiveAxis([16, 24], 8), [16, 24]);
});

test('an axis the family does not have stays empty', () => {
  // The SDXL second pass: no ladder, no default, nothing sent — and a missing
  // key means "keep the backend default", never "run zero steps".
  assert.deepEqual(effectiveAxis(null, null), []);
  assert.deepEqual(effectiveAxis(null, undefined), []);
});

test('an empty axis multiplies the grid by one, not by zero', () => {
  assert.equal(axisTotal({ cfgs: [1], steps: [8], steps2: [] }), 1);
  assert.equal(axisTotal({}), 1);
  assert.equal(axisTotal({ cfgs: [1, 2], steps: [8, 16, 24], steps2: [] }), 6);
});

test('only the axes that have values reach the request body', () => {
  assert.deepEqual(axisPayload({ cfgs: [1.0], steps: [24], steps2: [] }),
    { cfgs: [1.0], steps: [24] });
  assert.deepEqual(axisPayload({}), {});
  // No mutation of the component state that was handed in.
  const steps = [24];
  assert.notEqual(axisPayload({ steps }).steps, steps);
});

test('a ticked steps value really travels — the bug was an ABSENT key', () => {
  // Two LoRAs, blend or compare: the body must carry `steps`. Before this, the
  // panel had no picker and the body had no key, so every cell ran the default.
  assert.deepEqual(axisPayload({ steps: [24] }), { steps: [24] });
});

test('the toggle keeps at least one value, and stays sorted', () => {
  assert.deepEqual(toggleAxisValue([8], 24), [8, 24]);
  assert.deepEqual(toggleAxisValue([8, 24], 8), [24]);
  // Un-ticking the last one would launch an axis of nothing.
  assert.deepEqual(toggleAxisValue([24], 24), [24]);
  assert.deepEqual(toggleAxisValue(null, 16), [16]);
});
