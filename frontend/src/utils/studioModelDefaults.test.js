import test from 'node:test';
import assert from 'node:assert/strict';
import {
  modelDefaultsFor, defaultCfgFor, defaultStepsFor, mixedModelDefaults,
} from './studioModelDefaults.js';

const TURBO = 'z image\\z_image_turbo_bf16.safetensors';
const BASE = 'z image\\z_image_base.safetensors';

const payload = {
  default_cfg: 1.0,
  default_steps: 8,
  model_defaults: {
    [TURBO]: { cfg: 1.0, steps: 8 },
    [BASE]: { cfg: 4.0, steps: 30 },
  },
};

test('Z-Image Base does NOT inherit the Turbo defaults (GitHub #18)', () => {
  assert.notDeepEqual(modelDefaultsFor(payload, BASE), modelDefaultsFor(payload, TURBO));
  assert.equal(defaultCfgFor(payload, [BASE]), 4.0);
  assert.equal(defaultStepsFor(payload, [BASE]), 30);
  assert.equal(defaultCfgFor(payload, [TURBO]), 1.0);
  assert.equal(defaultStepsFor(payload, [TURBO]), 8);
});

test('a base the backend does not differentiate falls back to the family default', () => {
  assert.equal(defaultCfgFor(payload, ['z image\\mystery.safetensors']), 1.0);
  assert.equal(defaultStepsFor(payload, ['z image\\mystery.safetensors']), 8);
});

test('an older backend payload (no model_defaults) keeps the historical values', () => {
  const old = { default_cfg: 1.0, default_steps: 8 };
  assert.equal(modelDefaultsFor(old, BASE), null);
  assert.equal(defaultCfgFor(old, [BASE]), 1.0);
  assert.equal(defaultStepsFor(old, [BASE]), 8);
});

test('no payload at all degrades to the shipped constants, never NaN/undefined', () => {
  assert.equal(defaultCfgFor(null, [BASE]), 1.0);
  assert.equal(defaultStepsFor(undefined, []), 8);
});

test('a multi-base sweep takes the FIRST selected base, and says it is mixed', () => {
  assert.equal(defaultCfgFor(payload, [BASE, TURBO]), 4.0);
  assert.equal(defaultCfgFor(payload, [TURBO, BASE]), 1.0);
  assert.equal(mixedModelDefaults(payload, [BASE, TURBO]), true);
  assert.equal(mixedModelDefaults(payload, [TURBO]), false);
  assert.equal(mixedModelDefaults(payload, [BASE, BASE]), false);
});

/* The persistence contract these helpers live under (useStudioForm):
   `effectiveCfgs = selCfgs ?? [modelDefaultCfg]`. The helpers are only ever
   consulted when the axis has NEVER been touched, so a value the user saved
   cannot be rewritten by a model-default change. Asserted here on the same
   expression the hook uses, because that `??` is the entire guarantee. */
test('a persisted user selection is never replaced by a model default', () => {
  const persistedCfgs = [1.5];
  const persistedSteps = [12];
  assert.deepEqual(persistedCfgs ?? [defaultCfgFor(payload, [BASE])], [1.5]);
  assert.deepEqual(persistedSteps ?? [defaultStepsFor(payload, [BASE])], [12]);
  // …and an untouched axis (null) does take the model default.
  const untouched = null;
  assert.deepEqual(untouched ?? [defaultCfgFor(payload, [BASE])], [4.0]);
});

test('a malformed model_defaults entry degrades instead of throwing', () => {
  const junk = { default_cfg: 1.0, default_steps: 8, model_defaults: { [BASE]: 'nope' } };
  assert.equal(modelDefaultsFor(junk, BASE), null);
  assert.equal(defaultCfgFor(junk, [BASE]), 1.0);
  assert.equal(defaultCfgFor({ model_defaults: [] }, [BASE]), 1.0);
});
