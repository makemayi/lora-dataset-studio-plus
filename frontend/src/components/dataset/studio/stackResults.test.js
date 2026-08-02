import test from 'node:test';
import assert from 'node:assert/strict';
import {
  alignWeights, bestStackPayload, fmtWeight, isStackRun, stackMembers,
  variantSummary, weightVectorText, weightsIntoStackMap,
} from './stackResults.js';

const HEAD = { filename: 'z image\\a.safetensors', label: 'a', weight: 0.9, dataset_id: 4, trigger: 'aaa', head: true };
const MATE = { filename: 'z image\\b.safetensors', label: 'b', weight: 0.55, dataset_id: 7, trigger: 'bbb', head: false };

test('a run is a stack only when the backend reports several stacked LoRAs', () => {
  assert.equal(isStackRun({ stack: [HEAD, MATE] }), true);
  // A comparison run carries `stack: null`, and a one-LoRA run is not a pile.
  assert.equal(isStackRun({ stack: null }), false);
  assert.equal(isStackRun({ stack: [HEAD] }), false);
  assert.equal(isStackRun(undefined), false);
  assert.deepEqual(stackMembers({ stack: null }), []);
});

test('a variant is labelled by its weight VECTOR, missing weights stay visible', () => {
  assert.equal(weightVectorText([{ weight: 0.9 }, { weight: 0.55 }]), '0.90 / 0.55');
  assert.equal(weightVectorText([{ weight: 1 }, { weight: null }]), '1.00 / —');
  assert.equal(fmtWeight('nope'), '—');
});

test('variant weights align by FILE, not by position, and carry the delta', () => {
  // The relaunch listed the same two LoRAs in the other order: comparing row 1 with
  // row 1 would report two bogus changes.
  const variant = [{ filename: MATE.filename, weight: 1 }, { filename: HEAD.filename, weight: 0.9 }];
  const active = [{ filename: HEAD.filename, weight: 0.9 }, { filename: MATE.filename, weight: 0.55 }];
  const rows = alignWeights([HEAD, MATE], variant, active);
  assert.deepEqual(rows.map((r) => [r.label, r.weight, r.delta, r.changed]), [
    ['a', 0.9, 0, false],
    ['b', 1, 0.45, true],
  ]);
  // No reference vector (the active column itself) → no delta claimed.
  assert.equal(alignWeights([HEAD], [{ filename: HEAD.filename, weight: 0.9 }])[0].changed, false);
  // A member the variant does not price is shown empty, never as "unchanged".
  const missing = alignWeights([HEAD], [], active)[0];
  assert.equal(missing.weight, null);
  assert.equal(missing.changed, false);
});

test('variant summary falls back to counting the cells when totals are absent', () => {
  const cells = [
    { rating: 1, status: 'done', filename: 'a.png' },
    { rating: -1, status: 'done', filename: 'b.png' },
    { rating: 0, status: 'failed', filename: null },
  ];
  assert.deepEqual(variantSummary({ cells }), { likes: 1, dislikes: 1, net: 0, done: 2, total: 3 });
  assert.deepEqual(variantSummary({ cells, likes: 5, dislikes: 1, done: 6 }),
    { likes: 5, dislikes: 1, net: 4, done: 6, total: 3 });
});

test('a variant can be reloaded into the launch panel with the same keys as the sliders', () => {
  // The key shape MUST match loraStack.stackKey (`${dataset_id}:${checkpoint}`),
  // otherwise "use these weights" would silently set weights nobody reads.
  const map = weightsIntoStackMap([HEAD, MATE],
    [{ filename: HEAD.filename, weight: 1 }, { filename: MATE.filename, weight: 0.3 }]);
  assert.deepEqual(map, { '4:z image\\a.safetensors': 1, '7:z image\\b.safetensors': 0.3 });
  // Legacy runs have no dataset_id on the stacked members → that member is skipped
  // rather than written under a bogus "undefined:" key.
  assert.deepEqual(weightsIntoStackMap([{ ...MATE, dataset_id: null }],
    [{ filename: MATE.filename, weight: 0.3 }]), {});
});

test('the ★ best setting of a stack is its weights, head first', () => {
  assert.deepEqual(bestStackPayload([HEAD, MATE]), {
    dataset_id: 4,
    checkpoint: 'z image\\a.safetensors',
    strength: 0.9,
    stack: [{ dataset_id: 7, lora_filename: 'z image\\b.safetensors', weight: 0.55 }],
  });
  // Nothing to pin from an incomplete composition (an older run whose stacked members
  // never recorded their dataset) — the button hides instead of pinning half a stack.
  assert.equal(bestStackPayload([HEAD, { ...MATE, dataset_id: null }]), null);
  assert.equal(bestStackPayload([HEAD, { ...MATE, weight: null }]), null);
  assert.equal(bestStackPayload([HEAD]), null);
});
