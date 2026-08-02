import test from 'node:test';
import assert from 'node:assert/strict';
import { splitCaptionByLeakTerms } from './captionLeakHighlight.js';

test('no terms or no caption returns the caption as one clean run', () => {
  assert.deepEqual(splitCaptionByLeakTerms('a woman smiling', []), [
    { text: 'a woman smiling', leak: false },
  ]);
  assert.deepEqual(splitCaptionByLeakTerms('', ['hair']), [{ text: '', leak: false }]);
});

test('a matched term is split into its own leak run, case-insensitively', () => {
  const runs = splitCaptionByLeakTerms('a woman with long hair smiling', ['hair']);
  assert.deepEqual(runs, [
    { text: 'a woman with long ', leak: false },
    { text: 'hair', leak: true },
    { text: ' smiling', leak: false },
  ]);
});

test('matching is word-boundary safe, not a raw substring test', () => {
  // "hair" must not fire inside "hairstyle" (that's not what the detector matched)
  const runs = splitCaptionByLeakTerms('a striking hairstyle', ['hair']);
  assert.deepEqual(runs, [{ text: 'a striking hairstyle', leak: false }]);
});

test('multiple terms across the caption each get their own run', () => {
  const runs = splitCaptionByLeakTerms('blue eyes and freckles on pale skin',
    ['blue eyes', 'freckles', 'skin']);
  assert.deepEqual(runs, [
    { text: 'blue eyes', leak: true },
    { text: ' and ', leak: false },
    { text: 'freckles', leak: true },
    { text: ' on pale ', leak: false },
    { text: 'skin', leak: true },
  ]);
});

test('a longer term is not pre-empted by a shorter one it contains', () => {
  const runs = splitCaptionByLeakTerms('has freckles on her face', ['freckles', 'her face']);
  assert.deepEqual(runs, [
    { text: 'has ', leak: false },
    { text: 'freckles', leak: true },
    { text: ' on ', leak: false },
    { text: 'her face', leak: true },
  ]);
});
