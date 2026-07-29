import test from 'node:test';
import assert from 'node:assert/strict';
import { dualCaptionsSupport, DUAL_CAPTION_UNSUPPORTED_FAMILIES } from './dualCaptions.js';

test('families that cache text embeddings cannot train dual captions (issue #22)', () => {
  for (const fam of ['krea', 'anima']) {
    const s = dualCaptionsSupport(fam);
    assert.equal(s.supported, false, `${fam} must be flagged`);
    assert.match(s.note, /long caption alone/);
    // The note must name the family, not just "this family".
    assert.ok(s.note.length > 40 && /^[A-Z]/.test(s.note));
  }
});

test('every other family keeps dual captions, with no scary note', () => {
  for (const fam of ['zimage', 'sdxl', 'flux', 'flux2klein']) {
    const s = dualCaptionsSupport(fam);
    assert.equal(s.supported, true, `${fam} must stay supported`);
    assert.equal(s.note, '');
  }
});

test('an unknown family is assumed to support them (no false alarm)', () => {
  assert.equal(dualCaptionsSupport('brand_new').supported, true);
  assert.equal(dualCaptionsSupport(undefined).supported, true);
});

test('the unsupported list is the one the backend refuses', () => {
  // Mirrors lora_training.DUAL_CAPTION_UNSUPPORTED_FAMILIES.
  assert.deepEqual(DUAL_CAPTION_UNSUPPORTED_FAMILIES, ['krea', 'anima']);
});
