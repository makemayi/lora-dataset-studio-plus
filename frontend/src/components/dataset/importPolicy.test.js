import test from 'node:test';
import assert from 'node:assert/strict';
import {
  IMPORT_ENCODING_LABEL,
  IMPORT_IMAGE_ACCEPT,
  IMPORT_IMAGE_FORMATS,
  importInputLimitLine,
  importPolicyLine,
  preservesOriginalFiles,
} from './importPolicy.js';

test('the picker and policy name exactly the static formats that can be preserved', () => {
  assert.equal(IMPORT_IMAGE_ACCEPT, 'image/jpeg,image/png,image/webp,image/bmp');
  assert.equal(IMPORT_IMAGE_FORMATS, 'JPEG, PNG, WebP and BMP');
});

test('the default import policy preserves an eligible original file rather than guessing WebP', () => {
  assert.equal(preservesOriginalFiles(), true);
  assert.equal(importInputLimitLine(), '16 Mi-pixels and 8192 px per side');
  assert.equal(importPolicyLine(),
    'stored byte-for-byte in the original file and format (input limit: 16 Mi-pixels and 8192 px per side)');
  assert.equal(importPolicyLine({ encoding: 'preserve', max_side: 1024 }),
    'stored byte-for-byte in the original file and format (input limit: 16 Mi-pixels and 8192 px per side)');
});

test('input limits use the explicit capability aliases, with the old names as a rolling-update fallback', () => {
  assert.equal(importInputLimitLine({ input_max_pixels: 8 * 1024 * 1024, input_max_side: 4096 }),
    '8 Mi-pixels and 4096 px per side');
  assert.equal(importInputLimitLine({ preserve_max_pixels: 12 * 1024 * 1024, preserve_max_side: 6144 }),
    '12 Mi-pixels and 6144 px per side');
});

test('the capability flag keeps the preserve hint correct during a rolling update', () => {
  assert.equal(preservesOriginalFiles({ encoding: 'standard', preserve: true }), true);
  assert.equal(importPolicyLine({ encoding: 'standard', preserve: true }),
    'stored byte-for-byte in the original file and format (input limit: 16 Mi-pixels and 8192 px per side)');
});

test('legacy WebP policies still describe their explicit conversion behavior', () => {
  assert.equal(IMPORT_ENCODING_LABEL.standard, 'WebP q92');
  assert.equal(preservesOriginalFiles({ encoding: 'standard', max_side: 1536 }), false);
  assert.equal(importPolicyLine({ encoding: 'standard', max_side: 1536 }),
    'stored as WebP q92, resized to 1536 px on the long side, ratio kept (input limit: 16 Mi-pixels and 8192 px per side)');
  assert.equal(importPolicyLine({ encoding: 'lossless', max_side: 0, ceiling: 8192 }),
    'stored as WebP lossless at original size (input limit: 16 Mi-pixels and 8192 px per side)');
});
