import test from 'node:test';
import assert from 'node:assert/strict';

import {
  REFERENCE_COMPARISON,
  describeReferenceComparison,
} from './referenceCompare.js';

const generated = { id: 7, filename: 'var_0007.png', source: 'generated' };
const imported = { id: 8, filename: 'import_0008.jpg', source: 'import' };

test('any dataset image can be held against the reference photo', () => {
  const c = describeReferenceComparison(generated, 'ref.webp');
  assert.equal(c.available, true);
  assert.equal(c.parent.filename, 'ref.webp');
  assert.equal(c.beforeLabel, 'Reference');
  assert.equal(c.afterLabel, 'This image');
  assert.equal(c.reason, '');

  // Not a privilege of generated rows: an imported photo raises the same
  // "is this the same person" question.
  assert.equal(describeReferenceComparison(imported, 'ref.webp').available, true);
});

test('a dataset with no reference photo stays silent — no button AND no note', () => {
  // The reference panel already says "Add the reference photo"; a second
  // warning here would be noise on a screen that cannot act on it.
  assert.equal(describeReferenceComparison(generated, null), null);
  assert.equal(describeReferenceComparison(generated, undefined), null);
  assert.equal(describeReferenceComparison(generated, ''), null);
  // Nothing to inspect at all.
  assert.equal(describeReferenceComparison(null, 'ref.webp'), null);
  assert.equal(describeReferenceComparison({ id: 1 }, 'ref.webp'), null);
});

test('the reference photo itself offers no comparison against itself', () => {
  assert.equal(
    describeReferenceComparison({ id: 3, filename: 'ref.webp' }, 'ref.webp'),
    null,
  );
});

test('a reference recorded without a usable file says why instead of nothing', () => {
  // Distinguished from "no reference at all" on purpose: here the dataset
  // believes it HAS one, so a missing button would read as a bug.
  const blank = describeReferenceComparison(generated, '   ');
  assert.equal(blank.available, false);
  assert.equal(blank.parent, null);
  assert.match(blank.reason, /reference photo file is missing/i);

  const bogus = describeReferenceComparison(generated, { filename: 'ref.webp' });
  assert.equal(bogus.available, false);
  assert.match(bogus.reason, /reference photo file is missing/i);
});

test('the comparison names both sides, and names them differently', () => {
  // Same shape guarantee derivedCompare.test.js applies to its own table: each
  // pane is identified by TEXT, never by position or colour alone.
  assert.ok(REFERENCE_COMPARISON.beforeLabel.trim(), 'empty beforeLabel');
  assert.ok(REFERENCE_COMPARISON.afterLabel.trim(), 'empty afterLabel');
  assert.notEqual(REFERENCE_COMPARISON.beforeLabel, REFERENCE_COMPARISON.afterLabel);
});
