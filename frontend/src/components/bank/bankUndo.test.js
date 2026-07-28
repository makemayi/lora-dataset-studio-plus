import test from 'node:test';
import assert from 'node:assert/strict';
import { UNDO_HINT, undoBannerText, undoOffer, undoResultMessage } from './bankUndo.js';

test('undoOffer: an offer in the payload is what draws the bar', () => {
  const o = undoOffer({ undo: { label: 'Reject images', count: 412, at: 1 } });
  assert.equal(o.count, 412);
  assert.equal(o.label, 'Reject images');
});

test('undoOffer: no offer, or an offer that flipped nothing, draws nothing', () => {
  assert.equal(undoOffer(null), null);
  assert.equal(undoOffer({}), null);
  assert.equal(undoOffer({ undo: null }), null);
  // the actions that cannot be undone cleanly (🗑 Delete rejected, ⬆ Promote)
  // publish no offer at all — there must be no way to render one for them.
  assert.equal(undoOffer({ undo: { label: 'Delete rejected', count: 0 } }), null);
});

test('undoBannerText: says the action and the size, singular-aware', () => {
  assert.equal(undoBannerText({ label: 'Auto-reject by flag', count: 412 }),
    'Auto-reject by flag — 412 images');
  assert.equal(undoBannerText({ label: 'Keep images', count: 1 }),
    'Keep images — 1 image');
  assert.equal(undoBannerText(null), '');
});

test('the hint states the boundary of the promise up front', () => {
  assert.match(UNDO_HINT, /one step/i);
  assert.match(UNDO_HINT, /restart/i);
});

test('undoResultMessage: a full restore says so plainly', () => {
  const m = undoResultMessage({ total: 400, restored: 400, missing: 0, conflicts: 0 });
  assert.equal(m.type, 'success');
  assert.match(m.text, /Restored 400 images/);
});

test('undoResultMessage: a PARTIAL restore never over-claims — it counts and names', () => {
  const m = undoResultMessage({
    total: 400, restored: 340, missing: 48, conflicts: 12,
    conflict_names: ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'],
  });
  assert.match(m.text, /Restored 340 of 400/);
  assert.match(m.text, /48 are no longer in the bank/);
  assert.match(m.text, /12 changed since/);
  assert.match(m.text, /a\.jpg, b\.jpg, c\.jpg/);
  assert.ok(!/^↩ Restored 400/.test(m.text));
});

test('undoResultMessage: restoring NOTHING is an error, not a quiet success', () => {
  const m = undoResultMessage({ total: 400, restored: 0, missing: 400, conflicts: 0 });
  assert.equal(m.type, 'error');
  assert.match(m.text, /Restored 0 of 400/);
});

test('undoResultMessage: singular wording holds at 1', () => {
  const m = undoResultMessage({
    total: 3, restored: 1, missing: 1, conflicts: 1, conflict_names: ['x.jpg'],
  });
  assert.match(m.text, /1 is no longer in the bank/);
  assert.match(m.text, /1 changed since and was left alone \(x\.jpg\)/);
});

test('undoResultMessage: a missing reply is an error, never a fake success', () => {
  assert.equal(undoResultMessage(null).type, 'error');
});
