/* A scope line must count the WORK, not the pool.
 *
 * 🎨 Classify medium on a 50 397-image bank read as "nothing happened": the
 * window offered "Classify 2 images", the run answered "0 classified, 2 skipped
 * (not scored yet)". Both figures were exact and neither explained the screen —
 * the two rows in the default scope had never been scored, and this pass reads
 * the embeddings ✨ Score caches rather than computing any of its own.
 *
 * `blocked` is that subset, measured server-side per pile from the same clause
 * the pass's own "not scored yet" comes from. These tests pin what the user
 * READS off it, including the case where it must stay invisible.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  passLaunchDisabledReason, passScopeBlocked, passScopeCount, passScopeLineLabel,
} from './bankPassScope.js';

/** The measured bank, scaled to its shape: 2 undecided rows with no score, a bin
 *  full of unclassified rows that WERE scored before being rejected. */
const PAYLOAD = {
  pass_scopes: {
    medium: {
      todo: { keep: 0, pending: 2, reject: 25464 },
      all: { keep: 24931, pending: 2, reject: 25464 },
      blocked: { keep: 0, pending: 2, reject: 0 },
      blocked_all: { keep: 0, pending: 2, reject: 0 },
    },
    // A pass with no prerequisite at all sends no `blocked` table.
    framing: { todo: { keep: 0, pending: 2, reject: 25464 }, all: {} },
  },
};

test('the default scope line says the run would classify nothing', () => {
  const label = passScopeLineLabel(PAYLOAD, 'medium', '');
  // BOTH magnitudes: what the scope holds, and what is actually runnable.
  assert.match(label, /2 images in scope, 0 ready/);
  assert.match(label, /✨ Score has not reached 2 of them/);
});

test('a scope whose work is only partly blocked quotes the fraction', () => {
  const payload = {
    pass_scopes: {
      medium: {
        todo: { keep: 10, pending: 0, reject: 0 },
        blocked: { keep: 4, pending: 0, reject: 0 },
      },
    },
  };
  const label = passScopeLineLabel(payload, 'medium', 'keep');
  assert.match(label, /10 images in scope, 6 ready/);
  assert.match(label, /✨ Score has not reached 4 of them/);
});

test('a scope with nothing blocked reads exactly as it did before', () => {
  // The bin's 25 464 rows all carry a score, so its line gains NOTHING. A note
  // that shows up on every line is a note nobody reads.
  const label = passScopeLineLabel(PAYLOAD, 'medium', 'reject');
  assert.equal(label, '✕ Unkept only (the bin) — 25464 images');
  assert.equal(passScopeBlocked(PAYLOAD, 'medium', 'reject'), 0);
});

test('a pass that sends no blocked table is untouched', () => {
  assert.equal(passScopeBlocked(PAYLOAD, 'framing', ''), null);
  assert.equal(passScopeLineLabel(PAYLOAD, 'framing', ''),
    'Kept + undecided — 2 images');
});

test('a fully blocked run is refused before the click, with the fix named', () => {
  const why = passLaunchDisabledReason({ payload: PAYLOAD, passId: 'medium', scopeId: '' });
  assert.match(why, /2 image\(s\) in scope, 0 ready/);
  assert.match(why, /Run ✨ Score first/);
  // …and the pool it refuses is a REAL one — this is not the pre-existing
  // "0 images in this scope" refusal wearing new words.
  assert.equal(passScopeCount(PAYLOAD, 'medium', ''), 2);
});

test('a runnable scope is not refused', () => {
  assert.equal(
    passLaunchDisabledReason({ payload: PAYLOAD, passId: 'medium', scopeId: 'reject' }), '');
});

test('a selection still bypasses the scope refusals', () => {
  assert.equal(passLaunchDisabledReason({
    payload: PAYLOAD, passId: 'medium', scopeId: '', selectionSize: 3,
  }), '');
});
