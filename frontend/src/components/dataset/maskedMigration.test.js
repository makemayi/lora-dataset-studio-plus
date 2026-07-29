import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  LEGACY_MASKED_KEY, readLegacyMasked, clearLegacyMasked, maskedCarryOverAction,
} from './maskedMigration.js';

/** Minimal localStorage stand-in; `boom` makes every access throw (private mode). */
function store(initial = {}, boom = false) {
  const map = { ...initial };
  return {
    getItem: (k) => { if (boom) throw new Error('denied'); return k in map ? map[k] : null; },
    setItem: (k, v) => { if (boom) throw new Error('denied'); map[k] = String(v); },
    removeItem: (k) => { if (boom) throw new Error('denied'); delete map[k]; },
    _map: map,
  };
}

const ADV_UNANSWERED = { masked: true, masked_supported: true, masked_stored: null };

test('the legacy key keeps its name — it is already in real localStorage', () => {
  assert.equal(LEGACY_MASKED_KEY, 'trainMasked_v1');
});

test('reads the three states, and survives an unreadable storage', () => {
  assert.equal(readLegacyMasked(store({ trainMasked_v1: '0' })), false);
  assert.equal(readLegacyMasked(store({ trainMasked_v1: '1' })), true);
  assert.equal(readLegacyMasked(store()), null);
  assert.equal(readLegacyMasked(store({ trainMasked_v1: '0' }, true)), null);
  assert.equal(readLegacyMasked(null), null);
});

// --- the arbitration, direction A: never silently DISABLE masking -------------
test('a browser that had masking OFF is asked, never adopted silently', () => {
  const s = store({ trainMasked_v1: '0' });
  assert.equal(maskedCarryOverAction(s, ADV_UNANSWERED), 'prompt');
  // and nothing was written: the decision is the user's, the panel only asks.
  assert.deepEqual(s._map, { trainMasked_v1: '0' });
});

// --- the arbitration, direction B: never silently ENABLE it either ------------
test('a browser whose value AGREES with the default is not nagged', () => {
  // ON is the default, so there is nothing to disclose — the notice would be noise.
  assert.equal(maskedCarryOverAction(store({ trainMasked_v1: '1' }), ADV_UNANSWERED), 'clear');
});

test('no legacy key at all = nothing happened, nothing is said', () => {
  assert.equal(maskedCarryOverAction(store(), ADV_UNANSWERED), 'none');
});

test('a dataset that already answered wins over the browser', () => {
  // The server value is the truth now; the browser has no decision left to offer.
  for (const stored of [true, false]) {
    assert.equal(
      maskedCarryOverAction(store({ trainMasked_v1: '0' }),
        { ...ADV_UNANSWERED, masked_stored: stored }),
      'clear');
  }
});

test('a concept/style/slider dataset keeps the key for a dataset where it applies', () => {
  assert.equal(
    maskedCarryOverAction(store({ trainMasked_v1: '0' }),
      { masked: false, masked_supported: false, masked_stored: null }),
    'none');
});

test('nothing is decided before the server settings have loaded', () => {
  assert.equal(maskedCarryOverAction(store({ trainMasked_v1: '0' }), null), 'none');
});

test('clearing removes exactly the legacy key and tolerates private mode', () => {
  const s = store({ trainMasked_v1: '0', other: 'x' });
  clearLegacyMasked(s);
  assert.deepEqual(s._map, { other: 'x' });
  assert.doesNotThrow(() => clearLegacyMasked(store({}, true)));
});

// --- the panel actually wires it ---------------------------------------------
const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');

test('the panel reads masked from the SERVER settings, not from localStorage', () => {
  assert.doesNotMatch(panel, /localStorage[^\n]*trainMasked_v1/,
    'the browser preference must no longer be the source of truth');
  assert.match(panel, /adv\?\.masked !== false/);
  assert.match(panel, /saveAdv\(\{ masked:/);
});

test('the panel offers the carry-over notice with BOTH explicit answers', () => {
  assert.match(panel, /maskedCarryOverAction\(/);
  assert.match(panel, /Keep masking on/);
  assert.match(panel, /Turn it off for this dataset/);
});
