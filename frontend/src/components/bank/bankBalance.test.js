import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
  BALANCE_AXES, BALANCE_DEFAULT_AXIS, balanceNotes, balanceReadiness,
  balanceRows, bucketLabel, summarizeBalance,
} from './bankBalance.js';

const ws = fs.readFileSync(new URL('./BankWorkspace.jsx', import.meta.url), 'utf8');
const servicePy = fs.readFileSync(
  new URL('../../../../backend/app/services/image_bank_service.py', import.meta.url), 'utf8');

const bucket = (framing, selected, available, fairShare, short = false) => ({
  key: framing, framing, cluster: null, selected, available,
  fair_share: fairShare, short,
});

/** The shape the backend returns for a clean 20-image balanced pick. */
const EVEN = {
  ok: true, axis: 'framing', requested: 20, selected: 20, shortfall: 0,
  unlabelled: 0, unknown: 0, pool: 85,
  buckets: [bucket('face', 5, 10, 5), bucket('bust', 5, 30, 5),
    bucket('body', 5, 40, 5), bucket('back', 5, 5, 5)],
};

// --- the distribution is SAID, in numbers ------------------------------------
test('the obtained distribution is announced as a sentence of counts', () => {
  const line = summarizeBalance(EVEN);
  assert.match(line, /Selected 20 of 20 requested/);
  assert.match(line, /5 face, 5 bust, 5 body, 5 back/);
  // No chart needed to know what you got — this is the accessible carrier.
  assert.ok(!line.includes('%'));
});

test('rows carry selected / available / fair share per bucket', () => {
  const rows = balanceRows(EVEN);
  assert.equal(rows.length, 4);
  assert.deepEqual(rows.map((r) => r.label), ['face', 'bust', 'body', 'back']);
  assert.deepEqual(rows[0], {
    key: 'face', label: 'face', selected: 5, available: 10, fairShare: 5,
    short: false, share: 0.25,
  });
});

test('an even split reports nothing alarming', () => {
  assert.deepEqual(balanceNotes(EVEN).map((n) => n.tone), ['info']);
  assert.match(balanceNotes(EVEN)[0].text, /even share/);
});

// --- the impossible axis is NAMED, never padded in silence -------------------
test('a bucket that cannot be filled is reported with both numbers', () => {
  const res = {
    ...EVEN, selected: 20, shortfall: 0,
    buckets: [bucket('face', 6, 10, 5), bucket('bust', 6, 30, 5),
      bucket('body', 6, 43, 5), bucket('back', 2, 2, 5, true)],
  };
  const notes = balanceNotes(res);
  assert.equal(notes[0].tone, 'warn');
  assert.match(notes[0].text, /Only 2 back images exist/);
  assert.match(notes[0].text, /even split wanted 5/);
  // The top-up is stated, not hidden: the summary still totals 20.
  assert.match(summarizeBalance(res), /6 face, 6 bust, 6 body, 2 back/);
});

test('a pool that cannot reach the requested count says so', () => {
  const res = {
    ...EVEN, requested: 60, selected: 42, shortfall: 18,
    buckets: [bucket('face', 10, 10, 15, true), bucket('bust', 15, 30, 15),
      bucket('body', 15, 40, 15), bucket('back', 2, 2, 15, true)],
  };
  const texts = balanceNotes(res).map((n) => n.text).join(' | ');
  assert.match(texts, /42 images selected instead of the 60/);
});

// --- unlabelled images are the DEFAULT state, and are stated -----------------
test('unlabelled images are counted out loud with the pass that fixes it', () => {
  const notes = balanceNotes({ ...EVEN, unlabelled: 30398 });
  const text = notes.map((n) => n.text).join(' | ');
  assert.match(text, /30398 images in this filter have no label yet/);
  assert.match(text, /📐 Framing pass/);
});

test('“unknown” framing is distinguished from “not classified”', () => {
  const text = balanceNotes({ ...EVEN, unknown: 7 }).map((n) => n.text).join(' | ');
  assert.match(text, /“unknown” framing/);
  assert.ok(!/no label yet/.test(text));
});

test('readiness refuses before Score and before Framing, with the reason', () => {
  assert.deepEqual(balanceReadiness({ scored: 0 }).ready, false);
  assert.match(balanceReadiness({ scored: 0 }).reason, /✨ Score/);
  const noFraming = balanceReadiness({ scored: 10, coverage: { framing_available: false } });
  assert.equal(noFraming.ready, false);
  assert.match(noFraming.reason, /📐 Framing/);
  assert.equal(balanceReadiness({ scored: 10, coverage: { framing_available: true } }).ready, true);
});

test('the person axis labels its bucket with the cluster id', () => {
  assert.equal(bucketLabel({ framing: 'face', cluster: 2 }), 'face · person #2');
  assert.equal(bucketLabel({ framing: 'face', cluster: null }), 'face');
});

// --- contracts with the rest of the app --------------------------------------
test('axis ids match the backend axes (stored keys, never renamed)', () => {
  const ids = BALANCE_AXES.map((a) => a.id);
  assert.deepEqual(ids, ['framing', 'framing+person']);
  assert.equal(BALANCE_DEFAULT_AXIS, 'framing');
  assert.match(servicePy, /_BALANCE_AXES = \('framing', 'framing\+person'\)/);
  assert.match(servicePy, /_BALANCE_DEFAULT_AXIS = 'framing'/);
});

test('the workspace wires the balanced selector and reads it out', () => {
  assert.ok(ws.includes('/select-balanced'), 'the route is called');
  assert.ok(ws.includes('summarizeBalance'), 'the distribution is displayed');
  assert.ok(ws.includes('balanceNotes'), 'shortfalls are displayed');
});

test('empty / missing results never claim a balance', () => {
  assert.equal(summarizeBalance(null), '');
  assert.equal(summarizeBalance({ buckets: [] }), 'Nothing selected.');
  assert.deepEqual(balanceNotes(null), []);
});
