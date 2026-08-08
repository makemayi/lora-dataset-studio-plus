import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
  FLAG_PREREQ, flagCandidateLabel, flagPrereq, launchRejectNote,
  pickedCandidates, unscannedNotice,
} from './autoRejectReadiness.js';

const ws = fs.readFileSync(new URL('./BankWorkspace.jsx', import.meta.url), 'utf8');
const dialog = fs.readFileSync(new URL('./LaunchAllDialog.jsx', import.meta.url), 'utf8');

/* ── The defect, in one line ──────────────────────────────────────────────────
   Both surfaces used to print `flags[f]` — "every image carrying this flag" —
   next to a button that only ever touches undecided images. Same number on a
   fresh bank, wildly different on the second pass. These tests pin the wiring
   at the source, because the number is the whole feature. */

test('the auto-reject popover counts from flags_actionable, never from flags', () => {
  // The checkbox line reads the actionable map through the shared helper...
  assert.match(ws, /\{flagCandidateLabel\(f, flagsActionable\)\}/);
  // ...and the old wording, which advertised the facet count, is gone.
  assert.doesNotMatch(ws, /\{flags\[f\] \?\? 0\} flagged/);
  // The facet chips keep the OTHER map: "show me every blurry image" rightly
  // includes the ones a previous pass rejected.
  assert.match(ws, /\{FLAG_LABEL\[f\]\} \{flags\[f\] \?\? 0\}/);
  // Two maps, two questions — the second was added, not substituted. The chip
  // map now comes from the FILTERED counters (a chip prints the size of the page
  // it opens); the auto-reject map deliberately still comes from the payload,
  // because that pass runs over the whole bank and not over the current view.
  assert.match(ws, /const flags = chipPrint\.flags/);
  assert.match(ws, /const flagsActionable = payload\?\.flags_actionable \|\| \{\}/);
  assert.doesNotMatch(ws, /flagsActionable = (chipPrint|facets)/);
});

test('the Launch all dialog shows the same honest count and is fed it', () => {
  assert.match(dialog, /flagCandidateLabel\(f\.key, flagsActionable\)/);
  assert.match(dialog, /launchRejectNote\(counts, steps\.has\('scan'\)\)/);
  // The workspace actually passes both props, or the dialog would print zeros.
  assert.match(ws, /counts=\{counts\} flagsActionable=\{flagsActionable\}/);
});

test('the panel cannot clip the numbers it exists to show, at any width', () => {
  // `absolute left-0` at every width put a 288-px panel at the toolbar's middle:
  // at 400 px it ran off the right edge and cut its own text in half. Bottom
  // sheet on a phone, anchored popover from sm: up.
  assert.match(ws, /fixed inset-x-3 bottom-3 z-50 max-h-\[70vh\] overflow-y-auto/);
  assert.match(ws, /sm:absolute sm:inset-x-auto sm:bottom-auto sm:left-0 sm:mt-1 sm:w-72/);
});

test('a zero-result auto-reject is not announced as a success', () => {
  // "0 image(s) rejected" in green is the exact shape of a broken feature.
  assert.match(ws, /if \(n === 0\) \{[\s\S]{0,200}toast\.info/);
});

test('the shown number is the pending count, formatted for humans', () => {
  assert.equal(flagCandidateLabel('blur', { blur: 704 }), '704 to reject');
  assert.equal(flagCandidateLabel('blur', { blur: 5930 }), '5,930 to reject');
  // A flag the payload does not carry reads 0 rather than "undefined".
  assert.equal(flagCandidateLabel('nsfw', {}), '0 to reject');
  assert.equal(flagCandidateLabel('nsfw', null), '0 to reject');
});

test('every offered flag declares which pass has to have run', () => {
  // The popover offers these; each must map to a prerequisite, or its 0 could
  // never be told apart from "nothing matches".
  for (const f of ['blur', 'noise', 'uniform', 'small', 'soft_detail', 'bars',
    'low_aesthetic', 'nsfw', 'watermark']) {
    assert.ok(FLAG_PREREQ[f], `${f} has no declared prerequisite`);
  }
});

test('a 0 with no pass behind it reads as a missing prerequisite, not a result', () => {
  const virgin = { scanned: 0, scored: 0, watermark_scanned: 0 };
  assert.match(flagPrereq('blur', virgin, 0), /🔎 Scan/);
  assert.match(flagPrereq('low_aesthetic', virgin, 0), /✨ Score/);
  assert.match(flagPrereq('watermark', virgin, 0), /🚩 Find watermarks/);
  // soft_detail / bars read a column the ORIGINAL quality pass never wrote: a
  // bank scanned by an older build is 'scanned' yet blind to them.
  assert.match(flagPrereq('bars', { scanned: 9000 }, 0), /🔎 Rescan/);

  // Once a pass has produced anything, a 0 really does mean "nothing matches"
  // and no prerequisite is claimed.
  const measured = { scanned: 9000, scored: 9000, watermark_scanned: 9000 };
  assert.equal(flagPrereq('blur', measured, 9000), null);
  assert.equal(flagPrereq('low_aesthetic', measured, 9000), null);
  assert.equal(flagPrereq('watermark', measured, 9000), null);
  assert.equal(flagPrereq('bars', measured, 9000), null);
});

test('the never-scanned pile is named, with the gesture that reaches it', () => {
  // Measured shape of a real bank: a big never-scanned pile, four of which are
  // already rejected and therefore out of the scan pool.
  const n = unscannedNotice({
    total: 99493, unscanned: 12367, unscanned_scannable: 12363,
  });
  assert.match(n.text, /12,367 of 99,493/);
  // The honest reading, spelled out — this is the sentence the whole item is for.
  assert.match(n.text, /nothing has been measured/);
  assert.match(n.action, /🔎 Scan picks up 12,363/);
  assert.match(n.caveat, /other 4 are already rejected/);
});

test('the caveat is silent when the two numbers agree, and the notice absent when clean', () => {
  const equal = unscannedNotice({ total: 100, unscanned: 12, unscanned_scannable: 12 });
  assert.equal(equal.caveat, null);
  assert.match(equal.action, /picks up 12 of them/);
  // A fully scanned bank gets no reassuring line at all.
  assert.equal(unscannedNotice({ total: 100, unscanned: 0, unscanned_scannable: 0 }), null);
  assert.equal(unscannedNotice(null), null);
  // Everything unscanned is also rejected: 🔎 Scan would do nothing, and saying
  // "picks up 0 of them" would be an offer that goes nowhere.
  const stuck = unscannedNotice({ total: 100, unscanned: 5, unscanned_scannable: 0 });
  assert.match(stuck.action, /un-reject/);
});

test('the ticked total is exact for one flag and a labelled ceiling for several', () => {
  const actionable = { blur: 704, uniform: 12 };
  assert.equal(pickedCandidates(new Set(), actionable), null);

  const one = pickedCandidates(new Set(['blur']), actionable);
  assert.equal(one.sum, 704);
  assert.equal(one.exact, true);
  assert.match(one.text, /^704 undecided image\(s\) will be rejected\./);

  // An image can carry both flags, so the sum is a ceiling and says so — the
  // same class of over-claim the whole fix is about.
  const two = pickedCandidates(new Set(['blur', 'uniform']), actionable);
  assert.equal(two.sum, 716);
  assert.equal(two.exact, false);
  assert.match(two.text, /Up to 716/);
  assert.match(two.text, /counted twice/);

  // Nothing to do says so, instead of letting the click be the discovery.
  const none = pickedCandidates(new Set(['blur']), { blur: 0 });
  assert.equal(none.sum, 0);
  assert.match(none.text, /Nothing to reject/);
});

test('Launch all admits its counts move, because the scan runs first', () => {
  const counts = { total: 99493, unscanned: 12367, unscanned_scannable: 12363 };
  const withScan = launchRejectNote(counts, true);
  assert.match(withScan, /🔎 Scan runs first/);
  assert.match(withScan, /12,363/);
  assert.match(withScan, /they will grow/);

  // Scan unticked: the pile stays invisible, and the fix is one checkbox away.
  const noScan = launchRejectNote(counts, false);
  assert.match(noScan, /no flag can reach them/);
  assert.match(noScan, /Tick 🔎 Scan quality/);

  // Nothing to warn about on a fully scanned bank.
  assert.equal(launchRejectNote({ total: 10, unscanned: 0 }, true), null);
});
