import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SAMPLE_SIZE, assertionFor, assertionSummary, checkCostNote, folderLabel,
  folderMarker, revokeNote, scanOffer, suggestedFolders, suggestionFor,
  suggestionLine, suggestionTone, toCheckNote, verdictLine, verdictTone,
} from './folderPerson.js';

const ENTRY = { subfolder: 'anna', cluster_id: 3, images: 412, sample: null, to_check: [] };

test('the bank root is a subfolder like any other, and "" is not "nothing"', () => {
  const list = [{ ...ENTRY, subfolder: '' }];
  assert.ok(assertionFor(list, ''), 'the root assertion must be findable');
  assert.equal(assertionFor(list, null), null);       // no folder scoped = no panel
  assert.equal(assertionFor(list, 'anna'), null);
  assert.equal(folderLabel(''), 'the bank root');
  assert.equal(folderLabel('anna'), 'anna');
});

test('a missing or malformed list never throws — it just has no assertion', () => {
  assert.equal(assertionFor(undefined, 'anna'), null);
  assert.equal(assertionFor(null, 'anna'), null);
});

test('the summary attributes the grouping to the USER, not to a pass', () => {
  const s = assertionSummary(ENTRY);
  assert.match(s, /Person #3/);
  assert.match(s, /412 images/);
  assert.match(s, /by you/);
  assert.match(s, /no face pass/);
});

test('one image is not "1 images"', () => {
  assert.match(assertionSummary({ ...ENTRY, images: 1 }), /1 image grouped/);
});

test('a verdict only ever speaks about the sample', () => {
  const ok = { verdict: 'consistent', note: 'sample consistent (14/15 same person)' };
  assert.equal(verdictTone(ok), 'ok');
  assert.match(verdictLine(ok), /sample consistent \(14\/15 same person\)/);
  // The reassuring line must not claim anything about the FOLDER.
  assert.ok(!/folder is/i.test(verdictLine(ok)));
});

test('two faces warn, and the warning names the action, not a failure', () => {
  const mixed = { verdict: 'mixed', note: '2 different faces in the sample — check this folder' };
  assert.equal(verdictTone(mixed), 'warn');
  assert.match(verdictLine(mixed), /check this folder/);
});

test('an unknown or inconclusive verdict is muted, never read as good news', () => {
  assert.equal(verdictTone({ verdict: 'inconclusive', note: 'x' }), 'muted');
  assert.equal(verdictTone({ verdict: 'something-new' }), 'muted');
  assert.equal(verdictTone(null), 'muted');
  assert.equal(verdictLine(null), null);
});

test('the cost of a check is stated against the cost it replaces', () => {
  const note = checkCostNote(ENTRY);
  assert.match(note, new RegExp(`${SAMPLE_SIZE} images`));
  assert.match(note, /412/);
  // A small folder is not sold a saving it would not get.
  assert.match(checkCostNote({ images: 4 }), /^Embeds the 4 images of this folder\.$/);
  assert.match(checkCostNote({ images: 1 }), /the 1 image of this folder/);
});

test('unreadable images are flagged as a heads-up, never as an exclusion', () => {
  assert.equal(toCheckNote(ENTRY), null);
  assert.equal(toCheckNote(null), null);
  const n = toCheckNote({ ...ENTRY, to_check: [{ id: 1 }, { id: 2 }] });
  assert.match(n, /2 images/);
  assert.match(n, /Still in the group/);
});

test('revoking says what it does AND what it does not do', () => {
  const n = revokeNote('anna');
  assert.match(n, /normal clustering/);
  assert.match(n, /Nothing is deleted/);
  assert.match(revokeNote(''), /the bank root/);
});

// ── automatic suggestions ───────────────────────────────────────────────────
const LIKELY = {
  subfolder: 'anna', verdict: 'consistent', sample: 15, scorable: 15,
  largest: 15, faces: 1, stale: false,
};

test('a suggestion is phrased as a QUESTION, never as something already done', () => {
  const line = suggestionLine(LIKELY);
  assert.match(line, /Looks like one person/);
  assert.match(line, /assert\?$/);          // it asks
  assert.equal(suggestionTone(LIKELY), 'ok');
  // It never claims the folder IS one person, nor that anything was grouped.
  assert.ok(!/\bis one person\b/.test(line));
  assert.ok(!/grouped|asserted/i.test(line));
});

test('the offer says how much of the folder it actually looked at', () => {
  assert.match(suggestionLine(LIKELY), /15\/15 of the 15 sampled/);
});

test('a mixed or thin sample is never dressed up as an offer', () => {
  const mixed = { ...LIKELY, verdict: 'mixed', faces: 2 };
  assert.match(suggestionLine(mixed), /probably not one person/);
  assert.equal(suggestionTone(mixed), 'muted');
  const thin = { ...LIKELY, verdict: 'inconclusive', scorable: 1, sample: 60 };
  assert.match(suggestionLine(thin), /Almost no readable face here — 1 in 60 images tried/);
  assert.match(suggestionLine(thin), /little to work with in this folder/);
  assert.equal(suggestionTone(thin), 'muted');
});

test('a partial verdict is offered with its thinness in front of it', () => {
  const partial = { ...LIKELY, verdict: 'partial', sample: 40, scorable: 6, largest: 6 };
  const line = suggestionLine(partial);
  assert.match(line, /Looks like one person, on thin evidence/);
  assert.match(line, /only 6 usable faces in 40 images tried/);
  assert.match(line, /assert\?$/);          // still a question, never a claim
  assert.equal(suggestionTone(partial), 'ok');
  assert.deepEqual(suggestedFolders([partial]), ['anna']);
  assert.equal(folderMarker([partial], 'anna'), ' · 👤?');
});

test('a STALE probe never surfaces — the folder it describes is gone', () => {
  const stale = [{ ...LIKELY, stale: true }];
  assert.equal(suggestionFor(stale, 'anna'), null);
  assert.deepEqual(suggestedFolders(stale), []);
  assert.equal(folderMarker(stale, 'anna'), '');
});

test('only a "looks like one person" folder gets a marker in the picker', () => {
  const list = [LIKELY, { ...LIKELY, subfolder: 'bob', verdict: 'mixed' }];
  assert.equal(folderMarker(list, 'anna'), ' · 👤?');
  assert.equal(folderMarker(list, 'bob'), '');
  assert.equal(folderMarker(list, 'nobody'), '');
  assert.deepEqual(suggestedFolders(list), ['anna']);
  assert.deepEqual(suggestedFolders(undefined), []);
});

test('the scan states its cost BEFORE it is paid, and who decides', () => {
  const o = scanOffer({ scannable: 6, scan_limit: 20, sample_size: 15 });
  assert.match(o.label, /Scan 6 folders/);
  assert.match(o.note, /~15 images each/);
  assert.match(o.note, /only suggests; you confirm/);
  assert.ok(!/waiting/.test(o.note));      // nothing left over, so nothing claimed
});

test('the scan quotes the re-draw ceiling, not just the lucky case', () => {
  const o = scanOffer({ scannable: 6, scan_limit: 20, sample_size: 15, sample_max: 60 });
  assert.match(o.note, /~15 images each, up to 60 in the hard ones/);
});

test('a ceiling is announced rather than silently applied', () => {
  const o = scanOffer({ scannable: 200, scan_limit: 20, sample_size: 15 });
  assert.match(o.label, /Scan 20 folders/);
  assert.match(o.note, /200 are waiting/);
  assert.match(o.note, /biggest go first/);
});

test('nothing left to scan means no button at all, not a dead one', () => {
  assert.equal(scanOffer({ scannable: 0, scan_limit: 20 }), null);
  assert.equal(scanOffer(null), null);
  assert.match(scanOffer({ scannable: 1, scan_limit: 20 }).label, /Scan 1 folder$/);
});
