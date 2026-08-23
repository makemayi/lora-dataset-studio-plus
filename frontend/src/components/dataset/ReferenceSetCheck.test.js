/**
 * The reference-set self-check.
 *
 * The check exists because candidate scoring cannot perform it: `sim` is the MAX
 * over the references, so a photo of the WRONG person is never outvoted — it wins
 * that max and RAISES every candidate's score. These tests pin the three answers
 * that must stay distinct ("agrees" / "may not be the same person" / "never
 * compared"), and the render rules that keep the panel alive under Chrome's
 * auto-translate.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { render } from '../../../tests/support/mountJsx.mjs';
import { verdictFor, verdictTitle, wasCompared, summarise } from './referenceSetCheck.js';

const src = readFileSync(new URL('./ReferencePanel.jsx', import.meta.url), 'utf8');

const REPORT = {
  ok: true,
  floor: 0.2,
  compared: 3,
  refs: [
    { slot: 'primary', filename: null, state: 'scorable', agreement: 0.61, flagged: false },
    { slot: 'extra', filename: 'a.webp', state: 'scorable', agreement: 0.58, flagged: false },
    { slot: 'extra', filename: 'b.webp', state: 'scorable', agreement: 0.04, flagged: true },
  ],
};

// --- reading a report ------------------------------------------------------

test('verdictFor picks the primary by slot and an extra by filename', () => {
  assert.equal(verdictFor(REPORT, 'primary').agreement, 0.61);
  assert.equal(verdictFor(REPORT, 'extra', 'b.webp').flagged, true);
  assert.equal(verdictFor(REPORT, 'extra', 'gone.webp'), null);
  assert.equal(verdictFor(null, 'primary'), null);
});

test('a reference with no agreement was never compared, and is not "clear"', () => {
  // Three ways to end up here, and none of them means the photo is fine.
  assert.equal(wasCompared({ state: 'scorable' }), false);
  assert.equal(wasCompared({ state: 'no_face' }), false);
  assert.equal(wasCompared({ state: 'scorable', agreement: 0.0 }), true);
  assert.equal(wasCompared(null), false);
});

test('an uncompared reference is never titled as if it agreed', () => {
  assert.match(verdictTitle(REPORT, { state: 'no_face' }), /Not compared/);
  assert.match(verdictTitle(REPORT, { state: 'no_face' }), /no usable face/);
  assert.match(verdictTitle(REPORT, { state: 'scorable' }), /nothing here to compare/);
});

test('a flagged title names the number AND the floor it fell under', () => {
  const title = verdictTitle(REPORT, verdictFor(REPORT, 'extra', 'b.webp'));
  assert.match(title, /0\.04/);
  assert.match(title, /0\.2/);
  assert.match(title, /may not be the same person/);
});

test('a clear title states the agreement without a verdict word', () => {
  const title = verdictTitle(REPORT, verdictFor(REPORT, 'primary'));
  assert.match(title, /Agreement 0\.61/);
  assert.ok(!/may not be/.test(title));
});

test('summarise reports the lowest agreement as null, not 0, when nothing was compared', () => {
  // 0 is a measured floor-scraping score; null is "we did not measure". Rendering
  // the first for the second is how a panel says "terrible" when it means "unknown".
  const lone = { floor: 0.2, compared: 0, refs: [{ slot: 'primary', state: 'scorable' }] };
  assert.deepEqual(summarise(lone), { compared: 0, flaggedCount: 0, lowest: null });
  assert.deepEqual(summarise(REPORT), { compared: 3, flaggedCount: 1, lowest: 0.04 });
  assert.deepEqual(summarise(null), { compared: 0, flaggedCount: 0, lowest: null });
});

// --- the panel renders -----------------------------------------------------

function baseProps(overrides = {}) {
  return {
    refFilename: 'ref.webp',
    datasetId: 1,
    onSetRef: () => {},
    onCropRef: () => {},
    busy: false,
    extraRefs: ['a.webp', 'b.webp'],
    onAddExtraRef: () => {},
    onRemoveExtraRef: () => {},
    onCropExtraRef: () => {},
    ...overrides,
  };
}

async function panel(overrides) {
  const { default: ReferencePanel } = await import('./ReferencePanel.jsx');
  return render(ReferencePanel, baseProps(overrides));
}

test('the panel renders unchanged when no check is wired', async () => {
  const html = await panel({ onCheckRefs: undefined });
  assert.ok(html.includes('Reference photo'), 'the panel itself rendered');
  assert.ok(!html.includes('Check'), 'no Check button without onCheckRefs');
});

test('wiring the check mounts BOTH button labels, not one swapped in by a ternary', async () => {
  const html = await panel({ onCheckRefs: async () => REPORT });
  assert.ok(html.includes('Check'), 'idle label present');
  assert.ok(html.includes('Checking'), 'busy label present too');
});

test('the flag badge is mounted on every tile and hidden until it fires', async () => {
  const html = await panel({ onCheckRefs: async () => REPORT });
  // Three tiles (primary + two extras), each carrying a hidden badge before any
  // check has run — so the badge appearing is an attribute flip, not a new node.
  const badges = html.match(/This reference may not be the same person/g) || [];
  assert.equal(badges.length, 3, `one badge per reference tile (found ${badges.length})`);
  assert.ok(html.includes('hidden=""'), 'badges render hidden');
});

// --- the rules that a rendered DOM cannot show -----------------------------

test('every summary variant is a hidden-flipped sibling, never a ternary swap', () => {
  // Chrome auto-translate rewrites text nodes into its own <font> wrappers; React
  // then throws NotFoundError on removeChild and the error boundary eats the
  // section. Both variants must stay mounted.
  const summary = src.slice(src.indexOf('role="status"'),
                            src.indexOf('Lowest agreement') + 400);
  for (const phrase of ['Nothing to compare', 'Every reference agrees',
                        'may not be the same person', 'Lowest agreement']) {
    assert.ok(summary.includes(phrase), `${phrase} must be in the summary`);
  }
  assert.ok((summary.match(/hidden=\{/g) || []).length >= 4, 'each variant carries hidden=');
  assert.ok(!/\?\s*'Nothing to compare'/.test(src), 'no ternary text swap');
});

test('the verdict is dropped whenever the reference set changes', () => {
  // A verdict about a set that no longer exists reads as current, which is worse
  // than no verdict. nonce covers crops, extraKey covers add/remove, refFilename
  // covers replacing the primary.
  assert.match(src, /useEffect\(\(\) => \{ setRefCheck\(null\); \}, \[extraKey, refFilename, nonce\]\)/);
});

test('the flag is a corner badge — not a border, not a ring', () => {
  // The tile edge is already spoken for (✕ top-right, ✂ bottom-left), and a
  // coloured border on a filled surface is the grammar this codebase moved away from.
  const badge = src.slice(src.indexOf('const flagBadge'), src.indexOf('const flagBadge') + 500);
  assert.ok(badge.includes('rounded-br'), 'occupies the free top-left corner');
  assert.ok(!badge.includes('border-amber'), 'no coloured border');
  assert.ok(!badge.includes('ring-'), 'no ring');
  assert.ok(badge.includes('hidden={!row?.flagged}'), 'always mounted, hidden when clear');
});

test('green is not used for the verdict — green means "kept" everywhere in this app', () => {
  const summary = src.slice(src.indexOf('role="status"'),
                            src.indexOf('Lowest agreement') + 400);
  assert.ok(!/emerald|green/.test(summary), 'no green in the verdict line');
  assert.ok(summary.includes('text-amber-600'), 'the warning is amber');
});
