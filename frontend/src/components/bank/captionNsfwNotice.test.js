import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
  CAPTION_NSFW_MEASURED_MIN, CAPTION_NSFW_SHARE_MIN, captionNsfwCounts, captionNsfwNotice,
} from './captionNsfwNotice.js';
import { passScopeOption } from './bankPassScope.js';

/** A payload with the two per-pile figures the server now sends for the caption pass. */
function payloadOf(nsfw, measured) {
  return { pass_scopes: { caption: { todo: {}, all: {}, nsfw, nsfw_measured: measured } } };
}

function notice(nsfw, measured, { scopeId = '', engineId = '' } = {}) {
  return captionNsfwNotice({
    payload: payloadOf(nsfw, measured),
    scopeId,
    piles: passScopeOption(scopeId).piles,
    engineId,
  });
}

const HOT = { keep: 300, pending: 100, reject: 0 };
const HOT_MEASURED = { keep: 400, pending: 100, reject: 0 };

// --- when it stays quiet, which is most of the time -------------------------
test('a bank that is only incidentally spicy gets no sentence at all', () => {
  // 20 of 500 measured = 4%. Well under the bar, and the notice that fires
  // everywhere is the notice nobody opens.
  assert.equal(notice({ keep: 20, pending: 0, reject: 0 }, HOT_MEASURED), null);
});

test('a share computed on a handful of images is not a share', () => {
  // 8 of 8 measured is 100% — and it is still eight images.
  assert.ok(CAPTION_NSFW_MEASURED_MIN > 8);
  assert.equal(notice({ keep: 8, pending: 0, reject: 0 },
                      { keep: 8, pending: 0, reject: 0 }), null);
});

test('a server that never sent the figures produces silence, not a zero', () => {
  assert.equal(captionNsfwCounts({ pass_scopes: { caption: { todo: {}, all: {} } } },
                                 '', ['keep', 'pending']), null);
  assert.equal(captionNsfwCounts(null, '', ['keep', 'pending']), null);
  assert.equal(captionNsfwNotice({ payload: null, piles: ['keep'] }), null);
});

/* THE DENOMINATOR. An image ✨ Score never reached is not a SFW image. Dividing by
   the pile would count "unknown" as "clean" and stay silent on exactly the bank
   that needs the sentence — so the share is over the MEASURED rows. */
test('unscored images are unknown, not clean: they are out of the denominator', () => {
  // 30 flagged out of 40 measured (75%) inside a pile of 4 000 images.
  const n = notice({ keep: 30, pending: 0, reject: 0 },
                   { keep: 40, pending: 0, reject: 0 });
  assert.ok(n, 'must fire: three quarters of everything measured is NSFW');
  assert.match(n.paragraphs[0], /75% of the 40 scored images/);
  // The alternative reading (30/4000 = 1%) would have said nothing at all.
});

// --- when it speaks ---------------------------------------------------------
test('the default chained run warns, and names the lever it is about', () => {
  const n = notice(HOT, HOT_MEASURED);
  assert.equal(n.tone, 'warn');
  assert.match(n.paragraphs[0], /80% of the 500 scored images/);
  assert.ok(n.paragraphs.some((p) => /JoyCaption first, then the Ollama/.test(p)));
  assert.ok(n.paragraphs.some((p) => /pick JoyCaption above/i.test(p)));
});

test('the scope decides which piles are weighed', () => {
  // Kept only: 300/400 = 75%. Undecided only: 100/100 = 100%. The bin: nothing
  // measured at all, so no sentence.
  assert.match(notice(HOT, HOT_MEASURED, { scopeId: 'keep' }).paragraphs[0],
               /75% of the 400 scored/);
  assert.match(notice(HOT, HOT_MEASURED, { scopeId: 'pending' }).paragraphs[0],
               /100% of the 100 scored/);
  assert.equal(notice(HOT, HOT_MEASURED, { scopeId: 'reject' }), null);
});

test('picking JoyCaption turns the warning into its own confirmation', () => {
  const n = notice(HOT, HOT_MEASURED, { engineId: 'joycaption' });
  assert.equal(n.tone, 'info');
  assert.ok(n.paragraphs.some((p) => /is not called at all/.test(p)));
  // …and it still states what was measured: a protection nobody can see is a
  // protection nobody trusts.
  assert.ok(n.paragraphs.some((p) => /qwen3-vl/.test(p)));
});

test('picking Ollama says the whole run is that engine, not half of it', () => {
  const n = notice(HOT, HOT_MEASURED, { engineId: 'ollama' });
  assert.equal(n.tone, 'warn');
  assert.ok(n.paragraphs.some((p) => /every caption in this run/i.test(p)));
  assert.ok(!n.paragraphs.some((p) => /JoyCaption first, then/.test(p)));
});

test("an explicit 'auto' reads exactly like the Settings default", () => {
  assert.deepEqual(notice(HOT, HOT_MEASURED, { engineId: 'auto' }),
                   notice(HOT, HOT_MEASURED, { engineId: '' }));
});

// --- the claim itself, which is about somebody else's model -----------------
/* THE EVIDENCE BASE IS ONE IMAGE, ONE SESSION, THREE NAMED BUILDS. Over-claiming
   about a third party's model would be the same failure of honesty this feature
   exists to correct, pointed outward — so the wording is pinned, in every branch. */
test('every branch names the three builds it tested and nothing wider', () => {
  for (const engineId of ['', 'auto', 'joycaption', 'ollama']) {
    const n = notice(HOT, HOT_MEASURED, { engineId });
    const text = n.paragraphs.join(' ');
    assert.match(text, /three abliterated qwen3-vl builds \(30b-a3b, 8b-instruct, 8b\)/,
                 engineId);
    assert.match(text, /one image/i, engineId);
    assert.match(text, /not a benchmark/i, engineId);
    assert.match(text, /nothing is claimed about other vision models/i, engineId);
    assert.match(text, /JoyCaption read the same image correctly/, engineId);
  }
});

test('the sentence never generalises to models nobody here measured', () => {
  for (const engineId of ['', 'joycaption', 'ollama']) {
    const text = notice(HOT, HOT_MEASURED, { engineId }).paragraphs.join(' ');
    for (const overclaim of [
      /vision models (?:hallucinate|are unreliable|invent)/i,
      /\bVLMs?\b/,
      /\ball\b[^.]*\bmodels\b/i,
      /Ollama is (?:unreliable|broken|wrong)/i,
      /\balways\b/i,
    ]) {
      assert.doesNotMatch(text, overclaim, `${engineId}: ${overclaim}`);
    }
  }
});

/* And it must not describe the test image. The measurement is a PROPERTY (a person
   missed, an act that was not the one shown); the picture's content is not the
   app's to repeat, and a public repo is not where it goes. */
test('the wording states the property measured, never the picture', () => {
  const src = readFileSync(new URL('./captionNsfwNotice.js', import.meta.url), 'utf8');
  assert.match(src, /missed a person who was present/);
  assert.match(src, /generic act that was not the one in the picture/);
});

// --- the threshold, stated so a change is a decision and not a drift --------
test('the two gates are a quarter of the measured rows, and twenty of them', () => {
  assert.equal(CAPTION_NSFW_SHARE_MIN, 0.25);
  assert.equal(CAPTION_NSFW_MEASURED_MIN, 20);
  // Exactly at the bar fires; a hair under does not.
  assert.ok(notice({ keep: 25, pending: 0, reject: 0 },
                   { keep: 100, pending: 0, reject: 0 }));
  assert.equal(notice({ keep: 24, pending: 0, reject: 0 },
                      { keep: 100, pending: 0, reject: 0 }), null);
  assert.ok(notice({ keep: 20, pending: 0, reject: 0 },
                   { keep: 20, pending: 0, reject: 0 }));
  assert.equal(notice({ keep: 19, pending: 0, reject: 0 },
                      { keep: 19, pending: 0, reject: 0 }), null);
});

/* Shipped-but-invisible is indistinguishable from not shipped. The window that
   opens before the run has to render this. */
test('the caption launch window really renders the notice', () => {
  const src = readFileSync(new URL('./BankWorkspace.jsx', import.meta.url), 'utf8');
  assert.match(src, /import \{ captionNsfwNotice \} from '\.\/captionNsfwNotice\.js'/);
  assert.match(src, /captionNsfwNotice\(\{/);
  // It reads THIS RUN's engine, not the stored setting: warning about a half the
  // run will not call is worse than not warning.
  assert.match(src, /engineId: captionEngine/);
  assert.match(src, /captionNsfw\.paragraphs\.map/);
});
