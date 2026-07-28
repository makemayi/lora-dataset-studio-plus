import test from 'node:test';
import assert from 'node:assert/strict';

import {
  KREA_ASSET_LABELS, kreaMissingLabels, kreaUnavailableReason, groundingDescription,
  refOrientation, kreaFramingAdvisory, KREA_SUGGESTED_ASPECT, KREA_TIGHT_FRAMINGS,
} from './kreaEngine.js';
import {
  ENGINES, LOCAL_ENGINES, API_ENGINES, ENGINE_LABELS, ENGINE_RATES, ENGINE_ACCENTS,
  canonicalEngines, engineBatches, localOnly, localQueuesBehindApi, estimateCost,
  billingEngines, totalImages, readEngines, writeEngines,
} from '../components/dataset/engineSelection.js';

// ── The engine is a first-class member of the selection model ────────────────

test('krea is a real engine everywhere the selection model looks', () => {
  assert.ok(ENGINES.includes('krea'));
  assert.ok(LOCAL_ENGINES.includes('krea'));
  assert.ok(!API_ENGINES.includes('krea'), 'krea is local, never an API engine');
  assert.equal(ENGINE_LABELS.krea, 'Krea 2 Edit');
  assert.equal(ENGINE_RATES.krea, 0, 'local GPU time is free');
  assert.ok(ENGINE_ACCENTS.krea, 'a card with no accent renders unstyled');
  assert.deepEqual(canonicalEngines(['krea', 'nope', 'klein']), ['klein', 'krea']);
});

test('a Krea run is free and bills nobody', () => {
  assert.equal(estimateCost(30, ['krea'], 'split', { multiplier: 2 }), 0);
  assert.deepEqual(billingEngines(['krea', 'klein']), []);
  // Mixing with an API engine bills only the API share (split = round-robin).
  assert.deepEqual(billingEngines(['krea', 'nanobanana']), ['nanobanana']);
  assert.equal(totalImages(10, ['krea', 'klein'], 'all', 1), 20);
});

// ── Dispatch order: the GPU-bound engines go last, BOTH of them ──────────────

test('both local engines are dispatched after the API ones', () => {
  const shots = [1, 2, 3, 4, 5, 6];
  const order = engineBatches(shots, ['klein', 'krea', 'nanobanana'], 'split')
    .map((b) => b.generator);
  assert.deepEqual(order, ['nanobanana', 'klein', 'krea'],
    'API batches start returning immediately; the GPU ones queue behind');
});

test('localQueuesBehindApi is true for a Krea+API mix and false for local-only', () => {
  assert.equal(localQueuesBehindApi(['krea', 'chatgpt']), true);
  assert.equal(localQueuesBehindApi(['krea', 'klein']), false);
  assert.equal(localQueuesBehindApi(['chatgpt']), false);
});

// ── 🔞 gating widened from "Klein alone" to "every engine is local" ──────────

test('localOnly unlocks the uncensored catalog for any all-local run', () => {
  assert.equal(localOnly(['krea']), true);
  assert.equal(localOnly(['klein', 'krea']), true, 'two local engines together are fine');
  assert.equal(localOnly(['krea', 'openrouter']), false, 'an API engine locks it again');
  assert.equal(localOnly([]), false, 'nothing selected renders nothing');
});

// ── Storage compatibility: the legacy single-string key still rules ──────────

test('a profile that only ever knew the legacy key can still name krea', () => {
  const map = new Map([['datasetGenerator', 'krea']]);
  const storage = {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { map.set(k, String(v)); },
  };
  assert.deepEqual(readEngines(storage), ['krea']);
  writeEngines(storage, ['krea', 'chatgpt']);
  assert.equal(map.get('datasetGenerator'), 'krea', 'legacy mirror follows the primary');
});

// ── The "why can't I pick it?" sentence — one branch per real failure ────────

test('every missing asset key has a word, in a stable order', () => {
  assert.deepEqual(
    kreaMissingLabels(['krea_vae', 'krea_identity_lora', 'krea_model']),
    ['base model', 'identity edit LoRA', 'VAE']);
  assert.deepEqual(kreaMissingLabels([]), []);
  assert.deepEqual(kreaMissingLabels(undefined), []);
  assert.deepEqual(kreaMissingLabels(['who_knows']), [], 'unknown keys are dropped');
  // Every key the backend can send must have a word, or the sentence loses it.
  for (const key of ['krea_model', 'krea_identity_lora', 'krea_text_encoder', 'krea_vae']) {
    assert.ok(KREA_ASSET_LABELS[key], `no label for ${key}`);
  }
});

test('the unavailable reason names the FIRST thing to fix, never just "off"', () => {
  assert.equal(kreaUnavailableReason({}), null, 'nothing missing = pickable');

  assert.match(kreaUnavailableReason({ enabledInSettings: false }), /disabled in Settings/);

  assert.match(
    kreaUnavailableReason({ comfyuiReachable: false, missingAssets: ['krea_model'] }),
    /Configure ComfyUI/,
    'an unreachable ComfyUI makes the asset list meaningless — say that instead');

  assert.match(
    kreaUnavailableReason({
      missingNodes: ['Krea2EditModelPatch'], missingAssets: ['krea_model'],
    }),
    /comfyui-krea2edit node pack/,
    'the node pack comes first: without it nothing runs even with every file present');

  const assetsOnly = kreaUnavailableReason({
    missingAssets: ['krea_identity_lora', 'krea_vae'], missingNodes: [],
  });
  assert.match(assetsOnly, /identity edit LoRA \+ VAE/);
  assert.doesNotMatch(assetsOnly, /node pack/);
});

test('a missing asset points somewhere that actually mentions Krea', () => {
  // The rule this pins is NOT a specific word. This one line has been wrong
  // twice: it said "see Setup" when Setup covered Klein only and never
  // mentioned Krea, then it named the Guide -- right while reading was all a
  // user could do. Setup installs these files now, so pointing at prose would
  // send someone to read about a job the app can do for them.
  // The invariant: name a destination that EXISTS and covers Krea, and prefer
  // the one that acts over the one that explains.
  const msg = kreaUnavailableReason({ missingAssets: ['krea_model', 'krea_vae'] });
  assert.match(msg, /Setup/, 'Setup now installs these - send people to the button');
  assert.match(msg, /missing/i, 'still say what is wrong, not only where to go');
  assert.match(msg, /download|install/i, 'say that it can be done FOR them');
});

test('a file that is PRESENT but is not weights gets named, not left to ComfyUI', () => {
  // An interrupted, proxied or error-page download saves HTML (or half a file)
  // as .safetensors. It exists, so "missing" says nothing — and the only
  // symptom used to be ComfyUI's raw "Expecting value: line 1 column 1".
  const gate = kreaUnavailableReason({
    invalidAssets: [{ asset: 'krea_model', filename: 'krea2_turbo_fp8.safetensors',
      verdict: 'html_or_text', blocking: true }],
  });
  assert.match(gate, /base model/);
  assert.match(gate, /krea2_turbo_fp8\.safetensors/);
  assert.match(gate, /web page, not weights/);
  assert.match(gate, /Delete it and download it again/);

  assert.match(
    kreaUnavailableReason({
      invalidAssets: [{ asset: 'krea_identity_lora', filename: 'id.safetensors',
        verdict: 'truncated_or_garbage', blocking: true }],
    }),
    // Wording now comes from the SHARED helper (utils/modelIntegrityWords.js), so
    // Klein and Krea describe one corrupted file the same way. It also says what
    // "truncated" actually means, because the person reading it is looking at a
    // file of plausible size and needs to know why it counts as broken.
    /cut short or corrupted/);

  // An ADVISORY (too_small) is not a reason to refuse the engine.
  assert.equal(kreaUnavailableReason({
    invalidAssets: [{ asset: 'krea_vae', filename: 'v.safetensors',
      verdict: 'too_small', blocking: false }],
  }), null);

  // A genuinely missing file still comes first: "place it here" beats
  // "re-download that one".
  assert.match(kreaUnavailableReason({
    missingAssets: ['krea_vae'],
    invalidAssets: [{ asset: 'krea_model', filename: 'k.safetensors',
      verdict: 'html_or_text', blocking: true }],
  }), /VAE missing/);
});

// ── The dial has to MEAN something, not just show a number ───────────────────

test('grounding is described in words at every end of the range', () => {
  assert.match(groundingDescription(512), /follows the prompt/);
  assert.match(groundingDescription(768), /leans towards the prompt/);
  assert.match(groundingDescription(1024), /balanced/);
  assert.match(groundingDescription(1536), /sticks to the reference/);
  assert.match(groundingDescription(undefined), /default/);
  assert.match(groundingDescription('nonsense'), /default/);
});

// ── Reference shape vs body/back framing (MEASURED 2026-07-25) ──────────────
// Krea reproduces the reference's aspect ratio, so a square reference crops the
// wide shots. The two measured cases below are the anchors of this whole block:
// 1024x1024 came back a bust, 835x1024 came back a full figure.

const bodyShots = (n) => Array.from({ length: n }, () => 'body');

test('orientation buckets the two MEASURED references on the right side', () => {
  assert.equal(refOrientation(835, 1024), 'portrait', 'the one that produced a full figure');
  assert.equal(refOrientation(1024, 1024), 'square', 'the one that produced a bust');
  assert.equal(refOrientation(1920, 1080), 'landscape');
  // "Nearly square" behaves like square, so it is bucketed like square.
  assert.equal(refOrientation(1000, 1024), 'square');
  // Unmeasurable stays unmeasurable — never guessed into a bucket.
  for (const bad of [[0, 100], [100, 0], [null, 100], ['x', 'y'], [undefined, undefined]]) {
    assert.equal(refOrientation(bad[0], bad[1]), null, `bad size ${JSON.stringify(bad)}`);
  }
});

test('a square reference with body shots selected warns, and counts them', () => {
  const a = kreaFramingAdvisory({
    width: 1024, height: 1024,
    framings: [...bodyShots(17), 'back', 'face', 'bust'],
  });
  assert.ok(a, 'a square reference + wide shots is exactly the case to warn about');
  assert.equal(a.tight, 18, '17 body + 1 back');
  assert.equal(a.total, 20);
  assert.match(a.headline, /18 of your 20 selected shots/);
  assert.match(a.detail, /1024×1024/);
  assert.equal(a.suggestAspect, KREA_SUGGESTED_ASPECT);
  assert.equal(a.suggestLabel, '3:4');
});

test('the wording is honest: tighter, not impossible, and Krea-only', () => {
  const a = kreaFramingAdvisory({ width: 1024, height: 1024, framings: ['body'] });
  assert.match(a.detail, /still generate/i, 'a square reference DOES produce body shots');
  assert.doesNotMatch(a.detail, /impossible|cannot|won't work/i);
  assert.match(a.detail, /other engines follow each shot/i,
    'Klein and the API engines are untouched — the notice must not imply otherwise');
  // Singular/plural is not a cosmetic detail on a one-shot run.
  assert.match(a.headline, /1 of your 1 selected shot is a body or back framing/);
});

test('it stays quiet whenever it would be noise or a guess', () => {
  // Already portrait — nothing to fix.
  assert.equal(kreaFramingAdvisory({ width: 835, height: 1024, framings: bodyShots(17) }), null);
  // Wide reference but only close framings selected: the square is FINE for those.
  assert.equal(kreaFramingAdvisory({ width: 1024, height: 1024, framings: ['face', 'bust'] }), null);
  // Nothing selected yet.
  assert.equal(kreaFramingAdvisory({ width: 1024, height: 1024, framings: [] }), null);
  // Unmeasurable reference (exotic format, missing file, Pillow absent server-side):
  // an unknown shape is not a reason to alarm anyone.
  assert.equal(kreaFramingAdvisory({ width: null, height: null, framings: bodyShots(5) }), null);
  assert.equal(kreaFramingAdvisory({ framings: bodyShots(5) }), null);
  assert.equal(kreaFramingAdvisory(), null);
});

test('an exotic reference still gets a straight answer', () => {
  // Panoramic: the worst case, and the one most likely to surprise someone.
  const pano = kreaFramingAdvisory({ width: 3840, height: 1080, framings: ['body', 'back'] });
  assert.equal(pano.orientation, 'landscape');
  assert.equal(pano.tight, 2);
  // Tiny reference: the shape question is independent of the size question.
  const tiny = kreaFramingAdvisory({ width: 256, height: 256, framings: ['body'] });
  assert.equal(tiny.orientation, 'square');
  assert.match(tiny.sizeLabel, /^256×256$/);
});

test('only body and back are treated as needing vertical room', () => {
  assert.deepEqual(KREA_TIGHT_FRAMINGS, ['body', 'back']);
  for (const fr of ['face', 'bust', 'unknown', '']) {
    assert.equal(kreaFramingAdvisory({ width: 1024, height: 1024, framings: [fr] }), null, fr);
  }
});
