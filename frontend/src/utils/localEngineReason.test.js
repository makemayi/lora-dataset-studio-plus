import test from 'node:test';
import assert from 'node:assert/strict';
import {
  kleinUnavailableReason, localEngineUnavailableReason, localEngineReadiness, hasComfyui,
} from './localEngineReason.js';

/* zigzag4794's install (Discord), as capabilities reported it: ComfyUI up, every
   required Klein file in its folder at a plausible 9.5 GB, and both model files
   listed under `klein_invalid` with verdict `truncated_or_garbage` — an
   interrupted or corrupted download. Setup said "✓ Installed"; Generate said
   "⚠ Klein model missing — download it in the Setup step". */
const ZIGZAG = {
  engines: { klein: false },
  comfyui: {
    reachable: true,
    klein_missing: ['klein_lora'],          // recommended only — does not gate
    klein_unsupported_enums: [],
    klein_invalid: [
      { asset: 'klein_model', filename: 'flux-2-klein-9b-kv-fp8.safetensors',
        verdict: 'truncated_or_garbage', blocking: true,
        reason: 'flux-2-klein-9b-kv-fp8.safetensors is shorter than its header declares' },
      { asset: 'klein_text_encoder', filename: 'qwen_3_8b_fp8mixed.safetensors',
        verdict: 'truncated_or_garbage', blocking: true,
        reason: 'qwen_3_8b_fp8mixed.safetensors is shorter than its header declares' },
    ],
  },
};

test('a file ON DISK is never called "missing" — zigzag4794 (Discord)', () => {
  const reason = localEngineUnavailableReason('klein', ZIGZAG);
  // The guard-rail. Two OPPOSITE problems were collapsed into one sentence, and
  // their fixes differ: download it vs DELETE it and download it again. Sending
  // someone to fetch a file they are staring at is worse than saying nothing.
  assert.doesNotMatch(reason, /missing/i);
  assert.match(reason, /flux-2-klein-9b-kv-fp8\.safetensors/);
  assert.match(reason, /cannot be loaded/);
  assert.match(reason, /cut short or corrupted/);
  assert.match(reason, /Delete it and download it again/);
});

test('a corrupted weight and an absent one do not get the same sentence', () => {
  const absent = kleinUnavailableReason({ missingAssets: ['klein_model'] });
  const corrupt = kleinUnavailableReason({
    invalidAssets: [{ asset: 'klein_model', filename: 'k.safetensors',
      verdict: 'truncated_or_garbage', blocking: true }],
  });
  assert.match(absent, /missing/);
  assert.doesNotMatch(absent, /Delete it/);
  assert.doesNotMatch(corrupt, /missing/i);
  assert.match(corrupt, /Delete it/);
});

test('the advisory too_small is not dressed up as a broken install', () => {
  // The file loads; it is only suspiciously small. Nothing to delete.
  const reason = kleinUnavailableReason({
    invalidAssets: [{ asset: 'klein_model', filename: 'k.safetensors',
      verdict: 'too_small', blocking: false }],
  });
  assert.doesNotMatch(reason, /Delete it/);
});

test('the last resort states what is known instead of guessing "model missing"', () => {
  // Reachable, nothing missing, nothing broken, no unsupported value — and the
  // backend still says no. Every nameable gap is handled above this line, so the
  // old catch-all was wrong in 100% of the cases that reached it.
  const reason = kleinUnavailableReason({});
  assert.doesNotMatch(reason, /model missing/);
  assert.match(reason, /diagnostic/i);
});

test('readiness is READ from the backend, never recomputed', () => {
  // The whole point: one verdict, every screen. A caps payload that says the
  // engine is up is ready even though the raw ingredient lists below look alarming
  // — because the backend, not the ingredient lists, is the authority.
  const ready = localEngineReadiness('klein', {
    engines: { klein: true },
    comfyui: { reachable: true, klein_missing: ['klein_lora'] },
  });
  assert.deepEqual(ready, { engine: 'klein', ready: true, verified: true, reason: null });

  const zig = localEngineReadiness('klein', ZIGZAG);
  assert.equal(zig.ready, false);
  assert.equal(zig.verified, true);
  assert.equal(zig.reason, localEngineUnavailableReason('klein', ZIGZAG));
});

test('"could not check" is not "checked and fine"', () => {
  // ComfyUI down: the probes that need it fail OPEN (they report no gap rather
  // than inventing one), so an empty warning list here proves nothing. `verified`
  // is how a screen says that out loud instead of rendering a ✓ it did not earn.
  const down = localEngineReadiness('klein', {
    engines: { klein: false },
    comfyui: { reachable: false, klein_missing: [], klein_unsupported_enums: [] },
  });
  assert.equal(down.ready, false);
  assert.equal(down.verified, false);
});

/* One gap, one sentence. Both the generation panel and the ✦ Edit-reference modal
   read this module, so "why can't I pick Klein" cannot be answered two different
   ways two clicks apart. Each branch answers ONE question: what do I do next? */

test('the reasons are ordered by what has to be fixed FIRST', () => {
  assert.match(kleinUnavailableReason({ enabledInSettings: false }), /disabled in Settings/);
  // A disabled engine wins over everything: the asset list is meaningless until
  // the engine is on at all.
  assert.match(kleinUnavailableReason({
    enabledInSettings: false, comfyuiReachable: false, missingAssets: ['klein_model'],
  }), /disabled in Settings/);
  assert.match(kleinUnavailableReason({ comfyuiReachable: false }), /Configure ComfyUI/);
});

test('an enum gap is named BEFORE the weights — it is a different fix', () => {
  // Reachable ComfyUI, every file in place, but a widget VALUE the graph pins is
  // not offered. Blaming a weight here sends the user to re-check a green step.
  const reason = kleinUnavailableReason({
    unsupportedEnums: [{ node_id: '31', class_type: 'KSampler', input: 'scheduler',
      value: 'beta57', pack: 'RES4LYF', url: 'https://example.invalid/pack' }],
  });
  assert.doesNotMatch(reason, /missing — download/);
  assert.match(reason, /RES4LYF|scheduler|beta57/);
});

test('the missing Klein weights are NAMED, not collapsed into "the model"', () => {
  // The old text always blamed the UNET, sending people to models/unet/klein/
  // even when the real gap was the text encoder.
  const reason = kleinUnavailableReason({ missingAssets: ['klein_text_encoder', 'klein_vae'] });
  assert.match(reason, /text encoder/);
  assert.match(reason, /VAE/);
});

test('an available engine has no reason at all', () => {
  const caps = { engines: { klein: true, krea: true }, comfyui: { reachable: true } };
  assert.equal(localEngineUnavailableReason('klein', caps), null);
  assert.equal(localEngineUnavailableReason('krea', caps), null);
});

test('an API engine is never explained here — nothing local can predict its failures', () => {
  const caps = { engines: {}, comfyui: { reachable: false } };
  for (const e of ['nanobanana', 'chatgpt', 'openrouter']) {
    assert.equal(localEngineUnavailableReason(e, caps), null, e);
  }
});

test('Krea reports its own extra failure mode: the custom-node pack', () => {
  const caps = {
    engines: { krea: false },
    comfyui: { reachable: true, krea_nodes_missing: ['Krea2EditModelPatch'],
      krea_nodes_installed: false },
  };
  assert.match(localEngineUnavailableReason('krea', caps), /node pack/);
});

test('a node pack ON DISK but not loaded says "restart", not "install"', () => {
  // The app installs the pack itself now; telling someone to install what they
  // just watched install is a working feature reading as broken.
  const caps = {
    engines: { krea: false },
    comfyui: { reachable: true, krea_nodes_missing: ['Krea2EditModelPatch'],
      krea_nodes_installed: true },
  };
  assert.match(localEngineUnavailableReason('krea', caps), /restart ComfyUI/);
});

test('the Settings gate is honoured when the list is passed, skipped when it is not', () => {
  const caps = { engines: { krea: false }, comfyui: { reachable: true } };
  assert.match(localEngineUnavailableReason('krea', caps, ['klein']), /disabled in Settings/);
  // FOLLOW-UP, deliberately left visible: the ✦ Edit modal passes no list (it does
  // not fetch /api/settings), so an engine disabled there still appears in that
  // modal — gated only by live readiness. Passing null is that state, not an
  // accident: the gate is opt-in until the modal has the list to give it.
  assert.doesNotMatch(localEngineUnavailableReason('krea', caps, null) || '',
    /disabled in Settings/);
});

test('hasComfyui is the line between "a fixable gap" and "a product you have not got"', () => {
  assert.equal(hasComfyui(null), false);
  assert.equal(hasComfyui({ comfyui: {} }), false);
  // Configured but currently down still counts: that is fixable, and hiding the
  // engines while the user restarts ComfyUI would look like a lost feature.
  assert.equal(hasComfyui({ comfyui: { dir_configured: true, reachable: false } }), true);
  assert.equal(hasComfyui({ comfyui: { reachable: true } }), true);
});
