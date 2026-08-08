/* A broken OPTIONAL Klein asset must be VISIBLE without being dressed as a blocker.

   The defect these tests pin: the consistency LoRA is not part of
   KLEIN_REQUIRED_ASSETS — the backend's own readiness verdict ignores it
   (klein_engine_ready reads KLEIN_REQUIRED only), so a corrupted LoRA does not stop
   a single generation. Yet the Setup install menu judged every unreadable file with
   one unfiltered blockingInvalid() call, so that LoRA wore the exact red
   "⚠ On disk, unreadable" of a dead UNET — the screen read as blocked while nothing
   was. Meanwhile the ComfyUI step's download buttons filtered on the REQUIRED list
   and therefore printed "✓ Installed" for the very same unreadable file.

   Two opposite lies about one file, on two screens. The fix is a severity, not a
   filter: both surfaces show it, neither claims the engine is down.

   These assert the PROPERTY (does this file gate the engine / is it still visible),
   not the wording. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { deriveSetupSteps, installCatalog } from './useSetupSteps.js';
import { KLEIN_REQUIRED_ASSETS, KLEIN_OPTIONAL_ASSETS, kleinAssetBlocks }
  from '../utils/kleinAssets.js';

const BROKEN_LORA = {
  asset: 'klein_lora', filename: 'Flux2-Klein-9B-consistency-V2.safetensors',
  verdict: 'truncated_or_garbage', blocking: true,
  reason: 'Flux2-Klein-9B-consistency-V2.safetensors is shorter than its header declares',
};
const BROKEN_UNET = {
  asset: 'klein_model', filename: 'flux-2-klein-9b-kv-fp8.safetensors',
  verdict: 'truncated_or_garbage', blocking: true,
  reason: 'flux-2-klein-9b-kv-fp8.safetensors is shorter than its header declares',
};

// A fully-set-up install whose ONLY fault is the corrupted optional LoRA. Klein is
// ready — that is the backend's verdict, and it is the correct one.
const loraOnlyCaps = () => ({
  engines: { klein: true, krea: false },
  comfyui: {
    reachable: true, dir_valid: true,
    klein_missing: [], klein_unsupported_enums: [], klein_invalid: [BROKEN_LORA],
  },
});

const comfyStep = (caps) => deriveSetupSteps(caps).find((s) => s.id === 'comfyui');
const byAction = (rows) => Object.fromEntries(rows.map((r) => [r.action, r]));

test('the recommended LoRA is not in the required list, and knows it', () => {
  assert.ok(!KLEIN_REQUIRED_ASSETS.includes('klein_lora'));
  assert.deepEqual(KLEIN_OPTIONAL_ASSETS, ['klein_lora']);
  assert.equal(kleinAssetBlocks('klein_lora'), false);
  for (const a of KLEIN_REQUIRED_ASSETS) assert.equal(kleinAssetBlocks(a), true);
  // An asset this module never heard of must NOT be quietly downgraded.
  assert.equal(kleinAssetBlocks('krea_model'), true);
  assert.equal(kleinAssetBlocks('something_new'), true);
});

test('a corrupted consistency LoRA does not block the ComfyUI step', () => {
  const step = comfyStep(loraOnlyCaps());
  assert.equal(step.hasKlein, true);
  assert.equal(step.status, 'ready');
  assert.equal(step.kleinReason, null);
  // The GATING list stays empty — that is what readiness is derived from.
  assert.deepEqual(step.kleinBroken, []);
  assert.equal(step.kleinFilesReady, true);
});

test('…and is still VISIBLE — the fix must not trade one defect for another', () => {
  const step = comfyStep(loraOnlyCaps());
  // The list the download buttons read carries it, so the LoRA button can no longer
  // print "✓ Installed" over an unreadable file.
  assert.deepEqual(step.kleinBrokenAll.map((i) => i.asset), ['klein_lora']);
  assert.equal(step.kleinBrokenAll[0].filename, BROKEN_LORA.filename);
});

test('the install menu marks it broken-but-optional, never a dead engine', () => {
  const row = byAction(installCatalog(loraOnlyCaps())).klein_lora;
  assert.equal(row.state, 'broken_optional');
  assert.equal(row.blocking, false);
  // Not "✓ Installed" (it cannot load) and not a bare "✗" (the file is right there).
  assert.match(row.stateLabel, /unreadable/i);
  assert.match(row.brokenReason, /still works without it/i);
  // present:false is what keeps it in the install plans, so the button has work to do.
  assert.equal(row.present, false);
  assert.equal(row.available, true);
  // The required weights on the same install are untouched and green.
  const cat = byAction(installCatalog(loraOnlyCaps()));
  for (const a of KLEIN_REQUIRED_ASSETS) {
    assert.equal(cat[a].present, true, `${a} must stay installed`);
    assert.equal(cat[a].state, undefined, `${a} must carry no fault state`);
  }
});

test('a corrupted REQUIRED weight keeps the blocking treatment', () => {
  const caps = {
    engines: { klein: false, krea: false },
    comfyui: {
      reachable: true, dir_valid: true,
      klein_missing: [], klein_unsupported_enums: [],
      klein_invalid: [BROKEN_UNET, BROKEN_LORA],
    },
  };
  const step = comfyStep(caps);
  assert.equal(step.hasKlein, false);
  assert.equal(step.status, 'partial');
  assert.deepEqual(step.kleinBroken.map((i) => i.asset), ['klein_model']);
  assert.deepEqual(step.kleinBrokenAll.map((i) => i.asset), ['klein_model', 'klein_lora']);
  const cat = byAction(installCatalog(caps));
  assert.equal(cat.klein_model.state, 'broken');
  assert.equal(cat.klein_model.blocking, true);
  assert.equal(cat.klein_lora.state, 'broken_optional');
  assert.equal(cat.klein_lora.blocking, false);
});

test('every Krea asset is required, so none of them is ever downgraded', () => {
  const caps = {
    engines: { klein: false, krea: false },
    comfyui: {
      reachable: true, dir_valid: true, krea_nodes_installed: true,
      krea_missing: [], krea_invalid: [{
        asset: 'krea_identity_lora', filename: 'krea-identity.safetensors',
        verdict: 'html_or_text', blocking: true, reason: 'looks like an HTML page',
      }],
    },
  };
  const row = byAction(installCatalog(caps)).krea_identity_lora;
  assert.equal(row.state, 'broken');
  assert.equal(row.blocking, true);
});

test('an advisory too_small optional asset is not a fault at all', () => {
  // `blocking:false` never reaches either list — a small-but-loadable file is the
  // user's business, and inventing a badge for it is how alarms stop meaning things.
  const caps = {
    engines: { klein: true },
    comfyui: {
      reachable: true, dir_valid: true, klein_missing: [],
      klein_invalid: [{ asset: 'klein_lora', filename: 'l.safetensors',
        verdict: 'too_small', blocking: false, reason: 'l.safetensors is only 10 B' }],
    },
  };
  assert.deepEqual(comfyStep(caps).kleinBrokenAll, []);
  assert.equal(byAction(installCatalog(caps)).klein_lora.state, undefined);
  assert.equal(byAction(installCatalog(caps)).klein_lora.present, true);
});
