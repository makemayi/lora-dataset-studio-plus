import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { cloudTrainingLaunchPayload } from './checkpointBrowser.js';
import {
  TRAINING_MODE_FULL_TRANSFORMER,
  TRAINING_MODE_LORA,
  canRecheckFullTransformerDelivery,
  cloudTierEstimateView,
  fullTransformerArtifactView,
  fullTransformerRecheckOutcome,
  fullTransformerUnavailableReason,
  hfCloudTokenReadiness,
  isFullTransformerRun,
  isFullTransformerEligible,
  normalizeTrainingMode,
  trainingModeSettingsPayload,
  trainingModeLabel,
} from './trainingMode.js';
import { preflightUrl } from '../components/dataset/preflightLane.js';

const panel = readFileSync(new URL('../components/dataset/TrainingPanel.jsx', import.meta.url), 'utf8');
const datasetHook = readFileSync(new URL('../hooks/useDataset.js', import.meta.url), 'utf8');
const runsPage = readFileSync(new URL('../pages/CloudRunsPage.jsx', import.meta.url), 'utf8');

test('training mode enum is exact and every legacy or invalid value falls back to LoRA', () => {
  assert.equal(TRAINING_MODE_LORA, 'lora');
  assert.equal(TRAINING_MODE_FULL_TRANSFORMER, 'full_transformer');
  assert.equal(normalizeTrainingMode('full_transformer'), 'full_transformer');
  assert.equal(normalizeTrainingMode('full_model'), 'lora');
  assert.equal(normalizeTrainingMode(undefined), 'lora');
  assert.equal(trainingModeLabel('lora'), 'LoRA');
  assert.equal(trainingModeLabel('full_transformer'), 'Full model');
});

test('dense eligibility is limited to the official Krea 2 Raw recipe', () => {
  assert.equal(isFullTransformerEligible({ trainType: 'krea', variant: 'base', baseModel: '' }), true);
  assert.equal(isFullTransformerEligible({ trainType: 'krea', variant: 'turbo', baseModel: '' }), false);
  assert.equal(isFullTransformerEligible({ trainType: 'zimage', variant: 'base', baseModel: '' }), false);
  assert.equal(isFullTransformerEligible({ trainType: 'krea', variant: 'base', baseModel: 'custom.safetensors' }), false);
  assert.equal(isFullTransformerEligible({
    trainType: 'krea', variant: 'base', baseModel: '', customBase: true,
  }), false);
  assert.match(fullTransformerUnavailableReason({ trainType: 'krea', variant: 'turbo' }), /Raw/);
});

test('cloud launch and preflight carry the mode while LoRA stays the default regression path', () => {
  assert.equal(cloudTrainingLaunchPayload({ trainType: 'krea', variant: 'base' }).training_mode, 'lora');
  assert.equal(cloudTrainingLaunchPayload({
    trainType: 'krea', variant: 'base', trainingMode: 'full_transformer', gpuName: 'H100',
  }).training_mode, 'full_transformer');
  assert.equal(preflightUrl(9, {
    trainType: 'krea', variant: 'base', baseModel: '',
    trainingMode: 'full_transformer', lane: 'cloud',
  }), '/api/dataset/9/train/preflight?train_type=krea&variant=base&base_model=&training_mode=full_transformer&lane=cloud');
  assert.deepEqual(trainingModeSettingsPayload('full_transformer', {
    trainType: 'krea', variant: 'base', baseModel: '',
    disableSliderForFullTransformer: true,
  }), {
    training_mode: 'full_transformer', train_type: 'krea', variant: 'base', base_model: '',
    disable_slider_for_full_transformer: true,
  });
});

test('a full artifact link exists only after verified availability', () => {
  const pending = fullTransformerArtifactView({
    training_mode: 'full_transformer', artifact_status: 'verification_pending',
    hf_url: 'https://huggingface.co/me/private', artifact_status_detail: 'token timed out',
  });
  assert.equal(pending.href, null);
  assert.equal(pending.repositoryHref, 'https://huggingface.co/me/private');
  assert.equal(pending.tone, 'warning');
  assert.equal(pending.label, 'Hugging Face verification pending');
  assert.match(pending.detail, /token timed out/);

  const missing = fullTransformerArtifactView({
    artifact_status: 'missing', hf_url: 'https://huggingface.co/me/private',
  });
  assert.equal(missing.href, null);
  assert.equal(missing.tone, 'error');
  assert.equal(missing.label, 'Full model not found');

  const available = fullTransformerArtifactView({
    artifact_status: 'available', hf_url: 'https://huggingface.co/me/private',
  });
  assert.equal(available.href, 'https://huggingface.co/me/private');
  assert.equal(available.available, true);
  assert.equal(available.label, 'Full model available');
  assert.equal(isFullTransformerRun({ training_mode: 'full_transformer' }), true);
});

test('dense token readiness fails closed only when the backend explicitly reports a problem', () => {
  assert.deepEqual(hfCloudTokenReadiness({}), {
    signaled: false, ready: true, blocked: false, detail: null,
  });
  const offerFailure = hfCloudTokenReadiness({
    hf_cloud_token: {
      ok: false, configured: true,
      error: 'HF_CLOUD_TOKEN cannot write to the dedicated namespace',
    },
  });
  assert.equal(offerFailure.blocked, true);
  assert.match(offerFailure.detail, /dedicated namespace/);
  assert.equal(hfCloudTokenReadiness({
    hf_cloud_token: { ok: true, configured: true, namespace: 'lds-deliveries' },
  }).ready, true);
  assert.equal(hfCloudTokenReadiness({
    checks: [{ id: 'hf_cloud_token', status: 'fail', detail: 'fine-grained scope invalid' }],
    hf_cloud_token_status: { ok: false, configured: true },
  }).blocked, true);
});

test('kept dense runs can recheck verification or pending pod cleanup', () => {
  assert.equal(canRecheckFullTransformerDelivery({
    training_mode: 'full_transformer', status: 'error_pod_kept',
    artifact_status: 'verification_pending',
  }), true);
  assert.equal(canRecheckFullTransformerDelivery({
    training_mode: 'full_transformer', status: 'error_pod_kept',
    artifact_status: 'available',
  }), true);
  assert.equal(canRecheckFullTransformerDelivery({
    training_mode: 'full_transformer', status: 'error_pod_kept',
    artifact_status: 'available', artifact_cleanup_status: 'pending',
  }), true);
  assert.equal(canRecheckFullTransformerDelivery({
    training_mode: 'full_transformer', status: 'error_pod_kept',
    artifact_status: 'available', artifact_cleanup_status: 'complete',
  }), false);
  assert.equal(canRecheckFullTransformerDelivery({
    training_mode: 'lora', status: 'error_pod_kept', artifact_status: 'verification_pending',
  }), false);
});

test('legacy verified kept rows default to visible cleanup-pending state', () => {
  const legacy = fullTransformerArtifactView({
    training_mode: 'full_transformer', status: 'error_pod_kept',
    artifact_status: 'available',
    hf_url: 'https://huggingface.co/me/private',
  });
  assert.equal(legacy.available, true);
  assert.equal(legacy.cleanupPending, true);
  assert.equal(legacy.href, 'https://huggingface.co/me/private');
  assert.equal(legacy.tone, 'warning');
  assert.match(legacy.detail, /may still be billing/);
});

test('verified model and pending cleanup never produce a pod-released success', () => {
  const pending = fullTransformerRecheckOutcome({
    ok: true, delivery: 'available', cleanup_pending: true,
  });
  assert.equal(pending.kind, 'warning');
  assert.match(pending.text, /Hugging Face model verified and available/);
  assert.match(pending.text, /may still be billing/);

  const complete = fullTransformerRecheckOutcome({
    ok: true, delivery: 'available', cleanup_pending: false,
  });
  assert.equal(complete.kind, 'success');
  assert.match(complete.text, /pod cleanup is confirmed/);
});

test('dense offers never reuse an unlabelled or unavailable estimate', () => {
  assert.deepEqual(cloudTierEstimateView({
    est_minutes: 42, est_cost: 1.25, exceeds_cap: true,
  }, { fullMode: true }), {
    available: false, minutes: null, cost: null, exceedsCap: false, status: null,
  });
  assert.equal(cloudTierEstimateView({
    estimate_status: 'unavailable', est_minutes: 42, est_cost: 1.25, exceeds_cap: true,
  }, { fullMode: true }).available, false);
  assert.equal(cloudTierEstimateView({
    estimate_status: 'available', est_minutes: null, est_cost: null, exceeds_cap: true,
  }, { fullMode: true }).available, false);
  assert.deepEqual(cloudTierEstimateView({
    estimate_status: 'available', est_minutes: 42, est_cost: 1.25, exceeds_cap: true,
  }, { fullMode: true }), {
    available: true, minutes: 42, cost: 1.25, exceedsCap: true, status: 'available',
  });
  // Legacy LoRA offers remain usable while the backend rolls out estimate_status.
  assert.equal(cloudTierEstimateView({ est_minutes: 42 }, { fullMode: false }).available, true);
});

test('MVP copy and artifact actions distinguish a full model from a LoRA', () => {
  assert.match(panel, /LoRA/);
  assert.match(panel, /Full model/);
  assert.match(panel, /80 GB VRAM GPU/);
  assert.match(panel, /at least 200 GB disk/);
  assert.match(panel, /~26 GB/);
  assert.match(panel, /private Hugging Face repository/);
  assert.match(panel, /much larger, more diverse dataset/);
  assert.match(panel, /Open private model on Hugging Face/);
  assert.match(panel, /!fullMode && !cloudActiveHere/);
  assert.match(panel, /HF_CLOUD_TOKEN/);
  assert.match(panel, /ArrowLeft/);
  assert.match(panel, /tabIndex=\{!fullMode \|\| !fullTransformerEligible \? 0 : -1\}/);
  assert.match(panel, /aria-describedby/);
  assert.doesNotMatch(panel,
    /Modèle complet|Fine-tuning complet|Recette dense verrouillée|Lancer le fine-tuning complet|Choisir un GPU 80 Go|Livraison Hugging Face bloquée|estimation dense indisponible/);
});

test('dense Advanced renders only the locked server recipe and its honored steps input', () => {
  const recipe = panel.slice(
    panel.indexOf('// FULL_TRANSFORMER_ADVANCED_RECIPE_START'),
    panel.indexOf('// FULL_TRANSFORMER_ADVANCED_RECIPE_END'),
  );
  assert.match(panel, /Locked full-model recipe · steps/);
  assert.match(recipe, /Official Krea 2 Raw · full transformer · unquantized/);
  assert.match(recipe, /1024 px · batch 1 · bf16/);
  assert.match(recipe, /Adafactor · learning rate 1e-6/);
  assert.match(recipe, /Gradient checkpointing · cached latents \+ text embeddings/);
  assert.match(recipe, /Checkpoint \+ preview every 250 steps · keep 1 checkpoint/);
  assert.match(recipe, /80 GB VRAM GPU · at least 200 GB disk/);
  assert.match(recipe, /The only editable setting in this full-model recipe/);
  assert.equal([...recipe.matchAll(/<input\b/g)].length, 1);
  assert.match(recipe, /setStepsOverride\(event\.target\.value\)/);
  assert.doesNotMatch(recipe, /<select\b|<button\b|saveAdv\(|setBase\(|setVariant\(|setMasked\(/,
    'the dense recipe card must not grow an ignored LoRA control');
});

test('the full Advanced branch cannot render the unchanged LoRA controls', () => {
  const branch = panel.slice(
    panel.indexOf('{fullMode ? (', panel.indexOf('FULL_TRANSFORMER_ADVANCED_BRANCH_START')),
    panel.indexOf('FULL_TRANSFORMER_ADVANCED_BRANCH_END'),
  );
  const split = branch.indexOf(') : (<>');
  assert.ok(split > 0, 'Advanced must have explicit dense and LoRA render arms');
  const denseArm = branch.slice(0, split);
  const loraArm = branch.slice(split);
  assert.match(denseArm, /<FullTransformerAdvancedRecipe/);
  assert.doesNotMatch(denseArm, /Presets|CUSTOM_BASE_SENTINEL|advNetworkType|Masked \(bg 10%\)|saveAdv\(/);
  assert.match(loraArm, /Presets/);
  assert.match(loraArm, /CUSTOM_BASE_SENTINEL/);
  assert.match(loraArm, /advNetworkType/);
  assert.match(loraArm, /Masked \(bg 10%\)/);
  assert.match(loraArm, /saveAdv\(/);
  assert.match(panel, /advancedOpen && trainingMode !== TRAINING_MODE_FULL_TRANSFORMER/);
});

test('local hook persists and launches with the canonical mode', () => {
  assert.match(datasetHook, /trainingModeSettingsPayload\(trainingMode, selection\)/);
  assert.match(datasetHook, /slider: d\.slider \?\? null/);
  assert.match(datasetHook, /training_mode: normalizeTrainingMode\(opts\.trainingMode\)/);
  assert.match(datasetHook, /catch \(error\)[\s\S]*return null/);
});

test('offers use the exact recipe and refetch when any recipe input changes', () => {
  assert.match(panel, /new URLSearchParams\(\{\s*train_type: trainType,\s*variant,\s*base_model: base \?\? '',\s*training_mode:/);
  assert.match(panel, /\[datasetId, trainType, variant, base, trainingMode, steps\]/);
  assert.match(panel, /full-model estimate unavailable — hourly price only/);
  assert.match(panel, /hasUsableEstimate/);
  assert.match(panel, /hfCloudTokenReadiness\(data \|\| \{\}\)/);
  assert.match(panel, /disabled=\{!selected \|\| launching \|\| !customBaseReady \|\| hfTokenBlocked\}/);
  assert.match(panel, /focus="HF_CLOUD_TOKEN"/);
  assert.match(panel, /checksDenseCloudToken/);
});

test('empty cloud offers preserve the cap message and link to its exact setting', () => {
  const emptyOffersStart = panel.indexOf('{!loading && !error && tiers.length === 0 && (');
  const populatedOffersStart = panel.indexOf('{tiers.length > 0 && (', emptyOffersStart);
  assert.ok(emptyOffersStart >= 0 && populatedOffersStart > emptyOffersStart,
    'the empty-offers branch must remain distinct from the populated offer list');

  const emptyOffersBranch = panel.slice(emptyOffersStart, populatedOffersStart);
  assert.match(emptyOffersBranch,
    /No GPU available under \$\{data\?\.max_price_per_hour\}\/h right now/);
  assert.match(emptyOffersBranch,
    /<SettingsLink section="training" focus="cloud-max-price-per-hour">\s*increase the price cap in Settings\s*<\/SettingsLink>/);
  assert.equal([...panel.matchAll(/focus="cloud-max-price-per-hour"/g)].length, 1,
    'the price-cap link must appear only in the tiers.length === 0 branch');
});

test('mode persistence is atomic and the incompatible fallback is not optimistic', () => {
  assert.match(panel, /setDatasetTrainingMode\?\.\(TRAINING_MODE_LORA, nextSelection\)/);
  assert.match(panel, /setDatasetTrainingMode\?\.\(\s*TRAINING_MODE_LORA,\s*fullTransformerSelection/);
  const fallback = panel.slice(panel.indexOf('// Family, Krea variant'), panel.indexOf('const toggleSliderMode'));
  const persistAt = fallback.indexOf('await ds.setDatasetTrainingMode');
  const showLoraAt = fallback.indexOf('setTrainingMode(TRAINING_MODE_LORA)');
  assert.ok(persistAt >= 0 && showLoraAt > persistAt,
    'the UI must not claim LoRA before the save resolves');
  assert.match(fallback, /const info = await ds\.trainBaseInfo/);
  const modeChange = panel.slice(panel.indexOf('const onTrainingModeChange'), panel.indexOf('const onTrainingModeKeyDown'));
  assert.match(modeChange, /disableSliderForFullTransformer: nextMode === TRAINING_MODE_FULL_TRANSFORMER/);
  assert.match(modeChange, /saved\.slider\?\.enabled !== false/);
  assert.doesNotMatch(modeChange, /saveSlider\(/,
    'switching to full must not issue or roll back a separate Slider request');
  assert.ok(modeChange.indexOf('setTrainingMode(canonicalMode)') > modeChange.indexOf('saved.slider?.enabled !== false'),
    'the UI must wait for the canonical mode + disabled Slider response');
});

test('full run cards surface Hub status and suppress LoRA-only actions', () => {
  assert.match(runsPage, /function FullArtifactStatus/);
  assert.match(runsPage, /!fullModel && run\.checkpoint_ready/);
  assert.match(runsPage, /!fullModel && run\.dataset_id != null/);
  assert.match(runsPage, /!fullModel && run\.record_id != null/);
  assert.match(runsPage, /isFullTransformerRun\(run\) && \([\s\S]*?<FullArtifactStatus run=\{run\}/);
  assert.match(runsPage, /AI Toolkit uploads the full model to Hugging Face only when the run finishes cleanly/);
  assert.match(runsPage, /Verify Hugging Face delivery/);
  assert.match(runsPage, /Retry pod cleanup/);
  assert.match(runsPage, /\/api\/dataset\/train\/cloud\/recheck-delivery/);
  assert.match(runsPage, /Inspect Hugging Face repository \(delivery not verified\)/);
});

test('user-facing full-model recovery copy never falls back to dense terminology', () => {
  assert.match(panel, /the latest full-model checkpoint may not have reached Hugging Face/);
  assert.match(runsPage, /Verify or recover the full-model weights on Hugging Face/);
  assert.doesNotMatch(`${panel}\n${runsPage}`, /\bdense (?:checkpoint|weights)\b/i);
});
