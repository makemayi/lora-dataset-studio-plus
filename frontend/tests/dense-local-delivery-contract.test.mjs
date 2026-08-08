/**
 * A full model that lives on THIS computer — what the hub says about it.
 *
 * The delivery block used to answer one question ("is it on Hugging Face?").
 * It now answers three, and they can disagree: the model can be here and NOT on
 * the Hub (a full quota), on the Hub and not here (an old run), or still on the
 * pod (a transfer to retry). A run whose 26 GB master is safely on the disk must
 * never be described by a red "Full model not found", and a run with nothing
 * anywhere must never be described as delivered.
 *
 * Pure helpers are asserted directly; the block itself is MOUNTED, because the
 * state that ships broken is always the one no test ever rendered.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

const {
  canFetchDenseLocally, canRecheckFullTransformerDelivery, denseDelivery,
  denseHubBackupView, denseLocalArtifactView, denseResumeBlocker,
  fullTransformerArtifactFiles,
} = await import('../src/utils/trainingMode.js')
const { FullArtifactStatus } = await import('../src/pages/CloudRunsPage.jsx')

const legacyRun = (extra = {}) => ({
  run_id: 1, training_mode: 'full_transformer', status: 'done',
  artifact_status: 'available',
  hf_url: 'https://huggingface.co/tester/Krea-2-full-1-dense',
  hf_weight_filename: 'Krea_lds1_dense_000003000.safetensors',
  hf_artifact_proof: { size_bytes: 26_000_000_000 },
  ...extra,
})

const localRun = (extra = {}) => ({
  run_id: 2, training_mode: 'full_transformer', status: 'done',
  dense_delivery: 'both', local_artifact_status: 'available',
  local_artifact_dir: 'D:/lds/checkpoints/run_2',
  local_weight_filename: 'Krea_lds2_dense_000003000.safetensors',
  local_weight_bytes: 26_000_000_000,
  local_fp8_filename: 'Krea_lds2_dense_000003000_fp8.safetensors',
  local_fp8_bytes: 10_000_000_000,
  hub_backup_status: 'done', artifact_status: 'available',
  hf_url: 'https://huggingface.co/tester/Krea-2-full-2-dense',
  ...extra,
})

// --- what a run says about itself ---------------------------------------------

test('a run with no delivery stamp keeps its Hugging-Face-only meaning', () => {
  assert.equal(denseDelivery(legacyRun()), 'hub')
  assert.equal(denseLocalArtifactView(legacyRun()), null,
    'nothing may be said about a local file that never existed')
  assert.equal(denseHubBackupView(legacyRun()), null,
    'a hub-only run keeps the original delivery view, not a backup note')
  assert.equal(canRecheckFullTransformerDelivery(
    legacyRun({ status: 'error_pod_kept', artifact_status: 'missing' })), true)
})

test('a run delivered here only is never offered a Hugging Face verification', () => {
  const run = localRun({ dense_delivery: 'local', status: 'error_pod_kept',
    artifact_status: 'not_requested', local_artifact_status: 'pending' })
  assert.equal(canRecheckFullTransformerDelivery(run), false,
    'there is no Hub delivery to verify — the button would be a dead end')
  assert.equal(canFetchDenseLocally({ ...run, can_fetch_local: true }), true)
})

test('a failed backup reads as a missing backup, never as a missing model', () => {
  const view = denseHubBackupView(localRun({
    hub_backup_status: 'failed', artifact_status: 'missing',
    hub_backup_detail: 'No Hugging Face copy was made (403 storage limit).',
  }))
  assert.equal(view.tone, 'warning')
  assert.match(view.label, /No Hugging Face backup/)
  assert.match(view.detail, /403/)
  // ... and the model itself is still described as available, on the disk.
  const local = denseLocalArtifactView(localRun({ hub_backup_status: 'failed' }))
  assert.equal(local.available, true)
  assert.equal(local.dir, 'D:/lds/checkpoints/run_2')
})

test('the two files are told apart wherever they live', () => {
  const local = fullTransformerArtifactFiles(localRun())
  assert.deepEqual(local.map((f) => f.kind), ['fp8', 'bf16'])
  assert.match(local[0].note, /already on this computer/)
  assert.equal(local[0].primary, true, 'fp8 is the one to load in ComfyUI')
  const hub = fullTransformerArtifactFiles(legacyRun({
    fp8_export_status: 'done',
    fp8_weight_filename: 'Krea_lds1_dense_000003000_fp8.safetensors',
  }))
  assert.deepEqual(hub.map((f) => f.kind), ['fp8', 'bf16'])
  assert.match(hub[0].note, /Download this one/)
})

test('why a full model cannot be continued is said, not left blank', () => {
  assert.equal(denseResumeBlocker(localRun({ resume_steps: [3000] })), null)
  assert.match(denseResumeBlocker(localRun({
    dense_delivery: 'local', resume_steps: [] })), /no Hugging Face copy/)
  assert.match(denseResumeBlocker(localRun({ resume_steps: [] })),
    /no verified Hugging Face copy/)
})

// --- the block itself ----------------------------------------------------------

test('the delivered model, its folder and its backup all render', () => {
  const html = renderToStaticMarkup(createElement(FullArtifactStatus, {
    run: localRun(), onRecheck: () => {}, onFetch: () => {},
  }))
  assert.match(html, /Full model on this computer/)
  assert.match(html, /D:\/lds\/checkpoints\/run_2/)
  assert.match(html, /Krea_lds2_dense_000003000_fp8\.safetensors/)
  assert.match(html, /Hugging Face backup made/)
  assert.doesNotMatch(html, /Full model not found/)
})

test('a kept pod whose file is still up there offers to fetch it', () => {
  const html = renderToStaticMarkup(createElement(FullArtifactStatus, {
    run: localRun({
      status: 'error_pod_kept', local_artifact_status: 'pending',
      local_artifact_dir: '', local_weight_filename: '',
      hub_backup_status: '', artifact_status: 'not_requested',
      dense_delivery: 'local', can_fetch_local: true,
      local_artifact_detail: 'the full model could not be brought home (truncated)',
    }),
    onRecheck: () => {}, onFetch: () => {},
  }))
  assert.match(html, /Full model not downloaded/)
  assert.match(html, /Fetch to this computer/)
  assert.doesNotMatch(html, /Verify Hugging Face delivery/)
})

test('a transfer in flight offers to stop it, and says what stopping keeps', () => {
  const html = renderToStaticMarkup(createElement(FullArtifactStatus, {
    run: localRun({
      status: 'downloading', local_artifact_status: 'pending',
      phase_detail: 'Fetching the full model to this computer — 12 / 26 GB',
      dense_fetch_active: true, can_fetch_local: false,
      hub_backup_status: '', local_weight_filename: '',
    }),
    onFetch: () => {}, fetching: true,
  }))
  assert.match(html, /Downloading the full model/)
  assert.match(html, /12 \/ 26 GB/)
  assert.match(html, /what landed is kept/)
})

test('an old Hugging-Face-only run still renders its Hub block, and only that', () => {
  const html = renderToStaticMarkup(createElement(FullArtifactStatus, {
    run: legacyRun(), onRecheck: () => {},
  }))
  // "delivered", not "available": nothing has asked the Hub, and this block
  // used to offer the link below over a repository that had been deleted.
  assert.match(html, /Full model delivered/)
  assert.match(html, /not re-checked since/)
  assert.match(html, /Open private model on Hugging Face/)
  assert.doesNotMatch(html, /on this computer/)
})

test('the Hub block of a legacy run collapses to the truth once the repo is gone', () => {
  const html = renderToStaticMarkup(createElement(FullArtifactStatus, {
    run: legacyRun(), onRecheck: () => {}, presence: { state: 'gone' },
  }))
  assert.match(html, /Full model no longer on Hugging Face/)
  // The dead link is the thing a user actually clicked. It must be gone from
  // the markup, not merely relabelled.
  assert.doesNotMatch(html, /Open private model on Hugging Face/)
  assert.doesNotMatch(html, /Inspect Hugging Face repository/)
  assert.doesNotMatch(html, /huggingface\.co/)
})
