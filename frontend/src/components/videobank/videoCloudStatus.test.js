import test from 'node:test'
import assert from 'node:assert/strict'

import {
  isActive, launchBlockedReason, runSummary, canRetry, canContinue, stepLabel,
} from './videoCloudStatus.js'
import {
  videoDatasetCloudUrl, videoDatasetCloudProgressUrl,
  videoDatasetCloudCheckpointsUrl, videoDatasetCheckpointUrl,
  videoDatasetCloudRetryUrl, videoDatasetCloudContinueUrl,
} from './videoBankApi.js'

test('the cloud URLs name the VIDEO dataset table, never the face one', () => {
  // Same integer, two tables. `/api/dataset/7/...` is a face dataset and
  // `/api/video-dataset/7/...` is not, and the difference is which weights the
  // server hands over.
  assert.equal(videoDatasetCloudUrl(7), '/api/video-dataset/7/train/cloud')
  assert.equal(videoDatasetCloudProgressUrl(7),
    '/api/video-dataset/7/train/cloud/progress')
  assert.equal(videoDatasetCloudCheckpointsUrl(7),
    '/api/video-dataset/7/train/cloud/checkpoints')
  assert.equal(videoDatasetCloudRetryUrl(7), '/api/video-dataset/7/train/cloud/retry')
  assert.equal(videoDatasetCloudContinueUrl(7),
    '/api/video-dataset/7/train/cloud/continue')
})

test('a checkpoint filename is ENCODED, not interpolated', () => {
  // It carries the dataset's own name, which is whatever the user typed.
  assert.equal(
    videoDatasetCheckpointUrl(7, 12, 'video_surf & sun_000000400_high_noise.safetensors'),
    '/api/video-dataset/7/train/cloud/checkpoint?run_id=12&filename='
    + 'video_surf+%26+sun_000000400_high_noise.safetensors')
})

test('an unverified target is refused BEFORE a GPU is rented', () => {
  // The catalogue carries `training_verified` because for several video models
  // the geometry is known and no LoRA trainer is. Discovering that on a rented
  // 80 GB pod is the mistake this reason exists to prevent.
  assert.match(
    launchBlockedReason({ clips: 12, training_verified: false, target_label: 'LTX-2' }),
    /no lora trainer is known/i)
  assert.equal(launchBlockedReason({ clips: 12, training_verified: true }), null)
})

test('a dataset with no clips on disk cannot be launched', () => {
  assert.match(launchBlockedReason({ clips: 0, training_verified: true }),
    /no clips on disk/i)
})

test('a dataset already on a pod is not launched a second time', () => {
  // Two pods on one dataset is money spent twice for one answer.
  const ds = { clips: 12, training_verified: true }
  assert.match(launchBlockedReason(ds, { status: 'training' }), /already on a pod/i)
  assert.equal(launchBlockedReason(ds, { status: 'done' }), null)
})

test('an unknown status is treated as NOT active', () => {
  // The alternative is a panel frozen for good by one string this build has not
  // heard of — and the server refuses a real second launch anyway.
  assert.equal(isActive('training'), true)
  assert.equal(isActive('done'), false)
  assert.equal(isActive('something_new'), false)
  assert.equal(isActive(null), false)
})

test('the run line states what is being billed', () => {
  assert.equal(
    runSummary({ run_id: 12, status: 'training', gpu: 'RTX 5090', price_per_hour: 0.45 }),
    'Run #12 · training · RTX 5090 · $0.45/h')
  assert.equal(runSummary(null), null)
})

test('retry is for a failed run, continue for a terminal one with weights', () => {
  assert.equal(canRetry({ status: 'error' }), true)
  assert.equal(canRetry({ status: 'done' }), false)
  const group = { steps: [{ step: 400, files: ['a.safetensors'] }] }
  assert.equal(canContinue({ status: 'done' }, group), true)
  assert.equal(canContinue({ status: 'training' }, group), false)
  assert.equal(canContinue({ status: 'done' }, { steps: [] }), false)
})

test('a Wan step says out loud that it is TWO files', () => {
  // The bug this label exists against raises nothing: offer one half of a MoE
  // pair and the user downloads a LoRA no loader can complete.
  assert.equal(
    stepLabel({ step: 400, final: false, files: ['a_high_noise.safetensors', 'a_low_noise.safetensors'] }),
    'Step 400 — 2 files (both experts)')
  // MiniMax H3 writes ONE file per step (ai-toolkit's MinimaxH3Model defines no
  // save_lora override), and must not be labelled as if a half were missing.
  assert.equal(stepLabel({ step: 250, final: false, files: ['a.safetensors'] }),
    'Step 250')
  assert.equal(stepLabel({ step: 500, final: true, files: ['a.safetensors'] }),
    'Final (step 500)')
})
