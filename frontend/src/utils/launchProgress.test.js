import test from 'node:test';
import assert from 'node:assert/strict';
import {
  formatElapsed,
  launchButtonLabel,
  launchProgressView,
  podBootFailureView,
  stopButtonLabel,
  uploadStallFailureView,
} from './launchProgress.js';

// Verbatim shape of what the backend forwards while a pod boots, taken from
// launch_view() with the production config that killed run #134
// (ready_timeout_minutes: 25).
const BOOTING = {
  active_step: 'boot',
  detail: 'Waiting for the pod to boot — pod up — waiting for the UI to answer',
  elapsed_seconds: 8 * 60 + 12,
  steps: [
    { key: 'staging', label: 'Preparing the dataset', state: 'done' },
    { key: 'offer', label: 'Searching for a GPU offer', state: 'done' },
    { key: 'boot', label: 'Renting the machine and booting the pod', state: 'active' },
    { key: 'upload', label: 'Uploading the dataset', state: 'pending' },
    { key: 'start', label: 'Starting the training job', state: 'pending' },
  ],
  boot_idle_limit_seconds: 25 * 60,
  boot_budget_seconds: 90 * 60,
};

test('a booting pod reads as a step plus a clock, not as a frozen sentence', () => {
  const v = launchProgressView(BOOTING);
  assert.equal(v.headline, 'Renting the machine and booting the pod — 8m elapsed');
  assert.equal(v.activeKey, 'boot');
  assert.equal(v.detail, 'Waiting for the pod to boot — pod up — waiting for the UI to answer');
});

test('the boot step announces its deadline — the answer run #134 never gave', () => {
  const v = launchProgressView(BOOTING);
  assert.match(v.note, /25 min/);
  assert.match(v.note, /released automatically/);
});

// Same payload once the pod is up and the dataset is going across — the phase
// run #138 spent 2 h 07 in with nothing on screen but its label.
const UPLOADING = {
  ...BOOTING,
  active_step: 'upload',
  detail: 'Uploading the dataset — 912/12422 files, 2.1 of 24.0 GB',
  steps: BOOTING.steps.map((s) => ({
    ...s,
    state: s.key === 'upload' ? 'active' : s.key === 'boot' ? 'done' : s.state,
  })),
  upload_stall_limit_seconds: 25 * 60,
};

test('the upload step announces a deadline on IDLE BYTES, not on its duration', () => {
  const v = launchProgressView(UPLOADING);
  assert.equal(v.activeKey, 'upload');
  // The distinction is the whole point: a 24 GB upload is allowed to run for
  // hours, and a note that reads like a countdown would be a lie that makes
  // people cancel healthy runs.
  assert.match(v.note, /no time limit/i);
  assert.match(v.note, /NO data reaches the machine for 25 min/);
  assert.match(v.note, /released automatically/);
  // ... and the progress the backend now sends is what the step shows.
  assert.match(v.detail, /912\/12422 files/);
});

test('a step with no deadline in the payload does not get an invented one', () => {
  assert.equal(launchProgressView({ ...UPLOADING, upload_stall_limit_seconds: 0 }).note, null);
  const staging = {
    ...BOOTING,
    steps: BOOTING.steps.map((s) => ({
      ...s, state: s.key === 'staging' ? 'active' : 'pending',
    })),
  };
  assert.equal(launchProgressView(staging).note, null);
});

test('a boot deadline the install disabled is not invented', () => {
  assert.equal(launchProgressView({ ...BOOTING, boot_idle_limit_seconds: 0 }).note, null);
});

test('nothing to report degrades to null — the caller keeps its phase sentence', () => {
  assert.equal(launchProgressView(null), null);
  assert.equal(launchProgressView(undefined), null);
  assert.equal(launchProgressView({}), null);
  // Every step done and none active = the job is running; the checklist goes.
  assert.equal(launchProgressView({
    ...BOOTING, steps: BOOTING.steps.map((s) => ({ ...s, state: 'done' })),
  }), null);
});

test('an empty phase_detail is dropped rather than printed as a blank line', () => {
  assert.equal(launchProgressView({ ...BOOTING, detail: '   ' }).detail, null);
});

test('the clock stays readable from one second to two hours', () => {
  assert.equal(formatElapsed(0), '0s');
  assert.equal(formatElapsed(42), '42s');
  assert.equal(formatElapsed(59.6), '1m');
  assert.equal(formatElapsed(25 * 60), '25m');
  assert.equal(formatElapsed(3 * 3600 + 4 * 60), '3h 04m');
  assert.equal(formatElapsed(-5), '0s');
  assert.equal(formatElapsed(undefined), '0s');
});

// The real run-134 row: status 'error', pod destroyed, error verbatim.
const RUN_134 = {
  status: 'error',
  error: 'pod did not become ready in time — no boot progress for 25 min: '
    + 'pod up — waiting for the UI to answer',
};

test('the boot timeout is explained, including what became of the machine', () => {
  const v = podBootFailureView(RUN_134);
  assert.equal(v.title, 'The rented machine never started');
  assert.match(v.message, /released/);
  assert.match(v.message, /no longer billing/);
  assert.match(v.message, /again picks a different one/);
});

test('only a boot timeout gets the boot-timeout explanation', () => {
  assert.equal(podBootFailureView({ status: 'error', error: 'CUDA out of memory' }), null);
  // A kept pod is still billing — the "released" sentence would be false.
  assert.equal(podBootFailureView({ ...RUN_134, status: 'error_pod_kept' }), null);
  assert.equal(podBootFailureView({ status: 'training' }), null);
  assert.equal(podBootFailureView(null), null);
});

// Run #138 as the supervisor now closes it.
const RUN_138 = {
  status: 'error',
  error: 'upload stall watchdog',
  phase_detail: 'Dataset upload stalled — nothing reached the pod for 25 min; '
    + 'pod terminated by the supervisor',
};

test('a stalled upload is explained as a transfer, not as a crash', () => {
  const v = uploadStallFailureView(RUN_138);
  assert.equal(v.title, 'The dataset never reached the rented machine');
  assert.match(v.message, /no longer billing/);
  assert.match(v.message, /before any training happened/);
  assert.match(v.message, /connection/);
  // The one thing users would otherwise conclude on their own and act on: it
  // is not "my dataset is too big".
  assert.match(v.message, /not the problem on its own/);
});

test('only a stalled upload gets the stalled-upload explanation', () => {
  assert.equal(uploadStallFailureView({ status: 'error', error: 'CUDA out of memory' }), null);
  // A kept pod is still billing — the "released" sentence would be false.
  assert.equal(uploadStallFailureView({ ...RUN_138, status: 'error_pod_kept' }), null);
  assert.equal(uploadStallFailureView({ ...RUN_138, status: 'uploading' }), null);
  assert.equal(uploadStallFailureView(null), null);
});

test('the two launch teardowns never both answer for the same run', () => {
  assert.equal(podBootFailureView(RUN_138), null);
  assert.equal(uploadStallFailureView(RUN_134), null);
});

test('ending a launch is not worded like abandoning a trained run', () => {
  assert.equal(stopButtonLabel('preparing'), 'Cancel launch');
  assert.equal(stopButtonLabel('provisioning'), 'Cancel launch');
  assert.equal(stopButtonLabel('uploading'), 'Cancel launch');
  assert.equal(stopButtonLabel('training'), 'Stop run');
  assert.equal(stopButtonLabel('downloading'), 'Stop run');
  assert.equal(stopButtonLabel(undefined), 'Stop run');
});

test('the dialog button counts, so a working request is not read as a hang', () => {
  assert.equal(launchButtonLabel({ launching: false, elapsedSeconds: 0, fullMode: false }),
    '☁️ Rent & train');
  assert.equal(launchButtonLabel({ launching: false, elapsedSeconds: 0, fullMode: true }),
    '☁️ Rent GPU & train full model');
  // The first seconds still feel instant — no counter flicker for a fast POST.
  assert.equal(launchButtonLabel({ launching: true, elapsedSeconds: 1 }), 'Launching…');
  assert.equal(launchButtonLabel({ launching: true, elapsedSeconds: 34 }), 'Launching… 34s');
  assert.equal(launchButtonLabel({ launching: true, elapsedSeconds: 95 }), 'Launching… 1m');
});
