import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { comfyuiLauncherState, deriveSetupSteps } from './useSetupSteps.js';

function comfyStep(comfyui) {
  return deriveSetupSteps({ engines: {}, comfyui }).find((step) => step.id === 'comfyui');
}

test('the Start ComfyUI control is visible only for a valid stopped install', () => {
  const stopped = comfyuiLauncherState(comfyStep({
    dir_valid: true, reachable: false,
    portable_launcher_supported: true, portable_launcher_local_api: true,
  }), true);
  assert.deepEqual(stopped, { visible: true, enabled: true, reason: '' });

  const running = comfyuiLauncherState(comfyStep({
    dir_valid: true, reachable: true,
    portable_launcher_supported: true, portable_launcher_local_api: true,
  }), true);
  assert.equal(running.visible, false);

  const slow = comfyuiLauncherState(comfyStep({
    dir_valid: true, reachable: false, status: 'slow',
    portable_launcher_supported: true, portable_launcher_local_api: true,
  }), true);
  assert.equal(slow.visible, false);

  const invalid = comfyuiLauncherState(comfyStep({ reachable: false }), true);
  assert.equal(invalid.visible, false);
});

test('a live-valid unsaved folder reveals a disabled Start ComfyUI control', () => {
  const unpersisted = comfyuiLauncherState(comfyStep({
    dir_valid: false, reachable: false,
    portable_launcher_supported: false, portable_launcher_local_api: false,
  }), false, true);
  assert.deepEqual(unpersisted, {
    visible: true,
    enabled: false,
    reason: 'Save & re-check the ComfyUI settings before starting it from LDS.',
  });

  const liveValidButNotRechecked = comfyuiLauncherState(comfyStep({
    dir_valid: false, reachable: false,
    portable_launcher_supported: true, portable_launcher_local_api: true,
  }), true, true);
  assert.equal(liveValidButNotRechecked.visible, false);

  const liveValidButRunning = comfyuiLauncherState(comfyStep({
    dir_valid: false, reachable: true,
    portable_launcher_supported: true, portable_launcher_local_api: true,
  }), false, true);
  assert.equal(liveValidButRunning.visible, false);

  const liveValidButSlow = comfyuiLauncherState(comfyStep({
    dir_valid: false, reachable: false, status: 'slow',
    portable_launcher_supported: true, portable_launcher_local_api: true,
  }), false, true);
  assert.equal(liveValidButSlow.visible, false);
});

test('the Start ComfyUI control explains each disabled safety condition', () => {
  const portableStopped = comfyStep({
    dir_valid: true, reachable: false,
    portable_launcher_supported: true, portable_launcher_local_api: true,
  });
  const unsaved = comfyuiLauncherState(portableStopped, false);
  assert.equal(unsaved.visible, true);
  assert.equal(unsaved.enabled, false);
  assert.match(unsaved.reason, /Save & re-check/);

  const unsupported = comfyuiLauncherState(comfyStep({
    dir_valid: true, reachable: false,
    portable_launcher_supported: false, portable_launcher_local_api: true,
  }), true);
  assert.equal(unsupported.enabled, false);
  assert.match(unsupported.reason, /NVIDIA portable/);

  const remote = comfyuiLauncherState(comfyStep({
    dir_valid: true, reachable: false,
    portable_launcher_supported: true, portable_launcher_local_api: false,
  }), true);
  assert.equal(remote.enabled, false);
  assert.match(remote.reason, /port 8188/);
});

test('Setup uses a body-free CSRF-protected POST for the fixed launcher endpoint', () => {
  const source = fs.readFileSync(new URL('../pages/SetupPage.jsx', import.meta.url), 'utf8');
  assert.match(source, /const startComfyui = async \(\) =>/);
  assert.match(source, /apiFetch\('\/api\/setup\/comfyui\/start', \{\s*method: 'POST',\s*headers: \{ 'X-CSRFToken': getCsrfToken\(\) \},/);
  assert.doesNotMatch(source, /postJson\('\/api\/setup\/comfyui\/start/);
  assert.match(source, /const liveDirValid = !!liveCheck && liveCheck\.status === 'valid'/);
  assert.match(source, /comfyuiLauncherState\(step, configPersisted, liveDirValid\)/);
  assert.match(source, /disabled=\{startingComfyui \|\| !comfyLauncher\.enabled\}/);
  assert.match(source, /catch \(e\) \{\s*toast\.error\(e\.message \|\|/);
  assert.match(source, /connectionStatus === 'slow'/);
});
