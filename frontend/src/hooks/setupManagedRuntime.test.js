/* What survives of the managed-runtime tests after the Docker deployments were
   removed. The five cases that pinned "integrated"/"external-host" ComfyUI and
   the none/host/docker Ollama modes went with the feature; these two describe
   behaviour that is still the app's, and the third pins the readiness read that
   replaced the poll loop. */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { comfyuiLauncherState, deriveSetupSteps } from './useSetupSteps.js';

function setupStep(id, caps, runtimeReadiness) {
  return deriveSetupSteps(caps, runtimeReadiness).find((item) => item.id === id);
}

test('ComfyUI is the user\'s, and can expose its safe portable launcher', () => {
  const step = setupStep('comfyui', {
    engines: {},
    comfyui: {
      reachable: false,
      dir_valid: true,
      portable_launcher_supported: true,
      portable_launcher_local_api: true,
    },
  }, {
    comfyui: { mode: 'external', state: 'manual', ready: false, poll: false },
  });

  assert.equal(step.status, 'available');
  assert.equal(step.managedMode, 'external');
  assert.deepEqual(comfyuiLauncherState(step, true), {
    visible: true, enabled: true, reason: '',
  });
});

test('a stopped Ollama is reported from capabilities, not from a deployment mode', () => {
  const stopped = setupStep('ollama', {
    ollama: { reachable: false, installed: true, url: 'http://127.0.0.1:11434' },
  }, { ollama: { mode: 'local', state: 'unreachable', ready: false, poll: false } });
  const running = setupStep('ollama', {
    ollama: { reachable: true, installed: true, vision_model_ready: true },
  }, { ollama: { mode: 'local', state: 'ready', ready: true, poll: false } });

  assert.equal(stopped.status, 'available');
  assert.equal(stopped.reachable, false);
  assert.equal(stopped.installed, true);
  assert.equal(running.status, 'ready');
});

test('Setup reads the lightweight endpoint once and cleans up after itself', () => {
  const source = fs.readFileSync(new URL('../pages/SetupPage.jsx', import.meta.url), 'utf8');

  assert.match(source, /apiFetch\('\/api\/setup\/runtime-readiness'/);
  assert.match(source, /background: true/);
  assert.match(source, /cache: 'no-store'/);
  assert.match(source, /clearTimeout\(timer\)/);
  assert.match(source, /controller\?\.abort\(\)/);
  // Nothing bundled comes up on its own any more, so no state to wait for.
  assert.doesNotMatch(source, /docker/i);
});

test('capability refresh reports silent failure instead of stopping the caller', () => {
  const source = fs.readFileSync(
    new URL('../context/CapabilitiesContext.jsx', import.meta.url), 'utf8',
  );

  assert.match(source, /refresh = useCallback\(async \(force = false, options = \{\}\)/);
  assert.match(source, /apiFetch\([\s\S]*options,[\s\S]*\)/);
  assert.match(source, /setCaps\(data\)\s*return data/);
  assert.match(source, /catch \{[\s\S]*return null/);
});
