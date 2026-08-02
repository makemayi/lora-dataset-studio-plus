import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { comfyuiLauncherState, deriveSetupSteps } from './useSetupSteps.js';

function setupStep(id, caps, runtimeReadiness) {
  return deriveSetupSteps(caps, runtimeReadiness).find((item) => item.id === id);
}

test('integrated ComfyUI is initializing and never offers the portable launcher', () => {
  const step = setupStep('comfyui', {
    engines: {},
    comfyui: {
      reachable: false,
      skipped: true,
      dir_valid: true,
      portable_launcher_supported: true,
      portable_launcher_local_api: true,
    },
  }, {
    comfyui: { mode: 'integrated', state: 'starting', ready: false, poll: true },
  });

  assert.equal(step.status, 'initializing');
  assert.equal(step.managedMode, 'integrated');
  assert.equal(step.managedInitializing, true);
  assert.equal(step.skipped, false);
  assert.deepEqual(comfyuiLauncherState(step, true), {
    visible: false, enabled: false, reason: '',
  });

  // Runtime ownership must keep Start hidden even if lightweight readiness is
  // already ahead of the full capability refresh.
  assert.deepEqual(comfyuiLauncherState({ ...step, managedInitializing: false }, true), {
    visible: false, enabled: false, reason: '',
  });
});

test('external-host Docker ComfyUI stays manual and never offers portable Start', () => {
  const step = setupStep('comfyui', {
    engines: {},
    comfyui: {
      reachable: false,
      dir_valid: true,
      portable_launcher_supported: true,
      portable_launcher_local_api: true,
    },
  }, {
    comfyui: { mode: 'external-host', state: 'manual', ready: false, poll: false },
  });

  assert.equal(step.status, 'available');
  assert.equal(step.managedMode, 'external-host');
  assert.equal(step.managedInitializing, false);
  assert.deepEqual(comfyuiLauncherState(step, true), {
    visible: false, enabled: false, reason: '',
  });
});

test('external ComfyUI remains manual and can expose its safe portable launcher', () => {
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
  assert.equal(step.managedInitializing, false);
  assert.deepEqual(comfyuiLauncherState(step, true), {
    visible: true, enabled: true, reason: '',
  });
});

test('Ollama none, host and docker modes have distinct stopped states', () => {
  const caps = { ollama: { reachable: false, installed: false } };
  const none = setupStep('ollama', caps, {
    ollama: { mode: 'none', state: 'disabled', ready: false, poll: false },
  });
  const host = setupStep('ollama', caps, {
    ollama: { mode: 'host', state: 'unreachable', ready: false, poll: false },
  });
  const docker = setupStep('ollama', caps, {
    ollama: { mode: 'docker', state: 'starting', ready: false, poll: true },
  });

  assert.equal(none.status, 'skipped');
  assert.equal(none.disabled, true);
  assert.equal(none.managedInitializing, false);

  assert.equal(host.status, 'available');
  assert.equal(host.deploymentMode, 'host');
  assert.equal(host.managedInitializing, false);

  assert.equal(docker.status, 'initializing');
  assert.equal(docker.deploymentMode, 'docker');
  assert.equal(docker.managedInitializing, true);
});

test('Setup polls the lightweight endpoint without overlap and cleans up timers', () => {
  const source = fs.readFileSync(new URL('../pages/SetupPage.jsx', import.meta.url), 'utf8');

  assert.match(source, /apiFetch\('\/api\/setup\/runtime-readiness'/);
  assert.match(source, /background: true/);
  assert.match(source, /cache: 'no-store'/);
  assert.match(source, /setTimeout\(check, delay\)/);
  assert.match(source, /schedule\(3000\)/);
  assert.match(source, /clearTimeout\(timer\)/);
  assert.match(source, /controller\?\.abort\(\)/);
  assert.match(source, /capabilityRefreshPending = true/);
  assert.match(source, /refresh\(true, \{ background: true \}\)/);
  assert.match(source, /capabilityRefreshPending = !refreshed/);
  assert.match(source, /next\.ollama\?\.poll \|\| capabilityRefreshPending/);
});

test('capability refresh reports silent failure instead of stopping managed polling', () => {
  const source = fs.readFileSync(
    new URL('../context/CapabilitiesContext.jsx', import.meta.url), 'utf8',
  );

  assert.match(source, /refresh = useCallback\(async \(force = false, options = \{\}\)/);
  assert.match(source, /apiFetch\([\s\S]*options,[\s\S]*\)/);
  assert.match(source, /setCaps\(data\)\s*return data/);
  assert.match(source, /catch \{[\s\S]*return null/);
});

test('Setup explains every managed runtime state in the UI', () => {
  const source = fs.readFileSync(new URL('../pages/SetupPage.jsx', import.meta.url), 'utf8');

  assert.match(source, /Initializing ComfyUI…/);
  assert.match(source, /first startup can\s*\n?\s*take several minutes/);
  assert.match(source, /Starting Ollama…/);
  assert.match(source, /Ollama is optional and disabled/);
  assert.match(source, /start-docker\.bat --configure/);
  assert.doesNotMatch(source, /start-docker-gpu\.bat --configure/);
  assert.match(source, /start-docker-gpu\.bat/);
  assert.match(source, /Host Ollama is selected/);
  assert.match(source, /Windows Firewall/);
  assert.match(source, /No model is downloaded automatically/);
  assert.doesNotMatch(source, /LDS_OLLAMA_MODE/);
  assert.doesNotMatch(source, /docker-compose\.ollama-sidecar\.yml/);
  assert.match(source, /existing host ComfyUI, but its API is not reachable/);
  assert.match(source, /\/external-comfyui/);
  assert.match(source, /docs\/guide\/docker\.md/);
});
