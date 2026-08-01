/* JSX wiring contract for Docker lifecycle controls. The pure install-mode and
 * command data live in updateStatus.test.js; this file protects the places where
 * a future JSX refactor could accidentally put the unsafe buttons back. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = (url) => readFileSync(new URL(url, import.meta.url), 'utf8');
const maintenance = read('./MaintenanceSection.jsx');
const server = read('./ServerSection.jsx');
const app = read('../../App.jsx');
const instructions = read('../common/DockerUpdateInstructions.jsx');

test('settings and the global banner share one Docker update presentation', () => {
  assert.match(maintenance, /import DockerUpdateInstructions from ['"]\.\.\/common\/DockerUpdateInstructions['"]/);
  assert.match(app, /import DockerUpdateInstructions from ['"]\.\/components\/common\/DockerUpdateInstructions['"]/);
  assert.match(maintenance, /dockerMode && s\.update_available[\s\S]{0,500}<DockerUpdateInstructions \/>/,
    'the Settings update branch must replace the apply action with host commands');
  assert.match(app, /dockerMode \? \([\s\S]{0,120}<DockerUpdateInstructions \/>/,
    'the global banner must replace its apply action with the same host commands');
});

test('both apply callbacks refuse Docker mode even if stale UI invokes them', () => {
  assert.match(maintenance, /if \(mode === 'docker'\) return/);
  assert.match(app, /if \(installMode\(info\) === 'docker'\) return/);
});

test('the shared presentation renders the exact command contract and guide link', () => {
  assert.match(instructions, /DOCKER_UPDATE_COMMANDS\.map\(/);
  assert.match(instructions, /href=\{DOCKER_UPDATE_GUIDE_URL\}/);
  assert.match(instructions, /Docker GPU update guide/);
  assert.doesNotMatch(instructions, /Update &(?:amp;)? restart/);
});

test('a managed bind disables both port and host controls', () => {
  assert.match(server, /const bindManaged = runtime\.bind_managed === true/);
  assert.match(server, /const dirty = !bindManaged/,
    'a config/runtime difference must not offer an in-process restart for a managed bind');

  const port = server.match(/<input id="server-port"[\s\S]*?\/>/)?.[0] || '';
  assert.match(port, /disabled=\{bindManaged\}/);
  assert.match(port, /aria-describedby=\{bindManaged \? 'server-bind-managed-note'/);

  const host = server.match(/<button id="server-lan"[\s\S]*?>/)?.[0] || '';
  assert.match(host, /disabled=\{bindManaged\}/);
  assert.match(host, /aria-describedby=\{bindManaged \? 'server-bind-managed-note'/);
});

test('managed-bind guidance names the host variable and an explicit recreate', () => {
  assert.match(server, /LDS_HOST_PORT/);
  assert.match(server, /docker compose -f docker-compose\.gpu\.yml up -d --force-recreate/);
  assert.match(server, /Host and port are managed by Docker Compose/);
  assert.match(server, /\{!bindManaged && \([\s\S]{0,180}<ResetToDefault/,
    'the disabled port must not keep an apparently actionable reset control');
});
