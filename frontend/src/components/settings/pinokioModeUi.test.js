/* JSX wiring contract for the Pinokio install mode — the sibling of
 * dockerModeUi.test.js. The pure mode/steps data lives in updateStatus.test.js;
 * this file guards the two places where a refactor could put back an
 * "Update & restart" button that would relaunch the server detached from the
 * launcher that is supposed to stop and start it. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = (url) => readFileSync(new URL(url, import.meta.url), 'utf8');
const maintenance = read('./MaintenanceSection.jsx');
const app = read('../../App.jsx');
const instructions = read('../common/PinokioUpdateInstructions.jsx');
const startScript = read('../../../../start.js');

test('settings and the global banner share one Pinokio update presentation', () => {
  assert.match(maintenance, /import PinokioUpdateInstructions from ['"]\.\.\/common\/PinokioUpdateInstructions['"]/);
  assert.match(app, /import PinokioUpdateInstructions from ['"]\.\/components\/common\/PinokioUpdateInstructions['"]/);
  assert.match(maintenance, /pinokioMode && s\.update_available[\s\S]{0,700}<PinokioUpdateInstructions \/>/,
    'the Settings update branch must replace the apply action with the launcher steps');
  assert.match(app, /pinokioMode \? \([\s\S]{0,120}<PinokioUpdateInstructions \/>/,
    'the global banner must replace its apply action with the same steps');
});

test('both apply callbacks refuse Pinokio mode even if stale UI invokes them', () => {
  assert.match(maintenance, /if \(mode === 'pinokio'\) return/);
  assert.match(app, /if \(installMode\(info\) === 'pinokio'\) return/);
});

test('the shared presentation renders the steps and never an apply action', () => {
  assert.match(instructions, /PINOKIO_UPDATE_STEPS\.map\(/);
  assert.match(instructions, /href=\{PINOKIO_UPDATE_GUIDE_URL\}/);
  assert.doesNotMatch(instructions, /Update &(?:amp;)? restart/);
});

test('the launcher is what puts the app in this mode', () => {
  // Without LDS_RUNTIME the backend cannot know it was launched by Pinokio, and
  // the card would offer the button again.
  assert.match(startScript, /LDS_RUNTIME:\s*"pinokio"/);
});
