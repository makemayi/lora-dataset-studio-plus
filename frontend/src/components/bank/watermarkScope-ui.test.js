import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { BANK_PASSES, passScopeRows } from './bankPasses.js';

const panel = fs.readFileSync(new URL('./BankWatermarkPanel.jsx', import.meta.url), 'utf8');
const workspace = fs.readFileSync(new URL('./BankWorkspace.jsx', import.meta.url), 'utf8');

test('level 1 delegates Find to the workspace launch dialog', () => {
  const findCardStart = panel.indexOf('<LevelCard index={1} title="Find them"');
  const cropCardStart = panel.indexOf('<LevelCard index={2}', findCardStart);
  assert.ok(findCardStart >= 0, 'the Level 1 Find card is missing');
  assert.ok(cropCardStart > findCardStart, 'the Level 1 card boundary is missing');

  const findCard = panel.slice(findCardStart, cropCardStart);
  assert.match(findCard, /onRun=\{onFind\}/,
    'Level 1 must call the onFind callback instead of launching a request itself');
  assert.doesNotMatch(findCard, /\brun\s*\(/,
    'Level 1 must not use the panel POST helper');
  assert.doesNotMatch(findCard, /\/api\/bank\/\$\{bankId\}\/watermark(?:[`'"?])/,
    'Level 1 must not post directly to the watermark endpoint');

  const panelCallStart = workspace.indexOf('<BankWatermarkPanel');
  const panelCallEnd = workspace.indexOf('/>', panelCallStart);
  assert.ok(panelCallStart >= 0 && panelCallEnd > panelCallStart,
    'the workspace watermark panel is missing');
  const panelCall = workspace.slice(panelCallStart, panelCallEnd);
  assert.match(panelCall, /onFind=\{\(\) => setPassOpen\('watermark'\)\}/,
    'the workspace must open the shared watermark PassDialog');
  assert.match(workspace, /\{passOpen && \(\s*<PassDialog passId=\{passOpen\}/,
    'passOpen must render the shared PassDialog');
});

test('the watermark pass offers selection, all five scopes, and rescan', () => {
  const spec = BANK_PASSES.watermark;
  assert.ok(spec, 'the watermark pass spec is missing');
  assert.equal(spec.selection, true);
  assert.equal(spec.scopes, true);
  assert.deepEqual(passScopeRows('watermark').map(({ id }) => id),
    ['', 'keep', 'pending', 'reject', 'all']);
  assert.deepEqual(spec.redo && { key: spec.redo.key, label: spec.redo.label }, {
    key: 'rescan',
    label: 'Also re-check images that were already scanned',
  });
});

test('the watermark dialog reads the detector capability exposed by the backend', () => {
  const dialogStart = workspace.indexOf('<PassDialog passId={passOpen}');
  const dialogEnd = workspace.indexOf('>', dialogStart);
  assert.ok(dialogStart >= 0 && dialogEnd > dialogStart, 'the shared PassDialog is missing');
  const dialogCall = workspace.slice(dialogStart, dialogEnd);
  assert.match(dialogCall, /detectorReady=\{!!caps\.watermark_detect\}/);
  assert.doesNotMatch(dialogCall, /watermark_detector/);
});
