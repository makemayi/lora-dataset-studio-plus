/* Wiring contract for the ✨ Score interpreter picker.
 *
 * The logic is unit-tested in scoringPython.test.js; what a rewrite of the
 * workspace silently loses is the WIRING — the entry point, the refresh after a
 * change, the read-only promise in the dialog. Those are one-line facts with no
 * runtime assertion anywhere, exactly the kind that dies unnoticed.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const ws = fs.readFileSync(new URL('./BankWorkspace.jsx', import.meta.url), 'utf8');
const dialog = fs.readFileSync(new URL('./ScoringPythonDialog.jsx', import.meta.url), 'utf8');
const device = fs.readFileSync(new URL('./bankScoreDevice.js', import.meta.url), 'utf8');
const logic = fs.readFileSync(new URL('./scoringPython.js', import.meta.url), 'utf8');

test('the picker is reachable from BOTH dead ends: a CPU pass, and no extra at all', () => {
  assert.match(ws, /scoreNote\?\.tone === 'warn' \|\| !caps\.bank_scoring/);
  // …but never before the capabilities have loaded: EMPTY_CAPS has no
  // bank_scoring, so an ungated check flashes "Score needs its own packages" on
  // every open of a bank that is perfectly set up.
  assert.match(ws, /!capsLoading && \(scoreNote\?\.tone === 'warn'/);
  assert.match(ws, /const \{ caps, loading: capsLoading, refresh: refreshCaps \}/);
  assert.match(ws, /openerLabel\(scoreGpuPresent\)/);
  assert.match(ws, /<ScoringPythonDialog/);
  assert.match(ws, /setPythonPickerFor\('scoring'\)/);
});

test('ONE dialog serves both features — the semantic index gets the same picker', () => {
  // The detector, the read-only rule and the refusal are identical; only the
  // dependency list and the config key differ, and both live on the server. A
  // twin dialog would be a second place to fix every future wording bug.
  const semantic = fs.readFileSync(
    new URL('./BankSemanticEngine.jsx', import.meta.url), 'utf8');
  assert.match(ws, /profile=\{PICKER_PROFILES\[pythonPickerFor\]\}/);
  assert.match(ws, /onPickPython=\{\(\) => setPythonPickerFor\('semantic'\)\}/);
  assert.match(semantic, /offersSemanticGpuPython\(state, gpuPresent\)/);
  assert.match(semantic, /openerLabel\(gpuPresent\)/);
  // The dialog reads its endpoint from the profile — never a hardcoded route,
  // which is how the semantic picker would silently write Score's config key.
  assert.match(dialog, /apiFetch\(`\$\{picker\.endpoint\}/);
  assert.match(dialog, /postJson\(picker\.endpoint/);
  assert.doesNotMatch(dialog, /'\/api\/scoring-python'/);
  // Borrowing an interpreter for the index must never read as "and Setup will
  // install into it".
  assert.match(logic, /Setup ▸ Quality tools keeps installing into the app’s own environment/);
});

test('a machine with no NVIDIA card is never sold a CUDA fix', () => {
  // scoreDeviceNote only returns tone 'warn' when gpu_present, so the amber
  // "your GPU is idle" branch cannot fire on a card-less machine…
  assert.match(device, /if \(!info\.gpu_present\)/);
  assert.match(device, /tone: 'info'/);
  // …and where the picker IS still offered (it saves a second install), both
  // the button label and the dialog copy drop every mention of a GPU.
  assert.match(ws, /const scoreGpuPresent = /);
  assert.match(ws, /openerLabel\(scoreGpuPresent\)/);
  assert.match(dialog, /dialogCopy\(nvidia, picker\)/);
  assert.match(dialog, /result\.nvidia_present !== false/);
});

test('a hand-typed path is a first-class route, not a fallback', () => {
  // Most installs out there have neither ai-toolkit nor ComfyUI where we look.
  // The field must be present, checkable on its own, and never gated on the
  // automatic list having found something.
  assert.match(dialog, /Enter the path to a Python interpreter or its folder/);
  // An answer is ALWAYS produced, including for a path that lands on a row
  // that was already there — silence read as a broken button.
  assert.match(dialog, /enteredNote\(result\)/);
  assert.match(dialog, /load\(\{ force: true, path: typed\.trim\(\) \}\)/);
  assert.doesNotMatch(dialog, /rows\.length\s*&&[\s\S]{0,200}scoring-python-path/);
});

test('changing the interpreter re-probes the capabilities, not just the payload', () => {
  // bank_scoring / score_device are both server-side probes with their own
  // caches: refreshing one and not the other leaves the panel contradicting
  // itself (Score enabled, still warning about the CPU).
  assert.match(ws, /refreshCaps\(true\)/);
  assert.match(ws, /refresh: refreshCaps \} = useCapabilities\(\)/);
});

test('the dialog promises — and shows — that nothing is installed for the user', () => {
  assert.match(logic, /never changed/);
  assert.match(logic, /already carries them/);
  assert.match(dialog, /will not install into an environment we did not create/);
  assert.match(dialog, /install_command/);
});

test('the dialog can re-probe after a manual install, and can be undone', () => {
  assert.match(dialog, /force: true/);
  assert.match(dialog, /Check again/);
  assert.match(dialog, /Back to the app default/);
  assert.match(dialog, /choose\(''\)/);
});

test('the CPU warning offers the reuse route before the 2.5 GB download', () => {
  const warn = device.slice(device.indexOf("tone: 'warn'"));
  assert.ok(warn.indexOf('already has a working CUDA PyTorch') < warn.indexOf('CUDA_TORCH_DOWNLOAD'),
    'borrowing an existing interpreter is named first — the download is the fallback');
});
