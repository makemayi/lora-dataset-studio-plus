import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

/* `node --test` cannot parse JSX, so the modal itself is pinned by READING it.
   Crude on purpose: the helpers next door are unit-tested to death, and none of
   that is worth anything if the component quietly stops calling them. This file
   exists because that has already happened once in this modal — an invisible fix
   (the opaque panel token) died in a rewrite with every test still green. */

const here = dirname(fileURLToPath(import.meta.url));
const modal = readFileSync(join(here, 'ReferenceEditModal.jsx'), 'utf8');
const workspace = readFileSync(join(here, 'DatasetWorkspace.jsx'), 'utf8');

const datasetHook = readFileSync(join(here, '../../hooks/useDataset.js'), 'utf8');
test('the engine list is DERIVED, never spelled out in the component', () => {
  assert.ok(modal.includes('editEngineOptions('), 'the modal must build its list from the helper');
  // A literal engine label in the JSX is the hardcoded list coming back.
  for (const label of ['Krea 2 Edit', 'Nano Banana Pro', 'OpenRouter']) {
    assert.ok(!modal.includes(`>${label}<`), `${label} is spelled out in the JSX`);
  }
});

test('the cost and Keep lines come from the engine-aware helpers', () => {
  assert.ok(modal.includes('editCostNote('), 'cost line must be per-engine');
  assert.ok(modal.includes('editKeepNote('), 'Keep line must be per-engine');
  // The old unconditional sentence must be gone: it is a lie on a local engine.
  assert.ok(!modal.includes('Each edit is a paid API call'),
    'the hardcoded "paid API call" line is back in the JSX');
});

test('the reference picker is gated, and the reference note is rendered', () => {
  assert.ok(modal.includes('acceptsExtraEditRefsForBatch(engines)'),
    'the picker must be gated across the selected engine batch');
  assert.ok(modal.includes('editRefNote(') && modal.includes('localRefNotes.map('),
    'the modal must SHOW what each selected local engine does with extra references');
});

test('an unavailable engine renders its reason, not just a disabled button', () => {
  assert.ok(modal.includes('selectedBlocked.map('),
    'the blocked reason must reach the render');
  assert.ok(modal.includes('editBatchBlockedReason(prompt, engines, options)'),
    'every selected blocked engine must gate the Generate button');
});

test('the workspace hands the modal the SAME capabilities the generation panel reads', () => {
  assert.ok(workspace.includes('localEngineUnavailableReason'),
    'the reason must be the shared one, not a second inline copy');
  assert.ok(workspace.includes('comfyuiConfigured={hasComfyui(caps)}'),
    'an install with no ComfyUI must be told apart from one with a fixable gap');
  assert.ok(workspace.includes('engineAvailable={caps.engines'),
    'live readiness must reach the modal');
});

test('the engine pills wrap instead of overflowing a 400px screen', () => {
  // Five pills on a narrow phone. `flex-wrap` is the whole mechanism; losing it
  // pushes the modal wider than the viewport, which is not visible on a desktop
  // rewrite and is exactly how this regresses.
  const pills = modal.slice(modal.indexOf('{options.map('));
  assert.ok(modal.includes('flex-wrap min-w-0'),
    'the engine row must wrap and be allowed to shrink');
  assert.ok(pills.length > 0);
});

test('engine pills are an accessible multi-select toggle group', () => {
  assert.match(modal, /role="group" aria-label="Edit engines"/);
  assert.match(modal, /aria-pressed={engines\.includes\(o\.engine\)\}/);
  assert.match(modal, /onClick={\(\) => toggleEngine\(o\.engine\)\}/);
  assert.match(modal, /setEngines\(\(current\) => current\.includes\(engine\)/,
    'clicking a pill must toggle membership rather than replace one engine');
  assert.match(modal, /engines\.length === 0 && \([\s\S]*?role="alert"[\s\S]*?Select at least one engine\./,
    'an empty toggle set must expose its validation accessibly');
  assert.match(modal, /Generate \$\{engines\.length \|\| 0\} edit/,
    'the submit label must name the number of engine calls');
});

test('the modal traps focus and restores it to its opener', () => {
  assert.match(modal, /import \{ useFocusTrap \} from '\.\.\/\.\.\/hooks\/useFocusTrap';/);
  assert.match(modal, /const dialogRef = useRef\(null\);\s*useFocusTrap\(dialogRef\);/);
  assert.match(modal,
    /<div ref=\{dialogRef\} role="dialog" aria-modal="true" aria-label="Edit reference photo"/,
    'the actual modal root must own the focus trap');
});

test('Retry replays the exact session request, including transient reference files', () => {
  assert.match(modal, /onRetry = null, canRetry = false/,
    'the modal needs an explicit retry contract');
  assert.match(modal, /const retryEdit = async \(\) =>/,
    'the modal must invoke its retry callback instead of rebuilding the request');
  assert.match(datasetHook, /const retryRequest = \{ prompt, engines, files: Array\.from\(files \|\| \[\]\) \}/,
    'the hook must snapshot prompt, engines and File objects after a successful queue');
  assert.match(datasetHook,
    /referenceEditRetryRef\.current\.set\(String\(currentId\), confirmedRetry\)/,
    'only a request confirmed against the displayed opaque batch may remain retryable');
  assert.match(datasetHook,
    /retryRequest\.prompt, retryRequest\.engines, retryRequest\.files,[\s\S]*?retryRequest\.batchId/,
    'retry must replay the saved prompt, engine list, files and exact source batch');
  assert.match(workspace, /onRetry=\{ds\.retryReferenceEdit\}/,
    'the workspace must wire the hook retry callback to the modal');

  const retryHandler = modal.indexOf('const retryEdit = async');
  const keepHandler = modal.indexOf('const keep = async');
  assert.ok(retryHandler > modal.indexOf('const runEdit = async') && retryHandler < keepHandler,
    'retry must be a standalone handler, never nested inside Keep');

  const keepStart = datasetHook.indexOf('const keepEditedReference');
  const discardStart = datasetHook.indexOf('const discardEditedReference');
  const clear = 'referenceEditRetryRef.current.delete(String(currentId))';
  const keepClear = datasetHook.indexOf(clear, keepStart);
  const discardClear = datasetHook.indexOf(clear, discardStart);
  assert.ok(keepClear > keepStart && keepClear < discardStart,
    'Keep clears only the saved retry after a successful promotion');
  assert.ok(discardClear > discardStart,
    'Discard clears only the saved retry after a successful discard');

  const readyStart = modal.indexOf("phase === 'ready'");
  const readyEnd = modal.indexOf('\n        ) : (', readyStart);
  assert.ok(readyStart > 0 && readyEnd > readyStart, 'the ready branch must be isolated');
  const ready = modal.slice(readyStart, readyEnd);
  assert.match(ready, /ENGINE_LABELS\[candidate\.engine\] \|\| candidate\.engine/,
    'every result must use its canonical engine label');
  assert.match(ready, /onClick=\{\(\) => keep\(candidate\.engine\)\}/,
    'each successful candidate must own its Keep action');
  assert.match(ready, /Try another prompt\s*<\/button>\s*<button type="button" onClick=\{retryEdit\}/,
    'Retry must be its own action button, not nested inside another action');
});

test('a queued edit never leaves the modal bridge locked or retryable when refresh is unavailable', () => {
  assert.match(datasetHook,
    /const \[, bumpReferenceEditRetryRevision\] = useState\(0\)/,
    'the transient File snapshot must have reactive availability');
  assert.match(datasetHook, /const confirmedRetry = retryRequestForReferenceEdit\([\s\S]*?batchId: d\.batch_id[\s\S]*?refreshed\?\.data\?\.reference_edit/,
    'the 202 batch id must match the batch returned by the subsequent refresh');
  assert.match(datasetHook,
    /if \(refreshed\?\.status !== 'applied' \|\| !confirmedRetry\) \{[\s\S]*?referenceEditRetryRef\.current\.delete\(String\(currentId\)\);[\s\S]*?toast\.warning\('Edit queued, but its status could not be refreshed\.[^']*'\);[\s\S]*?return false;/,
    'a stale, replaced or unavailable refresh must disable Retry before releasing the spinner');
});

test('the completed or failed candidates come from the server per-engine registry', () => {
  assert.match(modal, /const candidates = referenceEditCandidates\(referenceEdit\)/,
    'the server candidate registry must drive the comparison');
  assert.match(modal, /candidate\.status === 'ready'/);
  assert.match(modal, /candidate\.status === 'failed'/);
  assert.match(modal, /candidate\.error \|\| 'This engine did not produce a candidate\.'/);
  assert.match(modal, /onKeep\(engine, referenceEdit\?\.batch_id \|\| null\)/,
    'Keep must forward the batch that rendered the candidate so a stale tab cannot promote a newer batch');
  assert.match(datasetHook,
    /const keepEditedReference = useCallback\(async \(engine = null, batchId = null\) => \{/);
  assert.match(datasetHook, /if \(engine\) payload\.engine = engine;/);
  assert.match(datasetHook, /if \(batchId\) payload\.batch_id = batchId;/,
    'Keep must identify both the selected engine and opaque server batch; no candidate filename comes from the client');
});

test('the hook submits and retries the exact selected engine list', () => {
  assert.match(datasetHook, /const retryRequest = \{ prompt, engines, files: Array\.from\(files \|\| \[\]\) \}/);
  assert.match(datasetHook,
    /retryRequest\.engines\.forEach\(\(engine\) => fd\.append\('engines', engine\)\)/);
  assert.match(datasetHook,
    /if \(retryRequest\.engines\.length === 1\) fd\.append\('engine', retryRequest\.engines\[0\]\)/,
    'single-engine requests keep the legacy field');
  assert.match(datasetHook, /if \(retryBatchId\) fd\.append\('retry_batch_id', retryBatchId\)/,
    'a Retry must let the server atomically reject a superseded source batch');
  assert.match(datasetHook,
    /const canRetryReferenceEdit = Boolean\(retryRequestForReferenceEdit\([\s\S]*?data\?\.reference_edit/,
    'Retry availability must follow the currently displayed opaque batch');
});

test('mixed batches keep the picker but explain its scope', () => {
  assert.match(modal, /acceptsExtraEditRefsForBatch\(engines\)/);
  // The split is no longer "API vs local" — Krea is local and reads these bytes.
  // The banner has to turn on what actually differs: an engine in the selection
  // that will NOT read them, whatever lane it sits in.
  assert.match(modal, /selectedLocalEngines\.some\(\(e\) => !acceptsExtraEditRefs\(e\)\)/);
  assert.match(modal, /Images added below go to the engines that read them/);
  // And the cap follows the selection, or deselecting the generous engine would
  // leave staged images the server is about to refuse.
  assert.match(modal, /maxEditRefsForBatch\(engines\)/);
  assert.match(modal, /editRefs\.length < maxRefs/);
});
