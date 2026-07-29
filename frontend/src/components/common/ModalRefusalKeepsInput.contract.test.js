/* A refusal must never throw away what the user just typed — in ANY modal.
 *
 * ContinueDialog was fixed first (ContinueDialogRefusal.contract.test.js); the
 * same close-before-posting shape was then found in four more places. All of
 * them date from the same era: the toast container shipped at z-[100], UNDER
 * every modal, so an error raised over an open dialog was invisible and closing
 * first was the only way to be heard. Toast.jsx is z-[10000] now
 * (Toast.contract.test.js keeps it there) and the workaround outlived its reason.
 *
 * What each one destroyed:
 *   CaptionEditorDialog  the long AND short caption just typed — the most used
 *                        screen in the app, and the worst case;
 *   PromptEditPopover    a hand-rewritten generation prompt;
 *   LaunchAllDialog      the seven pass checkboxes and the reject flags, reset
 *                        to defaults;
 *   FolderBrowserModal   the chosen path and the position in the tree.
 *
 * Two mechanical causes, both checked here: the close call sitting BEFORE the
 * await, and — worse — a handler prop invoked without `await`, which cannot
 * even know the call was refused.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const captionDialog = read('../dataset/CaptionEditorDialog.jsx');
const promptPopover = read('../dataset/PromptEditPopover.jsx');
const launchDialog = read('../bank/LaunchAllDialog.jsx');
const folderPicker = read('./FolderPicker.jsx');
const gridItem = read('../dataset/DatasetGridItem.jsx');
const bank = read('../bank/BankWorkspace.jsx');
const dsWorkspace = read('../dataset/DatasetWorkspace.jsx');
const useDataset = read('../../hooks/useDataset.js');

/* Comments are stripped before any ordering check: the comment explaining why a
   dialog no longer closes first is exactly where the old call gets written down
   in prose (the Toast contract test learned this the hard way). */
const code = (t) => t.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/(^|[^:])\/\/[^\n]*/g, '$1');

function slice(src, start, end) {
  const s = code(src);
  const a = s.indexOf(start);
  assert.ok(a >= 0, `not found: ${start}`);
  const b = s.indexOf(end, a + start.length);
  assert.ok(b > a, `end not found: ${end}`);
  return s.slice(a, b);
}

function assertPostsBeforeClosing(body, closeCall, who) {
  const firstAwait = body.indexOf('await ');
  const closed = body.indexOf(closeCall);
  assert.ok(firstAwait > 0, `${who}: nothing is awaited in the submit handler?`);
  assert.ok(closed > firstAwait,
    `${who}: still closes (${closeCall}) BEFORE the request is awaited — a refusal `
    + 'would again discard what the user typed.');
}

/* Every one of the four routes its answer through the SAME pure rule. Writing a
   fifth private notion of "what is a refusal" is how the hosts drifted apart the
   first time. */
for (const [who, src] of [
  ['CaptionEditorDialog', captionDialog],
  ['PromptEditPopover', promptPopover],
  ['LaunchAllDialog', launchDialog],
  ['FolderBrowserModal', folderPicker],
]) {
  test(`${who} classifies its answer with the shared rule`, () => {
    assert.match(src, /attemptModalSubmit/,
      `${who} must decide close/keep-open with attemptModalSubmit, not a private rule`);
  });

  test(`${who} shows the refusal INSIDE itself, and only success closes it`, () => {
    // an assertive live region, so it is announced and not merely drawn
    assert.match(src, /role="alert"[\s\S]{0,700}\{error\}/,
      `${who} must render the refusal in a role="alert" region inside the modal`);
    // 400 px: the card is a flex column with a max height, so a flex child is
    // free to be SQUASHED to a sliver of clipped text; and a long backend
    // sentence must scroll in its own box instead of pushing the submit button
    // off the screen. Both were measured on ContinueDialog before this rule.
    assert.match(src, /shrink-0[^"]*rounded[^"]*border-red-500\/40/,
      `${who}: the error box must be shrink-0 (measured: squashed to a 20-px sliver otherwise)`);
    assert.match(src, /max-h-\d+ overflow-y-auto/,
      `${who}: a long refusal must scroll inside its own box`);
  });

  test(`${who} cannot be dismissed out from under a request in flight`, () => {
    assert.match(code(src), /const dismiss = \(\) => \{ if \(!busy\) /,
      `${who} must funnel every way out through one busy-guarded dismiss()`);
    // …and busy is ALWAYS released, so a refused submit never leaves the modal
    // frozen with a disabled button and no way out.
    assert.match(code(src), /finally \{ setBusy\(false\) \}|finally \{\s*setBusy\(false\);?\s*\}/s,
      `${who}: setBusy(false) must be in a finally — otherwise a throw freezes the modal`);
  });
}

test('Escape and the backdrop still close a modal that is NOT posting', () => {
  // The fix is about the SERVER refusing, never about trapping the user. Every
  // dialog keeps its voluntary exits; they only stop working mid-request.
  for (const [who, src] of [
    ['CaptionEditorDialog', captionDialog],
    ['PromptEditPopover', promptPopover],
    ['LaunchAllDialog', launchDialog],
    ['FolderBrowserModal', folderPicker],
  ]) {
    assert.match(code(src), /'Escape'[\s\S]{0,80}dismiss\(\)/,
      `${who} must still close on Escape (through dismiss)`);
    assert.match(code(src), /currentTarget\) dismiss\(\)/,
      `${who} must still close on a backdrop click (through dismiss)`);
  }
});

test('the caption editor posts with the dialog open — the worst case of all', () => {
  const body = slice(captionDialog, 'const save = async ()', 'return createPortal');
  assertPostsBeforeClosing(body, 'onClose()', 'CaptionEditorDialog');
  // The tile is the host that actually calls the API, and it must AWAIT it:
  // an unawaited handler can never know it was refused.
  assert.match(code(gridItem), /onSave=\{async \(nextCaption, nextShort\) => \{[\s\S]{0,600}await onCaption\(/,
    'DatasetGridItem must await onCaption and hand its answer back to the dialog');
  // …and the hook must have an answer to give.
  assert.match(useDataset, /const setCaption = useCallback\(async \(imageId, captionText, shortText, \{ silent/,
    'useDataset.setCaption must report {ok,error} and be able to stay quiet');
});

test('the prompt bubble keeps a rewritten prompt when the server refuses', () => {
  const body = slice(promptPopover, 'const submit = async ()', 'return createPortal');
  assertPostsBeforeClosing(body, 'onClose()', 'PromptEditPopover');
  assert.match(code(gridItem), /onSubmit=\{\(prompt\) => onRegenerate\?\.\(img\.id, undefined, prompt, \{ silent: true \}\)\}/,
    'DatasetGridItem must return the regenerate outcome to the bubble');
  assert.match(useDataset, /const regenerate = useCallback\(async \(imageId, loraStrength, prompt, \{ silent/,
    'useDataset.regenerate must report {ok,error}');
  // …and the workspace must FORWARD the options. It dropped the 4th argument, so
  // the bubble drew the refusal AND a toast said the same thing — measured on a
  // 400-px capture, not reasoned about.
  assert.match(code(dsWorkspace),
    /onRegenerate=\{\(id, loraStrength, prompt, opts\) => ds\.regenerate\(id, loraStrength, prompt, opts\)\}/,
    'DatasetWorkspace must pass the options through to regenerate');
  // MEASURED at 400 px: pinned inside the tile (absolute inset-0) the bubble is
  // ~150 px tall, and the refusal rendered as clipped lines over the caption row.
  assert.match(promptPopover, /return createPortal\(/,
    'the bubble must portal out of the tile, or its refusal is unreadable at 400 px');
});

test('🚀 Launch all keeps its checkboxes when the bank is busy', () => {
  const body = slice(code(bank), 'const startPipeline = async (config)', 'const batchStatus');
  assertPostsBeforeClosing(body, 'setLaunchOpen(false)', 'BankWorkspace');
  // act() rewrites the 409 "another pass owns this bank" refusal for every bank
  // button. That wording must reach the dialog instead of only a toast — same
  // sentence, shown next to the checkboxes it is about.
  assert.match(code(bank), /const act = async \(fn, okMsg, \{ onRefusal/,
    'act() must be able to hand its refusal to a caller that owns a surface for it');
  assert.match(body, /onRefusal/, 'startPipeline must claim the refusal instead of toasting it');
});

test('the folder browser keeps the path and the tree position on a refusal', () => {
  const body = slice(folderPicker, 'const use = async ()', 'return (');
  assertPostsBeforeClosing(body, 'onClose()', 'FolderBrowserModal');
  assert.doesNotMatch(code(folderPicker), /onPick\(data\.path\); onClose\(\)/,
    'the unawaited onPick + unconditional close is the whole bug');
  // Both hosts must answer, so "no answer" stays a real error and not a silent close.
  assert.match(code(folderPicker), /onPick=\{\(p\) => \{ onChange\(p\); return \{ ok: true \}/,
    'the text-field host cannot fail, and says so explicitly');
  assert.match(code(dsWorkspace), /onPick=\{\(p\) => ds\.importDatasetFolder\(p, \{ silent: true \}\)\}/,
    'the dataset-import host must report {ok,error} to the browser');
  assert.match(useDataset, /const importDatasetFolder = useCallback\(async \(path, \{ silent/,
    'useDataset.importDatasetFolder must report {ok,error}');
});
