/**
 * THE FRAME-EXTRACTION DIALOG RENDERS, IN EVERY BRANCH IT INTRODUCES.
 *
 * Two of this screen's jobs are sentences, not controls, and a source-text test
 * cannot tell a removed sentence from a broken one:
 *
 *   · "frames per clip" is a CEILING — a clip with nothing sharp, well-exposed
 *     and (with the filter on) showing a usable face contributes fewer images,
 *     or none, and nothing pads to reach the number;
 *   · with the face filter OFF, frames are chosen on sharpness and exposure
 *     alone, so a character set can come back full of sharp pictures of the
 *     wrong person.
 *
 * Both filter states stay MOUNTED and flip `hidden` rather than swapping, for
 * the reason CLAUDE.md ▸ UI changes 5 gives: this app is read through Chrome
 * auto-translate, which rewrites text nodes into its own wrappers, and a React
 * removal then throws NotFoundError and the error boundary eats the section.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { render, createElement } from './support/mountJsx.mjs';

const { default: PromoteFramesDialog } =
  await import('../src/components/videobank/PromoteFramesDialog.jsx');
const { ToastProvider } = await import('../src/components/common/Toast.jsx');

function mount(props = {}) {
  return render(ToastProvider, {
    children: createElement(PromoteFramesDialog, {
      bankId: 1, keepCount: 40, selectedIds: [],
      ...props,
    }),
  });
}

test('the dialog mounts and names what it reads', () => {
  const html = mount();
  assert.match(html, /Build an image training set/);
  assert.match(html, /all 40 kept clips/);
});

test('a selection is counted instead of the whole bank', () => {
  assert.match(mount({ selectedIds: [1, 2, 3] }), /3 selected clips/);
});

test('the ceiling sentence is on screen, with the arithmetic done', () => {
  const html = mount({ keepCount: 40 });      // 40 clips x 3 frames
  assert.match(html, /Up to 120 images/);
  assert.match(html, /not a promise/);
});

test('BOTH face-filter explanations are mounted, one of them hidden', () => {
  const html = mount();   // the filter defaults ON
  assert.match(html, /Frames whose face is too small/);
  assert.match(html, /picked on sharpness and exposure alone/);
  // The OFF copy is the one hidden while the filter is on.
  assert.match(html,
    /hidden=""[^>]*>\s*Off: frames are picked on sharpness|Off: frames are picked/);
});

test('with no reference dataset loaded, the dead end is named, not implied', () => {
  // The list holds only datasets WITH a reference photo, so empty means "none
  // of them can answer this", which is a different fix from "make a dataset".
  const html = mount();
  assert.match(html, /None of your datasets has a reference photo yet/);
});

test('the reference picker is mounted whenever the filter is on', () => {
  const html = mount();
  assert.match(html, /id="frames-ref-ds"/);
  assert.match(html, /Compare against/);
});

test('the submit button is disabled until the request could actually be sent', () => {
  // No name yet -> framePromoteProblem refuses, so the button must not invite a click.
  assert.match(mount(), /<button[^>]*type="submit"[^>]*disabled=""/);
});

test('the refusal is shown as text, not only as a disabled button', () => {
  assert.match(mount(), /Name the dataset first/);
});

test('extraction is NOT gated on ffmpeg — it decodes through PyAV', () => {
  // The clip promotion beside it is; borrowing that gate here refused a run
  // that works, with a message about cutting clips this screen never does.
  assert.doesNotMatch(mount(), /Install ffmpeg/);
});

test('the per-source cap explains dominance rather than just naming itself', () => {
  const html = mount();
  assert.match(html, /Max clips per source/);
  assert.match(html, /one long video can supply most of the set/);
});

test('the hard stop warns that later sources never get a turn', () => {
  assert.match(mount(), /the later sources never get a turn/);
});

test('the dialog is a labelled modal', () => {
  const html = mount();
  assert.match(html, /role="dialog"/);
  assert.match(html, /aria-modal="true"/);
  assert.match(html, /aria-label="Build an image training set"/);
});
