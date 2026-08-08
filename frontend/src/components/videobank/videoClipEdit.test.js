import test from 'node:test';
import assert from 'node:assert/strict';
import {
  I2V_FIRST_FRAME_HINT, MIN_CLIP_S, boundsAtPlayhead, boundsChanged,
  draftSummary, firstShotBounds, frameStep, isLegalSpan, newShotBounds, nudgedBounds,
  playheadToSourceTime, retouchToast, splitAvailability,
} from './videoClipEdit.js';
import { clipFragmentSrc } from './videoClipFragment.js';

const SHOT = { start_s: 41.25, end_s: 50, source_id: 12 };

// --- the playhead is in SOURCE time ------------------------------------------
//
// THE test of this module. The lightbox's single <video> is pointed at the whole
// source file with a media fragment, so the element's timeline is the file's: a
// player showing the shot that starts at 41.25 s reports currentTime 43.9, not
// 2.65. Everything else here is arithmetic on top of that one fact, and getting
// it wrong is silent — you would get two shots, both plausible, neither where you
// clicked.

test('the player src is the whole SOURCE, which is why currentTime is source time', () => {
  // Not a restatement: this is the evidence for the invariant. The resource the
  // element loads is the rush, the fragment only moves the seek and the stop.
  const src = clipFragmentSrc('/api/video-bank/3/source/12/media',
    SHOT.start_s, SHOT.end_s);
  assert.equal(src, '/api/video-bank/3/source/12/media#t=41.25,50');
  assert.ok(!src.includes('/clip/'), 'a per-clip resource would make currentTime relative');
});

test('a reading inside the shot is already the source timestamp', () => {
  assert.equal(playheadToSourceTime(43.9, SHOT), 43.9);
});

test('a reading a hair before the fragment start still counts as the start', () => {
  // Browsers routinely report 41.249999 right after seeking to the fragment.
  assert.equal(playheadToSourceTime(41.2499, SHOT), 41.25);
});

test('a reading that cannot be in this shot is null, never clamped', () => {
  // Clamping would hand the split point the boundary itself — the one place a
  // split makes an empty clip — and would hide the invariant having broken.
  assert.equal(playheadToSourceTime(2.65, SHOT), null);
  assert.equal(playheadToSourceTime(300, SHOT), null);
  assert.equal(playheadToSourceTime(Number.NaN, SHOT), null);
  assert.equal(playheadToSourceTime(43.9, null), null);
});

// --- splitting ----------------------------------------------------------------

test('a playhead in the middle of a shot can split it', () => {
  assert.deepEqual(splitAvailability(SHOT, 45), { at: 45 });
});

test('a split too close to either bound is refused with a reason, not a clamp', () => {
  // Both halves have to be a shot in their own right — the same rule the server
  // applies, so the disabled button and the 400 cannot disagree.
  assert.ok(splitAvailability(SHOT, 41.4).why);
  assert.ok(splitAvailability(SHOT, 49.9).why);
});

test('a playhead outside the shot says what to do rather than what is wrong', () => {
  assert.match(splitAvailability(SHOT, 10).why, /Move the playhead/);
});

// --- nudging bounds -----------------------------------------------------------

test('one frame is one frame OF THE SOURCE, at its own rate', () => {
  // The same distinction that makes a 16 fps target accelerate motion if you read
  // the wrong column: bounds are timestamps in the source file.
  assert.ok(Math.abs(frameStep(25) - 0.04) < 1e-9);
  assert.ok(Math.abs(frameStep(59.94) - 1 / 59.94) < 1e-9);
});

test('an unprobed or absurd frame rate falls back to a conservative 30', () => {
  // Being one frame conservative on a 60 fps file costs 16 ms; assuming 120 on a
  // 24 fps one moves five frames per click.
  assert.ok(Math.abs(frameStep(null) - 1 / 30) < 1e-9);
  assert.ok(Math.abs(frameStep(0) - 1 / 30) < 1e-9);
  assert.ok(Math.abs(frameStep(100000) - 1 / 30) < 1e-9);
});

test('a nudge moves one edge and leaves the other alone', () => {
  assert.deepEqual(nudgedBounds(SHOT, 'start', 1, 120), { start_s: 42.25, end_s: 50 });
  assert.deepEqual(nudgedBounds(SHOT, 'end', -1, 120), { start_s: 41.25, end_s: 49 });
});

test('a nudge that would invert or shorten the shot past the floor is not applied', () => {
  // A held-down button has to stop at the wall rather than walk through it.
  assert.equal(nudgedBounds({ start_s: 5, end_s: 5.5 }, 'start', 0.2, 120), null);
  assert.equal(nudgedBounds({ start_s: 5, end_s: 5.5 }, 'end', -0.2, 120), null);
});

test('a nudge cannot leave the file at either end', () => {
  assert.equal(nudgedBounds({ start_s: 0.1, end_s: 4 }, 'start', -1, 120), null);
  assert.equal(nudgedBounds({ start_s: 110, end_s: 119.5 }, 'end', 1, 120), null);
});

test('with no probed duration the upper wall is left to the server', () => {
  // A bank whose probe has not run knows no duration. Refusing every edit there
  // would be a guess dressed as a rule; the server still validates.
  assert.deepEqual(nudgedBounds({ start_s: 110, end_s: 119.5 }, 'end', 1, null),
    { start_s: 110, end_s: 120.5 });
});

test('set-to-playhead moves the edge the user is looking at', () => {
  assert.deepEqual(boundsAtPlayhead(SHOT, 'start', 43.9, 120),
    { start_s: 43.9, end_s: 50 });
  assert.deepEqual(boundsAtPlayhead(SHOT, 'end', 43.9, 120),
    { start_s: 41.25, end_s: 43.9 });
});

test('set-to-playhead refuses to make a shot shorter than the floor', () => {
  assert.equal(boundsAtPlayhead(SHOT, 'end', 41.4, 120), null);
});

test('the floor is the one the server enforces', () => {
  assert.equal(MIN_CLIP_S, 0.5);
  assert.ok(isLegalSpan({ start_s: 1, end_s: 1.5 }, 120));
  assert.ok(!isLegalSpan({ start_s: 1, end_s: 1.4 }, 120));
});

// --- the cut the detector missed ----------------------------------------------

test('a hand-made shot starts at the playhead wherever it is in the file', () => {
  // NOT built on playheadToSourceTime: that one refuses a reading outside the open
  // shot, which is precisely where a missed boundary lives. The fragment only sets
  // the seek and the stop — the element's timeline is still the whole rush.
  assert.deepEqual(newShotBounds(70, 120), { start_s: 70, end_s: 75 });
  assert.deepEqual(newShotBounds(2.5, 120), { start_s: 2.5, end_s: 7.5 });
});

test('a hand-made shot near the end of the file is trimmed, not refused', () => {
  assert.deepEqual(newShotBounds(117, 120), { start_s: 117, end_s: 120 });
});

test('there is no room for a shot in the last fraction of a second', () => {
  assert.equal(newShotBounds(119.8, 120), null);
});

test('a file with no shots at all can still get a first one', () => {
  // The reachability hole: every other retouch gesture lives in the lightbox, and
  // the lightbox needs a shot to open. Without this, an install with no detector —
  // which the app tells the user can still "scan, cut, watch and triage" — could
  // not cut a single shot.
  assert.deepEqual(firstShotBounds({ duration_s: 40 }), { start_s: 0, end_s: 5 });
  assert.deepEqual(firstShotBounds({ duration_s: 3 }), { start_s: 0, end_s: 3 });
});

test('an unprobed or absurdly short file offers no first shot', () => {
  assert.equal(firstShotBounds({ duration_s: null }), null);
  assert.equal(firstShotBounds({ duration_s: 0.3 }), null);
  assert.equal(firstShotBounds(null), null);
});

// --- saving -------------------------------------------------------------------

test('float noise from a currentTime reading is not an edit', () => {
  // Saving on it would drop a real thumbnail and a full metrics pass for nothing.
  assert.equal(boundsChanged(SHOT, { start_s: 41.2504, end_s: 50 }), false);
  assert.equal(boundsChanged(SHOT, { start_s: 41.3, end_s: 50 }), true);
});

test('the draft is shown in the units being edited, not in timecode', () => {
  // "0:41 – 0:50" hides the very precision this tool exists to adjust.
  assert.equal(draftSummary({ start_s: 41.25, end_s: 50 }), '41.25s → 50.00s (8.75s)');
});

test('the toast says the thumbnail was dropped, so a blank tile is expected', () => {
  assert.match(retouchToast('bounds'), /thumbnail/);
  assert.match(retouchToast('split'), /split in two/);
  // A brand-new shot never had a thumbnail — claiming one was dropped is a small
  // lie that makes the user look for something that never existed.
  assert.ok(!/were dropped/.test(retouchToast('create')));
});

// --- the i2v discovery ---------------------------------------------------------

test('the tool says that moving the start picks the i2v conditioning frame', () => {
  // ai-toolkit conditions an image-to-video sample on the clip's FIRST frame. So
  // on an i2v target this control is not "trim", it is "choose the image the model
  // learns to animate from" — and no user would guess that from the buttons.
  assert.match(I2V_FIRST_FRAME_HINT, /first frame/i);
  assert.match(I2V_FIRST_FRAME_HINT, /image-to-video/i);
});
