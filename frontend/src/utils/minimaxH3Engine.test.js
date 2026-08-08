import test from 'node:test';
import assert from 'node:assert/strict';
import {
  minimaxH3UnavailableReason, minimaxH3SpeedWarning, h3MissingLabels,
  refImageSizeDescription, packetLengthDescription, H3_ASSET_LABELS,
} from './minimaxH3Engine.js';

test('a complete install has no reason at all', () => {
  assert.equal(minimaxH3UnavailableReason({}), null);
});

test('the reasons are ordered by what has to be fixed first', () => {
  // Disabled beats everything: the asset list is meaningless for an engine the
  // user turned off.
  assert.match(minimaxH3UnavailableReason({
    enabledInSettings: false, comfyuiReachable: false, missingNodes: ['H3FrameSelect'],
  }), /disabled in Settings/);
  // ComfyUI down beats the asset list for the same reason: nothing could have
  // been probed.
  assert.match(minimaxH3UnavailableReason({
    comfyuiReachable: false, missingAssets: ['h3_unet'],
  }), /ComfyUI/);
  // The node pack beats the weights: without it nothing runs even with every
  // file in place.
  assert.match(minimaxH3UnavailableReason({
    missingNodes: ['H3FrameSelect'], missingAssets: ['h3_unet'],
  }), /MinimaxH3-Image/);
});

test('missing weights are named individually, in a stable order', () => {
  const reason = minimaxH3UnavailableReason({
    missingAssets: ['h3_clip_vision', 'h3_unet'],
  });
  assert.match(reason, /Ref2VA model \+ CLIP-Vision tower/);
});

test('every asset key the labels claim to cover is spelled out', () => {
  // A key with no label would silently vanish from the sentence, leaving a
  // shorter list that looks complete.
  assert.deepEqual(h3MissingLabels(Object.keys(H3_ASSET_LABELS)),
    Object.values(H3_ASSET_LABELS));
});

test('the message never promises a download the app cannot do', () => {
  // Klein and Krea say "Setup can download them for you". H3's five files are
  // ~40 GB of community re-quantisations and are NOT in the installer, so
  // borrowing that sentence would be a lie.
  const reason = minimaxH3UnavailableReason({ missingAssets: ['h3_unet'] });
  assert.doesNotMatch(reason, /Setup can download/);
});

test('the VRAM advisory is separate from the reasons — it never darkens the card', () => {
  // A complete install that will merely crawl must still be pickable: the flag
  // costs a ComfyUI restart, and refusing to generate over it would be worse
  // than a slow generation.
  assert.equal(minimaxH3UnavailableReason({}), null);
  assert.equal(minimaxH3SpeedWarning({ minimax_h3: { vram_warning: 'add the flag' } }),
    'add the flag');
  assert.equal(minimaxH3SpeedWarning({ minimax_h3: { vram_warning: null } }), null);
  assert.equal(minimaxH3SpeedWarning(null), null);
});

test('the dials describe what they cost, not just their value', () => {
  assert.match(refImageSizeDescription('match'), /measured default/);
  assert.match(refImageSizeDescription('max'), /slower/);
  assert.match(packetLengthDescription(5), /floor/);
  assert.match(packetLengthDescription(22), /22 frames/);
  assert.match(packetLengthDescription(22), /slower/);
});
