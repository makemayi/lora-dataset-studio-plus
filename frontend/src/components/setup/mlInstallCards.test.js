/** WHY THIS FILE EXISTS — the day the video wave landed, its capability strip
 * told the first real user "→ Install the video extra from Setup", and Setup
 * had no such button: both installs existed only as API actions. A promise in
 * one file, its keeper in another, nothing holding them together. These tests
 * are that hold. */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { ML_INSTALL_CARDS, cardCaps, cardInstalled } from './mlInstallCards.js'
import { VIDEO_PIECES } from '../videobank/videoCapability.js'

test('every install the video capability strip points at has a Setup card', () => {
  // A piece whose `fix` line says "from Setup" MUST carry `setupCap`, and a
  // card must exist whose `cap` turns that exact probe green — otherwise the
  // remedy is a dead end the user walks into.
  const offered = new Set(ML_INSTALL_CARDS.flatMap(cardCaps))
  for (const piece of VIDEO_PIECES) {
    if (!/from Setup/i.test(piece.fix)) continue
    assert.ok(piece.setupCap,
      `"${piece.label}" sends the user to Setup but names no setupCap to look for`)
    assert.ok(offered.has(piece.setupCap),
      `the strip sends the user to Setup for "${piece.setupCap}" and Setup has no card for it`)
  }
})

test('the strip actually names at least one Setup-installable piece', () => {
  // Guards the test above against silently testing nothing: if no piece said
  // "from Setup" any more, the loop would pass while pinning zero promises.
  assert.ok(VIDEO_PIECES.some((p) => /from Setup/i.test(p.fix)),
    'no strip piece points at Setup — the contract test has gone blind')
})

test('cards are unique per action and carry the fields the page renders', () => {
  const actions = ML_INSTALL_CARDS.map((c) => c.action)
  assert.equal(new Set(actions).size, actions.length, 'duplicate action in ML_INSTALL_CARDS')
  for (const c of ML_INSTALL_CARDS) {
    for (const field of ['action', 'cap', 'icon', 'title', 'body']) {
      assert.ok(c[field], `card ${c.action || '?'} is missing "${field}"`)
    }
    assert.ok(cardCaps(c).length && cardCaps(c).every((k) => typeof k === 'string' && k),
      `card ${c.action} has a malformed cap`)
  }
})

// The video action installs PyAV *and* a bundled ffmpeg, and probe_video()
// reports them apart because they fail apart (imageio-ffmpeg returns a path even
// when its binary download never finished). A card badge reading only the first
// key said "✓ Installed" on a machine that could not encode a single clip — and
// a green badge is exactly where the user stops looking for the ↻ Reinstall.
test('a card is only "installed" when EVERY piece its action installs is present', () => {
  const video = ML_INSTALL_CARDS.find((c) => c.action === 'video')
  assert.ok(video, 'no video card')
  assert.deepEqual(cardCaps(video), ['video_decode', 'video_encode'])
  assert.equal(cardInstalled(video, { video_decode: true, video_encode: false }), false)
  assert.equal(cardInstalled(video, { video_decode: false, video_encode: true }), false)
  assert.equal(cardInstalled(video, { video_decode: true, video_encode: true }), true)
  // Single-key cards keep behaving exactly as before.
  const masks = ML_INSTALL_CARDS.find((c) => c.action === 'masks')
  assert.equal(cardInstalled(masks, { masks: true }), true)
  assert.equal(cardInstalled(masks, {}), false)
  assert.equal(cardInstalled(masks, null), false)
})
