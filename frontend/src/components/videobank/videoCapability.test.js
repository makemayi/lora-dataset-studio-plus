import test from 'node:test'
import assert from 'node:assert/strict'

import {
  VIDEO_PIECES, PASS_REQUIREMENTS, missingVideoPieces, passBlockedBy,
  videoCapabilityNotice, joinEnglish,
} from './videoCapability.js'

const caps = (o) => ({ ok: false, detail: 'missing: something', decode: false, detect: false, encode: false, ...o })
const READY = { ok: true, detail: 'video extra ready', decode: true, detect: true, encode: true }

test('a ready install says nothing at all', () => {
  // A green "all good" strip on a working install trains people to skip it.
  assert.equal(videoCapabilityNotice(READY), null)
  assert.deepEqual(missingVideoPieces(READY), [])
})

test('no encoder blocks promotion and NOTHING else', () => {
  // The single sentence this module exists to make true.
  const c = caps({ decode: true, detect: true, encode: false })
  assert.equal(passBlockedBy(c, 'probe'), null)
  assert.equal(passBlockedBy(c, 'detect'), null)
  assert.equal(passBlockedBy(c, 'thumbs'), null)
  assert.equal(passBlockedBy(c, 'pipeline'), null)
  assert.equal(passBlockedBy(c, 'promote').key, 'encode')
})

test('thumbnails need the DECODER, never the encoder', () => {
  // Thumbnails are decoded in-process (av + PIL); wiring them to ffmpeg would
  // grey out the grid on an install whose grid works perfectly.
  assert.deepEqual(PASS_REQUIREMENTS.thumbs, ['decode'])
  assert.equal(passBlockedBy(caps({ decode: true, encode: false }), 'thumbs'), null)
  assert.equal(passBlockedBy(caps({ decode: false, encode: true }), 'thumbs').key, 'decode')
})

test('finding scenes needs the decoder, and is not blocked by a missing encoder', () => {
  // The pass reads frames; it writes no media. Its OTHER requirement — an
  // interpreter that can run CLIP — is not a video piece and is deliberately not
  // modelled here: it belongs to the ✨ Score install step and is refused
  // server-side with its own sentence, so a machine with the video extra and no
  // torch gets told about torch rather than about ffmpeg.
  assert.deepEqual(PASS_REQUIREMENTS.embed, ['decode'])
  assert.equal(passBlockedBy(caps({ decode: true, encode: false }), 'embed'), null)
  assert.equal(passBlockedBy(caps({ decode: false }), 'embed').key, 'decode')
})

test('the chained pipeline needs both of its passes, and names the first gap', () => {
  // With no detector it would probe, find no shots, make no thumbnails, and
  // report success — the shape of a bug that reads as "my videos have no cuts".
  assert.equal(passBlockedBy(caps({ decode: true, detect: false }), 'pipeline').key, 'detect')
  assert.equal(passBlockedBy(caps({ decode: false, detect: false }), 'pipeline').key, 'decode')
  assert.equal(passBlockedBy(READY, 'pipeline'), null)
})

test('a blocked pass names the PIECE and the fix, never "video unavailable"', () => {
  const blocked = passBlockedBy(caps({ decode: true, detect: true }), 'promote')
  assert.match(blocked.why, /Cutting clips is unavailable/)
  assert.match(blocked.why, /ffmpeg/)
  assert.doesNotMatch(blocked.why.toLowerCase(), /video is unavailable/)
})

test('the notice names the missing pieces and what still works', () => {
  const n = videoCapabilityNotice(caps({ decode: true, detect: true, encode: false }))
  assert.equal(n.pieces.length, 1)
  assert.match(n.headline, /Cutting clips is missing\./)
  assert.match(n.stillWorks, /scan your files/)
  assert.match(n.stillWorks, /find the shots/)
  assert.match(n.stillWorks, /triage/)
})

test('watching and triaging survive a completely bare install', () => {
  // Playback is the BROWSER decoding bytes the app streams — it needs no local
  // package. A notice that said "nothing works" would be false.
  const n = videoCapabilityNotice(caps({}))
  assert.equal(n.pieces.length, 3)
  assert.equal(n.headline, 'None of the video pieces are installed yet.')
  assert.equal(n.stillWorks, 'You can still watch any shot and triage.')
})

test('the server’s own detail sentence is carried through verbatim', () => {
  // It names the exact package; paraphrasing it is how a user pip-installs the
  // wrong thing.
  const n = videoCapabilityNotice(caps({ detail: 'missing: av (video decoding)' }))
  assert.equal(n.detail, 'missing: av (video decoding)')
})

test('a missing/garbage capability payload is treated as "nothing installed"', () => {
  // The poll can answer before the probe has run. Failing OPEN here would show
  // enabled buttons that 503 on click.
  assert.equal(missingVideoPieces(undefined).length, 3)
  assert.equal(passBlockedBy(null, 'promote').key, 'encode')
})

test('the three pieces are exactly decode, detect and encode', () => {
  assert.deepEqual(VIDEO_PIECES.map((p) => p.key), ['decode', 'detect', 'encode'])
  for (const p of VIDEO_PIECES) {
    assert.ok(p.label && p.blurb && p.fix, `${p.key}: incomplete descriptor`)
  }
})

test('joinEnglish', () => {
  assert.equal(joinEnglish([]), '')
  assert.equal(joinEnglish(['a']), 'a')
  assert.equal(joinEnglish(['a', 'b']), 'a and b')
  assert.equal(joinEnglish(['a', 'b', 'c']), 'a, b and c')
})
