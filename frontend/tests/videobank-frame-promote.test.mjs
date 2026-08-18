import { test } from 'node:test'
import assert from 'node:assert/strict'
import { render } from './support/mountJsx.mjs'

// The JSX loader is registered while mountJsx is evaluated, so these imports
// have to stay dynamic (see support/mountJsx.mjs).
const { default: FramePromoteSummary } =
  await import('../src/components/videobank/FramePromoteSummary.jsx')

test('summary renders totals and per-clip rows with reasons', () => {
  const html = render(FramePromoteSummary, {
    result: { frames: 61,
      totals: { too_blurry: 68, face_blurry: 25, exposure: 9 },
      per_clip: [{ clip_id: 12, picked: 3,
                   rejected: { too_blurry: 5, face_blurry: 3 } }] },
    onClose: () => {}, onOpenClip: () => {},
  })
  assert.ok(html.includes('61 frames imported'), 'total missing')
  assert.ok(html.includes('68 blurry'), 'total rejection line missing')
  assert.ok(html.includes('Clip #12'), 'per-clip row missing')
  assert.ok(html.includes('blurry face'), 'reason label missing')
  assert.ok(html.includes('View'), 'jump button missing')
})

test('summary renders the empty report gracefully', () => {
  const html = render(FramePromoteSummary, {
    result: { frames: 10, totals: {}, per_clip: [] },
    onClose: () => {}, onOpenClip: () => {},
  })
  assert.ok(html.includes('10 frames imported'))
  assert.ok(html.includes('Nothing to report'))
})

test('summary survives a missing result object', () => {
  const html = render(FramePromoteSummary, {
    result: null, onClose: () => {}, onOpenClip: () => {},
  })
  assert.ok(html.includes('0 frames imported'))
})
