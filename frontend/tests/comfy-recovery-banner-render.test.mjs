/**
 * What a blocked user actually SEES, rendered.
 *
 * The banner's words are a tested pure function, but the order they appear in
 * is a property of the markup, and order is the whole fix here: someone on a
 * fresh install was told "A paused comfyui job is blocking new generation" while
 * his ComfyUI had never logged a single incoming connection — so he went looking
 * for a flag he had to pass, which was the one thing that could not help
 * (jerkyjunky, Discord). A model that says the right things in the wrong order
 * ships the same bug, and only a render can catch that.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { recoveryBannerModel } = await import('../src/utils/comfyRecovery.js')
const { RecoveryBannerBody } = await import(
  '../src/components/common/ComfyRecoveryBanner.jsx')

const render = (state) => renderToStaticMarkup(
  createElement(RecoveryBannerBody, { model: recoveryBannerModel(state) }))

const UNREACHABLE = {
  recovery: {
    kind: 'unknown_submit', job_id: 'job-1', can_confirm_restart: true,
    stalled_since: '2026-08-04T04:00:00Z',
    connection: { reachable: false, url: 'http://127.0.0.1:8188',
      status: 'unreachable', hint: null },
  },
}

const REACHABLE = {
  recovery: {
    kind: 'prompt', job_id: 'job-2', dataset_id: 7, dataset_name: 'Anna',
    variation_label: 'portrait', stalled_since: '2026-08-04T04:00:00Z',
    can_confirm_restart: true,
    connection: { reachable: true, url: 'http://127.0.0.1:8188', status: 'ok', hint: null },
  },
}

test('the connection comes FIRST in the markup, the paused job after it', () => {
  const html = render(UNREACHABLE)
  const connection = html.indexOf('cannot reach ComfyUI')
  const pausedJob = html.indexOf('could not confirm ComfyUI ever accepted')
  assert.ok(connection >= 0, 'the unreachable link is not rendered at all')
  assert.ok(pausedJob >= 0, 'the paused job vanished instead of being demoted')
  assert.ok(connection < pausedJob, 'the paused job still leads the banner')
})

test('the address and the concrete checks are on screen, not just in the model', () => {
  const html = render(UNREACHABLE)
  assert.match(html, /http:\/\/127\.0\.0\.1:8188/)
  assert.match(html, /host\.docker\.internal/)
  assert.match(html, /--listen/)
  assert.match(html, /<ul/)                      // a list, not one wall of text
  assert.match(html, /I restarted ComfyUI/)      // the exit is still offered
})

/* A URL has no spaces to wrap on: at 400 px the headline must break inside the
   word rather than push the card off the screen. */
test('the headline can break a long address instead of overflowing a phone', () => {
  assert.match(render(UNREACHABLE), /break-words[^"]*"[^>]*>LDS cannot reach/)
})

test('a reachable ComfyUI renders the unchanged paused-job banner', () => {
  const html = render(REACHABLE)
  assert.match(html, /A paused ComfyUI job is blocking new generations/)
  assert.doesNotMatch(html, /cannot reach ComfyUI/)
  assert.doesNotMatch(html, /host\.docker\.internal/)
  assert.match(html, /Anna/)
})

test('nothing blocking renders nothing at all', () => {
  assert.equal(renderToStaticMarkup(
    createElement(RecoveryBannerBody, { model: null })), '')
})

test('every state renders without throwing, including the unreadable record', () => {
  for (const state of [
    UNREACHABLE, REACHABLE,
    { recovery: { kind: 'unreadable', detail: 'invalid record', connection: null } },
    { recovery: { kind: 'prompt', job_id: 'j', connection: { reachable: false, url: '',
      status: 'unconfigured', hint: 'Set the ComfyUI API URL in Settings.' } } },
    { recovery: { kind: 'unknown_submit', job_id: 'j' } },
  ]) {
    assert.doesNotThrow(() => render(state))
  }
})
