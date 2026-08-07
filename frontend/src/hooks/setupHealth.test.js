import test from 'node:test'
import assert from 'node:assert/strict'
import { setupHealthPhase, shouldRedirectToSetup, joinLabels, regressionNotice,
  statusMessage } from './setupHealth.js'

const VERIFIED = { verified: true, checks: {}, regressions: [] }
const FRESH = { verified: false, checks: {}, regressions: [] }

// --- the redirect: what "coming back" must NOT do -----------------------------

test('a verified install is never sent back to the wizard', () => {
  // The bug this feature exists for: sessionStorage dies with the tab, so a new
  // tab re-offered Setup on a machine set up weeks ago.
  for (const alreadyRedirected of [false, true]) {
    assert.equal(shouldRedirectToSetup({
      loading: false, caps: { configured: true }, state: VERIFIED, alreadyRedirected,
    }), false)
  }
})

test('a verified install is not redirected even if config.json vanished', () => {
  assert.equal(shouldRedirectToSetup({
    loading: false, caps: { configured: false }, state: VERIFIED, alreadyRedirected: false,
  }), false)
})

test('a never-configured backend is still offered Setup, once', () => {
  assert.equal(shouldRedirectToSetup({
    loading: false, caps: { configured: false }, state: FRESH, alreadyRedirected: false,
  }), true)
  assert.equal(shouldRedirectToSetup({
    loading: false, caps: { configured: false }, state: FRESH, alreadyRedirected: true,
  }), false)
})

test('a configured-but-never-verified backend is left alone', () => {
  // Unchanged from before: someone who skipped the wizard on purpose keeps
  // working without being bounced.
  assert.equal(shouldRedirectToSetup({
    loading: false, caps: { configured: true }, state: FRESH, alreadyRedirected: false,
  }), false)
})

test('nothing is decided before the answers are in', () => {
  assert.equal(shouldRedirectToSetup({ loading: true, caps: {}, state: null }), false)
  assert.equal(shouldRedirectToSetup({ loading: false, caps: {}, state: null }), false)
})

// --- the phases ----------------------------------------------------------------

test('phases follow the check', () => {
  assert.equal(setupHealthPhase({ state: null }), 'waiting')
  assert.equal(setupHealthPhase({ state: FRESH }), 'first-run')
  assert.equal(setupHealthPhase({ state: VERIFIED, checking: true }), 'checking')
  assert.equal(setupHealthPhase({ state: VERIFIED, checking: false, result: null }), 'checking')
  assert.equal(setupHealthPhase({ state: VERIFIED, result: { regressions: [] } }), 'ok')
  assert.equal(setupHealthPhase({
    state: VERIFIED, result: { regressions: [{ key: 'masks', label: 'Person masks' }] },
  }), 'regressed')
})

test('on the wizard itself the check is skipped, and says so by saying nothing', () => {
  // Claiming "Setup checked — everything still works" on a page where no check
  // ran would be a report of work that was not done.
  assert.equal(setupHealthPhase({
    state: VERIFIED, result: { regressions: [], skipped: true },
  }), 'skipped')
  assert.equal(statusMessage('skipped'), null)
})

test('the first run says nothing — the wizard owns that screen', () => {
  assert.equal(statusMessage('first-run'), null)
  assert.equal(statusMessage('waiting'), null)
  assert.equal(statusMessage('regressed'), null)
  assert.match(statusMessage('checking'), /background/)
  assert.match(statusMessage('ok'), /still works/)
})

// --- the interruption's wording ------------------------------------------------

test('the warning names what broke, not "a check failed"', () => {
  const n = regressionNotice([
    { key: 'masks', label: 'Person masks' },
    { key: 'training_visible', label: 'LoRA training' },
  ])
  assert.match(n.body, /Person masks and LoRA training/)
  assert.match(n.body, /were working before/)
  assert.deepEqual(n.keys, ['masks', 'training_visible'])
})

test('one broken capability reads as singular', () => {
  const n = regressionNotice([{ key: 'masks', label: 'Person masks' }])
  assert.match(n.title, /A part of your setup/)
  assert.match(n.body, /Person masks was working before and is not responding now/)
})

test('no regression, no notice', () => {
  assert.equal(regressionNotice([]), null)
  assert.equal(regressionNotice(undefined), null)
})

test('label lists read like English', () => {
  assert.equal(joinLabels([]), '')
  assert.equal(joinLabels(['A']), 'A')
  assert.equal(joinLabels(['A', 'B']), 'A and B')
  assert.equal(joinLabels(['A', 'B', 'C']), 'A, B and C')
})
