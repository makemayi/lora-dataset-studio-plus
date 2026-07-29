import assert from 'node:assert/strict'
import test from 'node:test'

import { continueAttemptOutcome, continueRefusalMessage } from './continueOutcome.js'
import { postWithConfirmations } from './trainingRefusals.js'

test('a refusal NEVER closes the dialog — the inputs stay where they are', () => {
  // the never-throwing hook shape (dataset panel)
  assert.deepEqual(continueAttemptOutcome({ response: { ok: false, error: 'no checkpoint at step 750' } }),
    { close: false, error: 'no checkpoint at step 750' })
  // the throwing shape (Runs hub, board)
  assert.deepEqual(continueAttemptOutcome({ thrown: new Error('A training is already running') }),
    { close: false, error: 'A training is already running' })
})

test('a hint rides the refusal instead of being dropped', () => {
  assert.deepEqual(
    continueAttemptOutcome({ response: { ok: false, error: 'Cloud training refused', hint: 'add a vast.ai API key' } }),
    { close: false, error: 'Cloud training refused — add a vast.ai API key' })
})

test('only a success closes the dialog', () => {
  assert.deepEqual(continueAttemptOutcome({ response: { ok: true, resumed_from: 750, target_steps: 1750 } }),
    { close: true, error: null })
})

test('a decline keeps the dialog with its inputs and says nothing', () => {
  // The user answered "no" to "Continue anyway (force)?" — that answer IS the
  // explanation; an error banner would be the app arguing with them.
  assert.deepEqual(continueAttemptOutcome({ declined: true }), { close: false, error: null })
})

test('an empty answer is reported, never mistaken for a success', () => {
  const out = continueAttemptOutcome({})
  assert.equal(out.close, false)
  assert.match(out.error, /no answer from the server/)
})

test('a confirmable marker that survived the confirm loop is stripped for the eye', () => {
  // UNCAPTIONED: only reaches the dialog when the server refused AGAIN after we
  // sent allow_uncaptioned — the user must read the sentence, not the wire tag.
  assert.equal(continueRefusalMessage('UNCAPTIONED: 12 images have no caption.'),
    '12 images have no caption.')
  assert.equal(continueRefusalMessage('NOT_READY: only 3 kept image(s)'), 'only 3 kept image(s)')
  assert.equal(continueRefusalMessage(''), 'Continue failed')
  assert.equal(continueRefusalMessage(null, 'Continue failed'), 'Continue failed')
})

test('two refusals in a row cannot loop the dialog open — the flag is offered ONCE', async () => {
  // The trap this guards: a server that keeps refusing with the same confirmable
  // marker even after the force flag rode the retry. A hand-rolled loop would
  // re-ask forever; postWithConfirmations adds a flag once, then gives up — and
  // the give-up lands here as an ordinary refusal that keeps the dialog OPEN
  // exactly once, with the input intact.
  const confirms = []
  const realConfirm = globalThis.window?.confirm
  globalThis.window = { ...(globalThis.window || {}), confirm: (m) => { confirms.push(m); return true } }
  try {
    const posts = []
    const post = async (body) => {
      posts.push(body)
      throw new Error('UNCAPTIONED: 12 images have no caption.')
    }
    let outcome
    try {
      outcome = continueAttemptOutcome({ response: await postWithConfirmations(post, { extra_steps: 1000 }, 'Continue anyway (force)') })
    } catch (e) {
      outcome = continueAttemptOutcome({ thrown: e })
    }
    assert.equal(posts.length, 2, 'the request is retried once with the flag, never more')
    assert.equal(confirms.length, 1, 'the user is asked once, not in a loop')
    assert.equal(outcome.close, false)
    assert.equal(outcome.error, '12 images have no caption.')
  } finally {
    if (realConfirm) globalThis.window.confirm = realConfirm
  }
})
