import assert from 'node:assert/strict'
import test from 'node:test'

import { attemptModalSubmit, submitAttemptOutcome } from './submitOutcome.js'

test('a refusal NEVER closes the modal — the input stays where it is', () => {
  // the never-throwing shape (useDataset's postJson: caption, regenerate, import)
  assert.deepEqual(submitAttemptOutcome({ response: { ok: false, error: 'Image not found' } }),
    { close: false, error: 'Image not found' })
  // the throwing shape (fetchClient's apiFetch: the bank's pipeline route)
  assert.deepEqual(submitAttemptOutcome({ thrown: new Error('a scan job is already running') }),
    { close: false, error: 'a scan job is already running' })
})

test('only a success closes the modal', () => {
  assert.deepEqual(submitAttemptOutcome({ response: { ok: true } }), { close: true, error: null })
})

test('silence is not a success', () => {
  // The whole family started here: a handler fired without `await` answers
  // nothing, and closing on nothing is what destroyed the input.
  const out = submitAttemptOutcome({ fallback: 'Could not save the caption' })
  assert.equal(out.close, false)
  assert.equal(out.error, 'Could not save the caption — no answer from the server.')
})

test('a wordless refusal still says something', () => {
  assert.deepEqual(submitAttemptOutcome({ response: { ok: false }, fallback: 'Could not save the caption' }),
    { close: false, error: 'Could not save the caption' })
})

test('a hint rides the refusal instead of being dropped', () => {
  assert.deepEqual(
    submitAttemptOutcome({ response: { ok: false, error: 'Folder refused', hint: 'no images in it' } }),
    { close: false, error: 'Folder refused — no images in it' })
})

test('a decline keeps the modal and says nothing', () => {
  // The user answered "no" at a confirm. That answer IS the explanation; an
  // error banner would be the app arguing with them.
  assert.deepEqual(submitAttemptOutcome({ declined: true }), { close: false, error: null })
})

/* ── One per modal, with the handler shape that modal really receives ──────── */

test('the caption editor keeps BOTH captions when the write is refused', async () => {
  // DatasetGridItem's onSave: awaits useDataset.setCaption, which never throws.
  const draft = { long: 'a woman in a red coat, city street at night', short: 'red coat' }
  const onSave = async () => ({ ok: false, error: 'Image not found' })
  const out = await attemptModalSubmit(() => onSave(draft.long, draft.short),
    { fallback: 'Could not save the caption' })
  assert.equal(out.close, false, 'the editor must stay open, or the caption is gone')
  assert.equal(out.error, 'Image not found')
  // nothing in the rule touches the drafts — they are the dialog's own state
  assert.deepEqual(draft, { long: 'a woman in a red coat, city street at night', short: 'red coat' })
})

test('an unchanged caption closes without posting anything', async () => {
  // Save on a caption nobody edited must still behave like Cancel, not like a
  // failed request.
  const out = await attemptModalSubmit(async () => ({ ok: true }))
  assert.deepEqual(out, { close: true, error: null })
})

test('the prompt bubble keeps a hand-rewritten prompt when the GPU is busy', async () => {
  const out = await attemptModalSubmit(async () => ({ ok: false, error: 'A generation is already running' }),
    { fallback: 'Could not start the regeneration' })
  assert.equal(out.close, false)
  assert.equal(out.error, 'A generation is already running')
})

test('🚀 Launch all keeps its checkboxes when the bank refuses', async () => {
  // BankWorkspace.startPipeline answers {ok:false,error} carrying act()'s
  // reworded 409 — the same sentence the toast used to carry, now shown next to
  // the checkboxes it is about.
  const config = { steps: ['scan', 'score', 'faces'], reject_flags: ['blur'], resolve_dups: true }
  const startPipeline = async () => ({ ok: false, error: 'A scan pass owns this bank — Stop it or wait.' })
  const out = await attemptModalSubmit(() => startPipeline(config), { fallback: 'Could not start the run' })
  assert.equal(out.close, false)
  assert.match(out.error, /owns this bank/)
  assert.deepEqual(config.steps, ['scan', 'score', 'faces'], 'the picked passes are untouched')
})

test('the folder browser keeps the tree position when the path is refused', async () => {
  const out = await attemptModalSubmit(async () => ({ ok: false, error: 'No images in that folder.' }),
    { fallback: 'That folder was refused' })
  assert.deepEqual(out, { close: false, error: 'No images in that folder.' })
})

test('a throw is a refusal, not a crash that freezes the modal', async () => {
  // attemptModalSubmit exists so no dialog re-invents a try WITHOUT a catch: an
  // escaping rejection would leave busy=true and the modal stuck on "Saving…".
  const out = await attemptModalSubmit(async () => { throw new Error('Connection lost.') },
    { fallback: 'Could not save the caption' })
  assert.deepEqual(out, { close: false, error: 'Connection lost.' })
})

test('two refusals in a row do not loop — each is one closed answer', async () => {
  // The trap: a rule that retried by itself would re-post forever against a
  // server that keeps saying no. This rule decides ONCE per attempt and never
  // calls anything; the second Save is a second user gesture, and the input is
  // still intact after both.
  let posts = 0
  const post = async () => { posts += 1; return { ok: false, error: `refused #${posts}` } }
  const first = await attemptModalSubmit(post)
  const second = await attemptModalSubmit(post)
  assert.equal(posts, 2, 'exactly one request per attempt')
  assert.equal(first.close, false)
  assert.equal(second.close, false)
  assert.equal(second.error, 'refused #2', 'the second refusal replaces the first, it does not stack')
  // …and a success after two refusals still closes.
  const third = await attemptModalSubmit(async () => ({ ok: true }))
  assert.deepEqual(third, { close: true, error: null })
})
