/**
 * The OneTrainer lane's progress, RENDERED.
 *
 * Until 2026-08-29 this lane had none: the panel polled a route gated on
 * ai-toolkit, which then read ai-toolkit's own `training.log` — a file a
 * OneTrainer run never writes. The card therefore showed "Starting up… (the log
 * appears once ai-toolkit begins writing)" for the entire run, naming a program
 * that was not running.
 *
 * Mounted rather than grepped, because the two things worth protecting are both
 * render decisions: which sentence appears while nothing is parsed yet, and
 * whether the epoch pair survives being drawn.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { ProgressView } = await import(
  '../src/components/dataset/TrainingProgress.jsx')

const view = (prog, extra = {}) => renderToStaticMarkup(createElement(
  ProgressView, { prog, datasetId: 1, base: null, trainType: 'krea', variant: 'raw', ...extra }))

test('the waiting line names the trainer that is actually running', () => {
  const both = view({ log_exists: false, active: true, trainer: 'onetrainer' })
  // BOTH sentences stay mounted — a ternary here is what Chrome auto-translate
  // turns into a removeChild crash — so the assertion is on which one is hidden.
  assert.match(both, /the log appears once OneTrainer begins writing/)
  assert.match(both, /the log appears once ai-toolkit begins writing/)
  assert.match(both, /hidden=""[^>]*>\s*Starting up… \(the log appears once ai-toolkit/,
    'the ai-toolkit sentence is the hidden one on a OneTrainer run')

  const aitk = view({ log_exists: false, active: true })
  assert.match(aitk, /hidden=""[^>]*>\s*Starting up… \(the log appears once OneTrainer/,
    'and the other way round when no trainer is named')
})

test('a parsed OneTrainer payload draws steps, percent and the epoch pair', () => {
  const html = view({
    log_exists: true, active: true, trainer: 'onetrainer',
    step: 116, total: 3432, epoch: 5, epochs: 156,
    loss: 0.155, speed: '2.81s/it', eta: '3:32:37', loss_curve: [[100, 0.2], [116, 0.155]],
    samples: [], download: null, cache_pending: null,
  })
  assert.match(html, /116 \/ 3432 steps \(3%\)/)
  // +1: tqdm counts completed epochs, so it prints 5 while inside the sixth.
  assert.match(html, /epoch\s*6\s*\/\s*156/)
  assert.match(html, /2\.81s\/it/)
  assert.match(html, /ETA\s*3:32:37/)
  assert.match(html, /aria-valuenow="116"/)
})

test('the epoch pair is absent for a lane that has no epochs', () => {
  const html = view({
    log_exists: true, active: true, step: 500, total: 3500,
    loss: 0.1, loss_curve: [], samples: [],
  })
  assert.match(html, /500 \/ 3500 steps/)
  assert.doesNotMatch(html, /epoch/)
})

test('a run with no numbers yet does not throw', () => {
  assert.equal(typeof view(null), 'string')
  assert.equal(typeof view({ log_exists: true, active: true, samples: [] }), 'string')
})
