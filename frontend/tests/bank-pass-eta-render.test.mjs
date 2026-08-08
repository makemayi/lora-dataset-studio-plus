/**
 * "How long is left" — as it actually reaches the screen.
 *
 * The banner this was asked for, verbatim from a live 37 800-image bank:
 *
 *   Semantic index running — 12939 / 37800 · loading siglip2-base-p16-224@…
 *                                            on CUDA (local files only)
 *
 * Everything in that sentence is true and none of it answers "do I wait, or do
 * I go and do something else". These tests pin the answer where the user reads
 * it — in the rendered markup — because a source-text assertion cannot see the
 * separator logic that decides whether the clause lands next to the counter, in
 * place of it, or not at all.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { ProgressBar } = await import('../src/components/bank/BankWorkspace.jsx')

const text = (markup) => markup.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()

const bar = (activity) => text(renderToStaticMarkup(
  createElement(ProgressBar, { activity, onCancel: () => {} })))

const INDEXING = {
  kind: 'semantic_index', done: 12939, total: 37800, finished: false,
  detail: 'loading siglip2-base-p16-224 on CUDA (local files only)',
}

test('the semantic index banner now says how long it has left', () => {
  const seen = bar({ ...INDEXING, eta_state: 'ready', eta_seconds: 5220, eta_scope: 'job' })
  assert.equal(
    seen,
    '⏳ Semantic index running — 12939 / 37800 · about 1 hour 30 minutes left '
    + '· loading siglip2-base-p16-224 on CUDA (local files only) Stop')
})

test('while the estimate is still settling the banner says so and promises nothing', () => {
  // Rule one: "3 hours" that becomes "20 minutes" two minutes later does not
  // give the user a number, it teaches them to stop reading the number.
  const seen = bar({ ...INDEXING, eta_state: 'estimating' })
  assert.match(seen, /12939 \/ 37800 · estimating time left…/)
  assert.doesNotMatch(seen, /about/)
  assert.doesNotMatch(seen, /left in/)
})

test('a step with nothing to count gets a counter AND a duration of silence', () => {
  // ✨ Score's style grouping: 181 s measured on 23 000 images, on done=0 /
  // total=0. There is no unit of work here, so there is no honest estimate —
  // and the phase sentence must not be pushed off a 400 px screen by a guess.
  const seen = bar({
    kind: 'score', done: 0, total: 0, finished: false, eta_state: 'none',
    detail: 'grouping styles over 23000 image(s) — the slow tail of this pass',
  })
  assert.equal(seen, '⏳ Scoring pass running — grouping styles over 23000 image(s) '
    + '— the slow tail of this pass Stop')
  assert.doesNotMatch(seen, /left|estimating/)
})

test('after a phase change the banner scopes its promise to the current step', () => {
  // The ✨ Score write-back: ~21 000 rows, its own speed, its own counter. The
  // number measured here says nothing about the style grouping that follows it,
  // so the sentence does not pretend it covers the whole pass.
  const seen = bar({
    kind: 'score', done: 4200, total: 21220, finished: false,
    eta_state: 'ready', eta_seconds: 1200, eta_scope: 'phase',
    detail: 'writing 21220 score(s) to the database…',
  })
  assert.match(seen, /4200 \/ 21220 · about 20 minutes left in this step/)
})

test('a job with no estimator field renders exactly what it rendered before', () => {
  // An old snapshot still in memory across an upgrade, and every pass that
  // cannot count itself. No stray separator, no empty clause.
  const seen = bar({ kind: 'scan', done: 40, total: 100, finished: false, detail: 'quality scan' })
  assert.equal(seen, '⏳ Quality scan running — 40 / 100 · quality scan Stop')
})

test('the clause never doubles the width of the line on a phone', () => {
  // 400 px is where these banners are read. The wrapper already wraps; what
  // must not happen is the DURATION becoming the longest thing on the row.
  const clause = bar({ ...INDEXING, eta_state: 'ready', eta_seconds: 5220, eta_scope: 'phase' })
    .match(/· (about [^·]+left in this step)/)
  assert.ok(clause, 'the clause is present')
  // 45 characters is about one line of this text size inside a 400 px card, and
  // the longest clause the formatter can emit ("about 3 hours 30 minutes left
  // in this step") is 42. The row wraps, so overflowing costs a second line —
  // but a clause needing TWO would push the phase sentence, the thing that
  // explains what the pass is doing, off the visible card.
  assert.ok(clause[1].length <= 45, `too long for a phone: ${clause[1]}`)
})
