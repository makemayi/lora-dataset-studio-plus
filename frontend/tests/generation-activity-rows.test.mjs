/**
 * The per-batch progress rows — MOUNTED, because the thing that can break here
 * is a branch, and a source-text test cannot tell a removed branch from a
 * broken one (CLAUDE.md ▸ UI changes, rule 6).
 *
 * What they replaced: a single label on the Generate button reading
 * "Generating… 12/40". That worked only while exactly one batch could exist. It
 * can no longer — a local batch and an API batch are independent on the server —
 * and a single label would have shown one and silently dropped the other, which
 * is a worse failure than the block it replaced: the user launches something and
 * the screen says nothing about it.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const mod = await import('../src/components/dataset/VariationCatalog.jsx')
const Rows = mod.GenerationActivityRows

test('the rows component is exported so it can be mounted at all', () => {
  assert.equal(typeof Rows, 'function')
})

test('two live batches BOTH appear, with their own counts', () => {
  const html = renderToStaticMarkup(createElement(Rows, {
    activities: [
      { kind: 'generate', engine: 'chatgpt', done: 1, total: 4, started_at: 2 },
      { kind: 'generate', engine: 'klein', done: 12, total: 40, started_at: 1 },
    ],
  }))
  assert.match(html, /ChatGPT/)
  assert.match(html, /1\/4/)
  assert.match(html, /Klein/)
  assert.match(html, /12\/40/)
})

test('only generation batches show — a caption pass has its own indicator', () => {
  const html = renderToStaticMarkup(createElement(Rows, {
    activities: [{ kind: 'caption', done: 3, total: 9, started_at: 1 }],
  }))
  assert.equal(html, '')
})

test('nothing running renders nothing at all', () => {
  for (const activities of [null, undefined, []]) {
    assert.equal(renderToStaticMarkup(createElement(Rows, { activities })), '')
  }
})

test('a batch with no total says "running" rather than 0/0', () => {
  const html = renderToStaticMarkup(createElement(Rows, {
    activities: [{ kind: 'generate', engine: 'krea', done: 0, total: 0 }],
  }))
  assert.match(html, /running/)
  assert.doesNotMatch(html, /0\/0/)
})

test('both state labels stay mounted, toggled by hidden', () => {
  // Chrome auto-translate rewrites text nodes; a ternary swap throws there.
  const html = renderToStaticMarkup(createElement(Rows, {
    activities: [{ kind: 'generate', engine: 'klein', done: 2, total: 5,
                   cancelling: true }],
  }))
  assert.match(html, /stopping…/)
  assert.match(html, /2\/5/)          // the other half is present, just hidden
  assert.match(html, /hidden=""/)
})
