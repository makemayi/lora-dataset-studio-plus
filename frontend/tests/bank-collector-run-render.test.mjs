import assert from 'node:assert/strict'
import test from 'node:test'

import { render } from './support/mountJsx.mjs'

// The JSX loader is registered while mountJsx is evaluated, so this import has
// to stay dynamic (see support/mountJsx.mjs).
const { default: BankCollectorRun } =
  await import('../src/components/bank/BankCollectorRun.jsx')

const props = { destinationOf: () => ({ name: 'x' }), post: async () => ({}), busy: false }

/* Mounted rather than grepped: a source-text assertion cannot tell a branch that
   was DELETED from one that throws on render, and this block is nothing BUT two
   branches (CLAUDE.md ▸ UI changes, rule 6). */

test('with nothing configured it says so, and says where a collector comes from', () => {
  // The shipped state. It must read as a configuration this app HAS, not as an
  // error and not as a missing install step — no collector ships, on purpose.
  const html = render(BankCollectorRun, props)
  assert.match(html, /No collector is configured/)
  assert.match(html, /bank\.collectors/)
  assert.match(html, /stops working when that site changes/)
})

test('the run controls are mounted but hidden until a collector exists', () => {
  // Present in the DOM, hidden by attribute — the same rule the rest of this
  // panel follows so Chrome's translator never races an unmounting subtree.
  const html = render(BankCollectorRun, props)
  assert.match(html, /hidden[^>]*>[\s\S]*?aria-label="Collector to run"/)
  assert.match(html, /aria-label="Page or account URL to collect from"/)
})

test('both Run labels stay mounted so the starting swap is a hidden flip', () => {
  const html = render(BankCollectorRun, props)
  assert.match(html, />Run</)
  assert.match(html, /Starting…/)
})

test('the copy warns that this takes minutes and survives leaving the page', () => {
  // "Every limit stays visible": a browser-driving collector is not an API call,
  // and someone who does not know that reads a slow run as a hang.
  const html = render(BankCollectorRun, props)
  assert.match(html, /several minutes/)
  assert.match(html, /close this page/)
})

test('the block is addressable and its error line is polite', () => {
  const html = render(BankCollectorRun, props)
  assert.match(html, /id="bank-collector-run"/)
  assert.match(html, /aria-live="polite"/)
})
