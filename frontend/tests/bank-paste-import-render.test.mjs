import assert from 'node:assert/strict'
import test from 'node:test'

import { render } from './support/mountJsx.mjs'

// The JSX loader is registered while mountJsx is evaluated, so this import has
// to stay dynamic (see support/mountJsx.mjs).
const { default: BankPasteImport } =
  await import('../src/components/bank/BankPasteImport.jsx')

/* Mounted rather than grepped: a source-text assertion cannot tell a branch that
   was DELETED from one that throws on render, and this panel's whole job is to
   flip between an error line and a live count (CLAUDE.md ▸ UI changes, rule 6). */

test('the empty state renders both status branches, only one visible', () => {
  const html = render(BankPasteImport, { onImport: () => {}, busy: false })
  // Nothing pasted yet -> the error branch is the visible one...
  assert.match(html, /Nothing pasted yet/)
  // ...and the count branch is still IN the document, merely hidden. Swapping
  // the two with a ternary is what breaks under Chrome auto-translate.
  assert.match(html, /hidden[^>]*>\s*0 of 0 picked/)
})

test('the import button starts disabled — an empty paste must not queue a job', () => {
  const html = render(BankPasteImport, { onImport: () => {}, busy: false })
  assert.match(html, /<button[^>]*disabled[^>]*>[\s\S]*?Import[\s\S]*?into the bank/)
})

test('both button labels stay mounted so the busy swap is a hidden flip', () => {
  const html = render(BankPasteImport, { onImport: () => {}, busy: false })
  assert.match(html, /into the bank/)
  assert.match(html, /Importing…/)
})

test('the pick controls and grid are mounted but hidden while there is nothing to pick', () => {
  // Same rule as the status line: present in the DOM, hidden by attribute, so a
  // paste does not mount a subtree Chrome's translator is mid-way through.
  const html = render(BankPasteImport, { onImport: () => {}, busy: false })
  assert.match(html, /hidden[^>]*>[\s\S]*?Select all/)
  assert.match(html, /Select none/)
})

test('the copy states the limits instead of implying a crawler', () => {
  // "Every limit stays visible" — someone who expects this to follow an account
  // would otherwise sit waiting for a second page that never comes.
  const html = render(BankPasteImport, { onImport: () => {}, busy: false })
  assert.match(html, /does not follow an\s+account/)
  assert.match(html, /paste them whole/)
  // ...and that a dead thumbnail is a dropped link, not a silent failure later.
  assert.match(html, /dropped rather than\s+queued/)
})

test('the textarea is labelled and the region is addressable for help', () => {
  const html = render(BankPasteImport, { onImport: () => {}, busy: false })
  assert.match(html, /id="bank-paste-import"/)
  assert.match(html, /aria-label="Image links to import into the bank"/)
  assert.match(html, /aria-live="polite"/)
})
