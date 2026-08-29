import assert from 'node:assert/strict'
import test from 'node:test'

import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

// The JSX loader is registered while mountJsx is evaluated, so this import has
// to stay dynamic (see support/mountJsx.mjs).
const { default: PromoteFramesDialog } =
  await import('../src/components/videobank/PromoteFramesDialog.jsx')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')

const props = { bankId: 1, keepCount: 4, selectedIds: [], onClose: () => {} }
const renderDialog = () => renderToStaticMarkup(
  createElement(ToastProvider, null, createElement(PromoteFramesDialog, props)))

/* Mounted rather than grepped: the framings block is three checkboxes whose
   disabled state follows the person requirement, and a source-text assertion
   cannot see a branch that throws on render (CLAUDE.md ▸ UI changes, rule 6).
   The reference list fetch is an effect — server rendering runs none — so the
   dialog mounts in its default state (identity mode, full frame ticked). */

test('the framings block lists all three, with full frame ticked', () => {
  const html = renderDialog()
  assert.match(html, /Framings/)
  assert.match(html, /Full frame/)
  assert.match(html, /Waist-up/)
  assert.match(html, /Face close-up/)
  assert.match(html, /checked[^>]*type="checkbox"|type="checkbox"[^>]*checked/)
})

test('the copy says each framing takes its own moments', () => {
  // The whole point of the crops is MORE distinct usable instants; a reader
  // who expects triplets of one instant has to be corrected up front.
  const html = renderDialog()
  assert.match(html, /its own moments/)
  assert.match(html, /EACH framing/)
})

test('the dialog still renders its name and budget fields', () => {
  const html = renderDialog()
  assert.match(html, /frames-ds-name/)
  assert.match(html, /frames-per-clip/)
})
