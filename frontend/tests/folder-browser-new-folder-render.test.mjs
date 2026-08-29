import assert from 'node:assert/strict'
import test from 'node:test'

import { render } from './support/mountJsx.mjs'

// The JSX loader is registered while mountJsx is evaluated, so this import has
// to stay dynamic (see support/mountJsx.mjs). FolderBrowserModal is a NAMED
// export — the module's default is the field, not the modal.
const { FolderBrowserModal } =
  await import('../src/components/common/FolderPicker.jsx')

const props = { initial: null, onPick: async () => ({ ok: true }), onClose: () => {} }

/* Mounted rather than grepped: the new-folder button has two states that must
   both exist in the DOM (CLAUDE.md ▸ UI changes, rule 5), and the modal only
   ever renders for real after a fetch the render tests cannot run — so what a
   mount can prove is that both labels, the guard and the refusal slot exist. */

test('the new-folder control exists with both labels mounted', () => {
  const html = render(FolderBrowserModal, props)
  assert.match(html, /＋ New folder/)
  assert.match(html, /Creating…/)
})

test('it is disabled at the roots view, with the reason on hover', () => {
  // No drive open yet → there is no "inside" to create in. The button must not
  // just be inert: the title says why, so the disabled state reads as an
  // instruction rather than a broken button.
  const html = render(FolderBrowserModal, props)
  assert.match(html, /disabled[^>]*title="Open a drive first[^"]*"/)
})

test('the modal itself is what renders, address bar and all', () => {
  // The refusal box (role=alert) only exists in the error state, which a render
  // test cannot reach — what a mount CAN prove is that the modal is intact and
  // the button lives in its header, guarded at the roots view.
  const html = render(FolderBrowserModal, props)
  assert.match(html, /role="dialog" aria-modal="true" aria-label="Choose a folder"/)
  assert.match(html, /aria-label="Folder path"/)
})
