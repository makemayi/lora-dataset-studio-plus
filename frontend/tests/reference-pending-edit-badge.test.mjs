import assert from 'node:assert/strict'
import test from 'node:test'

import { render } from './support/mountJsx.mjs'

// The JSX loader is registered while mountJsx is evaluated, so this import has
// to stay dynamic (see support/mountJsx.mjs).
const { default: ReferencePanel } =
  await import('../src/components/dataset/ReferencePanel.jsx')

/* Why this badge is on the card and not only in the modal.
 *
 * A reference edit is a PAID call. When one lands, the server now keeps it
 * across a restart — but the ✦ Edit modal only opens on a click, so the restored
 * result was reachable and never announced: the user had no way to learn it was
 * there, and the TTL deleted it half an hour later. The recovery is only worth
 * what the user can SEE, which is what these assertions pin. */

const READY = {
  status: 'ready',
  engines: ['chatgpt'],
  candidates: {
    chatgpt: { engine: 'chatgpt', status: 'ready', candidate_filename: 'c.webp' },
  },
}

const base = { refFilename: 'ref.webp', datasetId: 7, onEditRef: () => {} }

test('a landed edit is announced on the reference card', () => {
  const html = render(ReferencePanel, { ...base, referenceEdit: READY })
  assert.match(html, /An edited version is waiting/)
})

test('the announcement is a button, so it leads to the Keep/Discard screen', () => {
  // A plain label would tell the user something is waiting and leave them to
  // find it. The badge has to be the way in.
  const html = render(ReferencePanel, { ...base, referenceEdit: READY })
  const badge = /<button[^>]*>[^<]*An edited version is waiting[^<]*<\/button>/
  assert.match(html.replace(/\s+/g, ' '), badge)
})

test('nothing is announced when there is nothing to decide', () => {
  for (const referenceEdit of [
    null,
    { status: 'running', engines: ['chatgpt'],
      candidates: { chatgpt: { engine: 'chatgpt', status: 'running' } } },
    { status: 'failed', engines: ['chatgpt'],
      candidates: { chatgpt: { engine: 'chatgpt', status: 'failed', error: 'boom' } } },
  ]) {
    const html = render(ReferencePanel, { ...base, referenceEdit })
    assert.doesNotMatch(html, /is waiting/, JSON.stringify(referenceEdit))
  }
})

test('an install that cannot edit is never told an edit is waiting', () => {
  // No onEditRef means the ✦ Edit button itself is not offered. A badge opening
  // a modal that does not exist would be a dead end.
  const html = render(ReferencePanel,
                      { ...base, onEditRef: undefined, referenceEdit: READY })
  assert.doesNotMatch(html, /is waiting/)
})

test('the card still renders its normal actions while announcing', () => {
  // The badge is additive: it must not displace Change/Crop/Edit.
  const html = render(ReferencePanel, { ...base, referenceEdit: READY })
  assert.match(html, /Change<\/button>|Change reference/)
  assert.match(html, /Crop/)
  assert.match(html, /Edit/)
})
