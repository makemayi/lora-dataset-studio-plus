/**
 * The two blocks every page now shares — the title block and the "nothing here
 * yet" block — rendered.
 *
 * They were extracted on 2026-08-10 from five pages that had each grown their
 * own version. A shared component is only worth it if every caller gets the
 * same thing, so this pins the parts that make them one: the eyebrow/title/
 * actions/description slots, and the fact that an empty state says what to do
 * rather than only that there is nothing.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement as h } from './support/mountJsx.mjs'

const { default: PageHeader } = await import('../src/components/common/PageHeader.jsx')
const { default: EmptyState } = await import('../src/components/common/EmptyState.jsx')

test('the page header renders every slot it is given', () => {
  const html = renderToStaticMarkup(h(PageHeader, {
    eyebrow: 'library',
    title: 'Datasets',
    badge: h('span', null, '19'),
    actions: h('button', { type: 'button' }, '+ New dataset'),
    description: h('p', null, 'What this page is for'),
  }))
  assert.match(html, /library/)
  assert.match(html, /<h1[^>]*>Datasets/)
  assert.match(html, /\+ New dataset/)
  assert.match(html, /What this page is for/)
})

test('the title is the biggest thing in the header, the eyebrow the smallest', () => {
  const html = renderToStaticMarkup(h(PageHeader, { eyebrow: 'bank', title: 'Image bank' }))
  const eyebrow = html.match(/<p[^>]*>bank<\/p>/)?.[0]
  const title = html.match(/<h1[^>]*>/)?.[0]
  assert.match(eyebrow, /text-\[11px\]/)
  assert.match(eyebrow, /tracking-\[0\.18em\]/)
  assert.match(title, /text-2xl/)
  assert.match(title, /tracking-tight/)
})

test('a header with no actions and no description renders neither', () => {
  const html = renderToStaticMarkup(h(PageHeader, { eyebrow: 'board', title: 'LoRA Canvas' }))
  assert.match(html, /LoRA Canvas/)
  assert.ok(!html.includes('ml-auto'), 'an empty actions slot must not draw its row')
})

test('an empty state carries a title, a reason and the way out', () => {
  const html = renderToStaticMarkup(h(EmptyState, {
    icon: h('span', null, '·'),
    title: 'No bank yet',
    action: h('button', { type: 'button' }, 'Create your first bank'),
  }, 'A bank points at a folder you already have.'))
  assert.match(html, /No bank yet/)
  assert.match(html, /A bank points at a folder/)
  assert.match(html, /Create your first bank/)
  // Dashed, not a card: it marks a slot that is empty rather than a surface
  // that holds something (see the component's own note).
  assert.match(html, /border-dashed/)
})

test('an empty state with nothing to click still renders', () => {
  const html = renderToStaticMarkup(h(EmptyState, { title: 'No trained LoRA yet' }, 'Train one first.'))
  assert.match(html, /No trained LoRA yet/)
  assert.match(html, /Train one first\./)
})
