/**
 * HelpText — the fold that keeps Settings readable.
 *
 * Settings carries thousands of characters of explanatory prose and it is the
 * most valuable thing about the app: it is why a setting is understandable at
 * all. It is also why the pages read as walls of text, so anything longer than
 * a sentence folds behind a "?" disclosure and anything shorter stays put.
 *
 * The regression this file exists for: the first version measured
 * `children.length`, which is the STRING length only when the help is a bare
 * string. Most field help is not — it carries <span className="font-medium">
 * emphasis, a <code>, a link — so `children` is an array of three and a
 * 500-character paragraph measured as "short" and rendered fully open. Thirteen
 * card blurbs folded, thirty field blurbs did not, and the pages that needed it
 * most (Captioning & quality) barely changed. Measure the tree, not the array.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'

import { render, createElement as h } from './support/mountJsx.mjs'
// The JSX loader is registered while mountJsx is evaluated, so this import has
// to be dynamic — a static one would be resolved before the hook exists.
const { HelpText } = await import('../src/components/settings/primitives.jsx')

const SHORT = 'Residual grain over this = noisy.'
const LONG = 'x'.repeat(200)

test('a sentence renders open, with no disclosure to click', () => {
  const html = render(HelpText, { children: SHORT })
  assert.match(html, /^<p /)
  assert.ok(html.includes(SHORT))
  assert.ok(!html.includes('<details'), 'a one-liner behind a click costs a click and saves nothing')
})

test('a paragraph folds, and its full text is still in the document', () => {
  const html = render(HelpText, { children: LONG })
  assert.match(html, /^<details/)
  assert.ok(html.includes('More'))
  /* Not a round "?" — help/HelpMode.jsx owns that glyph for "jump to the
     Guide", and the two badges can land on the same row. */
  assert.ok(!/rounded-full/.test(html), 'the ? badge means something else in this app')
  assert.ok(html.includes(LONG), 'folded, not truncated — the prose is the product')
})

/* The bug. `children` here is an array, so `children.length` is 3. */
test('a long paragraph made of JSX folds too', () => {
  const children = [
    h('span', { key: 'a', className: 'font-medium' }, 'Applies from now on:'),
    ' changing it mid-way can leave a dataset holding mixed formats or sizes. ',
    h('code', { key: 'b' }, 'ai-toolkit'),
    ' buckets and downscales on its own, so that is harmless for training — it only means the folder is no longer uniform.',
  ]
  const html = render(HelpText, { children })
  assert.match(html, /^<details/, 'measured the array length (3), not the prose')
  assert.ok(html.includes('no longer uniform'))
})

test('a short JSX help stays open', () => {
  const children = [h('span', { key: 'a', className: 'font-medium' }, 'Note:'), ' applies at the next scan.']
  assert.match(render(HelpText, { children }), /^<p /)
})

test('nothing renders for empty help', () => {
  assert.equal(render(HelpText, { children: null }), '')
  assert.equal(render(HelpText, { children: undefined }), '')
})

/* The caller's margin positions the help under its input. It has to travel to
   whichever element is actually in the flow, or a folded blurb loses its offset
   and the <details> body inherits a margin meant for the block above it. */
test('the caller margin lands on the outer element in both states', () => {
  const open = render(HelpText, { children: SHORT, className: 'mt-0.5 text-xs text-content-muted' })
  assert.match(open, /^<p class="[^"]*mt-0\.5/)
  assert.match(open, /^<p class="[^"]*text-content-muted/)

  const folded = render(HelpText, { children: LONG, className: 'mt-3 text-xs text-content-muted' })
  assert.match(folded, /^<details class="[^"]*mt-3/)
  assert.ok(!/<p class="[^"]*mt-3/.test(folded), 'the margin must not be duplicated onto the body')
  assert.match(folded, /<p class="mt-1\.5 text-xs text-content-muted"/)
})

test('the summary is overridable — Card uses the long form', () => {
  assert.ok(render(HelpText, { children: LONG, summary: 'Why this matters' }).includes('Why this matters'))
})

/* SectionHeader — see the comment on the component. Keyed on its own text so a
   route change REMOUNTS the heading instead of writing into a text node Chrome
   auto-translate may have replaced. Without this the Server page rendered
   "Image engines" over Storage's description. */
const { SectionHeader } = await import('../src/components/settings/primitives.jsx')

test('the section heading is keyed on its text, so a route change remounts it', async () => {
  const src = await readFile(
    new URL('../src/components/settings/primitives.jsx', import.meta.url), 'utf8')
  const header = src.slice(src.indexOf('export function SectionHeader'))
  for (const [el, k] of [['<p key={eyebrow}', 'eyebrow'], ['<h1 key={title}', 'title'],
    ['<p key={description}', 'description']]) {
    assert.ok(header.includes(el), `${k} must be keyed — a text update would go to a detached node`)
  }
})

test('the heading renders both states it is asked for', () => {
  const a = render(SectionHeader, { eyebrow: 'STORAGE', title: 'Storage', description: 'Where it lives.' })
  assert.ok(a.includes('Storage') && a.includes('Where it lives.'))
  const b = render(SectionHeader, { eyebrow: 'SERVER', title: 'Server', description: null })
  assert.ok(b.includes('Server'))
  assert.ok(!b.includes('text-sm text-content-muted'), 'no empty description paragraph')
})
