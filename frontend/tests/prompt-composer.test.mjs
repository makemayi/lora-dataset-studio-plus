/**
 * The prompt composer: ONE shape for "type a prompt, then act on it".
 *
 * Two things are pinned here. First the component itself, MOUNTED — both of its
 * branches, because the send button is a whole branch that a source-text test
 * cannot tell from a removed one. Second the adoption: three surfaces used to
 * carry three different prompt boxes, and the point of the composer is that a
 * fourth one cannot quietly appear beside them.
 *
 * The last test is a regression guard, not a style rule: a class-rewriting pass
 * once matched `bg-surface` inside `bg-surface-raised` and left
 * `bg-surface shadow-[…]-raised` behind — a wrong token plus a dead class that
 * Tailwind never emits. Nineteen call sites carried it silently.
 */
import assert from 'node:assert/strict'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import test from 'node:test'

import { render } from './support/mountJsx.mjs'

// Dynamic: the .jsx loader hook only exists once mountJsx has been EVALUATED,
// and a static import of a .jsx sibling is resolved before that happens.
const { default: PromptComposer, SEND_BUTTON } = await import(
  '../src/components/common/PromptComposer.jsx')

const SRC = fileURLToPath(new URL('../src/', import.meta.url))
const read = (rel) => readFileSync(new URL(`../src/${rel}`, import.meta.url), 'utf8')

test('the composer renders its send branch: a round gradient button, disabled when empty', () => {
  const html = render(PromptComposer, {
    value: '', onChange: () => {}, onSend: () => {}, sendDisabled: true,
    sendLabel: 'Add this shot', placeholder: 'describe a shot…',
  })
  assert.match(html, /aria-label="Add this shot"/)
  assert.match(html, /disabled=""/)
  assert.match(html, /data-icon="send"/)
  assert.match(html, /rounded-full bg-gradient-primary/)
})

test('and its no-send branch: tools, no button — the action lives elsewhere', () => {
  const html = render(PromptComposer, {
    value: 'a portrait', onChange: () => {}, ariaLabel: 'LoRA test prompt',
    tools: null,
  })
  assert.match(html, /aria-label="LoRA test prompt"/)
  assert.doesNotMatch(html, /data-icon="send"/)
  // No footer at all when there is neither a tool nor a send: an empty bar
  // would still take space under every prompt box that has no action.
  assert.doesNotMatch(html, /mt-1\.5 flex flex-wrap/)
})

test('the send button keeps the press-in feedback the brief asked for', () => {
  // 0.05s in (`active:duration-75` is the closest scale step), springy out.
  assert.match(SEND_BUTTON, /active:scale-95/)
  assert.match(SEND_BUTTON, /hover:scale-105/)
  assert.match(SEND_BUTTON, /disabled:opacity-40/)
})

test('every prompt-typing surface mounts the composer — no fourth prompt box', () => {
  const catalog = read('components/dataset/VariationCatalog.jsx')
  const field = read('components/dataset/studio/PromptField.jsx')
  const setup = read('components/dataset/studio/StudioRunSetup.jsx')
  for (const source of [catalog, field, setup]) {
    assert.match(source, /import PromptComposer from '[^']*PromptComposer\.jsx'/)
    assert.match(source, /<PromptComposer/)
    assert.doesNotMatch(source, /<textarea/)
  }
  // The catalog's Add action survived the reshape as the composer's send.
  assert.match(catalog, /onSend=\{addCustomShot\}/)
  assert.match(catalog, /sendDisabled=\{!customPrompt\.trim\(\)\}/)
})

test('no source carries the mangled `bg-surface …-raised` class', () => {
  const offenders = []
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const path = join(dir, name)
      if (statSync(path).isDirectory()) { walk(path); continue }
      if (!/\.(jsx?|mjs)$/.test(name)) continue
      if (/bg-surface\s+shadow-\[[^\]]*\]-raised/.test(readFileSync(path, 'utf8'))) offenders.push(path)
    }
  }
  walk(SRC)
  assert.deepEqual(offenders, [])
})
