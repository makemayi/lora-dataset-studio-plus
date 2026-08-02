/**
 * Contract of the two Test Studio additions:
 *   - 🧬 Combine — several LoRAs loaded in the SAME image, each at its own weight;
 *   - ✨ Enhance — prompt enrichment through the LOCAL Ollama model.
 *
 * The pure logic (loraStack.js, enhanceGate.js) is unit-tested directly. The JSX
 * that consumes it is asserted on its SOURCE, because `node --test` cannot parse
 * JSX — the same convention as the other studio contract tests here.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import {
  buildSelectionsPayload, cellCount, combineBlocker, stackKey, stackWeight,
} from '../src/components/dataset/studio/loraStack.js'
import { enhanceBlocker } from '../src/components/dataset/studio/enhanceGate.js'
import { getHelpTopic } from '../src/help/helpRegistry.js'
import { WHATS_NEW } from '../src/whatsNew.js'

const readStudio = (name) => readFileSync(
  new URL(`../src/components/dataset/studio/${name}`, import.meta.url), 'utf8')

const SEL = [
  { dataset_id: 1, checkpoint: 'z image\\a.safetensors', lora_label: 'A', family: 'zimage' },
  { dataset_id: 2, checkpoint: 'z image\\b.safetensors', lora_label: 'B', family: 'zimage' },
]

test('a combined payload carries one weight per LoRA, a comparison payload carries none', () => {
  const weights = { [stackKey(SEL[0])]: 0.9, [stackKey(SEL[1])]: 0.55 }
  assert.deepEqual(buildSelectionsPayload(SEL, { combine: true, weights }), [
    { dataset_id: 1, checkpoint: 'z image\\a.safetensors', weight: 0.9 },
    { dataset_id: 2, checkpoint: 'z image\\b.safetensors', weight: 0.55 },
  ])
  // Comparison keeps the historical shape byte for byte — no `weight` key at all.
  for (const entry of buildSelectionsPayload(SEL, { combine: false })) {
    assert.deepEqual(Object.keys(entry).sort(), ['checkpoint', 'dataset_id'])
  }
})

test('a per-LoRA weight is clamped to 0..2 and defaults to 1', () => {
  assert.equal(stackWeight({}, SEL[0]), 1)
  assert.equal(stackWeight({ [stackKey(SEL[0])]: 5 }, SEL[0]), 2)
  assert.equal(stackWeight({ [stackKey(SEL[0])]: -1 }, SEL[0]), 0)
  assert.equal(stackWeight({ [stackKey(SEL[0])]: 0.5555 }, SEL[0]), 0.56)
  assert.equal(stackWeight({ [stackKey(SEL[0])]: 'x' }, SEL[0]), 1)
})

test('combining is blocked below two LoRAs and across families, with a named reason', () => {
  assert.match(combineBlocker([SEL[0]]), /at least two/i)
  assert.equal(combineBlocker(SEL), null)
  const mixed = [SEL[0], { ...SEL[1], family: 'krea' }]
  const why = combineBlocker(mixed)
  assert.match(why, /zimage/)
  assert.match(why, /krea/)
})

test('a combined stack costs one configuration, not one per LoRA and strength', () => {
  // Comparison: 2 LoRAs × 3 strengths × 2 images = 12 cells.
  assert.equal(cellCount({ selectionCount: 2, strengthCount: 3, count: 2 }), 12)
  // Combine: the strength axis is gone — 1 stack × 2 images.
  assert.equal(cellCount({ selectionCount: 2, strengthCount: 3, count: 2, combine: true }), 2)
  // The ⚖ batch axis still multiplies both modes.
  assert.equal(
    cellCount({ selectionCount: 2, strengthCount: 3, count: 2, batchMult: 2, combine: true }), 4)
  // A stack of one is not a stack: nothing to launch.
  assert.equal(cellCount({ selectionCount: 1, strengthCount: 3, count: 2, combine: true }), 0)
})

test('Enhance is disabled with the reason when Ollama is missing, running or unpulled', () => {
  assert.match(enhanceBlocker(null, { capsLoading: true }), /Checking/)
  assert.match(enhanceBlocker({}), /install it/i)
  assert.match(enhanceBlocker({ installed: true, reachable: false }), /not running/i)
  assert.match(
    enhanceBlocker({ installed: true, reachable: true, vision_model_ready: false, vision_model: 'qwen' }),
    /qwen.*not downloaded/is)
  assert.equal(
    enhanceBlocker({ installed: true, reachable: true, vision_model_ready: true }), null)
})

test('the studio sends `combine` instead of the strength axis when the stack is on', () => {
  const source = readStudio('ComparisonStudio.jsx')
  assert.match(source, /buildSelectionsPayload\(selection, \{ combine, weights: stackWeights \}\)/)
  assert.match(source, /\.\.\.\(combine \? \{ combine: true \} : \{ strengths \}\)/)
  // A blocked stack must never reach the network.
  assert.match(source, /if \(!selection\.length \|\| combineBlocked\) return/)
})

test('the run panel hides the strength sweep and both prompt helpers stay reachable', () => {
  const setup = readStudio('StudioRunSetup.jsx')
  assert.match(setup, /\{!combine && \(\s*<StrengthPicker/)
  assert.match(setup, /<EnhancePromptButton prompt=\{prompt\} onResult=\{onPrompt\} \/>/)
  // The solo (single-LoRA) branch gets Enhance too, on its own prompt field.
  assert.match(readStudio('PromptField.jsx'),
    /<EnhancePromptButton prompt=\{value\} onResult=\{onChange\} \/>/)
})

test('the Enhance button posts to the studio route and stays disabled while blocked', () => {
  const source = readStudio('EnhancePromptButton.jsx')
  assert.match(source, /'\/api\/studio\/enhance-prompt'/)
  assert.match(source, /disabled=\{!!blocked \|\| empty \|\| busy\}/)
  assert.match(source, /title=\{title\}/)
  // No second Ollama probe: the shared capabilities context is the only source.
  assert.match(source, /useCapabilities\(\)/)
  assert.doesNotMatch(source, /\/api\/ollama\//)
})

test('a combined tile says so, so a stack is never mistaken for a solo render', () => {
  const tile = readStudio('ResultTile.jsx')
  assert.match(tile, /cell\.combined_loras/)
  assert.match(tile, /Combined with:/)
})

test('both features are documented and announced', () => {
  for (const id of ['studio-combine-loras', 'studio-enhance-prompt']) {
    const topic = getHelpTopic(id)
    assert.ok(topic, `missing help topic ${id}`)
    assert.ok(topic.guide?.chapter && topic.guide?.anchor, `${id} needs a Guide anchor`)
  }
  for (const id of ['2026-08-01-studio-combine-loras', '2026-08-01-studio-enhance-prompt']) {
    assert.ok(WHATS_NEW.some((e) => e.id === id), `missing What's new entry ${id}`)
  }
})
