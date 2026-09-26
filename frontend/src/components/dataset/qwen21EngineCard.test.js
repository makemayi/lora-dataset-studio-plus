/**
 * Qwen 2.1 — the fourth local engine card.
 *
 * MOUNTED where mounting is possible (the pure reason logic in
 * utils/qwen21Engine.js executes for real), and source-checked where it is
 * not (the card branch lives inside VariationCatalog, whose mount needs a
 * dataset payload, a jobs context and a capabilities provider that fetches —
 * the same honest split caption-engine-visibility documents).
 *
 * What can break silently, and what each test pins:
 * - a card with no accent renders `undefined` class names → accents complete;
 * - a reason branch that doesn't exist reads as "fine" on every screen →
 *   the localEngineUnavailableReason wiring is pinned by source;
 * - a missing asset key falls out of the sentence → labels complete.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'

import { ENGINES, LOCAL_ENGINES, ENGINE_ACCENTS, ENGINE_LABELS, ENGINE_RATES } from './engineSelection.js'

const read = (p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8')

test('every engine carries a full accent — no undefined class names', () => {
  for (const e of ENGINES) {
    const a = ENGINE_ACCENTS[e]
    assert.ok(a, `no accent for ${e}`)
    for (const key of ['card', 'title', 'text', 'icon', 'pill', 'dot', 'ring']) {
      assert.match(a[key], /^[a-z-]+-[a-z0-9]+/, `${e}.${key}`)
    }
  }
})

test('qwen21 is a local, free, editable engine in the canonical order', () => {
  assert.ok(ENGINES.indexOf('qwen21') > -1)
  assert.ok(LOCAL_ENGINES.includes('qwen21'))
  assert.equal(ENGINE_LABELS.qwen21, 'Qwen 2.1')
  assert.equal(ENGINE_RATES.qwen21, 0) // own GPU — free, like the other locals
})

test('the accent hue is not green (green means kept / free)', () => {
  assert.doesNotMatch(ENGINE_ACCENTS.qwen21.card, /green|emerald/)
})

test('the workspace card branch exists and reads its own reason', () => {
  const src = read('./VariationCatalog.jsx')
  assert.match(src, /EngineCard id="qwen21"/, 'the card is in the grid')
  assert.match(src, /enabledEngines\.includes\('qwen21'\)/, 'gated on Settings')
  assert.match(src, /qwen21UnavailableReason/, 'its hint names the real gap')
  assert.match(src, /Qwen21FrameIcon/, 'its own glyph, not another engine\'s')
})

test('localEngineReason routes qwen21 to its own capabilities key', () => {
  const src = read('../../utils/localEngineReason.js')
  assert.match(src, /import \{ qwen21UnavailableReason \}/)
  assert.match(src, /engine === 'qwen21'/)
  assert.match(src, /caps\?\.qwen21 \|\| \{\}/, 'reads caps.qwen21, like H3 does')
})

test('the reason logic executes: ordered asset words, node-class sentence', async () => {
  const { qwen21UnavailableReason, qwen21MissingLabels, QWEN21_ASSET_LABELS } =
    await import('../../utils/qwen21Engine.js')
  // Stable order regardless of the server's list order.
  assert.deepEqual(
    qwen21MissingLabels(['qwen21_vae', 'qwen21_model']),
    ['Qwen 2.1 base model', 'Qwen Image 2.1 VAE'])
  // Every capabilities key the backend publishes has a word.
  for (const k of ['qwen21_model', 'qwen21_text_encoder', 'qwen21_vae']) {
    assert.ok(QWEN21_ASSET_LABELS[k], k)
  }
  assert.equal(qwen21UnavailableReason({}), null)
  assert.match(
    qwen21UnavailableReason({ enabledInSettings: false }),
    /disabled in Settings/)
  assert.match(
    qwen21UnavailableReason({ comfyuiReachable: false }),
    /ComfyUI/i)
  assert.match(
    qwen21UnavailableReason({ missingNodes: ['TextEncodeQwenImage21'] }),
    /update ComfyUI/i)
  assert.match(
    qwen21UnavailableReason({ missingAssets: ['qwen21_vae'] }),
    /Qwen Image 2\.1 VAE missing/)
})
