/* SeedVR2 settings — every dial reaches the user, or it does not exist.
 *
 * Requested by SurpassHR (GitHub #32) alongside the engine: "DiT/VAE model
 * locations, target resolution, batch size, etc.". A setting that lives in
 * config.py and nowhere else is invisible; one with no Guide entry is
 * undocumented; one with no help topic cannot be found by search. This test
 * pins the four surfaces together so the next dial cannot ship half-wired.
 *
 * The dial set shrank on 2026-08-16 when the lane moved to the shipped manual
 * workflow: the model/VAE pins, the TTP tiling trio and blocks-to-swap have no
 * node to feed any more (the pipeline reads `diffusion_models`/`vae` through
 * core loaders and tiles with the core tiled VAE), so they are gone from all
 * four surfaces — only `resolution` (now an upscale MULTIPLIER, 2x by default)
 * and `color_correction` remain as settings.
 *
 * node --test parses no JSX, so the card is read as TEXT — which is exactly the
 * granularity that matters here: the ids, the config keys and the reset targets.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import { helpTopics } from '../src/help/helpRegistry.js'

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8')
const CARD = read('../src/components/settings/EnginesSection.jsx')
const GUIDE = read('../../docs/guide/settings-reference.md')
const DEFAULTS = read('../../backend/app/config.py')
const HELPER = read('../../backend/app/services/seedvr2_helper.py')

// field → the DOM id of its control in the card.
const FIELDS = {
  resolution: 'seedvr2-resolution',
  color_correction: 'seedvr2-color',
}

test('every seedvr2 setting has a labelled control and a reset in the card', () => {
  for (const [field, domId] of Object.entries(FIELDS)) {
    assert.ok(CARD.includes(`id="${domId}"`), `${field}: no control id="${domId}"`)
    assert.ok(CARD.includes(`htmlFor="${domId}"`), `${field}: control has no <label>`)
    assert.ok(CARD.includes(`setField('seedvr2', '${field}'`),
      `${field}: the control writes to some other config key`)
    assert.ok(CARD.includes(`section="seedvr2" field="${field}"`),
      `${field}: no ResetToDefault — a dial you cannot undo is a trap`)
  }
})

test('the config defaults carry every field the card writes', () => {
  const block = DEFAULTS.match(/'seedvr2': \{([\s\S]*?)\n    \},/)
  assert.ok(block, "backend defaults have no 'seedvr2' block")
  for (const field of Object.keys(FIELDS)) {
    assert.match(block[1], new RegExp(`'${field}':`), `${field}: missing from DEFAULTS`)
  }
})

test('the settings are documented and findable in Help', () => {
  const topics = new Set(helpTopics.map((t) => t.id))
  assert.ok(topics.has('seedvr2.resolution'), 'no help topic for the multiplier')
  assert.ok(topics.has('seedvr2.files'), 'no help topic for the model files')
  for (const field of Object.keys(FIELDS)) {
    assert.ok(GUIDE.includes(`\`seedvr2.${field}\``),
      `seedvr2.${field}: absent from docs/guide/settings-reference.md`)
  }
})

test('the dead dials are gone from every surface, not half-removed', () => {
  // The TTP tiling trio, the pins and blocks-to-swap have no node to feed.
  for (const key of ['model', 'vae', 'max_resolution', 'tiling', 'tile_px',
                     'tile_threshold', 'blocks_to_swap']) {
    assert.doesNotMatch(DEFAULTS, new RegExp(`'seedvr2'[\\s\\S]{0,400}'${key}':`),
      `seedvr2.${key} still in backend defaults`)
  }
  for (const id of ['seedvr2-model', 'seedvr2-vae', 'seedvr2-tiling',
                    'seedvr2-tile-px', 'seedvr2-tile-threshold',
                    'seedvr2-max-resolution', 'seedvr2-swap']) {
    assert.ok(!CARD.includes(`id="${id}"`), `dead control id="${id}" still in the card`)
  }
  assert.doesNotMatch(HELPER, /TTP_NODE_CLASSES|tile_plan|choose_lane|full_frame_ceiling_mp/,
    'TTP lane machinery still in the helper')
})

test('the multiplier mirrors the backend clamps and the shipped workflow value', () => {
  assert.match(HELPER, /RESOLUTION_MIN, RESOLUTION_MAX = 1\.0, 4\.0/)
  assert.match(CARD, /const SEEDVR2_RESOLUTION_MIN = 1\.0/)
  assert.match(CARD, /const SEEDVR2_RESOLUTION_MAX = 4\.0/)
  // 2x is the value of the user's verified workflow, not a guess.
  assert.match(DEFAULTS, /'resolution': 2\.0/)
})

test('the batch-size refusal is still stated, not silently dropped', () => {
  // The one item of #32 deliberately NOT shipped: batch_size is a temporal
  // window. Refusing it is defensible; refusing it without saying so is not.
  assert.match(CARD, /No batch size here, on purpose/)
  assert.match(GUIDE, /There is no batch-size setting, on purpose/)
})
