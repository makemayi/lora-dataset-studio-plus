/**
 * THE RESTYLED TILE RENDERS, IN EVERY STATE THE RESTYLE INTRODUCED.
 *
 * The 2026-08-15 restyle moved three things on the dataset tile:
 *   · the kept/rejected/pending decision left the tile's BORDER for a corner
 *     DOT (data-status, always visible, never hover-gated);
 *   · selection left the ring for an outward glow (data-selected on the tile);
 *   · the caption became a THIN LIGHT GLASS capsule at the photo's bottom,
 *     mounted and hover-gated by the CSS opacity contract — never unloaded,
 *     because Chrome auto-translate rewrites text nodes and a React removal
 *     would crash (CLAUDE.md ▸ UI changes 5).
 *
 * A source-text test cannot tell a removed branch from a broken one — Settings
 * white-screened behind a green suite once already. Every branch below is
 * MOUNTED (mountJsx.mjs) and its rendered markup asserted.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { render } from './support/mountJsx.mjs'

const { default: DatasetGridItem } = await import(
  '../src/components/dataset/DatasetGridItem.jsx')

const IMG = {
  id: 7, filename: 'shot.png', status: 'keep', source: 'import',
  caption: 'a caption', variation_label: 'portrait',
}

const tile = (props) => render(DatasetGridItem, {
  img: IMG, datasetId: 3,
  onStatus: () => {}, onCaption: () => {}, onCrop: () => {}, onDelete: () => {},
  onView: () => {}, onToggleSelect: () => {}, onMirror: () => {},
  ...props,
})

/* ── 1. The state dot: mounted, named, never hover-gated ─────────────────── */

test('the kept/rejected/pending decision is a corner dot carrying data-status', () => {
  for (const [status, cls] of [['keep', 'bg-green-500'], ['reject', 'bg-red-500'],
    ['pending', 'bg-amber-400']]) {
    const html = tile({ img: { ...IMG, status } })
    assert.match(html, new RegExp(`data-status="${status}"`),
      `${status} tile must expose its decision on the dot`)
    assert.match(html, new RegExp(cls), `${status} dot must wear its colour`)
    assert.match(html, /aria-label="Kept"|aria-label="Rejected"|aria-label="Undecided"/,
      `${status} dot must carry an accessible name, not colour alone`)
  }
})

test('a failed tile shows its own placeholder and a red dot', () => {
  const html = tile({ img: { ...IMG, filename: null, status: 'failed', fail_reason: 'ComfyUI said no' } })
  assert.match(html, /data-status="failed"/)
  assert.match(html, /bg-red-600/)
  assert.match(html, /⚠ failed/)
  assert.match(html, /ComfyUI said no/)
})

/* ── 2. Selection: data-selected on the tile, glow in the class list ─────── */

test('selection is data-selected on the tile, not a ring', () => {
  assert.doesNotMatch(tile({}), /data-selected=""/)
  const html = tile({ selected: true })
  assert.match(html, /data-selected="true"/)
  assert.match(html, /TILE_SELECTED_GLOW|rgba\(79,70,229,0\.55\)/,
    'selected must wear the outward glow token')
  assert.doesNotMatch(html, /ring-2 ring-indigo-400/, 'the ring is gone, by decision')
})

/* ── 3. The caption capsule: mounted when a caption exists, CSS-gated ────── */

test('a kept tile with a caption mounts the capsule, hover-gated, never unloaded', () => {
  const html = tile({})
  assert.match(html, /Edit the caption/)
  // The capsule is gated by the shared hover contract, not rendered on hover:
  // the same CSS class the toolbar uses (opacity), so the text node stays put
  // for Chrome auto-translate.
  assert.match(html, /dataset-grid-item__actions[^"]*absolute inset-x-0 bottom-0/)
  assert.match(html, /text-content/)
  assert.match(html, /bg-white\/70/, 'the light glass capsule recipe')
})

test('no caption means no capsule — nothing to gate', () => {
  const html = tile({ img: { ...IMG, caption: '' } })
  assert.doesNotMatch(html, /Edit the caption/)
  assert.doesNotMatch(html, /SCRIM_BOTTOM/)
})

test('a rejected tile never renders the caption capsule (captions live on keeps)', () => {
  const html = tile({ img: { ...IMG, caption: 'nope', status: 'reject' } })
  assert.doesNotMatch(html, /Edit the caption/)
})

/* ── 4. Face scoring: badge present when scored, absent when not ─────────── */

test('a scored tile mounts the face badge with its letter grade', () => {
  const html = tile({ img: { ...IMG, face_state: 'scorable', face_score: 0.93 } })
  assert.match(html, /Resemblance to the reference face — A/)
})

test('an unscored tile mounts no face badge', () => {
  assert.doesNotMatch(tile({}), /Resemblance to the reference face/)
})

/* ── 5. The photo is the card ────────────────────────────────────────────── */

test('the tile is the photo: full-bleed surface token, no inner card', () => {
  const html = tile({})
  assert.doesNotMatch(html, /rounded-\[28px\]/, 'the white card frame is gone')
  assert.doesNotMatch(html, /bg-black m-2/, 'no inner margin box')
  assert.match(html, /relative aspect-square bg-black/, 'the photo well remains square')
})
