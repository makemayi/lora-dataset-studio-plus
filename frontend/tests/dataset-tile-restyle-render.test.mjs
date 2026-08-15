/**
 * THE RESTYLED TILE RENDERS, IN EVERY STATE THE RESTYLE INTRODUCED.
 *
 * The 2026-08-15 restyle moved three things on the dataset tile:
 *   · the kept/rejected/pending decision left the tile's BORDER for a corner
 *     DOT (data-status, always visible, never hover-gated);
 *   · selection left the ring for an outward glow (data-selected on the tile);
 *   · the caption became a DARK glass capsule at the photo's bottom, mounted
 *     and hover-gated by its OWN CSS class (.dataset-grid-item__caption) —
 *     never unloaded, because Chrome auto-translate rewrites text nodes and a
 *     React removal would crash (CLAUDE.md ▸ UI changes 5).
 *
 * A source-text test cannot tell a removed branch from a broken one — Settings
 * white-screened behind a green suite once already. Every branch below is
 * MOUNTED (mountJsx.mjs) and its rendered markup asserted.
 *
 * The 2026-08-15 review killed three assertions here that could never fail:
 * a bare `bg-red-500` also matches the delete button's hover:bg-red-500/15; a
 * three-way aria-label alternation matches any state; and doesNotMatch on the
 * JS identifier SCRIM_BOTTOM can never see the rendered class name. Each is
 * now pinned to the element it means.
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
  // One dot, one colour, one accessible name. The three halves are asserted
  // INDEPENDENTLY on the data-status element (attribute order in the markup
  // is not guaranteed), and the colour is pinned to the dot's own tag so a
  // stray `hover:bg-red-500/15` elsewhere cannot satisfy it.
  for (const [status, cls, label] of [['keep', 'bg-green-500', 'Kept'],
    ['reject', 'bg-red-500', 'Rejected'], ['pending', 'bg-amber-400', 'Undecided']]) {
    const html = tile({ img: { ...IMG, status } })
    const dots = html.match(/<span data-status="[^"]*"[^>]*>/g) || []
    assert.ok(dots.length > 0, `${status} tile must render the state dot`)
    assert.ok(dots.some((d) => d.includes(`data-status="${status}"`)),
      `${status} tile must expose its decision on the dot`)
    assert.ok(dots.some((d) => d.includes(cls)),
      `${status} dot must wear its own colour (not a hover elsewhere)`)
    assert.ok(dots.some((d) => d.includes(`aria-label="${label}"`)),
      `${status} dot must carry its accessible name, not colour alone`)
  }
})

test('a failed tile shows its own placeholder and a red dot', () => {
  const html = tile({ img: { ...IMG, filename: null, status: 'failed', fail_reason: 'ComfyUI said no' } })
  assert.match(html, /data-status="failed"[^>]*bg-red-600/)
  assert.match(html, /⚠ failed/)
  assert.match(html, /ComfyUI said no/)
})

/* ── 2. Selection: data-selected on the tile, glow in the class list ─────── */

test('selection is data-selected on the tile, not a ring', () => {
  assert.doesNotMatch(tile({}), /data-selected=""/)
  const html = tile({ selected: true })
  assert.match(html, /data-selected="true"/)
  assert.match(html, /shadow-tile-selected/,
    'selected must wear the outward glow shadow, named after the config token')
  assert.doesNotMatch(html, /ring-2 ring-indigo-400/, 'the ring is gone, by decision')
})

/* ── 3. The caption capsule: mounted when a caption exists, CSS-gated ────── */

test('a kept tile with a caption mounts the capsule, hover-gated, never unloaded', () => {
  const html = tile({})
  assert.match(html, /Edit the caption/)
  // Gated by its OWN class (the caption is not an action — on touch screens
  // it must not resurrect the always-on bottom plate), positioned clear of
  // the checkbox (bottom-1 left-1) and the badges (bottom-right).
  assert.match(html, /dataset-grid-item__caption[^"]*absolute inset-x-2 bottom-14/)
  assert.match(html, /bg-black\/40/, 'the dark glass capsule recipe')
  assert.doesNotMatch(html, /bg-white\/70/, 'no light glass on this tile — one language')
})

test('no caption means no capsule — nothing to gate', () => {
  const html = tile({ img: { ...IMG, caption: '' } })
  assert.doesNotMatch(html, /Edit the caption/)
  assert.doesNotMatch(html, /dataset-grid-item__caption/)
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

test('a 3/4 profile shows its angle AND its score, not a silent not-scored', () => {
  // 2026-08-15: extreme_pose is scored (embedding stays discriminative to
  // ~70° yaw), shown as a number — never a letter grade, which is calibrated
  // on front-facing scores — and kept out of auto-triage (that filter only
  // reads face_state === 'scorable').
  const html = tile({ img: { ...IMG, face_state: 'extreme_pose',
    face_score: 0.61, face_yaw: 55 } })
  assert.match(html, /profile · 55° · 0\.61/, 'the tile names the angle and the score')
  assert.doesNotMatch(html, /profile — not scored/, 'the mute not-scored is gone')
})

test('a profile with no score still reads as profile, not a grade', () => {
  const html = tile({ img: { ...IMG, face_state: 'extreme_pose' } })
  assert.match(html, /profile/)
  assert.doesNotMatch(html, /Resemblance to the reference face — [A-D]/)
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

/* ── 6. The caption must stay clear of the tick (a reported click-stealer) ─ */

test('the caption capsule sits above the tick, never covering it', () => {
  const html = tile({})
  // Tick: bottom-1 left-1 (4–28px). Caption: bottom-14 (56px up) — disjoint.
  const tick = html.indexOf('Select portrait for bulk actions')
  const caption = html.indexOf('dataset-grid-item__caption')
  assert.ok(tick >= 0 && caption >= 0, 'both the tick and the caption render')
  assert.match(html, /dataset-grid-item__caption[^"]*bottom-14/)
})
