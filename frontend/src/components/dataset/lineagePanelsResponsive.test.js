/* The two drawers the lineage graph opens must be readable on a 400-px phone.
   Both are BOTTOM SHEETS below `lg` and side drawers from `lg` up. The detail
   panel used to be `w-80` at every width, which on a 400-px screen covered 80%
   of the very board it annotates — you could read a run's settings but no
   longer see the run. Pinned here because it is invisible to every other test:
   a rewrite of the panel would silently drop the fix (a class is not behaviour,
   nothing throws), and this is the panel opened most often on a phone.

   The switch used to happen at `sm` (640 px). Measured on the real board, that
   was one breakpoint too early: these drawers are a FIXED width, so a 768-px
   window — what a phone in landscape reports — got a 320/352-px drawer over a
   ~400-px sliver of board, which is the same failure the sheet exists to avoid,
   just less extreme. A side drawer only earns its keep once what remains is
   still a readable board, i.e. from `lg` (1024 px). Nothing at or above 1024 px
   changed. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const detail = fs.readFileSync(new URL('./LineageDetailPanel.jsx', import.meta.url), 'utf8');
const gallery = fs.readFileSync(
  new URL('../shared/CheckpointGalleryPanel.jsx', import.meta.url), 'utf8');
const diff = fs.readFileSync(new URL('./LineageDiffPanel.jsx', import.meta.url), 'utf8');
const canvasPanel = fs.readFileSync(
  new URL('../canvas/CanvasGenerationPanel.jsx', import.meta.url), 'utf8');

for (const [name, src] of [['LineageDetailPanel', detail],
  ['CheckpointGalleryPanel', gallery]]) {
  test(`${name} is a bottom sheet up to lg and a side drawer from lg up`, () => {
    // Phone AND tablet: full width, anchored to the bottom, never taller than
    // 70vh so the graph it describes stays on screen above it.
    assert.match(src, /fixed inset-x-0 bottom-0/);
    assert.match(src, /max-h-\[70vh\]/);
    // From `lg`: back to the original side drawer, fixed width, full height.
    assert.match(src, /lg:inset-x-auto/);
    assert.match(src, /lg:h-full/);
    assert.match(src, /lg:max-h-none/);
    assert.match(src, /lg:w-(80|\[22rem\])/);
    // No unconditional width — that is exactly the bug.
    assert.doesNotMatch(src, /className="fixed[^"]*\sw-80/);
    // …and no leftover `sm:` half, which would fire 384 px early and undo it.
    assert.doesNotMatch(src, /sm:inset-x-auto/);
  });
}

test('the canvas generation panel is a bottom sheet up to lg', () => {
  // The 🎨 panel is a FIXED 26 rem side drawer. At 768 px that is 54 % of the
  // window, over the board it is picking checkpoints from.
  assert.match(canvasPanel, /fixed inset-x-0 bottom-0/);
  assert.match(canvasPanel, /lg:inset-x-auto/);
  assert.match(canvasPanel, /lg:w-\[26rem\]/);
  assert.doesNotMatch(canvasPanel, /sm:inset-x-auto/);
});

test('the compare drawer stays full width up to lg', () => {
  // Two columns of run settings inside 384 px next to a sliver of board is
  // neither a comparison nor a board.
  assert.match(diff, /fixed right-0 top-0[^"]*\sw-full[^"]*lg:w-96/);
  assert.doesNotMatch(diff, /sm:w-96/);
});

/* ✕ The way back to the board. Every one of these surfaces covers the thing the
   user came for, so its close button is the most-reached control on a phone —
   and all four shipped it as a bare 14-px glyph. 44 px up to `lg`, the size the
   desktop had is kept from `lg`. */
for (const [name, src] of [['LineageDetailPanel', detail],
  ['CheckpointGalleryPanel', gallery], ['LineageDiffPanel', diff],
  ['CanvasGenerationPanel', canvasPanel]]) {
  test(`${name} closes with a thumb-sized target on a phone`, () => {
    const close = src.slice(src.indexOf('aria-label="Close"') - 700,
      src.indexOf('aria-label="Close"') + 700);
    assert.match(close, /h-11 w-11/);
    assert.match(close, /lg:h-8 lg:w-8/);
  });
}

/* 🗑 The gallery deletes, and a phone grid is scrolled with the same finger that
   would tap a tile. So deletion must NEVER be one tap away: it takes Select mode,
   a pick, then a confirmation. Pinned as text because node --test cannot parse
   JSX and because a rewrite of the panel would silently drop the guard — the
   worst possible thing to lose on a destructive control. */
test('the checkpoint gallery cannot delete on an accidental tap', () => {
  // A tile only deletes/selects inside `picking`; outside it, a tap zooms.
  assert.match(gallery, /picking\s*\n?\s*\?\s*setSelected\(\(cur\) => toggleGalleryImage/);
  // The 🗑 button lives in the Select-mode half of the action bar — the bar
  // itself is permanent, its destructive half is not…
  assert.match(gallery, /\{bar\.showsDelete && \(/);
  assert.match(gallery, /disabled=\{bar\.deleteDisabled\}/);
  // …and it opens a confirmation rather than firing the request.
  assert.match(gallery, /onClick=\{\(\) => setConfirming\(true\)\}/);
  // Cancel is the focused default in that confirmation.
  assert.match(gallery, /autoFocus onClick=\{\(\) => setConfirming\(false\)\}/);
  // The delete request itself never leaves the confirmation.
  assert.match(gallery, /data-testid="gallery-confirm-delete"[\s\S]{0,120}onClick=\{runDelete\}/);
});

/* 📌 An action nobody finds is an action that does not exist.
   Pin-to-canvas shipped INSIDE the viewer: you had to open an image to discover
   the board could hold it. The toolbar hint mentioned it, last in a seven-clause
   line that `lg:hidden` removes entirely below a laptop. So the tile carries it
   now — and that is only safe as long as it stays out of Select mode, which no
   screenshot can show. */
test('the run gallery offers 📌 on the thumbnail itself, never while arming a delete', () => {
  // The visibility RULE is the tested helper, not an inline `&&` a rewrite loosens.
  assert.match(gallery, /galleryTilePin\(\{ picking, canPin: typeof onPin === 'function' \}\)/);
  assert.match(gallery, /data-testid="gallery-tile-pin"/);
  assert.match(gallery, /\{showPin && \(/);
  // A button cannot nest in a button: the tile is wrapped, and the wrapper takes
  // no handler of its own (the image target must not grow or shrink).
  assert.match(gallery, /<div key=\{img\.id\} className="relative">/);
  // Bottom-right, because top-right is the 👍/👎 verdict — two thumb targets in
  // one corner is how you reject an image you meant to pin.
  assert.match(gallery, /data-testid="gallery-tile-pin"[\s\S]{0,600}absolute bottom-1 right-1/);
  assert.match(gallery, /aria-label="Pin this image to the canvas"/);
  // The badges are decoration; they must not eat the taps aimed at the tile.
  assert.match(gallery, /pointer-events-none absolute right-0\.5 top-0\.5/);
  // The viewer keeps its own, larger, labelled button — this is an addition.
  assert.match(gallery, /data-testid="gallery-pin-image"/);
});

/* 👍 One reachable place for the whole gesture. Select used to sit in the header,
   two thumb-lengths from the Select all / Delete it leads to; on a phone that is
   the most expensive reach in the panel. It now opens the SAME pinned bar, which
   is why the reach-distance argument below is inverted: what must stay far apart
   is no longer Select and the grid, but Select and Delete. */
test('Select rides the pinned bottom bar, kept apart from Delete', () => {
  // Not in the header any more — the header keeps the title and the ✕ only.
  const header = gallery.slice(gallery.indexOf('<header'), gallery.indexOf('</header>'));
  assert.doesNotMatch(header, /gallery-select-toggle/);
  // It is inside the pinned bar, which is itself gated on having images: an
  // empty gallery carries no bar, so no destructive control and no dead gate.
  const bar = gallery.slice(gallery.indexOf('data-testid="gallery-action-bar"'),
    gallery.indexOf('</aside>'));
  assert.match(bar, /gallery-select-toggle/);
  assert.match(bar, /gallery-delete/);
  assert.match(gallery, /\{bar\.shown && \(/);
  assert.match(gallery, /galleryActionBar\(\{/);
  // Select first, Delete last and pushed to the far edge: a thumb cannot slide
  // from the gate straight onto the destructive button.
  assert.ok(bar.indexOf('gallery-select-toggle') < bar.indexOf('gallery-delete'));
  assert.match(bar, /data-testid="gallery-delete"[\s\S]{0,600}ml-auto/);
  // The label carries the state, not just the colour — plus aria-pressed.
  assert.match(bar, /\{bar\.toggleLabel\}/);
  assert.match(bar, /aria-pressed=\{bar\.togglePressed\}/);
  assert.match(bar, /aria-label=\{picking/);
  // "A bit more visible": the resting gate is indigo, the app's accent, not the
  // muted hairline it used to be.
  assert.match(bar, /gallery-select-toggle[\s\S]{0,600}indigo/);
  // It stays a real button, and the bar still wraps rather than scrolling at 400 px.
  assert.match(bar, /flex-wrap/);
});

test('the checkpoint gallery lives in shared/, where both surfaces import it from', () => {
  // It is opened by the canvas board AND by the in-card run graph; sitting in
  // components/canvas/ made the dataset panel import a "canvas" component.
  const canvas = fs.readFileSync(
    new URL('../canvas/LineageCanvas.jsx', import.meta.url), 'utf8');
  const graph = fs.readFileSync(new URL('./RunLineageGraph.jsx', import.meta.url), 'utf8');
  assert.match(canvas, /from '\.\.\/shared\/CheckpointGalleryPanel'/);
  assert.match(graph, /from '\.\.\/shared\/CheckpointGalleryPanel'/);
  assert.ok(!fs.existsSync(new URL('../canvas/CheckpointGalleryPanel.jsx', import.meta.url)));
});
