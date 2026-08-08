import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  RAIL_WIDTH_PX,
  MIN_RAIL_VIEWPORT_PX,
  RAIL_EXIT_VIEWPORT_PX,
  PLACEMENT_HYSTERESIS,
  decideActionPlacement,
  rememberImageRatio,
  readImageRatio,
  _resetImageRatios,
} from './lightboxActionPlacement.js';

const PORTRAIT = { imageWidth: 832, imageHeight: 1216 };   // 0.684
const LANDSCAPE = { imageWidth: 1216, imageHeight: 832 };  // 1.462
const SQUARE = { imageWidth: 1024, imageHeight: 1024 };    // 1.000
const WIDE = { viewportWidth: 1440, viewportHeight: 900 };
const PHONE = { viewportWidth: 400, viewportHeight: 860 };

test('the reported case: a portrait image on a wide window puts the actions in a side rail', () => {
  assert.equal(decideActionPlacement({ ...WIDE, ...PORTRAIT }), 'rail');
});

test('a landscape image keeps the actions at the bottom — its scarce axis is width', () => {
  assert.equal(decideActionPlacement({ ...WIDE, ...LANDSCAPE }), 'bottom');
});

test('a phone-width window is always bottom, whatever the image shape', () => {
  assert.equal(decideActionPlacement({ ...PHONE, ...PORTRAIT }), 'bottom');
  assert.equal(decideActionPlacement({ ...PHONE, ...LANDSCAPE }), 'bottom');
  assert.equal(decideActionPlacement({ ...PHONE, ...SQUARE }), 'bottom');
});

test('a square image follows the window, not a portrait/landscape label', () => {
  // 1440x900: the leftover column (1168) still fits a square drawn at full
  // height (900), so the rail costs the image nothing and buys back the bar.
  assert.equal(decideActionPlacement({ ...WIDE, ...SQUARE }), 'rail');
  // 1280x1024: the leftover column (1008) is narrower than the full-height
  // square (1024) — a rail would shrink the image. Bottom.
  assert.equal(
    decideActionPlacement({ viewportWidth: 1280, viewportHeight: 1024, ...SQUARE }),
    'bottom',
  );
});

test('an unknown image size falls back to the bottom bar, never to a guess', () => {
  assert.equal(decideActionPlacement({ ...WIDE }), 'bottom');
  assert.equal(decideActionPlacement({ ...WIDE, imageWidth: 0, imageHeight: 0 }), 'bottom');
  assert.equal(decideActionPlacement({ ...WIDE, imageWidth: 832, imageHeight: undefined }), 'bottom');
});

test('an unknown viewport falls back to the bottom bar', () => {
  assert.equal(decideActionPlacement({ ...PORTRAIT }), 'bottom');
  assert.equal(
    decideActionPlacement({ viewportWidth: 0, viewportHeight: 0, ...PORTRAIT }),
    'bottom',
  );
});

test('side-by-side comparison forces the bottom bar — both panes want the width', () => {
  assert.equal(decideActionPlacement({ ...WIDE, ...PORTRAIT, comparing: true }), 'bottom');
  // …and it wins over a rail that is currently in place.
  assert.equal(
    decideActionPlacement({ ...WIDE, ...PORTRAIT, comparing: true, current: 'rail' }),
    'bottom',
  );
});

test('a locked decision is returned verbatim — buttons never move under a running action', () => {
  assert.equal(
    decideActionPlacement({ ...WIDE, ...LANDSCAPE, current: 'rail', locked: true }),
    'rail',
  );
  assert.equal(
    decideActionPlacement({ ...WIDE, ...PORTRAIT, current: 'bottom', locked: true }),
    'bottom',
  );
});

test('the hysteresis band keeps the decision still while the window is dragged', () => {
  // Find the width at which a growing window earns the rail…
  let current = 'bottom';
  let enteredAt = null;
  for (let w = 700; w <= 1600; w += 1) {
    const next = decideActionPlacement({
      viewportWidth: w, viewportHeight: 900, ...PORTRAIT, current,
    });
    if (next !== current && next === 'rail') enteredAt = w;
    current = next;
  }
  assert.ok(enteredAt, 'the rail must be reachable by widening the window');
  assert.equal(current, 'rail');

  // …then jitter around that exact width: a decision that flipped here would be
  // the "buttons jump while you aim at them" regression.
  for (const w of [enteredAt - 20, enteredAt + 20, enteredAt - 5, enteredAt, enteredAt + 40]) {
    current = decideActionPlacement({
      viewportWidth: w, viewportHeight: 900, ...PORTRAIT, current,
    });
    assert.equal(current, 'rail', `placement flipped at ${w}px (entered at ${enteredAt}px)`);
  }
});

test('a full grow-then-shrink sweep costs at most one move each way', () => {
  let current = 'bottom';
  let moves = 0;
  const widths = [];
  for (let w = 700; w <= 1600; w += 1) widths.push(w);
  for (let w = 1600; w >= 700; w -= 1) widths.push(w);
  for (const w of widths) {
    const next = decideActionPlacement({
      viewportWidth: w, viewportHeight: 900, ...PORTRAIT, current,
    });
    if (next !== current) moves += 1;
    current = next;
  }
  assert.equal(moves, 2, `the bar moved ${moves} times over one grow/shrink sweep`);
  assert.equal(current, 'bottom');
});

test('the viewport floor has its own dead band, so 1024px is not a flip-flop point', () => {
  const at = (viewportWidth, current) => decideActionPlacement({
    viewportWidth, viewportHeight: 900, ...PORTRAIT, current,
  });
  assert.equal(at(MIN_RAIL_VIEWPORT_PX, 'bottom'), 'rail');
  assert.equal(at(MIN_RAIL_VIEWPORT_PX - 1, 'bottom'), 'bottom');
  // Already railed: it takes a real shrink, not one pixel, to give it up.
  assert.equal(at(MIN_RAIL_VIEWPORT_PX - 1, 'rail'), 'rail');
  assert.equal(at(RAIL_EXIT_VIEWPORT_PX, 'rail'), 'rail');
  assert.equal(at(RAIL_EXIT_VIEWPORT_PX - 1, 'rail'), 'bottom');
});

test('the constants are the ones the layout is built from', () => {
  assert.equal(RAIL_WIDTH_PX, 272);            // w-[17rem]
  assert.equal(MIN_RAIL_VIEWPORT_PX, 1024);    // Tailwind lg
  assert.ok(RAIL_EXIT_VIEWPORT_PX < MIN_RAIL_VIEWPORT_PX);
  assert.ok(PLACEMENT_HYSTERESIS > 0 && PLACEMENT_HYSTERESIS < 0.5);
});

test('an image ratio, once measured, is remembered so reopening never re-flips', () => {
  _resetImageRatios();
  assert.equal(readImageRatio(7), null);
  rememberImageRatio(7, 832, 1216);
  assert.deepEqual(readImageRatio(7), { imageWidth: 832, imageHeight: 1216 });
  // A rotation changes the ratio of the same id — the last measure wins.
  rememberImageRatio(7, 1216, 832);
  assert.deepEqual(readImageRatio(7), { imageWidth: 1216, imageHeight: 832 });
});

test('a bogus measurement is not remembered — an unloaded <img> reports 0', () => {
  _resetImageRatios();
  rememberImageRatio(9, 0, 0);
  assert.equal(readImageRatio(9), null);
  rememberImageRatio(null, 832, 1216);
  assert.equal(readImageRatio(null), null);
});

test('a remembered portrait decides the rail before the image has painted', () => {
  _resetImageRatios();
  rememberImageRatio(3, 832, 1216);
  assert.equal(decideActionPlacement({ ...WIDE, ...readImageRatio(3) }), 'rail');
});

/* ── Source contracts ──────────────────────────────────────────────────────
 * `node --test` cannot parse JSX, so what the rule is WIRED to is pinned by
 * reading the file: a rule nobody applies, or applied by reordering the DOM,
 * would leave every test above green.
 */
const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const lightbox = read('./DatasetLightbox.jsx');
const gridItem = read('./DatasetGridItem.jsx');

test('the lightbox flips its axis from the rule, and only its axis', () => {
  assert.match(lightbox, /from '\.\/lightboxActionPlacement'/);
  assert.match(lightbox, /const rail = placement === 'rail'/);
  assert.match(lightbox, /flex \$\{rail \? 'flex-row' : 'flex-col'\}/);
  // The rail width in the markup IS the one the rule budgets for.
  assert.ok(lightbox.includes('w-[17rem]'));
  assert.equal(RAIL_WIDTH_PX, 17 * 16);
  // A short window must still be able to reach the last action.
  assert.match(lightbox, /flex w-\[17rem\] flex-col[^']*overflow-y-auto/);
  // …and the rail must clear the absolutely-positioned ✕ (top-3, h-9 = 48 px).
  assert.match(lightbox, /flex w-\[17rem\][^']*pt-14/);
  // The image cell must be allowed to shrink, or the rail leaves the viewport.
  assert.match(lightbox, /flex-1 min-h-0 min-w-0 flex items-center justify-center/);
});

test('tab order follows the eye: one DOM order, no CSS reordering', () => {
  const at = (needle) => {
    const i = lightbox.indexOf(needle);
    assert.ok(i > 0, `missing from the lightbox: ${needle}`);
    return i;
  };
  const order = [
    'const alt = displayLabel',      // the meta group is first in both modes
    '⧉ Compare with original',
    '✂ Crop',
    '⇆ Mirror horizontally',
    'Rotate left',
    'Rotate right',
    // The improve actions are now rendered from a list (one button per engine),
    // so the DOM-order anchor is the map that emits them rather than one label.
    'improveButtons.map',
    '<KleinImproveNote',
  ].map(at);
  for (let i = 1; i < order.length; i += 1) {
    assert.ok(order[i] > order[i - 1], 'the action DOM order must not change');
  }
  // No `order-*` / `flex-row-reverse` utility anywhere: those are exactly how a
  // rail ends up reading top-to-bottom on screen and elsewhere under Tab.
  assert.ok(!/className[^\n]*\border-\d/.test(lightbox));
  assert.ok(!lightbox.includes('flex-row-reverse'));
  assert.ok(!lightbox.includes('flex-col-reverse'));
});

test('the rail keeps words — these actions rotate, recrop and spend GPU time', () => {
  for (const label of ['✂ Crop', '⇆ Mirror horizontally', 'Rotate left',
    'Rotate right']) {
    assert.ok(lightbox.includes(label), `${label} must stay spelled out`);
  }
  // The improve labels are built in improveEngines.js ('✨ Improve via Klein',
  // '🔍 Upscale via SeedVR2'); that they stay spelled out is asserted there.
  assert.ok(lightbox.includes('{btn.label}'), 'the improve buttons must render their label');
  // No placement branch may swap a label for a bare glyph.
  assert.ok(!/rail \? '[↺↻✂⇆✨]'/.test(lightbox));
});

test('the decision is fed the lock and the comparison, not just the geometry', () => {
  assert.match(lightbox, /const actionsLocked = busy \|\| mirrorBusy \|\| improving \|\| improvePending/);
  assert.match(lightbox, /locked: actionsLocked/);
  // The lightbox now has TWO comparisons (against the original, against the
  // reference photo) held in one `compareMode`. This rule only cares that two
  // panes want the width, so it is fed the collapsed boolean — a mode name
  // leaking in here would make the placement rule grow a second reason to change.
  assert.match(lightbox, /comparing: compareMode !== 'none',/);
  // Resize is coalesced to one decision per frame.
  assert.match(lightbox, /requestAnimationFrame\(\(\) => \{ frame = 0; apply\(\); \}\)/);
  assert.match(lightbox, /removeEventListener\('resize', onResize\)/);
});

test('the intrinsic size is measured wherever the image already loads', () => {
  // Both zoom branches of the lightbox…
  assert.equal((lightbox.match(/onLoad=\{onImageLoad\}/g) || []).length, 2);
  // …and the grid tile, which requests the SAME url earlier, so a first open is
  // already decided instead of committing in front of the user.
  assert.match(gridItem, /from '\.\/lightboxActionPlacement\.js'/);
  assert.match(gridItem, /onLoad=\{\(e\) => rememberImageRatio\(/);
  // …and that remembered ratio decides the FIRST painted frame, rather than an
  // effect correcting a bar the user already saw somewhere else.
  assert.match(lightbox, /useState\(\(\) => decideActionPlacement\(\{/);
});

/* --- The opening decision, and the row that must stay a row ---------------
   A user reported, with screenshots: whatever the window size, opening the
   lightbox drew the actions as a bottom row with the SECOND improve button
   stranded alone and centred at the very bottom; resizing the window produced
   the correct right-hand rail. Two distinct defects behind one picture. */

test('opening carries no hysteresis — there is nothing to stabilise yet', () => {
  // The dead band exists so a bar cannot change side under the pointer while a
  // window edge is dragged. Applied to the FIRST answer it is not stability, it
  // is a bias towards the default ('bottom'). A geometry that qualifies for the
  // rail must therefore open in the rail.
  const geometry = { viewportWidth: 1200, viewportHeight: 1000,
    imageWidth: 900, imageHeight: 1000 };            // aspect 0.90
  const leftover = (1200 - 272) / 1000;              // 0.928
  assert.ok(0.9 <= leftover, 'the bare inequality must hold for this case');
  assert.ok(0.9 > leftover * (1 - PLACEMENT_HYSTERESIS),
    'and it must FAIL the entering dead band — otherwise this proves nothing');
  assert.equal(decideActionPlacement(geometry), 'rail',
    'the opening decision must read the geometry, not the default');
  // Once a placement is in force the dead band comes back, unchanged.
  assert.equal(decideActionPlacement({ ...geometry, current: 'bottom' }), 'bottom');
  assert.equal(decideActionPlacement({ ...geometry, current: 'rail' }), 'rail');
});

test('the viewport floor keeps its dead band only once a rail exists', () => {
  const g = { viewportWidth: 1000, viewportHeight: 1000, imageWidth: 600, imageHeight: 1000 };
  // 1000 px is below the entry floor: opening there is a bottom bar...
  assert.equal(decideActionPlacement(g), 'bottom');
  // ...but an EXISTING rail survives down to RAIL_EXIT_VIEWPORT_PX.
  assert.equal(decideActionPlacement({ ...g, current: 'rail' }), 'rail');
});

test('a small window still opens at the bottom, and 400 px always does', () => {
  const portrait = { imageWidth: 832, imageHeight: 1216 };
  assert.equal(decideActionPlacement({
    viewportWidth: 900, viewportHeight: 900, ...portrait }), 'bottom');
  assert.equal(decideActionPlacement({
    viewportWidth: 400, viewportHeight: 900, ...portrait }), 'bottom');
});

test('the bottom bar keeps the improve buttons on ONE row', () => {
  // THE reported break: the Klein note is full-width, and dropped between the
  // two buttons it pushes the second one onto its own line — "stranded alone,
  // centred, at the very bottom". It may sit between them in the rail (a
  // column, where that is what attaches it to Klein) and nowhere else.
  assert.match(lightbox, /\{rail && btn\.showKleinNote && !improvementActive && \(/);
  assert.match(lightbox, /\{!rail && improveButtons\.some\(\(b\) => b\.showKleinNote\)/);
  // Exactly two renders of the note, one per placement — never both at once.
  const notes = lightbox.match(/<KleinImproveNote\b/g) || [];
  assert.equal(notes.length, 2, 'one note per placement branch, no more');
});
