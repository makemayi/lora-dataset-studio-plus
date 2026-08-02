import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { canRegenerateGeneric } from './improveRerun.js';

const lightbox = readFileSync(new URL('./DatasetLightbox.jsx', import.meta.url), 'utf8');
const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');
const hook = readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');
const grid = readFileSync(new URL('./DatasetGrid.jsx', import.meta.url), 'utf8');
const settings = readFileSync(new URL('../settings/ScrapingSection.jsx', import.meta.url), 'utf8');
const attribution = readFileSync(new URL('./PexelsAttribution.jsx', import.meta.url), 'utf8');

test('lightbox exposes an accessible responsive image improvement action', () => {
  assert.match(lightbox, /✨ Upscale & improve/);
  assert.match(lightbox, /✨ Improving…/);
  assert.match(lightbox, /Review improvement first/);
  assert.match(lightbox, /aria-busy=\{improvementActive\}/);
  assert.match(lightbox, /w-full sm:w-auto/);
  assert.match(lightbox, /Klein creates a new 2 MP version to validate and leaves the original intact/);
  assert.match(lightbox, /busy \|\| improvementActive \|\| improveReady \|\| !kleinAvailable/);
});

// The comparison is what makes an improvement judgeable: before this, the
// lightbox showed the RESULT alone and the original had to be remembered.
// node --test cannot render JSX, so the contract is asserted on the source.
test('a derived image can be inspected next to the original it came from', () => {
  // Both panes live in ONE grid whose cells are equal, and both images are
  // object-contain: identical box, identical scale — the only honest way to
  // compare an upscale against its source (Klein rescales to a pixel budget and
  // keeps the aspect ratio, so equal boxes means identical framing).
  assert.match(lightbox, /grid-rows-2 grid-cols-1 sm:grid-rows-1 sm:grid-cols-2/);
  // h-full w-full, never max-h/max-w: an <img> at its intrinsic size is capped
  // but never scaled UP, so a small original rendered smaller than the result
  // that filled its pane — two different scales, the exact dishonesty this mode
  // exists to remove. Caught in a headless capture, pinned here.
  assert.match(lightbox, /className="h-full w-full select-none object-contain"/);
  assert.doesNotMatch(lightbox, /max-h-full max-w-full select-none object-contain/);
  // Real button, pressed state carried by aria (not colour alone).
  assert.match(lightbox, /aria-pressed=\{comparing\}/);
  assert.match(lightbox, /Compare with original/);
  assert.match(lightbox, /Exit comparison/);
  // Full-width control at phone width, like the other lightbox actions.
  assert.match(lightbox, /w-full sm:w-auto[^]{0,400}Compare with original/);
  // Each pane names its side in TEXT.
  assert.match(lightbox, /compare\.beforeLabel/);
  assert.match(lightbox, /compare\.afterLabel/);
  // Zoom is not silently broken: comparison says, in the same hint slot, that
  // 100 % lives outside the comparison.
  assert.match(lightbox, /exit comparison to zoom/i);
  // A vanished original explains itself instead of leaving a dead button.
  assert.match(lightbox, /compare && !compare\.available/);
  assert.match(lightbox, /\{compare\.reason\}/);
});

test('workspace feeds the lightbox the resolved parent of a derived image', () => {
  assert.match(workspace, /describeDerivedComparison/);
  assert.match(workspace, /compare=\{viewImgComparison\}/);
  assert.match(workspace, /parentNonce=/);
});

test('workspace guards rescue rows and detects a pending improvement child', () => {
  assert.match(workspace, /!viewImgLive\._rescueReviewPreview/);
  assert.match(workspace, /!isSmallImageRescueRow\(viewImgLive\)/);
  assert.match(workspace, /viewImgLive\.derivation_kind !== 'klein_image_improve'/);
  assert.match(workspace, /image\.derivation_kind === 'klein_image_improve'/);
  assert.match(workspace, /image\.parent_image_id === viewImgLive\.id/);
  assert.match(workspace, /const viewImgImproving[\s\S]*image\.status === 'pending'[\s\S]*\)\) : false/);
  assert.match(workspace, /const viewImgImprovementReady[\s\S]*image\.status === 'pending'[\s\S]*!!image\.filename/);
  assert.match(workspace, /kleinAvailable=\{Boolean\(caps\.engines\?\.klein\)\}/);
});

test('dataset hook starts improvement, reports the preserved original, then refreshes', () => {
  assert.match(hook, /`\/api\/dataset\/image\/\$\{imageId\}\/improve`, \{\}/);
  assert.match(hook, /original stays intact while a separate 2 MP candidate is generated for validation/);
  assert.match(hook, /Could not start image improvement/);
  assert.match(hook, /resolveSmallImageRescue, improveImage, reimproveImage, improveBatch, classify/);
});

test('the bulk improvement is ONE call that starts a server job, not a per-image loop', () => {
  assert.match(hook, /`\/api\/dataset\/\$\{currentId\}\/improve\/batch`,/);
  assert.match(hook, /\{ image_ids: ids, engine \}/);
  // The engine the user pressed rides along; absent, the server falls back to the
  // improve.engine setting (which is what the single-tile pass uses).
  assert.match(grid, /onImproveBatch\(eligible\.map\(\(image\) => image\.id\), engineId\)/);
  // No client-side sequential driver survives: that loop is what walked into the
  // fan-out cap and made ⏹ Stop powerless.
  assert.doesNotMatch(grid, /runSequentialKleinImprove/);
  // Progress is read from the server activity, so it survives a reload.
  assert.match(grid, /improveBatchLabel\(activity\)/);
  // ⏹ Stop generation stays reachable (and enabled) for a running batch. The
  // enabled-ness itself is decided by isStopGenerationBlocked (unit-tested in
  // scraperState.test.js) — this earlier inline expression only exempted
  // 'improve', which left the button dead for every plain generation batch.
  assert.match(workspace, /pending > 0 \|\| act\?\.kind === 'improve'/);
  assert.match(workspace, /disabled=\{isStopGenerationBlocked\(\{[\s\S]{0,120}?busy: ds\.busy, activity: act/);
});

test('settings separates scraper rescue instructions from manual lightbox improvement', () => {
  assert.match(settings, /title="Klein rescue — small scraped images"/);
  assert.match(settings, /automatic rescue of scraped images under 768 px/);
  // The point is that the two flows are distinct and the manual one is elsewhere —
  // asserted on that meaning, not on a fixed sentence. The old wording claimed the
  // manual pass had a fixed profile, which stopped being true once its strength and
  // step count became editable.
  assert.match(settings, /manual Upscale & improve is a different flow/);
  assert.match(settings, /Settings ▸ Image engines/);
  // the rescue card points at the separate manual "Identity & Klein prompts" card
  assert.match(settings, /separate from the manual .Klein upscale &amp; improve. prompt/i);
});

test('manual improvement candidates cannot use the unrelated generic regenerate path', () => {
  const gridItem = readFileSync(new URL('./DatasetGridItem.jsx', import.meta.url), 'utf8');
  // The guard moved into improveRerun.js (testable in node --test, which cannot
  // parse JSX) when the tile gained its own 🔄✨ re-run of the improve pass. Same
  // meaning, asserted at both ends: the tile delegates, and the decision refuses.
  assert.match(gridItem, /const isImageImproveCandidate = isImageImproveRow\(img\)/);
  assert.match(gridItem, /const canRegenerate = canRegenerateGeneric\(img, \{ isRescueDerived \}\)/);
  assert.match(gridItem, /if \(!isImageImproveCandidate && img\.status !== 'reject'/);
  assert.equal(canRegenerateGeneric({ source: 'generated', filename: 'a.png', status: 'keep',
    derivation_kind: 'klein_image_improve', parent_image_id: 2 }), false);
});

test('curation grid and lightbox render the persisted safe Pexels attribution', () => {
  const gridItem = readFileSync(new URL('./DatasetGridItem.jsx', import.meta.url), 'utf8');
  assert.match(gridItem, /<PexelsAttribution metadata=\{img\.source_metadata\}/);
  assert.match(lightbox, /<PexelsAttribution metadata=\{img\.source_metadata\}/);
  assert.match(attribution, /Photo by\{' '\}/);
  assert.match(attribution, /rel="noopener noreferrer"/);
  assert.match(attribution, /attribution\.photographerUrl/);
  assert.match(attribution, /attribution\.sourceUrl/);
});
