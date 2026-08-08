import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const read = (name) => readFileSync(new URL(`./${name}`, import.meta.url), 'utf8');
const studio = read('ComparisonStudio.jsx');
const panel = read('StackCompositionPanel.jsx');
const grid = read('StackVariantsGrid.jsx');

test('a stacked run swaps the per-LoRA ranking for its composition', () => {
  // The ranking compares LoRAs against each other; a stack has ONE tested LoRA, so
  // its ranking is a single meaningless line. The composition takes that slot.
  assert.match(studio, /const showStackView = isStackRun\(data\)/);
  assert.match(studio, /showStackView \? \(\s*<StackCompositionPanel/);
  assert.match(studio, /<LoraRankingPanel ranking=\{data\?\.lora_ranking\}/,
    'a comparison run keeps its ranking');
  // Decided by the RUN, never by the Compare/Blend toggle: an old stack must open
  // as a stack even when the toggle sits on Compare.
  assert.doesNotMatch(studio, /showStackView = combine/);
});

test('the stack results grid is columns-of-weight-variants, not columns-of-LoRA', () => {
  assert.match(studio, /showStackView \? \(\s*<StackVariantsGrid members=\{shownStack\} variants=\{data\?\.stack_variants\}/);
  assert.match(studio, /<LoraComparisonGrid loras=\{loras\}/, 'comparison runs keep the LoRA grid');
  assert.match(grid, /weightVectorText\(v\.weights\)/, 'a column is labelled by its weight vector');
  assert.match(grid, /members\.map\(\(m, i\) =>/, 'a row is a LoRA of the stack');
  assert.match(grid, /overflow-x-auto/, 'the wide table scrolls inside its own container');
});

test('every LoRA of the stack shows its weight and its trigger word', () => {
  assert.match(panel, /\{fmtWeight\(m\.weight\)\}/);
  assert.match(panel, /\{m\.trigger\}/);
  // Runs launched before the stack view never recorded the stacked triggers: say so
  // instead of rendering an empty chip that reads as "no trigger".
  assert.match(panel, /trigger not recorded/);
});

test('the ★ best setting of a stack is its weights, and hides when it cannot be built', () => {
  assert.match(panel, /const payload = bestStackPayload\(members\)/);
  assert.match(panel, /★ Save these weights as the best setting/);
  assert.match(panel, /payload \? \(/, 'no button when the composition is incomplete');
  assert.match(studio, /\/api\/dataset\/\$\{dsId\}\/lora-test\/best/);
});

test('a variant can be reopened and its weights reloaded into the sliders', () => {
  assert.match(grid, /onSelectRun\?\.\(v\.run_id\)/);
  assert.match(grid, /onUseWeights\?\.\(weightsIntoStackMap\(members, v\.weights\)\)/);
  assert.match(studio, /onSelectRun=\{setRunId\}/);
  assert.match(studio, /setStackWeights\(\(cur\) => \(\{ \.\.\.cur, \.\.\.map \}\)\)/);
  // Reloading weights only helps if the next run is a stack again.
  assert.match(studio, /setMode\('combine'\)/);
});

test('the lightbox can leaf through the tiles of every variant, not just the open run', () => {
  assert.match(studio, /const variantCells = \(data\?\.stack_variants \|\| \[\]\)\.flatMap/);
});
