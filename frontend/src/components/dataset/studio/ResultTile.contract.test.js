import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const tile = readFileSync(new URL('./ResultTile.jsx', import.meta.url), 'utf8');

test('a stalled Studio queue renders an accessible paused card instead of a spinner', () => {
  assert.match(tile, /cell\.queue_status === 'stalled'/);
  assert.match(tile, /cell\.status === 'pending' \|\| cell\.status === 'stalled'/,
    'the paused UI accepts the live pending-cell contract and an older stalled-cell payload');
  assert.match(tile, /const stalledReason = typeof cell\.queue_error === 'string'/);
  assert.match(tile, /\{isStalled && \(/);
  assert.match(tile, /role="status" aria-live="polite" aria-atomic="true"/);
  assert.match(tile, />paused</, 'the visible state is unambiguously paused');
  assert.match(tile, /\{stalledReason\}/, 'the paste-safe backend reason stays visible');
  assert.match(tile, /Recover or restart ComfyUI, then cancel and resume\./);
  assert.match(tile, /\{cell\.status === 'pending' && !isStalled && \(/,
    'the queued spinner path excludes stalled work');

  const pausedBlock = tile.match(/\{isStalled && \(([\s\S]*?)\n\s*\)\}/)?.[1];
  assert.ok(pausedBlock, 'the paused card must have its own conditional block');
  assert.doesNotMatch(pausedBlock, /animate-spin/, 'a paused queue must not look active');
  assert.doesNotMatch(pausedBlock, /\btitle=/,
    'the visible reason must not be redundantly repeated in a tooltip');
});
