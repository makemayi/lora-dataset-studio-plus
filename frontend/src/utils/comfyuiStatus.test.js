/* "ComfyUI is slow" must never be rendered as "ComfyUI isn't running".

   j_o_e_l. (Discord) hit this on a ComfyUI that was up: the /object_info
   enumeration took ~15 s against an 8 s budget, and every local-engine card
   answered "⚠ Configure ComfyUI in Settings" — sending him to re-check the one
   thing that was already correct. These tests are the guard against the
   catch-all coming back. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { comfyuiAnswering, comfyuiDownReason } from './comfyuiStatus.js';
import { kreaUnavailableReason } from './kreaEngine.js';
import { localEngineUnavailableReason } from './localEngineReason.js';

test('an answering ComfyUI has no reason at all', () => {
  assert.equal(comfyuiDownReason({ reachable: true, status: 'ok' }), null);
  assert.equal(comfyuiAnswering({ reachable: true, status: 'ok' }), true);
});

test('slow and unreachable produce DIFFERENT sentences', () => {
  const slow = comfyuiDownReason({ reachable: false, status: 'slow', object_info_timeout_s: 45 });
  const down = comfyuiDownReason({ reachable: false, status: 'unreachable' });
  assert.notEqual(slow, down);
  // The slow one must name the delay and the knob, and must NOT claim it is off.
  assert.match(slow, /45/);
  assert.match(slow, /timeout/i);
  assert.doesNotMatch(slow, /configure comfyui/i);
  assert.match(slow, /running/i);
});

test('the server sentence wins when it sends one — one wording, everywhere', () => {
  // The blocked-run 409 and the engine card must not word the same gap twice.
  const r = comfyuiDownReason({ reachable: false, status: 'slow', hint: 'SERVER WORDS' });
  assert.equal(r, '⚠ SERVER WORDS');
});

test('an older backend that sends only `reachable` still gets the old sentence', () => {
  // Forward-compat: no `status` field at all must not become "slow" by accident.
  assert.equal(comfyuiAnswering({ reachable: true }), true);
  assert.equal(comfyuiDownReason({ reachable: false }), '⚠ Configure ComfyUI in Settings');
});

test('the Krea card says "slow", not "configure ComfyUI", on a slow install', () => {
  const slow = kreaUnavailableReason({
    comfyuiReachable: false,
    comfyui: { reachable: false, status: 'slow', object_info_timeout_s: 45 },
  });
  assert.match(slow, /45/);
  assert.doesNotMatch(slow, /configure comfyui/i);
  // ...and the genuinely-stopped case is unchanged.
  assert.equal(
    kreaUnavailableReason({ comfyuiReachable: false, comfyui: { reachable: false, status: 'unreachable' } }),
    '⚠ Configure ComfyUI in Settings');
});

test('both local engines read the same capabilities block, so they agree', () => {
  const caps = {
    engines: { klein: false, krea: false },
    comfyui: { reachable: false, status: 'slow', hint: 'ComfyUI took more than 45s to answer.' },
  };
  const klein = localEngineUnavailableReason('klein', caps);
  const krea = localEngineUnavailableReason('krea', caps);
  assert.equal(klein, krea);
  assert.match(klein, /45s/);
});
