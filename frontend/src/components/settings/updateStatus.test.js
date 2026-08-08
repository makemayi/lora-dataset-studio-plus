import test from 'node:test';
import assert from 'node:assert/strict';

import {
  PINOKIO_UPDATE_STEPS,
  PINOKIO_UPDATE_GUIDE_URL,
  formatMB,
  installMode,
  isPinokioInstall,
  zipUpdateHeadline,
  progressPercent,
  progressLabel,
} from './updateStatus.js';

test('formatMB is compact and empty for unknown sizes', () => {
  assert.equal(formatMB(0), '');
  assert.equal(formatMB(null), '');
  assert.equal(formatMB(42_000_000), '42.0 MB');
  assert.equal(formatMB(150_000_000), '150 MB');   // >=100 MB -> whole number
});

test('installMode distinguishes git, zip, unavailable and unknown', () => {
  assert.equal(installMode({ ok: true, is_git: true, update_available: true }), 'git');
  assert.equal(installMode({ ok: true, is_git: false, can_apply: true }), 'zip');
  assert.equal(installMode({ ok: true, is_git: false, can_apply: false }), 'unavailable');
  assert.equal(installMode({ ok: false, reason: 'offline' }), 'unknown');
  assert.equal(installMode(null), 'unknown');
});

test('a Pinokio install outranks its own git checkout', () => {
  // The payload really does carry is_git/behind (its Update tab pulls those
  // commits); reading it as plain 'git' would put the stranding button back.
  const pinokio = {
    ok: true,
    install_mode: 'pinokio',
    is_git: true,
    behind: 3,
    update_available: true,
  };
  assert.equal(isPinokioInstall(pinokio), true);
  assert.equal(installMode(pinokio), 'pinokio');
  assert.equal(installMode({ ...pinokio, can_apply: true }), 'pinokio');
  assert.equal(isPinokioInstall({ install_mode: 'git' }), false);
});

test('pinokio update steps are three clicks, not shell commands', () => {
  assert.deepEqual(PINOKIO_UPDATE_STEPS, [
    'Stop the app in Pinokio',
    'Click the Update tab',
    'Click Start',
  ]);
  assert.equal(Object.isFrozen(PINOKIO_UPDATE_STEPS), true);
  for (const step of PINOKIO_UPDATE_STEPS) {
    assert.doesNotMatch(step, /git |pip /,
      'this install shape never opens a terminal — do not name commands');
  }
  assert.match(PINOKIO_UPDATE_GUIDE_URL, /#option-5--pinokio-one-click-any-os$/);
});

test('zipUpdateHeadline announces the release and its size when known', () => {
  assert.equal(zipUpdateHeadline({ latest: '2026.07.19', zip_size: 42_000_000 }),
    'Update to v2026.07.19 (download ~42.0 MB)');
  assert.equal(zipUpdateHeadline({ latest: '2026.07.19' }), 'Update to v2026.07.19');
  assert.equal(zipUpdateHeadline({}), 'Update available');
});

test('progressPercent clamps and is null without a total', () => {
  assert.equal(progressPercent({ downloaded: 21_000_000, total: 42_000_000 }), 50);
  assert.equal(progressPercent({ downloaded: 99, total: 0 }), null);
  assert.equal(progressPercent({ downloaded: 999, total: 100 }), 100);   // clamped
  assert.equal(progressPercent(null), null);
});

test('progressLabel renders each active phase and defers idle/done', () => {
  assert.match(progressLabel({ phase: 'downloading', downloaded: 21_000_000, total: 42_000_000 }),
    /Downloading… 50% \(21\.0 MB \/ 42\.0 MB\)/);
  assert.match(progressLabel({ phase: 'downloading', downloaded: 5_000_000, total: 0 }),
    /Downloading… 5\.0 MB/);            // unknown total -> no percent
  assert.match(progressLabel({ phase: 'extracting' }), /Extracting/);
  assert.match(progressLabel({ phase: 'installing' }), /Installing/);
  assert.match(progressLabel({ phase: 'restarting' }), /Restarting/);
  assert.equal(progressLabel({ phase: 'done' }), null);
  assert.equal(progressLabel({ phase: 'idle' }), null);
});
