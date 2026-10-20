/**
 * The Score panel's decisions, tested as data — not pixels.
 * Each case here is a real trap: a null dimension rendered as 0 makes a dataset
 * look bad for a pass nobody ran; a withheld total shown as 0 hides the server's
 * "measured < 50%" refusal; a disabled rule must not render a button.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyResultLine, causeRows, dimensionRows, identitySummary, ruleButtons, totalLabel,
} from './datasetScore.js';

test('an unmeasured dimension stays null instead of becoming 0', () => {
  const rows = dimensionRows({ consistency: 83, quality: null });
  const byKey = Object.fromEntries(rows.map((r) => [r.key, r.value]));
  assert.equal(byKey.consistency, 83);
  assert.equal(byKey.quality, null);
  assert.equal(byKey.caption, null, 'missing keys read as unmeasured, never 0');
  assert.equal(rows.length, 6);
});

test('a withheld total reads "不足", never 0 and never a fake number', () => {
  assert.equal(totalLabel({ total: null, measured: 35 }).text, '不足');
  assert.equal(totalLabel({ total: 74, measured: 80 }).text, '74');
});

test('rule buttons only come from enabled rows that carry an action', () => {
  const buttons = ruleButtons([
    { key: 'a', name: '① X', action: 'analyse_images', enabled: true, params: {} },
    { key: 'b', name: '② Y', action: null, enabled: true, params: {} },
    { key: 'c', name: '③ Z', action: 'purge', enabled: false, params: { dry_run: false } },
  ]);
  assert.deepEqual(buttons.map((b) => b.key), ['a']);
  assert.equal(buttons[0].dryRun, true, 'dry_run defaults to true when the rule omits it');
});

test('destructive rules are flagged so the panel can colour them', () => {
  const buttons = ruleButtons([
    { key: 'p', name: 'purge', action: 'purge', enabled: true, params: {} },
    { key: 'm', name: 'mark', action: 'mark_causes', enabled: true, params: {} },
    { key: 's', name: 'scan', action: 'score_cluster', enabled: true, params: {} },
  ]);
  const danger = Object.fromEntries(buttons.map((b) => [b.key, b.danger]));
  assert.equal(danger.p, true);
  assert.equal(danger.m, true);
  assert.equal(danger.s, false);
});

test('a single identity is reported as drift, not as "other people"', () => {
  const single = identitySummary({ identities: [{ group: 1, size: 88 }], multi_identity: false });
  assert.equal(single.multi, false);
  assert.match(single.text, /单身份/);
  const multi = identitySummary({ identities: [{ group: 1, size: 20 }, { group: 2, size: 9 }],
    multi_identity: true });
  assert.equal(multi.multi, true);
  assert.match(multi.text, /可能混了多个人/);
});

test('cause rows are ordered by severity, and zero-count classes stay visible', () => {
  // 保留 0 是刻意的：让"这一类没有"成为一条可读的事实，而不是靠缺失去暗示。
  const rows = causeRows({ duplicate: { count: 8, action: 'x' }, drift: { count: 0 } });
  assert.deepEqual(rows.map((r) => r.key), ['duplicate', 'drift']);
  assert.equal(rows[1].count, 0);
  const ordered = causeRows({ drift: { count: 5 }, other_identity: { count: 2 } });
  assert.deepEqual(ordered.map((r) => r.key), ['other_identity', 'drift']);
});

test('apply results read as sentences, including the refusals', () => {
  assert.match(applyResultLine('pass_mark_causes', { dry_run: true, count: 8 }), /命中 8 张/);
  assert.match(applyResultLine('pass_purge', { dry_run: true, would_delete: 0 }), /会删除 0 张/);
  assert.match(applyResultLine('pass_mark_low', { blocked: true, reason: '比例太高' }), /已拦截/);
  assert.match(applyResultLine('pass_purge', { dry_run: false, purged: 14 }), /已删除 14 张/);
});
