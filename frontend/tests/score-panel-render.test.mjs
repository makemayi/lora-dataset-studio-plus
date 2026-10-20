/**
 * The 🩺 Score panel, RENDERED. Three things the source alone cannot prove:
 * a withheld total must not print "0 / 100", an unmeasured dimension must not
 * print 0, and a disabled rule must not produce a button.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs';

const { default: DatasetScorePanel } = await import(
  '../src/components/dataset/DatasetScorePanel.jsx');

const panel = (props = {}) => renderToStaticMarkup(createElement(
  DatasetScorePanel, { datasetId: 1, refreshKey: 0, ...props }));

test('collapsed by default, and it says what it is', () => {
  const html = panel();
  assert.match(html, /Score/);
  assert.match(html, /aria-expanded="false"/);
  assert.doesNotMatch(html, /可执行规则/, 'body only renders once opened');
});
