import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const src = fs.readFileSync(new URL('./WebImageSource.jsx', import.meta.url), 'utf8');

test('the link cannot be hijacked: target and rel are always set together', () => {
  assert.match(src, /target="_blank" rel="noopener noreferrer"/);
});

test('it renders nothing for non-websearch metadata — webImageSource is the single gate', () => {
  assert.match(src, /import \{ webImageSource \} from '\.\.\/\.\.\/utils\/webImageSource'/);
  assert.match(src, /const origin = webImageSource\(metadata\);/);
  assert.match(src, /if \(!origin\) return null;/);
  // No parallel decision is taken in the component itself.
  assert.doesNotMatch(src, /metadata\?\.platform/);
});

test('the anchor points at the page, titled with the host it came from', () => {
  assert.match(src, /href=\{origin\.sourceUrl\}/);
  assert.match(src, /title=\{`Found on \$\{origin\.host\}`\}/);
  assert.match(src, /Source ↗/);
});
