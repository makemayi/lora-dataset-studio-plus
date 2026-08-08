import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const src = fs.readFileSync(new URL('./SourceAttribution.jsx', import.meta.url), 'utf8');

test('it imports both platform renderers and dispatches by platform, nothing else', () => {
  assert.match(src, /import PexelsAttribution from '\.\/PexelsAttribution'/);
  assert.match(src, /import WebImageSource from '\.\/WebImageSource'/);
  assert.match(src, /const platform = metadata && typeof metadata === 'object' \? metadata\.platform : null;/);
  assert.match(src, /const Renderer = PLATFORM_RENDERERS\[platform\];/);
});

test('every currently-supported platform is wired to its own renderer', () => {
  assert.match(src, /pexels:\s*PexelsAttribution/);
  assert.match(src, /websearch:\s*WebImageSource/);
});

test('an unrecognized platform (or no metadata at all) renders nothing — no second decision', () => {
  assert.match(src, /if \(!Renderer\) return null;/);
  // The dispatcher itself does not re-validate hosts/urls — that stays the sole
  // job of pexelsAttribution.js / webImageSource.js, so there is exactly ONE
  // place each platform's payload is judged.
  assert.doesNotMatch(src, /new URL\(/);
  assert.doesNotMatch(src, /\.hostname/);
});

test('the chosen renderer still receives metadata and className', () => {
  assert.match(src, /<Renderer metadata=\{metadata\} className=\{className\} \/>/);
});
