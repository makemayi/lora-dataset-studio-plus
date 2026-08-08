import test from 'node:test';
import assert from 'node:assert/strict';
import { webImageSource } from './webImageSource.js';

const valid = {
  platform: 'websearch',
  source_url: 'https://blog.example.test/post/42',
};

test('Web-search source resolves the origin link and host for valid metadata', () => {
  assert.deepEqual(webImageSource(valid), {
    sourceUrl: valid.source_url,
    host: 'blog.example.test',
  });
});

test('Web-search source rejects anything that is not the websearch platform', () => {
  assert.equal(webImageSource({ ...valid, platform: 'pexels' }), null);
  assert.equal(webImageSource({ ...valid, platform: 'other' }), null);
  assert.equal(webImageSource({ ...valid, platform: undefined }), null);
  const { platform, ...withoutPlatform } = valid;
  assert.equal(webImageSource(withoutPlatform), null);
});

test('Web-search source rejects unusable metadata containers', () => {
  assert.equal(webImageSource(null), null);
  assert.equal(webImageSource('not an object'), null);
  assert.equal(webImageSource(42), null);
  assert.equal(webImageSource([valid]), null);
});

test('Web-search source rejects a missing, empty or non-string source_url', () => {
  assert.equal(webImageSource({ ...valid, source_url: undefined }), null);
  assert.equal(webImageSource({ ...valid, source_url: '' }), null);
  assert.equal(webImageSource({ ...valid, source_url: '   ' }), null);
  assert.equal(webImageSource({ ...valid, source_url: 12 }), null);
});

test('Web-search source rejects a source_url longer than 2048 characters', () => {
  const longUrl = `https://blog.example.test/${'a'.repeat(2048)}`;
  assert.ok(longUrl.length > 2048);
  assert.equal(webImageSource({ ...valid, source_url: longUrl }), null);
});

test('Web-search source rejects a source_url carrying control characters', () => {
  assert.equal(webImageSource({ ...valid, source_url: 'https://blog.example.test/\x00post' }), null);
  assert.equal(webImageSource({ ...valid, source_url: 'https://blog.example.test/post\nmore' }), null);
});

test('Web-search source rejects http instead of https', () => {
  assert.equal(webImageSource({ ...valid, source_url: 'http://blog.example.test/post/42' }), null);
});

test('Web-search source rejects a source_url carrying embedded credentials', () => {
  assert.equal(webImageSource({ ...valid, source_url: 'https://user:pass@blog.example.test/x' }), null);
});

test('Web-search source rejects javascript: and data: schemes so the anchor cannot be hijacked', () => {
  assert.equal(webImageSource({ ...valid, source_url: 'javascript:alert(1)' }), null);
  assert.equal(webImageSource({ ...valid, source_url: 'data:text/html,<script>alert(1)</script>' }), null);
});
