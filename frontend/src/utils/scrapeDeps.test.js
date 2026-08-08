/** The scraper install banner must name the packages THIS machine is missing.
 *
 * It used to recite "curl_cffi, gallery-dl, cloudscraper…" from memory while the
 * backend probe watches seven modules. `ddgs` (keyless web image search) and
 * `yt_dlp` (video sources, launched as `python -m`) were added to the probe
 * later, so an install flagged because of those two read a warning naming
 * neither — and could not explain why it was asking for a reinstall.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { missingScrapeDeps, scrapeDepsBanner } from './scrapeDeps.js';

test('the banner quotes exactly what the probe reported missing', () => {
  const msg = scrapeDepsBanner('missing: curl_cffi, ddgs, yt_dlp');
  assert.match(msg, /curl_cffi/);
  assert.match(msg, /ddgs/);
  assert.match(msg, /yt_dlp/);
  // And nothing it did NOT report: naming a package that is installed sends the
  // user hunting for a problem they do not have.
  assert.ok(!/gallery_dl|gallery-dl|cloudscraper|instaloader/.test(msg), msg);
});

test('missingScrapeDeps parses the probe detail and ignores the rest', () => {
  assert.deepEqual(missingScrapeDeps('missing: bs4, instaloader'), ['bs4', 'instaloader']);
  assert.deepEqual(missingScrapeDeps('missing: ddgs'), ['ddgs']);
  // The healthy string carries no list — and must not be read as one.
  assert.deepEqual(missingScrapeDeps('scrape deps OK'), []);
  assert.deepEqual(missingScrapeDeps(undefined), []);
  assert.deepEqual(missingScrapeDeps(null), []);
  assert.deepEqual(missingScrapeDeps(''), []);
});

test('with no detail the banner names NO package rather than a stale list', () => {
  // An older backend sends only the boolean. Saying less is honest; reciting
  // three names that may all be installed is the bug this file exists for.
  for (const detail of [undefined, null, '', 'scrape deps OK']) {
    const msg = scrapeDepsBanner(detail);
    assert.match(msg, /scraper packages/i);
    assert.ok(!/curl_cffi|gallery|cloudscraper|ddgs|yt_dlp/.test(msg), `${detail}: ${msg}`);
  }
});

test('the panel renders the parsed list instead of a hard-coded one', () => {
  const panel = fs.readFileSync(
    new URL('../components/dataset/ConceptSourcesPanel.jsx', import.meta.url), 'utf8');
  const banner = panel.match(/caps\.scrape_deps === false[\s\S]{0,900}?<\/p>/);
  assert.ok(banner, 'the scrape-deps banner is no longer recognisable in the panel');
  assert.match(banner[0], /scrapeDepsBanner\(caps\.scrape_deps_detail\)/,
    'the banner stopped reading the probe detail');
  assert.ok(!/gallery-dl|cloudscraper/.test(banner[0]),
    'the banner grew a second, hand-maintained copy of the package list');
});

test('the backend publishes the detail the banner reads', () => {
  // The whole fix depends on this one field reaching the frontend; without it
  // the banner silently falls back to naming nothing.
  const caps = fs.readFileSync(
    new URL('../../../backend/app/capabilities.py', import.meta.url), 'utf8');
  assert.match(caps, /'scrape_deps_detail':\s*scrape_deps\['detail'\]/);
  assert.match(caps, /'scrape_deps':\s*scrape_deps\['ok'\]/);
});
