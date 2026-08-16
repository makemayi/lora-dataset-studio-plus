import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { matchesQuery, SETTINGS_SECTIONS } from './registry.js';

test('small-image Klein rescue terms find Scraping & sources', () => {
  const scraping = SETTINGS_SECTIONS.find((section) => section.id === 'scraping');
  for (const query of ['klein', 'small image', 'rescue', 'upscale']) {
    assert.equal(matchesQuery(scraping, query), true, query);
  }
});

test('Pexels API credential terms find Scraping & sources', () => {
  const scraping = SETTINGS_SECTIONS.find((section) => section.id === 'scraping');
  for (const query of ['pexels', 'pexels api', 'api key', 'quota']) {
    assert.equal(matchesQuery(scraping, query), true, query);
  }
});

test('Pexels key and attribution markup stay wired without nested controls', () => {
  const settingsSource = readFileSync(new URL('./ScrapingSection.jsx', import.meta.url), 'utf8');
  const panelSource = readFileSync(
    new URL('../dataset/ConceptSourcesPanel.jsx', import.meta.url), 'utf8');
  const scraperSourceSearchSource = readFileSync(
    new URL('../dataset/scraperSourceSearch.js', import.meta.url), 'utf8');
  const attributionSource = readFileSync(
    new URL('../dataset/PexelsAttribution.jsx', import.meta.url), 'utf8');
  const readmeSource = readFileSync(new URL('../../../../README.md', import.meta.url), 'utf8');
  const envSource = readFileSync(new URL('../../../../.env.example', import.meta.url), 'utf8');

  assert.match(settingsSource, /key:\s*'PEXELS_API_KEY'/);
  for (const [label, source] of [
    ['settings', settingsSource], ['README', readmeSource], ['env example', envSource],
  ]) {
    assert.match(source, /https:\/\/www\.pexels\.com\/api\/key\//, `${label}: current key URL`);
    assert.doesNotMatch(source, /pexels\.com\/api\/new\//, `${label}: obsolete key URL`);
    assert.match(source,
      /An API key alone does not authorize\s+dataset or machine-learning use/,
      `${label}: API key is not permission`);
    assert.match(source, /Pexels\s+has explicitly authorized this use case/,
      `${label}: explicit authorization gate`);
  }
  for (const source of [settingsSource, readmeSource]) {
    assert.match(source,
      /https:\/\/help\.pexels\.com\/hc\/en-us\/articles\/900005880463-What-are-the-Terms-and-Conditions/);
  }
  assert.match(panelSource, /\['pexels', 'Pexels'\]/);
  assert.match(panelSource, /buildPexelsSearchUrl/);
  assert.match(panelSource,
    /I confirm I have explicit Pexels authorization for dataset\/ML use/);
  assert.match(panelSource,
    /https:\/\/help\.pexels\.com\/hc\/en-us\/articles\/900005880463-What-are-the-Terms-and-Conditions/);
  assert.match(panelSource, /Photos provided by Pexels/);
  assert.match(panelSource, /<PexelsAttribution metadata=\{it\}/);
  // The url→payload mapping lives in scraperSourceSearch.js (scrapeItemToImportPayload),
  // not inlined in the panel — see scraperSourceSearch.test.js for its behavioural coverage.
  // Pinned on the CALL SITE, not just the import: a dangling import with nothing
  // calling it would still match a bare name-presence check, and the wiring this
  // test exists to guarantee (selected items are actually mapped before import)
  // would then be unpinned.
  assert.match(panelSource, /\.map\(scrapeItemToImportPayload\)/);
  for (const field of ['platform', 'source_url', 'photographer', 'photographer_url']) {
    assert.match(scraperSourceSearchSource, new RegExp(`${field}:`), `selected items forward ${field}`);
  }
  assert.match(attributionSource, /Photo by\{' '\}[\s\S]*\{' · '\}[\s\S]*Pexels/);
  assert.match(attributionSource, /rel="noopener noreferrer"/);

  // The tile markup moved to the shared ScrapePickGrid when the paste intake
  // needed the same grid, so the invariant is pinned THERE now. It matters more
  // after the move, not less: the button and the credit that must sit beside it
  // are in two different files, and nesting one in the other is now something
  // nobody would see while editing either.
  const pickGridSource = readFileSync(
    new URL('../shared/ScrapePickGrid.jsx', import.meta.url), 'utf8');
  const selectionButton = pickGridSource.match(
    /<button type="button" onClick=\{\(\) => onToggle\?\.\(it\.url\)\}[\s\S]*?<\/button>/);
  assert.ok(selectionButton, 'selection button markup must remain present');
  assert.doesNotMatch(selectionButton[0], /<a\b/i,
    'Pexels credit links must remain siblings of the selection button');
  // ...and the caption the panel supplies is rendered AFTER that button closes,
  // which is what keeps the credit a sibling rather than a child.
  assert.match(pickGridSource, /<\/button>[\s\S]*?renderCaption \? renderCaption\(it\) : null/);
});

test('Vast offer filter terms find Training', () => {
  const training = SETTINGS_SECTIONS.find((section) => section.id === 'training');
  for (const query of ['verified', 'secure cloud', 'community cloud', 'offer filter']) {
    assert.equal(matchesQuery(training, query), true, query);
  }
});
