import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// Same convention as ConceptSourcesPanel.pagination.test.js: this component has
// no rendering harness in this repo (no testing-library/jsdom), so the contract
// is pinned on the wiring code itself (state set from the backend field, reset
// on `resetScan`) rather than on phrasing.
const source = readFileSync(
  new URL('./ConceptSourcesPanel.jsx', import.meta.url), 'utf8');

test('the truncation flag is read from the scan response, not invented client-side', () => {
  assert.match(source, /setPartial\(!!body\.partial\);/);
});

test('resetting the scan also clears the truncation flag', () => {
  assert.match(source, /setPartial\(false\)/);
});

test('the truncation notice is gated on the partial flag, independently of pagination', () => {
  // Must NOT be folded into the `paginated &&` block: a truncated album dive
  // clears `paginated` (from_albums), so the notice has to render on its own
  // condition or it silently disappears exactly when it matters most.
  assert.match(source, /\{partial &&/);
  assert.doesNotMatch(source, /paginated && partial/);
});
