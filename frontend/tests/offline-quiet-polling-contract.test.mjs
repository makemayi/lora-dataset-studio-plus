/* Contract: the network layer stays quiet about background polls, the toast
 * queue merges repeats, and the offline state is rendered somewhere.
 *
 * These are source assertions on purpose. The logic itself is unit-tested
 * (utils/toastQueue.test.js, utils/connectionStatus.test.js,
 * components/bank/progressPresence.test.js) — what THIS file protects is the
 * WIRING, which lives in JSX `node --test` cannot execute. A rewrite of
 * Toast.jsx that quietly goes back to `[...prev, toast]` would pass every unit
 * test and re-ship the ten stacked banners.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const SRC = join(dirname(fileURLToPath(import.meta.url)), '..', 'src');
const read = (p) => readFileSync(join(SRC, p), 'utf8');

test('apiFetch accepts a background flag and strips it from the fetch init', () => {
  const s = read('api/fetchClient.js');
  assert.match(s, /const \{ background = false, \.\.\.init \} = options/,
    'background must be destructured out — it is not a valid RequestInit key');
  assert.match(s, /fetchWithCsrfRetry\(url, init\)/);
});

test('a network failure goes through the connection store, not straight to a toast', () => {
  const s = read('api/fetchClient.js');
  assert.match(s, /if \(reportRequestFailure\(\{ background \}\)\) toastRef\?\.error\(/,
    'the toast must be gated on the store deciding this failure is worth announcing');
  assert.doesNotMatch(s, /catch \{\s*toastRef\?\.error\('Connection lost/,
    'the unconditional per-failure banner must be gone');
});

test('recovery is announced', () => {
  const s = read('api/fetchClient.js');
  assert.match(s, /if \(reportRequestSuccess\(\)\) toastRef\?\.success\(CONNECTION_BACK_MESSAGE\)/);
});

test('the toast provider merges through the queue instead of appending blindly', () => {
  const s = read('components/common/Toast.jsx');
  assert.match(s, /from '\.\.\/\.\.\/utils\/toastQueue'/);
  assert.match(s, /pushToast\(prev, \{ id, message, type, expiresAt \}\)/);
  assert.doesNotMatch(s, /setToasts\(\(prev\) => \[\.\.\.prev,/,
    'the naive append is what stacked ten identical banners');
});

test('the repeat counter is aria-hidden so a merged banner is announced once', () => {
  const s = read('components/common/Toast.jsx');
  assert.match(s, /<span aria-hidden="true"[\s\S]{0,220}\{t\.count\}×/,
    'a visible counter inside the live region would re-announce on every repeat');
});

test('the app renders one persistent offline indicator', () => {
  assert.match(read('App.jsx'), /<ConnectionBanner \/>/);
  const banner = read('components/common/ConnectionBanner.jsx');
  assert.match(banner, /role="status"/, 'must be announceable');
  assert.match(banner, /Offline — reconnecting…/);
  // No timer/counter in the banner: its text must be stable while it is up,
  // or the live region re-announces on every tick.
  assert.doesNotMatch(banner, /setInterval|offlineSince/);
});

test('the bank progress zone knows about the offline state', () => {
  const s = read('components/bank/BankWorkspace.jsx');
  assert.match(s, /progressPresence\(activity, offline\)/);
  assert.match(s, /<ProgressBar activity=\{payload\?\.activity\} onCancel=\{cancelJob\} offline=\{!connection\.online\}/);
});

test('the polls that fire on a timer are marked background', () => {
  const marked = [
    ['components/bank/BankWorkspace.jsx', /refreshPayload\(\{ background: true \}\), 2000\)/],
    ['hooks/useTrainingActivity.js', /'\/api\/train\/activity', \{ background: true \}/],
    ['components/settings/MaintenanceSection.jsx', /background: true/],
    ['components/settings/EnginesSection.jsx', /background: true/],
    ['components/dataset/CaptionOptionsPopover.jsx', /background: true/],
    ['App.jsx', /update\/check\?auto=1', \{ background: true \}/],
  ];
  for (const [file, re] of marked) assert.match(read(file), re, `${file} poll not marked background`);
});
