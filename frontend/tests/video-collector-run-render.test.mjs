import assert from 'node:assert/strict'
import test from 'node:test'

import { render } from './support/mountJsx.mjs'

// The JSX loader is registered while mountJsx is evaluated, so this import has
// to stay dynamic (see support/mountJsx.mjs).
const { default: VideoCollectorRun } =
  await import('../src/components/videobank/VideoCollectorRun.jsx')

const unconfigured = { bankId: 1, busy: false }
const configured = { bankId: 1, busy: false, collectors: ['Kuaishou videos', 'Douyin videos'] }

/* Mounted rather than grepped: a source-text assertion cannot tell a branch that
   was DELETED from one that throws on render, and this block is nothing BUT two
   branches (CLAUDE.md ▸ UI changes, rule 6). The configured branch is reached by
   injecting the list — the same hook the fetch fills in production. */

test('with nothing configured it says so, and says where a collector comes from', () => {
  // The shipped state. It must read as a configuration this app HAS, not as an
  // error and not as a missing install step — no collector ships, on purpose.
  const html = render(VideoCollectorRun, unconfigured)
  assert.match(html, /No collector is configured/)
  assert.match(html, /video_collectors/)
  assert.match(html, /stops working when that site changes/)
  // The two tokens the config speaks, since the command is only half the story.
  assert.match(html, /\{url\}/)
  assert.match(html, /\{folder\}/)
})

test('the configured branch mounts the picker, the URL box and the Run button', () => {
  const html = render(VideoCollectorRun, configured)
  assert.match(html, /aria-label="Collector to run"/)
  assert.match(html, /aria-label="Account URL to collect from"/)
  assert.match(html, />Kuaishou videos</)
  assert.match(html, />Douyin videos</)
})

test('both Run labels stay mounted so the starting swap is a hidden flip', () => {
  const html = render(VideoCollectorRun, configured)
  assert.match(html, />Run</)
  assert.match(html, /Starting…/)
})

test('the copy warns that this takes minutes, holds the bank, and survives leaving', () => {
  // "Every limit stays visible": a browser-driving collector is not an API
  // call, and while it runs NO pass can start — the bank has one job slot.
  const html = render(VideoCollectorRun, configured)
  assert.match(html, /several minutes/)
  assert.match(html, /close this page/)
  assert.match(html, /passes are busy/)
})

test('the started confirmation is mounted before it is needed', () => {
  const html = render(VideoCollectorRun, configured)
  assert.match(html, /Collector started/)
})

test('the block is addressable and its live lines are polite', () => {
  const html = render(VideoCollectorRun, configured)
  assert.match(html, /id="video-collector-run"/)
  assert.match(html, /aria-live="polite"/)
})
