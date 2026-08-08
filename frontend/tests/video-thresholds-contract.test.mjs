import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import { thresholdFields, FLAG_LABELS } from '../src/components/videobank/videoMetricsFilter.js'

// 🎬 The quality cuts exist in TWO codebases and have to name the same things.
//
// This test exists because they did not. `first_frame_floor` was honoured by
// `video_metrics.verdicts` from the quality wave onwards and named by neither
// the config reader, the dry-run route, nor the panel table — so the `soft_start`
// flag it feeds could not fire anywhere in the app. Nothing failed: the backend
// test proved the rule worked, the frontend test proved the table was
// self-consistent, and each was checking its own side against a hard-coded copy
// of the list. The list is now declared ONCE, in the backend, and this is what
// holds the frontend to it.
const backend = readFileSync(
  new URL('../../backend/app/services/video_metrics.py', import.meta.url), 'utf8')

function backendThresholdKeys() {
  const m = backend.match(/^THRESHOLD_KEYS = \(([\s\S]*?)\)/m)
  assert.ok(m, 'could not find THRESHOLD_KEYS in video_metrics.py')
  return [...m[1].matchAll(/'([a-z_]+)'/g)].map((x) => x[1])
}

test('the panel offers every cut the backend honours, and no cut it does not', () => {
  assert.deepEqual(thresholdFields().map((f) => f.key), backendThresholdKeys())
})

test('every cut the backend honours feeds a flag the UI can name', () => {
  // A flag with no label renders as its raw key ("soft_start") in a chip the
  // user is meant to act on.
  for (const f of thresholdFields()) {
    assert.ok(FLAG_LABELS[f.flag], `${f.key} feeds an unlabelled flag ${f.flag}`)
  }
})

test('every flag the backend can raise has a label', () => {
  // Read off `verdicts()` itself rather than off the field table: a flag can be
  // raised by a rule whose threshold is spelled differently, and an unlabelled
  // one reaches the grid as a raw identifier.
  const raised = [...backend.matchAll(/flags\.add\('([a-z_]+)'\)/g)].map((m) => m[1])
  assert.ok(raised.length >= 5, 'no flags found — did verdicts() move?')
  for (const flag of raised) {
    assert.ok(FLAG_LABELS[flag], `the backend can raise "${flag}" with no label`)
  }
})
