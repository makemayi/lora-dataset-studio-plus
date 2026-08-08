/* Seen on a phone, on a real bank: the same phase sentence rendered twice on one
   screen — once under the threshold slider, once in the progress bar above it.
   Two surfaces, one narrator. The bar keeps the story; the slider's notice only
   has to say a pass is holding the setting. */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { busyLine } from './bankPassRun.js'

const GROUPING = 'grouping styles over 21220 image(s) — the slow tail of this pass'

test('the notice beside a setting names the pass, not its phase', () => {
  const line = busyLine({ kind: 'score', activity: { done: 0, total: 0, detail: GROUPING }, withDetail: false })
  assert.match(line, /Score pass is running on this bank/)
  assert.ok(!/grouping styles/.test(line), `the phase was echoed: ${line}`)
})

test('a refusal still carries the phase — it answers "why did nothing happen"', () => {
  // The opposite case, and the reason this is an option and not a deletion:
  // a refusal can appear with no progress bar in view, so dropping the detail
  // there would replace an explanation with a shrug.
  const line = busyLine({ kind: 'score', activity: { done: 0, total: 0, detail: GROUPING } })
  assert.match(line, /grouping styles over 21220/)
})

test('the thresholds panel is the surface that opts out', () => {
  // Pinned at the call site: an option nobody passes is an option that rots.
  const panel = fs.readFileSync(new URL('./BankThresholdsPanel.jsx', import.meta.url), 'utf8')
  assert.match(panel, /busyLine\(\{ activity, withDetail: false \}\)/)
})
