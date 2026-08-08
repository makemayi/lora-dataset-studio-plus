/* The busy notice next to a setting is a DIFFERENT surface from the progress bar
   at the top of the bank, and it was missed when the score pass learned to
   announce its silent tail. Both defects below were seen on a real 50 000-image
   bank, on a phone, and both made a working pass look broken or unreadable. */
import test from 'node:test'
import assert from 'node:assert/strict'
import { busyLine, jobProgress } from './bankPassRun.js'

// The exact string the server publishes while ✨ Score groups styles.
const GROUPING = 'grouping styles over 21220 image(s) — the slow tail of this '
  + 'pass; Stop now keeps every score already computed but discards the '
  + 'grouping, which can only be redone whole'

test('a phase with nothing to count prints NO number, not a zero', () => {
  // done=0/total=0 is how a phase says "no countable unit here". The old code
  // fell through to `${done}` and rendered "— 0 · grouping styles…", which
  // reads as "zero done" on a pass that is very much working.
  assert.equal(jobProgress({ done: 0, total: 0 }), '')
  const line = busyLine({ kind: 'score', activity: { done: 0, total: 0, detail: GROUPING } })
  assert.ok(!/—\s*0\s*[·-]/.test(line), `a bare zero leaked into: ${line}`)
  assert.match(line, /grouping styles over 21220/)
})

test('a count with no known total is still worth showing', () => {
  // The guard above must not silence a pass that counts up without knowing
  // where it stops — that number is the only progress the user gets.
  assert.equal(jobProgress({ done: 137, total: 0 }), '137')
  assert.equal(jobProgress({ done: 137, total: 412 }), '137 / 412')
})

test('the Stop cost is NOT echoed where there is no Stop button', () => {
  // It stays in the progress bar, which is where the control lives. Here it
  // would be ~150 characters about an off-screen button, and on a 400 px phone
  // it pushed the setting it annotates out of view.
  const line = busyLine({ kind: 'score', activity: { done: 0, total: 0, detail: GROUPING } })
  assert.ok(!/Stop now keeps/.test(line), `the Stop explanation leaked into: ${line}`)
  // 208 characters before this fix, 103 after. The bound below is a REGRESSION
  // guard against a future clause being appended — it is not a claim that 120
  // characters is what fits at 400 px, which nobody has measured. Naming what a
  // number is not is the only way it stays honest as the sentence evolves.
  assert.ok(line.length < 120, `grew past the guard (${line.length}): ${line}`)
})

test('a detail with no semicolon survives whole', () => {
  // Proof the split does not silently truncate the many details that carry no
  // Stop clause at all.
  const line = busyLine({ kind: 'score', activity: { done: 2, total: 9, detail: 'writing 21220 scores' } })
  assert.match(line, /writing 21220 scores$/)
  assert.match(line, /2 \/ 9/)
})
