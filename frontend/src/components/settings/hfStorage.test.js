import test from 'node:test'
import assert from 'node:assert/strict'

import {
  deletionSafety, formatBytes, lastRunLabel, sortedCaches, storageSummary,
} from './hfStorage.js'
import { CONFIRMABLE_REFUSALS, matchConfirmableRefusal } from '../../utils/trainingRefusals.js'

const GB = 1000 ** 3

test('sizes read in the same decimal GB huggingface.co shows', () => {
  assert.equal(formatBytes(25 * GB), '25.0 GB')
  assert.equal(formatBytes(500 * 1000 ** 2), '500 MB')
  // Not-measured is its own answer: a repo the Hub reported no size for must
  // never read as an empty one.
  assert.equal(formatBytes(null), '—')
  assert.equal(formatBytes(undefined), '—')
  assert.notEqual(formatBytes(null), formatBytes(0))
})

test('a cache with its local file still here is safe to delete', () => {
  const safety = deletionSafety({ local_available: true })
  assert.equal(safety.level, 'safe')
  assert.match(safety.text, /re-upload/)
})

test('a cache whose local source is gone is flagged as the last copy', () => {
  const safety = deletionSafety({ local_available: false, local_reason: 'weights_missing' })
  assert.equal(safety.level, 'only_copy')
  assert.equal(safety.label, 'Only copy')
  assert.match(safety.text, /last copy/)
  assert.match(safety.text, /local weights file is gone/)
})

test('an unmatched cache still says why, never nothing', () => {
  const safety = deletionSafety({ local_available: false, local_reason: 'unknown_source' })
  assert.equal(safety.level, 'only_copy')
  assert.match(safety.text, /no run or dataset/)
  // An unknown reason code must not produce "undefined" in a delete confirm.
  assert.doesNotMatch(deletionSafety({ local_reason: 'brand_new_code' }).text, /undefined/)
})

test('summary states the gap, the caches and that the ceiling is a guess', () => {
  const s = storageSummary({
    ok: true,
    namespace: 'tester',
    used_bytes: 90 * GB,
    private_repo_count: 3,
    forecast: {
      needed_bytes: 46 * GB, checkpoint_bytes: 26 * GB, keeps: 1,
      margin_bytes: 20 * GB, limit_bytes: 100 * GB, limit_is_estimate: true,
      fits: false, shortfall_bytes: 36 * GB,
    },
  })
  assert.equal(s.measured, true)
  assert.equal(s.tone, 'blocked')
  const text = s.lines.join(' ')
  assert.match(text, /36\.0 GB short/)
  assert.match(text, /ESTIMATE/)
  assert.match(text, /no quota endpoint/)
})

test('the breakdown adds up to the total it explains, fp8 export included', () => {
  // The shipped default: 26 GB checkpoint + a 14.3 GB fp8 twin + 20 GB margin.
  // The card used to say "26.0 GB checkpoint × 1 kept + 20.0 GB margin" under a
  // 60.3 GB total, and the missing 14 GB read as a mistake in the card.
  const s = storageSummary({
    ok: true,
    namespace: 'tester',
    used_bytes: 10 * GB,
    private_repo_count: 1,
    forecast: {
      needed_bytes: 60.3 * GB, checkpoint_bytes: 26 * GB, keeps: 1,
      fp8_bytes: 14.3 * GB, fp8_export: true, margin_bytes: 20 * GB,
      limit_bytes: 100 * GB, fits: true, free_bytes: 90 * GB,
    },
  })
  const text = s.lines.join(' ')
  assert.match(text, /14\.3 GB fp8 export/)
  const [, total, checkpoint, fp8, margin] = text
    .match(/needs about ([\d.]+) GB \(one ([\d.]+) GB checkpoint × 1 kept \+ a ([\d.]+) GB fp8 export \+ ([\d.]+) GB margin\)/)
  assert.equal(Number(checkpoint) + Number(fp8) + Number(margin), Number(total))
})

test('a run that exports no fp8 twin omits the term rather than showing 0 GB', () => {
  const s = storageSummary({
    ok: true, namespace: 'n', used_bytes: 0, private_repo_count: 0,
    forecast: {
      needed_bytes: 46 * GB, checkpoint_bytes: 26 * GB, keeps: 1, fp8_bytes: 0,
      margin_bytes: 20 * GB, limit_bytes: 100 * GB, fits: true, free_bytes: 100 * GB,
    },
  })
  const text = s.lines.join(' ')
  assert.doesNotMatch(text, /fp8/)
  assert.match(text, /one 26\.0 GB checkpoint × 1 kept \+ 20\.0 GB margin/)
})

test('an unmeasurable account says so instead of showing a reassuring zero', () => {
  const s = storageSummary({ ok: false })
  assert.equal(s.measured, false)
  assert.equal(s.tone, 'unknown')
  assert.match(s.lines.join(' '), /could not be measured/)
  // The same must hold for a fits:null forecast on a listing that answered.
  const partial = storageSummary({
    ok: true, namespace: 'n', used_bytes: 0, private_repo_count: 0,
    forecast: { fits: null, needed_bytes: 46 * GB, keeps: 1 },
  })
  assert.equal(partial.tone, 'unknown')
})

test('unsized repos downgrade the total to a floor, out loud', () => {
  const s = storageSummary({
    ok: true, namespace: 'n', used_bytes: 10 * GB, private_repo_count: 2,
    unsized_repo_count: 1,
    forecast: { fits: true, free_bytes: 5 * GB, needed_bytes: 1 * GB, keeps: 1 },
  })
  assert.match(s.lines.join(' '), /is a floor/)
})

test('caches list biggest first — the order for freeing space', () => {
  const rows = sortedCaches({
    caches: [{ name: 'a', used_bytes: 1 * GB }, { name: 'b', used_bytes: 24 * GB },
      { name: 'c' }],
  })
  assert.deepEqual(rows.map((r) => r.name), ['b', 'a', 'c'])
})

test('last-run label degrades to a sentence, never a blank', () => {
  assert.match(lastRunLabel({}), /Never used/)
  assert.match(
    lastRunLabel({ last_run: { id: 146, name: 'dense', created_at: '2026-08-03T10:00:00' } }),
    /#146 “dense” \(2026-08-03\)/,
  )
})

test('the storage refusal is confirmable, so a wrong guess never locks the GPU', () => {
  const hit = matchConfirmableRefusal('HF_STORAGE_FULL: 36 GB short', CONFIRMABLE_REFUSALS)
  assert.ok(hit, 'HF_STORAGE_FULL must be a confirmable refusal')
  assert.equal(hit[1], 'allow_hf_storage')
})
