import assert from 'node:assert/strict'
import test from 'node:test'
import {
  bankListOverview,
  bankOverviewModel,
  displayPercent,
  widthPercent,
} from './bankOverview.js'

test('an empty bank never presents zero as an analyzed pass', () => {
  const model = bankOverviewModel({ counts: { total: 0, scanned: 0, scored: 0 } })
  assert.equal(model.available, true)
  assert.equal(model.total, 0)
  assert.deepEqual(model.status.map((row) => row.value), [0, 0, 0])
  assert.ok(model.passes.every((pass) => pass.value === null))
  assert.ok(model.passes.every((pass) => pass.text.startsWith('Not measured · Run')))
})

test('an absent total is unavailable rather than a fake zero-percent analysis', () => {
  const model = bankOverviewModel({ counts: { scanned: 8 } })
  assert.equal(model.total, null)
  assert.equal(model.passes[0].value, 8)
  assert.equal(model.passes[0].percent, null)
  assert.equal(model.passes[0].text, '8 measured · total unavailable')
  assert.deepEqual(model.status.map((row) => row.value), [null, null, null])
})

test('a missing payload is unavailable, while a loaded total=0 remains a real empty bank', () => {
  const unavailable = bankOverviewModel(null)
  assert.equal(unavailable.available, false)
  assert.equal(unavailable.total, null)
  assert.deepEqual(unavailable.status, [])

  const empty = bankOverviewModel({ counts: { total: 0 } })
  assert.equal(empty.available, true)
  assert.equal(empty.total, 0)
  assert.deepEqual(empty.status.map((row) => row.value), [0, 0, 0])
})

test('partial analysis keeps exact bank-wide counts and exposes unfinished passes', () => {
  const model = bankOverviewModel({
    counts: { total: 10, keep: 3, pending: 5, reject: 2, scanned: 10,
      scored: 4, watermark_scanned: 0, framing_classified: 2 },
    res_buckets: { res_1_2: 7, res_2_4: 3 },
    framing: { face: 1, bust: 1 },
    dup: { groups: 1, images: 2 },
  })
  assert.deepEqual(model.status.map(({ value, percent }) => [value, percent]),
    [[3, 30], [5, 50], [2, 20]])
  assert.equal(model.passes.find((pass) => pass.key === 'scored').text, '4 of 10 · 40%')
  assert.equal(model.passes.find((pass) => pass.key === 'watermark_scanned').value, null)
  assert.equal(model.distributions[0].total, 10)
  assert.equal(model.kpis[0].value, '1')
})

test('duplicate KPIs headline the live unresolved queue, not historical groups', () => {
  const model = bankOverviewModel({
    counts: { total: 130, scanned: 130 },
    dup: { groups: 65, images: 130, unresolved: 0 },
    semantic_dup: { groups: 65, images: 130, unresolved: 0 },
  })
  const duplicates = model.kpis.find((kpi) => kpi.label === 'Duplicates')
  const sameShots = model.kpis.find((kpi) => kpi.label === 'Same shots')

  assert.equal(duplicates.value, '0')
  assert.equal(duplicates.detail, 'groups remaining to resolve')
  assert.equal(sameShots.value, '0')
  assert.equal(sameShots.detail, 'groups remaining to resolve')
})

test('duplicate KPIs expose non-zero remaining work for both analysis stages', () => {
  const model = bankOverviewModel({
    counts: { total: 130, scanned: 130 },
    dup: { groups: 65, images: 130, unresolved: 3 },
    semantic_dup: { groups: 40, images: 90, unresolved: 3 },
  })

  assert.equal(model.kpis.find((kpi) => kpi.label === 'Duplicates').value, '3')
  assert.equal(model.kpis.find((kpi) => kpi.label === 'Same shots').value, '3')
})

test('duplicate KPIs remain compatible with legacy payloads lacking unresolved', () => {
  const model = bankOverviewModel({
    counts: { total: 4, scanned: 4 },
    dup: { groups: 2, images: 4 },
    semantic_dup: { groups: 1, images: 2 },
  })

  assert.equal(model.kpis.find((kpi) => kpi.label === 'Duplicates').value, '2')
  assert.equal(model.kpis.find((kpi) => kpi.label === 'Same shots').value, '1')
  assert.equal(model.kpis.find((kpi) => kpi.label === 'Same shots').detail,
    'group remaining to resolve')
})

test('Faces coverage is separate from a completed no-face angle result', () => {
  const model = bankOverviewModel({
    counts: { total: 6, angle_measured: 0, angle_backfillable: 0 },
    faces_scanned: 6,
  })
  const faces = model.passes.find((pass) => pass.key === 'faces_scanned')
  const angles = model.passes.find((pass) => pass.key === 'angle_measured')
  assert.equal(faces.text, '6 of 6 · 100%')
  assert.equal(angles.value, 0)
  assert.equal(angles.text, '6 face-checked · no measurable head angle')
  assert.doesNotMatch(angles.text, /Run Faces/)
})

test('partial angles state reports both stored coverage and available backfill', () => {
  const model = bankOverviewModel({
    counts: { total: 10, angle_measured: 3, angle_backfillable: 2 },
    faces_scanned: 8,
  })
  const faces = model.passes.find((pass) => pass.key === 'faces_scanned')
  const angles = model.passes.find((pass) => pass.key === 'angle_measured')
  assert.equal(faces.text, '8 of 10 · 80%')
  assert.equal(angles.percent, 30)
  assert.equal(angles.text, '3 of 10 · 30% · Backfill available for 2 images')
})

test('the exact four-megapixel tier is labelled as inclusive', () => {
  const model = bankOverviewModel({
    counts: { total: 1 }, res_buckets: { res_gt_4: 1 },
  })
  assert.equal(model.distributions[0].rows[0].label, '≥ 4 MP')
})

test('tiny non-zero segments keep an exact width even when their label rounds to zero', () => {
  assert.equal(widthPercent(1, 1000), 0.1)
  assert.equal(displayPercent(1, 1000), 0)
  const overview = bankOverviewModel({
    counts: { total: 1000, keep: 1, pending: 999, reject: 0 },
  })
  assert.equal(overview.status[0].widthPercent, 0.1)
  assert.equal(overview.status[0].percent, 0)
  assert.equal(overview.status.reduce((sum, row) => sum + row.widthPercent, 0), 100)

  const list = bankListOverview({ total: 1000, keep: 1, reject: 0, scanned: 1 })
  assert.equal(list.status[0].widthPercent, 0.1)
  assert.equal(list.status[0].percent, 0)
})

test('exhaustive thirds close at exactly 100 while labels stay independently rounded', () => {
  const model = bankOverviewModel({
    counts: { total: 3, keep: 1, pending: 1, reject: 1 },
    framing: { face: 1, bust: 1, body: 1 },
  })
  assert.deepEqual(model.status.map((row) => row.percent), [33, 33, 33])
  assert.equal(model.status.reduce((sum, row) => sum + row.widthPercent, 0), 100)
  const framing = model.distributions.find((item) => item.id === 'framing')
  assert.deepEqual(framing.rows.map((row) => row.percent), [33, 33, 33])
  assert.equal(framing.rows.reduce((sum, row) => sum + row.widthPercent, 0), 100)
})

test('overview only consumes the root bank payload and has no coverage/facet input', () => {
  const model = bankOverviewModel({
    counts: { total: 5, keep: 2, pending: 2, reject: 1, scanned: 5 },
    framing: { face: 5 },
    coverage: { total: 2, framing: { body: 2 } },
    facets: { framing: { back: 5 } },
  })
  const framing = model.distributions.find((item) => item.id === 'framing')
  assert.deepEqual(framing.rows.map((row) => [row.id, row.value]), [['face', 5]])
})

test('bank list derives undecided and scan coverage without another payload', () => {
  const model = bankListOverview({ total: 20, keep: 7, reject: 3, scanned: 5 })
  assert.deepEqual(model.status.map((row) => row.value), [7, 10, 3])
  assert.equal(model.scanText, '5 of 20 · 25%')
  assert.equal(bankListOverview({ total: 20, keep: 0, reject: 0, scanned: 0 }).scanText,
    'Not measured · Run Scan')
})

test('capped cluster payloads never pretend that the top 40 are an exact total', () => {
  const clusters = Array.from({ length: 40 }, (_, id) => ({ id, size: 2 }))
  const model = bankOverviewModel({
    counts: { total: 100, scored: 100 }, faces_scanned: 100,
    clusters, style_clusters: clusters,
  })
  assert.equal(model.kpis[2].value, '40+')
  assert.match(model.kpis[2].detail, /top groups shown/)
  assert.equal(model.kpis[3].value, '40+')
})

test('saved scores from an interrupted pass never pretend style grouping exists', () => {
  const model = bankOverviewModel({
    counts: { total: 12, scored: 7 },
    style_clusters: [],
  })
  const styles = model.kpis.find((kpi) => kpi.label === 'Styles')
  assert.equal(styles.value, '—')
  assert.equal(styles.detail, 'No style grouping recorded; finish/re-run Score')
})
