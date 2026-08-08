const RESOLUTION_BUCKETS = [
  { id: 'res_lt_025', label: '< 0.25 MP' },
  { id: 'res_025_1', label: '0.25–1 MP' },
  { id: 'res_1_2', label: '1–2 MP' },
  { id: 'res_2_4', label: '2–4 MP' },
  { id: 'res_gt_4', label: '≥ 4 MP' },
]

const FRAMING_BUCKETS = [
  { id: 'face', label: 'Face' },
  { id: 'bust', label: 'Bust' },
  { id: 'body', label: 'Body' },
  { id: 'back', label: 'Back' },
  { id: 'unknown', label: 'Unknown' },
]

const MEDIUM_BUCKETS = [
  { id: 'photo', label: 'Photo' },
  { id: 'anime', label: 'Anime' },
  { id: 'render3d', label: '3D render' },
  { id: 'illustration', label: 'Illustration' },
  { id: 'unsure', label: 'Unsure' },
]

const ORIGIN_BUCKETS = [
  { id: 'ai', label: 'AI' },
  { id: 'camera', label: 'Camera' },
  { id: 'unknown', label: 'Unknown' },
]

function count(value) {
  if (value == null || value === '') return null
  const n = Number(value)
  return Number.isFinite(n) && n >= 0 ? Math.trunc(n) : null
}

export function bankListOverview(bank) {
  const total = count(bank?.total)
  const keep = count(bank?.keep) ?? (total == null ? null : 0)
  const reject = count(bank?.reject) ?? (total == null ? null : 0)
  const pending = total == null ? null : Math.max(0, total - keep - reject)
  const scanned = count(bank?.scanned)
  return {
    total,
    scanned,
    scanPercent: scanned > 0 && total > 0 ? displayPercent(scanned, total) : null,
    scanWidthPercent: scanned > 0 && total > 0 ? widthPercent(scanned, total) : null,
    scanText: scanned > 0 && total > 0
      ? `${scanned.toLocaleString()} of ${total.toLocaleString()} · ${displayPercent(scanned, total)}%`
      : scanned > 0 && total == null
        ? `${scanned.toLocaleString()} measured · total unavailable`
        : 'Not measured · Run Scan',
    status: percentageRows([
      { id: 'keep', label: 'Kept', value: keep },
      { id: 'pending', label: 'Undecided', value: pending },
      { id: 'reject', label: 'Rejected', value: reject },
    ], total),
  }
}

export function widthPercent(value, total) {
  if (!(total > 0) || value == null) return null
  return Math.max(0, Math.min(100, (100 * value) / total))
}

export function displayPercent(value, total) {
  const width = widthPercent(value, total)
  return width == null ? null : Math.round(width)
}

/** Add exact CSS widths and separately rounded human labels. When the rows are
 * an exhaustive integer partition, the last non-zero segment receives the
 * floating-point remainder so the widths add to exactly 100 (no hairline gap). */
function percentageRows(rows, total) {
  const exhaustive = total > 0
    && rows.every((row) => row.value != null)
    && rows.reduce((sum, row) => sum + row.value, 0) === total
  const lastPositive = exhaustive
    ? rows.reduce((last, row, index) => (row.value > 0 ? index : last), -1)
    : -1
  let usedWidth = 0
  return rows.map((row, index) => {
    let width = widthPercent(row.value, total)
    if (index === lastPositive) width = Math.max(0, 100 - usedWidth)
    if (width != null) usedWidth += width
    return {
      ...row,
      percent: displayPercent(row.value, total),
      widthPercent: width,
    }
  })
}

function distribution(values, buckets, emptyHint) {
  const rows = buckets
    .map((bucket) => ({ ...bucket, value: count(values?.[bucket.id]) || 0 }))
    .filter((row) => row.value > 0)
  const total = rows.reduce((sum, row) => sum + row.value, 0)
  return {
    rows: percentageRows(rows, total),
    total,
    emptyHint,
  }
}

function passCoverage(counts, total, key, label, runLabel) {
  const value = count(counts?.[key])
  if (total == null && value > 0) {
    return { key, label, value, total, percent: null,
      text: `${value.toLocaleString()} measured · total unavailable` }
  }
  if (!(total > 0) || !(value > 0)) {
    return { key, label, value: null, total, percent: null,
      text: `Not measured · ${runLabel}` }
  }
  const pct = displayPercent(value, total)
  return { key, label, value, total, percent: pct, widthPercent: widthPercent(value, total),
    text: `${value.toLocaleString()} of ${total.toLocaleString()} · ${pct}%` }
}

function headAngleCoverage(payload, total) {
  const counts = payload?.counts || {}
  const measured = count(counts.angle_measured) || 0
  const facesScanned = count(payload?.faces_scanned) || 0
  const backfillable = count(counts.angle_backfillable) || 0
  const base = { key: 'angle_measured', label: 'Head angle', value: measured,
    total,
    percent: measured > 0 ? displayPercent(measured, total) : null,
    widthPercent: measured > 0 ? widthPercent(measured, total) : null }

  if (backfillable > 0) {
    const current = measured > 0
      ? passCoverage(counts, total, 'angle_measured', 'Head angle', 'Run Faces')
      : { ...base, text: 'No stored head angles yet' }
    return { ...current,
      text: `${current.text} · Backfill available for ${backfillable} image${backfillable === 1 ? '' : 's'}` }
  }
  if (measured > 0) {
    return passCoverage(counts, total, 'angle_measured', 'Head angle', 'Run Faces')
  }
  if (facesScanned > 0) {
    return { ...base,
      text: `${facesScanned.toLocaleString()} face-checked · no measurable head angle` }
  }
  return { ...base, value: null, text: 'Not measured · Run Faces' }
}

function groupKpi(label, data, available, emptyText) {
  const groups = count(data?.groups) || 0
  const images = count(data?.images) || 0
  if (!available && groups === 0 && images === 0) {
    return { label, value: '—', detail: emptyText }
  }
  // `groups` and `images` describe everything the last analysis pass found,
  // including groups the user has already resolved.  `unresolved` is the live
  // curation queue and is therefore the only honest headline number.  Fall
  // back to `groups` for payloads produced by an older backend.
  const remaining = count(data?.unresolved) ?? groups
  return {
    label,
    value: remaining.toLocaleString(),
    detail: `${remaining === 1 ? 'group' : 'groups'} remaining to resolve`,
  }
}

function clusterKpi(label, clusters, available, emptyText) {
  const rows = Array.isArray(clusters) ? clusters : []
  if (!available && rows.length === 0) return { label, value: '—', detail: emptyText }
  const shownImages = rows.reduce((sum, item) => sum + (count(item?.size) || 0), 0)
  // The workspace payload intentionally caps both lists at the 40 largest
  // groups. At the cap we know there are AT LEAST 40, never that there are
  // exactly 40 or that the displayed image sum covers the whole bank.
  if (rows.length >= 40) {
    return { label, value: '40+',
      detail: `top groups shown · ${shownImages.toLocaleString()}+ images in shown groups` }
  }
  return { label, value: rows.length.toLocaleString(),
    detail: `${rows.length === 1 ? 'group' : 'groups'} · ${shownImages.toLocaleString()} clustered image${shownImages === 1 ? '' : 's'}` }
}

/**
 * Bank-wide, read-only view model. It only reads GET /api/bank/:id; filtered
 * facets and the kept-only /coverage endpoint must never enter this summary.
 */
export function bankOverviewModel(payload) {
  if (!payload || !payload.counts || typeof payload.counts !== 'object') {
    return { available: false, total: null, status: [], passes: [], distributions: [], kpis: [] }
  }
  const counts = payload.counts
  const total = count(counts.total)
  const statusValue = (key) => count(counts[key]) ?? (total == null ? null : 0)
  const status = percentageRows([
    { id: 'keep', label: 'Kept', value: statusValue('keep') },
    { id: 'pending', label: 'Undecided', value: statusValue('pending') },
    { id: 'reject', label: 'Rejected', value: statusValue('reject') },
  ], total)

  return {
    available: true,
    total,
    status,
    passes: [
      passCoverage(counts, total, 'scanned', 'Quality', 'Run Scan'),
      passCoverage(counts, total, 'scored', 'Score', 'Run Score'),
      passCoverage(counts, total, 'watermark_scanned', 'Watermarks', 'Run Watermarks'),
      passCoverage(counts, total, 'framing_classified', 'Framing', 'Run Framing'),
      passCoverage(counts, total, 'medium_classified', 'Medium', 'Run Medium'),
      passCoverage({ faces_scanned: payload?.faces_scanned }, total,
        'faces_scanned', 'Faces', 'Run Faces'),
      headAngleCoverage(payload, total),
    ],
    distributions: [
      { id: 'resolution', label: 'Resolution',
        ...distribution(payload?.res_buckets, RESOLUTION_BUCKETS, 'Not measured · Run Scan') },
      { id: 'framing', label: 'Framing',
        ...distribution(payload?.framing, FRAMING_BUCKETS, 'Not measured · Run Framing') },
      { id: 'medium', label: 'Medium',
        ...distribution(payload?.mediums, MEDIUM_BUCKETS, 'Not measured · Run Score, then Medium') },
      { id: 'origin', label: 'Origin',
        ...distribution(payload?.origins, ORIGIN_BUCKETS, 'Not measured · Run Scan') },
    ],
    kpis: [
      groupKpi('Duplicates', payload?.dup, (count(counts.scanned) || 0) > 0,
        'Not measured · Run Scan'),
      groupKpi('Same shots', payload?.semantic_dup,
        (count(payload?.semantic_dup?.groups) || 0) > 0,
        'No groups recorded'),
      clusterKpi('People', payload?.clusters,
        (count(payload?.faces_scanned) || 0) > 0, 'Not measured · Run Faces'),
      clusterKpi('Styles', payload?.style_clusters,
        Array.isArray(payload?.style_clusters) && payload.style_clusters.length > 0,
        'No style grouping recorded; finish/re-run Score'),
    ],
  }
}
