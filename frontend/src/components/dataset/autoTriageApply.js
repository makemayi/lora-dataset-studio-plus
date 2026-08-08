function successfulBatchResult(value) {
  // useDataset.batchImages returns a number on success and null on an explicit
  // API failure. Zero is still a completed request (the rows may have changed
  // between the render and the click), so only a number earns success.
  return typeof value === 'number' && Number.isFinite(value)
}

/**
 * Latest-run guard for Auto-triage.
 *
 * React state alone cannot identify which async Apply owns the result: changing
 * dataset resets `applying`, then the promise from the previous dataset may
 * settle and write into the new session. The gate is synchronous, so a render
 * with a new dataset invalidates the old generation before any effect runs.
 * In-flight tokens stay registered by dataset, however: A -> B -> A must not
 * start a second write to A while the first request is still settling.
 */
export function createAutoTriageRunGate(initialDatasetId) {
  let datasetId = initialDatasetId
  let generation = 0
  let nextRunId = 0
  const inFlightByDataset = new Map()

  const syncDataset = (nextDatasetId) => {
    if (Object.is(nextDatasetId, datasetId)) return false
    datasetId = nextDatasetId
    generation += 1
    return true
  }

  const isCurrent = (token) => Boolean(
    token
    && token === inFlightByDataset.get(token.datasetId)
    && token.generation === generation
    && Object.is(token.datasetId, datasetId)
  )

  const begin = (nextDatasetId = datasetId) => {
    syncDataset(nextDatasetId)
    if (inFlightByDataset.has(datasetId)) return null
    const token = Object.freeze({ datasetId, generation, runId: ++nextRunId })
    inFlightByDataset.set(datasetId, token)
    return token
  }

  const hasActive = (candidateDatasetId = datasetId) => inFlightByDataset.has(candidateDatasetId)

  const commit = (token, callback) => {
    if (!isCurrent(token)) return false
    callback()
    return true
  }

  const finish = (token, callback) => {
    if (!token || token !== inFlightByDataset.get(token.datasetId)) return false
    const current = isCurrent(token)
    inFlightByDataset.delete(token.datasetId)
    if (current) callback()
    return true
  }

  return {
    // Stable diagnostic identity: function names are mangled in the Vite bundle,
    // while release verification still needs to prove this guard shipped.
    kind: 'createAutoTriageRunGate',
    syncDataset,
    begin,
    hasActive,
    isCurrent,
    commit,
    finish,
  }
}

/** Token-aware immutable reducer for DatasetGrid's shared busy registry. */
export function updateAutoTriageRuns(previous, datasetId, token, active) {
  if (active) {
    if (previous.get(datasetId) === token) return previous
    const next = new Map(previous)
    next.set(datasetId, token)
    return next
  }
  if (previous.get(datasetId) !== token) return previous
  const next = new Map(previous)
  next.delete(datasetId)
  return next
}

/** Ownership that is safe to publish after a run result. A completed batch was
 * persisted even when the following batch failed; targets already in their
 * requested status were no-ops and keep their previous session ownership. */
export function autoTriageOwnershipForResult({
  result,
  keepTargets = [],
  rejectTargets = [],
  previousOwnership = {},
}) {
  const completed = new Set(result?.completed || [])
  const next = {}
  const record = (action, targets) => {
    const persisted = result?.ok === true || completed.has(action)
    targets.forEach((image) => {
      if (image?.id == null) return
      if (persisted) {
        next[image.id] = action
      } else if (previousOwnership?.[image.id] === image.status) {
        // A failed Re-apply flip leaves the row at its previous status. Retain
        // that old ownership only while the live row still proves it; assigning
        // the new target here would claim a write the server refused.
        next[image.id] = image.status
      }
    })
  }
  record('keep', keepTargets)
  record('reject', rejectTargets)
  return next
}

/** Run Auto-triage's two status batches as one UI transaction. The server calls
 * remain independent, so the first may have completed when the second fails;
 * callers may preserve ownership for `completed`, but must only record the
 * whole run as "applied" after this returns ok=true. */
export async function runAutoTriageBatches({
  onBatch,
  keepIds = [],
  rejectIds = [],
  shouldContinue = () => true,
}) {
  const completed = []
  let attempting = null
  try {
    if (!shouldContinue()) return { ok: false, stale: true, completed }
    if (keepIds.length) {
      attempting = 'keep'
      const kept = await onBatch(keepIds, 'keep', { silent: true })
      if (!successfulBatchResult(kept)) return { ok: false, failed: 'keep', completed }
      completed.push('keep')
    }
    // Navigation cannot abort a POST already on the wire, but it can prevent
    // the second half of the old transaction from starting on a stale dataset.
    if (!shouldContinue()) return { ok: false, stale: true, completed }
    if (rejectIds.length) {
      attempting = 'reject'
      const rejected = await onBatch(rejectIds, 'reject', { silent: true })
      if (!successfulBatchResult(rejected)) return { ok: false, failed: 'reject', completed }
      completed.push('reject')
    }
    return { ok: true, completed }
  } catch (error) {
    return { ok: false, failed: attempting, completed, error }
  }
}

export function autoTriageFailureMessage(result) {
  if (!result || result.ok) return ''
  return result.completed?.length
    ? 'Auto-triage stopped part-way through. Some changes are already saved; retry the remaining images.'
    : 'Auto-triage could not be applied. No successful batch was recorded; try again.'
}
