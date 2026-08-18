/* Task Center: pure helpers for the /tasks page and the nav badge.
   Pure on purpose: node --test cannot parse JSX, so the mapping and badge
   math — the parts easy to get subtly wrong — live here without a browser. */

export const EMPTY_OVERVIEW = { status: null, tasks: [], runningCount: 0,
                                failedCount: 0, failedIds: [] }

/* Tolerate anything the endpoint hands back rather than rendering
   "undefined" next to a count. `runningCount`/`failedCount` are DERIVED,
   never trusted from the payload. */
export function normalizeOverview(raw) {
  if (!raw || typeof raw !== 'object') return EMPTY_OVERVIEW
  const tasks = Array.isArray(raw.tasks) ? raw.tasks : []
  const failedIds = tasks.filter((t) => t.status === 'failed')
    .map((t) => String(t.job_id)).filter(Boolean)
  const runningCount = tasks.filter((t) =>
    ['pending', 'queued', 'awaiting_comfyui', 'processing', 'sent_to_comfy', 'running']
      .includes(t.status)).length
  return {
    status: raw.status && typeof raw.status === 'object' ? raw.status : null,
    tasks,
    runningCount,
    failedCount: failedIds.length,
    failedIds,
  }
}

/* The badge shows failures a user has NOT already seen. */
export function badgeCounts(overview, seenFailureIds) {
  const failed = overview.failedIds.filter((id) => !seenFailureIds.has(id)).length
  return { running: overview.runningCount, failed }
}

export const STATUS_LABEL = {
  pending: 'Queued',
  queued: 'Queued',
  awaiting_comfyui: 'Paused · waiting for ComfyUI',
  processing: 'Running',
  sent_to_comfy: 'Running',
  running: 'Running',
  completed: 'Done',
  failed: 'Failed',
  cancelled: 'Cancelled',
  stalled: 'Needs recovery',
}

/* Tailwind chip classes. Semantic only — green stays "done", red "failed",
   amber "running", sky "paused", slate "queued" (CLAUDE.md §UI rules). */
export const STATUS_CHIP = {
  pending: 'bg-slate-100 text-slate-600',
  queued: 'bg-slate-100 text-slate-600',
  awaiting_comfyui: 'bg-sky-50 text-sky-700',
  processing: 'bg-amber-50 text-amber-700',
  sent_to_comfy: 'bg-amber-50 text-amber-700',
  running: 'bg-amber-50 text-amber-700',
  completed: 'bg-emerald-50 text-emerald-700',
  failed: 'bg-red-50 text-red-700',
  cancelled: 'bg-slate-100 text-slate-500',
  stalled: 'bg-red-50 text-red-700',
}
