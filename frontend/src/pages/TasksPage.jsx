import { useCallback, useMemo, useState } from 'react'
import { useNavigate } from 'react-router'
import { postJson } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { useTaskOverview } from '../hooks/useTaskOverview'
import { StatusStrip } from '../components/tasks/StatusStrip'
import { TaskRow } from '../components/tasks/TaskRow'

const ACTIVE = new Set(['pending', 'queued', 'awaiting_comfyui', 'processing',
                        'sent_to_comfy', 'running'])

export default function TasksPage() {
  const overview = useTaskOverview()
  const toast = useToast()
  const navigate = useNavigate()
  const [tab, setTab] = useState('active')
  const [kindFilter, setKindFilter] = useState('all')

  const tasks = useMemo(() => {
    let list = overview.tasks
    if (tab === 'active') list = list.filter((t) => ACTIVE.has(t.status))
    if (kindFilter !== 'all') list = list.filter((t) => t.kind === kindFilter)
    return list
  }, [overview.tasks, tab, kindFilter])

  const onCancel = useCallback(async (task) => {
    try {
      await postJson(`/api/tasks/${task.job_id}/cancel`, {})
    } catch { /* toast from fetchClient */ }
  }, [])

  const onRetry = useCallback(async (task) => {
    try {
      await postJson(`/api/tasks/${task.job_id}/retry`, {})
      toast?.success?.('Task requeued')
    } catch { /* toast from fetchClient */ }
  }, [toast])

  const onOpenResource = useCallback((resource) => {
    if (resource?.dataset_id == null) return
    try { localStorage.setItem('datasetCurrentId', String(resource.dataset_id)) } catch { /* ignore */ }
    window.dispatchEvent(new CustomEvent('lds:open-image', {
      detail: { dataset_id: resource.dataset_id, image_id: resource.image_id ?? null },
    }))
    navigate('/datasets')
  }, [navigate])

  const tabCls = (active) => 'rounded-full px-3 py-1 text-sm ' +
    (active ? 'bg-surface-raised text-content' : 'text-content-muted')

  return (
    <div className="mx-auto max-w-5xl px-4 py-6">
      <h1 className="text-xl font-semibold text-content">Task Center</h1>
      <div className="mt-4">
        <StatusStrip comfyui={overview.status?.comfyui}
                     summary={overview.status?.summary}
                     training={overview.status?.gpu?.training}
                     vision={overview.status?.gpu?.vision} />
      </div>
      <div className="mt-4 flex items-center gap-3">
        <button type="button" className={tabCls(tab === 'active')} onClick={() => setTab('active')}>
          In progress
        </button>
        <button type="button" className={tabCls(tab === 'all')} onClick={() => setTab('all')}>
          All
        </button>
        <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value)}
                className="ml-auto rounded-lg border border-border bg-surface px-2 py-1 text-sm text-content">
          <option value="all">All types</option>
          <option value="image">Images</option>
          <option value="training">Training</option>
          <option value="vision">Vision</option>
          <option value="topaz">Topaz</option>
        </select>
      </div>
      <div className="mt-4 overflow-hidden rounded-xl bg-surface">
        {tasks.length === 0 ? (
          <div className="px-4 py-12 text-center">
            <p className="text-sm text-content-muted">No tasks {tab === 'active' ? 'in progress' : ''}.</p>
            <p className="mt-1 text-xs text-content-subtle">
              Generate from a dataset, Studio or the Canvas — everything lands here.
            </p>
          </div>
        ) : (
          tasks.map((t) => (
            <TaskRow key={t.job_id} task={t}
                     onCancel={onCancel} onRetry={onRetry} onOpenResource={onOpenResource} />
          ))
        )}
      </div>
    </div>
  )
}
