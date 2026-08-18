import { STATUS_CHIP, STATUS_LABEL } from '../../utils/taskOverview'
import { QUIET_BUTTON } from '../common/surfaces'

export function TaskRow({ task, onCancel, onRetry, onOpenResource }) {
  const chip = STATUS_CHIP[task.status] || STATUS_CHIP.pending
  const label = STATUS_LABEL[task.status] || task.status
  const { resource } = task
  const hasResource = resource && resource.dataset_id !== null && resource.dataset_id !== undefined
  return (
    <div className="flex items-center gap-3 border-b border-border px-4 py-3 last:border-b-0">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium text-content">{task.title}</span>
          <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${chip}`}>
            {label}
          </span>
        </div>
        <div className="mt-0.5 flex items-center gap-2 text-xs text-content-muted">
          {hasResource ? (
            <button type="button" className={QUIET_BUTTON} onClick={() => onOpenResource?.(resource)}>
              {resource.type === 'image' ? `Image #${resource.image_id}` : 'Dataset'}
            </button>
          ) : (
            <span className="text-content-subtle">Unknown source</span>
          )}
          {task.progress && <span>{task.progress}</span>}
          {task.error && <span className="truncate text-red-600">{task.error}</span>}
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {task.actions?.includes('cancel') && (
          <button type="button" className={QUIET_BUTTON} onClick={() => onCancel?.(task)}>
            Cancel
          </button>
        )}
        {task.actions?.includes('retry') && (
          <button type="button" className={QUIET_BUTTON} onClick={() => onRetry?.(task)}>
            Retry
          </button>
        )}
      </div>
    </div>
  )
}
