import { CARD_SURFACE } from '../common/surfaces'

export function StatusStrip({ comfyui, summary, training, vision }) {
  const reachable = Boolean(comfyui?.reachable)
  const paused = summary?.paused ?? 0
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
      <div className={`${CARD_SURFACE} p-3`}>
        <div className="text-xs text-content-muted">ComfyUI</div>
        <div className="text-sm font-medium text-content">
          {reachable ? '● Online' : 'Offline'}
        </div>
        <div className="text-xs text-content-subtle">
          {reachable
            ? `${summary?.running ?? 0} running · ${summary?.queued ?? 0} queued`
            : `${paused} paused · resumes automatically`}
        </div>
      </div>
      <div className={`${CARD_SURFACE} p-3`}>
        <div className="text-xs text-content-muted">GPU</div>
        <div className="text-sm font-medium text-content">
          {training ? 'Training in progress' : vision ? 'Vision inference' : 'Idle'}
        </div>
        <div className="text-xs text-content-subtle">
          {training ? 'Image jobs yield the GPU' : vision ? 'Ollama lane busy' : 'GPU free for generation'}
        </div>
      </div>
      <div className={`${CARD_SURFACE} p-3`}>
        <div className="text-xs text-content-muted">Queue</div>
        <div className="text-sm font-medium text-content">
          {summary?.queued ?? 0} queued · {paused} paused · {summary?.running ?? 0} running
        </div>
        <div className="text-xs text-content-subtle">
          {summary?.today_done ?? 0} done today · {summary?.today_failed ?? 0} failed
        </div>
      </div>
    </div>
  )
}
