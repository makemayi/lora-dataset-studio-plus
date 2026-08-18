import { CARD_SURFACE } from '../common/surfaces'
import { summarizeRejections } from './videoFrameReasons'

/* The completion report for a frame-promotion job: how many frames each clip
   contributed and why the rest were rejected, with a jump to the clip. */
export default function FramePromoteSummary({ result, onClose, onOpenClip }) {
  const totals = summarizeRejections(result?.totals)
  const rows = (result?.per_clip || []).map((row) => ({
    ...row, rejections: summarizeRejections(row.rejected),
  }))
  const rejectedTotal = totals.reduce((a, t) => a + t.count, 0)
  return (
    <div role="dialog" aria-modal="true" aria-label="Frame extraction summary"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-3 sm:p-4">
      <div className="max-h-[85vh] w-full max-w-xl space-y-3 overflow-y-auto rounded-2xl bg-surface-overlay/85 p-4 shadow-2xl backdrop-blur-md sm:p-5">
        <h2 className="text-base font-bold text-content">
          ✅ {result?.frames ?? 0} frame{result?.frames === 1 ? '' : 's'} imported
        </h2>
        {rejectedTotal > 0 && (
          <p className="text-sm text-content-muted">
            {rejectedTotal} rejected —{' '}
            {totals.map((t) => `${t.count} ${t.label.toLowerCase()}`).join(', ')}
          </p>
        )}
        <div className="space-y-1.5">
          {rows.map((r) => (
            <div key={r.clip_id}
              className={`${CARD_SURFACE} flex items-center gap-2 p-2 text-sm`}>
              <span className="font-medium text-content">Clip #{r.clip_id}</span>
              <span className="min-w-0 text-content-muted">
                {r.picked} frame{r.picked === 1 ? '' : 's'}
                {r.rejections.length > 0 &&
                  ` · rejected ${r.rejections.map((t) => `${t.count} ${t.label.toLowerCase()}`).join(', ')}`}
              </span>
              <button type="button"
                className="ml-auto shrink-0 rounded-full bg-surface-raised px-2 py-1 text-xs text-content"
                onClick={() => onOpenClip?.(r.clip_id)}>
                View
              </button>
            </div>
          ))}
          {rows.length === 0 && (
            <p className="py-6 text-center text-sm text-content-muted">
              Nothing to report — every clip contributed what was asked.
            </p>
          )}
        </div>
        <div className="flex justify-end">
          <button type="button" onClick={onClose}
            className="rounded-full px-3 py-1.5 text-sm text-content-muted hover:text-content">
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
