import { useState, useEffect, useCallback, useMemo, createContext, useContext } from 'react'
import { pushToast, dropToast, sweepToasts, toastLabel } from '../../utils/toastQueue'

// ── Context ──

const ToastContext = createContext(null)

let _nextId = 0

// Expiry is swept centrally instead of one setTimeout per emission: a merged
// repeat has to be able to PUSH its banner's expiry out, which a timer captured
// on the first emission's id cannot do. See utils/toastQueue.js.
const SWEEP_MS = 250

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const addToast = useCallback((message, type = 'info', duration = 4000) => {
    const id = ++_nextId
    const expiresAt = duration > 0 ? Date.now() + duration : null
    setToasts((prev) => pushToast(prev, { id, message, type, expiresAt }))
    return id
  }, [])

  const removeToast = useCallback((id) => {
    setToasts((prev) => dropToast(prev, id))
  }, [])

  useEffect(() => {
    const t = setInterval(() => {
      setToasts((prev) => {
        const next = sweepToasts(prev, Date.now())
        return next.length === prev.length ? prev : next
      })
    }, SWEEP_MS)
    return () => clearInterval(t)
  }, [])

  const toast = useMemo(() => ({
    info: (msg, d) => addToast(msg, 'info', d),
    success: (msg, d) => addToast(msg, 'success', d),
    error: (msg, d) => addToast(msg, 'error', d ?? 6000),
    warning: (msg, d) => addToast(msg, 'warning', d),
  }), [addToast])

  // Expose on window for non-React usage
  useEffect(() => { window.__adminToast = toast }, [toast])

  return (
    <ToastContext.Provider value={toast}>
      {children}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </ToastContext.Provider>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be inside ToastProvider')
  return ctx
}

// ── Renderer ──

const TYPE_STYLES = {
  info: 'border-blue-500/50 bg-blue-500/10 text-blue-300',
  success: 'border-green-500/50 bg-green-500/10 text-green-300',
  error: 'border-red-500/50 bg-red-500/10 text-red-300',
  warning: 'border-yellow-500/50 bg-yellow-500/10 text-yellow-300',
}

const ICONS = {
  info: '\u2139\uFE0F',
  success: '\u2705',
  error: '\u274C',
  warning: '\u26A0\uFE0F',
}

function ToastContainer({ toasts, onRemove }) {
  if (!toasts.length) return null

  // Plain positioning wrapper — NOT a live region. Each toast is its own live
  // region (per-type politeness), avoiding nested live regions (double-announce)
  // and per-new-toast re-announce-all.
  return (
    /* left-4 as well as right-4: at 400 px a max-w-sm card pinned only to the
       right still had to fit, and long messages pushed past the viewport.

       z-[10000] — ABOVE EVERY OVERLAY, and that is the whole point. At z-[100]
       this container sat *under* every modal and lightbox in the app (they run
       9990-9999), so a toast raised while a dialog was open rendered behind it:
       the app said something and nobody could read it. Callers worked around it
       by closing the dialog first, which throws away whatever the user had just
       typed into it. A notification that cannot be seen is worse than none —
       keep this the highest layer, and let TOAST_Z in Toast.contract.test.js
       fail the build if a new overlay ever climbs past it. */
    <div className="fixed top-4 right-4 left-4 sm:left-auto z-[10000] flex flex-col gap-2 sm:max-w-sm">
      {toasts.map((t) => (
        <div
          key={t.id}
          role={t.type === 'error' ? 'alert' : 'status'}
          aria-live={t.type === 'error' ? 'assertive' : 'polite'}
          aria-atomic="true"
          title={toastLabel(t)}
          className={`flex items-start gap-2 border rounded-lg px-4 py-3 shadow-lg backdrop-blur-sm animate-slideIn ${
            TYPE_STYLES[t.type] || TYPE_STYLES.info
          }`}
        >
          <span className="flex-shrink-0 mt-0.5">{ICONS[t.type] || ICONS.info}</span>
          {/* The message alone lives in the live region; the repeat counter is
              aria-hidden ON PURPOSE. It changes on every repeat, and a live
              region re-announces whenever its content changes — a banner
              merged twelve times would otherwise be twelve announcements,
              exactly the noise the merging was meant to remove. Sighted users
              see "(12×)", assistive tech hears the sentence once. */}
          <span className="text-sm flex-1 break-words">{t.message}</span>
          {(t.count || 1) > 1 && (
            <span aria-hidden="true"
              className="flex-shrink-0 self-start rounded-full bg-black/25 px-1.5 py-px text-[0.6875rem] font-semibold tabular-nums">
              {t.count}×
            </span>
          )}
          <button
            type="button"
            onClick={() => onRemove(t.id)}
            aria-label="Close notification"
            className="flex-shrink-0 text-content-muted hover:text-content ml-2"
          >
            <span aria-hidden="true">&times;</span>
          </button>
        </div>
      ))}
    </div>
  )
}
