import { useCallback, useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from './Toast'
import { autoClearedMessage, recoveryBannerModel } from '../../utils/comfyRecovery'

/**
 * 🛟 The app-wide way out of a stalled ComfyUI job.
 *
 * The recovery barrier is global — one paused prompt blocks EVERY local
 * generation — but until now its only resolution was the Stop button of the
 * dataset that happened to own the job. A user working on any other dataset hit
 * a refusal that named no dataset, no job and no button, did exactly what the
 * message said (restart ComfyUI), and stayed blocked.
 *
 * So: mounted once in the shell, it names what is stuck and clears it in one
 * click from wherever the user is. Most of the time it never appears at all —
 * the server clears provable cases by itself and this only says so.
 */
const POLL_MS = 20000

export default function ComfyRecoveryBanner() {
  const toast = useToast()
  const [state, setState] = useState(null)
  const [busy, setBusy] = useState(false)
  const seenNoticeRef = useRef(null)
  const aliveRef = useRef(true)

  const poll = useCallback(async () => {
    try {
      // background: a periodic poll must not toast when the server blinks —
      // the offline banner already owns that story.
      const data = await apiFetch('/api/system/comfyui-recovery', { background: true })
      if (!aliveRef.current) return data
      setState(data)
      const notice = autoClearedMessage(data, seenNoticeRef.current)
      if (notice) {
        seenNoticeRef.current = notice.id
        toast.success(notice.message)
      }
      return data
    } catch {
      // An older backend without this route, or a server that is down: the
      // banner simply stays quiet. It is not the outage reporter.
      return null
    }
  }, [toast])

  useEffect(() => {
    aliveRef.current = true
    poll()
    const timer = setInterval(poll, POLL_MS)
    // A refusal carrying `comfyui_recovery_required` is the one moment the user
    // is certainly looking: show the way out immediately instead of up to 20 s
    // later. fetchClient fires this on every such 409, wherever it came from.
    const onBlocked = () => poll()
    window.addEventListener('lds:comfyui-recovery-required', onBlocked)
    return () => {
      aliveRef.current = false
      clearInterval(timer)
      window.removeEventListener('lds:comfyui-recovery-required', onBlocked)
    }
  }, [poll])

  const model = recoveryBannerModel(state)
  if (!model) return null

  const clearIt = async () => {
    setBusy(true)
    try {
      const res = await postJson('/api/system/comfyui-recovery/resolve',
        { confirmed_comfyui_restart: true })
      if (res?.already_clear) toast.success('Nothing left to clear — you can generate again.')
      else toast.success('Paused job cleared. You can generate again.')
      await poll()
    } catch (e) {
      // The server refuses anything it cannot prove or identify, and its message
      // says which — surface it verbatim rather than a generic failure.
      toast.error(e?.message || 'The paused job could not be cleared.')
      await poll()
    } finally {
      setBusy(false)
    }
  }

  // No route opens a specific dataset (the workspace remembers its selection in
  // localStorage), so pointing at one means writing that selection and letting
  // the app boot on it. Only ever on an explicit click, in a state where the
  // user is already stuck.
  const openOwningDataset = () => {
    try { localStorage.setItem('datasetCurrentId', String(model.datasetId)) } catch { /* private mode */ }
    window.location.hash = '#/datasets'
    window.location.reload()
  }

  const warning = model.tone === 'warning'
  return (
    <div className="mx-auto max-w-5xl px-4 pt-3">
      <div role="status"
        className={`rounded-lg border px-3 py-2 text-sm ${warning
          ? 'border-amber-400/40 bg-amber-500/10'
          : 'border-rose-400/40 bg-rose-500/10'}`}>
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
          <span aria-hidden>{warning ? '⏸️' : '⚠️'}</span>
          <span className="font-medium text-content">{model.headline}</span>
        </div>
        <p className="mt-1 text-xs text-content-subtle">{model.detail}</p>
        {/* Stacks on a 400 px screen, sits on one row from `sm` up. */}
        <div className="mt-2 flex flex-col gap-2 sm:flex-row sm:items-center">
          {model.canConfirm && (
            <button type="button" onClick={clearIt} disabled={busy}
              className="rounded-md border border-amber-400/50 bg-amber-500/20 px-3 py-1.5
                         text-xs font-medium text-content hover:bg-amber-500/30
                         disabled:cursor-not-allowed disabled:opacity-60">
              {busy ? 'Clearing…' : model.actionLabel}
            </button>
          )}
          {model.datasetId != null && (
            <button type="button" onClick={openOwningDataset}
              className="text-left text-xs text-content-subtle underline hover:text-content">
              Open {model.datasetName ? `“${model.datasetName}”` : 'that dataset'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
