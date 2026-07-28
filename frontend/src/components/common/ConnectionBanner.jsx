import { useConnectionStatus } from '../../hooks/useConnectionStatus'

/**
 * 📡 The app's single, quiet voice for "I can't reach the server right now".
 *
 * Replaces the stack of "Connection lost" toasts a failing poll used to emit.
 * It is deliberately ONE node that mounts when the outage opens and unmounts
 * when it closes: `role="status"` announces it once, and nothing inside it
 * changes while it is up (no ticking counter), so it is never re-announced.
 *
 * The second sentence is the point of the whole fix: a bank pass, a caption
 * batch and a training run all live in server-side threads, and the page
 * losing contact says nothing about them. Silence used to read as "it stopped".
 */
export default function ConnectionBanner() {
  const { online } = useConnectionStatus()
  if (online) return null
  return (
    <div className="mx-auto max-w-5xl px-4 pt-3">
      <div role="status"
        className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-sm">
        <span aria-hidden>📡</span>
        <span className="font-medium text-content">Offline — reconnecting…</span>
        <span className="basis-full text-xs text-content-subtle sm:basis-auto">
          Anything already running keeps running on the server.
        </span>
      </div>
    </div>
  )
}
