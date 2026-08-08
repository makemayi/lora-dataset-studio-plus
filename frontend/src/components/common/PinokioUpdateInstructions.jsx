import {
  PINOKIO_UPDATE_STEPS,
  PINOKIO_UPDATE_GUIDE_URL,
} from '../settings/updateStatus'

/** Shared Pinokio update guidance. Pinokio starts and stops the server itself;
 * an in-app update would relaunch it in a process the launcher no longer
 * tracks, so it would show the app as stopped while the old one still held the
 * port. The Settings card and the global update banner render this exact same
 * component — three clicks, no terminal. */
export default function PinokioUpdateInstructions() {
  return (
    <div className="w-full space-y-2 rounded-md border border-sky-400/30 bg-sky-500/10 px-3 py-2 text-content">
      <p className="text-sm">
        This install is launched by Pinokio, which starts and stops the app. Update it from there:
      </p>
      <ol className="space-y-1" aria-label="Pinokio update steps">
        {PINOKIO_UPDATE_STEPS.map((step, index) => (
          <li key={step} className="flex min-w-0 items-start gap-2">
            <span aria-hidden className="w-4 shrink-0 text-right text-xs text-content-subtle">{index + 1}.</span>
            <span className="min-w-0 text-sm text-content">{step}</span>
          </li>
        ))}
      </ol>
      <p className="text-xs text-content-muted">
        Pinokio&rsquo;s Update runs the same fast-forward pull as this app, then reinstalls changed
        requirements; your datasets, config and Image Bank stay in place.{' '}
        <a href={PINOKIO_UPDATE_GUIDE_URL} target="_blank" rel="noreferrer"
          className="font-medium text-sky-300 underline hover:text-sky-200">
          Pinokio install guide ↗
        </a>
      </p>
    </div>
  )
}
