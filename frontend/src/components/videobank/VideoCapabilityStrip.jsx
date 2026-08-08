import { videoCapabilityNotice } from './videoCapability'

/** 🎬 What the video extra is missing — piece by piece, with what still works.
 *
 * The banner this replaces is "Video is unavailable", and that sentence is the
 * exact defect this lane was built to remove: it makes the user reinstall the
 * wrong thing, and it hides the fact that a bank with no encoder is entirely
 * usable right up to the moment it produces files.
 *
 * Silent on a healthy install: a green "all three present" strip on every visit
 * is what teaches people to skip this box on the day it says something.
 */
export default function VideoCapabilityStrip({ capability, compact = false }) {
  const notice = videoCapabilityNotice(capability)
  if (!notice) return null
  return (
    <section aria-label="Video tools status"
      className="rounded-lg border border-amber-500/50 bg-amber-500/10 p-3 text-sm text-amber-100">
      <p className="font-semibold">
        <span aria-hidden>⚠</span> {notice.headline}
      </p>
      {/* The half that keeps the bank usable. Deliberately above the fix list:
          "you can still do X" outranks "install Y" for someone who just wants
          to get on with triaging. */}
      <p className="mt-1 text-amber-100/90">{notice.stillWorks}</p>
      {!compact && (
        <ul className="mt-2 space-y-1.5">
          {notice.pieces.map((p) => (
            <li key={p.key} className="flex flex-wrap gap-x-1.5 text-xs">
              <span className="font-semibold text-amber-50">✗ {p.label}</span>
              <span className="text-amber-100/80">{p.blurb}</span>
              <span className="basis-full text-amber-200">→ {p.fix}</span>
            </li>
          ))}
        </ul>
      )}
      {!compact && notice.detail && (
        /* The server's own sentence, verbatim — it names the exact package, and
           a paraphrase is how someone pip-installs the wrong one. */
        <p className="mt-2 font-mono text-[0.6875rem] text-amber-200/80 break-words">
          {notice.detail}
        </p>
      )}
    </section>
  )
}
