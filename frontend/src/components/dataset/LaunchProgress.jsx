import { launchProgressView } from '../../utils/launchProgress';

/* The minutes between the click and step 1 of a cloud run, made readable: the
   ordered launch steps the monitor is walking, which one it is on, how long the
   whole launch has been running, and — while a pod boots — how long it is
   allowed to. Renting a machine legitimately takes minutes; before this, so did
   a launch that had already failed, and the two looked identical.

   aria-live is deliberately absent: the elapsed figure changes on every poll and
   a region that re-announces itself every few seconds is unusable. The steps are
   a plain list a screen reader reads on demand, and each carries its state in
   words, never in colour alone. */
export default function LaunchProgress({ launch }) {
  const view = launchProgressView(launch);
  if (!view) return null;
  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-sky-500/30 bg-sky-500/5 px-2.5 py-2">
      <p className="m-0 text-sky-200 text-[0.6875rem] font-semibold">☁ {view.headline}</p>
      <ol className="m-0 list-none p-0 flex flex-col gap-0.5">
        {view.steps.map((s) => (
          <li key={s.key}
            className={`flex items-start gap-1.5 text-[0.625rem] leading-snug ${
              s.state === 'active' ? 'text-content font-semibold'
                : s.state === 'done' ? 'text-content-muted' : 'text-content-subtle'}`}>
            <span aria-hidden className="w-3 shrink-0 text-center">
              {s.state === 'done' ? '✓' : s.state === 'active' ? '▶' : '·'}
            </span>
            {/* min-w-0 + break-words: the labels fold instead of overflowing on
                a 400 px phone — the screen this was reported from. */}
            <span className="min-w-0 break-words">
              {s.label}
              <span className="sr-only">
                {s.state === 'done' ? ' — done' : s.state === 'active' ? ' — in progress' : ' — pending'}
              </span>
              {s.state === 'active' && view.detail && (
                <span className="block font-normal text-content-muted">{view.detail}</span>
              )}
            </span>
          </li>
        ))}
      </ol>
      {view.note && (
        <p className="m-0 text-content-subtle text-[0.625rem] leading-snug">{view.note}</p>
      )}
    </div>
  );
}
