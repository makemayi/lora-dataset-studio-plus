/**
 * The "there is nothing here yet" block.
 *
 * Every list in the app had grown its own one-liner — a single grey sentence
 * floating where a grid of cards will be. That reads as a page that failed to
 * load, not as a page waiting for you: on a dark screen, one line of
 * `text-content-muted` at 14px is indistinguishable from a loading state, and
 * it never says what to do next.
 *
 * One block instead: a drawn glyph in a soft disc, a title in content weight,
 * one sentence of guidance, and — when the page has one — the action itself,
 * so the answer is a click away rather than a hunt back up the page.
 *
 * Deliberately NOT a card: it sits inside the space its future content will
 * occupy, so it must not look like one more panel stacked on the page. The
 * dashed hairline is the exception the surface rules name — it marks a slot
 * that is empty rather than a surface that holds something.
 */
export default function EmptyState({ icon, title, children, action }) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-border bg-surface px-4 py-10 text-center">
      {icon && (
        <span aria-hidden
          className="grid h-11 w-11 place-items-center rounded-full bg-surface-raised text-content-subtle">
          {icon}
        </span>
      )}
      <p className="m-0 text-sm font-medium text-content">{title}</p>
      {children && (
        <p className="m-0 max-w-sm text-xs leading-relaxed text-content-subtle">{children}</p>
      )}
      {action}
    </div>
  );
}
