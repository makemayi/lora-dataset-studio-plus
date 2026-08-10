/**
 * The title block every top-level page wears.
 *
 * Five pages had grown their own version of the same three parts — a mono
 * "rack tag" eyebrow, a title, and a row of actions — at four different sizes
 * (text-lg, text-xl, text-xl font-bold, text-2xl font-bold) with three
 * different gaps. On a dark screen a 20px semibold title one line under an 11px
 * eyebrow is not a hierarchy; it is two lines of grey.
 *
 * So: one component, and the contrast is deliberate — the eyebrow is small,
 * wide-tracked and subtle, the title is 24px with tight tracking, and the
 * description (when there is one) sits at a readable measure below both. That
 * step is what makes a page announce itself in one glance instead of after a
 * read.
 *
 * Props:
 *  · eyebrow     — the rack tag ("library", "bank", "training", "board").
 *  · title       — the page name. Keyed on its own text: Chrome auto-translate
 *                  replaces the text node, and React must remount rather than
 *                  edit a node it no longer owns (CLAUDE.md ▸ UI changes).
 *  · badge       — a chip that belongs to the TITLE (a Beta marker, a count).
 *  · actions     — the right-hand cluster; wraps under the title on a phone.
 *  · description — optional; pass a <HelpText> to fold a long one.
 */
export default function PageHeader({ eyebrow, title, badge, actions, description }) {
  return (
    <header className="flex flex-col gap-1">
      <p key={eyebrow}
        className="m-0 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">
        {eyebrow}
      </p>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <h1 key={title}
          className="m-0 flex items-center gap-2 text-2xl font-semibold tracking-tight text-content">
          {title}{badge}
        </h1>
        {actions && <div className="ml-auto flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {description}
    </header>
  );
}
