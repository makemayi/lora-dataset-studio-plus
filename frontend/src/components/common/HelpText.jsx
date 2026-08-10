/* ── Help copy: shown when it is a sentence, folded when it is a paragraph ────
   Written for Settings during the 2026-08-09 redesign, moved here on 08-10 when
   the same measurement came back for every other page: the workspace, the bank
   and the studio carry ~4,000 more characters of always-open explanation, and
   it is the most valuable thing about this app AND the reason a page feels
   heavy. Twenty-one paragraphs of 200–470 characters, all open, all the time,
   competing with the controls they explain and with each other.

   So: fold the paragraphs, keep the one-liners. The split in the real copy is
   bimodal — one-line notes sit at 60–140 characters, explanations start again
   at 200 — so the threshold is not a fussy judgement call; nothing sits near it.

   A native <details> on purpose, not a React toggle. help/revealTarget.js
   already walks up from a deep-linked field and opens every collapsed <details>
   ancestor, so a Guide link into a folded explanation keeps working with no new
   machinery. It also sidesteps the Chrome-auto-translate crash: the summary and
   the body are both always mounted, so React never removes a text node.

   ⚠️ What must NOT be folded, learned in Settings and re-applied here:
   a live readout (a current budget, what a slider is set to), a warning about
   something that is wrong right now, and anything a refusal explains. Those are
   not explanations — they are the state of the app, and they stay open. */
const HELP_FOLD_OVER = 140;

/* Most help is not a bare string — it carries <span className="font-medium">
   emphasis, a <code>, an <a> to the guide. Measuring `children.length` would see
   an array of three and call a 500-character paragraph short, which is exactly
   how the first pass folded thirteen card blurbs and left thirty field blurbs
   open. Walk the tree instead. Interpolated values count as ~1 char each; that
   is close enough for a threshold nothing sits near. */
export function plainLength(node) {
  if (node === null || node === undefined || node === false || node === true) return 0;
  if (typeof node === 'string') return node.length;
  if (typeof node === 'number') return String(node).length;
  if (Array.isArray(node)) return node.reduce((n, child) => n + plainLength(child), 0);
  if (node.props) return plainLength(node.props.children);
  return 0;
}

/* The caller's className positions the help relative to its input (`mt-0.5`
   under a field, `mt-3` under a card's grid). That margin belongs to whatever
   element is actually in the flow — the <p> when open, the <details> when
   folded — while the rest of the classes style the text. Leaving `mt-3` on the
   inner <p> of a <details> would both indent the body wrongly and fight the
   `mt-1.5` this component wants there (same specificity: Tailwind's own order
   decides, not the order written here). */
const MARGIN_CLASS_RE = /(?:^|\s)-?m[tby]-[^\s]+/g;

function splitMargins(className) {
  const margins = (className.match(MARGIN_CLASS_RE) || []).map((c) => c.trim());
  return { outer: margins.join(' '), text: className.replace(MARGIN_CLASS_RE, ' ').trim() };
}

/* A caret, NOT a round "?" — that glyph is taken. `help/HelpMode.jsx` renders a
   round indigo "?" badge beside titles in Help mode, and clicking it jumps to
   the Guide. A second round "?" that expands text in place, sometimes on the
   same row, is two meanings wearing one face. The caret is also just honest:
   this is a disclosure triangle, so it looks like one and rotates like one.

   `summary` defaults to the short form because most callers are FIELDS, where
   the disclosure sits directly under a labelled input and a long label repeated
   five times down one card is the same visual noise the fold was meant to
   remove. A card or a page intro passes the long form ("Why this matters",
   "What this page is for"): it introduces a whole section, and there is only
   ever one of it. */
export function HelpText({ children, className = '', summary = 'More' }) {
  if (!children) return null;
  const { outer, text } = splitMargins(className);
  if (plainLength(children) <= HELP_FOLD_OVER) {
    return <p className={`max-w-prose ${outer} ${text}`}>{children}</p>;
  }
  return (
    <details className={`group max-w-prose ${outer}`}>
      <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-xs text-content-subtle hover:text-content-muted [&::-webkit-details-marker]:hidden">
        <span aria-hidden="true"
          className="inline-block text-[0.625rem] leading-none transition-transform duration-150 group-open:rotate-90">
          ▶
        </span>
        {summary}
      </summary>
      <p className={`mt-1.5 ${text}`}>{children}</p>
    </details>
  );
}
