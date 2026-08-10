/**
 * The app's icon set — one system, drawn here, shipped in the bundle.
 *
 * Why hand-drawn SVG and not an icon package or emoji:
 *  - The app must work with no internet, so no CDN sprite and no webfont.
 *  - Emoji were what the UI had, and they were the problem: three of the five
 *    workspaces carried one and the other two carried nothing, each drawn by a
 *    different font at a different weight and baseline. Half-decorated is
 *    noise, not navigation. Same story on the bank cards, where a package and
 *    a cross stood in for "move" and "delete".
 *
 * Every icon is the same 24-box, the same 1.75 stroke, and paints in
 * `currentColor`, so a control's colour transition carries the glyph with it
 * and nothing has to be re-tinted per state.
 *
 * Emoji are NOT banned everywhere — they still carry meaning in prose, in
 * headings and in the Guide. What they may no longer do is stand in for a
 * control's icon.
 */

/** Shared frame. `className` reaches the <svg> for sizing (default 16px).
 *  `data-icon` is the icon's name: a rendered <svg> is otherwise an anonymous
 *  blob of path data, and the nav's render test needs to say WHICH glyph sits
 *  in which link. */
function Glyph({ name, children, className = 'h-4 w-4 shrink-0' }) {
  return (
    <svg aria-hidden="true" data-icon={name} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.75" strokeLinecap="round"
      strokeLinejoin="round" className={className}>
      {children}
    </svg>
  );
}

/* ── Workspaces (the nav bar) ────────────────────────────────────────────── */

export function DatasetsIcon(props) {
  return (
    <Glyph name="datasets" {...props}>
      <rect x="3" y="3" width="13" height="13" rx="3" />
      <path d="M6 20.5h11a3.5 3.5 0 0 0 3.5-3.5V6" />
      <path d="M3 12.5l3-3 4 4" />
    </Glyph>
  );
}

export function BankIcon(props) {
  return (
    <Glyph name="bank" {...props}>
      <rect x="3" y="4" width="18" height="4.5" rx="1.5" />
      <path d="M5 8.5V18a2.5 2.5 0 0 0 2.5 2.5h9A2.5 2.5 0 0 0 19 18V8.5" />
      <path d="M10 12.5h4" />
    </Glyph>
  );
}

export function RunsIcon(props) {
  return (
    <Glyph name="runs" {...props}>
      <path d="M3 12.5h3.2l2.6 6 4-13 2.8 7h5.4" />
    </Glyph>
  );
}

export function CanvasIcon(props) {
  return (
    <Glyph name="canvas" {...props}>
      <circle cx="6" cy="6" r="2.5" />
      <circle cx="18" cy="6" r="2.5" />
      <circle cx="12" cy="18" r="2.5" />
      <path d="M7.6 7.9l3 8M16.4 7.9l-3 8" />
    </Glyph>
  );
}

export function StudioIcon(props) {
  return (
    <Glyph name="studio" {...props}>
      <path d="M10.5 3.5l1.7 4.3 4.3 1.7-4.3 1.7-1.7 4.3-1.7-4.3L4.5 9.5l4.3-1.7z" />
      <path d="M17.5 14.5l.9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z" />
    </Glyph>
  );
}

/* ── Utilities (header chrome) ───────────────────────────────────────────── */

export function HelpIcon(props) {
  return (
    <Glyph name="help" {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M9.6 9.5a2.45 2.45 0 1 1 3.35 2.3c-.72.3-1.1.9-1.1 1.65v.35" />
      <path d="M11.85 17.2h.05" />
    </Glyph>
  );
}

export function SettingsIcon(props) {
  return (
    <Glyph name="settings" {...props}>
      <circle cx="12" cy="12" r="3.2" />
      <path d="M12 2.8l1.2 2.4 2.6-.6 1 2.5 2.6.5-.5 2.6 2 1.8-2 1.8.5 2.6-2.6.5-1 2.5-2.6-.6L12 21.2l-1.2-2.4-2.6.6-1-2.5-2.6-.5.5-2.6-2-1.8 2-1.8-.5-2.6 2.6-.5 1-2.5 2.6.6z" />
    </Glyph>
  );
}

export function GiftIcon(props) {
  return (
    <Glyph name="gift" {...props}>
      <rect x="3" y="8.5" width="18" height="4" rx="1.25" />
      <path d="M5 12.5V19a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-6.5M12 21V8.5" />
      <path d="M12 8.5H7.9a2.45 2.45 0 1 1 0-4.9c2.9 0 4.1 4.9 4.1 4.9zM12 8.5h4.1a2.45 2.45 0 1 0 0-4.9C13.2 3.6 12 8.5 12 8.5z" />
    </Glyph>
  );
}

export function UpdateIcon(props) {
  return (
    <Glyph name="update" {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 16.2V8M8.6 11.4L12 8l3.4 3.4" />
    </Glyph>
  );
}

/** The busy twin of UpdateIcon 鈥?a ring with one bright quarter, spinning. */
export function SpinnerIcon({ className = 'h-4 w-4 shrink-0' }) {
  return (
    <svg aria-hidden="true" data-icon="spinner" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.75" strokeLinecap="round"
      className={`${className} animate-spin`}>
      <circle cx="12" cy="12" r="9" className="opacity-25" />
      <path d="M21 12a9 9 0 0 0-9-9" />
    </svg>
  );
}

export function MenuIcon(props) {
  return (
    <Glyph name="menu" {...props}>
      <path d="M4 7h16M4 12h16M4 17h16" />
    </Glyph>
  );
}

export function CloseIcon(props) {
  return (
    <Glyph name="close" {...props}>
      <path d="M6.5 6.5l11 11M17.5 6.5l-11 11" />
    </Glyph>
  );
}

/* ── Actions (cards, lists, forms) ───────────────────────────────────────── */

/** Repoint something at a new folder — the bank cards' old package emoji. */
export function MoveIcon(props) {
  return (
    <Glyph name="move" {...props}>
      <path d="M3 7.5h6.5l1.8 2H21v9.5a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 19z" />
      <path d="M3 7.5V5.5A1.5 1.5 0 0 1 4.5 4h4l1.8 2" />
      <path d="M10.5 14.5h5M13.5 12.5l2 2-2 2" />
    </Glyph>
  );
}

export function FolderIcon(props) {
  return (
    <Glyph name="folder" {...props}>
      <path d="M3 6.5A1.5 1.5 0 0 1 4.5 5h4.2l1.8 2.2h8A1.5 1.5 0 0 1 20 8.7V18a1.5 1.5 0 0 1-1.5 1.5h-14A1.5 1.5 0 0 1 3 18z" />
    </Glyph>
  );
}

export function PlusIcon(props) {
  return (
    <Glyph name="plus" {...props}>
      <path d="M12 5v14M5 12h14" />
    </Glyph>
  );
}

export function ArrowRightIcon(props) {
  return (
    <Glyph name="arrow-right" {...props}>
      <path d="M4.5 12h15M14 6.5l5.5 5.5L14 17.5" />
    </Glyph>
  );
}

export function RefreshIcon(props) {
  return (
    <Glyph name="refresh" {...props}>
      <path d="M20 11.5a8 8 0 1 0-.8 4.3" />
      <path d="M20 5.5v6h-6" />
    </Glyph>
  );
}

/** Open this full-screen — the pinned images' old 🔍. */
export function MagnifierIcon(props) {
  return (
    <Glyph name="magnifier" {...props}>
      <circle cx="10.75" cy="10.75" r="6.25" />
      <path d="M15.4 15.4L20 20" />
    </Glyph>
  );
}

export function DownloadIcon(props) {
  return (
    <Glyph name="download" {...props}>
      <path d="M12 4v10.5M8.2 11l3.8 3.8L15.8 11" />
      <path d="M4.5 17.5V19a1.5 1.5 0 0 0 1.5 1.5h12a1.5 1.5 0 0 0 1.5-1.5v-1.5" />
    </Glyph>
  );
}

/** Tidy up / rearrange automatically — the canvas' old ✦. */
export function WandIcon(props) {
  return (
    <Glyph name="wand" {...props}>
      <path d="M4 20l10.5-10.5" />
      <path d="M14 4.5l.9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z" />
      <path d="M19 13l.6 1.4 1.4.6-1.4.6-.6 1.4-.6-1.4-1.4-.6 1.4-.6z" />
    </Glyph>
  );
}

/** Generate images — the canvas' and studio's old 🎨. */
export function PaletteIcon(props) {
  return (
    <Glyph name="palette" {...props}>
      <path d="M12 3.2a8.8 8.8 0 0 0 0 17.6c1.4 0 2.1-.9 2.1-1.9 0-1.5-1.2-1.8-1.2-3 0-1 .8-1.8 1.8-1.8h1.6a4.5 4.5 0 0 0 4.5-4.5c0-3.6-3.7-6.4-8.8-6.4z" />
      <circle cx="8" cy="10" r="1.1" />
      <circle cx="12" cy="7.6" r="1.1" />
      <circle cx="16" cy="9.6" r="1.1" />
    </Glyph>
  );
}

export function ImageIcon(props) {
  return (
    <Glyph name="image" {...props}>
      <rect x="3" y="4.5" width="18" height="15" rx="2.5" />
      <circle cx="8.75" cy="9.75" r="1.6" />
      <path d="M3.5 17l4.8-4.6 3.7 3.5 3-2.7 5.5 5" />
    </Glyph>
  );
}

export function FilmIcon(props) {
  return (
    <Glyph name="film" {...props}>
      <rect x="3" y="4.5" width="18" height="15" rx="2.5" />
      <path d="M7.5 4.5v15M16.5 4.5v15M3 12h18M3 8.25h4.5M3 15.75h4.5M16.5 8.25H21M16.5 15.75H21" />
    </Glyph>
  );
}

/**
 * The round icon-button shell, shared by every utility control in the header
 * (What's new, update check, the ? and gear menus, the mobile hamburger) and by
 * the small per-card actions, so a row of them reads as equal targets instead
 * of differently-sized text buttons. 32px is the hit target; the pill radius is
 * the same one the nav links use.
 */
export const ICON_BUTTON_BASE =
  'relative grid h-8 w-8 place-items-center rounded-full transition-colors';
export const ICON_BUTTON_QUIET =
  'text-content-muted hover:text-content hover:bg-surface';
