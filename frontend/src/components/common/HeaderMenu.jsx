import { useEffect, useRef, useState } from 'react';
import { ICON_BUTTON_BASE, ICON_BUTTON_QUIET } from './icons';

/** A small header dropdown (the ? Help menu and the ⚙ Settings menu share it).
 *  Follows the same interaction contract as the app's other popovers
 *  (CaptionOptionsPopover / PromptEditPopover): closes on Escape, on outside
 *  click, and — because every item navigates or toggles — on any item click.
 *
 *  Presentational only. `children` is a render-prop receiving `close`, so items
 *  (NavLinks, the Help-mode toggle) close the menu when they fire:
 *    <HeaderMenu …>{(close) => <NavLink onClick={close} … />}</HeaderMenu>
 *
 *  Props:
 *   - triggerLabel : node shown inside the trigger button (the ? / ⚙ glyph).
 *   - triggerTitle : tooltip + accessible name for the trigger.
 *   - active       : true when the current route lives in this menu — the
 *                    trigger then reflects the active-nav style, discreetly.
 *   - dot          : true to paint a small primary attention dot on the trigger
 *                    (Setup's "recommended steps unmet" indicator moved up here).
 *   - align        : menu horizontal alignment; 'right' (default) or 'left'. */
// The trigger is one of the header's round icon buttons — same 32px target as
// What's-new and the update check, so the utility cluster is one row of equal
// circles rather than a mix of text buttons and glyphs.
const TRIGGER_BASE = ICON_BUTTON_BASE;

export default function HeaderMenu({ triggerLabel, triggerTitle, active = false, dot = false, align = 'right', children }) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);
  const triggerRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        setOpen(false);
        triggerRef.current?.focus();   // return focus to the trigger on Escape
      }
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const close = () => setOpen(false);

  return (
    <div ref={wrapRef} className="relative">
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        title={triggerTitle}
        onClick={() => setOpen((v) => !v)}
        className={`${TRIGGER_BASE} ${
          open || active ? 'bg-surface-raised text-content' : ICON_BUTTON_QUIET
        }`}
      >
        {triggerLabel}
        {dot && (
          <span aria-hidden="true"
            className="absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-primary" />
        )}
        <span className="sr-only">{triggerTitle}</span>
      </button>
      {open && (
        <div
          role="menu"
          aria-label={triggerTitle}
          /* Elevation, not an outline: the shadow already separates the panel
             from the page, and the border only added a bright edge to it. */
          className={`absolute ${align === 'right' ? 'right-0' : 'left-0'} top-full mt-2 z-50 min-w-[11rem]
            flex flex-col gap-0.5 rounded-xl bg-surface-overlay/85 backdrop-blur-md p-1.5 shadow-2xl`}
        >
          {typeof children === 'function' ? children(close) : children}
        </div>
      )}
    </div>
  );
}
