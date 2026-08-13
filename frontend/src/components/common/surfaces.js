/**
 * The app's surface + control classes, in one place.
 *
 * These started life inside `components/settings/primitives.jsx` during the
 * Settings redesign, which is where the reference came from (Chrome's own
 * Settings, dark): a card is a raised surface with a soft shadow, not a boxed-in
 * outline, and it lifts slightly on hover. Two reasons that reads better than a
 * hairline here:
 *
 *   · the tokens are ALREADY an elevation system — `surface` is white at 4% and
 *     `surface-raised` at 9% — so `border border-border` on top of one meant two
 *     separation mechanisms doing the same job, and a page of them turned into a
 *     grid of boxes;
 *   · on a near-black ground a hairline reads as a hard edge, while a shadow
 *     reads as depth, which is what the hierarchy actually is.
 *
 * The shadow is deliberately heavier than a light-theme card's: at these
 * background values a subtle one is invisible. Hover is a lift, not a colour
 * change, so it never competes with the accent.
 *
 * Padding and radius-breaking layout stay with the caller — these strings are
 * the SURFACE, not the box.
 */

/** The elevation alone, for a card that brings its own background — a run card
 *  tinted by where it ran, a status panel tinted by its tone. */
export const CARD_SHADOW =
  'shadow-[0_4px_12px_rgba(0,0,0,0.05),0_12px_36px_rgba(0,0,0,0.06)]';

/** A card that does not react: a panel, a form, a section. Frosted glass —
 *  translucent white over a backdrop blur, with a thin white edge that reads
 *  as the light catching the pane. The page's colour blobs bleed through as a
 *  soft wash, so the card is clearly a pane of glass above the ground. */
export const CARD_SURFACE = `rounded-2xl bg-white/60 backdrop-blur-xl ring-1 ring-white/70 ${CARD_SHADOW}`;

/** The ONE hover gesture every interactive surface shares — cards, image tiles,
 *  covers: a 6px lift and a big diffuse shadow, so the thing under the pointer
 *  clearly floats above the page. This string is the source of truth. */
export const FLOAT_HOVER =
  'transition-[box-shadow,transform] duration-200 hover:-translate-y-2 ' +
  'hover:shadow-[0_24px_80px_rgba(0,0,0,0.12),0_8px_24px_rgba(0,0,0,0.08)]';

export const CARD_SURFACE_INTERACTIVE = `${CARD_SURFACE} ${FLOAT_HOVER}`;

/* `max-w-xl` is the tidy-up from the same wave: the settings shell went
   full-width on 2026-08-09 and every `w-full` control went with it, so a text
   input stretched ~1400px across while the sliders beside it stayed 112px. A
   control has a comfortable measure regardless of how wide the window is; the
   CARD still fills the pane, only its contents are bounded. */
export const INPUT_CLASS =
  'mt-1 w-full max-w-xl rounded-xl bg-surface-raised px-3 py-2 text-sm text-content ' +
  'placeholder:text-content-subtle focus:outline-none focus:ring-2 focus:ring-primary/40';

/** A secondary action: the outline button, minus the outline. */
export const QUIET_BUTTON =
  'inline-flex items-center gap-1.5 rounded-full bg-surface-raised px-3 py-1.5 text-xs font-semibold ' +
  'text-content transition-colors hover:bg-surface disabled:opacity-50';

/** The primary action on a page (create, promote, generate). */
export const PRIMARY_BUTTON =
  'inline-flex items-center gap-1.5 rounded-full bg-gradient-primary px-4 py-2 text-sm font-semibold ' +
  'text-white transition-[box-shadow,transform,opacity] duration-200 hover:opacity-95 ' +
  'hover:-translate-y-1 hover:shadow-[0_12px_28px_rgba(79,70,229,0.45),0_4px_10px_rgba(0,0,0,0.15)] ' +
  'disabled:opacity-50';
