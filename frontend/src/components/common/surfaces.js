/**
 * The app's surface + control classes, in one place.
 *
 * These started life inside `components/settings/primitives.jsx` during the
 * Settings redesign, which is where the reference came from (Chrome's own
 * Settings): a card is a raised surface with a soft shadow, not a boxed-in
 * outline, and it lifts slightly on hover. Two reasons that reads better than a
 * hairline here:
 *
 *   · the tokens are ALREADY an elevation system, so `border border-border` on
 *     top of one meant two separation mechanisms doing the same job, and a page
 *     of them turned into a grid of boxes;
 *   · a hairline reads as a hard edge, while a shadow reads as depth, which is
 *     what the hierarchy actually is.
 *
 * The APP WAS DARK when these were written and is LIGHT since the Vision-Pro
 * glass pass (2026-08-13/14) — the recipes below were re-tuned for it and the
 * numbers moved in the opposite direction to the intuition: a light ground
 * needs a HEAVIER shadow, because the 0.05-alpha ones that read fine on
 * near-black were invisible on `#F5F7FA`. Hover is a lift, not a colour change,
 * so it never competes with the accent.
 *
 * Padding and radius-breaking layout stay with the caller — these strings are
 * the SURFACE, not the box.
 */

/** The elevation alone, for a card that brings its own background — a run card
 *  tinted by where it ran, a status panel tinted by its tone. */
export const CARD_SHADOW =
  'shadow-[0_0_0_1px_rgba(255,255,255,0.4),0_2px_4px_rgba(0,0,0,0.10),0_10px_20px_rgba(0,0,0,0.10),0_12px_40px_rgba(79,70,229,0.10)]';

/** A card that does not react: a panel, a form, a section. THE glass recipe —
 *  the same one the hero "next step" card wears: a soft indigo-to-white
 *  gradient over a backdrop blur, with an indigo-tinted glow, so every card
 *  reads as a pane of tinted glass floating above the colourful ground. */
export const CARD_SURFACE =
  'rounded-2xl bg-gradient-to-br from-[#EBF0FF] to-white backdrop-blur-xl ' +
  `${CARD_SHADOW}`;

/** The ONE hover gesture every interactive surface shares — cards, image tiles,
 *  covers: a 4px lift and a deeper indigo glow, matching the hero card. */
export const FLOAT_HOVER =
  'transition-[box-shadow,transform] duration-200 hover:-translate-y-1 ' +
  'hover:shadow-[0_20px_60px_rgba(79,70,229,0.18),0_8px_20px_rgba(0,0,0,0.10)]';

export const CARD_SURFACE_INTERACTIVE = `${CARD_SURFACE} ${FLOAT_HOVER}`;

/* ── 2026-08-13 wave — the "floating" recipes first established on the dataset
   workspace (engine cards, preset/shot pills, the reference card, the sidebar)
   and now shared so every page reads the same. ──────────────────────────── */

/** At-rest two-layer shadow: a close definition layer + a larger diffuse layer.
 *  Heavy enough (0.12–0.14 alpha) that elevation reads WITHOUT a hover — the
 *  earlier 0.05-alpha shadows were invisible on the light ground. */
export const FLOAT_SHADOW =
  'shadow-[0_1px_2px_rgba(0,0,0,0.14),0_4px_10px_rgba(0,0,0,0.12)]';

/** The hover twin: lift a hair and deepen the shadow so the press-in feels real. */
export const FLOAT_HOVER_SHADOW =
  'hover:-translate-y-0.5 hover:shadow-[0_2px_4px_rgba(0,0,0,0.18),0_10px_20px_rgba(0,0,0,0.16)]';

/** A frosted-glass PILL for tags / chips: white glass + blur + float shadow. */
export const GLASS_PILL =
  'rounded-full bg-white/60 backdrop-blur-md ' +
  `${FLOAT_SHADOW} ${FLOAT_HOVER_SHADOW} hover:bg-white/80`;

/** The CURRENT thing as a filled pill — macOS selection, never an outline.
 *  Neutral (not accent-tinted): the accent belongs to the primary action. */
export const SELECTED_PILL =
  'rounded-full bg-surface-raised text-content ' +
  'shadow-[0_1px_2px_rgba(0,0,0,0.14),0_0_0_1px_rgba(0,0,0,0.08),0_4px_10px_rgba(0,0,0,0.10)]';

/* ── 2026-08-15 wave — the dataset tile restyle: the photo IS the card. ──
   The tile used to be a white glass card with a photo boxed inside it
   (`rounded-[28px]` + `m-2 rounded-[22px]`); now the photo fills the tile and
   the card is the photo. The kept/rejected/pending state left the border for a
   corner dot, and selection left the `ring` for an outward glow — both were
   competing for the same edge.

   One glass language, reviewed 2026-08-15: everything laid ON the photo is
   DARK glass (GLASS_CAPSULE_DARK). The first pass shipped a light capsule for
   the caption and toolbar, and the review killed it — a tile cannot carry
   half light glass, half dark. */

/** The photo block itself: the full-bleed tile. Rounded corners + the clip
 *  that makes the photo respect them. The tile's own shadow is separate
 *  (TILE_SHADOW / TILE_SELECTED_GLOW) because selection SWAPS it. */
export const TILE_SURFACE = 'relative overflow-hidden rounded-2xl';

/** A dataset tile at rest: a soft definition + diffuse shadow, no outline.
 *  Lighter than the Settings cards — a grid of photos reads as a wall, not
 *  a pile, and the photo supplies the contrast. */
export const TILE_SHADOW =
  'shadow-[0_1px_2px_rgba(0,0,0,0.12),0_4px_10px_rgba(0,0,0,0.10)]';

/** Selected = a spread of brand indigo around the tile, REPLACING the resting
 *  shadow (the two are the same visual channel — the outer edge). A hairline
 *  ring around a full-bleed photo reads as a frame; a glow reads as "this one".
 *  The colour lives ONCE in tailwind.config.js (BRAND → boxShadow); this class
 *  only names it. The hover twin keeps the glow through the lift — selection
 *  is the state, and FLOAT_HOVER's own hover shadow must not replace it. */
export const TILE_SELECTED_GLOW =
  'shadow-tile-selected transition-[box-shadow,transform] duration-200 ' +
  'hover:-translate-y-1 hover:shadow-tile-selected-hover';

/** The dark glass capsule every control laid ON a photo wears — toolbar,
 *  caption, keep/reject, the source credit, the rescue banner. Dark ground +
 *  light text reads over arbitrary pixels (CLAUDE.md ▸ UI changes: a control
 *  painted on a photograph sits on a dark scrim). One recipe, four call sites
 *  used to hand-write it. */
export const GLASS_CAPSULE_DARK =
  'rounded-full bg-black/40 backdrop-blur-md border border-white/15 ' +
  'shadow-[0_1px_2px_rgba(0,0,0,0.14),0_4px_10px_rgba(0,0,0,0.12)]';

/** Hover feedback for a control painted ON a photograph: a light veil so the
 *  hit target reads as interactive without adding a frame. */
export const ON_PHOTO_HOVER = 'hover:bg-white/20';

/* `max-w-xl` is the tidy-up from the same wave: the settings shell went
   full-width on 2026-08-09 and every `w-full` control went with it, so a text
   input stretched ~1400px across while the sliders beside it stayed 112px. A
   control has a comfortable measure regardless of how wide the window is; the
   CARD still fills the pane, only its contents are bounded. */
export const INPUT_CLASS =
  'mt-1 w-full max-w-xl rounded-xl bg-surface-raised px-3 py-2 text-sm text-content ' +
  'placeholder:text-content-subtle focus:outline-none focus:ring-2 focus:ring-primary/40';

/** A secondary action: the outline button, minus the outline — now a filled
 *  pill that floats via the shared shadow. */
export const QUIET_BUTTON =
  'inline-flex items-center gap-1.5 rounded-full bg-surface-raised px-3 py-1.5 text-xs font-semibold ' +
  'text-content ' + `${FLOAT_SHADOW} ` +
  'transition-[box-shadow,transform,background-color] duration-200 hover:bg-surface ' + `${FLOAT_HOVER_SHADOW} ` +
  'disabled:opacity-50';

/** The primary action on a page (create, promote, generate). Indigo gradient
 *  pill with its own indigo-tinted float shadow, deepened on hover. */
export const PRIMARY_BUTTON =
  'inline-flex items-center gap-1.5 rounded-full bg-gradient-primary px-4 py-2 text-sm font-semibold ' +
  'text-white shadow-[0_2px_4px_rgba(79,70,229,0.30),0_4px_10px_rgba(79,70,229,0.18)] ' +
  'transition-all duration-200 hover:scale-[1.02] ' +
  'hover:shadow-[0_4px_12px_rgba(79,70,229,0.40),0_8px_20px_rgba(79,70,229,0.25)] ' +
  'disabled:opacity-50';
