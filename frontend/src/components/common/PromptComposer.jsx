/**
 * ONE shape for "type a prompt, then act on it".
 *
 * Before this, every prompt box in the app was a bare `<textarea>` with its own
 * radius, its own ring colour and its own square button parked beside it — the
 * Custom shot in the variation catalog, the Test-studio prompt, the comparison
 * run's prompt. Three surfaces, three looks, all of them reading as a form
 * field rather than as the place where the work starts.
 *
 * The composer is the glass grammar the rest of the app already speaks: a
 * frosted pane (white glass + blur + the shared float shadow) that OWNS the
 * focus state, a borderless textarea sitting on it, a tools row along the
 * bottom, and — where the surface has a real action — a round gradient send
 * button on the right. The shell lifts on focus the same 0.5px every other
 * floating surface lifts on hover, so focus reads as the pane coming forward
 * rather than as a blue outline being switched on.
 *
 * The textarea keeps `bg-transparent` and drops its own ring on purpose: two
 * focus indicators (field ring + shell ring) is the same "two mechanisms doing
 * one job" mistake as a border on an elevated surface.
 *
 * `onSend` is optional. A surface whose action lives elsewhere (the Test studio
 * launches from its own panel, below the cost line) passes tools only and gets
 * no button — an inert send affordance would be a lie about where the run
 * starts.
 */
import { FLOAT_SHADOW } from './surfaces.js';
import { SendIcon } from './icons.jsx';

/** The frosted pane. Focus-within lifts it and deepens the shadow rather than
 *  painting a ring, matching FLOAT_HOVER_SHADOW's gesture. */
export const COMPOSER_SHELL =
  'rounded-[22px] bg-white/60 backdrop-blur-xl px-3 py-2.5 ' +
  `${FLOAT_SHADOW} ` +
  'transition-[box-shadow,transform] duration-200 ' +
  'focus-within:-translate-y-0.5 ' +
  'focus-within:shadow-[0_2px_4px_rgba(79,70,229,0.20),0_10px_24px_rgba(79,70,229,0.16)]';

/** The textarea on that pane: no background, no ring, no resize handle fighting
 *  the rounded corner — the shell is the visible control. */
export const COMPOSER_TEXTAREA =
  'w-full resize-y bg-transparent text-content placeholder:text-content-subtle ' +
  'focus:outline-none focus-visible:outline-none';

/** The round gradient send button — 40px, the primary gradient, its own indigo
 *  glow, and the press-in scale the design brief asked for (0.05s in, springy
 *  out). Disabled keeps the shape and drops the weight. */
export const SEND_BUTTON =
  'grid h-10 w-10 shrink-0 place-items-center rounded-full bg-gradient-primary text-white ' +
  'shadow-[0_2px_4px_rgba(79,70,229,0.30),0_6px_16px_rgba(79,70,229,0.25)] ' +
  'transition-all duration-200 hover:scale-105 ' +
  'hover:shadow-[0_4px_12px_rgba(79,70,229,0.40),0_10px_24px_rgba(79,70,229,0.30)] ' +
  'active:scale-95 active:duration-75 ' +
  'disabled:opacity-40 disabled:shadow-none disabled:hover:scale-100';

export default function PromptComposer({
  id = undefined,
  value,
  onChange,                 // receives the raw string
  placeholder = '',
  rows = 3,
  ariaLabel = undefined,
  tools = null,             // left side of the bottom row (Enhance, Describe, a framing select…)
  onSend = null,            // omit entirely when the action lives outside the composer
  sendLabel = 'Send',       // the button's accessible name AND its tooltip
  sendDisabled = false,
  textareaClassName = 'text-sm',
  className = '',
}) {
  // ⌘/Ctrl+Enter sends. Plain Enter stays a newline: these are multi-line
  // prompts, and a bare Enter that fires the action would eat paragraph breaks
  // people type by reflex.
  const onKeyDown = (e) => {
    if (!onSend || sendDisabled) return;
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      onSend();
    }
  };
  const hasFooter = !!tools || !!onSend;
  return (
    <div className={`${COMPOSER_SHELL} ${className}`}>
      <textarea
        id={id}
        value={value}
        rows={rows}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        aria-label={ariaLabel}
        className={`${COMPOSER_TEXTAREA} ${textareaClassName}`}
      />
      {hasFooter && (
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
          <div className="flex min-w-0 flex-wrap items-center gap-1.5">{tools}</div>
          {onSend && (
            <button type="button" onClick={onSend} disabled={sendDisabled}
              className={`${SEND_BUTTON} ml-auto`} title={sendLabel} aria-label={sendLabel}>
              <SendIcon className="h-[18px] w-[18px] shrink-0" />
            </button>
          )}
        </div>
      )}
    </div>
  );
}
