/**
 * Discreet segmented S/M/L control, not a slider (mouse-fragile, no useful
 * granularity for 3 steps). Shared by the workspace image grid (DatasetGrid)
 * and the Datasets library (DatasetListPanel) — callers pass their own
 * context-specific `titles` so the tooltips explain what each step is FOR.
 */
export default function TileSizeControl({ size, onChange, titles, className = '' }) {
  return (
    /* One rounded track holding three segments — the selected one is filled,
       the others are plain. Three separately outlined squares read as three
       controls; this reads as one control with a state. */
    <div role="group" aria-label="Thumbnail size"
      className={`flex shrink-0 items-center gap-0.5 rounded-full bg-surface p-0.5 ${className}`}>
      {['S', 'M', 'L'].map((s) => (
        <button key={s} type="button" onClick={() => onChange(s)}
          aria-pressed={size === s} title={titles[s]}
          aria-label={`${titles[s]}${size === s ? ' (active)' : ''}`}
          className={`h-6 w-6 rounded-full text-[0.6875rem] font-semibold transition-colors ${
            size === s
              ? 'bg-primary text-white'
              : 'text-content-muted hover:bg-surface-raised hover:text-content'}`}>
          {s}
        </button>
      ))}
    </div>
  );
}
