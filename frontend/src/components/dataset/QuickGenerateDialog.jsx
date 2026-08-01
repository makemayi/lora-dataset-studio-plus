/** 🎲 Quick generate — pick a total count and a rough framing/angle mix, then
 * go straight from sliders to queued images: compose the variation list
 * server-side, then feed it into the SAME engineBatches()/generate() flow a
 * manual multi-select already uses. Deliberately has NO preview step — that
 * was considered and rejected during design; reviewing composed prompts
 * before they queue would just be a second dialog for a feature whose whole
 * point is "I don't want to hand-pick shots".
 *
 * Purely presentational and props-driven, same shape as ContinueDialog:
 * onResolve(payload | null) — `true` once the batch is queued, or `null` on
 * cancel/Escape. Not wired into a parent yet (that's the next task) — every
 * prop below is expected to come from the dataset panel that mounts it.
 *
 * The rear-facing option exists in the component pools for other subject
 * types (object/other/creature) but is excluded here on purpose: the spec
 * for this dialog's sliders is face/bust/body only. */
import { useEffect, useMemo, useState } from 'react';
import { engineBatches } from './engineSelection.js';

const FRAMINGS = ['face', 'bust', 'body'];
const DEFAULT_FRAMING_RATIOS = { face: 50, bust: 30, body: 20 };

/** Redistribute the remainder across the OTHER keys proportionally to their
 * current weight, so nudging one slider doesn't silently zero the others out
 * — same spirit as the app's other ratio-editing UI.
 *
 * Uses the "largest remainder" (Hare quota) allocation method instead of
 * per-key rounding + a final drift patch: each other key first gets
 * Math.floor(its exact fractional share), which by construction can never be
 * negative (remaining >= 0, weights >= 0), and the leftover integer units
 * (always >= 0 and always <= others.length, since each key's fractional part
 * is < 1) are handed out one at a time to the keys with the largest
 * fractional remainder. Because the only adjustment ever made to a floor is
 * "+1", no key can end up negative, and the shares plus the floors sum to
 * exactly `remaining` by construction — so the full set of returned values
 * always sums to exactly 100 with none negative. */
function normalizeTo100(values, changedKey, changedValue) {
  const keys = Object.keys(values);
  const others = keys.filter((k) => k !== changedKey);
  const clampedChanged = Math.max(0, Math.min(100, changedValue));
  const next = { [changedKey]: clampedChanged };
  if (others.length === 0) return next;

  const remaining = 100 - clampedChanged;
  const weights = others.map((k) => Math.max(0, values[k] || 0));
  const othersTotal = weights.reduce((s, w) => s + w, 0);
  // Exact (fractional) share per key. If every "other" weight is 0 (nothing
  // to be proportional to), split the remainder evenly instead of the old
  // `|| 1` fallback, which used to dump everything on the last key.
  const exact = othersTotal > 0
    ? weights.map((w) => (w / othersTotal) * remaining)
    : others.map(() => remaining / others.length);
  const floors = exact.map((v) => Math.floor(v));
  const floorSum = floors.reduce((a, b) => a + b, 0);
  // Clamp for safety against floating-point noise only — mathematically this
  // is always in [0, others.length].
  const leftover = Math.max(0, Math.min(others.length, remaining - floorSum));
  const order = exact
    .map((v, i) => [v - floors[i], i])
    .sort((a, b) => b[0] - a[0]);
  const shares = floors.slice();
  for (let i = 0; i < leftover; i += 1) shares[order[i][1]] += 1;
  others.forEach((k, i) => { next[k] = shares[i]; });
  return next;
}

export default function QuickGenerateDialog({
  hasRef, engines, engineMode, klein, loraStrength, generationLoraPreset,
  onGenerate, quickGenerateCompose, quickGenerateComponents, busy, onResolve,
}) {
  const [total, setTotal] = useState(30);
  const [framingRatios, setFramingRatios] = useState(DEFAULT_FRAMING_RATIOS);
  const [angleRatios, setAngleRatios] = useState({});
  const [pools, setPools] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const d = await quickGenerateComponents('human');
      if (!cancelled && d?.pools) {
        setPools(d.pools);
        const evenAngles = {};
        FRAMINGS.forEach((f) => {
          const ids = (d.pools[f]?.angle || []).map((a) => a.id);
          const share = ids.length ? Math.floor(100 / ids.length) : 0;
          const ratios = {};
          ids.forEach((id, i) => { ratios[id] = i === ids.length - 1 ? 100 - share * (ids.length - 1) : share; });
          evenAngles[f] = ratios;
        });
        setAngleRatios(evenAngles);
      }
    })();
    return () => { cancelled = true; };
  }, [quickGenerateComponents]);

  /* ONE way out, closed while a request is in flight — same reasoning as
     ContinueDialog's dismiss(): cancelling mid-post would leave the compose
     or the generate call running with nothing on screen to report it. */
  const dismiss = () => { if (!submitting && !busy) onResolve(null); };

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') dismiss(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [submitting, busy]);  // eslint-disable-line react-hooks/exhaustive-deps

  const setFraming = (key, value) => setFramingRatios((prev) => normalizeTo100(prev, key, value));
  const setAngle = (framing, key, value) => setAngleRatios((prev) => ({
    ...prev, [framing]: normalizeTo100(prev[framing] || {}, key, value),
  }));

  const submit = async () => {
    // Re-entry guard: two click events dispatched in the same tick (before
    // React commits the `disabled` re-render) would otherwise both pass the
    // `disabled` check and both reach here, firing quickGenerateCompose +
    // onGenerate twice.
    if (submitting) return;
    setSubmitting(true);
    try {
      const activeAngleRatios = {};
      FRAMINGS.forEach((f) => { if (framingRatios[f] > 0) activeAngleRatios[f] = angleRatios[f]; });
      const variations = await quickGenerateCompose({
        total, framing_ratios: framingRatios, angle_ratios: activeAngleRatios, subject_type: 'human',
      });
      if (!variations) return;  // compose already toasted the error
      const batches = engineBatches(variations, engines, engineMode);
      await onGenerate(batches, 1, klein, loraStrength, generationLoraPreset);
      onResolve(true);
    } finally {
      setSubmitting(false);
    }
  };

  const activeFramings = useMemo(() => FRAMINGS.filter((f) => framingRatios[f] > 0), [framingRatios]);
  const disabled = submitting || busy || !hasRef;

  return (
    <div role="dialog" aria-modal="true" aria-label="Quick generate"
      className="fixed inset-0 z-[9990] bg-black/80 flex items-center justify-center p-3"
      onClick={(e) => { if (e.target === e.currentTarget) dismiss(); }}>
      <div className="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-xl border border-indigo-400/40 bg-app p-4 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="text-indigo-300 font-semibold"><span aria-hidden>🎲</span> Quick generate</span>
          <button type="button" onClick={dismiss} disabled={submitting || busy}
            className="ml-auto text-content-subtle hover:text-content disabled:opacity-40" aria-label="Cancel">✕</button>
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">Total images</span>
            <input type="number" min="1" max="200" value={total}
              onChange={(e) => setTotal(Math.max(1, Math.min(200, Number(e.target.value) || 1)))}
              aria-label="Total images to generate"
              className="w-28 px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] tabular-nums" />
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            Capped at 200 per run, same as the backend enforces.
          </span>
        </div>

        {/* Framing mix — face/bust/body only. The rear-facing option is
            intentionally not offered here (see file header). */}
        <div className="flex flex-col gap-2">
          <span className="text-content text-[0.75rem]">Framing mix</span>
          {FRAMINGS.map((f) => (
            <label key={f} className="flex flex-col gap-1">
              <span className="flex justify-between items-center">
                <span className="text-content-muted text-xs font-semibold uppercase tracking-wide">{f}</span>
                <span className="text-content-muted text-[0.8125rem] font-semibold">{framingRatios[f]}%</span>
              </span>
              <input type="range" min="0" max="100" value={framingRatios[f]}
                aria-label={`${f} framing share`}
                onChange={(e) => setFraming(f, Number(e.target.value))}
                className="w-full accent-primary" />
            </label>
          ))}
        </div>

        {activeFramings.map((f) => (
          <div key={f} className="flex flex-col gap-2 border-t border-border pt-2">
            <span className="text-content-subtle text-[0.6875rem]">{f} angle mix</span>
            {(pools?.[f]?.angle || []).map((a) => (
              <label key={a.id} className="flex flex-col gap-0.5">
                <span className="flex justify-between items-center">
                  <span className="text-content-muted text-[0.6875rem]">{a.label || a.id}</span>
                  <span className="text-content-muted text-[0.6875rem] tabular-nums">{angleRatios[f]?.[a.id] ?? 0}%</span>
                </span>
                <input type="range" min="0" max="100" value={angleRatios[f]?.[a.id] ?? 0}
                  aria-label={`${f} ${a.label || a.id} angle share`}
                  onChange={(e) => setAngle(f, a.id, Number(e.target.value))}
                  className="w-full accent-primary" />
              </label>
            ))}
          </div>
        ))}

        <div className="flex items-center gap-2 pt-1">
          <button type="button" onClick={dismiss} disabled={submitting || busy}
            className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40">Cancel</button>
          <button type="button" onClick={submit} disabled={disabled}
            title={!hasRef ? 'Add a reference image first' : undefined}
            className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            {submitting ? 'Generating…' : `🎲 Generate (${total})`}
          </button>
        </div>
      </div>
    </div>
  );
}
