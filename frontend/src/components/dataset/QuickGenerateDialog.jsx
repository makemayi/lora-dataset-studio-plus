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
 * cancel/Escape. Every prop below is expected to come from the dataset panel
 * that mounts it.
 *
 * The rear-facing option exists in the component pools for other subject
 * types (object/other/creature) but is excluded here on purpose: the spec
 * for this dialog's sliders is face/bust/body only.
 *
 * The dialog is a full-screen overlay, so the parent panel's engine picker is
 * out of reach while it's open: this dialog owns its OWN compact engine
 * checkbox list (`dialogEngines`), seeded from the `engines` prop on mount but
 * independent of the parent from then on. When the `nsfwMode` prop is on (the
 * same 🔞 toggle that gates the manual card-picking flow elsewhere), an NSFW
 * ratio slider appears; raising it above 0 filters that list down to the
 * LOCAL_ENGINES only, mirroring the server's fail-closed rule that NSFW shots
 * never reach an API engine. If that filtering would empty the selection, a
 * local fallback (klein, else krea) is auto-picked; if neither local engine is
 * available, the dialog says so and disables Generate rather than submitting
 * with zero engines. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { engineBatches, ENGINES, ENGINE_LABELS, LOCAL_ENGINES } from './engineSelection.js';

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
  nsfwMode, available,
}) {
  const [total, setTotal] = useState(30);
  const [framingRatios, setFramingRatios] = useState(DEFAULT_FRAMING_RATIOS);
  const [angleRatios, setAngleRatios] = useState({});
  const [pools, setPools] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [nsfwRatio, setNsfwRatio] = useState(0);
  // Dialog-owned engine selection: seeded once from whatever was checked in
  // the parent panel when this dialog opened, then fully independent of it
  // (see file header — the parent's picker is behind this full-screen overlay
  // and can't be live-mutated).
  const [dialogEngines, setDialogEngines] = useState(() => [...engines]);

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

  /* NSFW shots only ever run on a local engine (server fail-closed rule).
     Two DIFFERENT behaviours live in this one effect, and they must stay
     separate:
       1. Safety net (always applies while nsfwRatio > 0): drop any engine
          from the selection that isn't both in LOCAL_ENGINES AND currently
          `available` — covers a non-local engine carried over from the
          parent's picker, and an engine that WAS available/local but flips
          unavailable mid-session (e.g. the local server drops out).
       2. Fallback-if-empty (only on the 0 -> positive RATIO TRANSITION,
          i.e. the instant NSFW is switched on): if step 1 leaves nothing
          selected, auto-pick Klein (else Krea) so the user isn't dropped
          straight into a broken submit button. Once NSFW is already active,
          an empty selection is left alone — the user may have deliberately
          unchecked their only local engine, and re-adding it behind their
          back on the next slider tick would silently override that choice.
     `nsfwWasActiveRef` remembers whether the ratio was already positive the
     last time this effect ran, so a same-tick re-run (e.g. `available`
     flipping while the ratio doesn't change) is correctly treated as
     "already active", not as a fresh transition.
     A no-op while nsfwRatio is 0 — the exact prior behaviour. */
  const nsfwWasActiveRef = useRef(nsfwRatio > 0);
  useEffect(() => {
    const wasActive = nsfwWasActiveRef.current;
    const isActive = nsfwRatio > 0;
    nsfwWasActiveRef.current = isActive;
    if (!isActive) return;
    setDialogEngines((prev) => {
      const local = prev.filter((e) => LOCAL_ENGINES.includes(e) && available?.[e]);
      if (local.length > 0 || wasActive) {
        // Safety net only: either the selection already survives the
        // local+available filter, or NSFW was already on (respect an
        // intentionally-empty selection rather than fighting it).
        return local.length === prev.length ? prev : local;
      }
      // 0 -> positive transition left nothing selected: auto-pick a
      // fallback so Generate isn't silently disabled the moment NSFW turns on.
      if (available?.klein) return ['klein'];
      if (available?.krea) return ['krea'];
      // An install whose only local engine is H3 would otherwise be left with
      // an empty selection and a disabled Generate the moment NSFW turns on.
      if (available?.minimax_h3) return ['minimax_h3'];
      return local;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nsfwRatio, available?.klein, available?.krea, available?.minimax_h3]);

  const toggleDialogEngine = (name) => setDialogEngines((prev) => (
    prev.includes(name) ? prev.filter((e) => e !== name) : [...prev, name]
  ));

  /* The dialog's own picker: available engines, further narrowed to
     LOCAL_ENGINES while an NSFW ratio is set. Order follows the canonical
     ENGINES list, same as the parent panel's cards. */
  const offeredEngines = useMemo(() => ENGINES.filter(
    (name) => available?.[name] && (nsfwRatio <= 0 || LOCAL_ENGINES.includes(name)),
  ), [available, nsfwRatio]);

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
        nsfw_ratio: nsfwRatio,
      });
      if (!variations) return;  // compose already toasted the error
      const batches = engineBatches(variations, dialogEngines, engineMode);
      await onGenerate(batches, 1, klein, loraStrength, generationLoraPreset);
      onResolve(true);
    } finally {
      setSubmitting(false);
    }
  };

  const activeFramings = useMemo(() => FRAMINGS.filter((f) => framingRatios[f] > 0), [framingRatios]);
  const noLocalEngineForNsfw = nsfwMode && nsfwRatio > 0 && dialogEngines.length === 0;
  // Two distinct reasons the selection can be empty while NSFW is active —
  // say the accurate one. "Neither is currently available" is only true
  // when Klein AND Krea are both genuinely unavailable; if one of them IS
  // available, the user just hasn't (re)checked it (e.g. a deliberate
  // uncheck — see the effect above), so the copy must not claim otherwise.
  const noLocalEngineAvailable = !available?.klein && !available?.krea;
  const disabled = submitting || busy || !hasRef || dialogEngines.length === 0;

  return (
    <div role="dialog" aria-modal="true" aria-label="Quick generate"
      className="fixed inset-0 z-[9990] bg-black/80 flex items-center justify-center p-3"
      onClick={(e) => { if (e.target === e.currentTarget) dismiss(); }}>
      <div className="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-xl border border-indigo-200 bg-app p-4 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="text-indigo-700 font-semibold"><span aria-hidden>🎲</span> Quick generate</span>
          <button type="button" onClick={dismiss} disabled={submitting || busy}
            className="ml-auto text-content-subtle hover:text-content disabled:opacity-40" aria-label="Cancel">✕</button>
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">Total images</span>
            <input type="number" min="1" max="200" value={total}
              onChange={(e) => setTotal(Math.max(1, Math.min(200, Number(e.target.value) || 1)))}
              aria-label="Total images to generate"
              className="w-28 px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] tabular-nums focus:outline-none focus:ring-1 focus:ring-primary" />
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

        {nsfwMode && (
          <div className="flex flex-col gap-1 border-t border-border pt-2">
            <span className="flex justify-between items-center">
              <span className="text-content text-[0.75rem]"><span aria-hidden>🔞</span> NSFW ratio</span>
              <span className="text-content-muted text-[0.8125rem] font-semibold">{nsfwRatio}%</span>
            </span>
            <input type="range" min="0" max="100" value={nsfwRatio}
              aria-label="NSFW share of this batch"
              onChange={(e) => setNsfwRatio(Number(e.target.value))}
              className="w-full accent-rose-400" />
            <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
              NSFW shots only run on a local engine — raising this narrows the engine list below to Klein/Krea 2 Edit.
            </span>
          </div>
        )}

        {/* Dialog-owned engine picker — independent of the parent panel's
            selection while this full-screen overlay is open (see file
            header). Filtered to local-only engines while nsfwRatio > 0. */}
        <div className="flex flex-col gap-1 border-t border-border pt-2">
          <span className="text-content text-[0.75rem]">Engines for this batch</span>
          <fieldset className="flex flex-col gap-1">
            {offeredEngines.map((name) => (
              <label key={name} className="flex items-center gap-2 text-content-muted text-[0.75rem]">
                <input type="checkbox" checked={dialogEngines.includes(name)}
                  onChange={() => toggleDialogEngine(name)} />
                {ENGINE_LABELS[name] || name}
              </label>
            ))}
          </fieldset>
          {noLocalEngineForNsfw && (
            <span className="text-amber-700/90 text-[0.6875rem] leading-relaxed">
              {noLocalEngineAvailable
                ? 'NSFW content needs a local engine (Klein or Krea 2 Edit) — neither is currently available.'
                : 'Select at least one local engine (Klein or Krea 2 Edit) for NSFW content.'}
            </span>
          )}
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
            title={!hasRef ? 'Add a reference image first'
              : dialogEngines.length === 0 ? 'Pick at least one engine above' : undefined}
            className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            {submitting ? 'Generating…' : `🎲 Generate (${total})`}
          </button>
        </div>
      </div>
    </div>
  );
}
