// react-frontend/src/components/dataset/TrainingAdvancedGroups.jsx
//
// The 🔬 Expert — last-mile levers disclosure, extracted out of the already
// too-large TrainingPanel.jsx (see CLAUDE.md). It renders itself from
// GET /api/train/settings-map — never a client-kept copy of which setting
// reaches which lane, because that copy is exactly what drifted before.
//
// LAYOUT, per the two-sided map (`backend/app/services/training_settings_map.py`):
//   Shared     — applies on BOTH lanes.
//   This lane  — applies (enabled) or pinned (disabled, server reason) on the
//                CURRENT lane, whatever the other lane says about it.
//   The other lane — collapsed <details>; only settings the current lane has
//                no concept of (`absent`) AND the other lane does. Disabled.
// A setting `absent` on BOTH lanes (the map has never heard of it — the fail
// -safe default for an unknown key) has no honest home in either lane's own
// block, so it lands in "This lane", disabled, with a generic reason: it is
// certainly not something this lane's run reads, and there is no "other lane"
// to blame either.
//
// `settingsMap` is `{ lanes: { ai_toolkit: {...}, onetrainer: {...} }, groups }`
// exactly as the endpoint ships it, or `null` before the first fetch resolves.
// While null, every control just renders unlabelled and enabled (the same
// "nothing greyed yet" state the old one-sided fetch had before it resolved) —
// grouping only ever reflects data that actually arrived from the server.
import { HelpBadge } from '../../help/HelpMode';
import ConceptFaceMaskField from './ConceptFaceMaskField';
import { dualCaptionsSupport } from './dualCaptions.js';
import { MEMORY_KEYS, MEMORY_LABELS, memoryPatchFor } from './memorySavingAdvice';

const LANE_LABELS = { ai_toolkit: 'ai-toolkit', onetrainer: 'OneTrainer' };
const otherLaneOf = (lane) => (lane === 'onetrainer' ? 'ai_toolkit' : 'onetrainer');

const LR_SCHED_LABELS = {
  constant: 'Constant (default)', constant_with_warmup: 'Warmup → constant',
  linear: 'Linear decay', cosine: 'Cosine decay', cosine_with_restarts: 'Cosine + restarts',
};

/** `{state, why, group}` for one key on one lane, defaulting to the module's
 * own fail-safe (`absent`, no group) when the map hasn't heard of it. */
function laneEntry(settingsMap, key, lane) {
  return settingsMap?.lanes?.[lane]?.[key] || { state: 'absent', why: '', group: null };
}

/** Where a setting belongs for the CURRENT lane, and whether it is disabled.
 * Returns `null` while the map hasn't loaded — the caller then renders every
 * control enabled, in document order, with no section at all. */
function classify(settingsMap, key, lane) {
  if (!settingsMap) return null;
  const other = otherLaneOf(lane);
  const cur = laneEntry(settingsMap, key, lane);
  const otherE = laneEntry(settingsMap, key, other);
  const group = cur.group || otherE.group || null;
  if (cur.state === 'absent') {
    const why = `${LANE_LABELS[lane]} does not read this setting.`;
    // Known to the other lane → its own block. Known to neither → nowhere to
    // put it but here, disabled, rather than silently dropping a control.
    return { section: otherE.state !== 'absent' ? 'other' : 'this', group, disabled: true, why };
  }
  if (cur.state === 'applies' && otherE.state === 'applies') {
    return { section: 'shared', group, disabled: false, why: '' };
  }
  if (cur.state === 'pinned') {
    return { section: 'this', group, disabled: true, why: cur.why || `${LANE_LABELS[lane]} pins its own value for this setting.` };
  }
  // applies on this lane only (the other lane is pinned or absent)
  return { section: 'this', group, disabled: false, why: '' };
}

export default function TrainingAdvancedGroups({
  lane, settingsMap, caps, ds,
  advNetworkType, saveAdv, advNetworkChoices, advNetworkSupported, trainType,
  advLokrFactor, advLokrFactorChoices,
  advKreaRecipeSupported, advContentOrStyle, advContentOrStyleChoices, advContentOrStyleDefault,
  advDifferentialGuidance, differentialGuidanceScaleDraft, setDifferentialGuidanceScaleDraft, saveDifferentialGuidanceScale,
  advEma, advEmaChoices,
  advDualCaptions,
  advMaskFaces, advMaskFacesSupported, advMaskFacesConflict,
  advMemEff, advMemStateLabel, advMemTouched, advMemRiskLine, advMemAdviceText, advMemDefault, adv,
  advAlphaChoice, advAlphaChoices, advDefaultAlpha, advEffAlpha, advEffRank,
  advDropout, advDropoutChoices,
  advTimestepSupported, advTimestep, advTimestepChoices, advTimestepDefault,
  advOptimizer, advOptimizerChoices,
  advAdaptiveLr, learningRateDraft, setLearningRateDraft, saveLearningRate, learningRateError,
  minSnrDraft, setMinSnrDraft, saveMinSnr,
  advLrSched, advLrSchedChoices, advWarmup, advWarmupChoices,
  advGradAccum, advGradAccumChoices,
}) {
  const other = otherLaneOf(lane);
  const cls = (key) => classify(settingsMap, key, lane);
  const warmupCls = cls('warmup');

  // Original document order. When `settingsMap` hasn't loaded yet this is also
  // the render order — flat, nothing disabled, exactly the old behaviour.
  const defs = [
    {
      key: 'network_type',
      render: (rk, disabled, why) => (
        <div key={rk} className="flex flex-col gap-0.5" title={why || undefined}>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">Network</span>
            <select value={advNetworkType} onChange={(e) => saveAdv({ network_type: e.target.value })}
              aria-label="Network type"
              disabled={disabled}
              title={why || undefined}
              className="px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50">
              {advNetworkChoices.map((n) => <option key={n} value={n}>{n === 'lora' ? 'LoRA (default)' : 'LoKr'}</option>)}
            </select>
            {advNetworkType === 'lokr' && !advNetworkSupported && (
              <span className="text-amber-700 text-[0.625rem]" title={`LoKr isn't supported for ${trainType} — this run would fall back to LoRA.`}>⚠ not supported for {trainType}</span>
            )}
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> LoRA is the standard adapter; LoKr factorises
            the update differently. <b className="text-content-muted font-medium">How:</b> keep LoRA unless you are
            deliberately comparing it. The Krea Raw community starter below pins LoKr factor 16, but a network type
            alone cannot make up for the wrong images, captions or total steps.
          </span>
        </div>
      ),
    },
    {
      key: 'lokr_factor',
      when: advNetworkType === 'lokr',
      render: (rk, disabled, why) => (
        <div key={rk} className="flex items-center gap-2 flex-wrap" title={why || undefined}>
          <span className="text-content text-[0.75rem] w-28 shrink-0 inline-flex items-center gap-1">
            LoKr factor<HelpBadge topic="training.lokr_factor" />
          </span>
          <select value={advLokrFactor == null ? 'auto' : String(advLokrFactor)}
            onChange={(e) => saveAdv({ lokr_factor: e.target.value === 'auto' ? 'auto' : Number(e.target.value) })}
            aria-label="LoKr decomposition factor"
            disabled={disabled}
            title={why || undefined}
            className="px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50">
            <option value="auto">Auto (ai-toolkit)</option>
            {advLokrFactorChoices.map((factor) => <option key={factor} value={String(factor)}>{factor}</option>)}
          </select>
        </div>
      ),
    },
    {
      // Represents `content_or_style`, `do_differential_guidance` and
      // `differential_guidance_scale` together — one visual card, one entry.
      // Safe because the three keys carry IDENTICAL lane status in the map
      // (all ai-toolkit-applies / OneTrainer-absent): classifying by the
      // first is classifying by all three.
      key: 'content_or_style',
      when: advKreaRecipeSupported,
      render: (rk, disabled, why) => (
        <div key={rk} className="flex flex-col gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0 inline-flex items-center gap-1">
              Krea community recipe<HelpBadge topic="training.krea_community_recipe" />
            </span>
            <span className="text-amber-700 text-[0.6875rem] leading-relaxed">
              Community starting point, not a likeness promise — compare your own checkpoints.
            </span>
          </div>
          <div className="flex items-center gap-2 flex-wrap" title={why || undefined}>
            <span className="text-content text-[0.75rem] w-28 shrink-0">Content / style</span>
            <select value={advContentOrStyle ?? 'auto'}
              onChange={(e) => saveAdv({ content_or_style: e.target.value === 'auto' ? 'auto' : e.target.value })}
              aria-label="Krea content or style balance"
              disabled={disabled}
              title={why || undefined}
              className="px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50">
              <option value="auto">Auto ({advContentOrStyleDefault})</option>
              {advContentOrStyleChoices.map((mode) => <option key={mode} value={mode}>{mode}</option>)}
            </select>
          </div>
          <div className="flex items-center gap-2 flex-wrap" title={why || undefined}>
            <label className="flex items-center gap-2 flex-wrap cursor-pointer">
              <span className="text-content text-[0.75rem] w-28 shrink-0 inline-flex items-center gap-1">
                Differential guidance<HelpBadge topic="training.krea_community_recipe" />
              </span>
              <input type="checkbox" checked={advDifferentialGuidance}
                onChange={(e) => saveAdv({ do_differential_guidance: e.target.checked })}
                aria-label="Enable Krea differential guidance"
                disabled={disabled}
                title={why || undefined}
                className="h-4 w-4 rounded border-border bg-surface accent-indigo-500 disabled:cursor-not-allowed disabled:opacity-50" />
              <span className="text-content-muted text-[0.75rem]">enable</span>
            </label>
            <label className="flex items-center gap-2" title={why || undefined}>
              <span className="text-content-muted text-[0.6875rem]">scale</span>
              <input type="number" min="0.1" max="10" step="0.1"
                value={differentialGuidanceScaleDraft}
                disabled={!advDifferentialGuidance || disabled}
                onChange={(e) => setDifferentialGuidanceScaleDraft(e.target.value)}
                onBlur={saveDifferentialGuidanceScale}
                onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); }}
                aria-label="Krea differential guidance scale"
                title={why || undefined}
                className="w-16 px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50" />
            </label>
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Reported starter:</b> Balanced plus differential
            guidance at scale 3 is what the Krea 2 Raw · LoKr likeness preset applies. Change one variable at a
            time and keep the intermediate saves that actually work for your dataset.
          </span>
        </div>
      ),
    },
    {
      key: 'ema',
      render: (rk, disabled, why) => (
        <div key={rk} className="flex flex-col gap-0.5" title={why || undefined}>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">EMA</span>
            <select value={String(advEma)}
              onChange={(e) => saveAdv({ ema: e.target.value === '0' ? 'off' : Number(e.target.value) })}
              aria-label="EMA (exponential moving average)"
              disabled={disabled}
              title={why || undefined}
              className="px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50">
              <option value="0">Off (default)</option>
              {advEmaChoices.map((d) => <option key={d} value={String(d)}>{d}</option>)}
            </select>
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> exponential moving average of the weights —
            smoother, often better checkpoints. <b className="text-content-muted font-medium">How:</b> Off by
            default; 0.99 averages faster, 0.999 is slower and steadier. Test it as a separate variable: it is
            not part of the Krea Raw LoKr likeness starter.
          </span>
        </div>
      ),
    },
    {
      key: 'dual_captions',
      render: (rk) => (
        <div key={rk} className="flex flex-col gap-0.5">
          <label className="flex items-center gap-2 flex-wrap cursor-pointer">
            <span className="text-content text-[0.75rem] w-28 shrink-0 inline-flex items-center gap-1">
              Dual captions<HelpBadge topic="training.dual_captions" />
            </span>
            <input type="checkbox" checked={advDualCaptions}
              onChange={(e) => saveAdv({ dual_captions: e.target.checked })}
              aria-label="Dual long + short captions"
              className="h-4 w-4 rounded border-border bg-surface accent-indigo-500" />
            <span className="text-content-muted text-[0.75rem]">long + short (local training only)</span>
          </label>
          {/* Krea 2 / Anima cache their text embeddings, so the short caption
              can never be encoded. Say it here rather than let the user
              believe two wordings are training. Wraps — no fixed width. */}
          {advDualCaptions && !dualCaptionsSupport(trainType).supported && (
            <span className="text-amber-400 text-[0.6875rem] leading-relaxed">
              Ignored here: {dualCaptionsSupport(trainType).note}
            </span>
          )}
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> trains each image with both a full and a brief
            caption (text-side augmentation) so the LoRA leans less on any single wording.
            <b className="text-content-muted font-medium"> How:</b> the short variant is derived from the long one
            when you (re-)caption — same rules (no trigger, identity/concept/aesthetic kept out); edit it per image in
            the ⛶ caption editor. Cloud runs ignore this and train on the long caption only for now.
          </span>
        </div>
      ),
    },
    {
      // Represents `quantize`, `quantize_te` and `low_vram` (MEMORY_KEYS)
      // together for the same reason as the Krea recipe card above: identical
      // lane status for all three in the map (ai-toolkit-applies / OneTrainer
      // -absent), so classifying by the first classifies all three.
      key: 'quantize',
      render: (rk, disabled, why) => (
        <div key={rk} className="flex flex-col gap-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0 inline-flex items-center gap-1">
              Memory saving<HelpBadge topic="training.memory_saving" />
            </span>
            <span className="text-content-subtle text-[0.6875rem]">
              {advMemStateLabel}
            </span>
            {advMemTouched && (
              <button type="button"
                onClick={() => saveAdv(Object.fromEntries(MEMORY_KEYS.map((k) => [k, 'auto'])))}
                className="ml-auto text-[0.625rem] text-content-subtle hover:text-content underline underline-offset-2">
                Reset to default
              </button>
            )}
          </div>
          <div className="flex flex-col gap-1 sm:flex-row sm:flex-wrap sm:gap-x-4">
            {MEMORY_KEYS.map((k) => (
              <label key={k} className="flex items-center gap-2 cursor-pointer min-w-0" title={why || undefined}>
                <input type="checkbox" checked={Boolean(advMemEff[k])}
                  onChange={(e) => saveAdv(memoryPatchFor(k, e.target.checked, advMemDefault))}
                  aria-label={MEMORY_LABELS[k]}
                  disabled={disabled}
                  title={why || undefined}
                  className="h-4 w-4 shrink-0 rounded border-border bg-surface accent-indigo-500 disabled:cursor-not-allowed disabled:opacity-50" />
                <span className="text-content-muted text-[0.75rem] truncate">{MEMORY_LABELS[k]}</span>
              </label>
            ))}
          </div>
          {advMemRiskLine && (
            <span className={`text-[0.6875rem] leading-relaxed ${
              adv?.memory_risk?.verdict === 'can_disable' ? 'text-content-muted' : 'text-amber-700'}`}>
              {adv?.memory_risk?.verdict === 'can_disable' ? '' : '⚠️ '}{advMemRiskLine}
            </span>
          )}
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> the recipes are tuned so a 12B model fits in
            24 GB — quantisation costs precision and low-VRAM streaming costs a lot of speed. If your card is
            bigger than the target, you are paying for nothing.
            <b className="text-content-muted font-medium"> How:</b> {advMemAdviceText}
          </span>
        </div>
      ),
    },
    {
      key: 'alpha',
      render: (rk, disabled, why) => (
        <div key={rk} className="flex flex-col gap-0.5" title={why || undefined}>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">Alpha</span>
            <select value={String(advAlphaChoice)}
              onChange={(e) => saveAdv({ alpha: e.target.value === 'auto' ? 'auto' : Number(e.target.value) })}
              aria-label="LoRA alpha"
              disabled={disabled}
              title={why || undefined}
              className="px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50">
              <option value="auto">Auto (= {advDefaultAlpha})</option>
              {advAlphaChoices.map((a) => <option key={a} value={String(a)}>{a}</option>)}
            </select>
            <span className="text-content-subtle text-[0.625rem] tabular-nums">→ scale {(advEffAlpha / Math.max(1, advEffRank)).toFixed(2)}×</span>
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> alpha ÷ rank is the LoRA&apos;s effective strength while
            training — a soft learning-rate lever that isn&apos;t the LR. <b className="text-content-muted font-medium">How:</b> Auto
            ties alpha to rank (scale 1.0); a lower alpha (e.g. ½ rank) softens the fit — a clean way to stop a tiny
            (≤20-image) set from memorising without touching LR or rank.
          </span>
        </div>
      ),
    },
    {
      key: 'dropout',
      render: (rk, disabled, why) => (
        <div key={rk} className="flex flex-col gap-0.5" title={why || undefined}>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">Network dropout</span>
            <select value={String(advDropout)}
              onChange={(e) => saveAdv({ dropout: e.target.value === '0' ? 'off' : Number(e.target.value) })}
              aria-label="Network dropout"
              disabled={disabled}
              title={why || undefined}
              className="px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50">
              <option value="0">Off</option>
              {advDropoutChoices.map((d) => <option key={d} value={String(d)}>{d}</option>)}
            </select>
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> this is <i>network</i> dropout: it randomly drops
            adapter updates to reduce memorisation. It is separate from caption dropout. <b className="text-content-muted font-medium">How:</b> follow
            the preset; 0.05 is gentle, while larger values can underfit. Krea&apos;s text-embedding cache affects caption
            dropout, not this network control.
          </span>
        </div>
      ),
    },
    {
      key: 'timestep_type',
      when: advTimestepSupported,
      render: (rk, disabled, why) => (
        <div key={rk} className="flex flex-col gap-0.5" title={why || undefined}>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">Timestep weighting</span>
            <select value={advTimestep} onChange={(e) => saveAdv({ timestep_type: e.target.value })}
              aria-label="Timestep weighting"
              disabled={disabled}
              title={why || undefined}
              className="px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50">
              <option value="auto">Auto ({advTimestepDefault})</option>
              {advTimestepChoices.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> which noise levels the loss emphasises — the
            detail-versus-global-structure balance for flow-matching models. <b className="text-content-muted font-medium">How:</b> Auto
            uses the family recipe ({advTimestepDefault}); use the researched Style preset unless you are deliberately
            testing texture/detail versus composition/structure emphasis.
          </span>
        </div>
      ),
    },
    {
      key: 'optimizer',
      render: (rk, disabled, why) => (
        <div key={rk} className="flex flex-col gap-0.5" title={why || undefined}>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">Optimizer</span>
            <select value={advOptimizer} onChange={(e) => saveAdv({ optimizer: e.target.value })}
              aria-label="Optimizer"
              disabled={disabled}
              title={why || undefined}
              className="px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50">
              {advOptimizerChoices.map((o) => <option key={o} value={o}>{o}{o === 'adamw8bit' ? ' (default)' : ''}</option>)}
            </select>
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> how the weights are updated — the biggest training
            lever after the dataset. <b className="text-content-muted font-medium">How:</b> <i>adamw8bit</i> (default)
            is fast and VRAM-light; <i>adafactor</i> uses less memory and auto-scales; <i>automagic</i>/<i>automagic2</i>
            use ai-toolkit&apos;s adaptive update rule; the Krea community starter still records its reported initial
            LR of 1e-4. <i>prodigy</i> also auto-tunes the LR and is popular for tiny sets — but may need
            <code className="text-content-muted">pip install prodigyopt</code> in the ai-toolkit venv. Picking an
            adaptive optimiser is the &quot;push further without cranking the LR&quot; move.
          </span>
        </div>
      ),
    },
    {
      key: 'learning_rate',
      render: (rk, disabled, why) => (
        <div key={rk} className="flex flex-col gap-0.5" title={why || undefined}>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">Learning rate</span>
            <input type="text" inputMode="decimal"
              value={learningRateDraft}
              onChange={(e) => setLearningRateDraft(e.target.value)}
              onBlur={saveLearningRate}
              placeholder={advAdaptiveLr ? 'set by the optimiser' : 'auto (1e-4)'}
              aria-label="Learning rate"
              disabled={advAdaptiveLr || disabled}
              title={why
                || (advAdaptiveLr
                  ? 'This optimiser tunes the learning rate itself — a fixed rate would be ignored.'
                  : 'Blank = the family default (1e-4). Example: 3e-4')}
              className="px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] w-32 focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50" />
            <span className="text-content-subtle text-[0.6875rem]" hidden={!learningRateError}>
              {learningRateError}
            </span>
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> how far each step moves the weights. Too high
            bakes the source&apos;s look in; too low never converges.
            <b className="text-content-muted font-medium"> How:</b> leave it blank for the family default
            (<i>1e-4</i>) unless you have a reason. Adaptive optimisers (<i>prodigy</i>, <i>automagic</i>) set it
            themselves and this field is disabled for them.
          </span>
        </div>
      ),
    },
    {
      key: 'min_snr_gamma',
      render: (rk, disabled, why) => (
        <div key={rk} className="flex flex-col gap-0.5" title={why || undefined}>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">Min-SNR gamma</span>
            <input type="number" min={0} max={20} step={1}
              value={minSnrDraft}
              onChange={(e) => setMinSnrDraft(e.target.value)}
              onBlur={saveMinSnr}
              placeholder="off"
              aria-label="Min-SNR gamma"
              disabled={disabled}
              title={why || 'Empty = off. 5 is the usual value; both trainers ignore a 0.'}
              className="w-24 rounded-lg bg-surface-raised px-2 py-1 text-content tabular-nums text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50" />
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> weights the loss by how much signal each
            noise level carries, which mostly helps a high-resolution run converge instead of spending its
            early steps on the noisiest timesteps.
            <b className="text-content-muted font-medium"> How:</b> leave it empty unless a run is converging
            slowly at 1024; <i>5</i> is the value most recipes use.
          </span>
        </div>
      ),
    },
    {
      key: 'lr_scheduler',
      render: (rk, disabled, why) => (
        <div key={rk} className="flex flex-col gap-0.5" title={why || undefined}>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">LR schedule</span>
            <select value={advLrSched} onChange={(e) => saveAdv({ lr_scheduler: e.target.value })}
              aria-label="Learning-rate schedule"
              disabled={disabled}
              title={why || undefined}
              className="px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50">
              {advLrSchedChoices.map((s) => <option key={s} value={s}>{LR_SCHED_LABELS[s] || s}</option>)}
            </select>
            {advLrSched === 'constant_with_warmup' && (
              <select value={String(advWarmup || 100)} onChange={(e) => saveAdv({ warmup: Number(e.target.value) })}
                aria-label="Warmup steps"
                disabled={!!warmupCls?.disabled}
                title={warmupCls?.why || undefined}
                className="px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50">
                {advWarmupChoices.map((w) => <option key={w} value={String(w)}>{w} warmup</option>)}
              </select>
            )}
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> how the learning rate moves over the run.
            <b className="text-content-muted font-medium"> How:</b> <i>Constant</i> (default) holds it flat;
            <i> Warmup → constant</i> ramps it up over the first N steps (a gentler start that avoids early
            over-commitment on a small set) then holds; <i>Linear</i>/<i>Cosine</i> decay it toward 0 by the end for
            cleaner convergence. The warmup-steps box only applies to the warmup schedule.
          </span>
        </div>
      ),
    },
    {
      key: 'grad_accum',
      render: (rk, disabled, why) => (
        <div key={rk} className="flex flex-col gap-0.5" title={why || undefined}>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">Effective batch</span>
            <select value={String(advGradAccum)} onChange={(e) => saveAdv({ grad_accum: Number(e.target.value) })}
              aria-label="Gradient accumulation"
              disabled={disabled}
              title={why || undefined}
              className="px-2 py-1 rounded-lg bg-surface-raised text-content text-[0.75rem] focus:outline-none focus:ring-1 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-50">
              {advGradAccumChoices.map((g) => <option key={g} value={String(g)}>{g === 1 ? '1 (default)' : `${g} × accum`}</option>)}
            </select>
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> averages the gradient over N micro-batches before
            each update — a larger <i>effective</i> batch with no extra VRAM. <b className="text-content-muted font-medium">How:</b> 1
            (default); 2–4 smooths the noisy gradients a tiny dataset produces (steadier training), at the cost of a
            bit more time per update. A cheap stabiliser for small sets.
          </span>
        </div>
      ),
    },
  ].filter((d) => d.when !== false);

  // --- No map yet: flat, nothing disabled, document order. -----------------
  if (!settingsMap) {
    return (
      <ExpertShell>
        <ConceptFaceMaskField
          datasetId={ds.currentId}
          enabled={advMaskFaces}
          supported={advMaskFacesSupported}
          conceptConflict={advMaskFacesConflict}
          faceCapability={caps.face_scoring}
          expandDefault={undefined}
          onToggle={(v) => saveAdv({ mask_faces: v })}
        />
        <div className="flex flex-col divide-y divide-indigo-400/10 [&>div]:py-2.5 [&>div:first-child]:pt-1 [&>div:last-child]:pb-0">
          {defs.map((d) => d.render(d.key, false, ''))}
        </div>
      </ExpertShell>
    );
  }

  const groups = settingsMap.groups || [];
  const classified = defs.map((d) => ({ ...d, status: cls(d.key) }));
  const bySection = (section) => classified.filter((d) => d.status.section === section);
  const shared = bySection('shared');
  const thisLane = bySection('this');
  const otherItems = bySection('other');

  const renderGroups = (items) => {
    const known = groups
      .map((g) => ({ g, items: items.filter((d) => d.status.group === g.key) }))
      .filter((x) => x.items.length > 0);
    const ungrouped = items.filter((d) => !d.status.group);
    return (
      <>
        {known.map(({ g, items: gi }) => (
          <div key={g.key} className="flex flex-col gap-2.5">
            <span className="text-indigo-700/70 text-[0.6875rem] font-semibold uppercase tracking-wider">{g.label}</span>
            <div className="flex flex-col divide-y divide-indigo-400/10 [&>div]:py-2.5 [&>div:first-child]:pt-0 [&>div:last-child]:pb-0">
              {gi.map((d) => d.render(d.key, d.status.disabled, d.status.why))}
            </div>
          </div>
        ))}
        {ungrouped.length > 0 && (
          <div className="flex flex-col divide-y divide-indigo-400/10 [&>div]:py-2.5 [&>div:first-child]:pt-0 [&>div:last-child]:pb-0">
            {ungrouped.map((d) => d.render(d.key, d.status.disabled, d.status.why))}
          </div>
        )}
      </>
    );
  };

  return (
    <ExpertShell>
      <ConceptFaceMaskField
        datasetId={ds.currentId}
        enabled={advMaskFaces}
        supported={advMaskFacesSupported}
        conceptConflict={advMaskFacesConflict}
        faceCapability={caps.face_scoring}
        expandDefault={undefined}
        onToggle={(v) => saveAdv({ mask_faces: v })}
      />
      {shared.length > 0 && (
        <div className="flex flex-col gap-3">
          <span className="text-content-muted text-[0.625rem] font-semibold uppercase tracking-wider">Shared</span>
          {renderGroups(shared)}
        </div>
      )}
      {thisLane.length > 0 && (
        <div className="flex flex-col gap-3">
          <span className="text-content-muted text-[0.625rem] font-semibold uppercase tracking-wider">{LANE_LABELS[lane]} settings</span>
          {renderGroups(thisLane)}
        </div>
      )}
      {otherItems.length > 0 && (
        <details className="rounded-lg bg-surface px-2.5 py-2">
          <summary className="cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden text-content-subtle text-[0.6875rem] font-semibold uppercase tracking-wider">
            {LANE_LABELS[other]} only — this lane does not read these
          </summary>
          <div className="flex flex-col gap-3 mt-2.5">
            {renderGroups(otherItems)}
          </div>
        </details>
      )}
    </ExpertShell>
  );
}

/* The outer disclosure + its summary line, unchanged from the pre-restructure
   markup — every control inside still defaults to current behaviour, so a
   newcomer who never opens this is unaffected. */
function ExpertShell({ children }) {
  return (
    <details className="group rounded-lg border border-indigo-200 border-l-[3px] border-l-indigo-400 bg-indigo-500/[0.14] transition-colors hover:bg-indigo-100">
      <summary className="flex items-center gap-2 cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden px-2.5 py-2.5 text-[0.6875rem] font-semibold uppercase tracking-wider text-indigo-100 hover:text-white">
        <span aria-hidden className="text-indigo-700 transition-transform group-open:rotate-90">▸</span>
        <span aria-hidden>🔬</span>
        <span>Expert — last-mile levers</span>
        <span className="ml-auto hidden sm:inline normal-case font-normal tracking-normal text-indigo-700/50">shared · this lane · the other lane, collapsed</span>
      </summary>
      <div className="flex flex-col gap-3 px-2.5 pb-2.5 pt-1">
        {children}
      </div>
    </details>
  );
}
