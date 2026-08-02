// react-frontend/src/components/dataset/studio/LoraStackPanel.jsx
/**
 * Bascule Compare / Combine + poids par LoRA de la pile (≥2 LoRA cochés).
 *
 * « Compare » = comportement historique : chaque LoRA est testé SEUL, une colonne
 * par LoRA. « Combine » = les LoRA cochés sont chargés ENSEMBLE dans la même image,
 * chacun à son poids, et tous leurs triggers sont injectés dans le prompt.
 *
 * La logique (clé de poids, clamp, blocage inter-familles, coût) vit dans
 * ./loraStack.js — testée sous `node --test`, que le JSX rend inaccessible ici.
 */
import { HelpBadge } from '../../../help/HelpMode';
import {
  COMBINE_MAX_WEIGHT, COMBINE_MIN_WEIGHT, combineBlocker, stackKey, stackWeight,
} from './loraStack';

export default function LoraStackPanel({ selection, mode, onMode, weights, onWeight }) {
  const combine = mode === 'combine';
  const blocker = combine ? combineBlocker(selection) : null;

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-content-muted text-[0.6875rem] uppercase">
          How to use the {selection.length} LoRAs
        </span>
        <HelpBadge topic="studio-combine-loras" />
        <div role="group" aria-label="LoRA run mode"
          className="ml-auto flex rounded-lg border border-border bg-app/60 p-0.5">
          {[['compare', '⚖ Compare'], ['combine', '🧬 Combine']].map(([value, label]) => (
            <button key={value} type="button" onClick={() => onMode(value)}
              aria-pressed={mode === value}
              className={`px-2.5 py-1 rounded text-[0.6875rem] font-semibold ${
                mode === value ? 'bg-primary/30 text-content' : 'text-content-subtle hover:text-content'
              }`}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <p className="m-0 text-content-subtle text-[0.6875rem]">
        {combine
          ? 'All checked LoRAs load together in the same image, each at its own weight. '
            + 'Every trigger word is injected into the prompt. One image per seed — the '
            + 'strength sweep is replaced by these weights.'
          : 'Each LoRA is tested on its own, one column per LoRA, swept across the '
            + 'strengths below.'}
      </p>

      {combine && blocker && (
        <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-2.5 py-1.5 text-amber-200 text-[0.6875rem]"
          role="status">
          {blocker}
        </p>
      )}

      {combine && !blocker && (
        <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
          {selection.map((s, i) => {
            const k = stackKey(s);
            const w = stackWeight(weights, s);
            return (
              // Le nom et le curseur sont sur DEUX lignes : côte à côte, la colonne
              // étroite du studio (320 px, et 400 px en mobile) écrasait le nom en
              // « l.. » — un sélecteur de LoRA où on ne lit pas le LoRA.
              <li key={k}
                className="flex flex-col gap-1 rounded-lg border border-border bg-surface-raised px-2.5 py-1.5">
                <div className="flex min-w-0 items-center gap-2">
                  <span className="shrink-0 text-content-subtle text-[0.625rem] tabular-nums">{i + 1}.</span>
                  <span className="min-w-0 flex-1 truncate text-content text-sm" title={s.lora_label}>
                    {s.lora_label}
                  </span>
                  {s.trigger_word && (
                    <code className="shrink-0 rounded border border-indigo-400/40 bg-indigo-500/10 px-1.5 py-0.5 text-[0.625rem] font-semibold text-indigo-300">
                      {s.trigger_word}
                    </code>
                  )}
                </div>
                <label className="flex items-center gap-1.5 text-content-muted text-[0.6875rem]">
                  <span className="shrink-0 uppercase">Weight</span>
                  <input type="range" min={COMBINE_MIN_WEIGHT} max={COMBINE_MAX_WEIGHT} step="0.05"
                    value={w} onChange={(e) => onWeight(k, Number(e.target.value))}
                    aria-label={`Weight for ${s.lora_label}`}
                    className="min-w-0 flex-1 accent-primary" />
                  <span className="w-9 shrink-0 text-right tabular-nums text-content">{w.toFixed(2)}</span>
                </label>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
