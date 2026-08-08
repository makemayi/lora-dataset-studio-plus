// react-frontend/src/components/dataset/studio/StackVariantsGrid.jsx
/**
 * Vue résultats d'une PILE 🧬 : une COLONNE par VARIANTE DE POIDS de la même pile,
 * pas par LoRA.
 *
 * La grille de comparaison (colonnes = LoRA, lignes = strength) est le bon outil pour
 * opposer des LoRA ; sur une pile elle produit une colonne unique à une ligne. Ce qui
 * varie d'un run de pile à l'autre, ce sont les POIDS — donc les colonnes sont les
 * relances (backend : `stack_variants`, même composition, run courant compris), les
 * lignes sont les LoRA de la pile avec leur poids, et les images de chaque variante
 * sont posées sous sa colonne, votables sur place.
 *
 * Chaque variante donne deux gestes : « Open » (la charger comme run courant, pour
 * voter/relancer dessus) et « Use these weights » (recharger ses poids dans les
 * curseurs du panneau de lancement, pour repartir de là).
 *
 * Responsive : le tableau vit dans son PROPRE conteneur `overflow-x-auto` — à 400 px
 * il défile horizontalement au lieu de faire déborder la page.
 */
import { fmt } from '../../../utils/studioFormat';
import { HelpBadge } from '../../../help/HelpMode';
import ResultTile from './ResultTile';
import { alignWeights, comboLabelText, variantKey, variantSummary, weightVectorText, weightsIntoStackMap } from './stackResults';

export default function StackVariantsGrid({
  members, variants, onRate, onOpen, onSelectRun, onUseWeights,
}) {
  const list = variants || [];
  if (!members?.length || list.length === 0) return null;
  const activeWeights = (list.find((v) => v.active) || list[0])?.weights || [];

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-content-muted text-[0.6875rem] uppercase">
          🧬 Weight variants of this stack ({list.length})
        </span>
        <HelpBadge topic="studio-stack-results" />
        {list.length === 1 && (
          <span className="text-content-subtle text-[0.6875rem]">
            Change the weights on the left and run again — the next run lands here as a
            second column.
          </span>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-1">
          <caption className="sr-only">
            Stack weight variants: columns = one run of the stack, rows = each LoRA’s weight
          </caption>
          <thead>
            <tr>
              <th scope="col" className="px-1 text-left text-content-subtle text-[0.625rem] font-normal">
                LoRA \ run
              </th>
              {list.map((v) => (
                <th key={variantKey(v)} scope="col"
                  title={comboLabelText(v.weights)}
                  className={`px-1.5 py-1 text-[0.6875rem] font-semibold rounded ${v.active
                    ? 'bg-sky-500/15 text-sky-200 border border-sky-400/50'
                    : 'text-content border border-transparent'}`}>
                  <span className="tabular-nums">{weightVectorText(v.weights)}</span>
                  {v.active && <span className="ml-1 font-normal text-[0.625rem]">(shown)</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {members.map((m, i) => (
              <tr key={`${m.dataset_id}:${m.filename}`}>
                <th scope="row"
                  className="max-w-[9rem] truncate px-1 text-left text-content-muted text-[0.6875rem] font-normal"
                  title={`${m.label}${m.trigger ? ` — trigger ${m.trigger}` : ''}`}>
                  {i + 1}. {m.label}
                </th>
                {list.map((v) => {
                  const row = alignWeights(members, v.weights, v.active ? null : activeWeights)[i];
                  return (
                    <td key={variantKey(v)}
                      className={`px-1.5 py-0.5 text-center text-[0.6875rem] tabular-nums rounded ${row.changed
                        ? 'bg-amber-400/15 text-amber-200 font-semibold'
                        : 'text-content-muted'}`}
                      title={row.changed
                        ? `${row.delta > 0 ? '+' : ''}${row.delta.toFixed(2)} vs the run shown`
                        : undefined}>
                      {row.weight == null ? '—' : row.weight.toFixed(2)}
                      {row.changed && (
                        <span aria-label={`${row.delta > 0 ? 'higher' : 'lower'} than the run shown`}>
                          {' '}{row.delta > 0 ? '↑' : '↓'}
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
            <tr>
              <th scope="row" className="px-1 text-left text-content-subtle text-[0.625rem] font-normal">
                images
              </th>
              {list.map((v) => (
                <td key={variantKey(v)} className="align-top p-1">
                  {v.cells?.length ? (
                    <div className="flex flex-wrap items-start gap-1">
                      {v.cells.map((c) => (
                        <ResultTile key={c.id} cell={c}
                          row={{ label: c.label || '' }} strength={c.strength}
                          variant={{ aspect: c.aspect || '' }}
                          datasetId={c.dataset_id} onRate={onRate} onOpen={onOpen} fmt={fmt} />
                      ))}
                    </div>
                  ) : (
                    <span className="text-content-subtle text-[0.625rem]">—</span>
                  )}
                </td>
              ))}
            </tr>
            <tr>
              <th scope="row" className="px-1 text-left text-content-subtle text-[0.625rem] font-normal">
                votes
              </th>
              {list.map((v) => {
                const s = variantSummary(v);
                return (
                  <td key={variantKey(v)} className="px-1.5 py-0.5 text-center text-[0.6875rem] tabular-nums">
                    <span className="text-green-300">👍 {s.likes}</span>{' '}
                    <span className="text-red-300">👎 {s.dislikes}</span>
                    <span className="ml-1 text-content-subtle"
                      title="Likes minus dislikes on this variant only">
                      net {s.net > 0 ? `+${s.net}` : s.net}
                    </span>
                  </td>
                );
              })}
            </tr>
            <tr>
              <th scope="row" className="px-1" />
              {list.map((v) => (
                <td key={variantKey(v)} className="px-1 py-1">
                  <div className="flex flex-col gap-1">
                    <button type="button" disabled={v.active} onClick={() => onSelectRun?.(v.run_id)}
                      className="rounded border border-border bg-surface px-1.5 py-0.5 text-[0.625rem] text-content disabled:opacity-40">
                      {v.active ? 'Shown' : 'Open this run'}
                    </button>
                    <button type="button"
                      onClick={() => onUseWeights?.(weightsIntoStackMap(members, v.weights))}
                      title="Load these weights back into the sliders, then run again"
                      className="rounded border border-sky-400/40 bg-sky-400/10 px-1.5 py-0.5 text-[0.625rem] text-sky-200">
                      Use these weights
                    </button>
                  </div>
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      <p className="m-0 text-content-subtle text-[0.625rem] leading-relaxed">
        Votes are counted per variant, so the column with the best net score is the
        weight set to keep. Older relaunches of this stack are found by their LoRAs, and
        only the most recent ones are listed.
      </p>
    </div>
  );
}
