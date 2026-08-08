// react-frontend/src/components/dataset/studio/ResultsGrid.jsx
/**
 * Grille(s) de résultats : une `<table>` par variante (format × cfg × steps ×
 * prompt). Lignes = checkpoint, colonnes = strength ; chaque case = `<ResultCell>`
 * (bande de tuiles + score + ★). Extrait 1:1 du bloc `<table>` de l'ancien
 * LoraTestStudio : mêmes classes Tailwind, même en-tête « ckpt \ strength ».
 *
 * Le lot de prompts 📝 emprunte cette géométrie plutôt que d'en inventer une
 * troisième : un balayage de CFG donne déjà une table par valeur, sous une
 * légende qui la nomme. Un lot de N prompts donne donc N tables, chacune sous SON
 * prompt — `showPromptLabels` n'ajoute la ligne que lorsqu'il y a plusieurs
 * prompts à distinguer.
 */
import ResultCell from './ResultCell';
import { promptLabel } from './resultKeys';

export default function ResultsGrid({ gridRows, gridCols, variantsInData, showPromptLabels, cellList, scoreMap, best, datasetId, onRate, onOpen, fmt }) {
  return variantsInData.map((variant) => (
    <div key={variant.key} className="flex flex-col gap-1">
      {showPromptLabels && variant.prompt && (
        // Le prompt entier reste dans le `title` : la légende, elle, doit tenir
        // sur une ligne même à 400 px — un prompt de test fait des centaines de
        // caractères et repousserait la grille hors de l'écran.
        <span className="text-content text-[0.6875rem] font-medium truncate max-w-full"
          title={variant.prompt}>
          <span aria-hidden>📝</span> {promptLabel(variant.prompt)}
        </span>
      )}
      {variantsInData.length > 1 && (
        <span className="text-content-muted text-[0.625rem] uppercase">
          {variant.zModelLabel ? `${variant.zModelLabel} · ` : ''}Format {variant.aspect || '—'}{variant.cfg != null ? ` · CFG ${fmt(variant.cfg)}` : ''}{variant.steps != null ? ` · ${variant.steps}${variant.steps2 != null ? '/' + variant.steps2 : ''} steps` : ''}
        </span>
      )}
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-1">
          {/* Le libellé, pas la clé : `variant.key` est un identifiant technique
              (et il porte maintenant le prompt ENTIER) — le lire à voix haute
              n'apprend rien à personne. */}
          <caption className="sr-only">
            Test grid{variant.prompt ? ` for prompt “${promptLabel(variant.prompt)}”` : ''}:
            rows = checkpoint, columns = strength
          </caption>
          <thead>
            <tr>
              <th scope="col" className="text-content-subtle text-[0.625rem] font-normal text-left px-1">ckpt \ strength</th>
              {gridCols.map((s) => (
                <th key={s} scope="col" className="text-content-muted text-[0.6875rem] tabular-nums px-1">{fmt(s)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {gridRows.map((row) => (
              <tr key={row.filename}>
                <th scope="row" className="text-content text-[0.6875rem] font-medium text-left px-1 whitespace-nowrap">{row.label}</th>
                {gridCols.map((s) => (
                  <ResultCell key={s} row={row} strength={s} variant={variant}
                    cellList={cellList} scoreMap={scoreMap} best={best} datasetId={datasetId}
                    onRate={onRate} onOpen={onOpen} fmt={fmt} />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  ));
}
