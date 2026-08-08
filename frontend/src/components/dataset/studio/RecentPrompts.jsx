// Vignettes des prompts récents (clic pour recharger) — rétro-compat string vs objet.
// Extrait behavior-preserving de LoraTestStudio.jsx (bloc « Prompts récents »),
// + bouton 🗑 par preset (supprime le prompt et ses cellules/images de test).
//
// 📝 LOT — chaque carte porte une case à cocher. Cocher n'écrit RIEN dans le
// champ prompt (c'est le rôle du clic sur la carte, inchangé) : la coche décrit
// ce que le prochain lancement doit rejouer, une génération par prompt coché,
// avec les réglages courants du panneau. Aucune case cochée = le panneau se
// comporte exactement comme avant, prompt du champ compris.
//
// Le composant est monté par les DEUX surfaces (Test Studio du dataset et
// panneau « Generate from the board » du canvas) via PromptField/RunSetupPanel :
// le lot existe donc des deux côtés par construction, pas par duplication.
import { HelpBadge } from '../../../help/HelpMode';

export default function RecentPrompts({
  items, datasetId, selectedPrompt, onPick, onDelete,
  batch = null, onToggleBatch = null, onClearBatch = null,
}) {
  // Le lot n'est proposé que si l'hôte l'accepte — un appelant qui ne passe pas
  // onToggleBatch garde le composant d'avant, à l'octet près.
  const batchable = typeof onToggleBatch === 'function';
  const picked = Array.isArray(batch) ? batch : [];

  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
        <span className="text-content-subtle text-[0.5625rem] uppercase">
          Recent prompts (click to reload · 🗑 to delete) — thumbnail = image 👍
        </span>
        {batchable && <HelpBadge topic="studio-prompt-batch" />}
        {batchable && picked.length > 0 && (
          <span className="flex items-center gap-1.5">
            <span className="rounded bg-purple-500/20 px-1.5 py-0.5 text-[0.5625rem] font-semibold text-purple-200 tabular-nums">
              {picked.length} selected
            </span>
            <button type="button" onClick={onClearBatch}
              className="text-content-subtle text-[0.5625rem] underline decoration-dotted hover:text-content">
              Clear
            </button>
          </span>
        )}
      </div>
      {batchable && (
        <p className="m-0 text-content-subtle text-[0.5625rem]">
          Tick several prompts to generate them all in one run — same checkpoints,
          same settings, one image set per prompt.
        </p>
      )}
      <div className="flex gap-1.5 flex-wrap">
        {items.map((item) => {
          // rétro-compat : avant restart Flask, l'API renvoie des strings ;
          // après, des objets {prompt, thumbnail, thumb_rating, count}.
          const pr = typeof item === 'string' ? { prompt: item } : item;
          const sel = selectedPrompt === pr.prompt;
          const inBatch = picked.includes(pr.prompt);
          // Conteneur = la « carte » (porte la bordure) ; boutons frères à
          // l'intérieur (PAS imbriqués) : cocher (lot) + recharger + supprimer.
          return (
            <div key={pr.prompt}
              className={`flex items-stretch rounded-lg border text-[0.625rem] max-w-[260px] overflow-hidden ${
                inBatch
                  ? 'border-purple-400 bg-purple-500/25'
                  : sel ? 'border-purple-400/60 bg-purple-500/20' : 'border-border bg-surface'}`}>
              {batchable && (
                <button type="button" role="checkbox" aria-checked={inBatch}
                  onClick={() => onToggleBatch(pr.prompt)}
                  title={inBatch ? 'Remove this prompt from the batch' : 'Add this prompt to the batch'}
                  aria-label={inBatch ? 'Remove this prompt from the batch' : 'Add this prompt to the batch'}
                  className={`shrink-0 px-1.5 flex items-center border-r border-border ${
                    inBatch ? 'text-purple-200' : 'text-content-subtle hover:text-content'}`}>
                  <span aria-hidden>{inBatch ? '☑' : '☐'}</span>
                </button>
              )}
              <button type="button" onClick={() => onPick(pr.prompt)} title={pr.prompt}
                className={`flex items-center gap-1.5 p-1 text-left min-w-0 ${
                  sel ? 'text-purple-200' : 'text-content-muted'}`}>
                {pr.thumbnail
                  ? <img src={`/api/dataset/${pr.thumb_dataset_id ?? datasetId}/img/${encodeURIComponent(pr.thumbnail)}`}
                      alt="" loading="lazy"
                      className="w-8 h-10 object-cover rounded shrink-0" />
                  : <span className="w-8 h-10 rounded bg-app/60 shrink-0 flex items-center justify-center text-content-subtle">?</span>}
                <span className="flex flex-col items-start min-w-0">
                  <span className="truncate max-w-[150px]">{pr.prompt}</span>
                  {pr.count ? <span className="text-content-subtle">{pr.count} img{pr.thumb_rating === 1 ? ' · 👍' : ''}</span> : null}
                </span>
              </button>
              {onDelete && (
                <button type="button"
                  onClick={() => {
                    const n = pr.count ? ` and its ${pr.count} test image(s)` : '';
                    if (window.confirm(`Delete this recent prompt${n}?`)) onDelete(pr.prompt);
                  }}
                  title="Delete this recent prompt (and its test images)"
                  aria-label="Delete this recent prompt"
                  className="shrink-0 px-1.5 flex items-center border-l border-border text-red-300/70 hover:text-red-300 hover:bg-red-500/15">
                  🗑
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
