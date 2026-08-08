// react-frontend/src/components/dataset/studio/ResultsArea.jsx
/**
 * Zone « 📊 Résultats » du Studio de test LoRA. Possède l'état d'affichage
 * (repli `showResults`) et le run sélectionné (`selRun`), recalcule tout le
 * regroupement par run / config / variante à partir de `d.cells` et `d.scores`
 * (extraction behavior-preserving depuis l'ancien LoraTestStudio.jsx), puis rend
 * le sélecteur de run + une grille par variante (format × cfg × steps).
 */
import { useCallback, useMemo, useState } from 'react';
import { fmt } from '../../../utils/studioFormat';
import { flipOrder } from './flipOrder';
import { runKey, variantKey, variantOf, cellKey, distinctPrompts } from './resultKeys';
import RunSelector from './RunSelector';
import ResultsGrid from './ResultsGrid';
import ExportGridModal from './ExportGridModal';

export default function ResultsArea({ datasetId, d, studio, vote, onOpen }) {
  // Repli des grilles de résultats (pour ne pas encombrer la page).
  const [showResults, setShowResults] = useState(true);
  // Run sélectionné (null = run le plus récent par défaut).
  const [selRun, setSelRun] = useState(null);
  // Modale « Export grid » (compose le run affiché en UNE image partageable).
  const [exportOpen, setExportOpen] = useState(false);

  // --- Regroupement par RUN (un lancement = même seed + prompt + modèle). On
  // n'affiche que le run sélectionné (le plus récent par défaut) pour ne pas
  // mélanger d'anciens tests déjà votés avec un nouveau run.
  const runs = useMemo(() => {
    const groups = new Map();
    for (const c of d?.cells || []) {
      // Un lancement = son `run_id` (cf. resultKeys.runKey) : les N seeds d'un
      // batch, TOUS les modèles de base balayés ET tous les prompts du lot 📝.
      // Le modèle comme le prompt sont des axes de VARIANTE, pas des runs
      // distincts — les mettre dans cette clé coupait un lot de N prompts en N
      // pseudo-runs, dont la vue n'en montrait qu'un.
      const key = runKey(c);
      let g = groups.get(key);
      if (!g) {
        g = { key, seed: c.run_seed ?? c.seed, prompt: c.prompt || '', models: new Set(),
              prompts: new Set(), cells: [], latestId: 0, likes: 0, dislikes: 0 };
        groups.set(key, g);
      }
      g.cells.push(c);
      if (c.z_model_label) g.models.add(c.z_model_label);
      g.prompts.add(c.prompt || '');
      if (c.id > g.latestId) g.latestId = c.id;
      if (c.rating === 1) g.likes += 1; else if (c.rating === -1) g.dislikes += 1;
    }
    return [...groups.values()].map((g) => ({
      ...g, modelLabel: g.models.size > 1 ? `${g.models.size} models` : ([...g.models][0] || ''),
      // Même règle que les modèles : un lot annonce son COMPTE, pas le premier de
      // ses prompts — le sélecteur mentirait en nommant un seul des cinq.
      promptLabel: g.prompts.size > 1 ? `${g.prompts.size} prompts` : g.prompt,
    })).sort((a, b) => b.latestId - a.latestId);
  }, [d]);
  const activeRunKey = (runs.find((r) => r.key === selRun) ? selRun : runs[0]?.key) || null;
  const displayedCells = useMemo(() => {
    const r = runs.find((x) => x.key === activeRunKey);
    return r ? r.cells : [];
  }, [runs, activeRunKey]);

  // Cellules par config dans le run affiché (clé : resultKeys.cellKey — checkpoint
  // × strength × la variante, PROMPT COMPRIS ; c'est la même fonction que celle
  // dont `ResultCell` se sert pour retrouver sa case).
  // Batch : TOUTES les cellules par config (les N seeds), triées par seed → bande.
  const cellList = useMemo(() => {
    const m = new Map();
    for (const c of displayedCells) {
      const k = cellKey(c);
      if (!m.has(k)) m.set(k, []);
      m.get(k).push(c);
    }
    for (const arr of m.values()) arr.sort((a, b) => (a.seed || 0) - (b.seed || 0));
    return m;
  }, [displayedCells]);

  // --- Ordre de FEUILLETAGE de la lightbox --------------------------------------
  // Le set navigable = les cellules AFFICHÉES (run courant), triées pour que les
  // variantes de strength d'un même rendu soient adjacentes : variante (z_model /
  // aspect / cfg / steps / prompt) → checkpoint → seed → STRENGTH en dernier.
  // Le prompt se range AVEC les autres axes de variante, pour que feuilleter suive
  // les grilles affichées au lieu de sauter d'un prompt à l'autre. Sur un run
  // mono-prompt il est constant : l'ordre y est exactement celui d'avant.
  const navImages = useMemo(
    () => flipOrder(displayedCells, (c) => [
      c.z_model_label || c.z_model || '', c.aspect || '', c.cfg ?? 0,
      c.steps ?? 0, c.steps2 ?? 0, c.prompt || '', c.label || '',
      c.seed ?? 0, c.strength ?? 0,
    ]),
    [displayedCells],
  );
  // On remonte le set ordonné À CÔTÉ de la cellule ouverte (le parent tient l'état
  // lightbox mais ne connaît pas ce tri — il vit ici avec displayedCells).
  const handleOpen = useCallback((cell) => onOpen(cell, navImages), [onOpen, navImages]);

  // Score cross-runs PAR CONFIG (modèle + cfg + steps inclus) — aligné backend.
  const scoreMap = useMemo(() => {
    const m = new Map();
    for (const s of d?.scores || []) {
      m.set(`${s.checkpoint}|${s.strength}|${s.aspect || ''}|${s.z_model || ''}|${s.cfg ?? ''}|${s.steps ?? ''}|${s.steps2 ?? ''}`, s);
    }
    return m;
  }, [d]);

  // Variantes présentes dans le run affiché (format × cfg × steps × PROMPT) → une
  // grille par variante. Le prompt est un axe comme les autres : un lot de N
  // prompts rend N tables, exactement comme un balayage de N CFG.
  const variantsInData = useMemo(() => {
    const m = new Map();
    for (const c of displayedCells) {
      const k = variantKey(c);
      if (!m.has(k)) m.set(k, variantOf(c));
    }
    // Aucun critère sur le prompt : les variantes d'un lot ne diffèrent QUE par
    // lui, elles se comparent donc toutes à égalité et le tri (stable) leur laisse
    // l'ordre d'insertion — c'est-à-dire l'ordre dans lequel l'utilisateur a coché
    // ses prompts, le seul qui lui parle. Un tri alphabétique le lui reprendrait.
    return [...m.values()].sort((a, b) =>
      (a.zModelLabel || '').localeCompare(b.zModelLabel || '')
      || a.aspect.localeCompare(b.aspect) || ((a.cfg ?? 0) - (b.cfg ?? 0))
      || ((a.steps ?? 0) - (b.steps ?? 0)) || ((a.steps2 ?? 0) - (b.steps2 ?? 0)));
  }, [displayedCells]);

  // Un run mono-prompt n'a rien à étiqueter — sa légende répéterait le même texte
  // sur chacune de ses tables. Le lot 📝, lui, ne se lit pas sans le prompt.
  const showPromptLabels = useMemo(() => distinctPrompts(displayedCells) > 1, [displayedCells]);

  const gridRows = useMemo(() => {
    const seen = new Map();
    for (const c of displayedCells) if (!seen.has(c.checkpoint)) seen.set(c.checkpoint, c.label);
    return [...seen.entries()].map(([filename, label]) => ({ filename, label }))
      .sort((a, b) => a.label.localeCompare(b.label, undefined, { numeric: true }));
  }, [displayedCells]);

  const gridCols = useMemo(() => {
    const set = new Set(displayedCells.map((c) => c.strength));
    return [...set].sort((a, b) => a - b);
  }, [displayedCells]);  // dépend des cellules affichées (pas de d) — sinon colonnes figées au changement de run

  // Run actif (objet) + axes présents pour la modale d'export.
  const activeRun = useMemo(() => runs.find((r) => r.key === activeRunKey) || null, [runs, activeRunKey]);
  const exportAspects = useMemo(
    () => [...new Set(displayedCells.map((c) => c.aspect).filter(Boolean))].sort(),
    [displayedCells]);
  const canExport = displayedCells.some((c) => c.status === 'done' && c.filename);

  // --- Mode vote rapide : enchaîne les images non votées (swipe / 👍 / 👎) ----
  const unvoted = displayedCells.filter((c) => c.status === 'done' && c.filename && !c.rating);
  // 2e passe : revoter UNIQUEMENT les 👍 pour resserrer (un 👎 les bascule rouge,
  // un 👍 les reconfirme, passer les laisse vertes).
  const greens = displayedCells.filter((c) => c.status === 'done' && c.filename && c.rating === 1);

  if (gridRows.length === 0) return null;

  return (
    <div className="flex flex-col gap-1">
      <RunSelector
        runs={runs}
        activeRunKey={activeRunKey}
        onSelect={(key) => setSelRun(key)}
        unvotedCount={unvoted.length}
        onStartVote={() => vote.startVoting(unvoted)}
        greenCount={greens.length}
        onStartReVote={() => vote.startVoting(greens, '♻️ Reconfirm the 👍')}
        displayedCount={displayedCells.length}
        showResults={showResults}
        onToggleResults={() => setShowResults((v) => !v)}
        canExport={canExport}
        onExport={() => setExportOpen(true)}
      />
      {showResults && (
        <ResultsGrid
          gridRows={gridRows}
          gridCols={gridCols}
          variantsInData={variantsInData}
          showPromptLabels={showPromptLabels}
          cellList={cellList}
          scoreMap={scoreMap}
          best={d.best_cell}
          datasetId={datasetId}
          onRate={studio.rate}
          onOpen={handleOpen}
          fmt={fmt}
        />
      )}
      <ExportGridModal
        open={exportOpen}
        onClose={() => setExportOpen(false)}
        datasetId={datasetId}
        family={d.family}
        run={activeRun}
        aspects={exportAspects}
        rows={gridRows.length}
        cols={gridCols.length}
      />
    </div>
  );
}
