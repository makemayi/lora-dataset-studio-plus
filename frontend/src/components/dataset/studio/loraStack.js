// react-frontend/src/components/dataset/studio/loraStack.js
/**
 * Logique PURE du mode « pile » (combine) du Test Studio — extraite du JSX pour
 * être testable sous `node --test` (le runner ne parse pas le JSX).
 *
 * Deux modes, une seule sélection de LoRA :
 *   - 'compare' : chaque LoRA coché est testé SEUL, une colonne par LoRA
 *                 (comportement historique) ; l'axe strengths balaye chacun.
 *   - 'combine' : les LoRA cochés sont chargés ENSEMBLE dans la même génération,
 *                 chacun à SON poids ; l'axe strengths n'a plus de sens et
 *                 disparaît de l'UI comme du payload.
 *
 * Les poids vivent hors de la sélection (le LoraPicker n'en connaît pas) : une map
 * `{ "<dataset_id>:<checkpoint>": poids }`, pour qu'un poids réglé survive au
 * décochage/recochage d'un autre LoRA et à un changement de checkpoint.
 */

export const COMBINE_MIN_WEIGHT = 0;
export const COMBINE_MAX_WEIGHT = 2;
export const COMBINE_DEFAULT_WEIGHT = 1;

/** Clé stable d'un LoRA sélectionné dans la map de poids. */
export const stackKey = (sel) => `${sel?.dataset_id}:${sel?.checkpoint}`;

/** Poids retenu pour une sélection : clampé 0..2, arrondi au centième, 1 par défaut. */
export function stackWeight(weights, sel) {
  const raw = Number((weights || {})[stackKey(sel)]);
  if (!Number.isFinite(raw)) return COMBINE_DEFAULT_WEIGHT;
  return Math.round(Math.min(COMBINE_MAX_WEIGHT, Math.max(COMBINE_MIN_WEIGHT, raw)) * 100) / 100;
}

/**
 * Le mode combine exige ≥2 LoRA d'une MÊME famille. Retourne null si tout va bien,
 * sinon le message (anglais) à afficher — le backend refuse aussi, mais l'utilisateur
 * doit le savoir AVANT de dépenser du GPU.
 */
export function combineBlocker(selection) {
  const sel = selection || [];
  if (sel.length < 2) return 'Check at least two LoRAs to combine them.';
  const families = [...new Set(sel.map((s) => s.family || s.train_type || 'zimage'))];
  if (families.length > 1) {
    return `Combining needs one family: ${families.join(' + ')} use different base `
      + 'models and workflows. Uncheck one of them.';
  }
  return null;
}

/**
 * Corps `selections` du POST /api/studio/run. En combine, chaque entrée porte son
 * `weight` (le backend chaîne les LoRA au-delà du premier dans le même graphe et
 * injecte TOUS les triggers) ; en comparaison le payload reste celui d'avant, sans
 * `weight`, pour ne rien changer aux runs existants.
 */
export function buildSelectionsPayload(selection, { combine = false, weights = {} } = {}) {
  return (selection || []).map((s) => (combine
    ? { dataset_id: s.dataset_id, checkpoint: s.checkpoint, weight: stackWeight(weights, s) }
    : { dataset_id: s.dataset_id, checkpoint: s.checkpoint }));
}

/**
 * Nombre de cellules annoncé AVANT lancement. Une pile combinée = UNE configuration
 * (l'axe strengths est remplacé par les poids), d'où `1 × count × batchMult` au lieu
 * de `nbLoRA × nbStrengths × count × batchMult`.
 */
export function cellCount({ selectionCount, strengthCount, count, batchMult = 1, combine = false }) {
  const n = Math.max(0, Number(count) || 0);
  const mult = Math.max(1, Number(batchMult) || 1);
  if (combine) return selectionCount >= 2 ? n * mult : 0;
  return Math.max(0, selectionCount) * Math.max(0, strengthCount) * n * mult;
}
