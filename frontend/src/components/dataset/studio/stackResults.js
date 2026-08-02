// react-frontend/src/components/dataset/studio/stackResults.js
/**
 * Logique PURE de la VUE RÉSULTATS d'une pile 🧬 (mode combine du Test Studio) —
 * extraite du JSX pour être testable sous `node --test`, comme ./loraStack.js qui
 * porte, lui, la logique du LANCEMENT.
 *
 * Un run de pile ne produit qu'UNE colonne (le LoRA de tête) : la grille de
 * comparaison et le classement par-LoRA, conçus pour opposer des LoRA entre eux,
 * n'ont alors rien à dire. Ce que l'utilisateur veut savoir d'une pile, c'est :
 *   1. ce qu'il y a DEDANS (chaque LoRA, son poids, son trigger) ;
 *   2. ce que donnent d'AUTRES poids sur la même pile (les variantes, côte à côte) ;
 *   3. quel jeu de poids garder — le « best setting » d'une pile, ce sont ses poids,
 *      pas un checkpoint isolé.
 *
 * Le backend fournit `stack` (composition, null si ce n'est pas une pile) et
 * `stack_variants` (les relances de la MÊME pile, la courante marquée `active`).
 */

/** Composition de la pile du run affiché, [] si le run n'est pas une pile. */
export function stackMembers(data) {
  const members = data?.stack;
  return Array.isArray(members) ? members : [];
}

/** Le run affiché est-il une pile ? (≥2 LoRA dans la même image) */
export function isStackRun(data) {
  return stackMembers(data).length > 1;
}

/**
 * Poids numérique, ou NaN s'il est absent. `Number(null)` vaut 0 : sans ce garde, un
 * poids manquant (run ancien, cellule tronquée) s'afficherait comme un vrai 0.00 —
 * « ce LoRA était désactivé » au lieu de « on ne sait pas ».
 */
const numWeight = (weight) => (weight == null || weight === '' ? NaN : Number(weight));

/** Un poids affichable : 2 décimales, « — » quand le backend n'en a pas. */
export function fmtWeight(weight) {
  const n = numWeight(weight);
  return Number.isFinite(n) ? n.toFixed(2) : '—';
}

/**
 * Étiquette courte d'une variante : son VECTEUR de poids, dans l'ordre de la pile.
 * C'est ce qui distingue deux colonnes — les LoRA, eux, sont les mêmes partout.
 */
export function weightVectorText(weights) {
  return (weights || []).map((w) => fmtWeight(w?.weight)).join(' / ');
}

/**
 * Poids d'une variante alignés sur la composition de référence, avec le delta face
 * à la variante active. Aligné PAR FICHIER (et non par position) : deux relances de
 * la même pile peuvent avoir été lancées dans un ordre de sélection différent, et
 * comparer alors la ligne 1 de l'une avec la ligne 2 de l'autre mentirait.
 * Retourne une entrée par membre : { filename, label, weight, delta, changed }.
 */
export function alignWeights(members, variantWeights, activeWeights = null) {
  const byFile = new Map((variantWeights || []).map((w) => [w?.filename, w]));
  const activeByFile = new Map((activeWeights || []).map((w) => [w?.filename, w]));
  return (members || []).map((m) => {
    const here = numWeight(byFile.get(m?.filename)?.weight);
    const there = activeWeights ? numWeight(activeByFile.get(m?.filename)?.weight) : NaN;
    const comparable = Number.isFinite(here) && Number.isFinite(there);
    // Arrondi au centième avant comparaison : les poids sont réglés au pas de 0.05
    // et stockés arrondis, un delta de 1e-15 ne serait pas un changement.
    const delta = comparable ? Math.round((here - there) * 100) / 100 : 0;
    return {
      filename: m?.filename ?? null,
      label: m?.label ?? '',
      weight: Number.isFinite(here) ? here : null,
      delta,
      changed: comparable && delta !== 0,
    };
  });
}

/** Bilan d'une variante : combien d'images rendues, et le solde des votes. */
export function variantSummary(variant) {
  const cells = variant?.cells || [];
  const likes = variant?.likes ?? cells.filter((c) => c.rating === 1).length;
  const dislikes = variant?.dislikes ?? cells.filter((c) => c.rating === -1).length;
  const done = variant?.done ?? cells.filter((c) => c.status === 'done' && c.filename).length;
  return { likes, dislikes, net: likes - dislikes, done, total: cells.length };
}

/**
 * Poids d'une variante ramenés dans la map du panneau de lancement
 * (clé `${dataset_id}:${checkpoint}` — celle de loraStack.stackKey), pour rejouer la
 * pile à ces poids-là. Les dataset_id viennent de la COMPOSITION : une variante ne
 * transporte que des fichiers.
 */
export function weightsIntoStackMap(members, variantWeights) {
  const byFile = new Map((variantWeights || []).map((w) => [w?.filename, w]));
  const out = {};
  for (const m of members || []) {
    const w = numWeight(byFile.get(m?.filename)?.weight);
    if (m?.dataset_id != null && m?.filename && Number.isFinite(w)) {
      out[`${m.dataset_id}:${m.filename}`] = w;
    }
  }
  return out;
}

/**
 * Corps du POST « ★ best setting » pour une pile : le LoRA de tête dans
 * checkpoint/strength (ce que lisent déjà le pin du Canvas, ★ Appliquer et le
 * garde-fou de suppression), les autres dans `stack`. `null` si la composition est
 * incomplète — mieux vaut ne pas proposer le bouton que d'épingler une demi-pile.
 */
export function bestStackPayload(members) {
  const list = members || [];
  if (list.length < 2) return null;
  const [head, ...rest] = list;
  // `Number(null)` vaut 0 : un poids ABSENT doit être rejeté, pas pris pour un zéro.
  const incomplete = (m) => m?.dataset_id == null || !m?.filename
    || m?.weight == null || !Number.isFinite(Number(m.weight));
  if (incomplete(head) || rest.some(incomplete)) return null;
  return {
    dataset_id: head.dataset_id,
    checkpoint: head.filename,
    strength: Number(head.weight),
    stack: rest.map((m) => ({
      dataset_id: m.dataset_id,
      lora_filename: m.filename,
      weight: Number(m.weight),
    })),
  };
}
