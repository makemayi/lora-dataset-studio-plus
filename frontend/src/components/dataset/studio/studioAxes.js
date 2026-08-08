/**
 * 🎛 Axes de rendu (CFG / steps / 2e passe) de la branche COMPARAISON du Studio.
 *
 * Le studio mono-LoRA et le panneau du canvas les tiennent de `useStudioForm`,
 * qui lit le payload d'UN dataset. La comparaison multi-LoRA n'a pas de dataset —
 * elle n'avait donc aucun de ces axes, et « je ne peux pas régler les steps quand
 * j'ai deux LoRA » était exactement cela : pas un champ désactivé, un champ
 * absent, et un corps de requête qui ne portait pas la clé. Les échelles arrivent
 * maintenant avec les bases (`/api/studio/base-models` → `axes`) et ces trois
 * fonctions en font le même « effectif ou défaut » que partout ailleurs.
 */

/** Un axe tel qu'il sera lancé : la sélection de l'utilisateur si elle existe,
 *  sinon la valeur par défaut de la famille, seule. `null` par défaut ⇒ axe vide
 *  (la 2e passe hors SDXL) : rien n'est envoyé et le backend garde son défaut. */
export function effectiveAxis(selected, fallback) {
  if (Array.isArray(selected) && selected.length) return selected;
  return fallback == null ? [] : [fallback];
}

/** Le facteur par lequel ces axes multiplient la grille. Un axe vide compte
 *  pour 1 : il ne balaye rien, il ne doit pas annuler le compte. */
export function axisTotal({ cfgs, steps, steps2 } = {}) {
  const n = (a) => Math.max(1, (Array.isArray(a) ? a.length : 0));
  return n(cfgs) * n(steps) * n(steps2);
}

/** Ce que le lancement ajoute au corps. Une clé n'apparaît que si l'axe a des
 *  valeurs : un axe absent laisse le backend sur SON défaut, ce qui est le
 *  comportement d'avant pour toute install qui ne touche à rien. */
export function axisPayload({ cfgs, steps, steps2 } = {}) {
  const out = {};
  if (Array.isArray(cfgs) && cfgs.length) out.cfgs = [...cfgs];
  if (Array.isArray(steps) && steps.length) out.steps = [...steps];
  if (Array.isArray(steps2) && steps2.length) out.steps2 = [...steps2];
  return out;
}

/** Toggle multi-sélection qui garde toujours au moins une valeur — même règle
 *  que les pickers du studio mono-LoRA (`_toggleKeep` de useStudioForm). */
export function toggleAxisValue(current, value) {
  const base = Array.isArray(current) ? current : [];
  const next = base.includes(value)
    ? base.filter((v) => v !== value)
    : [...base, value].sort((a, b) => a - b);
  return next.length ? next : base;
}
