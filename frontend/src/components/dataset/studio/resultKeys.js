// react-frontend/src/components/dataset/studio/resultKeys.js
/**
 * Les CLÉS de la vue résultats du Studio : ce qui fait « un lancement », ce qui
 * fait « une variante », ce qui fait « une case ». Logique PURE extraite du JSX
 * pour être testable sous `node --test` (comme ./flipOrder.js et
 * ./stackResults.js) — parce qu'une clé fausse ne casse rien de visible : elle
 * fait DISPARAÎTRE des images, en silence.
 *
 * C'est exactement ce qui est arrivé au lot de prompts 📝 : le run était
 * identifié par `run_seed | prompt`, donc UN lancement de N prompts se coupait
 * en N pseudo-runs dont la vue n'en montrait qu'un — « une grille avec une
 * seule image ». Deux règles en sont sorties, et les tests de ce module les
 * tiennent :
 *
 *   1. le PROMPT n'identifie plus un lancement — `run_id` le fait (le backend
 *      pose un id opaque par invocation, cf. `create_run`) ;
 *   2. le PROMPT identifie une VARIANTE et une CASE — au même titre que le
 *      format, la CFG ou les steps. Sans cela, les N prompts d'un run
 *      retomberaient tous dans la même case, empilés sans étiquette.
 *
 * La case est lue de DEUX côtés (l'index construit depuis les cellules, et la
 * recherche faite par `ResultCell` depuis ligne × colonne × variante) : les deux
 * passent par `cellKey`, pour qu'aucun axe ne puisse être ajouté d'un côté et
 * oublié de l'autre.
 */

/** Une partie de clé : `null`/`undefined` valent la chaîne vide, comme les
 *  `?? ''` que ces clés utilisaient quand elles étaient écrites à la main. */
const part = (v) => (v == null ? '' : String(v));

/**
 * Assemblage d'une clé. JSON plutôt qu'un `join('|')` : un prompt est du texte
 * LIBRE, il contient des `|` quand ça lui chante, et deux prompts qui ne
 * diffèrent qu'autour d'un séparateur partageraient alors la même case.
 */
const join = (parts) => JSON.stringify(parts.map(part));

/**
 * Identité d'un LANCEMENT. `run_id` est l'id opaque que le backend pose sur
 * chaque cellule d'une invocation — la seule frontière juste, y compris quand un
 * run porte plusieurs prompts, plusieurs bases et plusieurs seeds.
 *
 * Les cellules ANTÉRIEURES à cette colonne n'en ont pas : elles gardent alors
 * strictement l'ancienne clé (`run_seed ?? seed`, plus le prompt). Le prompt y
 * reste nécessaire — deux lancements distincts à seed ÉPINGLÉ partagent leur
 * run_seed, et seul le prompt les séparait. Un vieux run s'affiche donc
 * exactement comme hier ; un run neuf, lui, ne se coupe plus en morceaux.
 */
export function runKey(cell) {
  const runId = cell?.run_id;
  if (runId) return `id:${runId}`;
  const runSeed = cell?.run_seed ?? cell?.seed;
  return `seed:${join([runSeed, cell?.prompt])}`;
}

/**
 * Identité d'une VARIANTE = une grille. Les axes de rendu du run, prompt
 * compris : chaque prompt reçoit sa propre table, comme chaque format et chaque
 * CFG en reçoivent une.
 */
export function variantKey(cell) {
  return join([cell?.z_model, cell?.aspect, cell?.cfg, cell?.steps, cell?.steps2, cell?.prompt]);
}

/** Le descripteur de variante que consomme la grille (clé + valeurs d'axes). */
export function variantOf(cell) {
  return {
    key: variantKey(cell),
    zModel: cell?.z_model || '',
    zModelLabel: cell?.z_model_label || '',
    aspect: cell?.aspect || '',
    cfg: cell?.cfg,
    steps: cell?.steps,
    steps2: cell?.steps2,
    prompt: cell?.prompt || '',
  };
}

/**
 * Identité d'une CASE = checkpoint × strength × variante. Une seule fonction,
 * deux appelants : l'index (`cellKey(cell)`) et la recherche de `ResultCell`
 * (`cellKey({ checkpoint, strength, ...variant })`). Les deux formes donnent la
 * même chaîne parce qu'elles empruntent le même chemin.
 */
export function cellKey(cell) {
  return join([cell?.checkpoint, cell?.strength,
               cell?.z_model, cell?.aspect, cell?.cfg, cell?.steps, cell?.steps2, cell?.prompt]);
}

/** La case cherchée par `ResultCell`, depuis sa ligne, sa colonne et sa variante. */
export function cellKeyFor(checkpoint, strength, variant) {
  return cellKey({
    checkpoint,
    strength,
    z_model: variant?.zModel,
    aspect: variant?.aspect,
    cfg: variant?.cfg,
    steps: variant?.steps,
    steps2: variant?.steps2,
    prompt: variant?.prompt,
  });
}

/**
 * Un prompt RÉDUIT à une étiquette : les prompts de test font couramment
 * plusieurs centaines de caractères, et une légende de grille doit tenir sur une
 * ligne — y compris à 400 px de large. Le texte entier reste dans le `title`.
 */
export function promptLabel(prompt, max = 48) {
  const text = String(prompt ?? '').trim().replace(/\s+/g, ' ');
  if (text.length <= max) return text;
  return `${text.slice(0, max - 1).trimEnd()}…`;
}

/** Combien de prompts DISTINCTS dans ce lot — ce qui décide si la vue doit les
 *  nommer (un run mono-prompt n'a rien à étiqueter, et la légende resterait
 *  identique sur toutes ses tables). */
export function distinctPrompts(cells) {
  return new Set((cells || []).map((c) => c?.prompt || '')).size;
}
