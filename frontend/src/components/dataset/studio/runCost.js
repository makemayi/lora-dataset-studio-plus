/**
 * ⏱ Ce qu'un lancement va coûter, dit AVANT le clic — et rien de plus.
 *
 * POURQUOI CE MODULE EXISTE
 * -------------------------
 * Le lot de prompts a d'abord refusé au-delà de 24 prompts. Le premier usage
 * réel en a coché 33 et s'est fait renvoyer. Ce 24 était un jugement, pas une
 * mesure : rien ne casse à 33 (le corps pèse quelques kilo-octets contre 64 Mo
 * autorisés, la colonne `prompt` est un TEXT, la file n'a pas de profondeur
 * maximale, aucune vue de résultats ne tronque). Et c'était un plafond sur UN
 * axe parmi six — cocher 24 prompts sur 8 checkpoints passait, cocher 25 prompts
 * sur un seul ne passait pas, alors que le second run est trente fois plus court.
 *
 * La grandeur qui compte est donc le TOTAL de passes, et son coût est du temps.
 * Ce module le chiffre au rythme RÉEL de la machine (médiane observée par le
 * backend, `seconds_per_image`) et se contente d'avertir : la règle écrite de ce
 * projet est « PAS de plafond : la file est sérielle et l'utilisateur voit le
 * compte + l'estimation de durée avant de lancer ». Un lot de prompts obéit à
 * cette règle, il n'en invente pas une seconde.
 */

/** Repli quand la machine n'a pas encore assez d'historique — le chiffre que
 *  l'UI affichait en dur partout. Il reste un ordre de grandeur, pas une mesure,
 *  et `measured` dit lequel des deux on est en train de montrer. */
export const DEFAULT_SECONDS_PER_IMAGE = 12;

/** Au-delà de cette DURÉE estimée, le lancement demande confirmation. Un seuil
 *  en temps et non en nombre d'images : c'est la seule grandeur qui veuille dire
 *  la même chose sur une 4090 et sur une carte cinq fois plus lente — et comme
 *  il est calculé au rythme mesuré, la machine lente demande confirmation plus
 *  tôt, ce qui est exactement le comportement voulu. Une heure de GPU est un
 *  engagement qui vaut un clic ; en dessous, on informe sans interrompre. */
export const CONFIRM_ABOVE_SECONDS = 3600;

/** Une durée en secondes, dite comme un humain la dit. */
export function durationLabel(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  if (s < 60) return `${s} s`;
  const minutes = Math.round(s / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `${hours} h ${String(rest).padStart(2, '0')}` : `${hours} h`;
}

/**
 * Le coût d'un lancement de `cells` générations.
 * `secondsPerImage` = la médiane mesurée envoyée par le backend, ou null/0 quand
 * il n'y a pas encore assez d'historique — on retombe alors sur le défaut, et
 * `measured` le dit pour que l'UI n'écrive pas « à ton rythme actuel » sur une
 * constante inventée.
 */
export function runCost(cells, secondsPerImage) {
  const measured = Number(secondsPerImage) > 0;
  const per = measured ? Number(secondsPerImage) : DEFAULT_SECONDS_PER_IMAGE;
  const n = Math.max(0, Math.floor(Number(cells) || 0));
  const seconds = n * per;
  return {
    cells: n,
    secondsPerImage: per,
    measured,
    seconds,
    label: durationLabel(seconds),
    // `heavy` n'interdit rien : il colore une ligne et il fait poser UNE question.
    heavy: seconds > CONFIRM_ABOVE_SECONDS,
  };
}

/** La ligne montrée quand un lancement est long. Elle chiffre, elle ne gronde
 *  pas : c'est la machine de celui qui clique. */
export function heavyRunNotice(cost) {
  return `${cost.cells} generations, about ${cost.label}`
    + (cost.measured ? ' at your current pace' : '')
    + '. The queue is serial — you can stop it at any time and keep what is done.';
}

/** La question posée juste avant de lancer. Une seule, et seulement au-delà du
 *  seuil : un lot de trois prompts ne doit pas coûter une boîte de dialogue. */
export function heavyRunConfirm(cost) {
  return `This run will queue ${cost.cells} generations — about ${cost.label}`
    + (cost.measured ? ' at your current pace' : '')
    + '.\n\nThe queue is serial: you can stop it at any time, and the images already '
    + 'generated are kept.\n\nStart it?';
}
