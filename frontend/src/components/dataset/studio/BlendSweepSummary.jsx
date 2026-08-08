/**
 * Ce que le balayage 🧬 va coûter, dit AVANT de lancer : combien de
 * combinaisons, combien d'images, combien de minutes.
 *
 * Il n'y a pas de plafond dur, et c'est délibéré — `build_matrix` porte la même
 * règle écrite côté serveur : « PAS de plafond sur le nombre de cellules : la
 * file est sérielle et l'utilisateur voit le compte + l'estimation de durée
 * avant de lancer ». Un balayage de poids est un balayage comme l'axe des
 * strengths ; il obéit donc à la même règle plutôt qu'à une seconde. Au-delà de
 * BLEND_WARN_CELLS on chiffre en ambre, on n'interdit pas : c'est la machine de
 * celui qui clique.
 */
import { blendSweepCost } from './loraStack';

export default function BlendSweepSummary({ configCount, count = 1, batchMult = 1,
  secondsPerImage = null }) {
  const cost = blendSweepCost({ configCount, count, batchMult, secondsPerImage });
  if (cost.configs <= 1) return null;   // une seule configuration : rien à annoncer

  return (
    <p data-testid="blend-sweep-summary"
      className={'m-0 rounded-lg border px-2.5 py-1.5 text-[0.6875rem] '
        + (cost.warn
          ? 'border-amber-400/40 bg-amber-500/10 text-amber-200'
          : 'border-border bg-surface text-content-muted')}
      role={cost.warn ? 'status' : undefined}>
      <span aria-hidden>{cost.warn ? '⚠' : '🧮'}</span>{' '}
      <strong className="tabular-nums">{cost.configs}</strong> weight combinations
      {' → '}<strong className="tabular-nums">{cost.cells}</strong> image
      {cost.cells > 1 ? 's' : ''}, about {cost.label}
      {cost.measured ? ' at your current pace' : ''}.
      {cost.warn && ' That is a long queue — untick a few weights if you did not mean it.'}
    </p>
  );
}
