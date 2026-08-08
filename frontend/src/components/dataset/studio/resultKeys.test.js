// react-frontend/src/components/dataset/studio/resultKeys.test.js
/**
 * 📝 Le lot de prompts vu par la GRILLE.
 *
 * Le symptôme rapporté — « quand je fais du multi-prompt sur le studio, cela me
 * montre une grille d'une seule image » — n'était pas une génération manquante :
 * les N images existaient, terminées, dans le payload. C'est la clé de RUN qui
 * portait le prompt, donc un lancement de N prompts se présentait comme N runs
 * différents, et la vue n'en affiche qu'un.
 *
 * Ces tests tiennent les deux moitiés de la correction, et surtout le contrat
 * qui empêche la rechute : le prompt ne peut plus disparaître d'une clé sans
 * qu'un test rougisse.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  runKey, variantKey, variantOf, cellKey, cellKeyFor, promptLabel, distinctPrompts,
} from './resultKeys.js';

/** Une cellule de lot : tout est partagé sauf le prompt — c'est le run que
 *  `create_run` écrit quand N cartes de prompt sont cochées (même seed, mêmes
 *  checkpoints, mêmes réglages, un `run_id` unique). */
const batchCell = (prompt, over = {}) => ({
  id: 1, run_id: 'RUN-A', run_seed: 424242, seed: 424242,
  checkpoint: 'z image\\char-1000.safetensors', strength: 1.0,
  z_model: 'z image\\base.safetensors', aspect: '1:1', cfg: 1.0, steps: 8, steps2: null,
  status: 'done', filename: 'a.png', prompt, ...over,
});

const PROMPTS = ['on a rooftop', 'in a neon alley', 'in a sunlit cafe'];

// --- 1. Un lancement reste UN lancement --------------------------------------

test('N prompts launched together stay ONE run', () => {
  const cells = PROMPTS.map((p) => batchCell(p));
  assert.equal(new Set(cells.map(runKey)).size, 1,
    'a prompt batch must not be split into one pseudo-run per prompt');
});

test('two SEPARATE launches remain two runs, even at the same pinned seed', () => {
  // Le cas qui justifiait le prompt dans la clé : seed épinglé, deux lancements.
  // `run_id` les sépare désormais sans avoir besoin du prompt.
  const a = batchCell('on a rooftop', { run_id: 'RUN-A' });
  const b = batchCell('on a rooftop', { run_id: 'RUN-B' });
  assert.notEqual(runKey(a), runKey(b));
});

test('a run predating run_id keeps EXACTLY its old identity', () => {
  // Ancienne colonne absente ⇒ ancienne clé, prompt compris : deux lancements à
  // seed épinglé qui ne différaient que par le prompt ne doivent pas fusionner.
  const old = (prompt) => batchCell(prompt, { run_id: null });
  assert.notEqual(runKey(old('on a rooftop')), runKey(old('in a neon alley')));
  assert.equal(runKey(old('on a rooftop')), runKey(old('on a rooftop')));
  // …et un run ancien n'emprunte jamais la clé d'un run neuf.
  assert.notEqual(runKey(old('on a rooftop')), runKey(batchCell('on a rooftop')));
});

test('run_seed absent falls back to the cell seed, as it always did', () => {
  const c = batchCell('on a rooftop', { run_id: null, run_seed: null, seed: 77 });
  assert.equal(runKey(c), runKey(batchCell('on a rooftop',
    { run_id: null, run_seed: null, seed: 77 })));
  assert.notEqual(runKey(c), runKey(batchCell('on a rooftop',
    { run_id: null, run_seed: null, seed: 78 })));
});

// --- 2. …mais chaque prompt garde sa propre place ----------------------------

test('N prompts of one run occupy N distinct cells', () => {
  const cells = PROMPTS.map((p) => batchCell(p));
  assert.equal(new Set(cells.map(cellKey)).size, PROMPTS.length,
    'prompts sharing a checkpoint/strength/seed must not collapse into one cell');
  assert.equal(new Set(cells.map(variantKey)).size, PROMPTS.length,
    'each prompt earns its own grid, like each format and each CFG');
});

test('the prompt is the ONLY thing separating them — every other axis is shared', () => {
  const cells = PROMPTS.map((p) => batchCell(p));
  for (const key of ['checkpoint', 'strength', 'z_model', 'aspect', 'cfg', 'steps']) {
    assert.equal(new Set(cells.map((c) => c[key])).size, 1);
  }
});

test('a single-prompt run is unchanged: one variant, one cell', () => {
  const cells = [batchCell('on a rooftop'), batchCell('on a rooftop', { seed: 999 })];
  assert.equal(new Set(cells.map(variantKey)).size, 1);
  assert.equal(new Set(cells.map(cellKey)).size, 1, 'the seeds of a batch share one cell');
});

test('the existing axes still separate cells on their own', () => {
  const base = batchCell('on a rooftop');
  for (const over of [{ aspect: '9:16' }, { cfg: 2.5 }, { steps: 20 },
                      { steps2: 12 }, { z_model: 'other.safetensors' },
                      { strength: 0.8 }, { checkpoint: 'z image\\char-2000.safetensors' }]) {
    assert.notEqual(cellKey(base), cellKey({ ...base, ...over }),
      `${Object.keys(over)[0]} must keep separating cells`);
  }
});

// --- 3. Le contrat : les deux côtés de la clé ne peuvent plus diverger --------

test('CONTRACT: the index and the grid lookup build the same cell key', () => {
  // L'index est construit depuis les cellules ; `ResultCell` cherche depuis
  // ligne × colonne × variante. Un axe ajouté d'un seul côté ferait afficher
  // « — » à la place des images, sans erreur. Ce test l'interdit.
  for (const prompt of PROMPTS) {
    const c = batchCell(prompt);
    assert.equal(cellKeyFor(c.checkpoint, c.strength, variantOf(c)), cellKey(c));
  }
});

test('CONTRACT: the prompt cannot silently leave the variant or the cell key', () => {
  // Formulé sur le RÉSULTAT, pas sur la forme de la clé : si quelqu'un retire le
  // prompt d'une de ces deux clés, ceci rougit, quelle que soit la réécriture.
  const a = batchCell('on a rooftop');
  const b = batchCell('on a rooftop, at night');
  assert.notEqual(variantKey(a), variantKey(b));
  assert.notEqual(cellKey(a), cellKey(b));
  assert.equal(runKey(a), runKey(b), 'the prompt must NOT identify the launch');
});

test('a key part containing the separator cannot merge two prompts', () => {
  // Un prompt est du texte libre : `a|b` et `a`+`b` ne doivent pas se confondre.
  assert.notEqual(cellKey(batchCell('a|b')), cellKey(batchCell('a')));
  assert.notEqual(variantKey(batchCell('x"|"y')), variantKey(batchCell('x')));
});

// --- 4. L'étiquette ----------------------------------------------------------

test('a long prompt is truncated to a legend that fits, with no word cut mid-air', () => {
  const long = 'a portrait on a rooftop at golden hour, cinematic lighting, 85mm lens, '
    + 'shallow depth of field, film grain'.repeat(3);
  const label = promptLabel(long);
  assert.ok(label.length <= 48, `legend too long: ${label.length}`);
  assert.ok(label.endsWith('…'), 'a cut prompt must say it is cut');
  assert.ok(long.startsWith(label.slice(0, -1).trimEnd()));
});

test('a short prompt is left alone, and whitespace is normalised', () => {
  assert.equal(promptLabel('on a rooftop'), 'on a rooftop');
  assert.equal(promptLabel('  on   a \n rooftop  '), 'on a rooftop');
  assert.equal(promptLabel(null), '');
  assert.equal(promptLabel(undefined), '');
});

test('distinctPrompts decides whether the view has anything to name', () => {
  assert.equal(distinctPrompts(PROMPTS.map((p) => batchCell(p))), 3);
  assert.equal(distinctPrompts([batchCell('one'), batchCell('one')]), 1);
  assert.equal(distinctPrompts([]), 0);
});
