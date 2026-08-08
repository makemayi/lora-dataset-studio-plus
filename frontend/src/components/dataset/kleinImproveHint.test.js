import test from 'node:test';
import assert from 'node:assert/strict';
import {
  readImproveInstruction, improveInstructionLine, improveAnimeCaution,
  shortenPrompt, IMPROVE_LINE_UNKNOWN, IMPROVE_LINE_OFF,
} from './kleinImproveHint.js';

const SHIPPED = 'add detailed texture, add sharp details, add candid shot, add soft focus effect';
const payload = (identity_prompts = {}) => ({
  config: { identity_prompts },
  identity_prompt_defaults: { klein_improve: SHIPPED },
});

test('with nothing saved, the effective instruction IS the shipped default', () => {
  const s = readImproveInstruction(payload());
  assert.deepEqual(s, { loaded: true, enabled: true, prompt: SHIPPED });
});

test('a non-blank override replaces it; a blank one falls back to the default', () => {
  assert.equal(readImproveInstruction(payload({ klein_improve: 'keep the line art' })).prompt,
    'keep the line art');
  // Blank-means-default is the backend contract (get_identity_prompt) — a hint
  // that showed '' here would tell an untouched install it sends no prompt.
  assert.equal(readImproveInstruction(payload({ klein_improve: '   ' })).prompt, SHIPPED);
  assert.equal(readImproveInstruction(payload({ klein_improve: '' })).prompt, SHIPPED);
});

test('the toggle off is read as off, and only `false` turns it off', () => {
  assert.equal(readImproveInstruction(payload({ klein_improve_enabled: false })).enabled, false);
  assert.equal(readImproveInstruction(payload({})).enabled, true);
  assert.equal(readImproveInstruction(payload({ klein_improve_enabled: true })).enabled, true);
});

test('no payload (still loading, or the request failed) never guesses', () => {
  for (const bad of [null, undefined, 'nope', 42]) {
    const s = readImproveInstruction(bad);
    assert.equal(s.loaded, false);
    assert.equal(improveInstructionLine(s).text, IMPROVE_LINE_UNKNOWN);
    assert.equal(improveInstructionLine(s).quote, null);
  }
});

test('the line QUOTES the instruction — that is the whole fix', () => {
  // Qeeyana could not connect "my anime turned realistic" to a setting she never
  // saw. Reading these words next to the button is the connection.
  const line = improveInstructionLine(readImproveInstruction(payload()));
  assert.match(line.quote, /detailed texture/);
  assert.match(line.quote, /sharp details/);
  assert.equal(line.full, SHIPPED);
  assert.equal(line.tone, 'quote');
});

test('toggle off says NO instruction is sent, and quotes nothing', () => {
  const line = improveInstructionLine(readImproveInstruction(payload({ klein_improve_enabled: false })));
  assert.equal(line.text, IMPROVE_LINE_OFF);
  assert.equal(line.quote, null);
  assert.equal(line.full, null);
});

test('an empty effective prompt reads as "no instruction", not as an empty quote', () => {
  const line = improveInstructionLine({ loaded: true, enabled: true, prompt: '   ' });
  assert.equal(line.text, IMPROVE_LINE_OFF);
  assert.equal(line.quote, null);
});

test('a long instruction is shortened for display but kept whole for the tooltip', () => {
  const long = Array.from({ length: 40 }, (_, i) => `token${i}`).join(', ');
  const line = improveInstructionLine({ loaded: true, enabled: true, prompt: long });
  assert.ok(line.quote.length < long.length);
  assert.ok(line.quote.endsWith('…'));
  assert.equal(line.full, long);           // the title attribute stays complete
  assert.ok(!/\s…$/.test(line.quote));     // no dangling space before the ellipsis
});

test('shortenPrompt collapses newlines so a multi-line prompt stays one line', () => {
  assert.equal(shortenPrompt('a\n\n  b\tc'), 'a b c');
});

test('the anime caution fires only for a drawn dataset with an instruction ON', () => {
  const on = readImproveInstruction(payload());
  assert.ok(improveAnimeCaution({ ...on, subjectType: 'anime' }));
  assert.ok(improveAnimeCaution({ ...on, subjectType: 'ANIME' }));   // case-insensitive
  for (const st of ['human', 'animal', 'creature', 'object', 'other', '', undefined]) {
    assert.equal(improveAnimeCaution({ ...on, subjectType: st }), null, String(st));
  }
  // Nothing is being sent -> nothing to warn about.
  assert.equal(improveAnimeCaution({ ...on, enabled: false, subjectType: 'anime' }), null);
  assert.equal(improveAnimeCaution({ loaded: false, subjectType: 'anime' }), null);
});

test('the caution CITES the subject type instead of asserting the images are drawn', () => {
  // A photoreal dataset left marked Anime used to be told "This dataset is drawn."
  // — a claim the app cannot make, since it never looked at the pixels. Quoting
  // the setting keeps the note true in that case AND points at the fix.
  const caution = improveAnimeCaution({
    ...readImproveInstruction(payload()), subjectType: 'anime' });
  assert.match(caution, /subject type is set to anime/i);
  assert.ok(!/this dataset is drawn/i.test(caution),
    'the note must not assert what the images are, only what the setting says');
  // The actual advice is unchanged — this was a rewording, not a removal.
  assert.match(caution, /detailed texture/);
  assert.match(caution, /sharp details/);
  assert.match(caution, /turn it off and upscale only/);
});
