import test from 'node:test';
import assert from 'node:assert/strict';

import {
  summarizeGeneration, refusalHeadline, failureHeadline,
  FAIL_REFUSED, FAIL_EMPTY, FAIL_ERROR,
} from './generationOutcome.js';

const made = (id) => ({ id, status: 'keep', filename: `img${id}.webp` });
const failed = (id, kind, engine) => (
  { id, status: 'failed', fail_kind: kind, filename: null, engine });

test('the notice can name which engine refused, without inventing one', () => {
  // The policy being described belongs to a provider. Attributing it to "the
  // engines" in general would misdescribe a run that also used a local one.
  const c = summarizeGeneration([
    failed(1, FAIL_REFUSED, 'nanobanana'), failed(2, FAIL_REFUSED, 'nanobanana'),
    failed(3, FAIL_ERROR, 'chatgpt'), made(4),
  ]);
  assert.deepEqual(c.refusedEngines, ['nanobanana']);
  // An error from another engine does not put that engine in the refused list.
  assert.equal(c.refusedEngines.includes('chatgpt'), false);
  // Rows with no engine recorded add nothing rather than a blank entry.
  assert.deepEqual(summarizeGeneration([failed(1, FAIL_REFUSED)]).refusedEngines, []);
});

test('a batch with refusals is counted exactly, not approximated', () => {
  // The scenario this module exists for: 40 asked, 12 refused by the provider.
  const images = [];
  for (let i = 0; i < 40; i += 1) {
    images.push(i % 10 === 3 || i % 10 === 7 || i % 10 === 8
      ? failed(i, FAIL_REFUSED) : made(i));
  }
  const c = summarizeGeneration(images);
  assert.equal(c.total, 40);
  assert.equal(c.refused, 12);
  assert.equal(c.made, 28);
  assert.equal(c.failed, 12);
  assert.equal(c.refused + c.made, 40, 'no image fell in a silent hole');
});

test('a refusal and a malfunction are never merged into one number', () => {
  const c = summarizeGeneration([
    made(1), failed(2, FAIL_REFUSED), failed(3, FAIL_ERROR), failed(4, FAIL_EMPTY),
  ]);
  assert.equal(c.refused, 1);
  assert.equal(c.errored, 1);
  assert.equal(c.empty, 1);
  assert.equal(c.failed, 3);
  // …and they are worded apart, too.
  assert.match(refusalHeadline(c), /refused by the provider's content filter/);
  assert.match(failureHeadline(c), /failed for a different reason/);
  assert.doesNotMatch(refusalHeadline(c), /connection|key|quota/);
  assert.doesNotMatch(failureHeadline(c), /content filter/);
});

test('rows the backend never classified are not folded into either bucket', () => {
  // Databases predating fail_kind keep their sentence and a null kind. Counting
  // them as refusals would invent a policy verdict; as errors, a malfunction.
  const c = summarizeGeneration([failed(1, null), failed(2, undefined), made(3)]);
  assert.equal(c.unclassified, 2);
  assert.equal(c.refused, 0);
  assert.equal(c.errored, 0);
  assert.equal(refusalHeadline(c), '');
  assert.equal(failureHeadline(c), '');
});

test('the refusal notice states the successes too, so a run does not read as broken', () => {
  const c = summarizeGeneration([made(1), made(2), failed(3, FAIL_REFUSED)]);
  const line = refusalHeadline(c);
  assert.match(line, /1 image in this dataset was refused/);
  assert.match(line, /2 generated normally/);
});

test('the notice never promises a workaround it cannot deliver', () => {
  // Measured: the same prompt passes roughly half the time, and the output
  // filter is not configurable. Any "try again" / "reword it" would be a guess.
  const c = summarizeGeneration([failed(1, FAIL_REFUSED), failed(2, FAIL_REFUSED)]);
  const line = refusalHeadline(c).toLowerCase();
  for (const forbidden of ['retry', 'try again', 'rephrase', 'reword', 'usually works']) {
    assert.ok(!line.includes(forbidden), `notice must not say "${forbidden}"`);
  }
});

test('nothing is said when nothing was refused', () => {
  assert.equal(refusalHeadline(summarizeGeneration([made(1), made(2)])), '');
  assert.equal(failureHeadline(summarizeGeneration([made(1)])), '');
});

test('singular and plural both read as English', () => {
  assert.match(refusalHeadline({ refused: 1, made: 0 }), /1 image .* was refused/);
  assert.match(refusalHeadline({ refused: 3, made: 0 }), /3 images .* were refused/);
  assert.match(failureHeadline({ errored: 1 }), /^1 image failed/);
  assert.match(failureHeadline({ errored: 2 }), /^2 images failed/);
});

test('junk input counts nothing instead of throwing', () => {
  for (const junk of [null, undefined, 'nope', 42, {}]) {
    assert.equal(summarizeGeneration(junk).total, 0);
  }
  assert.equal(summarizeGeneration([null, undefined, 'x', made(1)]).total, 1);
  assert.equal(refusalHeadline(null), '');
  assert.equal(failureHeadline(undefined), '');
});

test('a pending row is neither a success nor a failure', () => {
  const c = summarizeGeneration([{ id: 1, status: 'pending', filename: null }, made(2)]);
  assert.equal(c.total, 2);
  assert.equal(c.made, 1);
  assert.equal(c.failed, 0);
});
