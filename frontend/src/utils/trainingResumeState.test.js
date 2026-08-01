import assert from 'node:assert/strict';
import test from 'node:test';

import {
  defaultResumeMode,
  fullStateUnavailableReason,
  preferredCheckpointForStep,
} from './trainingResumeState.js';

const exact = {
  step: 500,
  resume_state: {
    bundle_id: '0123456789abcdef0123456789abcdef',
    status: 'ready',
    integrity: 'verified',
    state_level: 'exact',
  },
};

test('verified exact local bundle is the only full-state default', () => {
  assert.equal(defaultResumeMode(exact, 'local'), 'full_state');
  assert.equal(fullStateUnavailableReason(exact, 'local'), null);
  assert.equal(defaultResumeMode({
    ...exact, resume_state: { ...exact.resume_state, status: 'complete' },
  }, 'local'), 'full_state');
  assert.equal(defaultResumeMode(exact, 'cloud'), 'weights_only');
  assert.match(fullStateUnavailableReason(exact, 'cloud'), /cloud image/i);
});

test('legacy, corrupt and incomplete checkpoints fall back with a reason', () => {
  assert.equal(defaultResumeMode({ step: 1 }, 'local'), 'weights_only');
  assert.match(fullStateUnavailableReason({ step: 1 }, 'local'), /Legacy checkpoint/);

  const corrupt = {
    ...exact,
    resume_state: { ...exact.resume_state, status: 'invalid', integrity: 'failed',
      reason: 'Artifact hash mismatch.' },
  };
  assert.equal(defaultResumeMode(corrupt, 'local'), 'weights_only');
  assert.equal(fullStateUnavailableReason(corrupt, 'local'), 'Artifact hash mismatch.');

  const partial = {
    ...exact,
    resume_state: { ...exact.resume_state, state_level: 'weights_optimizer' },
  };
  assert.equal(defaultResumeMode(partial, 'local'), 'weights_only');
});

test('numbered checkpoint wins a same-step tie with the final save', () => {
  const final = { ...exact, final: true, resume_state: null };
  assert.equal(preferredCheckpointForStep([final, exact], 500), exact);
});
