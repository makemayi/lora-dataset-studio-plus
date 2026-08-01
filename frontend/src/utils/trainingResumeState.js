/** Truthful resume-mode decisions shared by every ContinueDialog host. */

export function preferredCheckpointForStep(checkpoints, step) {
  const list = Array.isArray(checkpoints) ? checkpoints : [];
  const matches = list.filter((checkpoint) => checkpoint?.step === step);
  // Matches the backend: a numbered checkpoint wins a tie with the bare final.
  return matches.find((checkpoint) => !checkpoint.final) || matches[0] || null;
}

export function fullStateUnavailableReason(checkpoint, lane) {
  if (lane !== 'local') {
    return 'The current cloud image cannot restore LDS state bundles. Continue in the cloud uses weights only.';
  }
  const state = checkpoint?.resume_state;
  if (!state) {
    return 'Legacy checkpoint: no optimizer, scheduler, RNG or dataloader state was saved with these weights.';
  }
  if (!['ready', 'complete'].includes(state.status)) {
    return state.reason || 'The state bundle is not ready.';
  }
  if (state.integrity !== 'verified') {
    return state.reason || 'The state bundle has not passed its integrity check.';
  }
  if (state.state_level !== 'exact' || !state.bundle_id) {
    return state.reason || 'This checkpoint does not contain every artifact required for an exact resume.';
  }
  return null;
}

export function defaultResumeMode(checkpoint, lane) {
  return fullStateUnavailableReason(checkpoint, lane) ? 'weights_only' : 'full_state';
}
