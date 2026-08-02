export const TRAINING_MODE_LORA = 'lora';
export const TRAINING_MODE_FULL_TRANSFORMER = 'full_transformer';

export function normalizeTrainingMode(value) {
  return value === TRAINING_MODE_FULL_TRANSFORMER
    ? TRAINING_MODE_FULL_TRANSFORMER
    : TRAINING_MODE_LORA;
}

export function trainingModeLabel(value) {
  return normalizeTrainingMode(value) === TRAINING_MODE_FULL_TRANSFORMER
    ? 'Full model'
    : 'LoRA';
}

export function isFullTransformerRun(run) {
  return normalizeTrainingMode(run?.training_mode) === TRAINING_MODE_FULL_TRANSFORMER;
}

/** Payload used by the atomic recipe-settings endpoint. Keep the official base
 * explicit (`base_model: ''`): omitting it would ask the server to reuse an old
 * custom base, which is not the dense Krea Raw recipe selected in the UI. */
export function trainingModeSettingsPayload(trainingMode, selection = {}) {
  const payload = { training_mode: normalizeTrainingMode(trainingMode) };
  if (selection.trainType !== undefined) payload.train_type = selection.trainType;
  if (Object.prototype.hasOwnProperty.call(selection, 'baseModel')) {
    payload.base_model = selection.baseModel == null ? '' : String(selection.baseModel);
  }
  if (selection.variant !== undefined) payload.variant = selection.variant;
  if (selection.disableSliderForFullTransformer === true) {
    payload.disable_slider_for_full_transformer = true;
  }
  return payload;
}

/** Normalize the two backend surfaces that can report whether a dense run may
 * use the dedicated Hugging Face delivery token. Missing metadata is not a
 * refusal (older servers did not expose it); an explicit failed check/status is. */
export function hfCloudTokenReadiness(payload = {}) {
  const check = Array.isArray(payload?.checks)
    ? payload.checks.find((item) => item?.id === 'hf_cloud_token')
    : null;
  const offerStatus = payload?.hf_cloud_token || null;
  const status = payload?.hf_cloud_token_status
    || payload?.hf_token_status
    || offerStatus
    || null;
  const combinedText = [...new Set([
    check?.detail,
    check?.hint,
    status?.error,
    status?.detail,
    payload?.error,
    payload?.hint,
  ].filter(Boolean).map(String))].join(' — ');
  const textSignalsTokenFailure = /HF_CLOUD_TOKEN|hugging\s*face[^\n]*token|token[^\n]*(scope|permission)/i
    .test(combinedText);
  const checkFailed = String(check?.status || '').toLowerCase() === 'fail';
  const statusFailed = status && (
    status.ok === false
    || status.configured === false
    || status.valid === false
    || status.ready === false
  );
  const offerStatusFailed = offerStatus && offerStatus.ok !== true;
  const signaled = !!check || !!status || textSignalsTokenFailure;
  const blocked = checkFailed || !!statusFailed || !!offerStatusFailed
    || (!check && !status && textSignalsTokenFailure);
  let detail = combinedText;
  if (!detail && blocked) {
    detail = status?.configured === false
      ? 'The dedicated HF_CLOUD_TOKEN is missing.'
      : 'The dedicated HF_CLOUD_TOKEN is invalid or does not have the required permissions.';
  }
  return {
    signaled,
    ready: !blocked,
    blocked,
    detail: detail || null,
  };
}

/** A full model is useful only after the backend has verified the Hub contents.
 * The model CTA stays gated by `artifact_status`; `hf_url` alone may expose only
 * a clearly labelled repository-inspection link while delivery is unverified. */
export function fullTransformerArtifactView(run = {}) {
  const status = String(run.artifact_status || '').trim().toLowerCase();
  const detail = String(run.artifact_status_detail ?? run.artifact_detail ?? '').trim();
  const available = status === 'available';
  const cleanupStatus = String(run.artifact_cleanup_status || '').trim().toLowerCase();
  // Older backend rows predate artifact_cleanup_status.  A kept pod with a
  // verified model is therefore pending by default unless cleanup is explicitly
  // complete; silence here could otherwise hide continued billing.
  const cleanupPending = available && run.status === 'error_pod_kept'
    && cleanupStatus !== 'complete';
  const cleanupDetail = String(run.artifact_cleanup_detail || '').trim();
  const rawRepositoryHref = String(run.hf_url || '').trim();
  const repositoryHref = /^https:\/\/huggingface\.co\//i.test(rawRepositoryHref)
    ? rawRepositoryHref
    : null;
  const href = available ? repositoryHref : null;

  if (available) {
    return {
      status, available, cleanupPending, href, repositoryHref,
      tone: cleanupPending ? 'warning' : 'success',
      label: 'Full model available',
      detail: cleanupPending
        ? (cleanupDetail
          || 'The model is verified, but pod cleanup has not been confirmed and the pod may still be billing.')
        : detail || (href
        ? 'The private Hugging Face repository contents have been verified.'
        : 'The contents were verified, but this status does not include the repository link.'),
    };
  }
  if (status === 'missing') {
    return {
      status, available: false, href: null, repositoryHref, tone: 'error',
      label: 'Full model not found',
      detail: detail || 'No full-model weights were verified in the repository. Check the run logs and Hugging Face repository before deleting any recovery copy.',
    };
  }
  if (status === 'verification_pending') {
    return {
      status, available: false, href: null, repositoryHref, tone: 'warning',
      label: 'Hugging Face verification pending',
      detail: detail || 'Check the dedicated HF_CLOUD_TOKEN in Settings ▸ Local tools and your connection, then refresh Runs. Do not treat the model as recoverable yet.',
    };
  }
  if (status === 'creating_repository' || status === 'pending' || status === 'uploading') {
    // 'pending' is stamped at LAUNCH and covers the whole run, so on its own it
    // cannot say whether weights are moving. Announcing 'Uploading full
    // model…' from it claimed a transfer that had not been started and could
    // not be: for the two hours run #138 spent pushing its DATASET to the pod,
    // this panel described the model going up to Hugging Face, next to a link
    // offering to inspect a repository holding nothing but licence files. The
    // run's own phase is what distinguishes them, and it is already here.
    const runStatus = String(run.status || '');
    const beforeTraining = ['preparing', 'provisioning', 'uploading'].includes(runStatus);
    const training = runStatus === 'training';
    // Delivery is the very end of a run, so anything that is no longer running
    // and still reads 'pending' never got there. Saying 'Uploading full model…'
    // on a terminated run is the worst version of this: it also tells the user
    // to keep a pod alive that the supervisor already destroyed.
    // An ABSENT status is not a finished run (an older payload, a caller that
    // does not carry one): claiming a delivery never happened is a statement,
    // and it is only made about a run whose phase actually says so.
    const ended = !!runStatus && !['preparing', 'provisioning', 'uploading',
      'training', 'downloading', 'terminating'].includes(runStatus);
    let label = 'Uploading full model…';
    let fallbackDetail = 'Keep the run and pod active until the repository is verified.';
    if (status === 'creating_repository') {
      label = 'Creating Hugging Face repository…';
    } else if (beforeTraining) {
      label = 'Full model not created yet';
      fallbackDetail = 'The run is still starting up — the weights are created on Hugging '
        + 'Face once training produces them. Nothing is uploading to Hugging Face yet.';
    } else if (training) {
      label = 'Full model not delivered yet';
      fallbackDetail = 'Training is running. The weights are delivered to Hugging Face at '
        + 'the end of the run — keep the run and pod active until then.';
    } else if (ended) {
      label = 'Full model was never delivered';
      fallbackDetail = 'The run ended before any weights reached Hugging Face, so the '
        + 'repository holds only the licence and model card. Check the run error above.';
    }
    return {
      status, available: false, href: null, repositoryHref,
      // A run that ended empty-handed is not neutral information.
      tone: ended && status !== 'creating_repository' ? 'warning' : 'info',
      label,
      detail: detail || fallbackDetail,
    };
  }
  return {
    status, available: false, href: null, repositoryHref, tone: 'warning',
    label: 'Full model status unavailable',
    detail: detail || 'Refresh Runs. If the status remains unavailable, check the run logs and your Hugging Face configuration.',
  };
}

/** Delivery verification is safe only for the recovery state whose pod was
 * deliberately kept alive. Rechecking a live/finished run could otherwise race
 * the monitor and tear down an instance that is still uploading. */
export function canRecheckFullTransformerDelivery(run = {}) {
  const artifactStatus = String(run.artifact_status || '').trim().toLowerCase();
  const cleanupPending = artifactStatus === 'available'
    && String(run.artifact_cleanup_status || '').trim().toLowerCase() !== 'complete';
  return isFullTransformerRun(run)
    && run.status === 'error_pod_kept'
    && (artifactStatus !== 'available' || cleanupPending);
}

/** Turn the transactional backend result into billing-safe user feedback. */
export function fullTransformerRecheckOutcome(result = {}) {
  if (!result?.ok) {
    return {
      kind: 'error',
      text: result?.error
        || 'Hugging Face delivery could not be verified. The pod remains available for recovery.',
    };
  }
  if (result.delivery === 'available' && result.cleanup_pending) {
    return {
      kind: 'warning',
      text: 'Hugging Face model verified and available. Pod cleanup is still pending, and the pod may still be billing; retry cleanup.',
    };
  }
  if (result.delivery === 'available') {
    return {
      kind: 'success',
      text: 'Hugging Face delivery verified. The model is available and pod cleanup is confirmed.',
    };
  }
  return {
    kind: 'info',
    text: result.delivery === 'missing'
      ? 'No full-model weights were verified in the repository. The pod remains available for recovery; check its logs before deleting anything.'
      : 'Hugging Face verification is still pending. Fix HF_CLOUD_TOKEN if needed, then try again.',
  };
}

/** Dense estimates must be explicitly backed by a dense benchmark. Older
 * servers can still return LoRA-derived numbers without an estimate status; for
 * a full run those numbers are deliberately treated as unavailable. */
export function cloudTierEstimateView(tier = {}, { fullMode = false } = {}) {
  const status = tier.estimate_status == null
    ? null
    : String(tier.estimate_status).trim().toLowerCase();
  const explicitlyAvailable = ['available', 'estimated', 'ok'].includes(status);
  const explicitlyUnavailable = status === 'unavailable' || status === 'pending';
  const minutes = tier.est_minutes == null || tier.est_minutes === ''
    ? Number.NaN
    : Number(tier.est_minutes);
  const available = Number.isFinite(minutes)
    && minutes >= 0
    && !explicitlyUnavailable
    && (!fullMode || explicitlyAvailable);
  const rawCost = tier.est_cost == null || tier.est_cost === ''
    ? Number.NaN
    : Number(tier.est_cost);
  return {
    available,
    minutes: available ? minutes : null,
    cost: available && Number.isFinite(rawCost) && rawCost >= 0 ? rawCost : null,
    exceedsCap: available && tier.exceeds_cap === true,
    status,
  };
}

/** Dense fine-tuning is deliberately a single, narrow cloud recipe for the MVP:
 * the official Krea 2 Raw base. A local/custom base is not equivalent, even when
 * its architecture happens to be Krea-compatible. */
export function isFullTransformerEligible({
  trainType, variant, baseModel = '', customBase = false,
} = {}) {
  return !customBase
    && trainType === 'krea'
    && variant === 'base'
    && String(baseModel || '').trim() === '';
}

export function fullTransformerUnavailableReason(selection = {}) {
  if (selection.trainType !== 'krea') return 'Choose the Krea 2 family.';
  if (selection.variant !== 'base') return 'Choose Krea 2 Raw.';
  if (selection.customBase === true) return 'This MVP supports only the official Krea 2 Raw base.';
  if (String(selection.baseModel || '').trim()) return 'This MVP supports only the official Krea 2 Raw base.';
  return null;
}
