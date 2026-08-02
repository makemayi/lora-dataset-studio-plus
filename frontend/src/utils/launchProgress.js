// "Is it launching, or is it dead?" — the question the cloud launch could not
// answer. Clicking 'Rent & train' froze the button on 'Launching…' with no
// step, no clock and no ceiling, while behind it the app was staging the
// dataset, searching vast.ai offers, renting a machine and waiting for a pod
// that may never boot (run #134: 'pod did not become ready in time — no boot
// progress for 25 min', watched as a mute button).
//
// The backend already knew all of it — status + the phase_detail the monitor
// writes on every poll — and now forwards it as `launch`. Nothing here invents
// a state: it formats what the monitor reported, and returns null when there is
// nothing to report so the caller keeps the phase sentence it always showed.

/* Whole-minute clock. Seconds only matter in the first minute (the click still
   feels instant); past that they add noise to a number that will run for
   minutes and are dropped. */
export function formatElapsed(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

/* Backend `launch` payload -> what the card renders, or null.
   `note` exists on the two steps with a real deadline, and announcing it is
   what turns a long wait from "it hung" into "it has N minutes, then the
   machine is released". The two deadlines are not the same shape and the
   sentences must not pretend they are: the pod boot is a clock on WALL TIME,
   the dataset upload a clock on IDLE BYTES — a 24 GB upload may run for hours
   and is in no danger as long as data keeps arriving. */
export function launchProgressView(launch) {
  if (!launch || typeof launch !== 'object') return null;
  const steps = Array.isArray(launch.steps) ? launch.steps : [];
  const active = steps.find((s) => s.state === 'active') || null;
  if (!active) return null;
  const elapsed = formatElapsed(launch.elapsed_seconds);
  const limit = Number(launch.boot_idle_limit_seconds) || 0;
  const uploadLimit = Number(launch.upload_stall_limit_seconds) || 0;
  return {
    steps,
    activeKey: active.key,
    // The step, then how long the whole launch has been running — the two
    // figures needed to tell a slow launch from a stuck one.
    headline: `${active.label} — ${elapsed} elapsed`,
    // phase_detail says WHERE inside the step ('pod up — waiting for the UI to
    // answer'). Dropped when it merely repeats the step label.
    detail: String(launch.detail || '').trim() || null,
    note: launchNote(active.key, limit, uploadLimit),
    elapsed,
  };
}

function launchNote(activeKey, bootLimit, uploadLimit) {
  if (activeKey === 'boot' && bootLimit > 0) {
    return `A pod that shows no boot progress for ${Math.round(bootLimit / 60)} min is given up: `
      + 'the machine is released automatically and you can launch again.';
  }
  if (activeKey === 'upload' && uploadLimit > 0) {
    return 'A large dataset can take a long time to upload — that is fine, this step '
      + `has no time limit. It is given up only if NO data reaches the machine for ${
        Math.round(uploadLimit / 60)} min, and then the machine is released automatically.`;
  }
  return null;
}

const NEVER_BOOTED = /pod did not become ready/i;

/* The boot timeout, said in full. The raw error names the timer but not what
   became of the machine the user rented — the only thing they actually want to
   know before relaunching. `_finish(run, 'error')` destroys the instance, and
   the run only reaches 'error' (rather than 'error_pod_kept') when that
   destroy was confirmed, so the sentence below is a fact, not a hope. */
export function podBootFailureView(run) {
  if (!run || run.status !== 'error' || !NEVER_BOOTED.test(String(run.error || ''))) {
    return null;
  }
  return {
    title: 'The rented machine never started',
    message: 'vast.ai handed over a host that never finished booting, so the run '
      + 'was given up before any training happened. The machine was released — it '
      + 'is no longer billing — and that host is skipped for a while. Launching '
      + 'again picks a different one.',
  };
}

const UPLOAD_STALLED = /upload stall watchdog|dataset upload stalled/i;

/* The upload stall, said in full. This failure is the one users are most
   likely to misread as "the training crashed": nothing was trained, nothing
   was lost, and the thing to look at is the link between this machine and the
   pod — not the dataset, not the recipe. Like podBootFailureView, the released
   machine is stated as a fact and not a hope: the supervisor only reaches
   'error' when the vast.ai destroy was confirmed, otherwise the run is parked
   in 'error_pod_kept' with its own warning. */
export function uploadStallFailureView(run) {
  if (!run || run.status !== 'error') return null;
  const text = `${run.error || ''} ${run.phase_detail || ''}`;
  if (!UPLOAD_STALLED.test(text)) return null;
  return {
    title: 'The dataset never reached the rented machine',
    message: 'The upload stopped making progress — no data at all was arriving — so the '
      + 'run was given up before any training happened. The machine was released and is no '
      + 'longer billing. This is almost always the connection between this computer and the '
      + 'pod rather than the dataset itself: check your upload connection and launch again. '
      + 'A very large dataset is not the problem on its own — a slow upload is allowed to '
      + 'take as long as it needs, as long as it keeps moving.',
  };
}

/* A launch has nothing to salvage, so the red button that stops it should not
   read like the one that throws away a trained checkpoint. */
export function stopButtonLabel(status) {
  return ['preparing', 'provisioning', 'uploading'].includes(String(status || ''))
    ? 'Cancel launch'
    : 'Stop run';
}

/* The dialog's own button while the POST is in flight. That request is not a
   formality — it freezes the dataset, checks the base repository and (for
   full-model runs) creates the delivery repository — so it can legitimately
   take tens of seconds, and a static 'Launching…' is exactly how a working
   request gets mistaken for a hung one. */
export function launchButtonLabel({ launching, elapsedSeconds, fullMode }) {
  if (!launching) return fullMode ? '☁️ Rent GPU & train full model' : '☁️ Rent & train';
  const s = Math.max(0, Math.round(Number(elapsedSeconds) || 0));
  return s < 3 ? 'Launching…' : `Launching… ${formatElapsed(s)}`;
}
