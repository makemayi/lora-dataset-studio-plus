/* Which lane a pre-launch preflight is asked for, and the URL that asks it.
 *
 * The preflight mixes two kinds of row: properties of the DATASET (image count,
 * captions, identity leaks, near-duplicates, untriaged images, face masks) and
 * reads of THIS MACHINE (GPU memory, torch build). Only the first kind means
 * anything when the job runs on a rented pod, so the lane rides in the query
 * string and the server drops the machine-scope rows for `cloud`.
 *
 * Plain .js on purpose: node --test does not parse JSX, so the logic worth
 * testing lives outside TrainingPanel.jsx.
 */

/** 'cloud' or 'local' — anything unrecognised falls back to 'local', which is
 * the server's default and the historical (unfiltered) payload. */
export function normalizeLane(lane) {
  return lane === 'cloud' ? 'cloud' : 'local';
}

/** The ▶ Continue dialog resolves to a payload carrying its own lane; the plain
 * Train button has none. Same normalisation either way. */
export function laneOfPayload(payload) {
  return normalizeLane(payload && payload.lane);
}

/** GET url for the preflight. `lane` is only sent for the cloud lane: a request
 * with no `lane` must stay byte-for-byte the historical one, so nothing that
 * already calls this route (the workspace readiness badge) changes behaviour.
 *
 * `masked` follows the same rule for the opposite reason. Whether the run wants
 * person masks is a localStorage preference the server cannot read, so a launch
 * states it and gets the "rembg is missing, this run trains unmasked" row; a
 * caller that has no opinion (the readiness badge) omits it and gets no row,
 * rather than a warning about a mask nobody asked for. */
export function preflightUrl(datasetId, { trainType, variant, lane, masked } = {}) {
  const qs = [];
  if (trainType) qs.push(`train_type=${encodeURIComponent(trainType)}`);
  if (variant) qs.push(`variant=${encodeURIComponent(variant)}`);
  if (normalizeLane(lane) === 'cloud') qs.push('lane=cloud');
  if (masked !== undefined && masked !== null) qs.push(`masked=${masked ? '1' : '0'}`);
  return `/api/dataset/${datasetId}/train/preflight${qs.length ? `?${qs.join('&')}` : ''}`;
}

/** Rows the modal must never show on a cloud lane, kept here as the client-side
 * mirror of the server filter — used by the test that guards the whole point of
 * this feature: a warning that fires wrongly teaches people to ignore all of
 * them. Belt and braces; the server remains authoritative. */
export const MACHINE_SCOPE_CHECKS = ['vram', 'torch_arch'];

/** Defensive filter over a preflight payload's `checks`, for the lane given. */
export function checksForLane(checks, lane) {
  const list = Array.isArray(checks) ? checks : [];
  if (normalizeLane(lane) !== 'cloud') return list;
  return list.filter((c) => c && c.scope !== 'machine'
    && !MACHINE_SCOPE_CHECKS.includes(c.id));
}
