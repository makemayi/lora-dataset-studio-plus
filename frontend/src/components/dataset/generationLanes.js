/* Which lane a generation batch runs in — and therefore what a running batch
   is allowed to block.
 *
 * PURE JS (no JSX) so `node --test` can exercise it directly, same split as
 * engineSelection.js.
 *
 * WHY THIS EXISTS
 * ---------------
 * A local batch and an API batch share NOTHING on the server: ComfyUI work goes
 * through job_queue's single worker, and the API engines fan out on their own
 * `ThreadPoolExecutor(max_workers=3)` inside dataset_generation_service. Neither
 * touches the other's scheduler, and `dataset_activity.begin` only refuses an
 * EXCLUSIVE operation (a Bank export), never a second generate.
 *
 * The UI did not know that. One boolean — `busy = localFlag || !!activity` —
 * disabled the Generate button and every engine card whenever ANY batch was
 * live, so a ComfyUI run made ChatGPT unclickable for its whole duration. On a
 * 40-image local batch that is twenty minutes of a paid API lane the machine
 * was never using. Reported as "本地引擎和云端的API为什么不能同时跑…这不合理",
 * which it was.
 *
 * The lane is the unit because it matches the thing that is actually scarce:
 * ONE GPU, so local batches still queue behind each other; API calls are
 * rate-limited by the provider, not by this machine.
 */
import { API_ENGINES, LOCAL_ENGINES } from './engineSelection.js';

export const LANE_LOCAL = 'local';
export const LANE_API = 'api';

/** The lane an engine id belongs to, or null for one we do not know. Null is
 *  deliberately NOT treated as either lane by the blockers below: an engine
 *  this build has never heard of must not silently inherit permissions. */
export function laneOf(engine) {
  const id = String(engine || '').toLowerCase();
  if (LOCAL_ENGINES.includes(id)) return LANE_LOCAL;
  if (API_ENGINES.includes(id)) return LANE_API;
  return null;
}

/** The lanes a set of selected engines would launch into. */
export function lanesFor(engines) {
  const lanes = new Set();
  for (const engine of Array.from(engines || [])) {
    const lane = laneOf(engine);
    if (lane) lanes.add(lane);
  }
  return lanes;
}

/** The lanes currently busy, from the server's `activities` list.
 *
 *  A batch with no `engine` (a caption pass, a watermark sweep, an ✨ improve
 *  run) counts as LOCAL: those all drive ComfyUI or the local ML environments.
 *  Guessing the other way would let an API launch collide with the one thing on
 *  this machine that is genuinely serialized. */
export function busyLanes(activities) {
  const lanes = new Set();
  for (const entry of Array.isArray(activities) ? activities : []) {
    if (!entry) continue;
    lanes.add(laneOf(entry.engine) || LANE_LOCAL);
  }
  return lanes;
}

/** Why THIS selection cannot launch right now, or null when it can.
 *
 *  Names the lane and the engine in the way, because "please wait" on a button
 *  that was clickable a second ago is the message that sends someone to the
 *  server log. */
export function launchBlockedReason(engines, activities) {
  const want = lanesFor(engines);
  if (!want.size) return null;
  const busy = busyLanes(activities);
  const clash = [...want].filter((lane) => busy.has(lane));
  if (!clash.length) return null;
  const list = Array.isArray(activities) ? activities : [];
  const offender = list.find(
    (entry) => clash.includes(laneOf(entry?.engine) || LANE_LOCAL));
  const what = offender?.engine ? `a ${offender.engine} batch` : 'a batch';
  return clash.includes(LANE_LOCAL)
    ? `${what} is already running on your GPU — local engines render one at a time`
    : `${what} is already running on the API lane`;
}

/** True when an engine CARD should be locked: its own lane is busy. The card is
 *  a launch control, so it follows the same rule as the button — a running
 *  ComfyUI batch must not grey out ChatGPT. */
export function engineCardLocked(engine, activities) {
  const lane = laneOf(engine);
  return !!lane && busyLanes(activities).has(lane);
}
