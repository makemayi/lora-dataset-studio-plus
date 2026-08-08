import { useEffect, useRef, useState } from 'react';
import { postJson } from './useDataset';

/** Ask, once per screen, whether these runs' Hugging Face repositories are
 *  still there — and never make a panel wait for the answer.
 *
 * WHY A HOOK AND NOT A FIELD OF THE LISTING
 * -----------------------------------------
 * The three surfaces that draw a dense run all render `artifact_status`, which
 * is stamped at delivery and never revisited. Adding a live Hub check to the
 * listing endpoints would have put a network round-trip in front of a page that
 * is mostly about local files, on every poll, and turned a Hub outage into a
 * slow Checkpoints tab. So the panels paint first, from the record, in the past
 * tense — and this replaces "not re-checked since" with something measured as
 * soon as the Hub answers.
 *
 * It deliberately does NOT poll. A repository does not come back, and a panel
 * that re-polls itself must not re-ask the Hub at that rate; the backend caches
 * with a short TTL for the same reason. One question per set of runs, re-asked
 * only when the set itself changes.
 *
 * Returns `{ [runId]: { state, detail, checked_at } }` — missing keys mean
 * "not checked", which every caller has to render honestly anyway.
 */
export default function useHubPresence(runIds) {
  const [presence, setPresence] = useState({});
  // The ids as a stable string: a fresh array identity on every render must not
  // re-fire the request, and `[runIds]` in a dep array would do exactly that.
  // Non-numeric ids are dropped here rather than sent as nulls the server has
  // to skip — a local run has no cloud id and simply has nothing to ask about.
  const key = (runIds || []).map(Number).filter(Number.isFinite)
    .sort((a, b) => a - b).join(',');
  const asked = useRef('');

  useEffect(() => {
    if (!key || asked.current === key) return undefined;
    asked.current = key;
    let alive = true;
    postJson('/api/dataset/train/cloud/hub-presence',
      { run_ids: key.split(',').map(Number) })
      .then((res) => {
        // A failed call leaves the map alone on purpose: "we could not ask" and
        // "we asked and it is gone" are not the same sentence, and only the
        // backend is in a position to tell them apart.
        if (!alive || !res?.ok || !res.results) return;
        setPresence((prev) => ({ ...prev, ...res.results }));
      })
      .catch(() => {});
    return () => { alive = false; };
  }, [key]);

  return presence;
}
