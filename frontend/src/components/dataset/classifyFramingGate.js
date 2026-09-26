// 📐 Classify framing — UI gate (PURE JS, JSX-free so node --test can import it).
//
// Imported images keep framing = NULL unless head-crop was on at import time, and
// the Composition bar only counts images whose framing is KNOWN. A drag-and-drop
// import (the default on body-fidelity datasets) therefore leaves the bar at 0 with
// no way out inside the app: the vision pass that fills those framings exists
// (POST /api/dataset/<id>/classify) but had no trigger.
//
// The set this module counts is EXACTLY the set the server pass acts on
// (face_dataset_service.classify_images: source='import' AND framing IS NULL) —
// including rejected rows, which the pass classifies too. Counting the composition
// buckets instead would under-report and the button would promise less than it does.

const IMPORT_SOURCE = 'import';

/** How many images the classify pass would actually process. */
export function countUnclassified(images) {
  if (!Array.isArray(images)) return 0;
  return images.filter((i) => i && i.source === IMPORT_SOURCE && !i.framing).length;
}

// The vision pass runs on the ACTIVE local engine (caps.vision: Ollama or
// llama.cpp — same field shape either way). Each state gets the ONE action that
// fixes it — a disabled button with no reason is the failure mode we are
// removing, not a style choice.
const LLAMACPP_ABSENT = 'The llama.cpp server is not answering at its URL. Start llama-server yourself '
  + '(the URL lives in Settings ▸ Local tools), then try again.';
const OLLAMA_ABSENT = 'Ollama is not installed. Install it, then pull a vision model — '
  + 'Settings ▸ Local tools walks through both.';
const OLLAMA_STOPPED = 'Ollama is installed but not running. Start it from '
  + 'Settings ▸ Local tools (▶ Start Ollama), then try again.';
const OLLAMA_NO_MODEL = 'Ollama is running but its vision model is not pulled yet. '
  + 'Pull it from Settings ▸ Local tools, then try again.';
const LLAMACPP_NO_MODEL = 'The llama.cpp server is running, but the model named in Settings '
  + '▸ Local tools is not loaded. Load it in llama-server or clear the field, then try again.';

/** Why the pass cannot run right now, or null. `loading` = capabilities not fetched
 * yet: unknown is NOT "missing", so we say nothing and simply keep the button idle. */
export function classifyBlockedReason(vision, loading = false) {
  if (loading) return null;
  const o = vision || {};
  if (o.backend === 'llamacpp') {
    if (!o.reachable) return LLAMACPP_ABSENT;
    if (!o.vision_model_ready) return LLAMACPP_NO_MODEL;
    return null;
  }
  if (!o.installed && !o.reachable) return OLLAMA_ABSENT;
  if (!o.reachable) return OLLAMA_STOPPED;
  if (!o.vision_model_ready) return OLLAMA_NO_MODEL;
  return null;
}

/** Everything the 📐 Classify framing affordance needs.
 *
 * `activity` is the dataset payload's persistent server-side batch indicator, so a
 * pass keeps showing its progress across a page reload. */
export function classifyFramingState({
  images, vision, capsLoading = false, busy = false, activity = null,
} = {}) {
  const count = countUnclassified(images);
  const running = !!activity && activity.kind === 'classify';
  const done = running ? (activity.done || 0) : 0;
  const total = running ? (activity.total || 0) : 0;
  const blockedReason = classifyBlockedReason(vision, capsLoading);
  return {
    count,
    running,
    // Nothing left to classify (or never anything: an empty dataset, a fully
    // classified one) → the button is not on screen at all. It stays while a pass
    // runs so its progress has somewhere to live.
    visible: running || count > 0,
    blocked: !!blockedReason,
    blockedReason,
    disabled: running || busy || capsLoading || !!blockedReason,
    label: running
      ? `📐 Classifying framing… ${done}/${total || count}`
      : `📐 Classify framing (${count})`,
    title: blockedReason
      || (running
        ? 'A framing pass is already running on this dataset'
        : `Reads ${count} imported image(s) with a local vision model and sorts each `
          + 'into face / bust / body / back, so they finally count in Composition. '
          + 'Uses the GPU; it waits rather than competing with a training run.'),
  };
}

/** Toast after the pass returns. `classified === 0` while there WAS work to do is the
 * silent failure to name: the vision call answers empty when Ollama is down, and the
 * server keeps framing NULL on purpose (so a retry can still work). */
export function classifyResultMessage(classified, expected) {
  const n = Number(classified) || 0;
  const want = Number(expected) || 0;
  if (n === 0 && want > 0) {
    return {
      tone: 'error',
      text: 'Nothing could be classified — the vision model returned no answer. '
        + 'Check Ollama in Settings ▸ Local tools (server running, vision model pulled), '
        + 'then try again. Nothing was changed.',
    };
  }
  if (n > 0 && want > n) {
    return { tone: 'info', text: `${n}/${want} classified — the rest returned no answer; run it again to retry them.` };
  }
  return { tone: 'success', text: `${n} classified` };
}
