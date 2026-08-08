/* Hugging Face private-storage card — the PURE half.

   Kept JSX-free so `node --test` can import it: the interesting part of this
   card is not the markup, it is the three sentences it is allowed to say about
   a repo the user is about to delete forever.

   The server measures (backend/app/services/hf_storage.py); this file only
   decides how the measurement reads, and — the load-bearing bit — whether
   deleting a given lds-base-* cache is serene or destructive. */

const GB = 1000 ** 3;

/* Decimal GB, the unit huggingface.co itself displays. `null`/undefined is a
   real state here ("the Hub did not report a size"), never 0. */
export function formatBytes(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return '—';
  const v = Number(n);
  if (v < 1000 ** 2) return `${Math.round(v / 1000)} kB`;
  if (v < GB) return `${Math.round(v / 1000 ** 2)} MB`;
  return `${(v / GB).toFixed(1)} GB`;
}

/* Why a cache exists on the Hub but has no local source any more. These are the
   codes hf_base_push.local_base_payload raises, plus our own 'unknown_source'
   for a repo no run and no dataset can be matched to. */
const LOCAL_REASON_TEXT = {
  weights_missing: 'the local weights file is gone',
  not_converted: 'the local Z-Image base is no longer converted',
  unsupported_family: 'its family no longer has a cloud custom lane',
  unknown_source: 'no run or dataset on this machine points at it',
  no_custom_base: 'no custom base is attached to it any more',
};

/* The one judgement this card exists to make. A lds-base-* repo is a CACHE of a
   local file — deleting it costs one re-upload. But if the local file is gone,
   that "cache" is the only copy of those weights left, and deleting it is not
   an undo away. Never soften this into a single "Delete" label. */
export function deletionSafety(cache) {
  if (cache?.local_available) {
    return {
      level: 'safe',
      label: 'Re-pushable',
      text: 'The local file is still on this machine — deleting only costs one re-upload the next time you launch on this base.',
    };
  }
  return {
    level: 'only_copy',
    label: 'Only copy',
    text: `This may be the last copy of these weights: ${
      LOCAL_REASON_TEXT[cache?.local_reason] || 'its local source cannot be found'
    }. Download it from Hugging Face first if you still want it.`,
  };
}

export function lastRunLabel(cache) {
  const run = cache?.last_run;
  if (!run) return 'Never used by a recorded run';
  const when = String(run.created_at || '').slice(0, 10);
  const name = run.name ? ` “${run.name}”` : '';
  return `Last used by cloud run #${run.id}${name}${when ? ` (${when})` : ''}`;
}

/* Headline lines for the storage meter. Returns { measured, lines[], tone }.
   `measured:false` is a first-class answer: the Hub publishes no quota endpoint
   and a listing can fail, and saying "unknown" is the honest output — never a
   reassuring zero. */
export function storageSummary(payload) {
  if (!payload || !payload.ok) {
    return {
      measured: false,
      tone: 'unknown',
      lines: [
        payload?.error
          || 'Private storage could not be measured — Hugging Face publishes no quota endpoint, so this card sums the sizes of your private repos and that listing did not answer.',
      ],
    };
  }
  const f = payload.forecast || {};
  // The breakdown must ADD UP to the total, and since the fp8 export shipped it
  // stopped doing so: `needed_bytes` silently gained a ~14 GB fp8 term while
  // this sentence still said "checkpoint × kept + margin", so a 60 GB forecast
  // explained itself as 46 GB and the 14 GB gap looked like a bug in the card.
  // Same terms, same order as the server's own refusal sentence
  // (hf_storage.storage_refusal_message) — one story about one number. The term
  // is omitted, not zeroed, when the run does not export fp8.
  const terms = [`one ${formatBytes(f.checkpoint_bytes)} checkpoint × ${f.keeps || 1} kept`];
  if (f.fp8_bytes) terms.push(`a ${formatBytes(f.fp8_bytes)} fp8 export`);
  terms.push(`${formatBytes(f.margin_bytes)} margin`);
  const lines = [
    `${payload.namespace} uses ${formatBytes(payload.used_bytes)} across ${
      payload.private_repo_count || 0
    } private repo(s).`,
    `A full-model run needs about ${formatBytes(f.needed_bytes)} (${terms.join(' + ')}).`,
  ];
  let tone = 'ok';
  if (f.fits === false) {
    tone = 'blocked';
    lines.push(
      `That does not fit: about ${formatBytes(f.shortfall_bytes)} short of the ${formatBytes(
        f.limit_bytes,
      )} allowance assumed below.`,
    );
  } else if (f.fits === true) {
    lines.push(`Room left: about ${formatBytes(f.free_bytes)}.`);
  } else {
    tone = 'unknown';
    lines.push('Headroom for a full-model run could not be computed.');
  }
  if (f.limit_is_estimate) {
    lines.push(
      `The ${formatBytes(f.limit_bytes)} allowance is an ESTIMATE from the published plan table — Hugging Face exposes no quota endpoint, and a superseded revision keeps counting until a repo history is squashed. Set cloud.full_transformer.private_storage_limit_gb if you know your real ceiling.`,
    );
  }
  if (payload.unsized_repo_count) {
    lines.push(
      `${payload.unsized_repo_count} private repo(s) reported no size, so the total above is a floor.`,
    );
  }
  return { measured: true, tone, lines };
}

/* Caches, biggest first — the order in which someone freeing space wants them. */
export function sortedCaches(payload) {
  return [...((payload && payload.caches) || [])].sort(
    (a, b) => (b.used_bytes || 0) - (a.used_bytes || 0),
  );
}

export default storageSummary;
