/* How a full model gets BACK to a pod, and what each way costs.

   A ~26 GB checkpoint can reach a rented pod by two roads, and until now the
   choice was made in the source code rather than by the person paying for it:

     - Hugging Face — the pod downloads the file itself, over a datacenter
       link. Minutes. It needs a Hub copy to exist, and the weights travel
       through a third party.
     - This computer — the file is sent straight up from here. Nothing outside
       the machine is involved, and it costs the user's uplink.

   THE NUMBER THAT DECIDES IT is neither speed nor privacy: the pod is RENTED
   AND BILLED while it waits for its file. Three hours of upload at $1.40/h is
   $4.20 of GPU computing nothing. So every line this module produces ends in a
   duration and a price, and it never claims a measurement it does not have.

   JSX-free on purpose: `node --test` then exercises the real formatting rules
   that the dialog renders, not a copy of them. */

export const TRANSPORTS = ['hub', 'direct'];

export const TRANSPORT_LABELS = {
  hub: '☁ Hugging Face',
  direct: '💻 This computer',
};

/* One road's option object out of the backend plan, or an empty stand-in.
   Never returns null: the dialog renders both roads ALWAYS, because a road
   that is missing from the UI is a trade-off the user never learns exists. */
export function transportOption(options, id) {
  const found = (Array.isArray(options) ? options : [])
    .find((o) => o && o.transport === id);
  return found || { transport: id, available: false, reason: null };
}

/* Which road the dialog opens on. The backend's default is honoured whenever
   it is usable; otherwise the first road that IS. A dialog that opens on a
   disabled option with a disabled button reads as broken software. */
export function initialTransport(plan) {
  const options = plan?.options;
  const wanted = plan?.default_transport;
  if (wanted && transportOption(options, wanted).available) return wanted;
  const usable = TRANSPORTS.find((id) => transportOption(options, id).available);
  return usable || wanted || 'hub';
}

export function formatSize(bytes) {
  const v = Number(bytes) || 0;
  if (v <= 0) return '';
  if (v < 1e6) return `${Math.round(v / 1e3)} kB`;
  if (v < 1e9) return `${Math.round(v / 1e6)} MB`;
  return `${(v / 1e9).toFixed(1)} GB`;
}

/* Rounded UP, always. A forecast that turns 89 minutes into "1 h" is the kind
   of small lie that makes a whole panel untrustworthy — and this one is
   attached to a price. */
export function formatDuration(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  if (s < 90) return `${Math.max(1, s)} s`;
  const minutes = Math.ceil(s / 60);
  if (minutes < 90) return `${minutes} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

export function formatCost(usd) {
  const v = Number(usd) || 0;
  if (v <= 0) return '';
  return v < 1 ? `$${v.toFixed(2)}` : `$${v.toFixed(2)}`;
}

/* Where the speed came from, in the user's words. The distinction is the whole
   credibility of the forecast: "measured on your own transfers" is worth
   believing, "assumed" is worth checking, and pretending the second is the
   first is worse than saying nothing. */
export function rateNote(option) {
  const mbps = Number(option?.rate_mbps) || 0;
  if (!mbps) return '';
  const samples = Number(option?.rate_samples) || 0;
  switch (option?.rate_source) {
    case 'measured':
      return `measured at ${mbps} Mbit/s on your last `
        + `${samples} transfer${samples === 1 ? '' : 's'}`;
    case 'configured':
      return `at the ${mbps} Mbit/s upload speed set in Settings`;
    case 'floor':
      return `at the ${mbps} Mbit/s a rented pod is required to have`;
    default:
      return `estimated — assuming ${mbps} Mbit/s upload, since no transfer of `
        + 'yours has been measured yet';
  }
}

/* The one line under the selected road: how big, how long, how much GPU.
   Returns '' when there is nothing honest to say (an unavailable road, or a
   size we never learned) rather than a row of zeroes. */
export function transportSummary(option) {
  if (!option || option.available === false) return '';
  const size = formatSize(option.bytes);
  const time = formatDuration(option.seconds);
  if (!size || !Number(option.seconds)) return '';
  const parts = [`${size} · ${time}`];
  const cost = formatCost(option.gpu_cost);
  if (cost) parts.push(`${cost} of rented GPU while it waits`);
  return parts.join(' · ');
}

/* Every road that cannot be taken, with its reason and its label.

   Rendered ALWAYS, not only when the closed road happens to be the selected
   one. A disabled button whose explanation lives in a `title` tooltip is
   greyed out with no reason as far as anyone reading the screen is concerned —
   it never appears in a screenshot, never on a touch screen, and never to a
   screen reader that is not hovering. The whole complaint this feature answers
   is that the trade-off was invisible; hiding half of it in a tooltip would
   reproduce it one level down. */
export function closedRoads(plan) {
  return TRANSPORTS
    .map((id) => ({ id, option: transportOption(plan?.options, id) }))
    .filter(({ option }) => option.available === false && option.reason)
    .map(({ id, option }) => ({
      transport: id,
      label: TRANSPORT_LABELS[id] || id,
      reason: option.reason,
    }));
}

/* Whether the dialog may submit on this road, and why not when it may not. */
export function transportBlockedReason(plan, transport) {
  if (!plan) return null;
  const option = transportOption(plan.options, transport);
  if (option.available) return null;
  return option.reason
    || `${TRANSPORT_LABELS[transport] || transport} is not available for this run.`;
}
