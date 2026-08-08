/* Settings › Storage — the PURE half.

   Kept JSX-free so `node --test` can import it. The interesting part of this tab
   is not the markup: it is what the app is allowed to CLAIM about a folder you
   are about to point it at, and what it promises to do with the files already
   there.

   The rule this file encodes, and the reason it exists: a location change never
   moves anything on its own. The user picks — move the existing files, or adopt
   the new folder as-is — and both choices are spelled out before the save. */

/* Decimal units, like every disk vendor and like the Trash card already does.
   `null`/undefined is a real state here — "not measured yet" — and must never
   render as 0, or an unmeasured 100 GB folder would read as empty. */
export function formatSize(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return '—';
  const v = Number(n);
  if (v <= 0) return 'empty';
  if (v < 1e6) return `${Math.max(1, Math.round(v / 1e3))} KB`;
  if (v < 1e9) return `${Math.round(v / 1e6)} MB`;
  return `${(v / 1e9).toFixed(1)} GB`;
}

/* Free space of the volume a location sits on, as one short sentence. Returns
   '' when the server could not read it — silence beats a made-up number. */
export function volumeLabel(volume) {
  if (!volume || !volume.total_bytes) return '';
  return `${formatSize(volume.free_bytes)} free of ${formatSize(volume.total_bytes)}`;
}

/* 0-100, or null when the job has not sized its work yet (the scan phase). */
export function movePercent(job) {
  if (!job || !job.bytes_total) return null;
  const pct = Math.round((Number(job.bytes || 0) / Number(job.bytes_total)) * 100);
  return Math.min(100, Math.max(0, pct));
}

export function moveLabel(job) {
  if (!job) return '';
  if (job.phase === 'scanning') return 'Looking at what has to move…';
  if (job.phase === 'error') return job.error || 'The move failed.';
  if (job.phase === 'done') return 'Move complete.';
  const pct = movePercent(job);
  const of = `${job.files || 0} / ${job.files_total || 0} files`;
  return `Moving ${of}${pct === null ? '' : ` — ${pct}%`}`;
}

/* What the user may do with a validated target, and what each choice means.

   `move` is offered only when there is something to move AND the target can
   take it; `adopt` is always offered, because "start fresh here, leave the old
   files where they are" is a legitimate answer — and on a full disk it is often
   the only one that fits. Nothing is ever implicit. */
export function relocationChoices({ validation, currentSize } = {}) {
  if (!validation || !validation.ok) return [];
  if (validation.default) {
    return [{
      id: 'adopt',
      label: 'Use the default folder',
      detail: 'Back to the folder inside the app’s data directory. Files already '
        + 'written elsewhere stay where they are.',
    }];
  }
  const out = [];
  const known = typeof currentSize === 'number' && currentSize > 0;
  const tooBig = known && typeof validation.free_bytes === 'number'
    && validation.free_bytes < currentSize;
  out.push({
    id: 'move',
    label: known ? `Move the ${formatSize(currentSize)} already there` : 'Move what is already there',
    detail: tooBig
      ? `That drive has ${formatSize(validation.free_bytes)} free — less than the `
        + `${formatSize(currentSize)} to move. Free some space first.`
      : 'Everything is copied first; the old folder is only removed once every '
        + 'file has landed.',
    disabled: !!tooBig,
  });
  out.push({
    id: 'adopt',
    label: 'Start using it empty',
    detail: 'The new folder is used from now on. Nothing is copied and nothing '
      + 'is deleted — the old folder keeps its files, and the app stops looking '
      + 'at them.',
  });
  return out;
}

/* One line per category for the "what lives where" table, sizes folded in when
   they have been measured. Order is the server's — it groups your data first
   and the app's own files last. */
export function locationRows(locations, sizes) {
  const measured = sizes || {};
  return (locations || []).map((loc) => ({
    ...loc,
    sizeBytes: Object.prototype.hasOwnProperty.call(measured, loc.key)
      ? measured[loc.key] : null,
    sizeLabel: Object.prototype.hasOwnProperty.call(measured, loc.key)
      ? formatSize(measured[loc.key]) : 'not measured',
    volumeLabel: volumeLabel(loc.volume),
  }));
}
