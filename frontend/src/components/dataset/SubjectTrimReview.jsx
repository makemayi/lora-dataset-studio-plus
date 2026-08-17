import { useMemo, useState } from 'react';
import { PRIMARY_BUTTON, QUIET_BUTTON } from '../common/surfaces';

/* The subject-trim review: every proposed crop, before a single file is written.
 *
 * BOTH PANES COME FROM THE SAME FILE THE DATASET ALREADY SERVES.
 * No preview images exist anywhere — the server's measuring pass writes a
 * manifest and no pixels. The "before" pane is the photo with a rectangle laid
 * over it; the "after" pane is the same photo scaled up and offset inside a
 * clipping box, which is exactly what the crop will look like. That is zero
 * disk writes and zero extra requests for a screen that can show a hundred
 * pairs.
 *
 * The percentages come from the manifest, so an inline `style` is the only way
 * to express them: Tailwind scans source TEXT, and a class built by
 * interpolation is silently absent from the build (CLAUDE.md, UI rule 4).
 */

const SKIP_TEXT = {
  'nothing-much-to-remove': 'Nothing much to remove — the subject already fills this frame.',
  'subject-too-small': 'Subject too small — the crop would be under 256px on its short edge.',
  'no-subject-found': 'No subject found in this image.',
  // A Stop is not a failure. The server keeps the two apart on purpose; a
  // screen that folded them together would report the user's own click as a
  // broken engine.
  stopped: 'Not measured — you stopped the pass before this one.',
  failed: 'This image could not be measured.',
};

function imageUrl(datasetId, filename) {
  return `/api/dataset/${datasetId}/img/${encodeURIComponent(filename)}`;
}

function BeforePane({ datasetId, item }) {
  const [iw, ih] = item.image || [1, 1];
  const frame = item.frame;
  return (
    <div className="relative overflow-hidden rounded-lg bg-black"
      style={{ aspectRatio: `${iw} / ${ih}` }}>
      <img src={imageUrl(datasetId, item.filename)} alt={`${item.filename} before cropping`}
        loading="lazy" className="h-full w-full select-none object-cover" />
      {frame && (
        /* Painted ON a photograph, so it takes the on-image treatment rather
           than a semantic border — see CLAUDE.md, UI rule 0's third exception. */
        <span aria-hidden className="absolute border-2 border-rose-500"
          style={{
            left: `${(frame[0] / iw) * 100}%`,
            top: `${(frame[1] / ih) * 100}%`,
            width: `${(frame[2] / iw) * 100}%`,
            height: `${(frame[3] / ih) * 100}%`,
          }} />
      )}
    </div>
  );
}

function AfterPane({ datasetId, item }) {
  const [iw, ih] = item.image || [1, 1];
  const frame = item.frame;
  if (!frame) {
    return (
      <div className="grid place-items-center rounded-lg bg-surface-raised p-4 text-center text-[0.6875rem] text-content-subtle"
        style={{ aspectRatio: '9 / 16' }}>
        No crop to show
      </div>
    );
  }
  const [fx, fy, fw, fh] = frame;
  return (
    <div className="relative overflow-hidden rounded-lg bg-black"
      style={{ aspectRatio: `${fw} / ${fh}` }}>
      <img src={imageUrl(datasetId, item.filename)} alt={`${item.filename} after cropping`}
        loading="lazy" className="absolute select-none"
        style={{
          width: `${(iw / fw) * 100}%`,
          height: `${(ih / fh) * 100}%`,
          left: `${(-fx / fw) * 100}%`,
          top: `${(-fy / fh) * 100}%`,
          maxWidth: 'none',
        }} />
    </div>
  );
}

export default function SubjectTrimReview({ datasetId, preview, onApply, onDiscard, busy = false }) {
  const items = preview?.items || [];
  // The manifest carries an `error` string when the measuring pass CRASHED —
  // the server writes an empty manifest rather than none precisely so the
  // failure has somewhere to surface. A panel that ignored it would render
  // "Nothing to crop", which is the screen a successful no-op produces.
  const failure = preview?.error || null;
  const applicable = useMemo(() => items.filter((i) => !i.skip && i.frame), [items]);
  const skipped = useMemo(() => items.filter((i) => i.skip || !i.frame), [items]);
  // Track what was UNCHECKED, so a row that arrives later is checked by default
  // rather than silently excluded.
  const [unchecked, setUnchecked] = useState(() => new Set());

  const chosen = applicable.filter((i) => !unchecked.has(i.image_id));

  const toggle = (id) => setUnchecked((prev) => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  return (
    <section id="ds-subject-trim-review"
      className="flex min-w-0 scroll-mt-20 flex-col gap-3 rounded-xl bg-surface-raised p-3 lg:scroll-mt-24"
      aria-labelledby="subject-trim-review-title">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 id="subject-trim-review-title" className="m-0 text-sm font-semibold text-content">
            ✂ Subject trim review
          </h3>
          {/* All three stay mounted — Chrome auto-translate rewrites text nodes
              and a ternary swap throws NotFoundError (CLAUDE.md, UI rule 5). */}
          <p className="m-0 mt-0.5 max-w-3xl text-[0.6875rem] leading-relaxed text-content-subtle"
            hidden={applicable.length === 0 || !!failure}>
            Each crop keeps the whole person and takes the least background that leaves it no
            narrower than 9:16. Nothing is written until you confirm; afterwards the whole batch
            can still be undone.
          </p>
          <p className="m-0 mt-0.5 max-w-3xl text-[0.6875rem] leading-relaxed text-content-subtle"
            hidden={applicable.length > 0 || !!failure}>
            Nothing to crop — every image here already fills its frame or has no subject to crop
            around.
          </p>
          <p className="m-0 mt-0.5 max-w-3xl text-[0.6875rem] leading-relaxed text-rose-700"
            hidden={!failure}>
            {failure}
          </p>
        </div>
        <span className="shrink-0 rounded-full bg-surface px-2 py-0.5 text-[0.6875rem] font-semibold text-content">
          {applicable.length} proposed
        </span>
      </div>

      <div className="grid min-w-0 grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {applicable.map((item) => {
          const [iw, ih] = item.image;
          const removed = Math.round(100 - (item.frame[2] * item.frame[3] * 100) / (iw * ih));
          const checked = !unchecked.has(item.image_id);
          return (
            <article key={item.image_id} className="min-w-0 rounded-lg bg-surface p-2">
              <label className="mb-1.5 flex min-w-0 items-center gap-2 text-[0.6875rem] text-content">
                <input type="checkbox" checked={checked} disabled={busy}
                  onChange={() => toggle(item.image_id)}
                  className="h-3.5 w-3.5 accent-indigo-600" />
                <span className="truncate font-semibold">{item.filename}</span>
                <span className="ml-auto shrink-0 text-content-subtle">
                  −{removed}% · {item.out[0]}×{item.out[1]}
                </span>
              </label>
              <div className="grid min-w-0 grid-cols-2 gap-1.5">
                <BeforePane datasetId={datasetId} item={item} />
                <AfterPane datasetId={datasetId} item={item} />
              </div>
            </article>
          );
        })}
      </div>

      {skipped.length > 0 && (
        <div className="rounded-lg bg-surface p-2">
          <h4 className="m-0 mb-1 text-[0.6875rem] font-semibold text-content">
            {skipped.length} left alone
          </h4>
          <ul className="m-0 flex list-none flex-col gap-0.5 p-0">
            {skipped.map((item) => (
              <li key={item.image_id}
                className="flex min-w-0 flex-wrap gap-2 text-[0.6875rem] text-content-subtle">
                <span className="truncate font-medium text-content-muted">{item.filename}</span>
                <span className="min-w-0 break-words">{SKIP_TEXT[item.skip] || 'Left alone.'}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <button type="button" disabled={busy || chosen.length === 0}
          onClick={() => onApply(chosen.map((i) => i.image_id))}
          className={PRIMARY_BUTTON}>
          <span hidden={chosen.length === 1}>Crop {chosen.length} images</span>
          <span hidden={chosen.length !== 1}>Crop 1 image</span>
        </button>
        <button type="button" disabled={busy} onClick={onDiscard} className={QUIET_BUTTON}>
          Cancel
        </button>
        <span className="text-[0.6875rem] text-content-subtle">
          Originals go to the Trash and the whole batch can be undone afterwards.
        </span>
      </div>
    </section>
  );
}
