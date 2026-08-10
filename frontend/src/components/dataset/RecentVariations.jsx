/** Recent variations — a strip of round faces in the workspace rail.
 *
 * The rail carried a nav and a checklist and then dead space, and the app had
 * nowhere at all that showed output ACROSS datasets: every other surface is
 * scoped to the one that is open. This is both — a glance at what came out
 * lately, and one click back into the dataset it came from.
 *
 * The crop is `object-cover` with the frame pulled UP (`object-position` at
 * 50%/28%), not a face detection: these are portrait variations, the head is
 * near the top of the frame in essentially all of them, and a detection pass
 * per thumbnail would cost more than the feature is worth. A body shot lands on
 * a chest rather than a face — a fair trade for something that is a shortcut,
 * not a gallery.
 */
import { useEffect, useState } from 'react';
import { apiFetch } from '../../api/fetchClient';

export const RECENT_LIMIT = 8;

export default function RecentVariations({ onOpen, currentId = null, refreshKey = 0 }) {
  const [images, setImages] = useState([]);

  useEffect(() => {
    let alive = true;
    apiFetch(`/api/dataset/recent-images?limit=${RECENT_LIMIT}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (alive && d) setImages(Array.isArray(d.images) ? d.images : []); })
      /* A shortcut strip is not worth a toast: it either shows faces or it
         shows nothing, and the page it lives on works either way. */
      .catch(() => {});
    return () => { alive = false; };
  }, [refreshKey]);

  if (!images.length) return null;

  return (
    <section aria-labelledby="recent-variations-heading" className="flex flex-col gap-2">
      <p id="recent-variations-heading"
        className="m-0 px-3 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">
        Recent
      </p>
      <ul className="m-0 flex list-none flex-wrap gap-1.5 p-0 px-2">
        {images.map((img) => {
          const here = img.dataset_id === currentId;
          const title = `${img.dataset_name}${img.label ? ` — ${img.label}` : ''}`;
          return (
            <li key={img.image_id}>
              <button
                type="button"
                onClick={() => onOpen?.(img.dataset_id)}
                title={here ? `${title} (this dataset)` : `Open ${title}`}
                aria-label={here ? `${title}, the dataset you are in`
                  : `Open the dataset ${img.dataset_name}`}
                aria-current={here ? 'true' : undefined}
                className={`block h-11 w-11 overflow-hidden rounded-full transition-transform hover:scale-105 ${
                  here ? 'ring-2 ring-indigo-400' : 'ring-1 ring-border'}`}
              >
                <img
                  src={`/api/dataset/${img.dataset_id}/img/${encodeURIComponent(img.filename)}`}
                  alt=""
                  loading="lazy"
                  className="h-full w-full object-cover"
                  style={{ objectPosition: '50% 28%' }}
                />
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
