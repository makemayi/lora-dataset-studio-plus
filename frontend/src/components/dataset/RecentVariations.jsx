/** Recent activity — the workspace rail's shortcut strip, SPLIT by what happened
 *  LAST to each image, so a face you swapped an hour ago is no longer buried
 *  under rows merely generated later.
 *
 *  Four independent buckets, newest first, same item shape:
 *    · generated — fresh generations / regenerates;
 *    · edited    — face swap or watermark inpaint;
 *    · upscaled  — Klein rescue / ✨ improve;
 *    · captioned — the caption changed (its own clock, see caption_changed_at).
 *  An image can appear in more than one (generated, then captioned): each list
 *  answers "what did I just do?", not a partition.
 *
 *  SPLIT IN TWO ON PURPOSE (same reason as the first version): the list is a
 *  pure function of its items, and the fetching lives in the wrapper. `apiFetch`
 *  resolves to parsed JSON, not to a Response — the first version stored nothing
 *  and shipped an empty strip behind a green suite, which is exactly what a
 *  pure, renderable list is here to prevent.
 */
import { useEffect, useState } from 'react';
import { apiFetch } from '../../api/fetchClient';

export const RECENT_LIMIT = 8;

/* Display order + English labels. Kept as data so the render can map it and a
   test can name each section without guessing at JSX. */
export const RECENT_SECTIONS = [
  { key: 'generated', label: 'Recent generations' },
  { key: 'edited', label: 'Recent edits' },
  { key: 'upscaled', label: 'Recent upscales' },
  { key: 'captioned', label: 'Recent captions' },
];

export function RecentVariationsList({ activity, onOpen, currentId = null }) {
  const sections = RECENT_SECTIONS
    .map((s) => ({ ...s, items: Array.isArray(activity?.[s.key]) ? activity[s.key] : [] }))
    .filter((s) => s.items.length > 0);
  if (!sections.length) return null;
  return (
    <div className="flex flex-col gap-5">
      {sections.map((s) => (
        <section key={s.key} aria-labelledby={`recent-${s.key}-heading`}
          className="flex flex-col gap-2">
          <p id={`recent-${s.key}-heading`}
            className="m-0 px-3 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">
            {s.label}
          </p>
          <ul className="m-0 flex list-none flex-wrap gap-1.5 p-0 px-2">
            {s.items.map((img) => {
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
                    className={`block h-11 w-11 overflow-hidden rounded-full shadow-md transition duration-200 hover:scale-110 hover:shadow-xl hover:ring-2 hover:ring-indigo-400/70 ${
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
      ))}
    </div>
  );
}

export default function RecentVariations({ onOpen, currentId = null, refreshKey = 0 }) {
  const [activity, setActivity] = useState({});
  useEffect(() => {
    let alive = true;
    // apiFetch RESOLVES TO THE PARSED BODY and throws on a bad status — it is
    // not fetch(). Treating it as a Response is what broke the first version.
    apiFetch(`/api/dataset/recent-activity?limit=${RECENT_LIMIT}`)
      .then((d) => { if (alive) setActivity(d || {}); })
      /* A shortcut strip is not worth a toast: it either shows faces or it
         shows nothing, and the page it lives on works either way. */
      .catch(() => {});
    return () => { alive = false; };
  }, [refreshKey]);

  return <RecentVariationsList activity={activity} onOpen={onOpen} currentId={currentId} />;
}
