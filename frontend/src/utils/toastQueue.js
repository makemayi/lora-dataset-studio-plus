/* 🔔 Toast queue arithmetic — merging, capping and expiry.
 *
 * WHY THIS IS A SEPARATE .js FILE
 * -------------------------------
 * Toast.jsx is JSX, which `node --test` cannot parse. Everything that can go
 * wrong here is arithmetic (does a repeat merge? does the cap hold?), so it
 * lives in a plain module the test runner can exercise directly.
 *
 * WHY MERGE AT THE QUEUE, NOT AT THE CALLER
 * -----------------------------------------
 * A repeated message is a repeated message whoever sent it. Deduplicating in
 * the ToastProvider fixes every call site at once — including the ones nobody
 * has written yet — instead of asking each caller to remember it is noisy.
 * The concrete case that motivated it: a 2 s poll failing for 20 s stacked 20
 * identical "Connection lost" banners, one on top of the other, each its own
 * assertive live region (20 screen-reader announcements) and together covering
 * the whole app on a phone.
 */

/* Hard ceiling on simultaneously visible banners. Four fits a 400 px viewport
   without burying the page; beyond that the oldest is dropped. This is the
   safety net for the case merging can't help with: many DIFFERENT messages
   firing at once (a page mount fanning out six failing requests). */
export const MAX_VISIBLE_TOASTS = 4;

/** Two toasts are "the same notification" when they say the same thing in the
 *  same tone. Nothing else — an id is per-emission and must not enter this. */
function isSame(a, b) {
  return a.type === b.type && a.message === b.message;
}

/* A merged toast must not vanish on the FIRST occurrence's timer while
   repeats are still arriving: the later expiry wins, and a sticky entry
   (expiresAt == null) stays sticky. */
function laterExpiry(a, b) {
  if (a == null || b == null) return null;
  return a > b ? a : b;
}

/**
 * Add `entry` ({ id, message, type, expiresAt }) to `list`, merging it into an
 * identical live toast (bumping `count` and refreshing the expiry) instead of
 * appending a duplicate. Returns a NEW array; never mutates `list`.
 */
export function pushToast(list, entry) {
  const existing = list.findIndex((t) => isSame(t, entry));
  if (existing >= 0) {
    const prev = list[existing];
    const next = list.slice();
    next[existing] = {
      ...prev,
      count: (prev.count || 1) + 1,
      expiresAt: laterExpiry(prev.expiresAt, entry.expiresAt),
    };
    return next;
  }
  const next = [...list, { count: 1, ...entry }];
  if (next.length <= MAX_VISIBLE_TOASTS) return next;
  // Over the cap: drop the oldest DISMISSIBLE toast. A sticky one is only
  // dropped when there is nothing else to give, so a banner that was meant to
  // stay put can never be evicted by a burst of transient chatter.
  const victim = next.findIndex((t) => t.expiresAt != null);
  const drop = victim >= 0 ? victim : 0;
  return next.filter((_, i) => i !== drop);
}

/** Remove one toast by id (the ✕ button). */
export function dropToast(list, id) {
  return list.filter((t) => t.id !== id);
}

/** Drop everything whose expiry has passed. `expiresAt == null` = sticky. */
export function sweepToasts(list, now) {
  return list.filter((t) => t.expiresAt == null || t.expiresAt > now);
}

/** What the banner actually reads: "Connection lost." → "Connection lost. (12×)" */
export function toastLabel(toast) {
  const n = toast.count || 1;
  return n > 1 ? `${toast.message} (${n}×)` : toast.message;
}
