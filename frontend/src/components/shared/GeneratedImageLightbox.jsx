import { useCallback, useEffect, useRef, useState } from 'react';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { useImageDownload } from '../../hooks/useImageDownload';
import {
  imageHeadlineFacts, imagePromptBlocks, imageSettingFacts, promptFold,
} from '../../utils/generatedImageFacts';

/* 🔍 ONE generated image, large, with what it was made from.

   It replaces two things: the zoom that lived inline in CheckpointGalleryPanel,
   and PreviewLightbox (which now renders this with the little it knows). The
   app already carries DatasetLightbox and BankReviewLightbox and neither could
   be reused here — DatasetLightbox is about a DATASET image (crop, mirror,
   Klein improve, Pexels attribution, a `filename` on a dataset route) and
   BankReviewLightbox is a triage queue with keep/reject/skip and its own
   session. Both are the wrong nouns and the wrong verbs. So this is not a
   fourth viewer: it is the third, and it absorbs the inline one that made four.

   ⚠️ What was wrong with the old one, because it is the whole reason this file
   exists. Under the image sat a single paragraph:

     step 2500 · seed 208607443 · strength 0 · <forty lines of prompt>

   — the three facts you are actually looking for buried at the head of a wall of
   text, set across the full width of a 2 000-px screen. Three fixes, all
   structural:

     1. HIERARCHY. The facts are not the same kind of thing as the prompt. Step,
        seed and strength are chips; the settings that decided the picture are a
        table; the prose is last.
     2. BOUNDED READING WIDTH. Above `lg` the metadata is a column beside the
        image, not a line under it, so it can never run wider than a paragraph.
     3. THE PROMPT FOLDS. Long by nature, it opens collapsed past a threshold
        and scrolls rather than pushing everything else off the screen.

   And the seed and the prompt COPY in one click, because that is what those two
   values are for — you re-play a seed, you re-use a prompt.

   Everything the panel decides (which facts, in which order, what folds) is in
   utils/generatedImageFacts.js, where `node --test` can reach it. */

/** A copy button that says it worked. Clipboard access fails on a plain-http
 *  remote origin; the value stays selectable, so the failure is a button that
 *  does nothing visible rather than an error the user cannot act on. */
function CopyButton({ value, label, className = '' }) {
  const [done, setDone] = useState(false);
  const timer = useRef(null);
  useEffect(() => () => clearTimeout(timer.current), []);
  const copy = useCallback(async (e) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(String(value));
      setDone(true);
      clearTimeout(timer.current);
      timer.current = setTimeout(() => setDone(false), 1400);
    } catch { /* clipboard blocked — the text is selectable */ }
  }, [value]);
  return (
    <button type="button" onClick={copy} title={`Copy ${label}`}
      aria-label={done ? `${label} copied` : `Copy ${label}`}
      className={'shrink-0 rounded border border-white/25 px-1.5 py-0.5 text-[0.625rem] '
        + 'text-white/70 hover:border-white/50 hover:text-white ' + className}>
      {done ? '✓ Copied' : '⧉ Copy'}
    </button>
  );
}

/** One prose block — prompt or negative — bounded, foldable, copyable. */
function PromptBlock({ block }) {
  const [expanded, setExpanded] = useState(false);
  const fold = promptFold(block.text, expanded);
  return (
    <section className="mt-3 border-t border-white/10 pt-2">
      <div className="mb-1 flex items-center gap-2">
        <h4 className="m-0 text-[0.6875rem] font-semibold text-white/80">{block.label}</h4>
        <CopyButton value={block.text} label={block.label.toLowerCase()} className="ml-auto" />
      </div>
      <p className={'m-0 whitespace-pre-wrap break-words text-[0.75rem] leading-relaxed text-white/70 '
        + (fold.collapsed
          // A fixed clamp, not a scroll: collapsed means "you can see there is
          // more", and a scrollbar inside a collapsed block invites scrolling a
          // thing that is meant to be opened.
          ? 'line-clamp-4 '
          // Expanded, it scrolls INSIDE its own box. A forty-line prompt must
          // not push the settings — or the image — off the screen.
          : 'max-h-[9rem] overflow-y-auto ')}>
        {block.text}
      </p>
      {fold.foldable && (
        <button type="button" onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          className="mt-1 rounded text-[0.6875rem] text-indigo-300 underline decoration-dotted hover:text-indigo-200">
          {fold.label}
        </button>
      )}
    </section>
  );
}

/**
 * `img` is a gallery image row (see services.cloud_training._gallery_image).
 * `alt` names it. `actions` is an optional node rendered in the metadata
 * column's footer — the canvas puts its 📌 Pin button there.
 */
export default function GeneratedImageLightbox({ img, alt, actions = null, onClose }) {
  const dialogRef = useRef(null);
  const closeRef = useRef(null);
  const dl = useImageDownload();
  useFocusTrap(dialogRef, !!img);
  useEffect(() => {
    if (!img) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') onClose?.(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [img, onClose]);
  useEffect(() => { if (img) closeRef.current?.focus(); }, [img]);

  if (!img) return null;
  const head = imageHeadlineFacts(img);
  const settings = imageSettingFacts(img);
  const blocks = imagePromptBlocks(img);
  const label = alt || 'Generated image';

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label={label}
      data-testid="generated-image-lightbox"
      onClick={onClose}
      // ⚠️ /95, not an arbitrary /92: Tailwind only emits the opacities in its
      // scale, so bg-black/92 compiled to NOTHING and the board behind stayed at
      // full brightness under what was supposed to be a backdrop.
      className="fixed inset-0 z-[9997] flex flex-col bg-black/95 lg:flex-row">
      <button type="button" ref={closeRef}
        onClick={(e) => { e.stopPropagation(); onClose?.(); }}
        title="Close (Esc)" aria-label="Close image"
        className="absolute right-3 top-3 z-10 flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-lg leading-none text-white hover:bg-white/20">✕</button>

      <div className="flex min-h-0 flex-1 items-center justify-center p-3">
        <img src={img.url} alt={label} onClick={(e) => e.stopPropagation()}
          className="max-h-full max-w-full select-none rounded-lg object-contain shadow-2xl" />
      </div>

      {/* THE fix for the wall of text: a column, not a line. `max-w-md` above
          `lg` and the panel's own width below it — either way the prose has a
          reading width, never the width of the screen. Its own scroll, so the
          image never shrinks to make room for a long prompt. */}
      <aside onClick={(e) => e.stopPropagation()}
        data-testid="generated-image-facts"
        /* OPAQUE, not a tint. At 60 % the page behind it stayed legible through
           the panel — a settings table you read the board through is a table you
           misread. The backdrop can be translucent; the thing you are reading
           cannot. */
        className="max-h-[45vh] w-full shrink-0 overflow-y-auto border-t border-white/10 bg-app px-3 py-2.5
                   lg:max-h-none lg:w-[22rem] lg:border-l lg:border-t-0 lg:px-4 lg:py-10">
        <div className="mx-auto max-w-md">
          {/* The three answers to "what am I looking at", as chips. Big, tabular,
              and the seed carries its own copy — a seed is a thing you re-play. */}
          <div className="flex flex-wrap items-center gap-1.5">
            {head.map((f) => (
              <span key={f.key} data-testid={`fact-${f.key}`}
                className="flex items-center gap-1 rounded-md border border-white/15 bg-white/5 px-2 py-1">
                <span className="text-[0.5625rem] uppercase tracking-wide text-white/45">{f.label}</span>
                <span className="text-[0.8125rem] font-semibold tabular-nums text-white">{f.value}</span>
                {f.copy && <CopyButton value={f.copy} label="seed" />}
              </span>
            ))}
          </div>

          {settings.length > 0 && (
            <section className="mt-3 border-t border-white/10 pt-2">
              <h4 className="m-0 mb-1 text-[0.6875rem] font-semibold text-white/80">
                <span aria-hidden>⚙</span> Made with
              </h4>
              {/* A grid, not a sentence: these are looked UP, one at a time,
                  while comparing two renders. Absent settings produce no row —
                  see imageSettingFacts. */}
              <dl className="m-0 grid grid-cols-[minmax(0,auto)_minmax(0,1fr)] gap-x-3 gap-y-0.5">
                {settings.map((r) => (
                  <div key={r.key} className="contents">
                    <dt className="m-0 text-[0.6875rem] text-white/45">{r.label}</dt>
                    <dd className="m-0 break-words text-[0.6875rem] tabular-nums text-white/80">{r.value}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}

          {blocks.map((b) => <PromptBlock key={b.key} block={b} />)}

          {/* ⬇ Keep it. In the SAME footer as 📌 Pin, because they are the two
              things you do once you have decided a render is good — and the
              file lands under a name that still says which dataset, run, step
              and seed made it (services/gallery_download.py). Without that name
              a saved render is anonymous within the week, which on a screen
              whose whole job is telling checkpoints apart is the whole loss.

              A refusal is shown here rather than swallowed: the gallery does
              list rows whose file a resume or a trash sweep has already taken
              off the disk. */}
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-white/10 pt-2.5">
            <button type="button" data-testid="lightbox-download"
              disabled={dl.busy || img.id == null}
              onClick={(e) => { e.stopPropagation(); dl.download(img.id); }}
              title="Download this image — the file name keeps its dataset, run, step and seed"
              className="rounded-md border border-white/25 px-3 py-1.5 text-[0.75rem] font-semibold text-white/85 hover:border-white/50 hover:text-white disabled:opacity-40">
              <span aria-hidden>⬇</span> {dl.busy ? 'Downloading…' : 'Download'}
            </button>
            {actions}
          </div>
          {dl.error && (
            <p role="alert" data-testid="lightbox-download-error"
              className="m-0 mt-1.5 rounded-md border border-amber-400/40 bg-amber-500/10 px-2 py-1 text-[0.6875rem] text-amber-100">
              {dl.error}
            </p>
          )}
        </div>
      </aside>
    </div>
  );
}
