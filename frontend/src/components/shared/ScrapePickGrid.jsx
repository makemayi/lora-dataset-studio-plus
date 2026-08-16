/**
 * The pick-before-you-download grid, shared by every scrape intake.
 *
 * WHY IT IS SHARED
 * ----------------
 * It started inside ConceptSourcesPanel as the scanner's own result grid. When
 * the paste intake needed the same thing — see the images, tick the ones you
 * want, then download — the choice was to copy thirty lines of tile markup or
 * to lift them here. A copy is how the two drift: one grows the dead-link
 * handling, the other keeps a stale selection ring, and a later change to
 * either has to be found twice.
 *
 * WHAT IT DOES NOT OWN
 * --------------------
 * Selection and dead-link state stay with the caller. The scanner persists its
 * selection across a reload and filters dead items before paging; the paste
 * panel throws both away when the text changes. Pushing that state in here
 * would force one of those behaviours on the other.
 *
 * Thumbnails go through `/api/scrape/thumb`, which is also what makes this work
 * at all for signed CDN links: the browser cannot fetch them cross-origin, but
 * the server can, and it already has the SSRF guard the rest of the intake uses.
 */

const thumbFor = (it) =>
  `/api/scrape/thumb?url=${encodeURIComponent(it.thumbnail || it.url)}`

export default function ScrapePickGrid({
  items,
  selected,
  onToggle,
  onBroken,
  tile = 96,
  labelFor,
  renderCaption,
  ariaLabel = 'Scraped images',
}) {
  const list = items || []
  return (
    <div className="grid gap-1.5 overflow-y-auto max-h-[34rem] pr-1"
      aria-label={ariaLabel}
      style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${tile}px, 1fr))` }}>
      {list.map((it) => {
        const on = selected?.has(it.url)
        const imageLabel = (labelFor && labelFor(it)) || it.title || 'scraped image'
        return (
          <div key={it.url} className="min-w-0">
            <button type="button" onClick={() => onToggle?.(it.url)}
              aria-pressed={!!on}
              aria-label={`${on ? 'Deselect' : 'Select'} ${imageLabel}`}
              title={imageLabel}
              className={`relative aspect-square w-full rounded-lg overflow-hidden border-2 transition-all
                ${on ? 'border-indigo-400' : 'border-transparent hover:border-border-strong'}`}>
              {/* A thumbnail that 404s is the cheapest possible liveness probe:
                  a dead gallery, or here an expired signature, is invisible in
                  the URL and obvious in the image. */}
              <img src={thumbFor(it)} alt="" loading="lazy"
                onError={() => onBroken?.(it.url)}
                className="w-full h-full object-cover" />
              <span aria-hidden
                className={`absolute top-1 right-1 w-4 h-4 rounded-full text-[0.625rem] leading-4 text-center font-bold
                  ${on ? 'bg-indigo-500 text-white' : 'bg-black/50 text-white/70'}`}>
                {on ? '✓' : ''}
              </span>
            </button>
            {renderCaption ? renderCaption(it) : null}
          </div>
        )
      })}
    </div>
  )
}
