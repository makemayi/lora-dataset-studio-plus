import PexelsAttribution from './PexelsAttribution';
import WebImageSource from './WebImageSource';

/**
 * Provenance is recorded for a GROWING list of scraper sources (Pexels today,
 * web search, and more platforms already tracked as debt). Each platform keeps
 * its own fail-closed renderer — the wording and links a photo credit needs are
 * not the wording and links a "found on this page" note needs — but every
 * render site only wants ONE tile: "show whatever provenance this image has,
 * or nothing". Stacking a new `<XAttribution>` at every render site (grid tile,
 * lightbox, …) for each new platform is exactly what would not scale past a
 * couple of sources; this is the single dispatch point instead.
 *
 * Adding platform #3 is one new line in PLATFORM_RENDERERS plus its own small
 * component — never a change at a render site.
 */
const PLATFORM_RENDERERS = {
  pexels: PexelsAttribution,
  websearch: WebImageSource,
};

export default function SourceAttribution({ metadata, className = '' }) {
  const platform = metadata && typeof metadata === 'object' ? metadata.platform : null;
  const Renderer = PLATFORM_RENDERERS[platform];
  // Fail-closed on two independent axes: an unrecognized/missing platform never
  // reaches a renderer, and the renderer itself re-validates the whole payload
  // (its own allowlisted hosts, schemes, lengths) before it will render anything
  // — a `platform: 'pexels'` tag alone is not proof the rest of the object is safe.
  if (!Renderer) return null;
  return <Renderer metadata={metadata} className={className} />;
}
