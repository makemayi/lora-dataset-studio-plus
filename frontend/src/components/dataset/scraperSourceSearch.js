const PEXELS_AUTH_STORAGE_KEY = 'lds:pexels-dataset-authorization:v1';

const PEXELS_ORIENTATIONS = new Set(['portrait', 'landscape', 'square']);

const storageFor = (storage) => {
  if (storage !== undefined) return storage;
  try { return globalThis.localStorage || null; } catch { return null; }
};

export function normalizePexelsKeyword(value) {
  return String(value || '')
    .replace(/[\\/]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function normalizePexelsLocale(value) {
  return value === 'en-US' ? 'en-US' : 'fr-FR';
}

export function normalizePexelsOrientation(value) {
  return PEXELS_ORIENTATIONS.has(value) ? value : '';
}

export function buildPexelsSearchUrl(keyword, locale = 'fr-FR', orientation = '') {
  const query = normalizePexelsKeyword(keyword);
  if (!query) return '';
  const localizedRoute = normalizePexelsLocale(locale) === 'en-US'
    ? 'en-us/search'
    : 'fr-fr/chercher';
  const normalizedOrientation = normalizePexelsOrientation(orientation);
  const suffix = normalizedOrientation
    ? `?orientation=${encodeURIComponent(normalizedOrientation)}`
    : '';
  return `https://www.pexels.com/${localizedRoute}/${encodeURIComponent(query)}/${suffix}`;
}

export function isPexelsUrl(value) {
  try {
    const parsed = new URL(String(value || '').trim());
    return (parsed.protocol === 'https:' || parsed.protocol === 'http:')
      && (parsed.hostname === 'pexels.com' || parsed.hostname === 'www.pexels.com');
  } catch { return false; }
}

/**
 * A page-zero scan uses the current form submission. Later pages are pinned to
 * the last successful scan, so editing another form can never retarget Load more.
 */
export function resolveScanTarget({ nextPage, explicitUrl, draftUrl, activeScanUrl }) {
  const candidate = nextPage > 0 ? activeScanUrl : (explicitUrl ?? draftUrl);
  return typeof candidate === 'string' ? candidate.trim() : '';
}

// DuckDuckGo's own SafeSearch flag. The scan API accepts a URL and nothing else,
// so the setting travels inside the URL and the backend source reads it back —
// no request field is added and the contract is unchanged.
const WEB_SEARCH_SAFE_OFF = '-2';
const WEB_SEARCH_SAFE_STRICT = '1';

export function normalizeWebSearchKeyword(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

export function buildWebSearchUrl(keyword, safe = false) {
  const query = normalizeWebSearchKeyword(keyword);
  if (!query) return '';
  const params = new URLSearchParams({
    q: query, iax: 'images', ia: 'images',
    kp: safe ? WEB_SEARCH_SAFE_STRICT : WEB_SEARCH_SAFE_OFF,
  });
  return `https://duckduckgo.com/?${params.toString()}`;
}

/**
 * Scan item → the payload posted to scrape-import. Pexels items carry
 * attribution fields websearch items don't have (photographer, photographer_url);
 * websearch items carry the page they were found on (source_url) instead. Both
 * need `platform` forwarded or the backend has nothing to key provenance on.
 */
export function scrapeItemToImportPayload(it) {
  const base = { url: it.url, title: it.title || '' };
  if (it.platform === 'pexels') {
    return { ...base, platform: 'pexels', source_url: it.source_url,
      photographer: it.photographer, photographer_url: it.photographer_url };
  }
  if (it.platform === 'websearch') {
    return { ...base, platform: 'websearch', source_url: it.source_url };
  }
  return base;
}

export function loadPexelsAuthorization(storage) {
  const target = storageFor(storage);
  if (!target) return false;
  try { return target.getItem(PEXELS_AUTH_STORAGE_KEY) === 'true'; } catch { return false; }
}

export function savePexelsAuthorization(confirmed, storage) {
  const target = storageFor(storage);
  if (!target) return;
  try {
    if (confirmed) target.setItem(PEXELS_AUTH_STORAGE_KEY, 'true');
    else target.removeItem(PEXELS_AUTH_STORAGE_KEY);
  } catch { /* private mode/storage policy must not break the form */ }
}
