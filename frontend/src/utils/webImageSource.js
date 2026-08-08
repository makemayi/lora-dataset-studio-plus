const MAX_URL_CHARS = 2048;

/** Fail-closed origin link for images found through web search. */
export function webImageSource(metadata) {
  if (!metadata || typeof metadata !== 'object' || metadata.platform !== 'websearch') return null;
  const value = typeof metadata.source_url === 'string' ? metadata.source_url.trim() : '';
  if (!value || value.length > MAX_URL_CHARS || /[\u0000-\u001f]/.test(value)) return null;
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== 'https:' || parsed.username || parsed.password) return null;
    return { sourceUrl: value, host: parsed.hostname };
  } catch {
    return null;
  }
}
