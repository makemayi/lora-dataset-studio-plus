// react-frontend/src/components/dataset/studio/enhanceGate.js
/**
 * Décide si le bouton « ✨ Enhance » est utilisable, et POURQUOI il ne l'est pas.
 *
 * Extrait du JSX pour être testé sous `node --test`. La règle : sur une install
 * SANS moteur visuel local le bouton est désactivé avec la raison exacte en
 * infobulle — jamais un appel qui part et échoue en silence. Les états viennent
 * de /api/capabilities (`caps.vision` — le moteur ACTIF, ollama ou llamacpp, dans
 * la même forme de champs que l'ancien `caps.ollama`).
 */

/** Raison de blocage (string, anglais) ou null si Enhance est utilisable. */
export function enhanceBlocker(vision, { capsLoading = false } = {}) {
  if (capsLoading) return 'Checking local tools…';
  const o = vision || {};
  // The ACTIVE engine names itself, so the fix named in the message is the real
  // one: llama.cpp is started by the user (nothing to install, nothing to pull),
  // Ollama is installed/started/pulled from Settings › Local tools.
  const llamacpp = o.backend === 'llamacpp';
  if (!o.installed && !o.reachable) {
    return llamacpp
      ? 'Enhance needs the llama.cpp server — nothing is answering at its URL. Check Settings › Local tools.'
      : 'Enhance needs Ollama — install it from Settings › Local tools.';
  }
  if (!o.reachable) {
    return llamacpp
      ? 'The llama.cpp server is not answering — start llama-server yourself (its URL is in Settings › Local tools).'
      : 'Ollama is installed but not running — start it from Settings › Local tools.';
  }
  if (!o.vision_model_ready) {
    return llamacpp
      ? `llama.cpp model "${o.vision_model || 'unset'}" is not loaded — load it in llama-server, or clear the model field to use whatever the server has.`
      : `Ollama model "${o.vision_model || 'unset'}" is not downloaded yet `
        + '— pull it from Settings › Local tools.';
  }
  return null;
}
