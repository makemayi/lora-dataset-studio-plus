// react-frontend/src/components/dataset/studio/EnhancePromptButton.jsx
/**
 * « ✨ Enhance » : passe le prompt tapé au modèle Ollama LOCAL (le même que le
 * captioning — pas de second client) pour l'enrichir avant génération.
 *
 * Échec PROPRE sur une install sans Ollama : le bouton est DÉSACTIVÉ, avec en
 * infobulle la raison exacte (Ollama pas installé / pas joignable / modèle pas
 * téléchargé) — jamais un appel qui part et revient en erreur opaque. La source de
 * vérité est /api/capabilities (caps.ollama), déjà consommée par les surfaces Bank
 * et Settings : aucun probe supplémentaire n'est ajouté ici.
 */
import { useState } from 'react';
import { postJson } from '../../../api/fetchClient';
import { useCapabilities } from '../../../context/CapabilitiesContext';
import { useToast } from '../../common/Toast';
import { enhanceBlocker } from './enhanceGate';

export default function EnhancePromptButton({ prompt, onResult, className = '' }) {
  const toast = useToast();
  const { caps, loading } = useCapabilities();
  const [busy, setBusy] = useState(false);
  const blocked = enhanceBlocker(caps?.ollama, { capsLoading: loading });
  const empty = !((prompt || '').trim());
  const title = blocked
    || (empty ? 'Write a prompt first' : 'Enrich this prompt with the local model (Ollama)');

  const run = async () => {
    if (blocked || empty || busy) return;
    setBusy(true);
    try {
      const d = await postJson('/api/studio/enhance-prompt', { prompt });
      if (d?.ok && d.prompt) onResult(d.prompt);
      else toast.error(d?.error || 'The model returned nothing');
    } catch (e) {
      toast.error(e?.message || 'Enhance failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <button type="button" onClick={run} disabled={!!blocked || empty || busy} title={title}
      aria-label="Enhance the prompt with the local model"
      className={`px-2 py-0.5 rounded border border-border bg-surface text-content-subtle text-[0.625rem] hover:text-content disabled:opacity-40 disabled:cursor-not-allowed ${className}`}>
      {busy ? '✨ …' : '✨ Enhance'}
    </button>
  );
}
