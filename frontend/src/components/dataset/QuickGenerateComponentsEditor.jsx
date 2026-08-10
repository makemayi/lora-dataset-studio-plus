/** Advanced, collapsible JSON editor for a subject's quick-generate custom
 *  components (`quick_generate.custom_components[subjectType]`) — sits next
 *  to VariationCatalog's shot-catalog import UI, same collapsed-by-default
 *  <details> convention.
 *
 *  The PUT route (`dataset_quick_generate_components_save`, backend) REPLACES
 *  a subject's whole custom-component tree on every save, it does not merge.
 *  So the initial load MUST show what's already saved — reading `d.custom`,
 *  the raw round-trip view the GET route now returns for this subject only
 *  (not merged with the shipped pool) — never a blind '{}'. Loading a blind
 *  '{}' here would mean: user saves some entries, reopens later, sees an
 *  empty editor, adds one more entry, hits Save — and everything they saved
 *  before is silently gone. */
import { useEffect, useState } from 'react';
import { useToast } from '../common/Toast';

export default function QuickGenerateComponentsEditor({
  subjectType, quickGenerateComponents, saveQuickGenerateCustomComponents,
}) {
  const toast = useToast();
  const [text, setText] = useState('{}');
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoaded(false);
    (async () => {
      const d = await quickGenerateComponents(subjectType);
      if (!cancelled && d) {
        // d.custom — the user's own additions only, never the shipped pool
        // (those live in d.pools). This is what makes reopening safe.
        setText(JSON.stringify(d.custom || {}, null, 2));
        setLoaded(true);
      }
    })();
    return () => { cancelled = true; };
  }, [subjectType, quickGenerateComponents]);

  const save = async () => {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      toast.error('Invalid JSON — fix the syntax before saving');
      return;
    }
    // saveQuickGenerateCustomComponents already toasts success/error itself.
    const result = await saveQuickGenerateCustomComponents(subjectType, parsed);
    // Reflect what actually landed: if any entries were dropped (a shadowing
    // id, a malformed framing/axis), the textarea should show the canonical
    // saved result — not the user's original input — so the drop is visible
    // immediately instead of only after closing and reopening the editor.
    if (result) setText(JSON.stringify(result.saved || {}, null, 2));
  };

  if (!loaded) return null;
  return (
    <div className="flex flex-col gap-1.5 mt-2">
      <p className="text-content-muted text-[0.6875rem]">
        Add your own angle/expression/pose/outfit/background entries per
        framing — shape: {'{'}"face": {'{'}"expression": [{'{'}"id": "...",
        "phrase": "..."{'}'}]{'}'}{'}'}. An id matching a built-in one is
        dropped, never overrides it. This replaces your ENTIRE saved set for
        this subject type each time you save — it's pre-loaded with what you
        already have, so edit in place rather than starting from scratch.
      </p>
      <textarea value={text} onChange={(e) => setText(e.target.value)}
        rows={8} spellCheck={false} aria-label="Quick-generate custom components JSON"
        className="bg-surface-raised rounded px-2 py-1 text-content text-[0.6875rem] font-mono resize-y" />
      <button type="button" onClick={save}
        className="self-end px-2.5 py-1 rounded-full bg-surface-raised text-content text-[0.6875rem] font-semibold hover:bg-surface-raised">
        Save custom components
      </button>
    </div>
  );
}
