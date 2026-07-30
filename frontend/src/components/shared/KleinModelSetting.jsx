/* The Klein model this dataset runs on: named always, chosen when there is a
   choice. Rendered next to ✨ Upscale & improve (lightbox + bulk toolbar) and in
   the Klein tuning block of the generation panel — ONE setting, one control,
   because both drive the same UNETLoader in the same graph and two near-identical
   dropdowns side by side is a durable confusion.

   The list is NOT scanned here: it comes from the capabilities probe that already
   walks every folder the resolver accepts (models/unet, models/diffusion_models,
   extra_model_paths roots, a relocated models_dir, klein-named subfolder or bare
   root file). A second scanner would drift from the first one — the backend test
   test_improve_klein_model_choice asserts the two ends layout by layout.

   WITHOUT a datasetId it becomes the naming half only: it reads the global
   /api/klein-model and states which model will run, with no <select>. That is the
   bank's watermark inpaint — a bank has no dataset to inherit a pick from, and
   adding a picker there would create a second place to choose a Klein model for
   the same UNETLoader. Naming and choosing are separate questions, and the naming
   one is the half that works on every screen (backend counterpart: the comment at
   image_bank_service's Klein call site).

   Wording and every conditional live in kleinModelChoice.js (pure JS) so
   `node --test` can cover them without a JSX parser. */
import { useCallback, useEffect, useState } from 'react';
import { apiFetch, postJson } from '../../api/fetchClient';
import {
  canChooseModel, legacyNotice, modelEndpoint, modelLine, readLegacyPick, selectValue,
} from './kleinModelChoice';

export default function KleinModelSetting({ datasetId = null, className = '', onChange }) {
  const [state, setState] = useState({ loaded: false, stored: null, effective: null,
    missing: null, choices: [] });
  const [legacy, setLegacy] = useState('');
  const [dismissed, setDismissed] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLegacy(datasetId
      ? readLegacyPick(typeof localStorage === 'undefined' ? null : localStorage)
      : '');
    // background: a model list is never worth a toast — the line degrades to
    // "Runs on the Klein model Studio detects" and the picker simply stays away.
    apiFetch(modelEndpoint(datasetId), { background: true })
      .then((d) => {
        if (cancelled || !d || !d.ok) return;
        setState({ loaded: true, stored: d.stored || null, effective: d.effective || null,
          missing: d.missing || null, choices: d.choices || [] });
        onChange?.(d.stored || null);
      })
      .catch(() => { /* stays loaded:false — see modelLine */ });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId]);

  const save = useCallback(async (value) => {
    setSaving(true);
    const d = await postJson(`/api/dataset/${datasetId}/klein-model`,
      { klein_model: value || '' });
    setSaving(false);
    if (!d || !d.ok) return;
    setState({ loaded: true, stored: d.stored || null, effective: d.effective || null,
      missing: d.missing || null, choices: d.choices || [] });
    onChange?.(d.stored || null);
  }, [datasetId, onChange]);

  const line = modelLine(state);
  const notice = dismissed ? null : legacyNotice({ ...state, legacy });
  const canChoose = canChooseModel({ datasetId, choices: state.choices });

  return (
    <div className={`min-w-0 space-y-1 text-[0.6875rem] leading-relaxed ${className}`}>
      {/* break-words: a model file name is arbitrary user content and this sits
          in flex rows that a single long word would widen past a 400 px screen. */}
      <p className={`break-words ${line.tone === 'warn' ? 'text-amber-300' : 'text-content-subtle'}`}>
        {line.text}
      </p>
      {canChoose && (
        // flex-wrap + min-w-0: label and select stack rather than overflow at 400 px.
        <label className="flex flex-wrap items-center gap-x-2 gap-y-1 min-w-0">
          <span className="text-content-muted">Klein model</span>
          <select
            aria-label="Klein model for this dataset"
            disabled={saving}
            value={selectValue(state)}
            onChange={(e) => save(e.target.value)}
            className="min-w-0 max-w-full flex-1 bg-white/[0.03] border border-white/10 rounded-md
                       px-2 py-1 text-[0.6875rem] text-content focus:outline-none
                       focus:border-primary/60 disabled:opacity-50"
          >
            <option value="" className="bg-surface-overlay">Auto (detected)</option>
            {state.choices.map((m) => (
              <option key={m} value={m} className="bg-surface-overlay">{m}</option>
            ))}
          </select>
        </label>
      )}
      {notice && (
        <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-content-subtle break-words">
          <span>{notice.text}</span>
          <button type="button" disabled={saving} onClick={() => save(notice.value)}
            className="underline text-content-muted hover:text-content disabled:opacity-40">
            Save {notice.value}
          </button>
          <button type="button" onClick={() => setDismissed(true)}
            className="underline text-content-subtle hover:text-content">Keep auto</button>
        </p>
      )}
    </div>
  );
}
