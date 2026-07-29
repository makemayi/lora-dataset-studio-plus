/* The "here is what improve will ask for, and here is where to change it" line,
   rendered wherever ✨ Upscale & improve can be triggered (the lightbox and the
   grid's bulk toolbar). Reasoning, and the report behind it, in kleinImproveHint.js.

   ONE fetch, shared: /api/settings is a full settings payload and this note can
   mount several times per screen (every lightbox open, the bulk toolbar). The
   module-level promise below is the whole cache — with a short TTL, because a
   user who goes to Settings, edits the instruction and comes straight back must
   not be shown the text they just replaced. A failed request stays loaded:false
   and the note degrades to "there IS an instruction, it is editable" rather than
   inventing its content.

   Deliberately NOT a second editor: the instruction is already editable in
   Settings ▸ Image engines ▸ Identity & Klein prompts, with a per-subject picker
   and a Reset. A copy of that field here would be a second write path to a value
   several screens read. This note quotes and points; it never sets. */
import { useEffect, useState } from 'react';
import { apiFetch } from '../../api/fetchClient';
import SettingsLink from '../common/SettingsLink';
import KleinModelSetting from '../shared/KleinModelSetting';
import { improveInstructionLine, improveAnimeCaution, readImproveInstruction } from './kleinImproveHint';

const TTL_MS = 15000;
let cache = { at: 0, promise: null };

function loadSettings() {
  const now = Date.now();
  if (!cache.promise || now - cache.at > TTL_MS) {
    cache = {
      at: now,
      // Resolve to null on failure: the caller renders the honest "unknown"
      // wording instead of a toast about a hint nobody asked for.
      promise: apiFetch('/api/settings', { background: true }).catch(() => null),
    };
  }
  return cache.promise;
}

/** Exported for tests/dev only — drops the shared payload cache. */
export function _resetKleinImproveNoteCache() {
  cache = { at: 0, promise: null };
}

export default function KleinImproveNote({ subjectType = '', className = '', datasetId = null }) {
  const [payload, setPayload] = useState(null);

  useEffect(() => {
    let cancelled = false;
    loadSettings().then((d) => { if (!cancelled) setPayload(d); });
    return () => { cancelled = true; };
  }, []);

  const state = readImproveInstruction(payload);
  const line = improveInstructionLine(state);
  const caution = improveAnimeCaution({ ...state, subjectType });

  return (
    // min-w-0 + break-words: the quoted instruction is user text of arbitrary
    // length and this sits inside flex rows that would otherwise be widened past
    // a 400 px screen by a single long word.
    <div className={`min-w-0 space-y-1 text-[0.6875rem] leading-relaxed ${className}`}>
      <p className="text-content-subtle break-words">
        {line.text}
        {line.quote && (
          <>
            {' '}
            <span className="text-content-muted italic" title={line.full}>“{line.quote}”</span>
          </>
        )}
      </p>
      {caution && (
        <p className="text-amber-300 break-words">{caution}</p>
      )}
      {/* WHICH model executes the instruction — the other half of the same
          question, and the half nothing on this screen ever answered. Named
          even on a one-model install; the picker itself only appears when
          there is more than one thing to pick. */}
      <KleinModelSetting datasetId={datasetId} />
      {/* Two targets because they are two different problems: the WORDS
          (why it turned realistic) and the AMOUNT (how far it moved).
          flex-wrap so they stack rather than overflow on a phone. */}
      <p className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <SettingsLink section="engines" focus="identity-prompt-klein-improve">
          Edit or turn off this instruction
        </SettingsLink>
        <SettingsLink section="engines" focus="klein-improve-strength">
          Adjust improve strength
        </SettingsLink>
      </p>
    </div>
  );
}
