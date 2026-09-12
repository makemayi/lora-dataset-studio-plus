import { RefModIcon, SpinnerIcon } from '../common/icons';
import { QUIET_BUTTON } from '../common/surfaces';

/** 🧩 Generate H3 RefMod — one click: the dataset's kept images (6 face + 6 half
 * + 4 full) VAE-encode into a MiniMax-H3 reference bundle the Load/Apply RefMod
 * nodes can use. Renders nothing on a dataset with no kept images. All labels
 * stay mounted and flip with `hidden` — see CLAUDE.md ▸ UI changes. */
export default function H3RefModButton({ onClick, busy = false, disabled = false, count = 0, stage = null }) {
  if (!count) return null;

  return (
    <button type="button" data-workspace-focus onClick={onClick}
      disabled={busy || disabled}
      title="Encode the kept images into a MiniMax-H3 RefMod (.safetensors) for the Load / Apply RefMod nodes"
      className={`${QUIET_BUTTON} self-start`}>
      <span hidden={busy} data-label="idle" className="inline-flex items-center gap-1.5">
        <RefModIcon className="h-3.5 w-3.5 shrink-0" /> Generate H3 RefMod
      </span>
      <span hidden={!busy} data-label="busy" className="inline-flex items-center gap-1.5">
        <SpinnerIcon className="h-3.5 w-3.5 shrink-0 animate-spin" />
        <span hidden={!!stage} data-label="busy-generic">Encoding RefMod…</span>
        <span hidden={!stage} data-label="busy-stage">{stage || ''}</span>
      </span>
    </button>
  );
}
