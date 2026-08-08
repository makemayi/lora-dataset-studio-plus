/* Shared visual vocabulary for a lineage run — used by BOTH the ☰ List
   (RunLineageTree) and the ◉ Graph (RunLineageGraph) views so a run reads the
   same in either: its family label, its status dot, its on-disk/gone chip. Kept
   in one place so the two views can never drift apart. */

// The family names live in a JSX-free module so the canvas's "you mixed Krea and
// Z-Image" refusal can name them in a unit-testable helper. Re-exported here so
// every existing import keeps working.
export { FAMILY_LABEL, famLabel } from '../../utils/familyLabels';

const STATUS_TONE = {
  done: 'bg-emerald-400',
  error: 'bg-rose-400',
  error_pod_kept: 'bg-amber-400',
};

/** Run status as a small dot; a finished run gets a soft emerald halo. */
export function StatusDot({ status }) {
  const tone = STATUS_TONE[status] || (status ? 'bg-sky-400' : 'bg-content-subtle');
  return (
    <span aria-hidden title={status || 'no recorded status'}
      className={`h-2 w-2 shrink-0 rounded-full ${tone} ${status === 'done' ? 'shadow-[0_0_6px] shadow-emerald-400/50' : ''}`} />
  );
}

/** "full model" for a run that trained the whole transformer, nothing for a LoRA.
 *
 *  Both kinds printed the SAME two words — "Krea 2 · Raw" — while being entirely
 *  different objects: one produces a ~26 GB checkpoint you load INSTEAD of the
 *  base, the other a small adapter you load ON TOP of it. On a board holding
 *  both, nothing on the card said which was which. Lives here so the ☰ List and
 *  the ◉ Graph cannot drift apart, like every other badge in this file. */
export function ModeChip({ node }) {
  if (node?.training_mode !== 'full_transformer') return null;
  return (
    <span title="This run trained the whole model, not a LoRA adapter"
      className="rounded bg-sky-500/20 px-1 py-px font-semibold text-sky-200">
      full model
    </span>
  );
}

/** LoRA/checkpoint availability chip: on-disk vs gone (superseded aside or
 *  deleted). A null availability (a scan we couldn't run) shows nothing.
 *
 *  A FULL MODEL answers first, because it has an address this chip never knew
 *  about. Its weights are a 26 GB file on this disk OR a private Hugging Face
 *  repository — and a run delivered to the Hub only has nothing in the
 *  checkpoint columns, which this chip used to read as "gone" for a model that
 *  is very much alive. `dense_artifact` is the backend's own answer for it. */
export function SavesChip({ node }) {
  if (node.dense_artifact === 'hub') {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-sky-400/40 bg-sky-500/10 px-1.5 py-0.5 text-sky-200 text-[0.5625rem] font-medium"
        title="The full model is in this run's private Hugging Face repository, not on this computer">
        <span aria-hidden>☁</span>on Hugging Face
      </span>
    );
  }
  if (node.dense_artifact === 'local') {
    const n = node.saves;
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-emerald-400/40 bg-emerald-500/10 px-1.5 py-0.5 text-emerald-200 text-[0.5625rem] font-medium"
        title={`The full model is on this computer${n > 1 ? ` (${n} files, master + fp8 twin)` : ''}`}>
        <span aria-hidden>💾</span>full model here
      </span>
    );
  }
  if (node.checkpoint_ready === true) {
    const n = node.saves;
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-emerald-400/40 bg-emerald-500/10 px-1.5 py-0.5 text-emerald-200 text-[0.5625rem] font-medium"
        title={n ? `${n} checkpoint${n > 1 ? 's' : ''} still on disk` : 'LoRA on disk'}>
        <span aria-hidden>💾</span>{n ? `${n} on disk` : 'on disk'}
      </span>
    );
  }
  if (node.checkpoint_ready === false) {
    return (
      <span className="inline-flex items-center rounded-full border border-border px-1.5 py-0.5 text-content-subtle text-[0.5625rem] font-medium"
        title="This run's checkpoint is no longer on disk (set aside by a later resume, or deleted)">
        gone
      </span>
    );
  }
  return null;
}
