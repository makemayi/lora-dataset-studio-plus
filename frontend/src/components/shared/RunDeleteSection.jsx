import { useEffect, useRef, useState } from 'react';
import { apiFetch, del } from '../../api/fetchClient';
import { cascadeConfirmation, cascadeResultMessage } from '../../utils/runCascadeDeletion';
import { useToast } from '../common/Toast';

/* ⚠ "Delete this run and everything it produced" — the one irreversible action
   of the run panel, kept in its own file for two reasons.

   The first is merge hygiene: the gallery panel it hangs under is a busy shared
   component (the canvas and the dataset graph both host it) and this feature
   only needs three lines there.

   The second is the safety argument, which is half the work here. The gesture
   fires from a board where cards get dragged all day, so:
     • the button sits at the BOTTOM of the scrolled body, under a rule and its
       own "Danger zone" label — the far end of the panel from the pinned action
       bar, where Select / 🗑 Delete-images live. The two destructive verbs are
       never one thumb-slide apart, and this one is never reached by accident;
     • it is outlined, not filled: destructive register, but the quietest button
       on the panel — the loud one is the action you take every day;
     • the confirmation COUNTS what disappears ("14 checkpoints · 24.0 GB, 37
       images") from a backend preview, and names what survives. A user must be
       able to back out by reading, which a generic "Are you sure?" never allows;
     • the dialog is a real modal: focus trapped, Escape cancels, Cancel is
       focused first so the destructive button is never one Enter away;
     • a run that is TRAINING says so and cannot be armed at all.

   A failure never reads as a success. The backend answers 409 for a partial
   deletion (some weights could not be moved, the run row kept) and that message
   is shown as-is — it is already path-redacted server side. */
export default function RunDeleteSection({ recordId, datasetId, onDeleted, onClose }) {
  const [impact, setImpact] = useState(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const toast = useToast();
  const dialogRef = useRef(null);
  const cancelRef = useRef(null);
  const openerRef = useRef(null);

  // The counted preview, loaded when the run opens. A failure leaves it null and
  // the dialog says "could not be counted" rather than inventing numbers.
  useEffect(() => {
    setImpact(null);
    setOpen(false);
    setError(null);
    if (!recordId) return undefined;
    let alive = true;
    apiFetch(`/api/dataset/train/runs/${recordId}/deletion-impact`)
      .then((d) => { if (alive) setImpact(d); })
      .catch(() => { if (alive) setImpact(null); });
    return () => { alive = false; };
  }, [recordId]);

  // Escape cancels, and focus is trapped inside the box while it is open. Both
  // are set up here rather than reused from useFocusTrap's effect ordering so
  // Cancel — not the destructive button — is what receives focus first.
  useEffect(() => {
    if (!open) return undefined;
    cancelRef.current?.focus();
    const onKey = (e) => {
      if (e.key === 'Escape') {
        // stopPropagation: the gallery panel above also listens for Escape (its
        // zoom lightbox); cancelling a confirmation must not close two things.
        e.stopPropagation();
        setOpen(false);
        // Back to the button that opened it — a keyboard user who cancels lands
        // where they were, not at the top of the document.
        openerRef.current?.focus();
        return;
      }
      if (e.key !== 'Tab') return;
      const box = dialogRef.current;
      if (!box) return;
      const els = [...box.querySelectorAll('button:not([disabled])')];
      if (!els.length) return;
      const first = els[0];
      const last = els[els.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault(); last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault(); first.focus();
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [open]);

  if (!recordId) return null;
  const confirmation = cascadeConfirmation(recordId, impact);
  const blocked = confirmation.blockedReason;

  const doDelete = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await del(`/api/dataset/train/runs/${recordId}?cascade=1`);
      setOpen(false);
      // The counts come back in the toast, not a flat "Run deleted": a 200 can
      // still have removed fewer images than the dialog promised (one already
      // gone, one still generating) and hiding that would be a small lie.
      toast?.success?.(cascadeResultMessage(res || {}));
      // The hosts already refetch a dataset lane on this callback (that is how
      // the image delete refreshes the board), so the deleted card disappears
      // without this component knowing anything about either graph.
      onDeleted?.([datasetId]);
      onClose?.();
    } catch (e) {
      setError(e?.message || 'Could not delete this run');
    } finally {
      setBusy(false);
    }
  };

  return (
    <section data-testid="run-delete-section"
      className="mt-4 border-t border-border pt-2">
      <h4 className="m-0 mb-1 text-content-subtle text-[0.625rem] font-semibold uppercase tracking-wide">
        Danger zone
      </h4>
      <button type="button" data-testid="run-delete-open" ref={openerRef}
        disabled={!!blocked}
        onClick={() => { setError(null); setOpen(true); }}
        title={blocked || 'Delete this run, its checkpoints and the images it produced'}
        className="rounded-md border border-rose-500/40 px-2 py-1.5 text-rose-300/90 text-[0.625rem] hover:bg-rose-500/10 disabled:opacity-40">
        <span aria-hidden>🗑</span> Delete run &amp; its files…
      </button>
      <p className="m-0 mt-1 break-words text-content-subtle text-[0.625rem] leading-snug">
        {blocked || 'Removes the checkpoints and generated images too. Runs that '
          + 'continued from it are kept.'}
      </p>
      {error && (
        <p className="m-0 mt-1 break-words rounded-lg border border-amber-400/40 bg-amber-500/10 px-2 py-1.5 text-amber-100 text-[0.625rem]">
          {error}
        </p>
      )}

      {open && (
        <div className="fixed inset-0 z-[75] flex items-center justify-center bg-black/70 p-3">
          <div ref={dialogRef} role="dialog" aria-modal="true"
            aria-label="Confirm run deletion" data-testid="run-delete-confirm"
            className="max-h-[85vh] w-full max-w-sm overflow-y-auto rounded-xl bg-surface-overlay/85 backdrop-blur-md p-4 shadow-xl">
            <h4 className="m-0 mb-2 break-words text-sm font-semibold text-content">
              {confirmation.title}
            </h4>
            {confirmation.unknown ? (
              <p className="m-0 mb-3 text-content-muted text-[0.75rem]">
                What this run holds could not be counted. It will still delete its
                checkpoints and the images it produced.
              </p>
            ) : (
              <>
                <p className="m-0 mb-1 text-content-subtle text-[0.6875rem]">This deletes:</p>
                <ul className="m-0 mb-3 list-disc space-y-1 pl-4 text-content-muted text-[0.75rem]">
                  {confirmation.losses.length
                    ? confirmation.losses.map((l) => <li key={l}>{l}</li>)
                    : <li>the run entry only — nothing else is attached to it</li>}
                </ul>
              </>
            )}
            <p className="m-0 mb-1 text-content-subtle text-[0.6875rem]">Kept:</p>
            <ul className="m-0 mb-3 list-disc space-y-1 pl-4 text-content-muted text-[0.75rem]">
              {confirmation.keeps.map((l) => <li key={l}>{l}</li>)}
            </ul>
            <p className="m-0 mb-3 rounded-lg border border-rose-400/40 bg-rose-500/10 px-2 py-1.5 text-rose-100 text-[0.6875rem]">
              Files go to the recycle bin / the app Trash. The run itself cannot be
              brought back.
            </p>
            {error && (
              <p className="m-0 mb-3 break-words rounded-lg border border-amber-400/40 bg-amber-500/10 px-2 py-1.5 text-amber-100 text-[0.6875rem]">
                {error}
              </p>
            )}
            <div className="flex flex-wrap justify-end gap-2">
              <button type="button" ref={cancelRef}
                onClick={() => { setOpen(false); openerRef.current?.focus(); }}
                className="rounded-md border border-border px-3 py-2 text-content-muted text-[0.75rem] hover:text-content">
                Cancel
              </button>
              <button type="button" data-testid="run-delete-confirm-go"
                disabled={busy} onClick={doDelete}
                className="rounded-md border border-rose-500/60 bg-rose-500/15 px-3 py-2 text-[0.75rem] text-rose-100 disabled:opacity-40 hover:bg-rose-500/25">
                {busy ? 'Deleting…' : 'Delete run'}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
