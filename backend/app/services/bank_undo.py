"""↩ Undo for the 🗃️ bank's bulk decisions — one step back, per bank.

WHY THIS EXISTS
---------------
Marking hundreds of images in one gesture (✓/✕ over a whole filter, auto-reject
at a threshold, collapsing every duplicate group) is the bank's biggest lever and
was its only irreversible-feeling one. A mis-set threshold used to cost a manual
re-triage of everything it caught.

WHAT IT REMEMBERS — and what it deliberately does not
-----------------------------------------------------
Only the STATUS DIMENSION: ``(status, reject_reason)`` per row, the two columns
those actions write. That is exactly the set of bulk actions that are reversible
*cleanly*, and the honesty rule is that we offer undo for nothing else:

* 🗑 Delete rejected sends source files to the OS trash (which we cannot pull
  back programmatically) and drops the rows with all their analysis — an
  "Undo" there would restore, at best, half of half. It CLEARS this registry
  instead, because the rows a pending snapshot points at are the ones it just
  deleted.
* ⬆ Promote copies through the dataset import path; un-promoting would mean
  deleting images in someone else's dataset.
* 🔄 Rotate lives on another column and already has a natural inverse (turn the
  other way), which the toast says.

DEPTH IS ONE, ON PURPOSE
------------------------
Not a shortcut — the only depth at which "every row goes back to exactly what it
was" stays checkable. A deeper stack would have to reason about whether older
states still hold after a re-scan dropped rows or a pipeline re-flagged them, and
would invite unwinding *through* a pipeline whose other effects do not unwind.
Zero → one is the whole value.

IN MEMORY, LIKE THE JOB REGISTRY
--------------------------------
Same declared lifetime as ``bank_jobs`` and ``dataset_activity``: the snapshot
lives in the SERVER, so it survives a page reload, a tab switch and a second tab
— which a React-state undo does not — and dies with the process. A restart is a
defensible session boundary for "undo my last click", and it costs no schema
migration on the databases users already have. The UI states the boundary rather
than implying permanence.

A snapshot entry is ``{image_id: {'before': (status, reason),
'after': (status, reason)}}``. Keeping ``after`` is what lets the restore tell
"nobody touched this since" from "someone did" — the latter is skipped and
NAMED, never clobbered.
"""
import threading
import time

_lock = threading.Lock()
_snapshots: dict = {}        # bank_id -> {'label', 'rows', 'at'}
_TTL = 60 * 60               # an offer older than this is stale — the user moved on


class Snapshot:
    """Collects the prior state of the rows an action is about to change.

    Mutators call :meth:`note` right before writing; rows whose value would not
    actually change are ignored, so the offer's count is the number of decisions
    genuinely flipped rather than the size of the selection.

    One snapshot can span several mutators (the pipeline's auto-reject step is
    ``apply_flags`` followed by a duplicate resolution): the earliest ``before``
    wins and the latest ``after`` is kept, so the pair still describes the whole
    step end to end.
    """

    def __init__(self, label):
        self.label = label
        self.rows: dict = {}

    def note(self, row, status, reason):
        before = (row.status, row.reject_reason)
        after = (status, reason)
        entry = self.rows.get(row.id)
        if entry is None:
            if before == after:
                return
            self.rows[row.id] = {'before': before, 'after': after}
            return
        entry['after'] = after
        if entry['before'] == after:
            # walked back to where it started within the same action
            self.rows.pop(row.id, None)

    def commit(self, bank_id):
        """Publish as THE undo offer for this bank, replacing any previous one.
        A snapshot that changed nothing publishes nothing."""
        if self.rows:
            record(bank_id, self.label, self.rows)
        return len(self.rows)


def record(bank_id, label, rows):
    """Replace the bank's undo offer. ``rows`` is {image_id: {'before','after'}}."""
    with _lock:
        _snapshots[int(bank_id)] = {'label': str(label), 'rows': dict(rows),
                                    'at': time.time()}


def peek(bank_id):
    """The offer as the UI shows it: {label, count, at} — or None. Purges a
    stale entry on the way through, like the job registry does."""
    with _lock:
        snap = _snapshots.get(int(bank_id))
        if not snap:
            return None
        if time.time() - snap['at'] > _TTL:
            _snapshots.pop(int(bank_id), None)
            return None
        return {'label': snap['label'], 'count': len(snap['rows']),
                'at': snap['at']}


def take(bank_id):
    """Pop the offer for a restore. One shot: an undo that half-succeeded is not
    re-offered, because a second press would find the same conflicts."""
    with _lock:
        snap = _snapshots.pop(int(bank_id), None)
    if snap and time.time() - snap['at'] > _TTL:
        return None
    return snap


def clear(bank_id):
    """Drop the offer — used by the actions that make it un-restorable
    (🗑 Delete rejected drops the very rows it points at) and by bank deletion."""
    with _lock:
        _snapshots.pop(int(bank_id), None)


def reset():
    """Test helper: forget every snapshot."""
    with _lock:
        _snapshots.clear()
