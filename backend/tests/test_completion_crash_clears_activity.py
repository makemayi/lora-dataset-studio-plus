"""A crashing completion callback must not leave the app saying "generating".

THE SECOND HALF OF THE 2026-08-09 GHOST PROGRESS BAR
----------------------------------------------------
`_dispatch_completion` already had a rescue: if the link callback throws, mark
the row `failed` so it does not sit in 'pending' looking alive. That flips the
ROW, and it was assumed to be enough.

It is not. The batch's progress indicator is a SEPARATE in-memory count
(`dataset_activity`), reconciled by `_sync_generate_activity` on enqueue, on
completion and on cancel. A crash in the callback is none of those three, so the
count kept its old value: dataset 11 reported **"generating 1 of 2" with nothing
running**, no job left that could ever finish it, and only the registry's TTL to
eventually clear it. The user's report was "it says it is generating variants,
but I am not running anything".

The row fix and the counter fix are independent, so this file pins the counter.
The trigger that exposed it (a held-open file on Windows) is fixed separately in
test_comfy_output_collect.py — but ANY exception in a link callback reaches this
same path, so the rescue has to be complete on its own.
"""
import json
from unittest.mock import patch

from app import db
from app.models import FaceDataset, FaceDatasetImage
from app.services import dataset_activity

LINK = 'app.services.face_dataset_service.link_completed_dataset_image'
SYNC = 'app.services.dataset_generation_service._sync_generate_activity'


def _dataset_with_two_pending_jobs():
    ds = FaceDataset(name='ghost-progress', trigger_word='ghostwoman')
    db.session.add(ds)
    db.session.commit()
    rows = []
    for i in range(2):
        img = FaceDatasetImage(dataset_id=ds.id, status='pending',
                               job_id=f'job-{i}', variation_label=f'shot {i}')
        db.session.add(img)
        rows.append(img)
    db.session.commit()
    return ds, rows


class _Job:
    """The shape `_dispatch_completion` reads off a finished queue job."""

    def __init__(self, job_id):
        self.job_id = job_id
        self.error_message = None
        self.job_type = 'image'
        # Stored as a JSON STRING on the row, and `model_name` must be one of
        # DATASET_IMAGE_JOB_NAMES or the dispatch skips the dataset branch and
        # this test would pass without exercising anything.
        self.job_metadata = json.dumps({'model_name': 'klein_face_swap_dataset'})
        self.status = 'completed'


def _still_generating(dataset_id):
    state = dataset_activity.get(dataset_id)
    return bool(state) and state.get('total', 0) > 0


def test_a_crashing_link_callback_clears_the_progress_counter(app):
    """The regression. Before the fix this ended with the dataset still
    reporting an in-flight generate and no job able to clear it."""
    from app.services.dataset_generation_service import _sync_generate_activity
    from app import job_queue

    with app.app_context():
        ds, rows = _dataset_with_two_pending_jobs()
        _sync_generate_activity(ds.id)
        assert _still_generating(ds.id), 'precondition: two jobs in flight'

        # Row 0 completes normally — one left, so the badge is honest so far.
        rows[0].status = 'keep'
        rows[0].filename = 'a.png'
        db.session.commit()
        _sync_generate_activity(ds.id)
        assert _still_generating(ds.id)

        # Row 1's callback explodes — exactly what a held-open source file did.
        with patch(LINK, side_effect=PermissionError(32, 'file in use')):
            job_queue._dispatch_completion(_Job('job-1'), 'b.png', failed=False)

        assert FaceDatasetImage.query.filter_by(job_id='job-1').first().status \
            == 'failed', 'the row rescue must still work'
        assert not _still_generating(ds.id), \
            'nothing is running, so nothing may still say "generating"'


def test_the_rescue_survives_a_dataset_it_cannot_resync(app):
    """A rescue that throws while cleaning up would re-raise inside the very
    handler meant to contain the first failure. It must stay contained, and the
    row must still be flipped."""
    from app import job_queue

    with app.app_context():
        _dataset_with_two_pending_jobs()
        with patch(LINK, side_effect=RuntimeError('boom')), \
             patch(SYNC, side_effect=RuntimeError('and the cleanup broke too')):
            job_queue._dispatch_completion(_Job('job-0'), 'a.png', failed=False)

        assert FaceDatasetImage.query.filter_by(job_id='job-0').first().status \
            == 'failed'
