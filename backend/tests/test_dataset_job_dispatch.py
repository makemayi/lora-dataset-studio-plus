"""Every local generation engine must link its finished image back to its row.

Regression test for a failure that produced NO error at all: Krea 2 Edit shipped
stamping `krea_identity_edit_dataset` on its jobs while `_dispatch_completion`
still matched only Klein's `klein_edit_dataset`. ComfyUI rendered the images, the
queue marked the jobs completed, and the dataset rows stayed `pending` with a
NULL filename — the panel read "0/12" forever and the logs were clean, because
nothing had failed.

It then happened AGAIN with SeedVR2, identically. This file covers the routing
itself, engine by engine; the guard that catches the next unregistered engine
lives in `test_dataset_job_harvest.py`, which discovers them instead of listing
them, and so does the repair of rows a miss has already stranded.
"""
import json
import pytest


def _job(model_name, job_id='j1'):
    from app.models import ImageGenerationQueue
    return ImageGenerationQueue(
        job_id=job_id, user_id='local', status='completed',
        workflow_data='{}', prompt='p',
        job_metadata=json.dumps({'model_name': model_name, 'dataset_id': 1}))


@pytest.mark.parametrize('model_name', [
    'klein_edit_dataset', 'krea_identity_edit_dataset',
    'klein_face_swap_dataset', 'seedvr2_upscale'])
def test_every_local_engine_links_its_finished_image(app, monkeypatch, model_name):
    """The heart of it: a completed job must reach link_completed_dataset_image."""
    from app import job_queue
    from app.services import face_dataset_service
    with app.app_context():
        seen = {}
        monkeypatch.setattr(face_dataset_service, 'link_completed_dataset_image',
                            lambda jid, fn, failed=False, reason=None:
                                seen.update(job_id=jid, filename=fn, failed=failed))
        job_queue._dispatch_completion(_job(model_name), 'out_00001_.png', False)
        assert seen.get('job_id') == 'j1', (
            f'{model_name} never reached the linker — its images would be '
            f'generated and then stranded as pending')
        assert seen['filename'] == 'out_00001_.png'
        assert seen['failed'] is False


def test_a_reference_edit_job_never_reaches_the_row_linker(app, monkeypatch):
    """A LOCAL ✦ Edit-reference render rides the same enqueue_*_edit helpers, so it
    carries the same `model_name` — but it has NO FaceDatasetImage row. It must be
    routed by its own marker (checked first), or link_completed_dataset_image
    would hunt for a row that does not exist and log a bogus 'no row for job'
    while the modal spun forever."""
    from app import job_queue
    from app.services import face_dataset_service
    with app.app_context():
        job = _job('krea_identity_edit_dataset', job_id='ref-1')
        job.job_metadata = json.dumps({'model_name': 'krea_identity_edit_dataset',
                                       'dataset_id': 1, 'is_reference_edit': True})
        seen = {}
        monkeypatch.setattr(face_dataset_service, 'link_completed_dataset_image',
                            lambda *a, **k: seen.update(rows=True))
        monkeypatch.setattr(face_dataset_service, 'link_completed_reference_edit',
                            lambda jid, fn, failed=False, reason=None:
                                seen.update(job_id=jid, filename=fn))
        job_queue._dispatch_completion(job, 'out_00001_.png', False)
        assert seen.get('job_id') == 'ref-1'
        assert 'rows' not in seen


# The set-vs-engines check used to live here, and it did not work. It iterated a
# HARDCODED tuple `(klein_edit_helper, krea_edit_helper)`, so it could only ever
# notice a name declared in the set but stamped by nobody. A THIRD helper module
# was invisible to it — which is precisely how SeedVR2 shipped stranding its
# results months after Krea did the same. Its docstring claimed it would catch
# "a third engine added to the enqueue side"; it could not, and that false
# promise is worse than no test.
#
# It now lives in test_dataset_job_harvest.py, where it DISCOVERS the engines by
# walking app/services with the AST instead of naming them. Both directions are
# covered there. Do not reintroduce a version of it that hardcodes a module list.


def test_an_unknown_model_name_is_ignored_without_crashing(app, monkeypatch):
    """A job from some other producer must not be routed to the dataset linker."""
    from app import job_queue
    from app.services import face_dataset_service
    with app.app_context():
        called = []
        monkeypatch.setattr(face_dataset_service, 'link_completed_dataset_image',
                            lambda *a, **k: called.append(a))
        job_queue._dispatch_completion(_job('something_else', 'j2'), 'x.png', False)
        assert called == []


def test_a_failed_job_still_reaches_the_linker(app, monkeypatch):
    """Failure must travel too — otherwise the tile stays 'generating' forever."""
    from app import job_queue
    from app.services import face_dataset_service
    with app.app_context():
        seen = {}
        monkeypatch.setattr(face_dataset_service, 'link_completed_dataset_image',
                            lambda jid, fn, failed=False, reason=None:
                                seen.update(failed=failed, reason=reason))
        job = _job('krea_identity_edit_dataset', 'j3')
        job.error_message = 'ComfyUI KSampler: out of memory'
        job_queue._dispatch_completion(job, None, True)
        assert seen.get('failed') is True
        assert seen.get('reason') == 'ComfyUI KSampler: out of memory'
