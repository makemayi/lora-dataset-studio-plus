"""Every local generation engine must link its finished image back to its row.

Regression test for a failure that produced NO error at all: Krea 2 Edit shipped
stamping `krea_identity_edit_dataset` on its jobs while `_dispatch_completion`
still matched only Klein's `klein_edit_dataset`. ComfyUI rendered the images, the
queue marked the jobs completed, and the dataset rows stayed `pending` with a
NULL filename — the panel read "0/12" forever and the logs were clean, because
nothing had failed.

The test walks `DATASET_IMAGE_JOB_NAMES` instead of naming engines one by one,
so a third engine added to the enqueue side without being added here fails here.
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
    'klein_edit_dataset', 'krea_identity_edit_dataset', 'klein_face_swap_dataset'])
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


def test_the_declared_set_matches_what_the_engines_actually_stamp(app):
    """The set is only useful if it stays in step with the enqueue side, so read
    the stamps from the helpers themselves rather than trusting a literal."""
    from app import job_queue
    from app.services import klein_edit_helper, krea_edit_helper, face_swap_helper  # noqa: F401
    import inspect
    stamped = set()
    for mod in (klein_edit_helper, krea_edit_helper, face_swap_helper):
        src = inspect.getsource(mod)
        for name in job_queue.DATASET_IMAGE_JOB_NAMES:
            if f"'{name}'" in src or f'"{name}"' in src:
                stamped.add(name)
    assert stamped == set(job_queue.DATASET_IMAGE_JOB_NAMES), (
        'a name in DATASET_IMAGE_JOB_NAMES is stamped by no engine, or an '
        f'engine stamps a name the dispatch does not know: {stamped} vs '
        f'{set(job_queue.DATASET_IMAGE_JOB_NAMES)}')


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
