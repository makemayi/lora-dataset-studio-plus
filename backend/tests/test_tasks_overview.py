"""Task Center API tests: metadata stamping, overview aggregation, retry/cancel.

These pin the resource-link contract (spec §2.5): in-place image jobs must
stamp `source_image_id` so the Task Center can answer "which image is this job
for?", and legacy jobs without it still resolve via the job_id reverse lookup.
"""
import json

import pytest


@pytest.fixture(autouse=True)
def _isolate_host_ollama_gpu_fence(monkeypatch):
    """Same fence isolation the queue unit tests rely on."""
    monkeypatch.setattr(
        'app.services.vision_keepalive.ensure_released_for_comfy',
        lambda: True,
    )


def _make_dataset(app, name='ds_test', trigger='tester'):
    from app.models import FaceDataset
    from app.extensions import db
    ds = FaceDataset(name=name, trigger_word=trigger)
    db.session.add(ds)
    db.session.commit()
    return ds


def test_seedvr2_replace_stamps_source_image_id(app, monkeypatch):
    """The destructive upscale lane stamps source_image_id into its metadata."""
    from app.services import dataset_generation_service as dgs
    from app.models import FaceDatasetImage
    from app.extensions import db

    with app.app_context():
        ds = _make_dataset(app)
        img = FaceDatasetImage(dataset_id=ds.id, filename='a.png', status='pending')
        db.session.add(img)
        db.session.commit()

        captured = {}
        monkeypatch.setattr('app.services.seedvr2_helper.preflight', lambda: None)

        def fake_enqueue(**kw):
            captured.update(kw)
            return 'fake-job'

        monkeypatch.setattr(
            'app.services.seedvr2_helper.enqueue_seedvr2_upscale', fake_enqueue)

        jid = dgs.seedvr2_upscale_replace('local', img.id)
        assert jid == 'fake-job'
        md = captured['extra_metadata']
        assert md['source_image_id'] == img.id
        assert md['dataset_id'] == ds.id


def test_enqueue_improve_stamps_source_image_id_for_face_images(app, monkeypatch):
    """The shared improve lane stamps it only for FaceDatasetImage sources
    (a LoraTestImage source would be a cross-table id and is left alone)."""
    from app.services import dataset_generation_service as dgs
    from app.models import FaceDatasetImage
    from app.extensions import db

    with app.app_context():
        ds = _make_dataset(app, name='ds_i')
        img = FaceDatasetImage(dataset_id=ds.id, filename='a.png', status='pending')
        db.session.add(img)
        db.session.commit()

        captured = {}
        monkeypatch.setattr('app.services.seedvr2_helper.preflight', lambda: None)

        def fake_enqueue(**kw):
            captured.update(kw)
            return 'fake-job'

        monkeypatch.setattr(
            'app.services.seedvr2_helper.enqueue_seedvr2_upscale', fake_enqueue)

        dgs._enqueue_improve('seedvr2', user_id='local', source=img,
                             source_path='/x/a.png', prompt='', label='x')
        assert captured['extra_metadata']['source_image_id'] == img.id


def test_overview_lists_image_training_and_vision_rows(client, app):
    """One poll returns the synthesized three kinds plus ComfyUI/GPU status."""
    from app.job_queue import queue_manager
    from app.models import SystemState
    from app.extensions import db

    with app.app_context():
        queue_manager.add_job(
            workflow_data={'1': {}},
            metadata={'model_name': 'klein_edit_dataset', 'dataset_id': 1})

        SystemState.query.filter_by(key='training_in_progress').delete()
        db.session.add(SystemState(
            key='training_in_progress',
            value=json.dumps({'v': True, 'exp': None})))
        db.session.commit()

    d = client.get('/api/tasks/overview').get_json()
    assert d['status']['comfyui'] is not None
    assert d['status']['summary']['queued'] >= 1
    kinds = {t['kind'] for t in d['tasks']}
    assert 'image' in kinds
    assert 'training' in kinds
    image = next(t for t in d['tasks'] if t['kind'] == 'image')
    assert image['title'] == 'klein_edit_dataset'
    assert image['resource']['type'] == 'dataset'


def test_overview_resource_reverse_lookup_when_metadata_missing(client, app):
    """Old jobs without source_image_id get their image via job_id reverse lookup."""
    from app.job_queue import queue_manager
    from app.models import FaceDatasetImage
    from app.extensions import db

    with app.app_context():
        ds = _make_dataset(app, name='ds_x')
        ds_id = ds.id
        jid = queue_manager.add_job(
            workflow_data={'1': {}},
            metadata={'model_name': 'seedvr2_upscale', 'dataset_id': ds.id})
        img = FaceDatasetImage(dataset_id=ds.id, filename='a.png',
                               status='pending', job_id=jid)
        db.session.add(img)
        db.session.commit()
        img_id = img.id

    d = client.get('/api/tasks/overview').get_json()
    row = next(t for t in d['tasks'] if t['job_id'] == jid)
    assert row['resource'] == {'type': 'image', 'dataset_id': ds_id,
                               'image_id': img_id}


def test_retry_requeues_failed_job(client, app):
    from app.models import ImageGenerationQueue
    from app.extensions import db
    from datetime import datetime, timezone

    with app.app_context():
        db.session.add(ImageGenerationQueue(
            job_id='retry-me', user_id='local', status='failed',
            workflow_data='{"1": {}}', error_message='boom',
            completed_at=datetime.now(timezone.utc).replace(tzinfo=None)))
        db.session.commit()

    d = client.post('/api/tasks/retry-me/retry').get_json()
    assert d['ok'] is True
    with app.app_context():
        row = ImageGenerationQueue.query.filter_by(job_id='retry-me').one()
        assert row.status == 'pending'
        assert row.retry_count == 1
        assert row.error_message is None

    # retrying a non-failed job is a 409
    r = client.post('/api/tasks/retry-me/retry')
    assert r.status_code == 409


def test_cancel_queued_and_paused_jobs(client, app):
    from app.job_queue import AWAITING_COMFYUI, queue_manager
    from app.models import ImageGenerationQueue
    from app.extensions import db

    with app.app_context():
        j1 = queue_manager.add_job(workflow_data={'1': {}})
        j2 = queue_manager.add_job(workflow_data={'1': {}})
        row = ImageGenerationQueue.query.filter_by(job_id=j2).one()
        row.status = AWAITING_COMFYUI
        db.session.commit()

        assert client.post(f'/api/tasks/{j1}/cancel').get_json()['ok'] is True
        assert client.post(f'/api/tasks/{j2}/cancel').get_json()['ok'] is True
        assert ImageGenerationQueue.query.filter_by(job_id=j1).one().status == 'cancelled'
        assert ImageGenerationQueue.query.filter_by(job_id=j2).one().status == 'cancelled'


def test_overview_lists_topaz_jobs(client, app):
    """Topaz jobs appear in the Task Center with image resources and actions."""
    from app.services.topaz_job_queue import topaz_queue
    from app.models import TopazJob
    from app.extensions import db

    with app.app_context():
        jid = topaz_queue.enqueue(user_id='local', dataset_id=5, image_id=9,
                                  input_filename='C:/nope/a.png')
        row = TopazJob.query.filter_by(job_id=jid).one()
        row.status = 'failed'
        row.error_message = 'open Topaz once'
        db.session.commit()

    d = client.get('/api/tasks/overview').get_json()
    row = next(t for t in d['tasks'] if t['job_id'] == jid)
    assert row['kind'] == 'topaz'
    assert row['status'] == 'failed'
    assert row['resource'] == {'type': 'image', 'dataset_id': 5, 'image_id': 9}
    assert 'retry' in row['actions']


def test_topaz_cancel_and_retry_routes(client, app):
    """The /api/tasks routes dispatch topaz- prefixed ids to the Topaz queue."""
    from app.services.topaz_job_queue import topaz_queue
    from app.models import TopazJob
    from app.extensions import db

    with app.app_context():
        jid = topaz_queue.enqueue(user_id='local', dataset_id=1, image_id=2,
                                  input_filename='C:/nope/a.png')

    assert client.post(f'/api/tasks/{jid}/cancel').get_json()['ok'] is True
    with app.app_context():
        assert TopazJob.query.filter_by(job_id=jid).one().status == 'cancelled'

    with app.app_context():
        row = TopazJob.query.filter_by(job_id=jid).one()
        row.status = 'failed'
        row.error_message = 'x'
        db.session.commit()
    assert client.post(f'/api/tasks/{jid}/retry').get_json()['ok'] is True
    with app.app_context():
        assert TopazJob.query.filter_by(job_id=jid).one().status == 'queued'
