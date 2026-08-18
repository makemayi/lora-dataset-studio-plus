"""The upscale-replace dispatcher must route by engines.upscale_engine — the
setting the user picks in Settings ▸ Image engines. Regression: a saved 'topaz'
that still ran ComfyUI (SeedVR2) means the route read the wrong key."""
import pathlib

import pytest


@pytest.fixture(autouse=True)
def _no_real_gpu(monkeypatch):
    """Both engines enqueue; neither should touch real services here."""
    monkeypatch.setattr('app.services.topaz_helper.preflight', lambda: None)


def _make_owned_image(app):
    from app.services import dataset_generation_service as dgs
    from app.models import FaceDataset, FaceDatasetImage
    from app.extensions import db

    ds = FaceDataset(name='ds', trigger_word='t')
    db.session.add(ds)
    db.session.commit()
    ds_dir = dgs._dataset_path(ds.id)
    pathlib.Path(ds_dir).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(ds_dir) / 'a.png').write_bytes(b'x')
    img = FaceDatasetImage(dataset_id=ds.id, filename='a.png', status='keep')
    db.session.add(img)
    db.session.commit()
    return img


def test_dispatcher_routes_to_topaz_when_setting_says_topaz(client, app, monkeypatch):
    """engines.upscale_engine = 'topaz' => the 🔍 dispatcher queues a TopazJob,
    not an ImageGenerationQueue row."""
    from app.services import dataset_generation_service as dgs
    from app.models import TopazJob
    from app.services.topaz_job_queue import topaz_queue
    from app.extensions import db
    import app.config as cfg

    with app.app_context():
        img = _make_owned_image(app)
        # persist the setting the way the Settings page does
        saved = cfg.load_config()
        saved.setdefault('engines', {})['upscale_engine'] = 'topaz'
        cfg.save_config({'engines': saved['engines']})

        resp = client.post(f'/api/dataset/image/{img.id}/upscale-replace')
        body = resp.get_json()
        assert resp.status_code == 202, body
        assert body['engine'] == 'topaz'
        assert body['job_id'].startswith('topaz-')
        assert TopazJob.query.filter_by(job_id=body['job_id']).one().image_id == img.id


def test_dispatcher_routes_to_seedvr2_by_default(client, app, monkeypatch):
    """No setting (or 'seedvr2') => the existing ComfyUI path is used."""
    from app.models import ImageGenerationQueue
    from app.extensions import db

    with app.app_context():
        img = _make_owned_image(app)
        from app.services import dataset_generation_service as dgs
        monkeypatch.setattr('app.services.seedvr2_helper.preflight', lambda: None)

        def fake_enqueue(**kw):
            from app.job_queue import queue_manager
            return queue_manager.add_job(workflow_data={'1': {}})
        monkeypatch.setattr('app.services.seedvr2_helper.enqueue_seedvr2_upscale',
                            fake_enqueue)

        resp = client.post(f'/api/dataset/image/{img.id}/upscale-replace')
        body = resp.get_json()
        assert resp.status_code == 202, body
        assert body['engine'] == 'seedvr2'
