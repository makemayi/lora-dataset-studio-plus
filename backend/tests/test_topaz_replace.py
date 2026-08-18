"""Topaz in-place upscale — swap-restore semantics, routes, settings."""
import json
from unittest.mock import patch

import pytest


def test_topaz_upscale_replace_snapshots_and_enqueues(app, monkeypatch):
    """Mirrors seedvr2_upscale_replace: tile -> pending + swap_restore, and a
    TopazJob queued carrying the source image id."""
    from app.services import dataset_generation_service as dgs
    from app.services.topaz_job_queue import topaz_queue
    from app.models import FaceDataset, FaceDatasetImage, TopazJob
    from app.extensions import db

    with app.app_context():
        ds = FaceDataset(name='ds', trigger_word='t')
        db.session.add(ds)
        db.session.commit()
        # The source file must exist on disk for the replace to enqueue.
        import os, pathlib
        ds_dir = dgs._dataset_path(ds.id)
        pathlib.Path(ds_dir).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(ds_dir) / 'a.png').write_bytes(b'fake-png')
        img = FaceDatasetImage(dataset_id=ds.id, filename='a.png', status='keep')
        db.session.add(img)
        db.session.commit()

        captured = {}
        real_enqueue = topaz_queue.enqueue

        def fake_enqueue(**kw):
            captured.update(kw)
            return real_enqueue(**kw)

        monkeypatch.setattr(topaz_queue, 'enqueue', fake_enqueue)

        jid = dgs.topaz_upscale_replace('local', img.id)
        assert jid.startswith('topaz-')
        assert captured['image_id'] == img.id
        assert captured['dataset_id'] == ds.id
        assert captured['input_filename'] is not None

        img2 = db.session.get(FaceDatasetImage, img.id)
        assert img2.status == 'pending'
        assert img2.job_id == jid
        assert json.loads(img2.swap_restore or '{}')['filename'] == 'a.png'

        tj = TopazJob.query.filter_by(job_id=jid).one()
        assert tj.image_id == img.id


def test_topaz_replace_route_404s_for_missing_image(app, client):
    r = client.post('/api/dataset/image/9999/topaz-replace')
    assert r.status_code == 404


def test_topaz_status_reports_exe(monkeypatch, client):
    from app.services import topaz_helper as th

    monkeypatch.setattr(th, 'resolve_exe', lambda: 'F:/x/tpai.exe')
    d = client.get('/api/settings/topaz/status').get_json()
    assert d['found'] is True
    assert 'tpai.exe' in d['path']


def test_topaz_status_reports_missing(monkeypatch, client):
    from app.services import topaz_helper as th

    monkeypatch.setattr(th, 'resolve_exe', lambda: None)
    d = client.get('/api/settings/topaz/status').get_json()
    assert d['found'] is False
