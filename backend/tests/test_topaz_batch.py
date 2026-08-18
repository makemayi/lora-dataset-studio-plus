"""Topaz BATCH upscale — one job = many images, one tpai call (models load once)."""
import json
import pathlib

import pytest

from app.services.topaz_job_queue import topaz_queue


def _dataset_with_images(app, n=3, prefix='a'):
    """Create a dataset with n images on disk. Returns (ds_id, [image_ids])."""
    from app.services import dataset_generation_service as dgs
    from app.models import FaceDataset, FaceDatasetImage
    from app.extensions import db

    with app.app_context():
        ds = FaceDataset(name='ds', trigger_word='t')
        db.session.add(ds)
        db.session.commit()
        ds_dir = dgs._dataset_path(ds.id)
        pathlib.Path(ds_dir).mkdir(parents=True, exist_ok=True)
        ids = []
        for i in range(n):
            name = f'{prefix}{i}.png'
            (pathlib.Path(ds_dir) / name).write_bytes(b'x')
            img = FaceDatasetImage(dataset_id=ds.id, filename=name, status='keep')
            db.session.add(img)
            db.session.flush()
            ids.append(img.id)
        db.session.commit()
        return ds.id, ids


def test_batch_enqueue_creates_one_row_with_all_images(app):
    """N images -> ONE TopazJob row carrying every image id + input path."""
    ds_id, img_ids = _dataset_with_images(app)
    with app.app_context():
        jid = topaz_queue.enqueue_batch(
            user_id='local', dataset_id=ds_id,
            inputs=[{'image_id': i, 'input': f'C:/nope/{i}.png'}
                    for i in img_ids])
        from app.models import TopazJob
        row = TopazJob.query.filter_by(job_id=jid).one()
        assert row.total_images == len(img_ids)
        ids = json.loads(row.image_ids or '[]')
        assert ids == img_ids
        assert row.status == 'queued'


def test_upscale_replace_batch_snapshots_every_tile(app, monkeypatch):
    """Every image is snapshotted + set pending with the batch job id."""
    from app.services import dataset_generation_service as dgs
    from app.models import FaceDatasetImage
    from app.extensions import db

    ds_id, img_ids = _dataset_with_images(app)
    with app.app_context():
        monkeypatch.setattr('app.services.topaz_helper.preflight', lambda: None)
        out = dgs.topaz_upscale_replace_batch('local', ds_id, img_ids)
        assert out['queued'] == len(img_ids)
        assert out['skipped'] == 0
        for img_id in img_ids:
            row = db.session.get(FaceDatasetImage, img_id)
            assert row.status == 'pending'
            assert row.job_id == out['job_id']
            assert json.loads(row.swap_restore)['filename'] == f'a{img_ids.index(img_id)}.png'


def test_upscale_replace_batch_filters_ineligible(app, monkeypatch):
    """Missing files and already-pending rows are skipped, not queued."""
    from app.services import dataset_generation_service as dgs
    from app.models import FaceDatasetImage
    from app.extensions import db

    ds_id, img_ids = _dataset_with_images(app, n=2)
    with app.app_context():
        missing = db.session.get(FaceDatasetImage, img_ids[1])
        missing.filename = None          # no file -> not eligible
        db.session.commit()
        monkeypatch.setattr('app.services.topaz_helper.preflight', lambda: None)
        out = dgs.topaz_upscale_replace_batch('local', ds_id, img_ids)
        assert out['queued'] == 1
        assert out['skipped'] == 1
