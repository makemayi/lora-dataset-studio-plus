"""REG.RESSION: single-image Topaz upscale end to end — the output must land as
a dataset-RELATIVE filename, not an absolute machine path.

2026-08-23: a single Topaz upscale completed but the tile showed no image
(「单独使用topaz升级图片不返图」). collect_output() returned the ABSOLUTE dataset
path, which link_topaz_image() stored as img.filename; the frontend then
requested /api/dataset/<id>/img/<absolute-path> and safe_join refused to serve a
machine path -> 404 -> broken tile. This test drives the REAL single-image
worker + REAL collector + REAL swap-restore link and asserts the relative name.
"""
import pathlib

import pytest

from app.services import dataset_generation_service as dgs
from app.services.topaz_job_queue import topaz_queue
from app.models import FaceDataset, FaceDatasetImage, TopazJob
from app.extensions import db


def _dataset_with_image(app):
    with app.app_context():
        ds = FaceDataset(name='ds', trigger_word='t')
        db.session.add(ds)
        db.session.commit()
        ds_dir = dgs._dataset_path(ds.id)
        pathlib.Path(ds_dir).mkdir(parents=True, exist_ok=True)
        name = 'orig.png'
        (pathlib.Path(ds_dir) / name).write_bytes(b'x')
        img = FaceDatasetImage(dataset_id=ds.id, filename=name, status='keep')
        db.session.add(img)
        db.session.commit()
        return ds.id, img.id


def test_single_upscale_lands_relative_output(app, monkeypatch):
    import os
    from app.services.topaz_helper import preflight
    ds_id, img_id = _dataset_with_image(app)
    with app.app_context():
        # svc path (route calls this): snapshots the tile + sets it pending.
        monkeypatch.setattr('app.services.topaz_helper.preflight', lambda: None)
        jid = dgs.topaz_upscale_replace('local', img_id)
        assert jid and jid.startswith('topaz-')

        def fake_run_tpai(exe, input_path, output_dir, **kw):
            out = pathlib.Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / 'orig.png').write_bytes(b'UPSCALED')
            return 'ok', ''

        monkeypatch.setattr('app.services.topaz_helper.run_tpai', fake_run_tpai)

        assert topaz_queue.process_one() is True

        job = TopazJob.query.filter_by(job_id=jid).one()
        assert job.status == 'completed'

        img = db.session.get(FaceDatasetImage, img_id)
        assert img.status == 'keep'
        assert img.job_id is None
        # The tile's filename is a RELATIVE basename inside the dataset folder —
        # the thing safe_join can serve. Not an absolute C:\\... path.
        assert img.filename
        assert not os.path.isabs(img.filename)
        assert os.path.basename(img.filename) == img.filename
        ds_dir = dgs._dataset_path(ds_id)
        assert os.path.isfile(os.path.join(ds_dir, img.filename))
        assert img.filename.startswith(f'topaz_{jid[-8:]}_')
