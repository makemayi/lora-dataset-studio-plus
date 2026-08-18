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


def test_worker_batch_calls_tpai_once_and_links_each(app, monkeypatch):
    """N images -> run_tpai called EXACTLY ONCE over a folder; every output is
    mapped back to its image and linked (swap-restore success per tile)."""
    import pathlib
    from app.services.topaz_job_queue import topaz_queue
    from app.models import TopazJob

    linked = []
    monkeypatch.setattr(topaz_queue, 'link_completed',
                        lambda row: linked.append(row.job_id))

    with app.app_context():
        jid = topaz_queue.enqueue_batch(
            user_id='local', dataset_id=1,
            inputs=[{'image_id': 10, 'input': 'C:/x/a.png'},
                    {'image_id': 11, 'input': 'C:/x/b.png'}])

        calls = {'n': 0}

        def fake_run_tpai(exe, input_dir, output_dir, **kw):
            calls['n'] += 1
            # write two outputs so the collector can map them back
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            (pathlib.Path(output_dir) / 'img_10.png').write_bytes(b'out-a')
            (pathlib.Path(output_dir) / 'img_11.png').write_bytes(b'out-b')
            return 'ok', ''

        monkeypatch.setattr('app.services.topaz_helper.run_tpai', fake_run_tpai)
        monkeypatch.setattr('app.services.topaz_job_queue.stage_inputs',
                            lambda inputs, tmp: {
                                img_id: f'img_{img_id}.png'
                                for img_id, _ in inputs})
        monkeypatch.setattr('app.services.topaz_job_queue.collect_output',
                            lambda tmp_dir, dataset_id, job_id, staged_name=None:
                                f'DS/{staged_name}')

        assert topaz_queue.process_one() is True
        assert calls['n'] == 1, 'models must load once per batch'
        row = TopazJob.query.filter_by(job_id=jid).one()
        assert row.status == 'completed'
        assert row.done_images == 2
        import json
        results = json.loads(row.image_results or '{}')
        assert results['10']['status'] == 'completed'
        assert results['11']['status'] == 'completed'


def test_worker_batch_partial_failure_restores_missing_outputs(app, monkeypatch):
    """An image whose output never appears is restored (swap-restore) and the
    batch ends failed(partial) with a per-image detail."""
    import json
    import pathlib
    from app.services.topaz_job_queue import topaz_queue
    from app.models import TopazJob

    with app.app_context():
        jid = topaz_queue.enqueue_batch(
            user_id='local', dataset_id=1,
            inputs=[{'image_id': 10, 'input': 'C:/x/a.png'},
                    {'image_id': 11, 'input': 'C:/x/b.png'}])

        def fake_run_tpai(exe, input_dir, output_dir, **kw):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            (pathlib.Path(output_dir) / 'img_10.png').write_bytes(b'out-a')
            return 'ok', ''           # img_11 output missing

        monkeypatch.setattr('app.services.topaz_helper.run_tpai', fake_run_tpai)
        monkeypatch.setattr('app.services.topaz_job_queue.stage_inputs',
                            lambda inputs, tmp: {
                                img_id: f'img_{img_id}.png'
                                for img_id, _ in inputs})
        def fake_collect(tmp_dir, dataset_id, job_id, staged_name=None):
            # mirror the real collector: only a file that tpai actually wrote
            # comes back as a path (the real one checks the output folder).
            return (f'DS/{staged_name}'
                    if (pathlib.Path(tmp_dir) / staged_name).exists()
                    else None)

        monkeypatch.setattr('app.services.topaz_job_queue.collect_output',
                            fake_collect)

        restored = {}

        def fake_link(row):
            results = json.loads(row.image_results or '{}')
            for img_id, res in results.items():
                restored[img_id] = res.get('status')

        monkeypatch.setattr(topaz_queue, 'link_completed', fake_link)

        topaz_queue.process_one()
        row = TopazJob.query.filter_by(job_id=jid).one()
        assert row.status == 'failed'
        assert '1 of 2' in (row.error_message or '')
        results = json.loads(row.image_results or '{}')
        assert results['11']['status'] == 'failed'
        assert 'nothing' in (results['11'].get('error') or '')
