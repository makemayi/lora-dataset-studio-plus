"""Regression coverage for the Bank -> Dataset promotion dedupe cache."""
import io
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from PIL import Image


def _png(seed):
    rnd = random.Random(seed)
    image = Image.frombytes(
        'RGB', (96, 96),
        bytes(rnd.randrange(256) for _ in range(96 * 96 * 3)))
    out = io.BytesIO()
    image.save(out, format='PNG')
    return out.getvalue()


def _run_promote_inline(banks, bank_id, image_ids, dataset_id):
    progress = []
    with patch.object(banks.bank_jobs, 'cancelled', lambda _job: False), \
         patch.object(banks.bank_jobs, 'bump', lambda _job, _n=1: None), \
         patch.object(banks.bank_jobs, 'progress',
                      lambda _job, **values: progress.append(values)):
        banks._promote_job('local', bank_id, image_ids, dataset_id)(object())
    return progress


def test_promote_loads_dataset_dhashes_once_and_dedupes_across_chunks(
        app, tmp_path, monkeypatch):
    """The second chunk must reuse the first chunk's updated cache."""
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage, FaceDatasetImage
        from app.services import face_dataset_service as datasets
        from app.services import image_bank_service as banks

        source = tmp_path / 'bank'
        source.mkdir()
        first = _png(1)
        (source / '00-first.png').write_bytes(first)
        (source / '10-unique.png').write_bytes(_png(2))
        (source / '20-copy.png').write_bytes(first)

        bank, _added = banks.create_bank('local', 'Performance', str(source))
        rows = (BankImage.query.filter_by(bank_id=bank.id)
                .order_by(BankImage.id.asc()).all())
        assert [row.relpath for row in rows] == [
            '00-first.png', '10-unique.png', '20-copy.png']
        for row in rows:
            row.status = 'keep'
        db.session.commit()
        dataset = datasets.create_dataset('local', 'Target', 'targetperf')

        real_existing = datasets._existing_dhash_rows
        existing_calls = []

        def counted_existing(dataset_id):
            existing_calls.append(dataset_id)
            return real_existing(dataset_id)

        monkeypatch.setattr(datasets, '_existing_dhash_rows', counted_existing)
        monkeypatch.setattr(banks, '_existing_dhash_rows', counted_existing)
        monkeypatch.setattr(banks, '_PROMOTE_CHUNK', 2)

        progress = _run_promote_inline(
            banks, bank.id, [row.id for row in rows], dataset.id)

        assert existing_calls == [dataset.id]
        detail = progress[-1]['detail']
        assert detail.startswith('done — 2 imported, 1 already in the dataset')
        assert 'failed' not in detail
        promoted = (FaceDatasetImage.query.filter_by(dataset_id=dataset.id)
                    .order_by(FaceDatasetImage.id.asc()).all())
        assert len(promoted) == 2
        assert {row.bank_image_id for row in promoted} == {rows[0].id, rows[1].id}


def test_two_banks_promoting_same_image_to_one_dataset_create_one_row(
        app, tmp_path, monkeypatch):
    """Both jobs may start together, but only one may snapshot/import at a time."""
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage, FaceDatasetImage
        from app.services import face_dataset_service as datasets
        from app.services import image_bank_service as banks

        payload = _png(40)
        bank_rows = []
        for suffix in ('a', 'b'):
            source = tmp_path / f'bank-{suffix}'
            source.mkdir()
            (source / 'same.png').write_bytes(payload)
            bank, _added = banks.create_bank(
                'local', f'Concurrent {suffix}', str(source))
            row = BankImage.query.filter_by(bank_id=bank.id).one()
            row.status = 'keep'
            bank_rows.append((bank.id, row.id))
        db.session.commit()
        dataset = datasets.create_dataset(
            'local', 'Concurrent target', 'concurrenttarget')

        real_existing = banks._existing_dhash_rows
        first_snapshot = threading.Event()
        second_started = threading.Event()
        second_snapshot = threading.Event()
        calls_lock = threading.Lock()
        calls = 0

        def coordinated_existing(dataset_id):
            nonlocal calls
            snapshot = real_existing(dataset_id)
            with calls_lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                first_snapshot.set()
                assert second_started.wait(2)
                # Without the job-long lock, the second job reaches this loader
                # and captures the same empty snapshot before either import.
                second_snapshot.wait(0.25)
            else:
                second_snapshot.set()
            return snapshot

        monkeypatch.setattr(banks, '_existing_dhash_rows',
                            coordinated_existing)

        def promote(bank_id, image_id, *, second=False):
            if second:
                second_started.set()
            with app.app_context():
                banks._promote_job(
                    'local', bank_id, [image_id], dataset.id)(object())

        with patch.object(banks.bank_jobs, 'cancelled', lambda _job: False), \
             patch.object(banks.bank_jobs, 'bump', lambda _job, _n=1: None), \
             patch.object(banks.bank_jobs, 'progress', lambda _job, **_kw: None), \
             ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                promote, bank_rows[0][0], bank_rows[0][1])
            assert first_snapshot.wait(2)
            second = pool.submit(
                promote, bank_rows[1][0], bank_rows[1][1], second=True)
            first.result(timeout=5)
            second.result(timeout=5)

        db.session.expire_all()
        assert calls == 2
        assert FaceDatasetImage.query.filter_by(
            dataset_id=dataset.id).count() == 1


def test_stale_cached_dhash_after_dataset_file_replacement_is_revalidated(app):
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDatasetImage
        from app.services import face_dataset_service as datasets

        dataset = datasets.create_dataset(
            'local', 'Stale hash target', 'stalehashtarget')
        original, replacement = _png(70), _png(71)
        ids, _ = datasets.import_images(
            'local', dataset.id, [original], dedupe=True)
        stale_seen = datasets._existing_dhash_rows(dataset.id)
        stale_hash, existing_id = stale_seen[0]
        existing = db.session.get(FaceDatasetImage, existing_id)
        path = os.path.join(
            datasets._dataset_dir(dataset.id), existing.filename)
        with open(path, 'wb') as handle:
            handle.write(replacement)
        with Image.open(io.BytesIO(replacement)) as image:
            replacement_hash = datasets._dhash(image)
        assert datasets._hamming(
            stale_hash, replacement_hash) > datasets.SCRAPE_DHASH_MAX_DISTANCE

        stats = {}
        imported, failed = datasets.import_images(
            'local', dataset.id, [original], dedupe=True, stats=stats,
            dedupe_seen=stale_seen)

        assert len(ids) == 1
        assert failed == 0
        assert len(imported) == 1
        assert stats.get('duplicates', 0) == 0
        assert stale_seen[0] == (replacement_hash, existing_id)
        assert FaceDatasetImage.query.filter_by(
            dataset_id=dataset.id).count() == 2


def test_import_images_default_still_loads_and_commits_per_image(app, monkeypatch):
    """Callers that omit the internal cache retain the historical semantics."""
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDatasetImage
        from app.services import face_dataset_service as datasets

        dataset = datasets.create_dataset('local', 'Ordinary', 'ordinaryperf')
        real_existing = datasets._existing_dhash_rows
        existing_calls = []

        def counted_existing(dataset_id):
            existing_calls.append(dataset_id)
            return real_existing(dataset_id)

        monkeypatch.setattr(datasets, '_existing_dhash_rows', counted_existing)
        session = db.session()
        real_commit = session.commit
        commit_calls = []

        def counted_commit():
            commit_calls.append(True)
            return real_commit()

        monkeypatch.setattr(session, 'commit', counted_commit)
        ids, failed = datasets.import_images(
            'local', dataset.id, [_png(10), _png(11)], dedupe=True)

        assert failed == 0
        assert len(ids) == 2
        assert existing_calls == [dataset.id]
        assert len(commit_calls) == 2
        assert FaceDatasetImage.query.filter_by(dataset_id=dataset.id).count() == 2
