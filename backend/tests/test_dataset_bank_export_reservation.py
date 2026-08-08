"""Exclusive Dataset -> Bank export reservation and cleanup contracts."""
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time

from PIL import Image
import pytest
from sqlalchemy import event


def _seed_dataset(app, name='Dataset source'):
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDataset, FaceDatasetImage
        from app.services.dataset_storage import ensure_dataset_dir

        dataset = FaceDataset(
            user_id='local', name=name, trigger_word='reservation_subject')
        db.session.add(dataset)
        db.session.flush()
        folder = str(ensure_dataset_dir(dataset.id))
        path = os.path.join(folder, 'source.png')
        Image.new('RGB', (8, 8), (20, 40, 60)).save(path, format='PNG')
        image = FaceDatasetImage(
            dataset_id=dataset.id, filename='source.png', status='keep',
            caption='kept caption',
        )
        db.session.add(image)
        db.session.commit()
        return dataset.id, image.id, folder, path


def _assert_export_released(dataset_id):
    from app.services import dataset_activity

    assert dataset_activity.get(dataset_id) is None
    token = dataset_activity.begin_exclusive(
        dataset_id, 'bank_export', total=1, detail='re-acquired')
    assert token is not None
    dataset_activity.end(token)


def test_begin_exclusive_has_exactly_one_winner_under_concurrency():
    from app.services import dataset_activity

    workers = 12
    gate = threading.Barrier(workers)

    def reserve():
        gate.wait(timeout=3)
        return dataset_activity.begin_exclusive(7001, 'bank_export', total=1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        tokens = list(pool.map(lambda _n: reserve(), range(workers)))

    winners = [token for token in tokens if token is not None]
    assert len(winners) == 1
    assert dataset_activity.get(7001)['kind'] == 'bank_export'
    dataset_activity.end(winners[0])
    assert dataset_activity.get(7001) is None


def test_dataset_activity_normalizes_numeric_string_ids_and_end_releases():
    """The registry and its token decoder must address the same canonical key."""
    from app.services import dataset_activity

    token = dataset_activity.begin_exclusive('7002', 'bank_export', total=1)
    assert token is not None
    assert dataset_activity.get(7002)['kind'] == 'bank_export'
    assert set(dataset_activity._active) == {7002}

    dataset_activity.end(token)
    assert dataset_activity.get(7002) is None
    assert dataset_activity._active == {}


@pytest.mark.parametrize('invalid', [
    None, True, False, 0, -1, 1.0, '', ' 1', '1 ', '1.0', 'abc',
    str(1 << 63),
])
def test_dataset_activity_rejects_noncanonical_or_out_of_range_ids(invalid):
    from app.services import dataset_activity

    with pytest.raises(ValueError, match='positive integer'):
        dataset_activity.begin_exclusive(invalid, 'bank_export')
    with pytest.raises(ValueError, match='positive integer'):
        dataset_activity.begin(invalid, 'caption')


def test_ordinary_activity_cannot_join_an_exclusive_export():
    from app.services import dataset_activity

    export = dataset_activity.begin_exclusive(7003, 'bank_export', total=3)
    with pytest.raises(dataset_activity.DatasetActivityBusy,
                       match='exclusive bank_export'):
        dataset_activity.begin(7003, 'caption', total=1)

    assert [entry['kind'] for entry in dataset_activity._active[7003].values()] == [
        'bank_export']
    dataset_activity.end(export)


def test_sync_pending_cannot_join_an_exclusive_export():
    from app.services import dataset_activity

    export = dataset_activity.begin_exclusive(7004, 'bank_export', total=3)
    dataset_activity.sync_pending(7004, 'generate', pending=2)

    assert [entry['kind'] for entry in dataset_activity._active[7004].values()] == [
        'bank_export']
    dataset_activity.end(export)


def test_running_purges_expired_exclusive_activity_before_answering():
    from app.services import dataset_activity

    token = dataset_activity.begin_exclusive(7005, 'bank_export')
    dataset_activity._active[7005][token]['_touched'] = (
        time.time() - dataset_activity._TTL_SECONDS - 1)

    assert dataset_activity.running(7005, ('bank_export',)) is False
    assert 7005 not in dataset_activity._active
    replacement = dataset_activity.begin(7005, 'caption')
    dataset_activity.end(replacement)


def test_export_reservation_exists_before_the_selected_generation_is_snapshotted(app):
    """No delete/edit can slip between selecting rows and reserving the Dataset."""
    with app.app_context():
        from app.extensions import db
        from app.models import ImageBank
        from app.services import dataset_activity
        from app.services import image_bank_service as banks

        dataset_id, _image_id, _folder, _path = _seed_dataset(app)
        observations = []

        def before_cursor_execute(_conn, _cursor, statement, _params, _ctx, _many):
            sql = ' '.join(statement.lower().split())
            if (not observations and 'from face_dataset_image' in sql
                    and 'order by' in sql and 'status' in sql):
                observations.append(dataset_activity.running(
                    dataset_id, ('bank_export',)))

        event.listen(db.engine, 'before_cursor_execute', before_cursor_execute)
        try:
            bank_id = banks.start_dataset_import(
                app, 'local', dataset_id, 'Reserved export')
        finally:
            event.remove(db.engine, 'before_cursor_execute', before_cursor_execute)

        assert observations == [True]
        assert db.session.get(ImageBank, bank_id) is not None
        _assert_export_released(dataset_id)


def test_live_export_refuses_edit_delete_batch_delete_and_second_export(
        app, client):
    """Every Dataset mutation sees the same exclusive generation guard."""
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDataset, FaceDatasetImage, ImageBank
        from app.services import dataset_activity
        from app.services import image_bank_service as banks

        dataset_id, image_id, _folder, path = _seed_dataset(app)
        before_banks = ImageBank.query.count()
        sources_root = str(banks.cfg.bank_sources_root())
        before_folders = set(os.listdir(sources_root)) if os.path.isdir(sources_root) else set()
        token = dataset_activity.begin_exclusive(
            dataset_id, 'bank_export', total=1, detail='copying to Bank')
        assert token is not None

    try:
        edit = client.post(
            f'/api/dataset/image/{image_id}/status', json={'status': 'reject'})
        assert edit.status_code == 409, edit.get_json()

        batch_delete = client.post(
            f'/api/dataset/{dataset_id}/images/batch',
            json={'ids': [image_id], 'action': 'delete'},
        )
        assert batch_delete.status_code == 409, batch_delete.get_json()

        delete = client.post(f'/api/dataset/{dataset_id}/delete')
        assert delete.status_code == 409, delete.get_json()

        second = client.post(
            '/api/bank/from-dataset',
            json={'dataset_id': dataset_id, 'name': 'Second export'},
        )
        assert second.status_code == 409, second.get_json()

        with app.app_context():
            assert db.session.get(FaceDataset, dataset_id) is not None
            assert db.session.get(FaceDatasetImage, image_id).status == 'keep'
            assert os.path.isfile(path)
            assert ImageBank.query.count() == before_banks
            after_folders = (
                set(os.listdir(sources_root)) if os.path.isdir(sources_root) else set())
            assert after_folders == before_folders
            activity = dataset_activity.get(dataset_id)
            assert activity is not None and activity['kind'] == 'bank_export'
    finally:
        dataset_activity.end(token)

    _assert_export_released(dataset_id)


def test_success_releases_the_dataset_export_reservation(app):
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage, ImageBank
        from app.services import image_bank_service as banks

        dataset_id, _image_id, _folder, _path = _seed_dataset(app)
        bank_id = banks.start_dataset_import(app, 'local', dataset_id, 'Success')
        bank = db.session.get(ImageBank, bank_id)
        assert bank is not None and os.path.isdir(bank.source_path)
        assert BankImage.query.filter_by(bank_id=bank_id).count() == 1
        _assert_export_released(dataset_id)


def test_route_normalizes_string_dataset_id_and_releases_activity(app, client):
    """A JSON numeric string must not create an unreachable string-key bucket."""
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage, ImageBank
        from app.services import dataset_activity

        dataset_id, _image_id, _folder, _path = _seed_dataset(app)

    response = client.post('/api/bank/from-dataset', json={
        'dataset_id': str(dataset_id), 'name': 'String id export',
    })
    assert response.status_code == 202, response.get_json()

    with app.app_context():
        bank_id = response.get_json()['id']
        assert db.session.get(ImageBank, bank_id) is not None
        assert BankImage.query.filter_by(bank_id=bank_id).count() == 1
        assert dataset_activity.get(dataset_id) is None
        assert dataset_activity._active == {}


@pytest.mark.parametrize('invalid', [True, False, 1.0, '1.0', '', None])
def test_route_rejects_invalid_dataset_id_without_creating_bank_or_folder(
        app, client, invalid):
    with app.app_context():
        from app.models import ImageBank
        from app.services import image_bank_service as banks

        _seed_dataset(app)
        before_banks = ImageBank.query.count()
        sources_root = str(banks.cfg.bank_sources_root())
        before_folders = (
            set(os.listdir(sources_root)) if os.path.isdir(sources_root) else set())

    response = client.post('/api/bank/from-dataset', json={
        'dataset_id': invalid, 'name': 'Invalid id export',
    })
    assert response.status_code == 400, response.get_json()
    assert 'positive integer' in response.get_json()['error']

    with app.app_context():
        assert ImageBank.query.count() == before_banks
        after_folders = (
            set(os.listdir(sources_root)) if os.path.isdir(sources_root) else set())
        assert after_folders == before_folders


def test_cancel_discards_destination_folder_row_and_activity(
        app, monkeypatch):
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDatasetImage, ImageBank
        from app.services import bank_jobs
        from app.services import image_bank_service as banks

        dataset_id, image_id, _folder, path = _seed_dataset(app)
        before_banks = ImageBank.query.count()
        created = {}
        real_create = banks._stage_import_bank

        def remember_create(user_id, name):
            bank = real_create(user_id, name)
            created.update(id=bank.id, folder=bank.source_path)
            return bank

        monkeypatch.setattr(banks, '_stage_import_bank', remember_create)
        monkeypatch.setattr(bank_jobs, 'cancelled', lambda _job: True)

        banks.start_dataset_import(app, 'local', dataset_id, 'Cancelled')

        assert db.session.get(ImageBank, created['id']) is None
        assert ImageBank.query.count() == before_banks
        assert not os.path.exists(created['folder'])
        assert db.session.get(FaceDatasetImage, image_id) is not None
        assert os.path.isfile(path)
        _assert_export_released(dataset_id)


def test_copy_failure_discards_destination_folder_row_and_activity(
        app, monkeypatch):
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDatasetImage, ImageBank
        from app.services import image_bank_service as banks

        dataset_id, image_id, _folder, path = _seed_dataset(app)
        before_banks = ImageBank.query.count()
        created = {}
        real_create = banks._stage_import_bank

        def remember_create(user_id, name):
            bank = real_create(user_id, name)
            created.update(id=bank.id, folder=bank.source_path)
            return bank

        def fail_read(*_args, **_kwargs):
            raise OSError('source read failed')

        monkeypatch.setattr(banks, '_stage_import_bank', remember_create)
        monkeypatch.setattr(banks, '_read_safe_bank_source_bytes', fail_read)

        banks.start_dataset_import(app, 'local', dataset_id, 'Failed copy')

        assert db.session.get(ImageBank, created['id']) is None
        assert ImageBank.query.count() == before_banks
        assert not os.path.exists(created['folder'])
        assert db.session.get(FaceDatasetImage, image_id) is not None
        assert os.path.isfile(path)
        _assert_export_released(dataset_id)


def test_launch_failure_discards_destination_folder_row_and_activity(
        app, monkeypatch):
    """Failure after row/folder creation but before worker ownership is atomic."""
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDatasetImage, ImageBank
        from app.services import image_bank_service as banks

        dataset_id, image_id, _folder, path = _seed_dataset(app)
        before_banks = ImageBank.query.count()
        created = {}
        real_create = banks._stage_import_bank

        def remember_create(user_id, name):
            bank = real_create(user_id, name)
            created.update(id=bank.id, folder=bank.source_path)
            return bank

        def fail_start(*_args, **_kwargs):
            raise RuntimeError('worker unavailable')

        monkeypatch.setattr(banks, '_stage_import_bank', remember_create)
        monkeypatch.setattr(banks.bank_jobs, 'start', fail_start)

        with pytest.raises(RuntimeError, match='worker unavailable'):
            banks.start_dataset_import(app, 'local', dataset_id, 'Launch failed')

        assert db.session.get(ImageBank, created['id']) is None
        assert ImageBank.query.count() == before_banks
        assert not os.path.exists(created['folder'])
        assert db.session.get(FaceDatasetImage, image_id) is not None
        assert os.path.isfile(path)
        _assert_export_released(dataset_id)
