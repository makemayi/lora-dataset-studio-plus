"""Fail-closed cleanup for cancelled/failed internal Bank transfers."""

import os
import shutil


def _partial_bank(app, name='Partial destination'):
    from app.extensions import db
    from app.services import image_bank_service as banks

    bank = banks._stage_import_bank('local', name)
    bank_id, folder = bank.id, bank.source_path
    with open(os.path.join(folder, 'partial.bin'), 'wb') as stream:
        stream.write(b'partial copy')
    db.session.commit()
    return bank_id, folder


def test_partial_destination_falls_back_to_bounded_rmtree_when_trash_fails(
        app, monkeypatch):
    with app.app_context():
        from app.extensions import db
        from app.models import ImageBank
        from app.services import image_bank_service as banks

        bank_id, folder = _partial_bank(app)

        def refuse_trash(*_args, **_kwargs):
            raise OSError('trash unavailable')

        monkeypatch.setattr(banks.trash, 'send_to_trash', refuse_trash)

        assert banks._discard_promoted_bank('local', bank_id) is True
        assert db.session.get(ImageBank, bank_id) is None
        assert not os.path.lexists(folder)


def test_partial_destination_cleanup_failure_is_reported_on_the_job(
        app, monkeypatch):
    with app.app_context():
        from app.services import image_bank_service as banks

        bank_id, folder = _partial_bank(app, 'Stuck partial destination')
        native_rmtree = shutil.rmtree

        def refuse_trash(*_args, **_kwargs):
            raise OSError('trash unavailable')

        def refuse_partial_rmtree(path, *args, **kwargs):
            if os.path.normcase(os.path.realpath(path)) == os.path.normcase(
                    os.path.realpath(folder)):
                raise PermissionError('folder is locked')
            return native_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(banks.trash, 'send_to_trash', refuse_trash)
        monkeypatch.setattr(banks.shutil, 'rmtree', refuse_partial_rmtree)
        job = {'done': 0, 'total': 1}

        banks._fail_discarding_promoted_bank(
            job, 'local', bank_id,
            'Copy cancelled — the partial Bank was discarded.')

        assert os.path.isdir(folder)
        assert 'Cleanup also failed' in job['error']
        assert 'could not be removed' in job['detail']

        # The monkeypatch deliberately simulated a transient lock; leave no test
        # artifact behind after proving the failure signal.
        native_rmtree(folder)


def test_partial_cleanup_never_removes_a_folder_outside_the_import_root(
        app, tmp_path, monkeypatch):
    with app.app_context():
        from app.services import image_bank_service as banks

        outside = tmp_path / 'user-owned'
        outside.mkdir()
        called = []
        monkeypatch.setattr(
            banks.shutil, 'rmtree',
            lambda path, *args, **kwargs: called.append(str(path)))

        assert banks._remove_partial_import_folder(
            str(outside), context='test boundary') is False
        assert outside.is_dir()
        assert called == []
