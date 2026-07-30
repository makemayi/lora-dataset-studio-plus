""" Promote a selection into a NEW BANK — the second destination.

Promoting used to lead to exactly one place: a dataset. A dataset is a strict,
training-bound container, and isolating 200 candidates out of a 9 000-image dump
to work on them further is not the same intent. So the promote dialog gained a
second door, built on the SAME machinery as ``/bank/from-dataset``: a name, a
202 + background job, and the new bank's id back so the UI can jump to it.

The one rule this file exists to prove: **banks never share their files**. The
copy is a real copy — different path AND different inode — because the app
rewrites images in place (re-crop, watermark cleaning), and a shared inode would
turn "two independent banks" into one at the first edit. The source bank keeps
its rows, marked promoted with the destination that received them.
"""
import os

import pytest
from PIL import Image


def _img(path, colour=(10, 20, 30)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new('RGB', (64, 64), colour).save(path)


@pytest.fixture()
def source(app, tmp_path):
    """A bank over a folder of the user's own, three images, two of them kept.

    Returns (bank_id, folder, image_ids) and LEAVES the app context — Flask reuses
    an already-pushed app context for a test-client request, so a fixture that
    held one open would hand the route the fixture's own session, complete with
    its pre-promotion identity map."""
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        from app.services import image_bank_service as banks
        folder = tmp_path / 'dump'
        for i, n in enumerate(('a.png', 'b.png', 'c.png')):
            _img(str(folder / n), (i * 40, 20, 30))
        bank, _added = banks.create_bank('local', 'Big dump', str(folder))
        rows = (BankImage.query.filter_by(bank_id=bank.id)
                .order_by(BankImage.relpath).all())
        rows[0].status = 'keep'
        rows[1].status = 'keep'
        rows[2].status = 'reject'
        db.session.commit()
        out = bank.id, str(folder), [r.id for r in rows]
    return out


def _ids_of(bank_id):
    from app.models import BankImage
    return (BankImage.query.filter_by(bank_id=bank_id)
            .order_by(BankImage.relpath).all())


def test_promoting_a_selection_makes_a_bank_with_files_of_its_own(app, source):
    """Distinct paths AND distinct inodes: a hardlink would cost zero bytes and
    read as two banks right up to the first in-place rewrite."""
    bank_id, folder, ids = source
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage, ImageBank
        from app.services import image_bank_service as banks

        new_id = banks.start_bank_promote(app, 'local', bank_id, ids[:2],
                                          'Candidates')
        assert new_id != bank_id
        new_bank = db.session.get(ImageBank, new_id)
        assert new_bank.name == 'Candidates'
        assert os.path.realpath(new_bank.source_path) != os.path.realpath(folder)

        copies = _ids_of(new_id)
        assert [r.relpath for r in copies] == ['a.png', 'b.png']
        for r in copies:
            dest = os.path.join(new_bank.source_path, r.relpath)
            src = os.path.join(folder, r.relpath)
            assert os.path.isfile(dest) and os.path.isfile(src)
            assert r.file_size == os.path.getsize(dest) > 0
            # THE assertion: two files, not two names for one file.
            assert not os.path.samefile(src, dest)
            assert os.stat(src).st_ino != os.stat(dest).st_ino
            # A fresh bank starts un-triaged, like any other.
            assert r.status == 'pending'

        # The source keeps every row, and says where they went.
        kept = BankImage.query.filter_by(bank_id=bank_id).all()
        assert len(kept) == 3
        promoted = {r.id: r.promoted_bank_id for r in kept}
        assert promoted[ids[0]] == new_id and promoted[ids[1]] == new_id
        assert promoted[ids[2]] is None
        # ...without disturbing the dataset pointer, a separate destination.
        assert all(r.promoted_dataset_id is None for r in kept)


def test_curating_one_bank_never_touches_the_other(app, source):
    """The point of the copy. Rewriting a file on one side leaves the other's
    bytes alone, and a triage decision stays on the bank that made it."""
    bank_id, folder, ids = source
    with app.app_context():
        from app.extensions import db
        from app.models import ImageBank
        from app.services import image_bank_service as banks

        new_id = banks.start_bank_promote(app, 'local', bank_id, ids[:2], 'Candidates')
        new_bank = db.session.get(ImageBank, new_id)
        src = os.path.join(folder, 'a.png')
        dest = os.path.join(new_bank.source_path, 'a.png')
        before = open(src, 'rb').read()

        _img(dest, (200, 100, 50))                       # an in-place re-crop
        assert open(src, 'rb').read() == before          # source untouched
        assert open(dest, 'rb').read() != before

        _img(src, (5, 5, 5))                             # and the other way round
        assert open(dest, 'rb').read() != open(src, 'rb').read()

        copies = _ids_of(new_id)
        copies[0].status = 'reject'
        db.session.commit()
        assert _ids_of(bank_id)[0].status == 'keep'      # the source still keeps it


def test_a_copy_that_cannot_be_written_leaves_no_phantom_bank(app, source):
    """A full disk mid-copy must not leave a half-filled bank presenting as a
    finished one. Nothing is registered, and the failure is reported."""
    from unittest.mock import patch
    bank_id, _folder, ids = source
    with app.app_context():
        from app.models import BankImage, ImageBank
        from app.services import bank_jobs, image_bank_service as banks

        banks_before = ImageBank.query.count()
        with patch.object(banks.shutil, 'copy2',
                          side_effect=OSError(28, 'No space left on device')):
            banks.start_bank_promote(app, 'local', bank_id, ids[:2], 'Candidates')

        # The job body reports through bank_jobs (the runner owns exceptions), and
        # the message is the user's, not a traceback.
        err = bank_jobs.get(bank_id)['error']
        assert 'discarded' in err and 'free space' in err

        assert ImageBank.query.count() == banks_before
        assert ImageBank.query.filter_by(name='Candidates').first() is None
        # ...and the source was not marked as promoted to a bank that is gone.
        assert all(r.promoted_bank_id is None
                   for r in BankImage.query.filter_by(bank_id=bank_id))


def test_a_bank_born_from_another_is_not_reported_as_overlapping(app, source):
    """Overlap is nesting of FOLDERS. The copy lands in a folder of its own, so
    the two must not warn each other about shared files — they share none."""
    bank_id, _folder, ids = source
    with app.app_context():
        from app.services import image_bank_service as banks
        new_id = banks.start_bank_promote(app, 'local', bank_id, ids[:2], 'Candidates')
        assert banks.overlapping_banks('local', new_id) == []
        assert banks.overlapping_banks('local', bank_id) == []


def test_two_promotions_of_the_same_name_never_share_a_folder(app, source):
    bank_id, _folder, ids = source
    with app.app_context():
        from app.extensions import db
        from app.models import ImageBank
        from app.services import image_bank_service as banks
        a = banks.start_bank_promote(app, 'local', bank_id, ids[:1], 'Same')
        b = banks.start_bank_promote(app, 'local', bank_id, ids[1:2], 'Same')
        pa = db.session.get(ImageBank, a).source_path
        pb = db.session.get(ImageBank, b).source_path
        assert os.path.realpath(pa) != os.path.realpath(pb)


def test_an_empty_selection_means_every_kept_image(app, source):
    bank_id, _folder, _ids = source
    with app.app_context():
        from app.services import image_bank_service as banks
        new_id = banks.start_bank_promote(app, 'local', bank_id, [], 'All kept')
        assert [r.relpath for r in _ids_of(new_id)] == ['a.png', 'b.png']


def test_it_refuses_without_a_name_or_anything_to_copy(app, source):
    bank_id, _folder, ids = source
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        from app.services import image_bank_service as banks
        with pytest.raises(ValueError):
            banks.start_bank_promote(app, 'local', bank_id, ids[:1], '   ')
        with pytest.raises(ValueError):
            banks.start_bank_promote(app, 'local', 99999, ids[:1], 'Nope')
        BankImage.query.filter_by(bank_id=bank_id).update({'status': 'reject'})
        db.session.commit()
        with pytest.raises(ValueError):                  # nothing kept
            banks.start_bank_promote(app, 'local', bank_id, [], 'Nope')


def test_the_selection_size_is_the_real_weight_on_disk(app, source):
    """The confirmation announces bytes, not an order of magnitude — images are
    ~300 KB but video banks are three orders above."""
    bank_id, folder, ids = source
    with app.app_context():
        from app.services import image_bank_service as banks
        out = banks.selection_size('local', bank_id, ids[:2])
        expected = sum(os.path.getsize(os.path.join(folder, n))
                       for n in ('a.png', 'b.png'))
        assert out == {'count': 2, 'bytes': expected}
        # empty selection = every kept image, the same set 'promote all' copies
        assert banks.selection_size('local', bank_id, []) == out
        assert banks.selection_size('local', 99999, []) is None


def test_a_running_job_on_the_source_bank_is_a_409(client, app, source):
    bank_id, _folder, ids = source
    from app.services import bank_jobs
    bank_jobs.start(app, bank_id, 'scan', lambda job: None)
    bank_jobs._jobs[bank_id]['finished'] = False          # pretend it is still live
    r = client.post(f'/api/bank/{bank_id}/promote-to-bank',
                    json={'name': 'Candidates', 'image_ids': ids[:2]})
    assert r.status_code == 409


def test_a_busy_snapshot_that_expires_between_reads_still_refuses_cleanly(
        app, source, monkeypatch):
    """A TTL race is a 409, never a ``get(None)['kind']`` server error."""
    bank_id, _folder, ids = source
    with app.app_context():
        from app.services import bank_jobs, image_bank_service as banks

        monkeypatch.setattr(banks.bank_jobs, 'running', lambda _bank_id: True)
        monkeypatch.setattr(banks.bank_jobs, 'get', lambda _bank_id: None)
        with pytest.raises(bank_jobs.BankJobBusy, match='background'):
            banks.start_bank_promote(app, 'local', bank_id, ids[:1], 'Candidates')


def test_the_route_returns_202_and_the_new_bank_id(client, app, source):
    bank_id, _folder, ids = source
    r = client.post(f'/api/bank/{bank_id}/promote-to-bank',
                    json={'name': 'Candidates', 'image_ids': ids[:2]})
    assert r.status_code == 202
    body = r.get_json()
    assert body['ok'] is True and isinstance(body['id'], int)

    bad = client.post(f'/api/bank/{bank_id}/promote-to-bank', json={'name': ''})
    assert bad.status_code == 400


def test_the_grid_payload_carries_the_new_destination(client, app, source):
    """The ⬆ badge has to survive a destination that is a bank, without the
    dataset key changing meaning (it is stored in user databases)."""
    bank_id, _folder, ids = source
    with app.app_context():
        from app.services import image_bank_service as banks
        new_id = banks.start_bank_promote(app, 'local', bank_id, ids[:1], 'Candidates')
    r = client.get(f'/api/bank/{bank_id}/images')
    imgs = {i['relpath']: i for i in r.get_json()['images']}
    assert imgs['a.png']['promoted_bank_id'] == new_id
    assert imgs['a.png']['promoted_dataset_id'] is None
    assert imgs['b.png']['promoted_bank_id'] is None


def test_the_size_route_answers_for_a_selection(client, app, source):
    bank_id, _folder, ids = source
    r = client.get(f'/api/bank/{bank_id}/selection-size',
                   query_string={'ids': ','.join(str(i) for i in ids[:2])})
    assert r.status_code == 200
    body = r.get_json()
    assert body['count'] == 2 and body['bytes'] > 0
    assert client.get('/api/bank/99999/selection-size').status_code == 404
