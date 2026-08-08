"""🗃️ Image bank — "already promoted into this dataset" is MEASURED, not remembered.

A bank used to mark an image promoted with a one-way flag. Delete that image in
the dataset and the flag stayed: the bank then advertised nothing promotable
into a dataset that held none of its images, which reads exactly like "the bank
lost my images" — the report these tests come from.

The bank now reads the back-link the promotion writes on the dataset row, so
deleting the image there gives the image back. Everything is asserted through
the service (the promote job body is run inline; in production it is a thread).
"""
import os
import random
from unittest.mock import patch

import pytest
from PIL import Image


@pytest.fixture()
def ds(app):
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDataset
        d = FaceDataset(user_id='local', name='Lola', trigger_word='Lola69382')
        db.session.add(d)
        db.session.commit()
        yield d


def _noise(path, seed):
    """A genuinely distinct image — flat colours collapse into one another under
    the import's perceptual dedup, which would hide what these tests measure."""
    rnd = random.Random(seed)
    img = Image.new('RGB', (600, 600))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
                 for _ in range(600 * 600)])
    img.save(path)


def _bank_of(folder, names, seed0=0):
    from app.services import image_bank_service as banks
    folder.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        _noise(folder / n, seed0 + i)
    bank, _added = banks.create_bank('local', 'Dump', str(folder))
    from app.models import BankImage
    rows = BankImage.query.filter_by(bank_id=bank.id).all()
    banks.set_status('local', bank.id, [r.id for r in rows], 'keep')
    return bank.id


def _promote(app, bank_id, dataset_id, ids=None):
    """Run the promote job inline so the assertions see a finished promotion."""
    from app.services import image_bank_service as banks
    with patch.object(banks.bank_jobs, 'start',
                      lambda _a, _b, _k, fn, total=0, **_kw: fn(object())), \
         patch.object(banks.bank_jobs, 'cancelled', lambda job: False), \
         patch.object(banks.bank_jobs, 'bump', lambda job, n=1: None), \
         patch.object(banks.bank_jobs, 'progress', lambda job, **kw: None):
        banks.start_promote(app, 'local', bank_id, ids or [], dataset_id)


def test_deleting_the_promoted_images_makes_them_promotable_again(app, ds, tmp_path):
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as fds
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank_of(tmp_path / 'src', ['a.png', 'b.png', 'c.png'])
        assert banks.promotable_count('local', bank_id, ds.id) == 3

        _promote(app, bank_id, ds.id)
        imgs = FaceDatasetImage.query.filter_by(dataset_id=ds.id).all()
        assert len(imgs) == 3
        assert all(i.bank_image_id is not None for i in imgs), 'promotion must record where it came from'
        assert banks.promotable_count('local', bank_id, ds.id) == 0

        for img in imgs:
            fds.delete_image('local', img.id)

        # THE REGRESSION: the bank has to offer them again — the dataset holds none.
        assert banks.promotable_count('local', bank_id, ds.id) == 3
        # …and the source files were never involved.
        assert sorted(os.listdir(tmp_path / 'src')) == ['a.png', 'b.png', 'c.png']


def test_deleting_one_image_gives_back_exactly_that_one(app, ds, tmp_path):
    from app.models import BankImage, FaceDatasetImage
    from app.services import face_dataset_service as fds
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank_of(tmp_path / 'src', ['a.png', 'b.png', 'c.png'])
        _promote(app, bank_id, ds.id)
        victim = FaceDatasetImage.query.filter_by(dataset_id=ds.id).first()
        origin = victim.bank_image_id
        fds.delete_image('local', victim.id)

        assert banks.promotable_count('local', bank_id, ds.id) == 1
        back = banks._promotable_query(bank_id, ds.id).all()
        assert [r.id for r in back] == [origin]
        # The other two are still held, so they are still not on offer.
        assert db_status(BankImage, origin) == 'keep'


def db_status(model, row_id):
    from app.extensions import db
    return db.session.get(model, row_id).status


def test_promoting_to_another_dataset_is_unaffected(app, ds, tmp_path):
    from app.extensions import db
    from app.models import FaceDataset
    from app.services import image_bank_service as banks

    with app.app_context():
        other = FaceDataset(user_id='local', name='Other', trigger_word='Other11111')
        db.session.add(other)
        db.session.commit()
        bank_id = _bank_of(tmp_path / 'src', ['a.png', 'b.png'])
        _promote(app, bank_id, ds.id)

        assert banks.promotable_count('local', bank_id, ds.id) == 0
        assert banks.promotable_count('local', bank_id, other.id) == 2   # per-target


def test_an_identical_image_from_another_bank_lands_independently(
        app, ds, tmp_path):
    """Transfer metadata is per source row, so byte twins never get merged."""
    from app.models import BankImage, FaceDatasetImage
    from app.services import image_bank_service as banks

    with app.app_context():
        folder = tmp_path / 'src'
        bank_id = _bank_of(folder, ['a.png'])
        _promote(app, bank_id, ds.id)
        assert FaceDatasetImage.query.filter_by(dataset_id=ds.id).count() == 1

        # A second bank over a COPY of the same photo — same pixels, new rows.
        second = tmp_path / 'src2'
        second.mkdir()
        (second / 'copy.png').write_bytes((folder / 'a.png').read_bytes())
        bank2, _n = banks.create_bank('local', 'Dump 2', str(second))
        rows = BankImage.query.filter_by(bank_id=bank2.id).all()
        banks.set_status('local', bank2.id, [r.id for r in rows], 'keep')
        assert banks.promotable_count('local', bank2.id, ds.id) == 1

        _promote(app, bank2.id, ds.id)
        assert FaceDatasetImage.query.filter_by(dataset_id=ds.id).count() == 2
        assert {row.bank_image_id for row in FaceDatasetImage.query.filter_by(
            dataset_id=ds.id).all()} == {
                BankImage.query.filter_by(bank_id=bank_id).one().id,
                rows[0].id,
            }
        assert banks.promotable_count('local', bank2.id, ds.id) == 0


def test_a_promotion_that_predates_the_backlink_still_counts(app, ds, tmp_path):
    """Legacy rows carry only the old one-way flag. They must keep being treated
    as promoted — an upgrade must not suddenly re-offer a user's whole bank."""
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank_of(tmp_path / 'src', ['a.png', 'b.png'])
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        rows[0].promoted_dataset_id = ds.id       # as an older version left it
        db.session.commit()

        assert banks.promotable_count('local', bank_id, ds.id) == 1
        assert banks.list_banks('local', dataset_id=ds.id)[0]['promotable'] == 1


def test_the_grid_stops_showing_a_promoted_badge_once_the_copy_is_deleted(
        app, ds, tmp_path):
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as fds
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank_of(tmp_path / 'src', ['a.png'])
        _promote(app, bank_id, ds.id)
        page = banks.list_images('local', bank_id)
        assert page['images'][0]['promoted_dataset_id'] == ds.id
        assert banks.bank_payload('local', bank_id)['counts']['promoted'] == 1

        img = FaceDatasetImage.query.filter_by(dataset_id=ds.id).first()
        fds.delete_image('local', img.id)

        page = banks.list_images('local', bank_id)
        assert page['images'][0]['promoted_dataset_id'] is None
        assert banks.bank_payload('local', bank_id)['counts']['promoted'] == 0
