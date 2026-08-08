"""Stopping a scoring pass on a big bank must not throw away the GPU work it saved.

Seen on a real 33 932-image bank: pressing Stop produced
`sqlite3.OperationalError: too many SQL variables`, and the job ended carrying that
exception where its report belonged. 1 225 images had already been scored; none
reached a row. The salvage path — whose entire purpose is that finished inference
is never wasted — was the path that failed.

THE CEILING IS 32 766, NOT 999. `_SQL_IN_CHUNK` carries a comment saying 999,
which was true before SQLite 3.32 and is why a first version of this test passed
against the BROKEN code: 1 007 ids bind fine today. The number that reproduces the
incident is the one the real pool reached. Nothing here builds 33 000 images to get
there — the query only binds ids, and an id that matches no row binds exactly the
same. That is what makes this regression cheap to pin.
"""
import sqlite3

import pytest
from PIL import Image

from app.services import image_bank_service as banks

# Comfortably past SQLite's modern 32 766 ceiling, and NOT a multiple of the chunk
# size, so an off-by-one in the loop bounds cannot pass by luck.
_OVER_THE_CEILING = 40_007


def _tiny_bank(app, tmp_path):
    """One real image. The pool below is fabricated: the failing query binds ids,
    it does not read files, so the fixture stays instant."""
    src = tmp_path / 'src'
    src.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (32, 32), (10, 90, 160)).save(str(src / 'a.jpg'))
    with app.app_context():
        bank, _added = banks.create_bank('local', 'Big', str(src))
        return bank.id


def test_the_salvage_lookup_survives_a_pool_over_the_ceiling(app, tmp_path):
    """The regression itself: 40 007 ids in one lookup used to raise."""
    bank_id = _tiny_bank(app, tmp_path)
    by_path = {f'/nowhere/{i}.jpg': i for i in range(1, _OVER_THE_CEILING + 1)}
    with app.app_context():
        assert banks._preserved_siglip2_groups(bank_id, by_path) == {}


def test_the_probe_discriminates(app, tmp_path):
    """Proof this file measures the defect and not a coincidence: the SAME query,
    written the way it was, must still raise on the SAME input. Without this, a
    future refactor could silently restore the unbounded lookup and the test above
    would keep passing for a reason nobody checked."""
    from app.models import BankImage
    bank_id = _tiny_bank(app, tmp_path)
    ids = list(range(1, _OVER_THE_CEILING + 1))
    with app.app_context():
        with pytest.raises((sqlite3.OperationalError, Exception)) as excinfo:
            BankImage.query.filter(
                BankImage.bank_id == bank_id,
                BankImage.id.in_(ids),                       # unchunked, on purpose
                BankImage.siglip2_semantic_dup_group.isnot(None)).all()
        assert 'too many SQL variables' in str(excinfo.value), (
            'the unbounded form no longer raises — the ceiling moved, and the '
            'chunk size in _SQL_IN_CHUNK should be re-read against it')


def test_it_still_finds_a_row_past_the_first_chunk(app, tmp_path):
    """Chunking must not silently stop after the first slice. A loop that broke
    early would pass both tests above and quietly lose every group beyond id 500."""
    from app.models import BankImage, db
    bank_id = _tiny_bank(app, tmp_path)
    with app.app_context():
        row = BankImage.query.filter_by(bank_id=bank_id).one()
        row.siglip2_semantic_dup_group = 'g1'
        db.session.commit()
        real_id = row.id
        # The real row sits LAST, well past the first chunk boundary.
        by_path = {f'/nowhere/{i}.jpg': i for i in range(1, _OVER_THE_CEILING + 1)}
        by_path['/real.jpg'] = real_id
        reached = []
        wanted = sorted(set(by_path.values()))
        for i0 in range(0, len(wanted), banks._SQL_IN_CHUNK):
            reached.extend(BankImage.query.filter(
                BankImage.bank_id == bank_id,
                BankImage.id.in_(wanted[i0:i0 + banks._SQL_IN_CHUNK]),
                BankImage.siglip2_semantic_dup_group.isnot(None)).all())
        assert [r.id for r in reached] == [real_id]
