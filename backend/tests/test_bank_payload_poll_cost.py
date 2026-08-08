"""⏱ What one workspace poll COSTS, and the guard it may not trade away.

Reported from a real 50 397-image bank (36 870 of them eligible): "I can't see my
banks, and I can't even stop the scan because I have no progress bar" — followed
by seven Stop clicks in 20 ms against a UI that answered nothing. Measured on a
copy of that bank:

    /api/bank/11/images?limit=60      114 ms
    /api/bank/11/facets               359 ms
    /api/banks                      1 341 ms
    /api/bank/11                   12 500 ms at rest, 28 858 ms during a pass

and 90 % of that 12.5 s was not the database. ``_semantic_eligible_paths`` called
``abs_image_path`` per row, which calls ``os.path.realpath`` TWICE per image — once
for the unchanging bank folder, once for the file — and on Windows each realpath
OPENS the file through ``nt._getfinalpathname``: 147 502 syscalls per poll. The
workspace polls that payload every 2 s while a job runs, so the requests stacked;
and that payload is what carries the ``activity`` block, hence the missing banner
and the unreachable Stop button.

Two independent defects, so two independent sets of tests:

  1. the poll must not walk the disk — pinned by SYSCALL COUNTS, because a
     stopwatch assertion is a flaky assertion;
  2. the containment guard that walk existed for must still hold. That is the
     one thing an optimisation here is not allowed to buy speed with, so it is
     asserted on the property (a relpath that escapes the bank folder is refused,
     and stays refused on the memoized second call), never on the mechanism.

And the banner itself no longer rides on the heaviest read of the page.
"""
import os

import pytest
from PIL import Image


def _uid():
    from app.config import LOCAL_USER
    return LOCAL_USER


def _bank(client, tmp_path, count=12, name='Poll cost'):
    source = tmp_path / 'source'
    source.mkdir()
    for index in range(count):
        Image.new('RGB', (32, 32), (10 + index * 7,) * 3).save(
            source / f'i{index:03d}.jpg')
    response = client.post('/api/bank/create',
                           json={'name': name, 'folder': str(source)})
    assert response.status_code == 200, response.get_json()
    return response.get_json()['id'], source


class _Counter:
    """Counts ``os.path.realpath`` — the call that opens a file on Windows."""

    def __init__(self, monkeypatch):
        self.paths = []
        real = os.path.realpath
        monkeypatch.setattr(
            os.path, 'realpath',
            lambda p, **k: (self.paths.append(str(p)), real(p, **k))[1])

    def __len__(self):
        return len(self.paths)


# --- 1. the poll must not walk the disk --------------------------------------
def test_first_poll_resolves_the_bank_folder_once_not_once_per_image(
        client, tmp_path, app, monkeypatch):
    """"Once for the base, once per image" — the same principle
    ``test_curation_selection_cost.py`` already pins for the curation pool, which
    this path simply never applied. It was doing TWO per image."""
    bank_id, _ = _bank(client, tmp_path, count=12)
    with app.app_context():
        from app.services import image_bank_service as banks
        banks.reset_poll_path_memo()
        bank = banks.get_bank(_uid(), bank_id)
        counter = _Counter(monkeypatch)
        total, paths = banks._semantic_eligible_paths(bank)
    assert total == 12 and len(paths) == 12
    assert len(counter) <= 13, (
        f'{len(counter)} realpath calls for 12 rows — the bank folder is being '
        f're-resolved per row')


def test_a_second_poll_costs_no_filesystem_call_at_all(
        client, tmp_path, app, monkeypatch):
    """THE regression. The workspace polls every 2 s and nothing about an
    unchanged row's location changed in between; on the real bank this is the
    difference between 147 502 syscalls per poll and none."""
    bank_id, _ = _bank(client, tmp_path, count=12)
    with app.app_context():
        from app.services import image_bank_service as banks
        banks.reset_poll_path_memo()
        bank = banks.get_bank(_uid(), bank_id)
        banks._semantic_eligible_paths(bank)          # first poll pays
        counter = _Counter(monkeypatch)
        _total, paths = banks._semantic_eligible_paths(bank)
    assert len(paths) == 12, 'the memoized poll lost images'
    assert len(counter) <= 1, (
        f'{len(counter)} realpath calls on a repeat poll — only the bank folder '
        f'itself may be re-resolved')


def test_the_repeat_poll_returns_exactly_the_same_paths(client, tmp_path, app):
    """An optimisation that returns a different set is not an optimisation: these
    paths are the membership key deciding which cached rows count as indexed, so
    dropping or altering one would silently reset the bank's ``ok / total``."""
    bank_id, _ = _bank(client, tmp_path, count=12)
    with app.app_context():
        from app.services import image_bank_service as banks
        banks.reset_poll_path_memo()
        bank = banks.get_bank(_uid(), bank_id)
        first = banks._semantic_eligible_paths(bank)
        second = banks._semantic_eligible_paths(bank)
        banks.reset_poll_path_memo()
        cold = banks._semantic_eligible_paths(bank)
    assert first == second == cold


def test_the_memo_is_dropped_when_the_bank_points_at_another_folder(
        client, tmp_path, app):
    """Keyed on the RESOLVED bank folder, so re-pointing a bank cannot serve
    paths under the old one."""
    bank_id, _ = _bank(client, tmp_path, count=4)
    moved = tmp_path / 'moved'
    moved.mkdir()
    with app.app_context():
        from app.extensions import db
        from app.services import image_bank_service as banks
        banks.reset_poll_path_memo()
        bank = banks.get_bank(_uid(), bank_id)
        before = banks._semantic_eligible_paths(bank)[1]
        for name in os.listdir(str(tmp_path / 'source')):
            Image.new('RGB', (32, 32)).save(moved / name)
        bank.source_path = str(moved)
        db.session.commit()
        after = banks._semantic_eligible_paths(bank)[1]
    assert before and after
    assert all(os.path.normcase(str(moved)) in os.path.normcase(p) for p in after), (
        'a re-pointed bank was served paths memoized under its previous folder')


# --- 2. the guard the walk existed for ---------------------------------------
@pytest.mark.parametrize('relpath', [
    os.path.join('..', 'escaped.jpg'),
    os.path.join('sub', '..', '..', 'escaped.jpg'),
])
def test_a_relpath_that_escapes_the_bank_folder_is_still_refused(
        client, tmp_path, app, relpath):
    """``_abs_under`` is a directory-escape guard, not a formality. Memoizing its
    VERDICT is only sound while the verdict is still 'no' — asserted on the
    property (the path is absent from what the poll reports), never on whether
    some resolver was called."""
    bank_id, source = _bank(client, tmp_path, count=3)
    Image.new('RGB', (32, 32), (9, 9, 9)).save(tmp_path / 'escaped.jpg')
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        from app.services import image_bank_service as banks
        banks.reset_poll_path_memo()
        row = BankImage.query.filter_by(bank_id=bank_id).order_by(
            BankImage.id.asc()).first()
        row.relpath = relpath
        db.session.commit()
        bank = banks.get_bank(_uid(), bank_id)
        first = banks._semantic_eligible_paths(bank)[1]
        second = banks._semantic_eligible_paths(bank)[1]
    outside = os.path.normcase(str(tmp_path / 'escaped.jpg'))
    for reported in (first, second):
        assert len(reported) == 2, f'expected 2 contained rows, got {reported}'
        assert all(os.path.normcase(str(source)) + os.sep
                   in os.path.normcase(p) + os.sep for p in reported)
        assert outside not in [os.path.normcase(p) for p in reported]


def test_a_refusal_is_remembered_as_a_refusal_not_dropped_from_the_memo(
        client, tmp_path, app, monkeypatch):
    """A rejected relpath must be CACHED as rejected. Caching only the successes
    would re-run the guard for the escaping row on every single poll — the cheap
    way to end up with an unbounded miss path that reintroduces the disk walk —
    and, worse, invites 'absent from the memo means allow'."""
    bank_id, _ = _bank(client, tmp_path, count=3)
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        from app.services import image_bank_service as banks
        banks.reset_poll_path_memo()
        row = BankImage.query.filter_by(bank_id=bank_id).order_by(
            BankImage.id.asc()).first()
        row.relpath = os.path.join('..', 'escaped.jpg')
        db.session.commit()
        bank = banks.get_bank(_uid(), bank_id)
        banks._semantic_eligible_paths(bank)
        counter = _Counter(monkeypatch)
        reported = banks._semantic_eligible_paths(bank)[1]
        _base, memo = banks._poll_path_memo(bank)
    assert len(reported) == 2
    assert memo[os.path.join('..', 'escaped.jpg')] is None
    assert len(counter) <= 2, (
        f'{len(counter)} realpath calls on a repeat poll — the refused row is '
        f'being re-resolved every time')


def test_abs_image_path_itself_is_never_memoized(client, tmp_path, app,
                                                 monkeypatch):
    """The memo serves ONE read-only counting path. Every caller that turns a
    path into a capability — serving bytes, cleaning, promoting, deleting — goes
    through ``abs_image_path``, which must keep paying the live guard on every
    call. This pins that the fast path did not leak into the general resolver."""
    bank_id, _ = _bank(client, tmp_path, count=3)
    with app.app_context():
        from app.models import BankImage
        from app.services import image_bank_service as banks
        banks.reset_poll_path_memo()
        bank = banks.get_bank(_uid(), bank_id)
        row = BankImage.query.filter_by(bank_id=bank_id).order_by(
            BankImage.id.asc()).first()
        banks.abs_image_path(bank, row)
        counter = _Counter(monkeypatch)
        again = banks.abs_image_path(bank, row)
    assert again is not None
    assert len(counter) >= 2, (
        'abs_image_path stopped re-checking containment — the poll memo must '
        'not be its resolver')


# --- 3. the banner no longer rides on the heaviest read of the page ----------
def test_activity_route_answers_the_job_without_building_the_payload(
        client, tmp_path, app, monkeypatch):
    """The Stop button's data is a handful of bytes. It must not be gated on the
    ~60 bank-wide aggregates of the workspace payload, which is precisely the
    call that was taking 28.9 s while the pass the user wanted to stop ran."""
    bank_id, _ = _bank(client, tmp_path, count=4)
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, 'bank_payload', lambda *a, **k: pytest.fail(
        'the activity route built the whole workspace payload'))
    with app.app_context():
        from app.services import bank_jobs
        bank_jobs.reset()
        job = bank_jobs.reserve(bank_id, 'score', total=100)
        bank_jobs.progress(job, 7)
    response = client.get(f'/api/bank/{bank_id}/activity')
    assert response.status_code == 200
    activity = response.get_json()['activity']
    assert activity['kind'] == 'score'
    assert activity['done'] == 7 and activity['total'] == 100
    assert activity['finished'] is False


def test_activity_route_does_not_touch_the_filesystem(client, tmp_path, app,
                                                      monkeypatch):
    """Its cost must not depend on the size of the bank at all — no source-folder
    re-walk, no path resolution, one indexed row."""
    bank_id, _ = _bank(client, tmp_path, count=6)
    counter = _Counter(monkeypatch)
    response = client.get(f'/api/bank/{bank_id}/activity')
    assert response.status_code == 200
    assert len(counter) == 0, f'the activity poll resolved paths: {counter.paths}'


def test_activity_route_is_404_on_a_bank_that_is_gone(client, app):
    assert client.get('/api/bank/999999/activity').status_code == 404


def test_activity_route_reports_no_job_as_null(client, tmp_path, app):
    """The banner's absence is a value, not an error — the client reads
    ``activity == null`` as 'nothing running'."""
    bank_id, _ = _bank(client, tmp_path, count=2)
    with app.app_context():
        from app.services import bank_jobs
        bank_jobs.reset()
    response = client.get(f'/api/bank/{bank_id}/activity')
    assert response.status_code == 200
    assert response.get_json() == {'activity': None}
