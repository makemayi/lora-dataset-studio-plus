"""🗃️ Image bank — when the quality scan re-groups duplicates, and how it says so.

The owner's report was "Scan quality freezes LDS". What actually froze was the
TAIL of the pass: `rebuild_dup_groups` ran unconditionally, over every hashed row
of the bank, on a scan that had almost nothing to scan. Measured on the real
database: 50 397 rows, scan pool = 2, and the phase re-grouped the other 50 389
for 96 to 124 s with the bar sitting at 100 %.

Two rules come out of that, and both are easy to get wrong in opposite ways:

  • a scan that changed no stored hash must NOT re-group — its input is
    byte-for-byte what produced the groups already on screen;
  • but the 🎚 panel's "↻ Re-group duplicates" runs THROUGH the scan endpoint
    precisely because that tail existed, and its pool is empty by construction on
    an already-scanned bank. Gating on the pool alone turns that button into a
    no-op that reports success — which is worse than the freeze, because nothing
    tells the user their new dup_distance was ignored.

And the phase now reports progress and honours Stop, which it did not: there was
no `progress`, no `bump` and no `cancelled` anywhere in its body.
"""
import hashlib
from unittest.mock import patch

from PIL import Image


def _bank(banks, tmp_path, n=4, name='Dump'):
    src = tmp_path / name
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new('RGB', (500, 500), (11 * i, 80, 150)).save(str(src / f'a{i}.jpg'))
    bank, _added = banks.create_bank('local', name, str(src))
    return bank.id, src


def _run_scan(banks, bank_id, rescan=False, regroup=False):
    """Run the scan inline and report (regrouped?, progress details seen)."""
    calls = []
    details = []
    real = banks.rebuild_dup_groups

    def spy(bank_id_, *a, **kw):
        calls.append((a, kw))
        return real(bank_id_, *a, **kw)

    with patch.object(banks, 'rebuild_dup_groups', spy), \
         patch.object(banks.bank_jobs, 'cancelled', lambda job: False), \
         patch.object(banks.bank_jobs, 'bump', lambda job, n=1: None), \
         patch.object(banks.bank_jobs, 'progress',
                      lambda job, **kw: details.append(kw)):
        banks._scan_job(bank_id, rescan, regroup=regroup)({})
    return bool(calls), details


def test_a_second_scan_with_nothing_new_does_not_regroup(app, tmp_path):
    """The owner's case, in miniature: everything is already scanned, so the
    pool is empty and the grouping must not run at all."""
    from app.extensions import db
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id, _src = _bank(banks, tmp_path)
        first, _ = _run_scan(banks, bank_id)
        assert first, 'the FIRST scan must group what it just hashed'
        db.session.commit()

        second, details = _run_scan(banks, bank_id)
        assert not second, (
            're-grouped every hash of the bank for a scan that had nothing to '
            'scan — that is the freeze this fix is about')
        last = [d['detail'] for d in details if d.get('detail')][-1]
        assert 'already scanned' in last, (
            f'the pass must say why it did nothing, got {last!r}')


def test_a_scan_that_hashes_a_new_image_still_regroups(app, tmp_path):
    """The gate must not become "never": one new file is a new hash, and a new
    hash can join or create a duplicate group."""
    from app.extensions import db
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id, src = _bank(banks, tmp_path, name='Growing')
        _run_scan(banks, bank_id)
        db.session.commit()
        Image.new('RGB', (500, 500), (200, 30, 30)).save(str(src / 'new.jpg'))
        banks.refresh_bank('local', bank_id, force=True)
        db.session.commit()

        regrouped, _details = _run_scan(banks, bank_id)
        assert regrouped, 'a scan that hashed a new image must re-group'


def test_a_rescan_that_finds_the_same_hashes_does_not_regroup(app, tmp_path):
    """rescan walks every file again, but walking is not changing: if every hash
    comes back identical the grouping's input never moved."""
    from app.extensions import db
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id, _src = _bank(banks, tmp_path, name='Stable')
        _run_scan(banks, bank_id)
        db.session.commit()

        regrouped, details = _run_scan(banks, bank_id, rescan=True)
        assert not regrouped
        last = [d['detail'] for d in details if d.get('detail')][-1]
        assert 'no new hash' in last, f'got {last!r}'


def test_the_regroup_button_regroups_a_bank_with_nothing_to_scan(app, tmp_path):
    """What "↻ Re-group duplicates" sends. Without this the gate above would have
    silently disabled the only way to apply a new dup_distance."""
    from app.extensions import db
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id, _src = _bank(banks, tmp_path, name='Regroup')
        _run_scan(banks, bank_id)
        db.session.commit()

        regrouped, _details = _run_scan(banks, bank_id, regroup=True)
        assert regrouped, (
            'the ↻ Re-group duplicates button asked for a grouping and got '
            'nothing — a new dup_distance would be ignored without a word')


def test_the_scan_route_forwards_the_regroup_intent(app, client, tmp_path):
    """End to end from the POST body, because that is where the button's intent
    would be dropped without anyone noticing."""
    from app.services import image_bank_service as banks

    seen = {}
    with app.app_context():
        bank_id, _src = _bank(banks, tmp_path, name='Route')

    # The launch window added `statuses` and `image_ids` to every pass body, so
    # the route forwards two more keys. They are pinned here rather than absorbed
    # by a **kwargs: what matters is that a body WITHOUT them still reaches the
    # service as None/None — the request that shipped before the window existed
    # has to stay byte-for-byte the same run.
    def fake_start(app_, user_id, bid, rescan=False, regroup=False,
                   statuses=None, ids=None):
        seen.update(rescan=rescan, regroup=regroup, statuses=statuses, ids=ids)
        return {}

    with patch.object(banks, 'start_scan', fake_start):
        assert client.post(f'/api/bank/{bank_id}/scan',
                           json={'regroup': True}).status_code == 202
        assert seen == {'rescan': False, 'regroup': True,
                        'statuses': None, 'ids': None}
        assert client.post(f'/api/bank/{bank_id}/scan', json={}).status_code == 202
        assert seen == {'rescan': False, 'regroup': False,
                        'statuses': None, 'ids': None}
        # …and a scoped body arrives whole, not silently dropped on the floor.
        assert client.post(f'/api/bank/{bank_id}/scan',
                           json={'statuses': ['keep'], 'image_ids': [7, 9]}).status_code == 202
        assert seen == {'rescan': False, 'regroup': False,
                        'statuses': ['keep'], 'ids': [7, 9]}


# --- the phase is visible, and it stops --------------------------------------
def _bank_of_pairs(banks, db, tmp_path, pairs=40):
    from app.models import BankImage
    src = tmp_path / 'pairs'
    src.mkdir(parents=True, exist_ok=True)
    bank, _added = banks.create_bank('local', 'Pairs', str(src))
    for g in range(pairs):
        base = (g * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        for suffix, h in (('a', base), ('b', base ^ 1)):
            relpath = f'{g:04d}{suffix}.jpg'
            path = src / relpath
            Image.new(
                'RGB', (8, 8),
                ((g * 17) % 256, 40 if suffix == 'a' else 80, 150),
            ).save(str(path), 'JPEG')
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
            db.session.add(BankImage(
                bank_id=bank.id, relpath=relpath, status='pending',
                dhash=f'{h:016x}', analysis_fingerprint=fingerprint))
    db.session.commit()
    return bank.id


def test_the_regrouping_publishes_its_progress(app, tmp_path):
    """It used to publish nothing: the bar stayed where the scan left it — at
    100 % — under the words "grouping duplicates", for up to two minutes. A user
    reads that as a dead application, and reported it as one."""
    from app.extensions import db
    from app.services import image_bank_service as banks

    reported = []
    with app.app_context():
        bank_id = _bank_of_pairs(banks, db, tmp_path)
        with patch.object(banks.bank_jobs, 'cancelled', lambda job: False), \
             patch.object(banks.bank_jobs, 'progress',
                          lambda job, **kw: reported.append(kw)):
            banks.rebuild_dup_groups(bank_id, max_distance=1, job={})

    details = ' | '.join(str(r.get('detail')) for r in reported if r.get('detail'))
    assert 'comparing' in details, f'no comparison phase reported: {details}'
    assert 'writing' in details, f'no write phase reported: {details}'
    assert any(r.get('total') for r in reported), 'the phase never sized its work'
    assert any(r.get('done') for r in reported), 'the phase never moved'


def test_stopping_during_the_comparison_leaves_the_bank_untouched(app, tmp_path):
    """Stop before anything is written: the groups on screen stay exactly as they
    were, rather than being half cleared."""
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank_of_pairs(banks, db, tmp_path)
        BankImage.query.filter_by(bank_id=bank_id).update({'dup_group': 7})
        db.session.commit()

        with patch.object(banks.bank_jobs, 'cancelled', lambda job: True), \
             patch.object(banks.bank_jobs, 'progress', lambda job, **kw: None):
            assert banks.rebuild_dup_groups(bank_id, max_distance=1, job={}) == 0
        left = {r.dup_group for r in BankImage.query.filter_by(bank_id=bank_id)}
        assert left == {7}, f'a stopped comparison wrote to the bank: {left}'


def test_stopping_during_the_write_keeps_what_it_had_reached(app, tmp_path, monkeypatch):
    """Stop mid-write. The groups already committed stay — which is what the
    panel already tells the user ("the groups are whatever it had reached")."""
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    monkeypatch.setattr(banks, '_DUP_WRITE_ROWS', 10)   # → many small batches
    writing = {'now': False}

    def note(job, **kw):
        if 'writing' in str(kw.get('detail') or ''):
            writing['now'] = True

    with app.app_context():
        bank_id = _bank_of_pairs(banks, db, tmp_path, pairs=60)
        with patch.object(banks.bank_jobs, 'cancelled',
                          lambda job: writing['now']), \
             patch.object(banks.bank_jobs, 'progress', note):
            written = banks.rebuild_dup_groups(bank_id, max_distance=1, job={})
        assert 0 < written < 60, f'the write did not stop part-way (wrote {written})'
        landed = (BankImage.query.filter_by(bank_id=bank_id)
                  .filter(BankImage.dup_group.isnot(None)).count())
        assert landed == written * 2, (
            f'{landed} rows carry a group but {written} groups were reported')
