"""✂ Auto-crop and 🧽 Inpaint take the same scope as every other bank pass.

WHY THIS FILE EXISTS, AND WHY IT IS STRICTER THAN THE OTHER SCOPE TESTS. These
two levels do not compute a verdict — they produce a cleaned IMAGE, thousands at
a time (a real bank offered "Auto-crop (16 052)" and "Inpaint (16 507)" from one
click each, with no way to say WHICH images). Adding a dial to an action that
writes files is only safe if two things are nailed down:

  1. the default is unchanged. A body with no `statuses` and no `image_ids` must
     reach the service as None/None and walk exactly the pool it walked before
     this feature existed — that is `test_default_*` below, and it asserts the
     ARGUMENTS as well as the outcome, so a client that silently defaulted to
     "kept only" could not pass by accident;
  2. the scope INTERSECTS the flagged set and never widens it. The pool of both
     levels is "flagged, geometry attested, something to act on"; a scope can
     only ever pick inside it.
"""
import json
import os

import pytest
from PIL import Image


def _photo(size=1000, value=90):
    im = Image.new('RGB', (size, size), (value, value, value))
    for y in range(0, size, 40):
        for x in range(size):
            im.putpixel((x, y), (250, 250, 250))
    return im


# Same marks as test_bank_watermark_clean.py: a croppable border band, and a
# small off-centre mark that only the repaint level handles.
TOP_MARK = [0.1, 0.01, 0.4, 0.05]
OFFCENTER_MARK = [0.30, 0.30, 0.36, 0.36]


def _mkbank(client, tmp_path, files, name='WMScope'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        p = src / rel
        os.makedirs(p.parent, exist_ok=True)
        im.save(str(p), 'JPEG', quality=92)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _setup(app, bank_id, marks_and_status):
    """[(bbox, status)] applied in id order. Returns the ids, in the same order.

    The fingerprint is part of the state the detection pass leaves behind, and
    the cleaning pool requires it: a geometry never attested against the bytes on
    disk may not drive a write (see _clean_todo_clause)."""
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        ids = []
        for row, (bbox, status) in zip(rows, marks_and_status):
            row.watermark_state = 'detected'
            row.watermark_bbox = json.dumps(bbox)
            row.watermark_fingerprint = transfer.content_fingerprint_path(
                banks.abs_image_path(bank, row))
            row.status = status
            ids.append(row.id)
        db.session.commit()
        return ids


def _states(app, ids):
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        return [(db.session.get(BankImage, i).watermark_state,
                 db.session.get(BankImage, i).watermark_clean_method) for i in ids]


def _three_piles(client, tmp_path, app, mark=TOP_MARK):
    """One flagged image in each pile — the shape every scope question needs."""
    bank_id, src = _mkbank(client, tmp_path, {
        'k.jpg': _photo(), 'p.jpg': _photo(value=120), 'r.jpg': _photo(value=160)})
    ids = _setup(app, bank_id, [(mark, 'keep'), (mark, 'pending'), (mark, 'reject')])
    return bank_id, src, ids


def _fake_lama(monkeypatch):
    """Repaint = rewrite the staged copy. Records what it was handed."""
    from app.services import watermark_lama
    calls = []
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr(watermark_lama, 'resolve_device', lambda: 'cpu')

    def fake_batch(jobs, device='cpu'):
        out = {}
        for job in jobs:
            calls.append(job['image_path'])
            with Image.open(job['image_path']) as im:
                im.load()
                im.save(job['image_path'], 'WEBP', quality=92)
            out[job['image_path']] = (True, None)
        return out
    monkeypatch.setattr(watermark_lama, 'inpaint_batch', fake_batch)
    return calls


# --- 1. the default did not move --------------------------------------------
@pytest.mark.parametrize('endpoint, service', [
    ('crop', 'start_watermark_crop'),
    ('inpaint', 'start_watermark_inpaint'),
])
def test_default_request_reaches_the_service_as_none_none(client, tmp_path, app,
                                                          monkeypatch, endpoint,
                                                          service):
    """A body with no scope keys arrives as statuses=None, ids=None.

    Asserted on the ARGUMENTS rather than the outcome: a client defaulting to
    ['keep','pending'] would produce the same cleaned images on most banks and
    quietly change the request for everyone with a populated bin."""
    from app.routes import bank as bank_routes
    bank_id, _src, _ids = _three_piles(client, tmp_path, app)
    seen = {}
    original = getattr(bank_routes.banks, service)

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(bank_routes.banks, service, spy)

    r = client.post(f'/api/bank/{bank_id}/watermark/{endpoint}', json={})
    assert r.status_code in (202, 400, 503), r.get_json()
    assert seen.get('statuses', 'MISSING') is None
    assert seen.get('ids', 'MISSING') is None


def test_default_crop_walks_the_historical_pool_kept_and_undecided(client, tmp_path,
                                                                   app):
    """The pool itself, not just the arguments: kept + undecided, bin excluded —
    which is what `status != 'reject'` has always meant here."""
    bank_id, _src, ids = _three_piles(client, tmp_path, app)

    assert client.post(f'/api/bank/{bank_id}/watermark/crop').status_code == 202

    assert _states(app, ids) == [('cleaned', 'crop'), ('cleaned', 'crop'),
                                 ('detected', None)]


# --- 2. the four scopes actually aim the run --------------------------------
@pytest.mark.parametrize('statuses, expected', [
    (['keep'], [True, False, False]),
    (['pending'], [False, True, False]),
    (['reject'], [False, False, True]),
    (['keep', 'pending', 'reject'], [True, True, True]),
])
def test_crop_scope_picks_exactly_that_pile(client, tmp_path, app, statuses,
                                            expected):
    bank_id, _src, ids = _three_piles(client, tmp_path, app)

    r = client.post(f'/api/bank/{bank_id}/watermark/crop', json={'statuses': statuses})
    assert r.status_code == 202, r.get_json()

    assert [s == 'cleaned' for s, _m in _states(app, ids)] == expected


def test_inpaint_scope_picks_exactly_that_pile(client, tmp_path, app, monkeypatch):
    bank_id, _src, ids = _three_piles(client, tmp_path, app, mark=OFFCENTER_MARK)
    _fake_lama(monkeypatch)

    r = client.post(f'/api/bank/{bank_id}/watermark/inpaint',
                    json={'method': 'lama', 'statuses': ['pending']})
    assert r.status_code == 202, r.get_json()

    assert [s for s, _m in _states(app, ids)] == ['detected', 'cleaned', 'detected']


def test_a_selection_is_intersected_with_the_scope_never_widened(client, tmp_path,
                                                                 app):
    """The rule the caption pass set: ids ∩ scope. A selection naming an image in
    another pile cannot pull it in."""
    bank_id, _src, ids = _three_piles(client, tmp_path, app)

    r = client.post(f'/api/bank/{bank_id}/watermark/crop',
                    json={'statuses': ['keep'], 'image_ids': [ids[0], ids[1]]})
    assert r.status_code == 202, r.get_json()

    # ids[1] is undecided: named by the selection, dropped by the scope.
    assert [s == 'cleaned' for s, _m in _states(app, ids)] == [True, False, False]


def test_a_selection_alone_still_stays_inside_the_default_scope(client, tmp_path,
                                                               app):
    """No `statuses` at all: the historical filter still applies, so a selected
    REJECTED image is not cleaned. A selection is a narrowing, never a bypass."""
    bank_id, _src, ids = _three_piles(client, tmp_path, app)

    r = client.post(f'/api/bank/{bank_id}/watermark/crop',
                    json={'image_ids': [ids[0], ids[2]]})
    assert r.status_code == 202, r.get_json()

    assert [s == 'cleaned' for s, _m in _states(app, ids)] == [True, False, False]


def test_an_unknown_status_is_refused(client, tmp_path, app):
    bank_id, _src, _ids = _three_piles(client, tmp_path, app)
    r = client.post(f'/api/bank/{bank_id}/watermark/crop',
                    json={'statuses': ['kept']})
    assert r.status_code == 400
    assert 'invalid status' in (r.get_json() or {}).get('error', '')


# --- 3. the refusal and the end-of-pass line both name the scope -------------
def test_empty_scope_says_the_work_is_in_another_pile(client, tmp_path, app):
    """"Run the watermark scan first" on a bank whose flagged images are all in
    another pile sends the user to re-run a pass that already did its job."""
    bank_id, _src, _ids = _three_piles(client, tmp_path, app)
    # Empty the two default piles by cleaning them, leaving only the rejected one.
    assert client.post(f'/api/bank/{bank_id}/watermark/crop').status_code == 202

    r = client.post(f'/api/bank/{bank_id}/watermark/crop', json={'statuses': ['keep']})
    assert r.status_code == 400
    assert 'another pile' in (r.get_json() or {}).get('error', '')


def test_inpaint_empty_scope_says_the_work_is_in_another_pile(client, tmp_path,
                                                              app, monkeypatch):
    bank_id, _src, _ids = _three_piles(client, tmp_path, app, mark=OFFCENTER_MARK)
    _fake_lama(monkeypatch)

    r = client.post(f'/api/bank/{bank_id}/watermark/inpaint',
                    json={'method': 'lama', 'statuses': ['keep'],
                          'image_ids': [-1]})
    assert r.status_code == 400
    assert 'another pile' in (r.get_json() or {}).get('error', '')


def test_crop_bilan_names_what_the_scope_left_out(client, tmp_path, app):
    from app.services import bank_jobs
    bank_id, _src, _ids = _three_piles(client, tmp_path, app)

    with app.app_context():
        assert client.post(f'/api/bank/{bank_id}/watermark/crop',
                           json={'statuses': ['keep']}).status_code == 202
        detail = (bank_jobs.get(bank_id) or {}).get('detail') or ''
    assert '1 cropped' in detail
    assert '2 image(s) left out by the scope (1 undecided + 1 rejected)' in detail


def test_the_same_run_at_full_scope_has_nothing_to_report(client, tmp_path, app):
    """The discriminating half: widen the scope and the sentence must disappear.
    Without this, a note printed unconditionally would pass the test above."""
    from app.services import bank_jobs
    bank_id, _src, _ids = _three_piles(client, tmp_path, app)

    with app.app_context():
        assert client.post(
            f'/api/bank/{bank_id}/watermark/crop',
            json={'statuses': ['keep', 'pending', 'reject']}).status_code == 202
        detail = (bank_jobs.get(bank_id) or {}).get('detail') or ''
    assert '3 cropped' in detail
    assert 'left out by the scope' not in detail


def test_a_selection_run_says_nothing_about_the_scope(client, tmp_path, app):
    """The user named the images one by one — "left out by the scope" would be
    describing their own click back at them."""
    from app.services import bank_jobs
    bank_id, _src, ids = _three_piles(client, tmp_path, app)

    with app.app_context():
        assert client.post(f'/api/bank/{bank_id}/watermark/crop',
                           json={'image_ids': [ids[0]]}).status_code == 202
        detail = (bank_jobs.get(bank_id) or {}).get('detail') or ''
    assert 'left out by the scope' not in detail


def test_inpaint_bilan_names_what_the_scope_left_out(client, tmp_path, app,
                                                     monkeypatch):
    from app.services import bank_jobs
    bank_id, _src, _ids = _three_piles(client, tmp_path, app, mark=OFFCENTER_MARK)
    _fake_lama(monkeypatch)

    with app.app_context():
        assert client.post(f'/api/bank/{bank_id}/watermark/inpaint',
                           json={'method': 'lama',
                                 'statuses': ['keep']}).status_code == 202
        detail = (bank_jobs.get(bank_id) or {}).get('detail') or ''
    assert 'left out by the scope' in detail


# --- 4. the numbers the launch windows quote --------------------------------
def test_pass_scopes_price_both_levels_from_their_own_pool(client, tmp_path, app):
    """The counter and the run must read the SAME clause. Asserted by comparing
    the payload's per-pile table against the pool query itself, pile by pile —
    a second copy of the predicate would drift and this would catch it."""
    from app.services import image_bank_service as banks
    bank_id, _src, _ids = _three_piles(client, tmp_path, app)

    payload = client.get(f'/api/bank/{bank_id}').get_json()
    with app.app_context():
        for pile in ('keep', 'pending', 'reject'):
            pool = banks._clean_pool_query(bank_id, [pile])
            assert payload['pass_scopes']['watermark_inpaint']['todo'][pile] == \
                pool.count(), f'inpaint/{pile} priced from another pool'
            assert payload['pass_scopes']['watermark_crop']['todo'][pile] == \
                pool.filter(banks.BankImage.watermark_regions.is_(None)).count(), \
                f'crop/{pile} priced from another pool'


def test_crop_does_not_price_hand_masked_rows_it_will_skip(client, tmp_path, app):
    """A hand mask is the repaint level's material. Counting it for ✂ would offer
    work the crop job can only skip — and 🧽 must still count it."""
    from app.extensions import db
    from app.models import BankImage
    bank_id, _src, ids = _three_piles(client, tmp_path, app)
    with app.app_context():
        db.session.get(BankImage, ids[0]).watermark_regions = json.dumps(
            [[0.1, 0.1, 0.2, 0.2]])
        db.session.commit()

    scopes = client.get(f'/api/bank/{bank_id}').get_json()['pass_scopes']
    assert scopes['watermark_crop']['todo']['keep'] == 0
    assert scopes['watermark_inpaint']['todo']['keep'] == 1


def test_an_unattested_row_is_priced_by_neither_level(client, tmp_path, app):
    """A flagged row whose geometry was never attested against the bytes on disk
    cannot be cleaned (the job's own guard refuses it), so no window may offer
    it. This is the filter the pool gained after the scope work was first
    drafted; counting it would price work that is provably skipped."""
    from app.extensions import db
    from app.models import BankImage
    bank_id, _src, ids = _three_piles(client, tmp_path, app)
    with app.app_context():
        db.session.get(BankImage, ids[0]).watermark_fingerprint = None
        db.session.commit()

    scopes = client.get(f'/api/bank/{bank_id}').get_json()['pass_scopes']
    assert scopes['watermark_crop']['todo']['keep'] == 0
    assert scopes['watermark_inpaint']['todo']['keep'] == 0
    assert scopes['watermark_inpaint']['todo']['pending'] == 1
