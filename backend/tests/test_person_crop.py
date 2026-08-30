"""✂ Crop to person — the Bank pass built from the operator's own batch script.

The geometry (`services/person_crop.py`) is the script's rules verbatim — merge
overlapping detections, keep the largest person, pad 3 %, cut the tightest
window that keeps the SOURCE aspect ratio — and the geometry is where the
interesting cases live (a box on an edge, a box bigger than the picture, two
people in one frame), so it is tested as pure functions.

The PASS itself is tested through the routes with the detector injected: what
matters now is that a detected box is written into a BRAND-NEW bank named after
the source plus `-crop`, that a picture with no person is skipped (the new bank
is crop-only, never padded), that already-cleaned rows are not re-cropped, that
the SOURCE bank keeps no working-copy blob and no marker, and that an absent
interpreter is a 503 that names the fix.
"""
import json
import os

import pytest
from PIL import Image

from app.services import person_crop as geo


def _photo(size=1000, value=90):
    return Image.new('RGB', (size, size), (value, value, value))


# ── the geometry ──────────────────────────────────────────────────────────────

def test_iou_of_identical_and_disjoint_boxes():
    assert geo.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert geo.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_duplicate_detections_merge_and_distinct_people_survive():
    a = [0, 0, 100, 100]
    dup = [2, 2, 98, 98]
    other = [400, 400, 480, 480]
    merged = geo.merge_boxes([a, dup, other])
    assert len(merged) == 2
    assert merged[0] == [0, 0, 100, 100]
    assert merged[1] == [400, 400, 480, 480]


def test_largest_person_box_is_none_when_nobody_was_found():
    assert geo.largest_person_box([]) is None


def test_the_window_keeps_the_source_aspect_ratio_and_contains_the_box():
    box = (400, 500, 600, 700)
    x1, y1, x2, y2 = geo.ar_window(box, 1000, 1000)
    assert (x2 - x1) == (y2 - y1)
    assert x1 <= box[0] and box[2] <= x2
    assert y1 <= box[1] and box[3] <= y2


def test_a_tall_person_in_a_wide_frame_gets_a_wider_window_never_a_shorter_one():
    box = (700, 100, 900, 850)
    x1, y1, x2, y2 = geo.ar_window(box, 1600, 900)
    assert round((x2 - x1) / (y2 - y1), 2) == round(1600 / 900, 2)
    assert x1 <= 700 and 850 <= y2


def test_a_box_touching_an_edge_clamps_without_leaving_the_frame():
    box = (0, 0, 800, 900)
    assert geo.ar_window(box, 1600, 900) == (0, 0, 1600, 900)


def test_a_box_bigger_than_the_frame_crops_nothing():
    assert geo.ar_window((-50, -50, 1700, 950), 1600, 900) == (0, 0, 1600, 900)


# ── the pass, through the routes, detector injected ──────────────────────────

@pytest.fixture()
def bank_with_photos(client, tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    for name in ('one.jpg', 'two.jpg'):
        _photo().save(str(src / name), 'JPEG', quality=92)
    r = client.post('/api/bank/create', json={'name': 'P', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _inject_detector(monkeypatch, boxes_by_path):
    """Replace the detector subprocess with a table of {path: [boxes]}."""
    from app.services import image_bank_service as banks

    def fake_run(_python, _script, payload_json, _timeout):
        payload = json.loads(payload_json)
        results = {p: {'boxes': boxes_by_path.get(p, [])}
                   for p in payload['images']}
        return type('R', (), {'stdout': json.dumps(
            {'ok': True, 'results': results}) + '\n', 'returncode': 0})()

    monkeypatch.setattr(banks, '_run_person_detector', fake_run)


def _row_paths(app, bank_id):
    """[(id, watermark_clean_method, abs_path)] for every image of the bank."""
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services.image_bank_service import abs_image_path
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        out = []
        for r in (BankImage.query.filter_by(bank_id=bank_id)
                  .order_by(BankImage.id.asc()).all()):
            out.append((r.id, r.watermark_clean_method, abs_image_path(bank, r)))
        return out


def _new_bank_by_name(app, name):
    """The bank created with this name (a crop destination), or None."""
    from app.extensions import db
    from app.models import ImageBank
    with app.app_context():
        return db.session.query(ImageBank).filter_by(name=name).first()


def test_a_detected_person_becomes_a_new_bank_and_the_source_stays_put(
        client, app, bank_with_photos, monkeypatch):
    bank_id, src = bank_with_photos
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_person_crop_prereq', lambda: None)
    rows = _row_paths(app, bank_id)
    boxes = {path: [[10, 10, 900, 900]] for _iid, _m, path in rows}
    _inject_detector(monkeypatch, boxes)

    r = client.post(f'/api/bank/{bank_id}/crop-person', json={})
    assert r.status_code == 202, r.get_json()
    new_id = r.get_json().get('id')

    # A brand-new bank named after the source plus `-crop`, holding the crops.
    new_bank = _new_bank_by_name(app, 'P-crop')
    assert new_bank is not None
    assert new_bank.id == new_id
    from app.models import BankImage
    with app.app_context():
        new_rows = (BankImage.query.filter_by(bank_id=new_bank.id).all())
        assert len(new_rows) == 2
        for row in new_rows:
            assert row.status == 'keep'
            target = os.path.join(new_bank.source_path, row.relpath)
            assert os.path.isfile(target)
    # THE source bank keeps no working-copy blob and no marker.
    methods = {iid: m for iid, m, _p in _row_paths(app, bank_id)}
    assert all(m is None for m in methods.values())
    from app.services.image_bank_service import _bank_dir
    with app.app_context():
        clean_dir = _bank_dir(bank_id) / 'clean'
    assert not clean_dir.exists() or len(list(clean_dir.iterdir())) == 0
    for name in ('one.jpg', 'two.jpg'):
        assert (src / name).is_file()


def test_a_picture_with_no_person_is_skipped_not_copied(
        client, app, bank_with_photos, monkeypatch):
    bank_id, _src = bank_with_photos
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_person_crop_prereq', lambda: None)
    rows = _row_paths(app, bank_id)
    boxes = {path: ([[10, 10, 900, 900]] if iid == rows[0][0] else [])
             for iid, _m, path in rows}
    _inject_detector(monkeypatch, boxes)

    r = client.post(f'/api/bank/{bank_id}/crop-person', json={})
    assert r.status_code == 202
    new_bank = _new_bank_by_name(app, 'P-crop')
    from app.models import BankImage
    with app.app_context():
        new_rows = BankImage.query.filter_by(bank_id=new_bank.id).all()
        assert len(new_rows) == 1, 'crop-only: no person → not copied'
    # The source bank is untouched by the crop (its second image had no person,
    # so nothing was written beside it either).
    methods = {iid: m for iid, m, _p in _row_paths(app, bank_id)}
    assert all(m is None for m in methods.values())


def test_an_already_cleaned_image_is_not_re_cropped_into_a_bank(
        client, app, bank_with_photos, monkeypatch):
    bank_id, _src = bank_with_photos
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_person_crop_prereq', lambda: None)
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        row = (BankImage.query.filter_by(bank_id=bank_id)
               .order_by(BankImage.id.asc()).first())
        row.watermark_clean_method = 'crop'      # some earlier cleaning
        cleaned_id = row.id
        db.session.commit()
    # Give the NOT-cleaned row a person box; the cleaned row is excluded by
    # _person_crop_todo_clause, so the detector never even sees it read.
    rows = _row_paths(app, bank_id)
    the_other = [p for mid, m, p in rows if mid != cleaned_id][0]
    _inject_detector(monkeypatch, {the_other: [[10, 10, 900, 900]]})

    r = client.post(f'/api/bank/{bank_id}/crop-person', json={})
    assert r.status_code == 202
    # The cleaned row stays out of the crop pool; the other row has a box, so
    # the new bank holds exactly one crop and the cleaned row is untouched.
    new_bank = _new_bank_by_name(app, 'P-crop')
    with app.app_context():
        new_rows = BankImage.query.filter_by(bank_id=new_bank.id).all()
    assert len(new_rows) == 1
    methods = {iid: m for iid, m, _p in _row_paths(app, bank_id)}
    assert any(m == 'crop' for m in methods.values())


def test_a_missing_interpreter_is_a_503_that_names_the_fix(
        client, app, bank_with_photos, monkeypatch):
    bank_id, _src = bank_with_photos
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_person_crop_prereq',
                        lambda: 'Crop to person needs an interpreter — set the '
                                '✨ Score interpreter in Setup ▸ Quality tools')
    r = client.post(f'/api/bank/{bank_id}/crop-person', json={})
    assert r.status_code == 503
    assert 'interpreter' in r.get_json()['error']


def test_an_unknown_bank_follows_the_image_lane_s_400(client):
    r = client.post('/api/bank/999999/crop-person', json={})
    assert r.status_code == 400
    assert 'error' in r.get_json()


def test_a_pool_with_nothing_to_crop_is_a_400(client, tmp_path, monkeypatch):
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_person_crop_prereq', lambda: None)
    src = tmp_path / 'empty'
    src.mkdir()
    r = client.post('/api/bank/create', json={'name': 'E', 'folder': str(src)})
    bank_id = r.get_json()['id']
    r = client.post(f'/api/bank/{bank_id}/crop-person', json={})
    assert r.status_code == 400
    assert 'nothing to crop' in r.get_json()['error']
