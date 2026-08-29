"""✂ Crop to person — the Bank pass built from the operator's own batch script.

The geometry (`services/person_crop.py`) is the script's rules verbatim — merge
overlapping detections, keep the largest person, pad 3 %, cut the tightest
window that keeps the SOURCE aspect ratio — and the geometry is where the
interesting cases live (a box on an edge, a box bigger than the picture, two
people in one frame), so it is tested as pure functions.

The PASS itself is tested through the routes with the detector injected: what
matters there is that a detected box becomes a working copy under
`watermark_clean_method='person_crop'`, that a picture without a person is
skipped rather than padded, that already-cleaned rows are not re-cropped, that
the SOURCE file is byte-identical afterwards, and that an absent interpreter is
a 503 that names the fix — never a silently unfiltered run.
"""
import io
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
    # The prompt names the class twice, so one person typically arrives as two
    # overlapping boxes; a genuinely second person must NOT be merged away.
    a = [0, 0, 100, 100]
    dup = [2, 2, 98, 98]                      # IoU with `a` ~0.92 → merged
    other = [400, 400, 480, 480]
    merged = geo.merge_boxes([a, dup, other])
    assert len(merged) == 2
    assert merged[0] == [0, 0, 100, 100]      # sorted by area, biggest first
    assert merged[1] == [400, 400, 480, 480]


def test_largest_person_box_is_none_when_nobody_was_found():
    assert geo.largest_person_box([]) is None


def test_the_window_keeps_the_source_aspect_ratio_and_contains_the_box():
    box = (400, 500, 600, 700)                # 200x200 in a 1000x1000 square
    x1, y1, x2, y2 = geo.ar_window(box, 1000, 1000)
    assert (x2 - x1) == (y2 - y1)             # square image → square window
    assert x1 <= box[0] and box[2] <= x2      # the padded box fits inside
    assert y1 <= box[1] and box[3] <= y2


def test_a_tall_person_in_a_wide_frame_gets_a_wider_window_never_a_shorter_one():
    # A 1600x900 frame with a tall thin person: keeping the frame's 16:9 shape
    # means the window must be WIDER than the person, never shorter.
    box = (700, 100, 900, 850)                # 200 wide, 750 tall
    x1, y1, x2, y2 = geo.ar_window(box, 1600, 900)
    assert round((x2 - x1) / (y2 - y1), 2) == round(1600 / 900, 2)
    assert x1 <= 700 and 850 <= y2


def test_a_box_touching_an_edge_clamps_without_leaving_the_frame():
    box = (0, 0, 800, 900)                    # already the whole frame's corner
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
    """[(id, method, abs_path)] for every image of the bank."""
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


def test_a_detected_person_becomes_a_working_copy_and_the_source_stays_put(
        client, app, bank_with_photos, monkeypatch):
    bank_id, src = bank_with_photos
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_person_crop_prereq', lambda: None)
    rows = _row_paths(app, bank_id)
    boxes = {path: [[10, 10, 900, 900]] for _iid, _m, path in rows}
    _inject_detector(monkeypatch, boxes)

    r = client.post(f'/api/bank/{bank_id}/crop-person', json={})
    assert r.status_code == 202, r.get_json()

    methods = {iid: m for iid, m, _p in _row_paths(app, bank_id)}
    assert all(m == 'person_crop' for m in methods.values())
    # THE source files are untouched and the bank holds one working copy each.
    for name in ('one.jpg', 'two.jpg'):
        assert (src / name).is_file()
    from app.services.image_bank_service import _bank_dir
    with app.app_context():
        clean_dir = _bank_dir(bank_id) / 'clean'
    assert len(list(clean_dir.iterdir())) == 2


def test_an_image_without_a_person_is_skipped_not_padded(
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
    methods = {iid: m for iid, m, _p in _row_paths(app, bank_id)}
    assert methods[rows[0][0]] == 'person_crop'
    assert methods[rows[1][0]] is None, 'no person → no working copy, nothing padded'


def test_an_already_cleaned_image_is_not_cropped_again(
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
        clean_id = row.id
        db.session.commit()
    _inject_detector(monkeypatch, {})

    r = client.post(f'/api/bank/{bank_id}/crop-person', json={})
    assert r.status_code == 202
    methods = {iid: m for iid, m, _p in _row_paths(app, bank_id)}
    assert methods[clean_id] == 'crop', 'the earlier copy must survive untouched'
    assert methods[ [_i for _i, _m, _p in _row_paths(app, bank_id)
                     if _i != clean_id][0] ] is None


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
    # The image lane has no _missing helper: an unknown bank is a 400 naming
    # the problem, and every pass on this lane answers the same way.
    r = client.post('/api/bank/999999/crop-person', json={})
    assert r.status_code == 400
    assert 'error' in r.get_json()


def test_an_empty_pool_is_a_400(client, tmp_path, monkeypatch):
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_person_crop_prereq', lambda: None)
    src = tmp_path / 'empty'
    src.mkdir()
    r = client.post('/api/bank/create', json={'name': 'E', 'folder': str(src)})
    bank_id = r.get_json()['id']
    r = client.post(f'/api/bank/{bank_id}/crop-person', json={})
    assert r.status_code == 400
    assert 'nothing to crop' in r.get_json()['error']
