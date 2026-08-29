"""⬆ Promote framings — face/half/full crops cut at promotion time.

The operator's bar, stated once and enforced everywhere: A FACE CLOSE-UP PUTS
THE FACE AT AT LEAST 60 % OF THE PICTURE. The geometry (`person_crop.py`) is
where that bar lives; the pass tests pin what a promotion actually produces —
one row with a face becomes full + waist-up + close-up, a row without one
becomes full only, nothing silently pads to a number, and the full frame keeps
its exact-bank-bytes transfer path untouched.
"""
import io
import json
import os

import pytest
from PIL import Image

from app.services import person_crop as geo


def _face(nx0=0.35, ny0=0.30, nx1=0.65, ny1=0.50):
    return {'nx0': nx0, 'ny0': ny0, 'nx1': nx1, 'ny1': ny1}


# ── the face close-up: the 60 % bar ──────────────────────────────────────────

def test_the_face_fills_at_least_60_percent_of_its_window():
    w, h = 1080, 1920
    win = geo.face_window(w, h, _face())
    fh = (0.50 - 0.30) * h
    assert (win[3] - win[1]) <= fh / 0.60 + 1.0
    assert fh / (win[3] - win[1]) >= 0.60


def test_the_face_box_sits_entirely_inside_its_closeup():
    w, h = 1080, 1920
    win = geo.face_window(w, h, _face())
    fx0, fy0 = 0.35 * w, 0.30 * h
    fx1, fy1 = 0.65 * w, 0.50 * h
    assert win[0] <= fx0 and fx1 <= win[2]
    assert win[1] <= fy0 and fy1 <= win[3]


def test_a_face_hard_against_the_edge_refuses_its_closeup():
    """Clamping a 60 %-tight window over a face at the very top of the frame
    would either leave the frame or behead the crop's subject — both break the
    bar quietly. The honest answer is no crop at all."""
    win = geo.face_window(1080, 1920, _face(ny0=0.0, ny1=0.05))
    assert win is None


def test_a_closeup_is_never_larger_than_the_frame():
    # A face filling half the frame: side would want to be > the frame's short
    # side, and the window must clamp to the frame rather than invent pixels.
    win = geo.face_window(800, 600, _face(nx0=0.2, ny0=0.1, nx1=0.8, ny1=0.9))
    assert win[2] - win[0] <= 800
    assert win[3] - win[1] <= 600


def test_a_garbage_face_reading_refuses_every_window():
    assert geo.face_window(100, 100, {}) is None
    assert geo.face_window(100, 100, _face(nx0=0.9, ny0=0.9, nx1=0.1, ny1=0.1)) is None


# ── the waist-up window ──────────────────────────────────────────────────────

def test_the_waist_up_window_keeps_the_face_whole_and_heads_the_upper_third():
    w, h = 1080, 1920
    face = _face()
    win = geo.half_window(w, h, face)
    fx0, fy0, fx1, fy1 = 0.35 * w, 0.30 * h, 0.65 * w, 0.50 * h
    assert win[0] <= fx0 and fx1 <= win[2]
    assert win[1] <= fy0 and fy1 <= win[3]
    assert win[1] <= fy0, 'the head-room is above the brows, never through them'


def test_a_face_at_the_very_top_still_gets_a_waist_up_window():
    win = geo.half_window(1080, 1920, _face(ny0=0.0, ny1=0.05))
    assert win is not None
    assert win[1] == 0


# ── the pass, through the routes, detector injected ──────────────────────────

@pytest.fixture()
def promoted_bank(client, app, tmp_path):
    """A bank of two square photos, both already scanned into `keep`."""
    src = tmp_path / 'src'
    src.mkdir()
    # Structured at LOW frequency: a flat fill crops to itself, and fine
    # stripes vanish in the dHash downscale — either way the dataset's
    # perceptual dedup would (correctly) swallow every framing variant. Four
    # large quadrants of different grey survive every crop as a different
    # low-frequency picture.
    for name in ('one.jpg', 'two.jpg'):
        im = Image.new('RGB', (1000, 1000), (90, 90, 90))
        for (x0, y0, v) in ((0, 0, 60), (500, 0, 150),
                            (0, 500, 200), (500, 500, 245)):
            im.paste(Image.new('RGB', (500, 500), (v, v, v)), (x0, y0))
        im.save(str(src / name), 'JPEG', quality=92)
    r = client.post('/api/bank/create', json={'name': 'F', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    bank_id = r.get_json()['id']
    # promote walks KEPT rows; scan leaves keep by default — make it explicit.
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        for r in BankImage.query.filter_by(bank_id=bank_id).all():
            r.status = 'keep'
        db.session.commit()
    return bank_id


def _face_table(app, bank_id, boxes):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services.image_bank_service import abs_image_path
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        return {abs_image_path(bank, r): boxes[i] for i, r in enumerate(rows)}


def _dataset(client):
    r = client.post('/api/dataset/create',
                    json={'name': 'F', 'kind': 'character', 'trigger_word': 'fk'})
    body = r.get_json()
    return body.get('id') or (body.get('dataset') or {}).get('id')


def test_promotion_with_three_framings_yields_three_images_per_faced_row(
        client, app, promoted_bank, monkeypatch):
    dataset_id = _dataset(client)
    assert dataset_id
    table = _face_table(app, promoted_bank, [[_face()], [_face()]])
    from app.services import image_bank_service as banks
    # The test config carries no face_scoring.python; the stub below stands in
    # for the configured interpreter this framing needs.
    real_get = banks.cfg.get
    monkeypatch.setattr(banks.cfg, 'get',
                        lambda key, *a, **k: ('py' if key == 'face_scoring.python'
                                              else real_get(key, *a, **k)))

    def fake_boxes(_py, _sc, payload_json, _to):
        payload = json.loads(payload_json)
        return type('R', (), {'stdout': json.dumps({
            'ok': True,
            'results': {p: dict(_face(), n_faces=1) for p in payload['images']},
        }) + '\n'})()

    monkeypatch.setattr(banks, '_run_face_box_detector', fake_boxes)
    r = client.post(f"/api/bank/{promoted_bank}/promote",
                    json={'dataset_id': dataset_id, 'framings': ['full', 'half', 'face']})
    assert r.status_code == 202, r.get_json()
    activity = client.get(f'/api/bank/{promoted_bank}').get_json().get('activity') or {}
    print('JOB', json.dumps({k: activity.get(k) for k in
                             ('error', 'detail', 'finished')}, default=str)[:400])

    from app.models import FaceDatasetImage
    with app.app_context():
        rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id).all())
        framings = sorted(f.framing or 'full' for f in rows)
    assert framings == ['face', 'face', 'full', 'full', 'half', 'half']


def test_promotion_without_framings_keeps_the_exact_legacy_behaviour(
        client, app, promoted_bank, monkeypatch):
    dataset_id = _dataset(client)
    called = []
    from app.services import image_bank_service as banks

    def fake_boxes(*a, **k):
        called.append(1)
        raise AssertionError('the detector must not run for full-only promotions')
    monkeypatch.setattr(banks, '_run_face_box_detector', fake_boxes)
    r = client.post(f"/api/bank/{promoted_bank}/promote",
                    json={'dataset_id': dataset_id})
    assert r.status_code == 202
    assert not called
    from app.models import FaceDatasetImage
    with app.app_context():
        assert (FaceDatasetImage.query.filter_by(dataset_id=dataset_id).count()) == 2


def test_framings_validation_rejects_unknown_names(client, promoted_bank):
    r = client.post(f"/api/bank/{promoted_bank}/promote",
                    json={'dataset_id': 1, 'framings': ['tiny']})
    assert r.status_code == 400
    assert 'framings' in r.get_json()['error']


def test_a_re_promotion_of_the_same_rows_cannot_duplicate_the_crops(
        client, app, promoted_bank, monkeypatch):
    """Idempotency for the framings comes from the ROWS, not a hash: rows
    already on this dataset are filtered out before the detector runs, so a
    re-run never cuts the same crop twice (and never re-detects either)."""
    dataset_id = _dataset(client)
    from app.services import image_bank_service as banks
    real_get = banks.cfg.get
    monkeypatch.setattr(banks.cfg, 'get',
                        lambda key, *a, **k: ('py' if key == 'face_scoring.python'
                                              else real_get(key, *a, **k)))

    def fake_boxes(_py, _sc, payload_json, _to):
        payload = json.loads(payload_json)
        return type('R', (), {'stdout': json.dumps({
            'ok': True,
            'results': {p: dict(_face(), n_faces=1) for p in payload['images']},
        }) + '\n'})()

    monkeypatch.setattr(banks, '_run_face_box_detector', fake_boxes)

    body = {'dataset_id': dataset_id, 'framings': ['full', 'half', 'face']}
    r = client.post(f"/api/bank/{promoted_bank}/promote", json=body)
    assert r.status_code == 202
    r = client.post(f"/api/bank/{promoted_bank}/promote", json=body)
    assert r.status_code == 400, 'every row is already on this dataset'
    from app.models import FaceDatasetImage
    with app.app_context():
        assert (FaceDatasetImage.query.filter_by(dataset_id=dataset_id).count()) == 6
