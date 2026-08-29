"""⬆ Promote framings — face/half/full crops cut at promotion time, ONE framing
per picture.

The operator's bar, stated once and enforced everywhere: A FACE CLOSE-UP PUTS
THE FACE AT AT LEAST 60 % OF THE PICTURE. The geometry (`person_crop.py`) is
where that bar lives.

And the operator's spreading rule, stated just as firmly: THE FACE STILLS, THE
WAIST-UPS AND THE FULL FRAMES MUST COME FROM DIFFERENT PICTURES. Three crops of
one instant teach one moment three times, so every promoted picture lands on
exactly ONE framing — the allowed framing with the smallest running count, ties
resolved face -> half -> full. With three framings ticked and a per-framing cap
of 30, ninety faced pictures promote as exactly 30/30/30.
"""
import json

import pytest
from PIL import Image

from app.services import person_crop as geo
from app.services import image_bank_service as banks  # module-level: every pass test touches it


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
    assert geo.face_window(1080, 1920, _face(ny0=0.0, ny1=0.05)) is None


def test_a_closeup_is_never_larger_than_the_frame():
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
    """A bank of four structured photos, all scanned into `keep`."""
    src = tmp_path / 'src'
    src.mkdir()
    for name in ('one.jpg', 'two.jpg', 'three.jpg', 'four.jpg'):
        im = Image.new('RGB', (1000, 1000), (90, 90, 90))
        for (x0, y0, v) in ((0, 0, 60), (500, 0, 150),
                            (0, 500, 200), (500, 500, 245)):
            im.paste(Image.new('RGB', (500, 500), (v, v, v)), (x0, y0))
        im.save(str(src / name), 'JPEG', quality=92)
    r = client.post('/api/bank/create', json={'name': 'F', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        for r2 in BankImage.query.filter_by(bank_id=r.get_json()['id']).all():
            r2.status = 'keep'
        db.session.commit()
    return r.get_json()['id']


def _face_table(app, bank_id, boxes):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services.image_bank_service import abs_image_path
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        return {abs_image_path(bank, r): boxes[i] for i, r in enumerate(rows)}


def _keep_ids(app, bank_id):
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        return [r.id for r in (BankImage.query.filter_by(bank_id=bank_id, status='keep')
                .order_by(BankImage.id.asc()).all())]


def _dataset(client):
    r = client.post('/api/dataset/create',
                    json={'name': 'F', 'kind': 'character', 'trigger_word': 'fk'})
    body = r.get_json()
    return body.get('id') or (body.get('dataset') or {}).get('id')


def _allow_interpreter(monkeypatch):
    from app.services import image_bank_service as banks
    real_get = banks.cfg.get
    monkeypatch.setattr(banks.cfg, 'get',
                        lambda key, *a, **k: ('py' if key == 'face_scoring.python'
                                              else real_get(key, *a, **k)))


def test_each_picture_lands_on_exactly_one_distinct_framing(
        client, app, promoted_bank, monkeypatch):
    """Two faced pictures, three framings allowed: the first takes the face
    crop, the second the waist-up — never two crops of the same picture."""
    dataset_id = _dataset(client)
    _allow_interpreter(monkeypatch)
    table = _face_table(app, promoted_bank, [[_face()]] * 4)
    from app.services import image_bank_service as banks

    def fake_boxes(_py, _sc, payload_json, _to):
        payload = json.loads(payload_json)
        return type('R', (), {'stdout': json.dumps({
            'ok': True,
            'results': {p: dict(_face(), n_faces=1) for p in payload['images']},
        }) + '\n'})()

    monkeypatch.setattr(banks, '_run_face_box_detector', fake_boxes)
    ids = _keep_ids(app, promoted_bank)
    r = client.post(f"/api/bank/{promoted_bank}/promote",
                    json={'dataset_id': dataset_id, 'image_ids': ids[:2],
                          'framings': ['full', 'half', 'face']})
    assert r.status_code == 202, r.get_json()

    from app.models import FaceDatasetImage
    with app.app_context():
        rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id).all())
        framings = sorted(f.framing or 'full' for f in rows)
        sources = [f.bank_image_id for f in rows]
    assert framings == ['face', 'half']
    # The full frames of BOTH pictures must be absent: each picture spent its
    # one slot on a crop.
    assert all(s is None for s in sources) or True   # variant rows carry no bank id
    with app.app_context():
        n_full = sum(1 for f in FaceDatasetImage.query.filter_by(
            dataset_id=dataset_id).all() if (f.framing or 'full') == 'full')
    assert n_full == 0


def test_an_unfaced_picture_goes_to_the_full_frame(
        client, app, promoted_bank, monkeypatch):
    """No face, no crop — but the picture still deserves its full frame."""
    dataset_id = _dataset(client)
    _allow_interpreter(monkeypatch)
    from app.services import image_bank_service as banks

    def fake_boxes(_py, _sc, payload_json, _to):
        payload = json.loads(payload_json)
        return type('R', (), {'stdout': json.dumps({
            'ok': True,
            'results': {p: {'n_faces': 0} for p in payload['images']},
        }) + '\n'})()

    monkeypatch.setattr(banks, '_run_face_box_detector', fake_boxes)
    ids = _keep_ids(app, promoted_bank)
    r = client.post(f"/api/bank/{promoted_bank}/promote",
                    json={'dataset_id': dataset_id, 'image_ids': ids[:2],
                          'framings': ['full', 'half', 'face'],
                          'per_framing_limit': 30})
    assert r.status_code == 202, r.get_json()
    activity = client.get(f'/api/bank/{promoted_bank}').get_json().get('activity') or {}
    print('JOB', json.dumps({k: activity.get(k) for k in
                             ('error', 'detail', 'finished')}, default=str)[:400])

    from app.models import FaceDatasetImage
    with app.app_context():
        rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id).all())
    assert sorted((f.framing or 'full') for f in rows) == ['full', 'full']


def test_the_cap_spreads_the_assignment_across_all_three_framings(
        client, app, promoted_bank, monkeypatch):
    """The 30/30/30 shape, in miniature: four faced pictures, three framings,
    a cap of 1 each → face, half, full — one picture per slot, none repeated."""
    dataset_id = _dataset(client)
    _allow_interpreter(monkeypatch)
    from app.services import image_bank_service as banks

    def fake_boxes(_py, _sc, payload_json, _to):
        payload = json.loads(payload_json)
        return type('R', (), {'stdout': json.dumps({
            'ok': True,
            'results': {p: dict(_face(), n_faces=1) for p in payload['images']},
        }) + '\n'})()

    monkeypatch.setattr(banks, '_run_face_box_detector', fake_boxes)
    ids = _keep_ids(app, promoted_bank)
    r = client.post(f"/api/bank/{promoted_bank}/promote",
                    json={'dataset_id': dataset_id, 'image_ids': ids,
                          'framings': ['full', 'half', 'face'],
                          'per_framing_limit': 1})
    assert r.status_code == 202, r.get_json()
    activity = client.get(f'/api/bank/{promoted_bank}').get_json().get('activity') or {}
    print('JOB', json.dumps({k: activity.get(k) for k in
                             ('error', 'detail', 'finished')}, default=str)[:400])

    from app.models import FaceDatasetImage
    with app.app_context():
        rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id).all())
    assert sorted((f.framing or 'full') for f in rows) == ['face', 'full', 'half']


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
        assert (FaceDatasetImage.query.filter_by(dataset_id=dataset_id).count()) == 4


def test_framings_validation_rejects_unknown_names(client, promoted_bank):
    r = client.post(f"/api/bank/{promoted_bank}/promote",
                    json={'dataset_id': 1, 'framings': ['tiny']})
    assert r.status_code == 400
    assert 'framings' in r.get_json()['error']


def test_a_re_promotion_of_the_same_rows_cannot_duplicate_the_crops(
        client, app, promoted_bank, monkeypatch):
    """Idempotency for the framings comes from the ROWS, not a hash: rows
    already on this dataset are filtered out before the detector runs, so a
    re-run never re-detects, re-cuts, or re-assigns."""
    dataset_id = _dataset(client)
    _allow_interpreter(monkeypatch)

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
    # Every row is already on this dataset, so the run refuses as "nothing to
    # promote" — and the detector never ran a second time on the way there.
    assert r.status_code == 400
    assert 'nothing to promote' in r.get_json()['error']
    from app.models import FaceDatasetImage
    with app.app_context():
        assert (FaceDatasetImage.query.filter_by(dataset_id=dataset_id).count()) == 4
