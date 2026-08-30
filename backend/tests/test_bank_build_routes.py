"""The two Bank->dataset build endpoints: read-only plan, and the build gate.

The plan endpoint WRITES NOTHING — the dialog polls it while the sliders move,
so a plan that mutated the dataset would turn preview into action. The build
endpoint's quota validation is the gate before anything expensive runs.
"""
import json

import pytest
from PIL import Image

from app.services import image_bank_service as banks


@pytest.fixture()
def seeded_bank(client, app, tmp_path):
    """A bank with two kept pictures + an empty target dataset."""
    src = tmp_path / 'src'
    src.mkdir()
    for name in ('one.jpg', 'two.jpg'):
        im = Image.new('RGB', (1000, 1000), (90, 90, 90))
        im.save(str(src / name), 'JPEG', quality=92)
    r = client.post('/api/bank/create', json={'name': 'B', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    bank_id = r.get_json()['id']
    # Scan imports land as 'pending'; a build's quota allocator walks KEEP rows.
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        for row in BankImage.query.filter_by(bank_id=bank_id).all():
            row.status = 'keep'
        db.session.commit()
    r = client.post('/api/dataset/create',
                    json={'name': 'D', 'kind': 'character', 'trigger_word': 'db'})
    assert r.status_code == 200, r.get_json()
    dataset_id = r.get_json()['id']
    return bank_id, dataset_id


def test_the_plan_endpoint_is_read_only(client, app, seeded_bank):
    bank_id, dataset_id = seeded_bank
    from app.extensions import db
    from app.models import FaceDatasetImage
    with app.app_context():
        before = FaceDatasetImage.query.filter_by(dataset_id=dataset_id).count()
    r = client.post(f'/api/bank/{bank_id}/promote/plan',
                    json={'dataset_id': dataset_id,
                          'quotas': {'face': 3, 'half': 3, 'full': 3}})
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) >= {'usable', 'with_face_box', 'counts', 'total',
                         'reused', 'shortfall'}
    with app.app_context():
        after = FaceDatasetImage.query.filter_by(dataset_id=dataset_id).count()
    assert after == before, 'a plan writes no row'


def test_the_plan_endpoint_validates_its_body(client, seeded_bank):
    bank_id, dataset_id = seeded_bank
    r = client.post(f'/api/bank/{bank_id}/promote/plan',
                    json={'dataset_id': dataset_id, 'quotas': {'tiny': 3}})
    assert r.status_code == 400
    assert 'quota' in r.get_json()['error'].lower()


def test_the_build_endpoint_rejects_a_bad_quota(client, seeded_bank):
    bank_id, dataset_id = seeded_bank
    r = client.post(f'/api/bank/{bank_id}/build',
                    json={'dataset_id': dataset_id, 'quotas': {'face': -1}})
    assert r.status_code == 400
    assert 'quota' in r.get_json()['error'].lower()


def test_the_build_endpoint_refuses_crops_without_the_interpreter(
        client, seeded_bank):
    """The design's edge case: `face_scoring.python` unset and the quotas ask
    for a crop — refused with the message that names the setting, verbatim.
    The test config has no interpreter, which is exactly the shipped state."""
    bank_id, dataset_id = seeded_bank
    r = client.post(f'/api/bank/{bank_id}/build',
                    json={'dataset_id': dataset_id, 'quotas': {'face': 1}})
    assert r.status_code == 400
    error = r.get_json()['error']
    assert 'face scoring interpreter' in error
    assert 'face_scoring.python' in error


def test_the_build_endpoint_accepts_a_valid_body(client, app, seeded_bank,
                                                 monkeypatch):
    """A valid build launches (202) without Topaz configured — the upscale is
    skipped with a note, and the import itself is what counts."""
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import dataset_generation_service as dgs
    from app.services.image_bank_service import BankImage
    # The quotas ask for crops, so the detector's interpreter must be
    # configured; the stub below replaces the subprocess entirely.
    real_get = banks.cfg.get
    monkeypatch.setattr(banks.cfg, 'get',
                        lambda key, *a, **k: ('C:/py/python.exe'
                                              if key == 'face_scoring.python'
                                              else real_get(key, *a, **k)))

    def fake_boxes(_py, _sc, payload_json, _to):
        payload = json.loads(payload_json)
        return type('R', (), {'stdout': json.dumps({
            'ok': True,
            'results': {p: {'n_faces': 0} for p in payload['images']},
        }) + '\n'})()

    monkeypatch.setattr(banks, '_run_face_box_detector', fake_boxes)
    monkeypatch.setattr(dgs, 'topaz_upscale_replace_batch',
                        lambda *a, **k: {'queued': 0, 'skipped': 0, 'job_id': None})
    bank_id, dataset_id = seeded_bank
    r = client.post(f'/api/bank/{bank_id}/build',
                    json={'dataset_id': dataset_id,
                          'quotas': {'face': 1, 'half': 1, 'full': 2}})
    assert r.status_code == 202, r.get_json()

    with app.app_context():
        rows = (FaceDatasetImage.query
                .filter_by(dataset_id=dataset_id)
                .order_by(FaceDatasetImage.id.asc()).all())
        counts = {}
        for row in rows:
            key = row.framing or 'full'
            counts[key] = counts.get(key, 0) + 1
    # Both pictures have no face (the stub says so), so both land on the full
    # frame — the quotas' face/half slots simply go unfilled.
    assert counts == {'full': 2}
