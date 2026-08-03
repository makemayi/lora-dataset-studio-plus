"""Angle reference photos for Krea 2 Edit (pose slots): left45/right45 now,
back/left90/right90 reserved for a later wave. See
docs/superpowers/specs/2026-08-03-krea-pose-slots-design.md.
"""
import io
import os

from app.extensions import db
from app.services import face_dataset_service as svc


def _dataset_with_ref():
    ds = svc.create_dataset('local', 'ds', 'trg')
    folder = svc._dataset_path(ds.id)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'ref.webp'), 'wb') as fh:
        fh.write(svc.normalize_to_webp(_png(512, 512)))
    ds.ref_filename = 'ref.webp'
    db.session.commit()
    return ds


def _png(w, h, color=(120, 40, 40)):
    import io
    from PIL import Image
    b = io.BytesIO()
    Image.new('RGB', (w, h), color).save(b, 'PNG')
    return b.getvalue()


def test_ref_pose_slot_table_exists_with_no_migration(app):
    """New table -> created by db.create_all() alone (see models.CheckpointNote
    for the precedent this follows) — no entry in _SCHEMA_ADDITIONS needed."""
    with app.app_context():
        from app.models import RefPoseSlot
        ds = _dataset_with_ref()
        row = RefPoseSlot(dataset_id=ds.id, pose_key='left45',
                          filename='x.webp', enabled=False)
        db.session.add(row)
        db.session.commit()
        fetched = RefPoseSlot.query.filter_by(dataset_id=ds.id, pose_key='left45').first()
        assert fetched is not None
        assert fetched.filename == 'x.webp'
        assert fetched.enabled is False
        assert fetched.original_filename is None


def test_ref_pose_slot_unique_per_dataset_and_pose_key(app):
    from app.models import RefPoseSlot
    from sqlalchemy.exc import IntegrityError
    with app.app_context():
        ds = _dataset_with_ref()
        db.session.add(RefPoseSlot(dataset_id=ds.id, pose_key='left45', filename='a.webp'))
        db.session.commit()
        db.session.add(RefPoseSlot(dataset_id=ds.id, pose_key='left45', filename='b.webp'))
        try:
            db.session.commit()
            assert False, 'expected a UniqueConstraint violation'
        except IntegrityError:
            db.session.rollback()


def test_set_pose_slot_uploads_disabled_by_default(app):
    with app.app_context():
        ds = _dataset_with_ref()
        fn = svc.set_pose_slot('local', ds.id, 'left45', _png(800, 800))
        assert fn
        folder = svc._dataset_path(ds.id)
        assert os.path.isfile(os.path.join(folder, fn))
        rows = svc.pose_slot_rows(ds)
        assert rows['left45'].filename == fn
        assert rows['left45'].enabled is False
        assert svc.enabled_pose_slot_paths(ds) == {}   # not enabled yet


def test_set_pose_slot_rejects_unknown_pose_key(app):
    with app.app_context():
        ds = _dataset_with_ref()
        try:
            svc.set_pose_slot('local', ds.id, 'upside_down', _png(400, 400))
            assert False, 'expected ValueError'
        except ValueError:
            pass


def test_enabling_a_slot_makes_it_appear_in_enabled_paths(app):
    with app.app_context():
        ds = _dataset_with_ref()
        fn = svc.set_pose_slot('local', ds.id, 'right45', _png(800, 800))
        assert svc.set_pose_slot_enabled('local', ds.id, 'right45', True) is True
        paths = svc.enabled_pose_slot_paths(ds)
        assert list(paths.keys()) == ['right45']
        assert paths['right45'] == os.path.join(svc._dataset_path(ds.id), fn)


def test_enabling_a_slot_with_no_upload_yet_is_refused(app):
    with app.app_context():
        ds = _dataset_with_ref()
        assert svc.set_pose_slot_enabled('local', ds.id, 'left45', True) is False


def test_re_uploading_a_slot_keeps_its_enabled_state_and_trashes_the_old_file(app):
    with app.app_context():
        ds = _dataset_with_ref()
        fn1 = svc.set_pose_slot('local', ds.id, 'left45', _png(800, 800))
        svc.set_pose_slot_enabled('local', ds.id, 'left45', True)
        fn2 = svc.set_pose_slot('local', ds.id, 'left45', _png(900, 900, (5, 5, 5)))
        assert fn2 != fn1
        folder = svc._dataset_path(ds.id)
        assert not os.path.exists(os.path.join(folder, fn1))   # trashed, not left behind
        assert os.path.isfile(os.path.join(folder, fn2))
        assert svc.pose_slot_rows(ds)['left45'].enabled is True   # preserved across re-upload


def test_crop_pose_slot_can_widen_back_out(app):
    from PIL import Image
    with app.app_context():
        ds = _dataset_with_ref()
        svc.set_pose_slot('local', ds.id, 'left45', _png(2000, 1500))
        folder = svc._dataset_path(ds.id)
        fn = svc.pose_slot_rows(ds)['left45'].filename
        path = os.path.join(folder, fn)

        assert svc.crop_pose_slot('local', ds.id, 'left45', 100, 100, 300, 300) is True
        with Image.open(path) as im:
            assert im.size == (300, 300)
        # Wider box in the ORIGINAL's pixel space — only works if the original
        # full frame was kept untouched by the first crop.
        orig = svc.pose_slot_rows(ds)['left45'].original_filename
        with Image.open(os.path.join(folder, orig)) as im:
            assert im.size == (2000, 1500)   # never overwritten
        assert svc.crop_pose_slot('local', ds.id, 'left45', 0, 0, 2000, 1500) is True
        with Image.open(path) as im:
            assert im.size == (1024, 768)   # capped at 1024 long side


def test_mirror_pose_slot_flips_the_current_file_in_place(app):
    from PIL import Image, ImageChops
    with app.app_context():
        ds = _dataset_with_ref()
        svc.set_pose_slot('local', ds.id, 'right45', _png(200, 100, (200, 0, 0)))
        folder = svc._dataset_path(ds.id)
        fn = svc.pose_slot_rows(ds)['right45'].filename
        path = os.path.join(folder, fn)
        with Image.open(path) as im:
            before = im.convert('RGB').copy()

        assert svc.mirror_pose_slot('local', ds.id, 'right45') is True

        with Image.open(path) as im:
            after = im.convert('RGB')
            mirrored_before = before.transpose(Image.FLIP_LEFT_RIGHT)
            assert ImageChops.difference(after, mirrored_before).getbbox() is None
        assert svc.pose_slot_rows(ds)['right45'].filename == fn   # same filename, in place


def test_mirror_pose_slot_on_missing_slot_returns_false(app):
    with app.app_context():
        ds = _dataset_with_ref()
        assert svc.mirror_pose_slot('local', ds.id, 'left45') is False


def test_remove_pose_slot_deletes_row_and_files(app):
    with app.app_context():
        ds = _dataset_with_ref()
        svc.set_pose_slot('local', ds.id, 'left45', _png(800, 800))
        folder = svc._dataset_path(ds.id)
        row = svc.pose_slot_rows(ds)['left45']
        fn, orig = row.filename, row.original_filename

        assert svc.remove_pose_slot('local', ds.id, 'left45') is True
        assert not os.path.exists(os.path.join(folder, fn))
        assert not os.path.exists(os.path.join(folder, orig))
        assert 'left45' not in svc.pose_slot_rows(ds)


def test_remove_pose_slot_on_missing_slot_returns_false(app):
    with app.app_context():
        ds = _dataset_with_ref()
        assert svc.remove_pose_slot('local', ds.id, 'left45') is False


def test_dataset_payload_lists_all_five_pose_keys_even_when_unused(app):
    with app.app_context():
        ds = _dataset_with_ref()
        svc.set_pose_slot('local', ds.id, 'left45', _png(800, 800))
        svc.set_pose_slot_enabled('local', ds.id, 'left45', True)
        payload = svc.dataset_payload('local', ds.id)
        assert set(payload['pose_slots'].keys()) == set(svc.POSE_SLOT_KEYS)
        left = payload['pose_slots']['left45']
        assert left['filename'] and left['enabled'] is True
        right = payload['pose_slots']['right45']
        assert right == {'filename': None, 'original_filename': '', 'enabled': False}


def test_pose_slot_upload_route(client, app):
    with app.app_context():
        ds = _dataset_with_ref()
        dsid = ds.id
    r = client.post(f'/api/dataset/{dsid}/ref/pose/left45',
                    data={'file': (io.BytesIO(_png(800, 800)), 'p.png')},
                    content_type='multipart/form-data')
    assert r.status_code == 200 and r.get_json()['ok'] is True

    r = client.post(f'/api/dataset/{dsid}/ref/pose/upsidedown',
                    data={'file': (io.BytesIO(_png(800, 800)), 'p.png')},
                    content_type='multipart/form-data')
    assert r.status_code == 400


def test_pose_slot_enable_and_delete_routes(client, app):
    with app.app_context():
        ds = _dataset_with_ref()
        dsid = ds.id
    client.post(f'/api/dataset/{dsid}/ref/pose/left45',
                data={'file': (io.BytesIO(_png(800, 800)), 'p.png')},
                content_type='multipart/form-data')

    r = client.post(f'/api/dataset/{dsid}/ref/pose/left45/enable', json={'enabled': True})
    assert r.status_code == 200 and r.get_json()['ok'] is True

    r = client.post(f'/api/dataset/{dsid}/ref/pose/right45/enable', json={'enabled': True})
    assert r.status_code == 404   # nothing uploaded there yet

    r = client.post(f'/api/dataset/{dsid}/ref/pose/left45/delete')
    assert r.status_code == 200 and r.get_json()['ok'] is True

    r = client.post(f'/api/dataset/{dsid}/ref/pose/left45/delete')
    assert r.status_code == 404   # already gone


def test_pose_slot_crop_and_mirror_routes(client, app):
    with app.app_context():
        ds = _dataset_with_ref()
        dsid = ds.id
    client.post(f'/api/dataset/{dsid}/ref/pose/right45',
                data={'file': (io.BytesIO(_png(800, 800)), 'p.png')},
                content_type='multipart/form-data')

    r = client.post(f'/api/dataset/{dsid}/ref/pose/right45/crop',
                    json={'x': 10, 'y': 10, 'w': 300, 'h': 300})
    assert r.status_code == 200 and r.get_json()['ok'] is True

    r = client.post(f'/api/dataset/{dsid}/ref/pose/right45/crop',
                    json={'x': 'nope', 'y': 0, 'w': 10, 'h': 10})
    assert r.status_code == 400

    r = client.post(f'/api/dataset/{dsid}/ref/pose/right45/mirror')
    assert r.status_code == 200 and r.get_json()['ok'] is True

    r = client.post(f'/api/dataset/{dsid}/ref/pose/left45/mirror')
    assert r.status_code == 404   # nothing uploaded there


def _krea_dataset_with_pose_slot(pose_key='left45', enabled=True):
    """Mirrors test_krea_generation_loras._krea_dataset, plus one pose slot."""
    ds = svc.create_dataset('local', 'ds', 'trg', train_type='krea')
    folder = svc._dataset_path(ds.id)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'ref.webp'), 'wb') as fh:
        fh.write(svc.normalize_to_webp(_png(512, 512)))
    ds.ref_filename = 'ref.webp'
    db.session.commit()
    svc.set_pose_slot('local', ds.id, pose_key, _png(600, 600, (30, 30, 30)))
    if enabled:
        svc.set_pose_slot_enabled('local', ds.id, pose_key, True)
    return ds


def test_zero_pose_slot_data_is_byte_identical_to_the_old_behaviour(app, monkeypatch):
    """The regression guard: a dataset with NO RefPoseSlot rows must keep using
    the primary reference for every shot, exactly like before this feature."""
    from app.services import krea_edit_helper as keh
    seen = []
    monkeypatch.setattr(keh, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(keh, 'enqueue_krea_edit',
                        lambda **kw: (seen.append(kw), 'job')[1])
    with app.app_context():
        ds = _dataset_with_ref()
        ref_path = svc._ref_path(ds)
        svc.generate_variations_krea(
            'local', ds.id,
            [{'label': 'a', 'prompt': 'left profile view', 'framing': 'face'}], 1)
    assert seen[0]['source_path'] == ref_path


def test_a_matched_enabled_slot_becomes_the_source_for_that_shot(app, monkeypatch):
    from app.services import krea_edit_helper as keh
    seen = []
    monkeypatch.setattr(keh, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(keh, 'enqueue_krea_edit',
                        lambda **kw: (seen.append(kw), 'job')[1])
    with app.app_context():
        ds = _krea_dataset_with_pose_slot('left45', enabled=True)
        expected = svc.enabled_pose_slot_paths(ds)['left45']
        svc.generate_variations_krea(
            'local', ds.id,
            [{'label': 'a', 'prompt': 'left profile view, neutral expression', 'framing': 'face'}], 1)
    assert seen[0]['source_path'] == expected
    assert seen[0]['source_filename'] == os.path.basename(expected)


def test_a_matched_but_disabled_slot_falls_back_to_front(app, monkeypatch):
    from app.services import krea_edit_helper as keh
    seen = []
    monkeypatch.setattr(keh, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(keh, 'enqueue_krea_edit',
                        lambda **kw: (seen.append(kw), 'job')[1])
    with app.app_context():
        ds = _krea_dataset_with_pose_slot('left45', enabled=False)
        svc.generate_variations_krea(
            'local', ds.id,
            [{'label': 'a', 'prompt': 'left profile view', 'framing': 'face'}], 1)
        expected_front = svc._ref_path(ds)
    assert seen[0]['source_path'] == expected_front


def test_ambiguous_direction_uses_the_single_enabled_slot(app, monkeypatch):
    from app.services import krea_edit_helper as keh
    seen = []
    monkeypatch.setattr(keh, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(keh, 'enqueue_krea_edit',
                        lambda **kw: (seen.append(kw), 'job')[1])
    with app.app_context():
        ds = _krea_dataset_with_pose_slot('right45', enabled=True)
        expected = svc.enabled_pose_slot_paths(ds)['right45']
        # No left/right word — ambiguous — but only ONE non-front slot is enabled.
        svc.generate_variations_krea(
            'local', ds.id, [{'label': 'a', 'prompt': 'side view', 'framing': 'face'}], 1)
    assert seen[0]['source_path'] == expected


def test_ambiguous_direction_with_both_sides_enabled_falls_back_to_front(app, monkeypatch):
    from app.services import krea_edit_helper as keh
    seen = []
    monkeypatch.setattr(keh, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(keh, 'enqueue_krea_edit',
                        lambda **kw: (seen.append(kw), 'job')[1])
    with app.app_context():
        ds = _krea_dataset_with_pose_slot('left45', enabled=True)
        svc.set_pose_slot('local', ds.id, 'right45', _png(600, 600))
        svc.set_pose_slot_enabled('local', ds.id, 'right45', True)
        svc.generate_variations_krea(
            'local', ds.id, [{'label': 'a', 'prompt': 'side view', 'framing': 'face'}], 1)
        expected_front = svc._ref_path(ds)
    assert seen[0]['source_path'] == expected_front


def test_ambiguous_direction_with_only_a_back_slot_enabled_falls_back_to_front(app, monkeypatch):
    """'back' is a valid POSE_SLOT_KEY (reachable via direct API even with no
    UI for it yet), but it is NOT a side-facing slot: an ambiguous prompt like
    'side view' must never resolve to the back-of-head photo just because it
    is the only enabled slot."""
    from app.services import krea_edit_helper as keh
    seen = []
    monkeypatch.setattr(keh, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(keh, 'enqueue_krea_edit',
                        lambda **kw: (seen.append(kw), 'job')[1])
    with app.app_context():
        ds = _krea_dataset_with_pose_slot('back', enabled=True)
        svc.generate_variations_krea(
            'local', ds.id, [{'label': 'a', 'prompt': 'side view', 'framing': 'face'}], 1)
        expected_front = svc._ref_path(ds)
    assert seen[0]['source_path'] == expected_front


def test_each_variation_in_one_batch_picks_its_own_source(app, monkeypatch):
    """A single generate_variations_krea call can carry a front shot and a left
    profile shot together — each must get its OWN source_path, not whichever
    was computed for the first variation."""
    from app.services import krea_edit_helper as keh
    seen = []
    monkeypatch.setattr(keh, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(keh, 'enqueue_krea_edit',
                        lambda **kw: (seen.append(kw), 'job')[1])
    with app.app_context():
        ds = _krea_dataset_with_pose_slot('left45', enabled=True)
        expected_left = svc.enabled_pose_slot_paths(ds)['left45']
        expected_front = svc._ref_path(ds)
        svc.generate_variations_krea(
            'local', ds.id,
            [{'label': 'a', 'prompt': 'front view, neutral expression', 'framing': 'face'},
             {'label': 'b', 'prompt': 'left profile view', 'framing': 'face'}], 1)
    assert seen[0]['source_path'] == expected_front
    assert seen[1]['source_path'] == expected_left
