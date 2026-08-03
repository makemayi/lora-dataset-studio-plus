"""Angle reference photos for Krea 2 Edit (pose slots): left45/right45 now,
back/left90/right90 reserved for a later wave. See
docs/superpowers/specs/2026-08-03-krea-pose-slots-design.md.
"""
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
