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
