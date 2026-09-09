"""👥👥 Multi-person on the DATASET side: analyze_faces persists n_faces.

The scorer subprocess (infer/face_score_infer.py) already reports how many
faces it saw per image — including the ones too small or too off-pose to
score. The pass used to drop that number on the floor, so the Triage bar's
multi-person one-click reject had nothing to read. NULL stays NULL: an image
the pass never measured must never read as "single person".
"""
import io
import os

import pytest

from test_dataset_activity import _img_bytes, _kept_image


def _ds_with_image(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'F', 'f')
        d = svc._dataset_dir(ds.id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
            fh.write(_img_bytes())
        ds.ref_filename = 'ref.webp'
        svc.db.session.commit()
        _kept_image(svc, ds.id, 'a.webp')
        _kept_image(svc, ds.id, 'b.webp')
        return ds.id


def test_analyze_faces_persists_n_faces(app, monkeypatch):
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import face_scoring_service as fss
    from app.services import face_similarity as fs
    from app.config import LOCAL_USER

    ds_id = _ds_with_image(app, monkeypatch)

    def fake_score(ref_path, paths, on_progress=None, **kw):
        results = {}
        for i, p in enumerate(paths):
            results[p] = {'state': 'scorable', 'sim': 0.5, 'yaw': 0.0,
                          'n_faces': 2 if i == 0 else 1}
        return results, None

    monkeypatch.setattr(fs, 'score_dataset_faces', fake_score)

    with app.app_context():
        fss.analyze_faces(LOCAL_USER, ds_id)
        rows = (FaceDatasetImage.query.filter_by(dataset_id=ds_id)
                .order_by(FaceDatasetImage.filename).all())
        assert [r.n_faces for r in rows] == [2, 1]


def test_analyze_faces_leaves_unscored_n_faces_null(app, monkeypatch):
    """A face that fails detection this run must leave the row unmeasured —
    the button reads absence as "not measured", never as "single person"."""
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import face_scoring_service as fss
    from app.services import face_similarity as fs
    from app.config import LOCAL_USER

    ds_id = _ds_with_image(app, monkeypatch)

    def fake_score(ref_path, paths, on_progress=None, **kw):
        return {p: {'state': 'no_face', 'n_faces': None} for p in paths}, None

    monkeypatch.setattr(fs, 'score_dataset_faces', fake_score)

    with app.app_context():
        fss.analyze_faces(LOCAL_USER, ds_id)
        nulls = [r.n_faces for r in
                 FaceDatasetImage.query.filter_by(dataset_id=ds_id).all()]
        assert all(v is None for v in nulls)
