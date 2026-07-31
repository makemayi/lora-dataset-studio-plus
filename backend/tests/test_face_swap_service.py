"""face_dataset_service.face_swap_image: in-place tile overwrite via the
fixed Klein face-swap workflow. Same cancel/trash/pending-transition shape
as regenerate_image, minus the provenance-column reset (variation_prompt/
variation_label/klein_model are left untouched — a face swap is an identity
post-process, not a re-generation from the catalog prompt)."""
import io
import os
import struct

import pytest
from PIL import Image


_VALID_ST = struct.pack('<Q', 2) + b'{}'


def _png(color=(0, 128, 255)):
    buf = io.BytesIO(); Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _install(base, *relparts, data=_VALID_ST):
    p = base.joinpath(*relparts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _comfy(tmp_path, cfg):
    base = tmp_path / 'comfyui'
    (base / 'input').mkdir(parents=True)
    (base / 'output').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    _install(base, 'models', 'unet', 'klein', 'flux-2-klein-9b-fp8.safetensors')
    _install(base, 'models', 'vae', 'flux2-vae.safetensors')
    _install(base, 'models', 'text_encoders', 'qwen_3_8b_fp8mixed.safetensors')
    _install(base, 'models', 'loras', 'klein', 'Klein2-9B-SmartCharacterSwap.safetensors')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


def _dataset_with_image(svc, cfg_module, tmp_path):
    from app.config import LOCAL_USER
    ds = svc.create_dataset(LOCAL_USER, 'Swap', 'swp')
    d = svc._dataset_dir(ds.id)
    import os
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'ref.png'), 'wb') as fh:
        fh.write(_png((200, 10, 10)))
    ds.ref_filename = 'ref.png'
    img = svc.FaceDatasetImage(dataset_id=ds.id, source='generated',
                               status='finished', filename='tile.png',
                               variation_prompt='p', variation_label='x')
    svc.db.session.add(img)
    svc.db.session.commit()
    with open(os.path.join(d, 'tile.png'), 'wb') as fh:
        fh.write(_png((10, 200, 10)))
    return ds, img


def test_face_swap_requires_reference_image(app, tmp_path):
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds = svc.create_dataset(LOCAL_USER, 'NoRef', 'nr')
        d = svc._dataset_dir(ds.id)
        import os
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'tile.png'), 'wb') as fh:
            fh.write(_png())
        img = svc.FaceDatasetImage(dataset_id=ds.id, source='generated',
                                   status='finished', filename='tile.png')
        svc.db.session.add(img)
        svc.db.session.commit()
        with pytest.raises(ValueError, match='reference image required'):
            svc.face_swap_image(LOCAL_USER, img.id)


def test_face_swap_returns_none_when_tile_has_no_current_image(app, tmp_path):
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds = svc.create_dataset(LOCAL_USER, 'Empty', 'em')
        img = svc.FaceDatasetImage(dataset_id=ds.id, source='generated',
                                   status='pending', filename=None)
        svc.db.session.add(img)
        svc.db.session.commit()
        assert svc.face_swap_image(LOCAL_USER, img.id) is None


def test_face_swap_transitions_row_and_leaves_provenance_untouched(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds, img = _dataset_with_image(svc, cfg, tmp_path)
        img_id, ds_id = img.id, ds.id
        captured = {}
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.update(kw), kw['job_id'])[1])
        job_id = svc.face_swap_image(LOCAL_USER, img_id)
        assert job_id and job_id == captured['job_id']
        row = svc.db.session.get(svc.FaceDatasetImage, img_id)
        assert row.status == 'pending'
        assert row.filename is None
        assert row.job_id == job_id
        # Provenance untouched — this is a post-process, not a re-generation.
        assert row.variation_prompt == 'p'
        assert row.variation_label == 'x'
        assert captured['metadata']['dataset_id'] == ds_id
        assert captured['metadata']['variation_label'] == 'x'


def test_face_swap_route_starts_a_job(app, client, tmp_path):
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds, img = _dataset_with_image(svc, cfg, tmp_path)
        img_id = img.id
    import unittest.mock as mock
    with mock.patch.object(queue_manager, 'add_job', lambda **kw: kw['job_id']):
        resp = client.post(f'/api/dataset/image/{img_id}/face-swap')
    assert resp.status_code == 200
    assert resp.get_json()['ok'] is True
    assert resp.get_json()['job_id']


def test_face_swap_route_404_for_unknown_image(client):
    resp = client.post('/api/dataset/image/999999/face-swap')
    assert resp.status_code == 404


def test_face_swap_trash_failure_restores_previous_row_and_cancels_new_job(
        app, tmp_path, monkeypatch):
    """Highest-risk branch: the row has already been flipped to pending/no
    filename/new job_id (DB committed) when Trash raises. Mirrors
    test_regenerate_trash_failure_restores_previous_row_and_cancels_new_job
    in test_data_integrity_trash.py, adapted to face_swap_image's fields."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.services import face_swap_helper, trash
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds, img = _dataset_with_image(svc, cfg, tmp_path)
        img_id = img.id
        old_path = os.path.join(svc._dataset_dir(ds.id), 'tile.png')

        monkeypatch.setattr(face_swap_helper, 'enqueue_face_swap',
                            lambda **_kwargs: 'new-job')

        cancellations = []

        def cancel(job_id, user_id=None, job_type='image', *, commit=True):
            cancellations.append((job_id, user_id, job_type, commit))
            return True

        monkeypatch.setattr(queue_manager, 'cancel_job', cancel)

        def fail_trash(_path, context=''):
            svc.db.session.expire_all()
            pending = svc.db.session.get(svc.FaceDatasetImage, img_id)
            assert pending.filename is None and pending.status == 'pending'
            assert pending.job_id == 'new-job'
            raise OSError('injected Trash failure')

        monkeypatch.setattr(trash, 'send_to_trash', fail_trash)

        with pytest.raises(OSError, match='injected Trash failure'):
            svc.face_swap_image(LOCAL_USER, img_id)

        row = svc.db.session.get(svc.FaceDatasetImage, img_id)
        assert (row.filename, row.status, row.job_id) == ('tile.png', 'finished', None)
        # Provenance was never touched by face_swap_image in the first place,
        # but assert it for symmetry with the regenerate rollback test.
        assert row.variation_prompt == 'p'
        assert row.variation_label == 'x'
        assert os.path.isfile(old_path)
        assert ('new-job', LOCAL_USER, 'image', False) in cancellations
