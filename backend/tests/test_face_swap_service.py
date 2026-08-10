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


# --- The original survives every way a swap can end ---------------------------
# A swap overwrites the tile IN PLACE, so while the job runs the row carries no
# file and the picture exists only as the file the snapshot names. Both ways a
# swap ends badly used to destroy it: ⏹ Stop deletes rows that are `pending`
# with no file — exactly a swapping tile — and a ComfyUI failure left an empty
# ⚠ tile. The original was then recoverable by hand, out of the app trash, and
# nowhere else.

def _swap_started(svc, monkeypatch, cfg, tmp_path):
    """A dataset whose tile has a swap in flight. Returns (ds, img_id, job_id)."""
    from app.services import face_swap_helper
    from app.config import LOCAL_USER
    ds, img = _dataset_with_image(svc, cfg, tmp_path)
    img_id = img.id
    monkeypatch.setattr(face_swap_helper, 'enqueue_face_swap',
                        lambda **_kwargs: 'swap-job')
    job_id = svc.face_swap_image(LOCAL_USER, img_id)
    return ds, img_id, job_id


def test_the_original_stays_on_disk_while_the_swap_runs(app, tmp_path, monkeypatch):
    """It used to go to Trash the moment the job was queued, which is what left
    both failure paths with nothing to put back."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds, img_id, job_id = _swap_started(svc, monkeypatch, cfg, tmp_path)
        row = svc.db.session.get(svc.FaceDatasetImage, img_id)
        assert (row.filename, row.status, row.job_id) == (None, 'pending', job_id)
        assert os.path.isfile(os.path.join(svc._dataset_dir(ds.id), 'tile.png'))
        assert 'tile.png' in (row.swap_restore or '')


def test_stopping_a_swap_restores_the_tile_instead_of_deleting_it(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds, img_id, _job = _swap_started(svc, monkeypatch, cfg, tmp_path)
        monkeypatch.setattr(queue_manager, 'cancel_job_outcome',
                            lambda *a, **k: 'cancelled')

        out = svc.cancel_pending(LOCAL_USER, ds.id)

        row = svc.db.session.get(svc.FaceDatasetImage, img_id)
        assert row is not None, 'Stop deleted the tile it was asked to leave alone'
        assert (row.filename, row.status) == ('tile.png', 'finished')
        assert row.job_id is None and row.swap_restore is None
        assert os.path.isfile(os.path.join(svc._dataset_dir(ds.id), 'tile.png'))
        # It was restored, not cancelled-away: the count is what the UI reports.
        assert out['cancelled'] == 0
        assert 'cancelled' in (row.fail_reason or '').lower()


def test_a_failed_swap_restores_the_tile(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import face_dataset_service as svc
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds, img_id, job_id = _swap_started(svc, monkeypatch, cfg, tmp_path)

        svc.link_completed_dataset_image(job_id, None, failed=True,
                                         reason='ComfyUI said no')

        row = svc.db.session.get(svc.FaceDatasetImage, img_id)
        assert (row.filename, row.status) == ('tile.png', 'finished')
        assert row.swap_restore is None
        assert os.path.isfile(os.path.join(svc._dataset_dir(ds.id), 'tile.png'))
        # The failure is not swallowed: a restored tile shows its own picture
        # again, so the reason is the only trace that anything was attempted.
        assert 'ComfyUI said no' in (row.fail_reason or '')


def test_a_swap_that_lands_trashes_the_picture_it_replaced(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import face_dataset_service as svc
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds, img_id, job_id = _swap_started(svc, monkeypatch, cfg, tmp_path)
        out_dir = svc._comfy_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'swapped.png'), 'wb') as fh:
            fh.write(_png((5, 5, 200)))

        svc.link_completed_dataset_image(job_id, 'swapped.png')

        row = svc.db.session.get(svc.FaceDatasetImage, img_id)
        assert row.filename == 'swapped.png'
        assert row.swap_restore is None
        assert row.unseen is True
        # NOW the replaced picture leaves — not one step earlier.
        assert not os.path.exists(os.path.join(svc._dataset_dir(ds.id), 'tile.png'))


def test_deleting_a_swapping_tile_takes_the_held_original_with_it(app, tmp_path, monkeypatch):
    """Otherwise the retained file outlives the only row that could name it."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds, img_id, _job = _swap_started(svc, monkeypatch, cfg, tmp_path)
        monkeypatch.setattr(queue_manager, 'cancel_job', lambda *a, **k: True)

        assert svc.delete_image(LOCAL_USER, img_id) is True
        assert not os.path.exists(os.path.join(svc._dataset_dir(ds.id), 'tile.png'))


# --- the fence must refuse the CLICK, not swallow it -------------------------

def test_a_swap_is_refused_while_ollama_holds_the_gpu(app, client, tmp_path, monkeypatch):
    """Reported as "submitted but ComfyUI does nothing": the row went pending,
    the tile said generating, the queue re-deferred it once a second, and the
    only trace was a WARNING in the server log. The queue may WAIT behind that
    fence — the click may not be accepted in silence."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.services import ollama_gpu_fence
    with app.app_context():
        _comfy(tmp_path, cfg)
        _ds, img = _dataset_with_image(svc, cfg, tmp_path)
        img_id = img.id
    monkeypatch.setattr(ollama_gpu_fence, 'fence_status',
                        lambda: {'applies': True, 'blocked': True, 'scope': 'local',
                                 'reachable': True, 'models': ['gemma:7b']})
    resp = client.post(f'/api/dataset/image/{img_id}/face-swap')
    assert resp.status_code == 409
    body = resp.get_json()
    assert body['code'] == 'ollama_gpu_fence'
    # It names the model in the way and the command that clears it, because the
    # fence itself will never unload a model LDS did not load.
    assert 'gemma:7b' in body['error']
    assert 'ollama stop gemma:7b' in body['error']
    with app.app_context():
        row = svc.db.session.get(svc.FaceDatasetImage, img_id)
        assert (row.filename, row.status) == ('tile.png', 'finished')


def test_an_unreadable_fence_does_not_ground_the_swap(app, client, tmp_path, monkeypatch):
    """A probe hiccup is not evidence of a blocked GPU."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.services import ollama_gpu_fence
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        _ds, img = _dataset_with_image(svc, cfg, tmp_path)
        img_id = img.id
    monkeypatch.setattr(ollama_gpu_fence, 'fence_status',
                        lambda: (_ for _ in ()).throw(OSError('probe down')))
    monkeypatch.setattr(queue_manager, 'add_job', lambda **kw: kw['job_id'])
    resp = client.post(f'/api/dataset/image/{img_id}/face-swap')
    assert resp.status_code == 200 and resp.get_json()['ok'] is True
