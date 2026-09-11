"""The dataset → H3 RefMod route and its service.

The subprocess is the unit boundary: every test here stubs
`refmod_service.generate_for_dataset` (route tests) or `subprocess.run`
(service tests) — no torch, no ComfyUI, no real VAE.
"""
import json
import os
from types import SimpleNamespace

import pytest


# ── service: image picking ────────────────────────────────────────────────────
def _img(id_, status, framing, filename='f.webp'):
    return SimpleNamespace(id=id_, status=status, framing=framing, filename=filename)


def test_pick_images_identity_first():
    from app.services import refmod_service as svc

    # 30 faces available: face-dominant, half only as secondary, NO fulls —
    # full frames are the clothing carrier and clothing is what the user wants
    # OUT of the reference.
    rows = ([_img(i, 'keep', 'face') for i in range(30)]
            + [_img(100 + i, 'keep', 'half') for i in range(8)]
            + [_img(200 + i, 'keep', 'full') for i in range(5)]
            + [_img(300, 'pending', 'face'), _img(301, 'reject', 'half')])
    picked = svc.pick_images(rows)
    assert len(picked) == 16          # 12 face + 4 half (cap), 0 full
    assert sum(r.framing == 'face' for r in picked) == 12
    assert sum(r.framing == 'half' for r in picked) == 4
    assert not any(r.framing == 'full' for r in picked)
    assert all(r.status == 'keep' for r in picked)


def test_pick_images_tops_up_fulls_only_when_thin():
    from app.services import refmod_service as svc

    # 4 face + 1 half: fulls top the set up to the 6-row floor.
    rows = ([_img(i, 'keep', 'face') for i in range(4)]
            + [_img(50, 'keep', 'half')]
            + [_img(100 + i, 'keep', 'full') for i in range(9)])
    picked = svc.pick_images(rows)
    assert len(picked) == 6
    assert sum(r.framing == 'full' for r in picked) == 1


def test_pick_images_full_only_dataset_takes_twelve():
    from app.services import refmod_service as svc

    rows = [_img(i, 'keep', None) for i in range(20)]   # no framing → full
    picked = svc.pick_images(rows)
    assert len(picked) == 12


def test_pick_images_treats_unknown_framing_as_full():
    from app.services import refmod_service as svc

    rows = [_img(1, 'keep', None), _img(2, 'keep', 'back'), _img(3, 'keep', 'full')]
    picked = svc.pick_images(rows)
    assert len(picked) == 3


# ── service: path resolution fails with a remedy ─────────────────────────────
def test_python_exe_missing_is_a_valueerror_with_remedy(tmp_path, monkeypatch):
    from app.services import refmod_service as svc

    monkeypatch.setattr(svc.cfg, 'get',
                        lambda key: str(tmp_path) if key == 'comfyui.base_dir' else None)
    with pytest.raises(ValueError, match='No ComfyUI python'):
        svc._python_exe(tmp_path)


def test_node_dir_missing_names_the_pack(tmp_path, monkeypatch):
    from app.services import refmod_service as svc

    monkeypatch.setattr(svc.cfg, 'get',
                        lambda key: str(tmp_path) if key == 'comfyui.base_dir' else None)
    with pytest.raises(ValueError, match='MiniMaxH3Mod is not installed'):
        svc._node_dir(tmp_path)


def test_vae_scan_skips_audio_and_prefers_video(tmp_path):
    from app.services import refmod_service as svc

    vae_dir = tmp_path / 'models' / 'vae'
    vae_dir.mkdir(parents=True)
    (vae_dir / 'minimax_h3_audio_vae_fp32.safetensors').write_bytes(b'x')
    (vae_dir / 'minimax_h3_t1_image_vae.safetensors').write_bytes(b'x')
    (vae_dir / 'minimax_h3_video_vae_fp16.safetensors').write_bytes(b'x')
    assert svc._vae_path(tmp_path).name == 'minimax_h3_video_vae_fp16.safetensors'


def test_vae_scan_without_any_h3_vae_is_a_valueerror(tmp_path):
    from app.services import refmod_service as svc

    (tmp_path / 'models' / 'vae').mkdir(parents=True)
    (tmp_path / 'models' / 'vae' / 'sdxl_vae.safetensors').write_bytes(b'x')
    with pytest.raises(ValueError, match='No H3 video VAE'):
        svc._vae_path(tmp_path)


# ── service: subprocess contract ─────────────────────────────────────────────
class _FakeRows:
    """A dataset whose kept images exist as real (empty) files on disk."""

    def __init__(self, tmp_path):
        storage = tmp_path / 'datasets' / '7'
        storage.mkdir(parents=True)
        rows = []
        for i, framing in enumerate(['face', 'half', 'full']):
            f = storage / f'img{i}.webp'
            f.write_bytes(b'x')
            rows.append(SimpleNamespace(id=i, status='keep', framing=framing,
                                        filename=f.name))
        self.images = rows
        self.id = 7
        self.name = 'test person'
        self.storage = storage


def test_generate_for_dataset_parses_worker_json(tmp_path, monkeypatch):
    from app.services import refmod_service as svc

    ds = _FakeRows(tmp_path)
    seen = {}

    def fake_run(argv, input=None, capture_output=True, timeout=None, env=None):
        seen['argv'] = argv
        seen['manifest'] = json.loads(input.decode('utf-8'))
        return SimpleNamespace(returncode=0, stdout=json.dumps(
            {'ok': True, 'tokens': 8192, 'frames': 8, 'path': 'out.safetensors',
             'mb': 1.5}).encode('utf-8'), stderr=b'')

    monkeypatch.setattr(svc.subprocess, 'run', fake_run)
    monkeypatch.setattr(svc, '_kept_rows', lambda d: d.images)
    monkeypatch.setattr(svc, '_comfy_root', lambda: tmp_path)
    monkeypatch.setattr(svc, '_python_exe', lambda root: tmp_path / 'py.exe')
    monkeypatch.setattr(svc, '_node_dir', lambda root: tmp_path / 'custom_nodes' / 'pack')
    monkeypatch.setattr(svc, '_vae_path', lambda root: tmp_path / 'models' / 'vae' / 'h3.safetensors')
    monkeypatch.setattr(svc, 'dataset_path', lambda ds_id: str(ds.storage))

    res = svc.generate_for_dataset(ds)
    assert res['tokens'] == 8192 and res['frames'] == 8
    assert seen['manifest']['name'] == 'minimaxh3_test_person_v1_refmod'
    assert len(seen['manifest']['images']) == 3
    assert all(str(ds.storage) in p for p in seen['manifest']['images'])


def test_worker_failure_raises_runtimeerror_with_stderr_tail(tmp_path, monkeypatch):
    from app.services import refmod_service as svc

    ds = _FakeRows(tmp_path)

    def fake_run(argv, input=None, capture_output=True, timeout=None, env=None):
        return SimpleNamespace(returncode=1, stdout=b'',
                               stderr='Traceback ... GatedRepoError: 401'.encode('utf-8'))

    monkeypatch.setattr(svc.subprocess, 'run', fake_run)
    monkeypatch.setattr(svc, '_kept_rows', lambda d: d.images)
    monkeypatch.setattr(svc, '_comfy_root', lambda: tmp_path)
    monkeypatch.setattr(svc, '_python_exe', lambda root: tmp_path / 'py.exe')
    monkeypatch.setattr(svc, '_node_dir', lambda root: tmp_path / 'custom_nodes' / 'pack')
    monkeypatch.setattr(svc, '_vae_path', lambda root: tmp_path / 'models' / 'vae' / 'h3.safetensors')

    with pytest.raises(RuntimeError, match='GatedRepoError'):
        svc.generate_for_dataset(ds)


def test_unsafe_dataset_name_sanitizes(tmp_path, monkeypatch):
    from app.services import refmod_service as svc

    ds = _FakeRows(tmp_path)
    ds.name = '人世间-宋佳!'
    seen = {}

    def fake_run(argv, input=None, capture_output=True, timeout=None, env=None):
        seen['manifest'] = json.loads(input.decode('utf-8'))
        return SimpleNamespace(returncode=0, stdout=json.dumps(
            {'ok': True, 'tokens': 100, 'frames': 1, 'path': 'o', 'mb': 0.1}).encode('utf-8'),
            stderr=b'')

    monkeypatch.setattr(svc.subprocess, 'run', fake_run)
    monkeypatch.setattr(svc, '_kept_rows', lambda d: d.images)
    monkeypatch.setattr(svc, '_comfy_root', lambda: tmp_path)
    monkeypatch.setattr(svc, '_python_exe', lambda root: tmp_path / 'py.exe')
    monkeypatch.setattr(svc, '_node_dir', lambda root: tmp_path / 'custom_nodes' / 'pack')
    monkeypatch.setattr(svc, '_vae_path', lambda root: tmp_path / 'models' / 'vae' / 'h3.safetensors')

    svc.generate_for_dataset(ds)
    # CJK chars are isalnum() in Python — they survive; punctuation becomes '_'
    # and a trailing '_' is stripped by .strip('_').
    assert seen['manifest']['name'] == 'minimaxh3_人世间_宋佳_v1_refmod'


# ── route, against the REAL models (the 500 regression) ──────────────────
def test_route_works_against_real_models(app, client, tmp_path, monkeypatch):
    """A dataset built through the real ORM — no SimpleNamespace anywhere —
    must reach the worker. Guards the regression where the route read a
    ``ds.images`` relationship that FaceDataset does not have (live 500)."""
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDataset, FaceDatasetImage
        from app.routes import datasets as routes
        from app.services import refmod_service as svc

        d = FaceDataset(user_id='local', name='RefModLive', trigger_word='rm00001')
        db.session.add(d)
        db.session.commit()
        db.session.add(FaceDatasetImage(dataset_id=d.id, filename='a.png',
                                        status='keep', framing='face'))
        db.session.commit()
        ds_id = d.id

        seen = {}

        def fake_run(argv, input=None, capture_output=True, timeout=None, env=None):
            seen['manifest'] = json.loads(input.decode('utf-8'))
            return SimpleNamespace(returncode=0, stdout=json.dumps(
                {'ok': True, 'tokens': 256, 'frames': 1, 'path': 'o',
                 'mb': 0.1}).encode('utf-8'), stderr=b'')

        monkeypatch.setattr(svc.subprocess, 'run', fake_run)
        monkeypatch.setattr(svc, '_comfy_root', lambda: tmp_path)
        monkeypatch.setattr(svc, '_python_exe', lambda root: tmp_path / 'py.exe')
        monkeypatch.setattr(svc, '_node_dir', lambda root: tmp_path / 'pack')
        monkeypatch.setattr(svc, '_vae_path', lambda root: tmp_path / 'h3.safetensors')

        resp = client.post(f'/api/dataset/{ds_id}/refmod')
        assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
        body = resp.get_json()
        assert body['ok'] is True and body['frames'] == 1
        assert len(seen['manifest']['images']) == 1


# ── route ─────────────────────────────────────────────────────────────────────
def test_route_refmod_happy_path(app, client, monkeypatch):
    from app.routes import datasets as routes
    from app.services import refmod_service as svc

    rows = [SimpleNamespace(id=1, status='keep', framing='face', filename='a.webp')]
    monkeypatch.setattr(routes.svc, 'get_dataset',
                        lambda user, ds_id: SimpleNamespace(id=ds_id, name='x',
                                                            images=rows))
    monkeypatch.setattr(routes.refmod_service, 'generate_for_dataset',
                        lambda ds: {'name': 'n', 'tokens': 10, 'frames': 2,
                                    'path': 'p', 'mb': 0.2})
    resp = client.post('/api/dataset/5/refmod')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True and body['tokens'] == 10


def test_face_masks_align_and_degrade(tmp_path, monkeypatch):
    """'masked' entries get their PNG path, everything else None — and an
    unavailable detector degrades to all-None instead of failing the run."""
    from app.services import refmod_service as svc

    paths = [str(tmp_path / f'img{i}.png') for i in range(3)]
    for p in paths:
        open(p, 'wb').close()
    mask_dir = tmp_path / 'masks' / 'refmod' / '9'
    mask_dir.mkdir(parents=True)
    for i in (0, 2):   # img1 = 'no_face'
        (mask_dir / f'img{i}.png').write_bytes(b'x')

    calls = {}

    def fake_generate(imgs, out_dir, expand=None, timeout=1800):
        calls['imgs'] = list(imgs)
        calls['out_dir'] = out_dir
        return {'ok': True, 'written': 2, 'results': {
            paths[0]: {'state': 'masked'},
            paths[1]: {'state': 'no_face'},
            paths[2]: {'state': 'masked'}}}

    monkeypatch.setattr(svc.face_mask_service, 'generate_face_masks', fake_generate)
    monkeypatch.setattr(svc, 'dataset_path', lambda ds_id: str(tmp_path / 'datasets' / 'ds'))
    masks = svc._face_masks_for(paths, ds_id=9)
    assert calls['imgs'] == paths and calls['out_dir'].endswith(os.path.join('refmod', '9'))
    assert masks[0] == str(mask_dir / 'img0.png')
    assert masks[1] is None
    assert masks[2] == str(mask_dir / 'img2.png')

    monkeypatch.setattr(svc.face_mask_service, 'generate_face_masks',
                        lambda imgs, out_dir, expand=None, timeout=1800: {})
    assert svc._face_masks_for(paths, ds_id=9) == [None, None, None]


def test_route_refmod_404_and_no_kept(app, client, monkeypatch):
    from app.routes import datasets as routes

    monkeypatch.setattr(routes.svc, 'get_dataset', lambda user, ds_id: None)
    assert client.post('/api/dataset/5/refmod').status_code == 404

    def no_images(ds):
        raise ValueError('No kept images to encode.')

    monkeypatch.setattr(routes.svc, 'get_dataset',
                        lambda user, ds_id: SimpleNamespace(id=ds_id, name='x'))
    monkeypatch.setattr(routes.refmod_service, 'generate_for_dataset', no_images)
    resp = client.post('/api/dataset/5/refmod')
    assert resp.status_code == 400
    assert 'No kept images' in resp.get_json()['error']


def test_route_refmod_gpu_busy_maps_to_503(app, client, monkeypatch):
    from app.routes import datasets as routes
    from app.gpu_window import GpuBusyError

    def fake_generate(ds):
        raise GpuBusyError('a vision task is already running')

    monkeypatch.setattr(routes.svc, 'get_dataset',
                        lambda user, ds_id: SimpleNamespace(id=ds_id, name='x'))
    monkeypatch.setattr(routes.refmod_service, 'generate_for_dataset', fake_generate)
    resp = client.post('/api/dataset/5/refmod')
    assert resp.status_code == 503
