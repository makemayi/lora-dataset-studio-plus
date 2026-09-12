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
def _img(id_, status, framing, filename='f.webp', face_score=None, face_yaw=None):
    # face_score/face_yaw: real rows carry the scorer's values; pick_images
    # sorts on score and spreads the face bucket across yaw, so stubs must
    # carry both (None sorts/fills last, deterministic).
    return SimpleNamespace(id=id_, status=status, framing=framing, filename=filename,
                           face_score=face_score, face_yaw=face_yaw)


def test_pick_images_identity_first():
    from app.services import refmod_service as svc

    # 30 faces available: face-dominant, half only as secondary, NO fulls —
    # full frames are the clothing carrier and clothing is what the user wants
    # OUT of the reference.
    rows = ([_img(i, 'keep', 'face') for i in range(30)]
            + [_img(100 + i, 'keep', 'bust') for i in range(8)]
            + [_img(200 + i, 'keep', 'full') for i in range(5)]
            + [_img(300, 'pending', 'face'), _img(301, 'reject', 'half')])
    picked = svc.pick_images(rows)
    assert len(picked) == 16          # 12 face + 4 half (cap), 0 full
    assert sum(r.framing == 'face' for r in picked) == 12
    assert sum(r.framing == 'bust' for r in picked) == 4
    assert not any(r.framing == 'full' for r in picked)
    assert all(r.status == 'keep' for r in picked)


def test_pick_images_tops_up_fulls_only_when_thin():
    from app.services import refmod_service as svc

    # 4 face + 1 half: fulls top the set up to the 6-row floor.
    rows = ([_img(i, 'keep', 'face') for i in range(4)]
            + [_img(50, 'keep', 'bust')]
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
    # Unknown framing lands in the full bucket; a back view is never an
    # identity reference and is never picked.
    assert [r.id for r in picked] == [1, 3]
    assert not any(r.framing == 'back' for r in picked)


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
        for i, framing in enumerate(['face', 'bust', 'full']):
            f = storage / f'img{i}.webp'
            f.write_bytes(b'x')
            rows.append(SimpleNamespace(id=i, status='keep', framing=framing,
                                        filename=f.name, face_score=None,
                                        face_yaw=None))
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
    monkeypatch.setattr(svc, '_harvest_face_crops',
                        lambda ds, sources, deficit, note: ([], note))
    monkeypatch.setattr(svc.face_mask_service, 'generate_face_masks',
                        lambda imgs, out_dir, expand=None, timeout=1800: {})
    monkeypatch.setattr(svc, '_comfy_root', lambda: tmp_path)
    monkeypatch.setattr(svc, '_python_exe', lambda root: tmp_path / 'py.exe')
    monkeypatch.setattr(svc, '_node_dir', lambda root: tmp_path / 'custom_nodes' / 'pack')
    monkeypatch.setattr(svc, '_vae_path', lambda root: tmp_path / 'models' / 'vae' / 'h3.safetensors')
    monkeypatch.setattr(svc, 'dataset_path', lambda ds_id: str(ds.storage))

    res = svc.generate_for_dataset(ds)
    assert res['tokens'] == 8192 and res['frames'] == 8
    assert seen['manifest']['name'] == 'minimaxh3_test_person_v1_mask_refmod'
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
    monkeypatch.setattr(svc, '_harvest_face_crops',
                        lambda ds, sources, deficit, note: ([], note))
    monkeypatch.setattr(svc.face_mask_service, 'generate_face_masks',
                        lambda imgs, out_dir, expand=None, timeout=1800: {})
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
    monkeypatch.setattr(svc, '_harvest_face_crops',
                        lambda ds, sources, deficit, note: ([], note))
    monkeypatch.setattr(svc.face_mask_service, 'generate_face_masks',
                        lambda imgs, out_dir, expand=None, timeout=1800: {})
    monkeypatch.setattr(svc, '_comfy_root', lambda: tmp_path)
    monkeypatch.setattr(svc, '_python_exe', lambda root: tmp_path / 'py.exe')
    monkeypatch.setattr(svc, '_node_dir', lambda root: tmp_path / 'custom_nodes' / 'pack')
    monkeypatch.setattr(svc, '_vae_path', lambda root: tmp_path / 'models' / 'vae' / 'h3.safetensors')

    svc.generate_for_dataset(ds)
    # CJK chars are isalnum() in Python — they survive; punctuation becomes '_'
    # and a trailing '_' is stripped by .strip('_').
    assert seen['manifest']['name'] == 'minimaxh3_人世间_宋佳_v1_mask_refmod'


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
                        lambda ds, masked=True: {'name': 'n', 'tokens': 10, 'frames': 2,
                                                 'path': 'p', 'mb': 0.2, 'masked': 1})
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

    def no_images(ds, masked=True):
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

    def fake_generate(ds, masked=True):
        raise GpuBusyError('a vision task is already running')

    monkeypatch.setattr(routes.svc, 'get_dataset',
                        lambda user, ds_id: SimpleNamespace(id=ds_id, name='x'))
    monkeypatch.setattr(routes.refmod_service, 'generate_for_dataset', fake_generate)
    resp = client.post('/api/dataset/5/refmod')
    assert resp.status_code == 503


# ── harvest: smart face recovery tiers ────────────────────────────────────────
def test_harvest_tiers_crops_into_face_recovery_values(app, tmp_path, monkeypatch):
    """_harvest_face_crops KNOWS each crop's face pixel size (it just measured
    the box) — the Topaz pass must tier crops and run one pass per value, so
    no crop meets Autopilot's 0.8-on-every-face default. Big-face crop -> off,
    small-face crop -> its tier strength, persisted as face rows either way."""
    from app.services import refmod_service as rs
    from app.services import topaz_helper as th
    from app.services.dataset_storage import dataset_path
    from app.models import FaceDataset, FaceDatasetImage
    from app.extensions import db
    from PIL import Image

    with app.app_context():
        ds = FaceDataset(name='harvest ds', trigger_word='t')
        db.session.add(ds)
        db.session.commit()
        ddir = dataset_path(ds.id)
        os.makedirs(ddir, exist_ok=True)

        # 1000x1000 source, face box 0.6x0.6 -> face 600px short side -> OFF.
        # 500x400 source, face box 0.5x0.5 -> face 200px short side -> 0.50.
        specs = [('a.png', 1000, 1000, 0.6, {'area': 0.36}),
                 ('b.png', 500, 400, 0.5, {'area': 0.25})]
        sources, recs = [], {}
        for name, w, h, box, face in specs:
            Image.new('RGB', (w, h), (128, 128, 128)).save(os.path.join(ddir, name))
            img = FaceDatasetImage(dataset_id=ds.id, filename=name, status='keep')
            db.session.add(img)
            db.session.flush()
            sources.append(img)
            half = (1 - box) / 2
            recs[name] = {'state': 'masked', 'faces': [dict(face, det_score=0.9)],
                          'boxes': [[half, half, half + box, half + box]]}
        db.session.commit()

        from pathlib import Path as _P

        def fake_detect(imgs):
            return {'ok': True, 'results': {
                p: recs.get(_P(p).name) or {'faces': [{'det_score': 0.9}],
                                            'boxes': []}
                for p in imgs}}

        monkeypatch.setattr(rs.face_mask_service, 'detect_faces', fake_detect)
        monkeypatch.setattr(th, 'preflight', lambda: 'tpai')
        monkeypatch.setattr(th, 'resolve_exe', lambda: 'tpai')
        seen = []

        def fake_run_tpai(exe, input_path, output_dir, **kw):
            seen.append(kw.get('face_recovery'))
            import pathlib as _pathlib
            out = _pathlib.Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for f in _pathlib.Path(input_path).iterdir():
                (out / (f.stem + '.png')).write_bytes(b'up')
            return 'ok', ''

        monkeypatch.setattr(th, 'run_tpai', fake_run_tpai)

        rows, note = rs._harvest_face_crops(ds, sources, deficit=4, note='')
        assert sorted(seen, key=str) == [0.5, False], \
            f'one tpai pass per tier, got {seen}'
        assert len(rows) == 2 and all(r.derivation_kind == rs._CROP_KIND
                                      for r in rows), note


# ── live stage progress (the button's poll-fed label) ─────────────────────────
def test_generation_publishes_and_clears_stage(tmp_path, monkeypatch):
    """generate_for_dataset publishes its stage while the worker runs (that is
    what the button's poll reads) and ALWAYS clears it — success or failure."""
    from app.services import refmod_service as svc

    ds = _FakeRows(tmp_path)
    seen = {}

    def fake_run(argv, input=None, capture_output=True, timeout=None, env=None):
        seen['during'] = svc.current_stage(ds.id)
        return SimpleNamespace(returncode=0, stdout=json.dumps(
            {'ok': True, 'tokens': 1, 'frames': 1, 'path': 'o', 'mb': 0}).encode(),
            stderr=b'')

    monkeypatch.setattr(svc.subprocess, 'run', fake_run)
    monkeypatch.setattr(svc, '_kept_rows', lambda d: d.images)
    monkeypatch.setattr(svc, '_harvest_face_crops',
                        lambda ds_, sources, deficit, note: ([], note))
    monkeypatch.setattr(svc.face_mask_service, 'generate_face_masks',
                        lambda imgs, out_dir, expand=None, timeout=1800: {})
    monkeypatch.setattr(svc, '_comfy_root', lambda: tmp_path)
    monkeypatch.setattr(svc, '_python_exe', lambda root: tmp_path / 'py.exe')
    monkeypatch.setattr(svc, '_node_dir', lambda root: tmp_path / 'pack')
    monkeypatch.setattr(svc, '_vae_path', lambda root: tmp_path / 'h3.safetensors')

    svc.generate_for_dataset(ds)
    assert seen['during'], 'a stage must be live while the worker runs'
    assert svc.current_stage(ds.id) is None, 'cleared on success'


def test_generation_clears_stage_on_failure(tmp_path, monkeypatch):
    from app.services import refmod_service as svc

    ds = _FakeRows(tmp_path)

    def fake_run(argv, input=None, capture_output=True, timeout=None, env=None):
        return SimpleNamespace(returncode=1, stdout=b'',
                               stderr='boom'.encode('utf-8'))

    monkeypatch.setattr(svc.subprocess, 'run', fake_run)
    monkeypatch.setattr(svc, '_kept_rows', lambda d: d.images)
    monkeypatch.setattr(svc, '_harvest_face_crops',
                        lambda ds_, sources, deficit, note: ([], note))
    monkeypatch.setattr(svc.face_mask_service, 'generate_face_masks',
                        lambda imgs, out_dir, expand=None, timeout=1800: {})
    monkeypatch.setattr(svc, '_comfy_root', lambda: tmp_path)
    monkeypatch.setattr(svc, '_python_exe', lambda root: tmp_path / 'py.exe')
    monkeypatch.setattr(svc, '_node_dir', lambda root: tmp_path / 'pack')
    monkeypatch.setattr(svc, '_vae_path', lambda root: tmp_path / 'h3.safetensors')

    with pytest.raises(RuntimeError):
        svc.generate_for_dataset(ds)
    assert svc.current_stage(ds.id) is None, 'cleared on failure too'


def test_refmod_progress_route_reports_stage(app, client):
    from app.services import refmod_service as svc

    svc.set_refmod_stage(7, 'encoding (VAE load takes a minute)')
    try:
        resp = client.get('/api/dataset/7/refmod/progress')
        assert resp.get_json() == {'stage': 'encoding (VAE load takes a minute)'}
    finally:
        svc.set_refmod_stage(7, None)
    assert client.get('/api/dataset/7/refmod/progress').get_json() == {'stage': None}
