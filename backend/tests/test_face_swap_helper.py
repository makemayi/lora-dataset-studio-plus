"""face_swap_helper.enqueue_face_swap: stages the tile's current image
(target) and the dataset reference (identity source) into ComfyUI's input
dir, rewires the fixed 'face swap.json' workflow's loader/seed/output nodes,
and enqueues it. Mirrors the fixture style of test_klein_models.py."""
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


def _comfy(tmp_path, cfg, swap_lora=True):
    """A ComfyUI tree with the shared Klein assets + (optionally) the
    fixed face-swap LoRA this workflow specifically requires."""
    base = tmp_path / 'comfyui'
    (base / 'input').mkdir(parents=True)
    (base / 'output').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    _install(base, 'models', 'unet', 'klein', 'flux-2-klein-9b-fp8.safetensors')
    _install(base, 'models', 'vae', 'flux2-vae.safetensors')
    _install(base, 'models', 'text_encoders', 'qwen_3_8b_fp8mixed.safetensors')
    if swap_lora:
        _install(base, 'models', 'loras', 'klein', 'Klein2-9B-SmartCharacterSwap.safetensors')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


def test_enqueue_rewires_target_ref_and_loaders(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        target = tmp_path / 'target.png'; target.write_bytes(_png((10, 200, 10)))
        ref = tmp_path / 'ref.png'; ref.write_bytes(_png((200, 10, 10)))
        captured = {}
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.update(kw), kw['job_id'])[1])
        job_id = fsh.enqueue_face_swap(user_id='local', target_path=str(target),
                                       ref_path=str(ref))
        assert job_id
        wf = captured['workflow_data']
        assert wf['151']['inputs']['image'].endswith('.png')
        assert wf['121']['inputs']['image'].endswith('.png')
        assert wf['151']['inputs']['image'] != wf['121']['inputs']['image']
        assert wf['165:126']['inputs']['unet_name'] == os.path.join('klein', 'flux-2-klein-9b-fp8.safetensors')
        assert wf['165:102']['inputs']['vae_name'] == 'flux2-vae.safetensors'
        assert wf['165:146']['inputs']['clip_name'] == 'qwen_3_8b_fp8mixed.safetensors'
        assert wf['9']['inputs']['filename_prefix'].startswith('local_FaceSwap_')
        assert captured['metadata']['model_name'] == 'klein_face_swap_dataset'
        assert len(captured['metadata']['staged_inputs']) == 2


def test_missing_target_raises_value_error(app, tmp_path):
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    with app.app_context():
        _comfy(tmp_path, cfg)
        ref = tmp_path / 'ref.png'; ref.write_bytes(_png())
        with pytest.raises(ValueError, match='target image not found'):
            fsh.enqueue_face_swap(user_id='local',
                                  target_path=str(tmp_path / 'gone.png'),
                                  ref_path=str(ref))


def test_missing_swap_lora_raises_face_swap_lora_missing(app, tmp_path):
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    with app.app_context():
        _comfy(tmp_path, cfg, swap_lora=False)   # shared Klein assets present, swap LoRA absent
        target = tmp_path / 'target.png'; target.write_bytes(_png())
        ref = tmp_path / 'ref.png'; ref.write_bytes(_png())
        with pytest.raises(fsh.FaceSwapLoraMissing):
            fsh.enqueue_face_swap(user_id='local', target_path=str(target), ref_path=str(ref))


def test_missing_shared_klein_model_raises_klein_models_missing(app, tmp_path):
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    from app.services.klein_edit_helper import KleinModelsMissing
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        os.remove(str(base / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-fp8.safetensors'))
        target = tmp_path / 'target.png'; target.write_bytes(_png())
        ref = tmp_path / 'ref.png'; ref.write_bytes(_png())
        with pytest.raises(KleinModelsMissing):
            fsh.enqueue_face_swap(user_id='local', target_path=str(target), ref_path=str(ref))
