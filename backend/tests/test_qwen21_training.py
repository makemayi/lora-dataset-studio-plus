"""Wave 3 — the Qwen 2.1 TRAINING family (train_type='qwen21', ai-toolkit lane).

Facts these tests pin, each verified against UPSTREAM ostris/ai-toolkit main
(the local install predates the feature):
  - the 2.1 arch lives in the `qwen_image_2` extension, `arch = "qwen_image_2"`,
    targeting QwenImage21Transformer2DModel (32 layers — the gen-1
    qwen_image extension hardcodes 60 and cannot load a 2.1 checkpoint);
  - the default base is the OFFICIAL repo Qwen/Qwen-Image-2.1 (downloaded on
    the first run, the Krea-Raw mechanic); a custom pin may be a local path,
    and a structured int8/fp8 export is refused at build time.
"""
import json
import struct

import pytest


_VALID_ST = struct.pack('<Q', 2) + b'{}'


def _write_st(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_VALID_ST)
    return path


def _write_structured_int8(path):
    """A safetensors whose tensors carry .scale_weight siblings — the shape the
    integrity reader flags as a STRUCTURED export (trainer cannot strict-load)."""
    tensors = {
        'blocks.0.weight': {'dtype': 'I8', 'shape': [4], 'data_offsets': [0, 4]},
        'blocks.0.scale_weight': {'dtype': 'BF16', 'shape': [4], 'data_offsets': [4, 12]},
    }
    header = json.dumps(tensors).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 12)
    return path


@pytest.fixture
def qtrain(app, tmp_path):
    """Throwaway ComfyUI + ai-toolkit trees, caches cleared around the test."""
    from app import config as cfg
    from app.services import comfy_model_paths, comfy_model_picker
    cfg.save_config({'comfyui': {'base_dir': str(tmp_path / 'comfy')},
                     'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    comfy_model_paths.clear_cache()
    comfy_model_picker.clear_cache()
    yield tmp_path
    comfy_model_paths.clear_cache()
    comfy_model_picker.clear_cache()


# --- the emitted job config --------------------------------------------------

def test_job_config_arch_and_official_base(app, qtrain):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Q', 'qtrig', train_type='qwen21')
        proc = lt.build_job_config(ds, str(qtrain / 'ds'), steps=1500)['config']['process'][0]
    model = proc['model']
    assert model['arch'] == 'qwen_image_2'
    assert model['name_or_path'] == 'Qwen/Qwen-Image-2.1'   # official repo, first-run download
    # The memory defaults mirror Krea: a 20B-class DiT trains quantized + low_vram.
    assert model['quantize'] is True and model['quantize_te'] is True
    assert model['low_vram'] is True
    assert proc['train']['timestep_type'] == 'weighted'   # options.ts qwen_image_2 default
    assert proc['train']['noise_scheduler'] == 'flowmatch'
    assert proc['sample']['guidance_scale'] == 4          # QWEN21_CFG, the generation engine's
    assert proc['network']['linear'] == 32                # parity with Krea/SDXL


def test_job_config_prefers_the_pinned_base(app, qtrain):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    pinned = _write_st(qtrain / 'comfy' / 'models' / 'diffusion_models' / 'qwen'
                       / 'my_own_21_build.safetensors')
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Q', 'qtrig', train_type='qwen21')
        ds.train_base_model = str(pinned)
        svc.db.session.commit()
        proc = lt.build_job_config(ds, str(qtrain / 'ds'), steps=1500)['config']['process'][0]
    assert proc['model']['name_or_path'] == str(pinned)


def test_a_structured_int8_pin_is_refused_at_build_time(app, qtrain):
    """The int8_convrot build ComfyUI loads fine is a STRUCTURED export: the
    trainer's strict load chokes on its .scale_weight tensors. A pin to one must
    refuse at build time with the shared fix sentence — never after the dataset
    export (or, on the cloud lane, after a GPU was rented). The DEFAULT base is
    the official repo, so this exercises the CUSTOM-pin path."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.services.model_integrity import QUANT_REFUSAL
    pinned = _write_structured_int8(qtrain / 'comfy' / 'models' / 'diffusion_models' / 'qwen'
                                    / 'qwen_image_2.1_int8_convrot.safetensors')
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Q', 'qtrig', train_type='qwen21')
        ds.train_base_model = str(pinned)
        svc.db.session.commit()
        with pytest.raises(ValueError, match='packed inference export'):
            lt.build_job_config(ds, str(qtrain / 'ds'), steps=1500)


def test_no_base_on_disk_names_the_fix(app, qtrain):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Q', 'qtrig', train_type='qwen21')
        # Blank base = the official repo: nothing on disk is required to BUILD
        # the config (the download happens inside ai-toolkit on the first run).
        proc = lt.build_job_config(ds, str(qtrain / 'ds'), steps=1500)['config']['process'][0]
        assert proc['model']['name_or_path'] == 'Qwen/Qwen-Image-2.1'


# --- guard / deploy / detection ----------------------------------------------

def test_aitoolkit_guard_is_exact_string(app, qtrain):
    """True only when an extensions .py carries the EXACT emitted string — the
    gen-1 qwen_image arch (60 layers) must NOT satisfy the 2.1 guard: its
    loaders cannot open a 2.1 checkpoint at all."""
    from app.services import lora_training as lt
    ext = qtrain / 'aitoolkit' / 'extensions_built_in' / 'diffusion_models' / 'qwen_image_2'
    assert lt._aitoolkit_supports_qwen_image() is False        # empty install
    _write_st(ext / 'qwen_image_2.py')                         # exists but silent
    (ext / 'qwen_image_2.py').write_text('class X:\n    arch = "qwen_image_2"\n',
                                         encoding='utf-8')
    assert lt._aitoolkit_supports_qwen_image() is True
    # The GEN-1 arch must not count: same family name, wrong architecture.
    gen1 = qtrain / 'aitoolkit' / 'extensions_built_in' / 'diffusion_models' / 'qwen_image'
    gen1.mkdir(parents=True, exist_ok=True)
    (gen1 / 'qwen_image.py').write_text('class Y:\n    arch = "qwen_image"\n',
                                        encoding='utf-8')
    (ext / 'qwen_image_2.py').unlink()
    assert lt._aitoolkit_supports_qwen_image() is False


def test_deploy_dir_and_detection_route_to_qwen21(app, qtrain):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Q', 'qtrig', train_type='qwen21')
        assert lt._lora_dest_dir(ds).replace('\\', '/').endswith('models/loras/qwen')
        # Metadata first: the trainer stamps ss_base_model_version = 'qwen_image_2'
        # (QwenImage2Model.get_base_model_version returns self.arch — upstream main).
        assert lt._family_from_base_model_version('qwen_image_2') == 'qwen21'
        # A gen-1 Qwen-Image LoRA stamps 'qwen_image' — a DIFFERENT architecture;
        # it must never be mistaken for ours (silent-drop protection).
        assert lt._family_from_base_model_version('qwen_image') is None
        # Key sniff: lora_keys_use_comfy_prefix + block name 'transformer_blocks'.
        keys = ['diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight',
                'diffusion_model.transformer_blocks.0.attn.to_k.lora_B.weight']
        assert lt._lora_arch_from_keys(keys) == 'qwen21'
        # No cross-detect: a Krea file (wq/gate) stays Krea even though both live
        # under diffusion_model.*.
        krea_keys = ['diffusion_model.blocks.0.attn.wq.lora_A.weight']
        assert lt._lora_arch_from_keys(krea_keys) == 'krea'


def test_qwen21_is_a_registered_train_type(app):
    from app.services import face_dataset_service as svc
    from app.services import cloud_training
    import inspect
    assert 'qwen21' in svc.TRAIN_TYPES
    # The cloud lane refuses the family BEFORE any reservation (the only
    # verified 2.1 weights are local files a pod cannot receive).
    src = inspect.getsource(cloud_training.launch_cloud_training)
    assert 'qwen21' in src and 'train it locally for now' in src
