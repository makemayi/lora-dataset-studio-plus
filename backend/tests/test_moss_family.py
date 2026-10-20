"""MOSS-Image ('moss') — a first-class training family (OpenMOSS-Team
MOSS-Image-Base/SFT, 5B single-stream DiT, upstream ai-toolkit 2026-10-16).

What matters here: the family is accepted end-to-end, the extension-arch guard
refuses a launch on an ai-toolkit that lacks the 'moss_image' arch (silent
SD-loader fallback otherwise), the job-config is a FAITHFUL PORT of the verified
local run jcl001_moss_image_bs4 (rank 32/32, bf16, quantize both with convrot8
default, sigmoid timesteps — the run omitted the key so ai-toolkit's default
applies —, TE frozen per OpenMOSS guidance, resolution [768,1024], preview
CFG 3 / 25 steps / 832x1216 / empty negative), the official base is PUBLIC,
deploy routing lands in loras/moss, and the cloud path is REFUSED (the pod
image predates the moss_image extension).
"""
import pytest


def _configure_aitoolkit(tmp_path, app, supports_moss=True):
    """Fake ai-toolkit install (venv python + run.py), with or without the
    'moss_image' extension arch under extensions_built_in."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    ext = root / 'extensions_built_in' / 'diffusion_models' / 'moss_image'
    ext.mkdir(parents=True)
    if supports_moss:
        (ext / 'moss_image.py').write_text(
            'class MossImageModel:\n    arch = "moss_image"\n', encoding='utf-8')
    else:
        # An incidental 'moss' mention in a comment must NOT count (exact arch).
        (ext / 'placeholder.py').write_text(
            '# moss support not merged yet\n'
            'class Placeholder:\n    arch = "flux"\n', encoding='utf-8')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


# --- 1) train_type accepted / normalized ---------------------------------------

def test_normalize_train_type_accepts_moss():
    from app.services import face_dataset_service as svc
    assert svc.normalize_train_type('moss') == 'moss'
    assert svc.normalize_train_type('MOSS') == 'moss'       # case-fold
    assert svc.normalize_train_type('bogus') == 'zimage'    # unknown -> default
    assert svc.normalize_train_type(None) == 'zimage'


# --- 2) extension-arch guard ----------------------------------------------------

def test_aitoolkit_supports_moss_scans_extension_arch(app, tmp_path):
    from app.services import lora_training as lt
    _configure_aitoolkit(tmp_path, app, supports_moss=True)
    with app.app_context():
        assert lt._aitoolkit_supports_moss() is True


def test_aitoolkit_supports_moss_false_without_arch(app, tmp_path):
    """No 'moss_image' arch on disk -> False, even with 'moss' mentioned in a
    comment (exact-arch match, no substring false positive)."""
    from app.services import lora_training as lt
    with app.app_context():
        assert lt._aitoolkit_supports_moss() is False       # not configured
    _configure_aitoolkit(tmp_path, app, supports_moss=False)
    with app.app_context():
        assert lt._aitoolkit_supports_moss() is False


# --- 3) job-config: faithful port of the verified local run ---------------------

def test_build_job_config_moss(app, tmp_path):
    """arch 'moss_image', PUBLIC Base (OpenMOSS LoRA guidance), quantize BOTH on
    with convrot8 (the run's trio pattern), sigmoid timesteps (the run omitted
    the key -> ai-toolkit default), TE frozen, NO unload_text_encoder (the run
    didn't use it), rank 32/32, preview CFG 3 / 25 steps / 832x1216 / empty neg."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'MO', 'zchar_mo', train_type='moss')
        folder = tmp_path / 'ds'; folder.mkdir()
        p = lt.build_job_config(ds, str(folder), steps=1200)['config']['process'][0]
        m = p['model']
        assert m['arch'] == 'moss_image'
        assert m['name_or_path'] == 'OpenMOSS-Team/MOSS-Image-Base'
        assert m['quantize'] is True and m['quantize_te'] is True
        assert m['qtype'] == 'convrot8' and m['qtype_te'] == 'convrot8'
        assert 'low_vram' not in m
        assert 'unload_text_encoder' not in p['train']
        assert p['train']['train_text_encoder'] is False     # official: freeze the TE
        # 3090 24G 实测：关掉它会溢出显存（Windows 换页 → 步数不动），必须开启。
        assert p['train']['gradient_checkpointing'] is True
        assert p['train']['batch_size'] == 1                  # the run's final decision
        assert p['train']['timestep_type'] == 'sigmoid'
        assert p['train']['noise_scheduler'] == 'flowmatch'
        assert p['train']['dtype'] == 'bf16'
        assert p['datasets'][0]['cache_text_embeddings'] is True
        assert p['datasets'][0]['cache_latents_to_disk'] is True
        assert p['datasets'][0]['caption_ext'] == 'txt'
        assert p['datasets'][0]['resolution'] == [768, 1024]
        assert p['sample']['sampler'] == 'flowmatch'
        assert p['sample']['neg'] == ''                       # the run's empty negative
        assert p['sample']['guidance_scale'] == 3             # MOSS official CFG
        assert p['sample']['sample_steps'] == 25
        assert p['sample']['width'] == 832 and p['sample']['height'] == 1216
        assert p['network'] == {'type': 'lora', 'linear': 32, 'linear_alpha': 32,
                                'network_kwargs': {'ignore_if_contains': ['layers.22.']}}
        assert m['model_kwargs'] == {'uncond_skip_layers': [22]}   # paired with the exclusion


def test_build_job_config_moss_explicit_sft_rides_through(app, tmp_path):
    """Picking MOSS-Image-SFT in the base list must reach name_or_path verbatim
    (a repo id is NOT custom weights, and must not silently fall back to Base)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'MS', 'zchar_ms', train_type='moss')
        ds.train_base_model = lt.MOSS_SFT
        svc.db.session.commit()
        folder = tmp_path / 'ds'; folder.mkdir()
        p = lt.build_job_config(ds, str(folder), steps=100)['config']['process'][0]
        assert p['model']['name_or_path'] == 'OpenMOSS-Team/MOSS-Image-SFT'


def test_build_job_config_moss_custom_trio_pins_te_vae(app, tmp_path):
    """The verified run's pattern: local convrot DiT + pinned TE/VAE files ride
    through as te_name_or_path / vae_path."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'MC', 'zchar_mc', train_type='moss')
        ds.train_base_model = r'F:\weights\moss_image_sft_int8_convrot.safetensors'
        ds.train_te_path = r'F:\weights\qwen3.5_9b_int8_convrot.safetensors'
        ds.train_vae_path = r'F:\weights\flux2-vae.safetensors'
        svc.db.session.commit()
        folder = tmp_path / 'ds'; folder.mkdir()
        p = lt.build_job_config(ds, str(folder), steps=100)['config']['process'][0]
        m = p['model']
        assert m['name_or_path'] == r'F:\weights\moss_image_sft_int8_convrot.safetensors'
        assert m['te_name_or_path'] == r'F:\weights\qwen3.5_9b_int8_convrot.safetensors'
        assert m['vae_path'] == r'F:\weights\flux2-vae.safetensors'


def test_official_base_repo_moss_is_public(app):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'MB', 'zchar_mb', train_type='moss')
        assert lt.official_base_repo(ds, 'moss') == 'OpenMOSS-Team/MOSS-Image-Base'
        ds.train_base_model = r'C:\weights\my-moss.safetensors'
        svc.db.session.commit()
        assert lt.official_base_repo(ds, 'moss') is None


# --- 4) cloud path REFUSED (pod image predates the extension) -------------------

def test_cloud_training_refuses_moss(app, tmp_path, monkeypatch):
    from app.services import cloud_training as ct
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    # Fake VAST key so the moss refusal (which sits AFTER the key check) is reached.
    monkeypatch.setattr(ct.cfg, 'secret',
                        lambda k: 'fake-key' if k == 'VAST_API_KEY' else None)
    _configure_aitoolkit(tmp_path, app, supports_moss=True)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'MR', 'zchar_mr', train_type='moss')
        with pytest.raises(ValueError, match='MOSS-Image cloud training'):
            ct.gpu_tiers(LOCAL_USER, ds.id)
        with pytest.raises(ValueError, match='MOSS-Image cloud training'):
            ct.launch_cloud_training(LOCAL_USER, ds.id, train_type='moss')


# --- 5) deploy routing + run-name isolation -------------------------------------

def test_lora_dest_dir_routes_moss(app, tmp_path):
    import os
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'comfyui': {'base_dir': str(tmp_path / 'comfy')}})
        ds = svc.create_dataset(LOCAL_USER, 'MD', 'zchar_md', train_type='moss')
        dest = lt._lora_dest_dir(ds)
        assert dest.replace('/', os.sep).endswith(
            os.sep.join(('models', 'loras', 'moss')))


def test_dest_base_tag_moss_isolated_from_zimage(app):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'MT', 'zchar_mt', train_type='moss')
        assert lt._dest_base_tag(ds) == '_MOSS-Image-Base'
        assert lt._run_name(ds).endswith('_MOSS-Image-Base')


def test_family_of_lora_classifies_moss_folder():
    from app.utils.comfyui import family_of_lora, FAMILY_LABELS
    assert family_of_lora(r'moss\x.safetensors') == 'moss'
    assert family_of_lora('moss/x.safetensors') == 'moss'
    assert FAMILY_LABELS['moss'] == 'MOSS-Image'


# --- 6) LoRA/keyword detectors ---------------------------------------------------

def test_lora_arch_from_keys_detects_moss_before_zimage():
    """MOSS keys use 'diffusion_model.layers.*' like Z-Image, but the modulation
    heads are unique — the moss check must win (it is ordered first)."""
    from app.services import lora_training as lt
    keys = ['diffusion_model.layers.0.attn.qkv.lora_A.weight',
            'diffusion_model.video_head_modulation.lora_B.weight']
    assert lt._lora_arch_from_keys(keys) == 'moss'


def test_lora_metadata_version_maps_moss():
    from app.services import lora_training as lt
    # ai-toolkit stamps MossImageModel.get_base_model_version() → exactly 'moss_image'
    assert lt._family_from_base_model_version('moss_image') == 'moss'
    assert lt._family_from_base_model_version('anima') == 'anima'  # no cross-talk


def test_safetensors_base_sniff_detects_moss_dit():
    from app.services import lora_training as lt
    keys = ['layers.0.attn.proj.qkv.weight',
            'video_head_modulation.weight',
            'shared_block_modulation.weight']
    assert lt._detect_safetensors_arch(keys) == 'moss_image'


# --- 7) built-in presets ---------------------------------------------------------

def test_builtin_presets_include_moss():
    """One Character + one Concept preset for MOSS-Image; the character one is
    the verified run (rank 32/32, sigmoid), the concept one extrapolated."""
    from app.services.lora_training import BUILTIN_TRAIN_PRESETS
    moss = [pr for pr in BUILTIN_TRAIN_PRESETS if pr['train_type'] == 'moss']
    kinds = {pr['dataset_kind'] for pr in moss}
    assert kinds == {'character', 'concept'}
    for pr in moss:
        assert pr['settings']['timestep_type'] == 'sigmoid'
    char = next(pr for pr in moss if pr['dataset_kind'] == 'character')
    assert char['settings']['rank'] == 32 and char['settings']['alpha'] == 32


def test_moss_timestep_type_is_moss_only(app, tmp_path):
    """'moss' (MOSS-Image pretraining noise, local ai-toolkit patch) is offered and
    emitted for the moss family only; other families refuse it at save time."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'MO', 'zchar_mo', train_type='moss')
        eff = lt.update_train_settings(LOCAL_USER, ds.id, {'timestep_type': 'moss'})
        assert eff['timestep_type'] == 'moss' and 'moss' in eff['timestep_type_choices']
        folder = tmp_path / 'ds'; folder.mkdir()
        p = lt.build_job_config(ds, str(folder), steps=100)['config']['process'][0]
        assert p['train']['timestep_type'] == 'moss'
        other = svc.create_dataset(LOCAL_USER, 'ZI', 'zchar_zi', train_type='zimage')
        with pytest.raises(ValueError):
            lt.update_train_settings(LOCAL_USER, other.id, {'timestep_type': 'moss'})


def test_moss_character_steps_follow_epochs():
    """MOSS character steps = images x epochs (60 up to 30 images, 50 above),
    rounded up to the 250-step save cadence; grad accumulation divides."""
    from types import SimpleNamespace
    from app.services import lora_training as lt
    ds = SimpleNamespace(train_type='moss', train_settings=None)
    cases = {15: 1000, 20: 1250, 25: 1500, 30: 2000, 31: 1750, 40: 2000,
             50: 2500, 60: 3000, 1: 250}
    for n, want in cases.items():
        assert lt._moss_epoch_steps(ds, n) == want, n
    assert lt._moss_epochs(30) == 60 and lt._moss_epochs(31) == 50


def test_moss_recommended_steps_uses_epochs(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'ME', 'zchar_me', train_type='moss')
        for i in range(40):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, filename=f'{i}.png',
                                                status='keep'))
        svc.db.session.commit()
        assert lt.recommended_steps(ds.id) == 2000          # 40 x 50
        info = lt.recommended_steps_info(ds.id)
        assert info['steps'] == 2000 and '50 epochs' in info['rationale']
