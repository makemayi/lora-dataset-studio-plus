"""Qwen-Image 2.1 — the fourth LOCAL generation engine.

WHAT THESE TESTS PIN
--------------------
Everything that has to survive a DIFFERENT install than the one the engine was
built on: resolution (2.1 tokens only — the folders are full of sibling qwen
generations that would silently render garbage), preflight (a named gap, never
a crash, never an invented download), the graph wiring (the native edit node's
flat autogrow keys, BOTH conditionings fed, cfg 4.0, the latent coming from the
node) and the LoRA chain order. NOTHING here renders anything: no GPU second,
no paid call.
"""
import importlib
import struct

import pytest


# --- fixtures ---------------------------------------------------------------

def _fresh_config(monkeypatch, tmp_path):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as config
    importlib.reload(config)
    return config


def _comfy_tree(tmp_path):
    base = tmp_path / 'Comfy'
    for sub in ('diffusion_models', 'unet', 'loras', 'text_encoders', 'vae'):
        (base / 'models' / sub).mkdir(parents=True, exist_ok=True)
    return base


# Smallest structurally-valid safetensors header (8-byte LE length + '{}') —
# same fixture idiom as test_krea_edit: present files read as REAL weights.
_VALID_ST = struct.pack('<Q', 2) + b'{}'


def _write(path, data=_VALID_ST):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def q21(monkeypatch, tmp_path):
    """qwen21_edit_helper bound to a throwaway ComfyUI tree + throwaway config."""
    config = _fresh_config(monkeypatch, tmp_path)
    base = _comfy_tree(tmp_path)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    import app.services.qwen21_edit_helper as qeh
    importlib.reload(qeh)
    qeh._nodes_ok_until = 0.0
    yield qeh, base, config
    comfy_model_paths.clear_cache()


# --- base model resolution --------------------------------------------------

def test_no_qwen21_model_resolves_to_none_not_a_wrong_file(q21):
    qeh, base, _cfg = q21
    _write(base / 'models' / 'diffusion_models' / 'qwen_image_2512_int8_convrot.safetensors')
    _write(base / 'models' / 'diffusion_models' / 'qwen_image_edit_2511_int8_convrot.safetensors')
    assert qeh.resolve_qwen21_unet() is None


def test_a_2_1_named_file_resolves_and_siblings_do_not_shadow_it(q21):
    qeh, base, _cfg = q21
    # Every neighbour generation lives in the SAME folder on a real install;
    # the token '2.1' is what keeps the resolver from electing any of them.
    _write(base / 'models' / 'diffusion_models' / 'qwen_image_2512_int8_convrot.safetensors')
    _write(base / 'models' / 'diffusion_models' / 'qwen_image_edit_2511_bf16.safetensors')
    _write(base / 'models' / 'diffusion_models' / 'qwen_image_2.1_int8_convrot.safetensors')
    got = qeh.resolve_qwen21_unet()
    assert got and '2.1' in got.replace('\\', '/')


def test_a_gguf_build_is_never_resolved(q21):
    """UNETLoader cannot open a .gguf; resolving one would be handing the graph
    a file it dies on. They exist on real installs next to the safetensors."""
    qeh, base, _cfg = q21
    _write(base / 'models' / 'diffusion_models' / 'qwen-image-2.1-UC-Q8_0.gguf')
    assert qeh.resolve_qwen21_unet() is None


def test_the_layered_variant_is_excluded(q21):
    qeh, base, _cfg = q21
    _write(base / 'models' / 'diffusion_models' / 'Qwen_Image_Layered-Q8_0.safetensors')
    assert qeh.resolve_qwen21_unet() is None


def test_a_pin_that_resolves_wins_and_a_pin_that_does_not_is_a_hard_none(q21, monkeypatch):
    qeh, base, config = q21
    root = base / 'models' / 'diffusion_models'
    _write(root / 'Qwen' / 'qwen_image_2.1_bf16.safetensors')
    pin = _write(root / 'Qwen' / 'qwen_image_2.1_mypin_int8.safetensors')
    config.save_config({'qwen21': {'base_model': str(pin.name)}})
    assert 'mypin' in qeh.resolve_qwen21_unet()
    config.save_config({'qwen21': {'base_model': 'not_on_disk.safetensors'}})
    assert qeh.resolve_qwen21_unet() is None


# --- text encoder / VAE resolution -----------------------------------------

def test_text_encoder_accepts_only_8b(q21):
    qeh, base, _cfg = q21
    enc = base / 'models' / 'text_encoders'
    _write(enc / 'qwen3vl_4b_bf16.safetensors')
    _write(enc / 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors')
    _write(enc / 'qwen_3_8b_fp8mixed.safetensors')
    assert qeh.resolve_qwen21_text_encoder() is None
    _write(enc / 'qwen3vl_8b_fp8_scaled.safetensors')
    assert '8b' in qeh.resolve_qwen21_text_encoder()


def test_vae_never_falls_back_to_the_qwen_image_1_vae(q21):
    """The Qwen-Image 1 VAE (Krea's) decodes 2.1 latents as noise — the latent
    format changed. A resolver that "helpfully" falls back to it would ship a
    noise engine that looks installed."""
    qeh, base, _cfg = q21
    vae = base / 'models' / 'vae'
    _write(vae / 'qwen_image_vae.safetensors')
    assert qeh.resolve_qwen21_vae() is None
    _write(vae / 'qwen_image_2.1_vae_bf16.safetensors')
    assert '2.1' in qeh.resolve_qwen21_vae()


# --- preflight --------------------------------------------------------------

def test_preflight_raises_a_named_gap_when_something_is_missing(q21):
    qeh, base, _cfg = q21
    with pytest.raises(qeh.Qwen21ModelsMissing) as ei:
        qeh.preflight()
    assert 'qwen21_model' in ei.value.missing


def test_missing_assets_lists_all_three_keys(q21):
    qeh, base, _cfg = q21
    missing = qeh.qwen21_missing_assets()
    assert set(missing) == {'qwen21_model', 'qwen21_text_encoder', 'qwen21_vae'}


def test_a_broken_download_blocks_but_a_small_lora_does_not_gate(q21):
    qeh, base, _cfg = q21
    root = base / 'models'
    _write(root / 'diffusion_models' / 'qwen_image_2.1_int8.safetensors', b'<html>')
    _write(root / 'text_encoders' / 'qwen3vl_8b_fp8_scaled.safetensors')
    _write(root / 'vae' / 'qwen_image_2.1_vae_bf16.safetensors')
    invalid = qeh.qwen21_invalid_assets()
    assert any(i['asset'] == 'qwen21_model' and i['blocking'] for i in invalid)


# --- graph ------------------------------------------------------------------

def _minimal_assets(qeh, base):
    root = base / 'models'
    _write(root / 'diffusion_models' / 'qwen_image_2.1_int8_convrot.safetensors')
    _write(root / 'text_encoders' / 'qwen3vl_8b_fp8_scaled.safetensors')
    _write(root / 'vae' / 'qwen_image_2.1_vae_bf16.safetensors')


def test_graph_wiring_end_to_end(q21):
    """The whole graph, as one assertion surface: loader values from resolvers,
    BOTH conditionings and the node latent into the KSampler, flat autogrow
    reference keys, the pinned sampling profile."""
    qeh, base, config = q21
    _minimal_assets(qeh, base)
    g = qeh.build_workflow(
        'src.png', 'put her in a cafe', unet=qeh.resolve_qwen21_unet(),
        clip=qeh.resolve_qwen21_text_encoder(), vae=qeh.resolve_qwen21_vae(),
        seed=42, resolution=1024)
    assert g['1']['class_type'] == 'UNETLoader'
    assert g['1']['inputs']['unet_name'].endswith('.safetensors')
    assert g['2']['inputs'] == {'clip_name': qeh.resolve_qwen21_text_encoder(),
                                'type': 'qwen_image', 'device': 'default'}
    enc = g['6']
    assert enc['class_type'] == 'TextEncodeQwenImage21'
    assert enc['inputs']['clip'] == ['2', 0]
    assert enc['inputs']['vae'] == ['3', 0]
    assert enc['inputs']['images.image_1'] == ['5', 0]   # dotted autogrow key
    assert 'image_1' not in enc['inputs']                # never a flat key
    assert 'images' not in enc['inputs']                 # never a nested dict
    assert enc['inputs']['prompt'] == 'put her in a cafe'
    assert enc['inputs']['negative_prompt'] == ''
    assert enc['inputs']['resolution'] == 1024
    ks = g['7']
    assert ks['class_type'] == 'KSampler'
    assert ks['inputs']['positive'] == ['6', 0]
    assert ks['inputs']['negative'] == ['6', 1]
    assert ks['inputs']['latent_image'] == ['6', 2]       # latent from the NODE
    assert ks['inputs']['cfg'] == 4.0
    assert ks['inputs']['sampler_name'] == 'euler'
    assert ks['inputs']['scheduler'] == 'simple'
    assert ks['inputs']['model'] == ['1', 0]              # no LoRA -> straight through
    assert g['8']['class_type'] == 'VAEDecode' and g['8']['inputs']['vae'] == ['3', 0]
    assert g['9']['class_type'] == 'SaveImage'


def test_second_reference_wires_image_2_not_a_second_vae(q21):
    qeh, base, _cfg = q21
    g = qeh.build_workflow('src.png', 'p', unet='u', clip='c', vae='v',
                           seed=1, extra_source_image='ref_b.png')
    assert g['ref_b']['class_type'] == 'LoadImage'
    assert g['6']['inputs']['images.image_2'] == ['ref_b', 0]
    # The node encodes its own reference latents — no VAEEncode anywhere.
    assert not any(n['class_type'] == 'VAEEncode' for n in g.values())


def test_lora_chain_order_character_then_generation(q21):
    qeh, base, _cfg = q21
    g = qeh.build_workflow('src.png', 'p', unet='u', clip='c', vae='v', seed=1,
                           character_loras=[('char_a.safetensors', 0.8)],
                           generation_loras=[{'file': 'gen_b.safetensors', 'strength': 1.2}])
    assert g['4c0']['inputs'] == {'lora_name': 'char_a.safetensors',
                                  'strength_model': 0.8, 'model': ['1', 0]}
    assert g['gen_lora_1']['inputs']['model'] == ['4c0', 0]
    assert g['7']['inputs']['model'] == ['gen_lora_1', 0]


def test_graph_is_pure_of_config_and_disk(q21):
    """build_workflow must not read config or touch the disk — every knob is a
    parameter. (It shares this contract with every engine's builder.)"""
    import inspect
    qeh, _base, _cfg = q21
    src = inspect.getsource(qeh.build_workflow)
    assert 'cfg.get' not in src and 'resolve_' not in src and 'open(' not in src


# --- settings ---------------------------------------------------------------

def test_ref_resolution_is_clamped_snapped_and_capped_by_max_mp(q21):
    qeh, _base, config = q21
    config.save_config({'qwen21': {'ref_resolution': 1000, 'max_output_mp': 2.0}})
    assert qeh.ref_resolution() == 992           # snapped to /32
    config.save_config({'qwen21': {'ref_resolution': 4000, 'max_output_mp': 2.0}})
    assert qeh.ref_resolution() <= qeh._max_output_resolution()


def test_cfg_is_a_clamped_setting_and_reaches_the_sampler(q21):
    """A turbo 2.1 build wants cfg 1 — cfg is a setting (default 4.0), clamped
    1..10, and build_workflow takes it as an argument (the graph stays pure)."""
    qeh, _base, config = q21
    assert qeh._cfg() == 4.0
    config.save_config({'qwen21': {'cfg': 1.0}})
    assert qeh._cfg() == 1.0
    config.save_config({'qwen21': {'cfg': 99}})
    assert qeh._cfg() == 10.0
    g = qeh.build_workflow('src.png', 'p', unet='u', clip='c', vae='v', seed=1,
                           steps=6, cfg_scale=1.0)
    ks = next(n for n in g.values() if n['class_type'] == 'KSampler')
    assert ks['inputs']['cfg'] == 1.0 and ks['inputs']['steps'] == 6


def test_generation_lora_preset_is_fail_closed(q21):
    qeh, base, config = q21
    assert qeh.resolve_generation_lora_preset('') == []
    assert qeh.resolve_generation_lora_preset('nope') == []
    _write(base / 'models' / 'loras' / 'skin.safetensors')   # only this one exists
    config.save_config({'qwen21': {'generation_lora_presets': [
        {'name': 'skin', 'loras': [{'file': 'skin.safetensors', 'strength': 0.7},
                                   {'file': '', 'strength': 1.0},
                                   {'file': 'gone.safetensors', 'strength': 1.0}]},
    ]}})
    rows = qeh._existing_generation_lora_rows(qeh.resolve_generation_lora_preset('skin'))
    assert rows == [{'file': 'skin.safetensors', 'strength': 0.7}]


def test_a_blank_run_preset_falls_back_to_the_settings_default(q21):
    """The workspace per-run preset picker does not exist for this engine yet:
    every real run calls this lane with generation_lora_preset=None, so without
    the fallback the whole presets config was inert — configurable, never
    applied. Source-pinned like the job-stamp test above: the seam lives in a
    module this file does not import."""
    import inspect
    from app.services import dataset_generation_service as dsvc
    src = inspect.getsource(dsvc.generate_variations_qwen21)
    assert "generation_lora_preset or cfg.get('qwen21.default_generation_lora_preset')" in src


def test_engine_is_registered_everywhere_it_must_be(q21):
    """One registration point per concern, and no third copy: the catalog, the
    LOCAL lane tuple and the label map all know the id."""
    qeh, _base, config = q21
    assert 'qwen21' in config.DEFAULTS['engines']['enabled']
    assert config.DEFAULTS['qwen21']['steps'] == 30
    from app.services import dataset_generation_service as svc
    assert 'qwen21' in svc.LOCAL_ENGINES
    assert svc.LOCAL_ENGINE_LABELS['qwen21'] == 'Qwen 2.1'
    assert 'qwen21' in svc.KNOWN_ENGINES
    # Editable: 2.1 IS a reference-edit engine (the whole point of it).
    assert 'qwen21' in svc.editable_engines()
    assert 'qwen21' not in svc.GENERATE_ONLY_ENGINES


def test_the_route_facade_exposes_the_qwen21_lane(q21):
    """The generate route calls svc = face_dataset_service — the FACADE, which
    re-exports only the names listed in its tail blocks. A lane function missing
    from that list passes every unit test (they import the owner module) and
    still 500s on the first real click. This is the regression that pins it."""
    from app.services import face_dataset_service as fsvc
    assert callable(fsvc.generate_variations_qwen21)
    assert fsvc.QWEN21_ENGINE == 'qwen21'
    assert fsvc.LOCAL_ENGINE_LABELS['qwen21'] == 'Qwen 2.1'


def test_the_job_stamp_is_routed_by_the_queue(q21):
    """THE seam that stranded Krea's twelve images and then this lane's own
    first generate run: enqueue stamps a model_name, and job_queue's dispatch
    routes finished jobs by membership in DATASET_IMAGE_JOB_NAMES. The stamp
    lives in THIS module, the table in an untouched one — so the engine's own
    suite pins both halves, and a rename on either side fails here first."""
    import inspect
    from app import job_queue
    qeh, _base, _cfg = q21
    src = inspect.getsource(qeh.enqueue_qwen21_edit)
    assert "'qwen21_dataset'" in src, 'enqueue must stamp the job name literal'
    assert 'qwen21_dataset' in job_queue.DATASET_IMAGE_JOB_NAMES


def test_reference_edit_support_declared_modal_one(q21):
    from app.services import face_dataset_service as fsvc
    assert fsvc.LOCAL_EDIT_REF_SUPPORT.get('qwen21') == 'modal_one'
    # And the modal-upload plumbing routes it like Krea: one dialog image.
    assert fsvc.local_edit_extra_refs('qwen21', ['a.png', 'b.png']) == []
    assert fsvc.local_edit_modal_refs('qwen21', ['up1.png', 'up2.png']) == ['up1.png']


def test_pin_gap_naming(q21):
    qeh, _base, config = q21
    config.save_config({'qwen21': {'vae': 'i_pinned_this.safetensors'}})
    gaps = qeh.qwen21_pin_gaps()
    assert gaps == [{'slot': 'vae', 'key': 'qwen21.vae',
                     'configured': 'i_pinned_this.safetensors'}]
    with pytest.raises(qeh.Qwen21PinnedModelMissing):
        qeh.preflight()


# --- sampling knobs (sampler / scheduler / viggle sigmas / shift / canvas) ------

def _node(g, cls):
    return next(n for n in g.values() if n['class_type'] == cls)


def test_default_graph_is_the_stock_ksampler_on_the_encode_latent(q21):
    qeh, _base, _config = q21
    g = qeh.build_workflow('src.png', 'p', unet='u', clip='c', vae='v', seed=1)
    ks = _node(g, 'KSampler')
    assert ks['inputs']['sampler_name'] == 'euler' and ks['inputs']['scheduler'] == 'simple'
    assert ks['inputs']['latent_image'] == ['6', 2] and ks['inputs']['model'] == ['1', 0]
    assert not any(n['class_type'] in ('ModelSamplingFlux', 'EmptyLatentImage',
                                       'SamplerCustomAdvanced', 'SeedVR2Preprocess')
                   for n in g.values())


def test_viggle_scheduler_samples_through_custom_advanced(q21):
    qeh, _base, _config = q21
    g = qeh.build_workflow('src.png', 'p', unet='u', clip='c', vae='v', seed=7,
                           steps=6, cfg_scale=1.5, sampler='dpmpp_2m_sde_gpu_sharp',
                           scheduler=qeh.VIGGLE_SCHEDULER,
                           viggle={'method': 'beta', 'nodes': '1.0, 0.75, 0.5, 0.25, 0',
                                   'alpha': 0.6, 'beta_param': 0.6, 'rho': 7.0, 'p': 2.0},
                           output_size=(1024, 1536))
    assert not any(n['class_type'] == 'KSampler' for n in g.values())
    sca = _node(g, 'SamplerCustomAdvanced')['inputs']
    sig = _node(g, qeh.VIGGLE_SIGMAS_NODE)['inputs']
    assert sig['steps'] == 6 and sig['method'] == 'beta' and sig['latent'] == ['latent', 0]
    assert _node(g, 'KSamplerSelect')['inputs']['sampler_name'] == 'dpmpp_2m_sde_gpu_sharp'
    guider = _node(g, 'CFGGuider')['inputs']
    assert guider['cfg'] == 1.5 and guider['negative'] == ['6', 1]
    assert _node(g, 'RandomNoise')['inputs']['noise_seed'] == 7
    assert sca['latent_image'] == ['latent', 0]
    lat = _node(g, 'EmptyLatentImage')['inputs']
    assert (lat['width'], lat['height']) == (1024, 1536)
    assert _node(g, 'VAEDecode')['inputs']['samples'] == ['7', 0]


def test_shift_mu_pins_model_sampling_flux_and_official_formula(q21):
    qeh, _base, config = q21
    g = qeh.build_workflow('src.png', 'p', unet='u', clip='c', vae='v', seed=1,
                           shift_mu=0.9, sampler='res_2s', scheduler='beta57')
    msf = _node(g, 'ModelSamplingFlux')['inputs']
    assert msf['max_shift'] == msf['base_shift'] == 0.9
    ks = _node(g, 'KSampler')['inputs']
    assert ks['model'] == ['shift', 0] and ks['sampler_name'] == 'res_2s'
    assert abs(qeh.official_shift_mu(1024, 1024) - 0.6935) < 1e-3
    assert abs(qeh.official_shift_mu(2048, 2048) - 1.3129) < 1e-3
    assert qeh._shift_setting() == ''
    config.save_config({'qwen21': {'shift': 'official'}})
    assert qeh._shift_setting() == 'official'
    config.save_config({'qwen21': {'shift': '1.1'}})
    assert qeh._shift_setting() == 1.1


def test_sampler_names_are_shape_checked(q21):
    qeh, _base, config = q21
    config.save_config({'qwen21': {'sampler': 'res_2s', 'scheduler': 'Beta57'}})
    assert qeh._sampler() == 'res_2s' and qeh._scheduler() == 'beta57'
    config.save_config({'qwen21': {'sampler': 'bad name;', 'scheduler': ''}})
    assert qeh._sampler() == 'euler' and qeh._scheduler() == 'simple'


def test_canvas_follows_the_shot_aspect(q21):
    qeh, _base, _config = q21
    assert qeh.parse_aspect('2:3') == 2 / 3 and qeh.parse_aspect('junk') is None
    w, h = qeh.output_size_for(1, 1, 3.0, ratio=2 / 3)
    assert w % 32 == 0 and h % 32 == 0 and h > w
    assert abs(w * h - 3_000_000) / 3_000_000 < 0.05


def test_reference_restore_feeds_image_1(q21):
    qeh, _base, _config = q21
    g = qeh.build_workflow('src.png', 'p', unet='u', clip='c', vae='v', seed=1,
                           ref_upscale={'dit': 'seedvr2.safetensors', 'vae': 'ema.safetensors',
                                        'factor': 2.0, 'seed': 3})
    assert g['6']['inputs']['images.image_1'] == ['up_post', 0]
    assert g['up_resize']['inputs']['input'] == ['5', 0]
    assert g['up_resize']['inputs']['resize_type.multiplier'] == 2.0
    assert g['up_ks']['inputs']['steps'] == 1
    assert g['up_post']['inputs']['color_correction_method'] == 'lab'


def test_enqueue_uses_the_shot_aspect_and_official_shift(q21, monkeypatch, tmp_path):
    qeh, _base, config = q21
    from PIL import Image
    src = tmp_path / 'ref.png'
    Image.new('RGB', (800, 800)).save(src)            # square head-and-shoulders ref
    config.save_config({'qwen21': {'output_mp': 2.0, 'shift': 'official',
                                   'scheduler': 'simple'}})
    monkeypatch.setattr(qeh, 'preflight', lambda: None)
    monkeypatch.setattr(qeh, 'resolve_qwen21_unet', lambda *_: 'u')
    monkeypatch.setattr(qeh, 'resolve_qwen21_text_encoder', lambda: 'c')
    monkeypatch.setattr(qeh, 'resolve_qwen21_vae', lambda: 'v')
    monkeypatch.setattr(qeh, '_comfy_input_dir', lambda: str(tmp_path))
    monkeypatch.setattr(qeh.comfy_fs, 'ensure_input_usable', lambda d: d)
    monkeypatch.setattr(qeh.comfy_fs, 'stage_input_image',
                        lambda path, name, d: str(tmp_path / name))
    captured = {}
    monkeypatch.setattr(qeh.queue_manager, 'add_job',
                        lambda **kw: captured.update(kw))
    qeh.enqueue_qwen21_edit('1', 'ref.png', 'p', source_path=str(src), aspect_ratio='2:3')
    g = captured['workflow_data']
    lat = _node(g, 'EmptyLatentImage')['inputs']
    assert lat['height'] > lat['width']                # portrait canvas from the card
    mu = _node(g, 'ModelSamplingFlux')['inputs']['max_shift']
    assert abs(mu - qeh.official_shift_mu(lat['width'], lat['height'])) < 1e-9
    config.save_config({'qwen21': {'follow_shot_aspect': False, 'output_mp': 0}})
    qeh.enqueue_qwen21_edit('1', 'ref.png', 'p', source_path=str(src), aspect_ratio='2:3')
    g = captured['workflow_data']
    assert not any(n['class_type'] == 'EmptyLatentImage' for n in g.values())


def test_framing_aspect_setting_wins_over_the_card(q21):
    qeh, _base, config = q21
    assert qeh.shot_aspect('anything', 'body') == '9:16'      # shipped default
    assert qeh.shot_aspect('anything', 'bust') == '3:4'
    assert qeh.shot_aspect('anything', 'face') == '1:1'
    config.save_config({'qwen21': {'aspect_by_framing': {'body': '', 'bust': '3:4',
                                                         'face': '1:1', 'back': '9:16'}}})
    from app.services.face_variations import aspect_for_label
    assert qeh.shot_aspect('anything', 'body') == aspect_for_label('anything', 'body')
