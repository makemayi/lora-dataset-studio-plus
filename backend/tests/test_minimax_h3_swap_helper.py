"""The MiniMax H3 head swap — the second engine behind the 🎭↔ button.

WHAT THESE PIN
--------------
The graph was validated on ONE ComfyUI, and everything that made it work there
is a fact about that install: an INT8 H3 build, six custom-node packs, a
Z-Image checkpoint and a private LoRA stack. Nobody else has that machine, so
what has to survive a different one is:

* the three optional stages are REMOVED cleanly when off — including their
  loaders, their samplers and their custom-node requirements, or an install
  without the Impact Pack would be told to install it for a stage it is not
  running;
* the two LoadImage roles never swap (backwards still renders something
  plausible, with the wrong face surviving);
* the seed moves on EVERY sampler the job contains, or a batch shares a stage;
* the save prefix is unique per job, or ComfyUI's own counter re-issues one
  filename and every tile displays the same image.

NOTHING here renders anything: not one GPU second.
"""
import importlib
import struct

import pytest


def _fresh_config(monkeypatch, tmp_path):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as config
    importlib.reload(config)
    return config


_VALID_ST = struct.pack('<Q', 2) + b'{}'


def _write(path, data=_VALID_ST):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _h3_install(base):
    m = base / 'models'
    _write(m / 'diffusion_models' / 'minimax_h3_ref2va_pruned_int8_convrot.safetensors')
    _write(m / 'text_encoders' / 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
    _write(m / 'vae' / 'minimax_h3_video_vae_fp16.safetensors')
    _write(m / 'vae' / 'minimax_h3_audio_vae_fp32.safetensors')
    _write(m / 'clip_vision' / 'CLIP-ViT-H-fp16.safetensors')


@pytest.fixture
def swap(monkeypatch, tmp_path):
    """minimax_h3_swap_helper bound to a throwaway ComfyUI tree + config, with a
    complete H3 install and an /object_info probe that answers "everything is
    installed" — each test then takes away what it is about."""
    config = _fresh_config(monkeypatch, tmp_path)
    base = tmp_path / 'Comfy'
    for sub in ('diffusion_models', 'unet', 'loras', 'text_encoders', 'vae',
                'clip_vision'):
        (base / 'models' / sub).mkdir(parents=True, exist_ok=True)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    _h3_install(base)
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    import app.services.minimax_h3_helper as mh
    import app.services.minimax_h3_swap_helper as sh
    importlib.reload(mh)
    importlib.reload(sh)
    mh._nodes_ok_until = 0.0
    # Fail-open probe by default: [] missing, every accelerator available.
    import app.utils.comfyui as comfyui
    monkeypatch.setattr(comfyui, 'fetch_object_info_classes', lambda: None)
    # load_workflow_local logs through current_app; a bare context is enough and
    # keeps this file independent of the DB-backed `app` fixture.
    import flask
    ctx = flask.Flask(__name__).app_context()
    ctx.push()
    try:
        yield sh, mh, base, config
    finally:
        ctx.pop()
        comfy_model_paths.clear_cache()


def _set(config, section, **values):
    """Write settings the way the app does — there is no config.set()."""
    config.save_config({section: values})


def _build(sh, **kw):
    kw.setdefault('filename_prefix', 'local_H3Swap_abcd1234')
    return sh.build_swap_workflow('t.png', 'r.png', **kw)


def _classes(wf):
    return {n['class_type'] for n in wf.values()}


# --- the three optional stages ----------------------------------------------

def test_every_stage_off_by_default(swap):
    sh, _mh, _base, _config = swap
    assert sh.enabled_stages() == {'hair_removal': False, 'lama': False,
                                   'face_detail': False}


def test_stages_off_removes_their_whole_branch(swap):
    """Not just the node: the Klein loaders, the Z-Image checkpoint, the LaMa
    node and every custom class only they needed must leave with them."""
    sh, _mh, _base, _config = swap
    wf, kept = _build(sh)
    assert kept == []
    gone = {'LayerUtility: LaMa', 'DetailerForEach', 'FaceSegment', 'MaskToSEGS',
            'ZImageTurboLoraLoader', 'Power Lora Loader (rgthree)',
            'ModelSamplingAuraFlow', 'EmptyFlux2LatentImage', 'Flux2Scheduler',
            'CFGGuider'}
    assert not (_classes(wf) & gone), _classes(wf) & gone
    for tail in ('426:423:65', '426:411', '426:396:370'):
        assert tail not in wf
    # ...and the graph still runs end to end.
    assert wf['412']['inputs']['images']
    assert 'MiniMaxH3ReferenceToVideo' in _classes(wf)


def test_a_stage_that_is_off_hands_its_consumers_the_right_fallback(swap):
    """The fallbacks are what the maintainer's own bypassed export wired. Get
    one wrong and the job still renders — from the wrong picture."""
    sh, _mh, _base, _config = swap
    wf, _ = _build(sh)
    assert wf['426:340']['inputs']['images'] == ['426:401', 0]
    assert wf['426:402']['inputs']['image'] == ['426:401', 0]
    assert wf['426:413']['inputs']['image'] == ['426:402', 1]
    assert wf['426:406']['inputs']['inpainted_image'] == ['426:304', 0]


def test_lama_on_sits_between_the_crop_and_the_overlay(swap):
    sh, _mh, _base, _config = swap
    wf, kept = _build(sh, stages={'lama': True})
    assert kept == ['lama']
    assert wf['426:411']['inputs']['image'] == ['426:402', 1]
    assert wf['426:411']['inputs']['mask'] == ['426:408', 0]
    assert wf['426:413']['inputs']['image'] == ['426:411', 0]


def test_face_detail_on_moves_its_own_seed(swap):
    """The detailer is a second sampler. Leaving its seed at the committed 0
    makes every tile of a batch share that pass."""
    sh, _mh, _base, _config = swap
    wf, kept = _build(sh, stages={'face_detail': True})
    assert kept == ['face_detail']
    assert wf['426:406']['inputs']['inpainted_image'] == ['426:396:370', 0]
    assert wf['426:396:370']['inputs']['seed'] != 0


def test_hair_removal_on_requires_and_resolves_the_klein_assets(swap):
    """The stage is a full second engine. With no Klein weights on disk it must
    say so BEFORE anything is queued, not fail inside ComfyUI."""
    sh, _mh, base, _config = swap
    from app.services.klein_edit_helper import KleinModelsMissing
    with pytest.raises(KleinModelsMissing):
        _build(sh, stages={'hair_removal': True})

    m = base / 'models'
    _write(m / 'diffusion_models' / 'klein' / 'flux-2-klein-9b-int8.safetensors')
    _write(m / 'text_encoders' / 'qwen_3_8b_fp8mixed.safetensors')
    _write(m / 'vae' / 'flux2-vae.safetensors')
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    wf, kept = _build(sh, stages={'hair_removal': True})
    assert kept == ['hair_removal']
    assert wf['426:340']['inputs']['images'] == ['426:423:65', 0]
    assert wf['426:423:73']['inputs']['noise_seed'] != 0
    assert 'klein' in wf['426:423:424']['inputs']['unet_name'].lower()


# --- resolution and degradation ---------------------------------------------

def test_every_h3_loader_is_re_resolved(swap):
    """The committed graph names the maintainer's files. Every one of them has
    to be replaced by what this install actually has."""
    sh, _mh, _base, _config = swap
    wf, _ = _build(sh)
    assert wf['426:312']['inputs']['unet_name'].endswith(
        'minimax_h3_ref2va_pruned_int8_convrot.safetensors')
    assert wf['426:130']['inputs']['clip_name'].endswith(
        'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
    assert wf['426:121']['inputs']['vae_name'].endswith(
        'minimax_h3_video_vae_fp16.safetensors')
    assert wf['426:172']['inputs']['vae_name'].endswith(
        'minimax_h3_audio_vae_fp32.safetensors')
    assert wf['426:305']['inputs']['clip_name'].endswith('CLIP-ViT-H-fp16.safetensors')


def test_a_non_int8_build_degrades_the_w8a8_loader(swap, monkeypatch):
    """Handing a W8A8 loader an fp16 checkpoint is not slower — it fails."""
    sh, mh, _base, _config = swap
    monkeypatch.setattr(mh, 'resolve_h3_unet',
                        lambda *a, **k: 'minimax_h3_ref2va_fp16.safetensors')
    wf, _ = _build(sh)
    assert wf['426:312']['class_type'] == 'UNETLoader'
    assert wf['426:312']['inputs']['unet_name'] == 'minimax_h3_ref2va_fp16.safetensors'


def test_a_missing_h3_asset_is_named_before_anything_is_queued(swap):
    sh, _mh, base, _config = swap
    (base / 'models' / 'clip_vision' / 'CLIP-ViT-H-fp16.safetensors').unlink()
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    with pytest.raises(sh.mh.MinimaxH3ModelsMissing) as e:
        _build(sh)
    assert 'h3_clip_vision' in e.value.missing


def test_the_accelerators_are_dropped_when_the_setting_is_off(swap):
    sh, _mh, _base, config = swap
    _set(config, 'minimax_h3', use_rtx_upscale=False, use_speed_nodes=False)
    wf, _ = _build(sh)
    assert 'RTXVideoSuperResolution' not in _classes(wf)
    assert 'SpectrumApplyMiniMaxH3' not in _classes(wf)
    assert 'PathchSageAttentionKJ' not in _classes(wf)
    # ...and the chain closed over the gaps rather than dangling.
    assert wf['426:311']['inputs']['images'] == ['114', 0]
    assert wf['412']['inputs']['images'] == ['426:406', 0]
    assert wf['426:126']['inputs']['model'] == ['426:312', 0]


def test_a_missing_custom_node_names_its_pack(swap, monkeypatch):
    """Six packs, so "install MinimaxH3-Image" is the wrong instruction most of
    the time — and a missing pack must be reported, never rendered around."""
    sh, _mh, _base, _config = swap
    import app.utils.comfyui as comfyui
    monkeypatch.setattr(comfyui, 'fetch_object_info_classes',
                        lambda: {'LoadImage', 'SaveImage'})
    with pytest.raises(sh.H3SwapNodesMissing) as e:
        _build(sh)
    packs = {h['pack'] for h in e.value.node_packs}
    assert 'ComfyUI_LayerStyle' in packs
    assert 'ComfyUI-Inpaint-CropAndStitch' in packs
    # A stage that is OFF must never appear in the shopping list.
    assert 'ComfyUI-Impact-Pack' not in packs


def test_an_unreachable_probe_does_not_block_the_swap(swap):
    """A probe timeout is not evidence of a missing pack."""
    sh, _mh, _base, _config = swap          # fixture's probe returns None
    wf, _ = _build(sh)
    assert wf['412']['inputs']['filename_prefix'] == 'local_H3Swap_abcd1234'


# --- the per-run values -----------------------------------------------------

def test_the_two_load_images_take_the_staged_names(swap):
    sh, _mh, _base, _config = swap
    wf, _ = _build(sh)
    assert wf['313']['inputs']['image'] == 't.png'
    assert wf['114']['inputs']['image'] == 'r.png'


def test_the_seed_moves_every_run(swap):
    sh, _mh, _base, _config = swap
    seeds = {_build(sh)[0]['426:131']['inputs']['noise_seed'] for _ in range(5)}
    assert len(seeds) > 1 and 0 not in seeds


def test_the_packet_length_is_snapped_onto_the_nodes_grid(swap):
    """An off-step length is a validation error, i.e. a whole batch of dead
    tiles — the node's minimum is 5 and its step is 17."""
    sh, _mh, _base, config = swap
    _set(config, 'minimax_h3', length=30)
    wf, _ = _build(sh)
    assert wf['426:139']['inputs']['value'] == 22


def test_the_prompt_override_replaces_the_graphs_own(swap):
    sh, _mh, _base, config = swap
    shipped = _build(sh)[0]['426:170']['inputs']['prompt']
    assert shipped
    _set(config, 'minimax_h3', swap_prompt='  put the head from <Picture 1> in  ')
    wf, _ = _build(sh)
    assert wf['426:170']['inputs']['prompt'] == 'put the head from <Picture 1> in'


def test_a_blank_prompt_override_keeps_the_tuned_prompt(swap):
    sh, _mh, _base, config = swap
    _set(config, 'minimax_h3', swap_prompt='   ')
    assert _build(sh)[0]['426:170']['inputs']['prompt'].strip()


# --- which engine the button runs -------------------------------------------

def test_the_swap_engine_setting_picks_the_helper(swap):
    sh, _mh, _base, config = swap
    from app.services import dataset_generation_service as dgs
    importlib.reload(dgs)
    assert dgs.resolve_face_swap_engine() == 'klein'
    _set(config, 'face_swap', engine='minimax_h3')
    assert dgs.resolve_face_swap_engine() == 'minimax_h3'
    # An explicit pick wins; junk falls BACK rather than raising, so a stale tab
    # degrades to the historical behaviour instead of refusing the click.
    assert dgs.resolve_face_swap_engine('klein') == 'klein'
    assert dgs.resolve_face_swap_engine('nope') == 'minimax_h3'
    _set(config, 'face_swap', engine='nonsense')
    assert dgs.resolve_face_swap_engine() == 'klein'


def test_the_two_engines_share_one_enqueue_contract(swap):
    """dataset_generation_service picks between them by config alone, so a
    signature drift would only show up as a TypeError on a user's click."""
    import inspect
    sh, _mh, _base, _config = swap
    from app.services import face_swap_helper as fsh
    assert (list(inspect.signature(sh.enqueue_h3_swap).parameters)
            == list(inspect.signature(fsh.enqueue_face_swap).parameters))


def test_the_job_name_is_registered_for_completion_dispatch(swap):
    """The failure this guards produced NO error at all: the image renders, the
    queue marks it done, and the row stays pending forever."""
    from app.job_queue import DATASET_IMAGE_JOB_NAMES
    assert 'minimax_h3_face_swap_dataset' in DATASET_IMAGE_JOB_NAMES


# --- how much of the shot the swap looks at ---------------------------------

def test_the_crop_reaches_past_the_head_by_default(swap):
    """The graph shipped at 1.3 — head only, in every photo. That is the most
    pixels per head, and it also crops away the shoulders the prompt asks the
    model to size the head against."""
    sh, _mh, _base, _config = swap
    wf, _ = _build(sh)
    assert sh.DEFAULT_CONTEXT_FACTOR == 3.0
    assert wf['426:402']['inputs']['context_from_mask_extend_factor'] == 3.0


def test_the_context_factor_is_configurable_and_bounded(swap):
    """The node accepts up to 100, which is meaningless: past the point where
    the crop covers the frame the number does nothing, and every value on the
    way there costs head pixels."""
    sh, _mh, _base, config = swap
    _set(config, 'face_swap', h3_context_factor=2.0)
    assert _build(sh)[0]['426:402']['inputs']['context_from_mask_extend_factor'] == 2.0
    _set(config, 'face_swap', h3_context_factor=99)
    assert (_build(sh)[0]['426:402']['inputs']['context_from_mask_extend_factor']
            == sh.CONTEXT_FACTOR_MAX)
    _set(config, 'face_swap', h3_context_factor=0.2)
    assert (_build(sh)[0]['426:402']['inputs']['context_from_mask_extend_factor']
            == sh.CONTEXT_FACTOR_MIN)


def test_junk_in_the_context_factor_frames_the_shot_rather_than_refusing_it(swap):
    """It decides framing, not correctness. A bad value in config must not cost
    the user the swap."""
    sh, _mh, _base, config = swap
    _set(config, 'face_swap', h3_context_factor='wide please')
    assert (_build(sh)[0]['426:402']['inputs']['context_from_mask_extend_factor']
            == sh.DEFAULT_CONTEXT_FACTOR)


# --- the "white face" levers -------------------------------------------------
# The overlay paints the head out with `mask * opacity` as its alpha, so at 1.0
# the model is handed a structureless slab and sometimes paints the slab back.
# And the frame selector shipped weighing sharpness and exposure only, so a
# blank face competed on equal terms with a good one.

def test_the_frame_selector_asks_whether_the_face_looks_like_the_reference(swap):
    sh, mh, _base, _config = swap
    wf, _ = _build(sh)
    assert wf['426:304']['inputs']['weight_reference'] == mh.DEFAULT_FRAME_WEIGHT_REFERENCE
    assert wf['426:304']['inputs']['weight_reference'] > 0


def test_the_frame_reference_weight_follows_the_h3_engine_setting(swap):
    """One setting, both lanes — a swap that judged frames differently from the
    generation engine would be a second dial nobody knew they owned."""
    sh, _mh, _base, config = swap
    _set(config, 'minimax_h3', frame_weight_reference=0.25)
    assert _build(sh)[0]['426:304']['inputs']['weight_reference'] == 0.25


def test_the_head_placeholder_opacity_is_configurable(swap):
    sh, _mh, _base, config = swap
    assert _build(sh)[0]['426:413']['inputs']['mask_opacity'] == 1.0
    _set(config, 'face_swap', h3_mask_opacity=0.75)
    assert _build(sh)[0]['426:413']['inputs']['mask_opacity'] == 0.75
    _set(config, 'face_swap', h3_mask_opacity=5)
    assert _build(sh)[0]['426:413']['inputs']['mask_opacity'] == 1.0
    _set(config, 'face_swap', h3_mask_opacity='opaque-ish')
    assert _build(sh)[0]['426:413']['inputs']['mask_opacity'] == sh.DEFAULT_MASK_OPACITY


# --- where the mask comes from ----------------------------------------------

def test_the_graph_masks_by_itself_by_default(swap):
    sh, _mh, _base, _config = swap
    assert sh.mask_source() == 'graph'
    wf, _ = _build(sh)
    assert wf['426:340']['class_type'] == 'LayerMask: PersonMaskUltra'
    assert wf['426:402']['inputs']['mask'] == ['426:340', 1]


def test_an_app_mask_replaces_person_mask_ultra_entirely(swap):
    """And takes the ComfyUI_LayerStyle dependency with it: the pack is only
    still needed if the LaMa stage is on."""
    sh, _mh, _base, _config = swap
    wf, _ = _build(sh, mask_image='mask.png')
    assert wf['426:402']['inputs']['mask'] == ['app_mask_to_mask', 0]
    assert 'LayerMask: PersonMaskUltra' not in _classes(wf)
    assert wf['app_mask_load']['inputs']['image'] == 'mask.png'
    assert wf['app_mask_to_mask']['inputs']['channel'] == 'red'


def test_the_app_mask_is_resized_by_the_graphs_own_node(swap):
    """Not app-side: ResizeImagesByLongerEdge truncates
    (new_h = int(h * edge / w)), and reproducing that arithmetic anywhere else
    is how an off-by-one size mismatch reaches ComfyUI instead of a test."""
    sh, _mh, _base, _config = swap
    wf, _ = _build(sh, mask_image='mask.png')
    target_edge = wf['426:401']['inputs']['longer_edge']
    assert wf['app_mask_resize']['class_type'] == 'ResizeImagesByLongerEdge'
    assert wf['app_mask_resize']['inputs']['longer_edge'] == target_edge
    assert wf['app_mask_resize']['inputs']['images'] == ['app_mask_load', 0]


def test_the_default_region_includes_what_is_WORN_on_the_head(swap):
    """A head is not one object to a segmenter: asked for as 'head' alone the
    mask comes back with a hole where the glasses are, and the swap then paints
    a new face around the old pair. Reported on a real tile."""
    sh, _mh, _base, _config = swap
    phrases = [p.strip() for p in sh.mask_prompt().split(',')]
    assert phrases[0] == 'head'
    for worn in ('glasses', 'sunglasses', 'hat', 'earrings'):
        assert worn in phrases, f'{worn} would be left behind by the swap'


def test_the_mask_source_and_phrase_are_configurable(swap):
    sh, _mh, _base, config = swap
    assert sh.mask_prompt().startswith('head')
    _set(config, 'face_swap', h3_mask_source='app', h3_mask_prompt='  hair  ')
    assert sh.mask_source() == 'app'
    assert sh.mask_prompt() == 'hair'
    # Fail-safe: junk falls back to the graph rather than refusing the swap.
    _set(config, 'face_swap', h3_mask_source='telepathy', h3_mask_prompt='')
    assert sh.mask_source() == 'graph'
    assert sh.mask_prompt() == sh.DEFAULT_MASK_PROMPT


def test_the_lama_stage_does_not_run_the_model_that_crashes(swap):
    """The graph asks for 'zits', which pads to a multiple of 32 and drives a
    256->512 structure upsampler — on an arbitrarily sized inpaint crop it dies
    inside TorchScript. Reported as "enabling LaMa errors"."""
    sh, _mh, _base, config = swap
    assert sh.DEFAULT_LAMA_MODEL == 'lama'
    wf, kept = _build(sh, stages={'lama': True})
    assert kept == ['lama']
    assert wf['426:411']['inputs']['lama_model'] == 'lama'
    _set(config, 'face_swap', h3_lama_model='mat')
    assert _build(sh, stages={'lama': True})[0]['426:411']['inputs']['lama_model'] == 'mat'
    # Blank falls back rather than sending an empty enum ComfyUI would refuse.
    _set(config, 'face_swap', h3_lama_model='   ')
    assert _build(sh, stages={'lama': True})[0]['426:411']['inputs']['lama_model'] == 'lama'


def test_the_blend_width_is_configurable_and_clamped(swap):
    """The composite is the half of "it does not blend" that no prompt reaches:
    the join is a feather this wide, written on the crop node and read back off
    the stitcher."""
    sh, _mh, _base, config = swap
    wf, _ = _build(sh)
    assert wf['426:402']['inputs']['mask_blend_pixels'] == sh.DEFAULT_BLEND_PIXELS
    _set(config, 'face_swap', h3_blend_pixels=64)
    assert _build(sh)[0]['426:402']['inputs']['mask_blend_pixels'] == 64
    # The node's own ceiling is 64 — past it ComfyUI refuses the whole prompt.
    _set(config, 'face_swap', h3_blend_pixels=200)
    assert _build(sh)[0]['426:402']['inputs']['mask_blend_pixels'] == sh.BLEND_PIXELS_MAX
    _set(config, 'face_swap', h3_blend_pixels='soft please')
    assert _build(sh)[0]['426:402']['inputs']['mask_blend_pixels'] == sh.DEFAULT_BLEND_PIXELS


def test_the_pose_hint_is_APPENDED_to_the_instruction(swap):
    """Never substituted: the base instruction is what the graph was tuned with,
    and the hint is one sentence of per-tile fact on top of it."""
    sh, _mh, _base, _config = swap
    base = _build(sh)[0]['426:170']['inputs']['prompt']
    wf, _ = _build(sh, pose_hint='In this shot the head is in full profile.')
    assert wf['426:170']['inputs']['prompt'].startswith(base.rstrip()[:80])
    assert wf['426:170']['inputs']['prompt'].endswith('full profile.')


def test_the_hint_survives_a_user_rewriting_the_instruction(swap):
    """Rewording the instruction is not the same as no longer wanting the head
    to face the right way."""
    sh, _mh, _base, config = swap
    _set(config, 'minimax_h3', swap_prompt='swap the head please')
    wf, _ = _build(sh, pose_hint='In this shot the face wears a smile.')
    assert wf['426:170']['inputs']['prompt'] == 'swap the head please In this shot the face wears a smile.'


def test_no_hint_leaves_the_instruction_untouched(swap):
    sh, _mh, _base, _config = swap
    assert _build(sh)[0]['426:170']['inputs']['prompt'] == \
        _build(sh, pose_hint=None)[0]['426:170']['inputs']['prompt']
