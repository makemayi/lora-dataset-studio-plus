"""The NEW MiniMax H3 head swap (2026-08-14 redesign) — what has to survive a
machine that is not the maintainer's.

The graph it came from was exported from ONE install: an INT8 H3 build, an INT8
Klein build, six node packs and one specific Ollama tag. What must hold
elsewhere:

* both optional stages are OFF by default, and switching one off REMOVES its
  whole branch — including the custom-node classes only it needed, or an
  install without OllamaAPI would be told to install it for a stage it is not
  running;
* the prompt is written into the TEXT node, because on this graph the H3 node's
  own `prompt` input is a link and anything typed into it is discarded;
* Klein is not optional here (it is how the head is erased), so its absence is
  reported before anything is queued;
* the hybrid loader gets BOTH models, with Fl2VA as the base — the one file
  every other H3 path in this app refuses to pick;
* Ollama runs on the app's configured vision model, never the graph's tag;
* the pose hint is not sent when Ollama is analysing the same picture.

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
    _write(m / 'diffusion_models' / 'minimax_h3_fl2va_pruned_int8_convrot.safetensors')
    _write(m / 'text_encoders' / 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
    _write(m / 'vae' / 'minimax_h3_video_vae_fp16.safetensors')
    _write(m / 'vae' / 'minimax_h3_audio_vae_fp32.safetensors')
    _write(m / 'clip_vision' / 'CLIP-ViT-H-fp16.safetensors')


def _klein_install(base):
    m = base / 'models'
    _write(m / 'diffusion_models' / 'klein' / 'flux-2-klein-9b-int8.safetensors')
    _write(m / 'text_encoders' / 'qwen_3_8b_fp8mixed.safetensors')
    _write(m / 'vae' / 'flux2-vae.safetensors')


@pytest.fixture
def swap(monkeypatch, tmp_path):
    """minimax_h3_swap_new_helper bound to a throwaway ComfyUI tree + config,
    with a complete H3 AND Klein install (this graph needs both) and a fail-open
    /object_info probe — each test then takes away what it is about."""
    config = _fresh_config(monkeypatch, tmp_path)
    base = tmp_path / 'Comfy'
    for sub in ('diffusion_models', 'unet', 'loras', 'text_encoders', 'vae',
                'clip_vision'):
        (base / 'models' / sub).mkdir(parents=True, exist_ok=True)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    _h3_install(base)
    _klein_install(base)
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    import app.services.minimax_h3_helper as mh
    import app.services.minimax_h3_swap_helper as old
    import app.services.minimax_h3_swap_new_helper as sh
    importlib.reload(mh)
    importlib.reload(old)
    importlib.reload(sh)
    mh._nodes_ok_until = 0.0
    import app.utils.comfyui as comfyui
    monkeypatch.setattr(comfyui, 'fetch_object_info_classes', lambda: None)
    import flask
    ctx = flask.Flask(__name__).app_context()
    ctx.push()
    try:
        yield sh, mh, base, config
    finally:
        ctx.pop()
        comfy_model_paths.clear_cache()


def _set(config, section, **values):
    config.save_config({section: values})


def _build(sh, **kw):
    kw.setdefault('filename_prefix', 'local_H3SwapNew_abcd1234')
    return sh.build_swap_workflow('t.png', 'r.png', **kw)


def _classes(wf):
    return {n['class_type'] for n in wf.values()}


# --- the two optional stages -------------------------------------------------

def test_both_stages_off_by_default(swap):
    sh, _mh, _base, _config = swap
    assert sh.enabled_stages() == {'mask_overlay': False, 'ollama': False}


def test_stages_off_removes_their_whole_branch(swap):
    """Not just the node: the Ollama call, the text concatenation and every
    custom class only they needed must leave with them, or the preflight names a
    pack for a stage this job does not contain."""
    sh, _mh, _base, _config = swap
    wf, kept = _build(sh)
    assert kept == []
    assert not (_classes(wf) & {'OllamaAPI', 'Text Concatenate',
                                'AILab_MaskOverlay'})
    for tail in ('988', '991', '983:1002'):
        assert tail not in wf
    # ...and the graph still runs end to end.
    assert wf['165']['inputs']['images']
    assert 'MiniMaxH3ReferenceToVideo' in _classes(wf)


def test_a_stage_that_is_off_hands_its_consumers_the_right_fallback(swap):
    """The fallbacks are what ComfyUI's own bypass does to those nodes. Get one
    wrong and the job still renders — from the wrong picture, or with an empty
    prompt."""
    sh, _mh, _base, _config = swap
    wf, _ = _build(sh)
    # H3's second reference is the head-removed Klein output, not the overlay.
    assert wf['170']['inputs']['ref_images.ref_image_1'] == ['983:973', 0]
    # ...and the prompt comes straight from the text node.
    assert wf['170']['inputs']['prompt'] == ['990', 0]


def test_mask_overlay_on_sits_between_klein_and_h3(swap):
    sh, _mh, _base, _config = swap
    wf, kept = _build(sh, stages={'mask_overlay': True, 'ollama': False})
    assert kept == ['mask_overlay']
    assert wf['983:1002']['inputs']['image'] == ['983:973', 0]
    assert wf['170']['inputs']['ref_images.ref_image_1'] == ['983:1002', 0]
    assert wf['983:1002']['inputs']['mask_opacity'] == 1.0


def test_mask_opacity_only_applies_while_the_overlay_stage_is_on(swap):
    sh, _mh, _base, config = swap
    _set(config, 'face_swap', h3_mask_opacity=0.5)
    wf, _ = _build(sh, stages={'mask_overlay': True, 'ollama': False})
    assert wf['983:1002']['inputs']['mask_opacity'] == 0.5
    # With the stage off the node is not in the job at all — the setting is not
    # ignored quietly, there is nothing left to ignore it.
    wf, _ = _build(sh)
    assert '983:1002' not in wf


def test_ollama_on_uses_the_app_model_not_the_graph_tag(swap, monkeypatch):
    """A graph naming one Ollama tag fails on every machine that lacks it."""
    sh, _mh, _base, config = swap
    _set(config, 'ollama', vision_model='my/own-vlm:latest')
    monkeypatch.delenv('VISION_OLLAMA_MODEL', raising=False)
    wf, kept = _build(sh, stages={'mask_overlay': False, 'ollama': True})
    assert kept == ['ollama']
    assert wf['988']['inputs']['ollama_model'] == 'my/own-vlm:latest'
    assert wf['988']['inputs']['seed'] != 0
    assert wf['170']['inputs']['prompt'] == ['991', 0]
    assert wf['991']['inputs']['text_b'] == ['988', 0]


# --- the prompt --------------------------------------------------------------

def test_the_prompt_override_is_written_into_the_text_node(swap):
    """The H3 node's `prompt` is a LINK here. Writing the override there would
    be silently discarded at execution time — the whole reason this helper is
    not a copy of the old one with different ids."""
    sh, _mh, _base, config = swap
    _set(config, 'minimax_h3', swap_prompt='  put this head on that body  ')
    wf, _ = _build(sh)
    assert wf['990']['inputs']['text'] == 'put this head on that body'
    assert isinstance(wf['170']['inputs']['prompt'], list)


def test_the_pose_hint_is_appended_to_the_text_node(swap):
    sh, _mh, _base, _config = swap
    wf, _ = _build(sh, pose_hint='The head is turned three-quarters left.')
    assert wf['990']['inputs']['text'].endswith(
        'The head is turned three-quarters left.')


def test_the_pose_hint_stands_down_when_ollama_is_looking(swap):
    """Both describe how this head sits; only one of them is looking at the
    actual picture. Sending both means sending two that can disagree."""
    sh, _mh, _base, _config = swap
    wf, kept = _build(sh, stages={'mask_overlay': False, 'ollama': True},
                      pose_hint='The head is turned three-quarters left.')
    assert kept == ['ollama']
    assert 'three-quarters left' not in wf['990']['inputs']['text']


# --- models ------------------------------------------------------------------

def test_the_hybrid_loader_gets_fl2va_as_the_base_and_ref2va_over_it(swap):
    sh, _mh, _base, _config = swap
    wf, _ = _build(sh)
    loader = wf['925:427']['inputs']
    assert 'fl2va' in loader['base_model'].lower()
    assert 'ref2va' in loader['overlay_model'].lower()


def test_a_missing_fl2va_build_is_named_before_anything_is_queued(swap):
    sh, mh, base, _config = swap
    (base / 'models' / 'diffusion_models'
     / 'minimax_h3_fl2va_pruned_int8_convrot.safetensors').unlink()
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    with pytest.raises(mh.MinimaxH3ModelsMissing) as err:
        _build(sh)
    # ...and the banner has a file to name, or the user gets an error with
    # nothing to act on: this key is deliberately outside H3_REQUIRED.
    entries = mh.missing_file_entries(err.value.missing)
    assert entries and 'fl2va' in entries[0]['path']


def test_the_generation_lane_still_refuses_fl2va(swap):
    """The exclusion that keeps Fl2VA out of every OTHER H3 path must survive
    this graph asking for it by name — it loads without complaint and then does
    a different job."""
    _sh, mh, base, _config = swap
    (base / 'models' / 'diffusion_models'
     / 'minimax_h3_ref2va_pruned_int8_convrot.safetensors').unlink()
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    assert mh.resolve_h3_unet() is None
    assert mh.resolve_h3_fl2va() is not None


def test_klein_is_required_here_rather_than_optional(swap):
    """On the old graph Klein was one of three stages you could switch on. Here
    the Klein pass IS how the head is removed, so a missing Klein asset has to
    be named before anything is queued."""
    sh, _mh, base, _config = swap
    for rel in ('diffusion_models/klein/flux-2-klein-9b-int8.safetensors',
                'text_encoders/qwen_3_8b_fp8mixed.safetensors',
                'vae/flux2-vae.safetensors'):
        (base / 'models' / rel).unlink()
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    from app.services.klein_edit_helper import KleinModelsMissing
    with pytest.raises(KleinModelsMissing):
        _build(sh)


def test_the_klein_sampler_reseeds_every_job(swap):
    """Otherwise every tile of a batch erases its head the identical way."""
    sh, _mh, _base, _config = swap
    wf, _ = _build(sh)
    assert wf['983:969']['inputs']['seed'] != 0
    assert wf['131']['inputs']['noise_seed'] != 0


def test_the_save_prefix_is_the_one_the_caller_asked_for(swap):
    """A shared prefix makes ComfyUI's counter re-issue one filename, and every
    tile of a batch ends up displaying the same image."""
    sh, _mh, _base, _config = swap
    wf, _ = _build(sh, filename_prefix='local_H3SwapNew_deadbeef')
    assert wf['165']['inputs']['filename_prefix'] == 'local_H3SwapNew_deadbeef'


# --- app-side mask -----------------------------------------------------------

def test_an_app_mask_replaces_the_in_graph_segmenter(swap):
    sh, _mh, _base, _config = swap
    wf, _ = _build(sh, mask_image='mask.png')
    assert '957' not in wf, 'ClothesSegment must leave with its own dependency'
    assert wf['983:971']['inputs']['mask'] == ['app_mask_to_mask', 0]
    # The mask is resized by the SAME node value as the target, never app-side.
    assert (wf['app_mask_resize']['inputs']['longer_edge']
            == wf['928']['inputs']['longer_edge'])


# --- engine selection --------------------------------------------------------

def test_the_new_graph_owns_the_id_that_was_already_in_config(swap):
    """Anyone who had picked H3 gets the redesign without touching a setting;
    the old graph moves to its own id."""
    _sh, _mh, _base, config = swap
    from app.services import dataset_generation_service as dgs
    importlib.reload(dgs)
    assert set(dgs.FACE_SWAP_ENGINES) == {'klein', 'minimax_h3', 'minimax_h3_old'}
    _set(config, 'face_swap', engine='minimax_h3_old')
    assert dgs.resolve_face_swap_engine() == 'minimax_h3_old'
    _set(config, 'face_swap', engine='minimax_h3')
    assert dgs.resolve_face_swap_engine() == 'minimax_h3'


def test_all_three_engines_share_one_enqueue_contract(swap):
    """dataset_generation_service picks between them by config alone, so a
    signature drift would be a runtime TypeError on a click."""
    import inspect
    sh, _mh, _base, _config = swap
    from app.services import face_swap_helper, minimax_h3_swap_helper
    expected = ['user_id', 'target_path', 'ref_path', 'extra_metadata']
    for fn in (face_swap_helper.enqueue_face_swap,
               minimax_h3_swap_helper.enqueue_h3_swap,
               sh.enqueue_h3_swap_new):
        assert list(inspect.signature(fn).parameters) == expected
