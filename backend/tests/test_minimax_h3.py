"""MiniMax H3 — the third LOCAL generation engine.

WHAT THESE TESTS ARE FOR
------------------------
Same reason as test_krea_edit.py: the engine was validated by hand against ONE
ComfyUI, and everything that made it work there is a fact about that install —
an `int8_convrot` UNET sitting next to an `fl2va` sibling that does a different
job, a 32B text encoder sitting next to three smaller Qwen3-VL encoders that
would load and produce garbage, four custom nodes of which exactly one is
mandatory. Nobody else has that machine.

So these pin what has to survive a DIFFERENT install: resolution (never the
first plausible file, never the fl2va sibling), preflight (name every gap at
once, never crash, never invent a downloader), the graph wiring (the frame
harvest is what turns a video model into a stills engine — it cannot quietly
lose a node), and the degradation rules (three speed nodes and the RTX upscaler
are optional BY DESIGN; an install with only core + MinimaxH3-Image must still
generate).

NOTHING here renders anything: not one GPU second.
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
    for sub in ('diffusion_models', 'unet', 'loras', 'text_encoders', 'vae',
                'clip_vision'):
        (base / 'models' / sub).mkdir(parents=True, exist_ok=True)
    return base


_VALID_ST = struct.pack('<Q', 2) + b'{}'


def _write(path, data=_VALID_ST):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def h3(monkeypatch, tmp_path):
    """minimax_h3_helper bound to a throwaway ComfyUI tree + throwaway config."""
    config = _fresh_config(monkeypatch, tmp_path)
    base = _comfy_tree(tmp_path)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    import app.services.minimax_h3_helper as mh
    importlib.reload(mh)
    mh._nodes_ok_until = 0.0
    yield mh, base, config
    comfy_model_paths.clear_cache()


def _full_install(base):
    m = base / 'models'
    _write(m / 'diffusion_models' / 'minimax_h3_ref2va_pruned_int8_convrot.safetensors')
    _write(m / 'text_encoders' / 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
    _write(m / 'vae' / 'minimax_h3_video_vae_fp16.safetensors')
    _write(m / 'vae' / 'minimax_h3_audio_vae_fp32.safetensors')
    _write(m / 'clip_vision' / 'CLIP-ViT-H-fp16.safetensors')


def _graph(mh, **over):
    kw = dict(unet='u.safetensors', clip='c.safetensors',
              video_vae='vv.safetensors', audio_vae='av.safetensors',
              clip_vision='cv.safetensors', width=1152, height=864, seed=42)
    kw.update(over)
    return mh.build_workflow('ref.png', 'a prompt', **kw)


def _by_class(graph, class_type):
    return [n for n in graph.values() if n['class_type'] == class_type]


# --- resolution -------------------------------------------------------------

def test_an_empty_install_resolves_to_none_not_a_wrong_file(h3):
    mh, _base, _cfg = h3
    assert mh.resolve_h3_unet() is None
    assert mh.resolve_h3_text_encoder() is None
    assert mh.resolve_h3_video_vae() is None


def test_the_fl2va_sibling_is_never_picked_for_ref2va(h3):
    """Both files carry 'minimax_h3'. fl2va is the first/last-frame model: it
    loads, and then does the wrong job. A token match alone would take it."""
    mh, base, _cfg = h3
    _write(base / 'models' / 'diffusion_models'
           / 'minimax_h3_fl2va_pruned_int8_convrot.safetensors')
    assert mh.resolve_h3_unet() is None

    _write(base / 'models' / 'diffusion_models'
           / 'minimax_h3_ref2va_pruned_int8_convrot.safetensors')
    assert mh.resolve_h3_unet() == 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'


def test_a_renamed_ref2va_quant_still_resolves(h3):
    """int4 / mixed / nvfp4 builds of the same model are all legitimate — the
    resolver keys on ref2va, not on one quantisation."""
    mh, base, _cfg = h3
    _write(base / 'models' / 'diffusion_models'
           / 'minimax_h3_ref2va_pruned_int4_convrot.safetensors')
    assert mh.resolve_h3_unet() == 'minimax_h3_ref2va_pruned_int4_convrot.safetensors'


def test_the_text_encoder_never_matches_a_bare_qwen3vl(h3):
    """A stock install carries qwen3vl_4b / _8b for other engines. They load
    under type='minimax' and produce garbage, so a bare qwen3vl must not match."""
    mh, base, _cfg = h3
    te = base / 'models' / 'text_encoders'
    _write(te / 'qwen3vl_4b_bf16.safetensors')
    _write(te / 'qwen3vl_8b_fp8_scaled.safetensors')
    assert mh.resolve_h3_text_encoder() is None

    _write(te / 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
    assert mh.resolve_h3_text_encoder() == 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'


def test_the_two_vaes_are_told_apart(h3):
    """Video and audio VAE live in the same folder and share a prefix. Swapping
    them is a runtime shape error, not a quality nuance."""
    mh, base, _cfg = h3
    _full_install(base)
    assert mh.resolve_h3_video_vae() == 'minimax_h3_video_vae_fp16.safetensors'
    assert mh.resolve_h3_audio_vae() == 'minimax_h3_audio_vae_fp32.safetensors'


# --- preflight --------------------------------------------------------------

def test_an_empty_install_lists_every_gap_instead_of_raising_on_the_first(h3):
    mh, _base, _cfg = h3
    assert set(mh.h3_missing_assets()) == set(mh.H3_REQUIRED)


def test_preflight_names_assets_and_nodes_and_stays_actionable(h3, monkeypatch):
    mh, _base, _cfg = h3
    monkeypatch.setattr(mh, 'h3_missing_nodes', lambda: ['H3FrameSelect'])
    with pytest.raises(mh.MinimaxH3ModelsMissing) as e:
        mh.preflight()
    assert e.value.missing_nodes == ['H3FrameSelect']
    assert set(e.value.missing) == set(mh.H3_REQUIRED)
    for entry in mh.missing_file_entries(e.value.missing):
        assert entry['path'] and entry['kind'] and entry['source']


def test_a_complete_install_preflights_clean(h3, monkeypatch):
    mh, base, _cfg = h3
    _full_install(base)
    monkeypatch.setattr(mh, 'h3_missing_nodes', lambda: [])
    mh.preflight()


def test_the_node_probe_fails_OPEN_when_comfyui_cannot_be_reached(h3, monkeypatch):
    """Unreachable ComfyUI is not evidence of a missing pack. Blocking there
    would refuse to generate whenever the probe times out."""
    mh, _base, _cfg = h3
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes', lambda: None)
    assert mh.h3_missing_nodes() == []


def test_only_the_frame_selector_is_mandatory(h3, monkeypatch):
    """The three speed nodes and the RTX upscaler are optional BY DESIGN. If a
    missing one ever became fatal, an ordinary install would stop generating."""
    mh, _base, _cfg = h3
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                        lambda: {'MiniMaxH3ReferenceToVideo', 'H3FrameSelect'})
    assert mh.h3_missing_nodes() == []
    assert mh.H3_REQUIRED_NODE_CLASSES == ('MiniMaxH3ReferenceToVideo', 'H3FrameSelect')


# --- graph wiring -----------------------------------------------------------

def test_the_graph_harvests_a_still_from_a_video_packet(h3):
    """The whole engine is this chain. Losing any link silently turns it into
    something else: no frame select = a five-frame packet saved as five tiles."""
    mh, _base, _cfg = h3
    g = _graph(mh)
    for cls in ('MiniMaxH3ReferenceToVideo', 'SamplerCustomAdvanced', 'VAEDecode',
                'H3FrameSelect', 'SaveImage', 'CLIPVisionLoader'):
        assert _by_class(g, cls), f'{cls} missing from the graph'
    sel = _by_class(g, 'H3FrameSelect')[0]
    assert sel['inputs']['select_count'] == 1
    save = _by_class(g, 'SaveImage')[0]
    assert save['inputs']['images'][0] != _find_id(g, 'VAEDecode'), \
        'SaveImage must take the SELECTED frame, never the raw decode'


def _find_id(graph, class_type):
    for nid, node in graph.items():
        if node['class_type'] == class_type:
            return nid
    return None


def test_the_text_encode_does_not_depend_on_the_seed(h3):
    """MEASURED, and the batch ordering rests on it: the seed reaches the
    sampler through RandomNoise only, so a second image from the same catalog
    card re-uses ComfyUI's cached conditioning (38 s instead of 78 s). Wiring
    the seed anywhere upstream of the encode would silently double a batch."""
    mh, _base, _cfg = h3
    g = _graph(mh, seed=12345)
    noise = _by_class(g, 'RandomNoise')[0]
    assert noise['inputs']['noise_seed'] == 12345
    encode = _by_class(g, 'MiniMaxH3ReferenceToVideo')[0]
    assert 'seed' not in encode['inputs']
    assert _find_id(g, 'RandomNoise') not in _upstream_ids(g, _find_id(g, 'MiniMaxH3ReferenceToVideo'))


def _upstream_ids(graph, node_id, seen=None):
    seen = seen if seen is not None else set()
    for value in (graph.get(node_id) or {}).get('inputs', {}).values():
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
            if value[0] not in seen:
                seen.add(value[0])
                _upstream_ids(graph, value[0], seen)
    return seen


def test_the_reference_is_downscaled_before_it_reaches_the_encoder(h3):
    """Reference tokens ride through the 32B VL encoder. A full-size reference
    is paid for on every prompt change."""
    mh, _base, _cfg = h3
    g = _graph(mh, ref_longer_edge=1024)
    resize = _by_class(g, 'ResizeImagesByLongerEdge')[0]
    assert resize['inputs']['longer_edge'] == 1024
    encode = _by_class(g, 'MiniMaxH3ReferenceToVideo')[0]
    assert encode['inputs']['ref_images.ref_image_0'][0] == _find_id(
        g, 'ResizeImagesByLongerEdge')


def test_the_speed_nodes_are_optional_and_the_model_chain_survives_without_them(h3):
    """With the pack absent the graph must still reach the sampler — same image,
    slower. This is the difference between 'ships' and 'works on one machine'."""
    mh, _base, _cfg = h3
    on = _graph(mh, use_speed_nodes=True)
    off = _graph(mh, use_speed_nodes=False)
    assert _by_class(on, 'SpectrumApplyMiniMaxH3') and _by_class(on, 'PathchSageAttentionKJ')
    assert not _by_class(off, 'SpectrumApplyMiniMaxH3')
    assert not _by_class(off, 'PathchSageAttentionKJ')
    for g in (on, off):
        guider = _by_class(g, 'BasicGuider')[0]
        assert _find_id(g, 'UNETLoader') in _upstream_ids(g, _find_id(g, 'BasicGuider'))
        assert guider['inputs']['model']


def test_spectrum_never_ships_with_debug_on(h3):
    mh, _base, _cfg = h3
    spectrum = _by_class(_graph(mh, use_speed_nodes=True), 'SpectrumApplyMiniMaxH3')[0]
    assert spectrum['inputs']['debug'] is False


def test_the_rtx_upscale_can_be_dropped_without_breaking_the_save(h3):
    """Default ON (maintainer's call), but a non-RTX card must lose the 2x, not
    the engine."""
    mh, _base, _cfg = h3
    with_sr = _graph(mh, use_rtx_upscale=True)
    without = _graph(mh, use_rtx_upscale=False)
    assert _by_class(with_sr, 'RTXVideoSuperResolution')
    assert not _by_class(without, 'RTXVideoSuperResolution')
    saved_from = _by_class(without, 'SaveImage')[0]['inputs']['images'][0]
    assert saved_from == _find_id(without, 'H3FrameSelect')


def test_frame_selection_scores_on_identity_by_default(h3):
    """The debugged graph had weight_reference at 0 by accident. For a dataset
    the point is the frame that looks most like the reference."""
    mh, _base, _cfg = h3
    sel = _by_class(_graph(mh), 'H3FrameSelect')[0]
    assert sel['inputs']['weight_reference'] > 0
    assert sel['inputs']['clip_vision'], 'reference scoring needs the CLIP-Vision tower'


def test_the_text_encoder_is_loaded_as_minimax(h3):
    mh, _base, _cfg = h3
    clip = _by_class(_graph(mh), 'CLIPLoader')[0]
    assert clip['inputs']['type'] == 'minimax'


def test_both_vaes_reach_the_encode_and_the_video_one_decodes(h3):
    """The node REQUIRES an audio VAE even for a still. Passing the video VAE
    twice loads, then fails at sample time."""
    mh, _base, _cfg = h3
    g = _graph(mh, video_vae='video.safetensors', audio_vae='audio.safetensors')
    encode = _by_class(g, 'MiniMaxH3ReferenceToVideo')[0]
    vae_id = encode['inputs']['vae'][0]
    audio_id = encode['inputs']['audio_vae'][0]
    assert vae_id != audio_id
    assert g[vae_id]['inputs']['vae_name'] == 'video.safetensors'
    assert g[audio_id]['inputs']['vae_name'] == 'audio.safetensors'
    decode = _by_class(g, 'VAEDecode')[0]
    assert decode['inputs']['vae'][0] == vae_id


# --- geometry and packet length ---------------------------------------------

@pytest.mark.parametrize('w,h', [(4000, 3000), (1024, 1024), (832, 1216), (5000, 1000)])
def test_output_size_is_a_multiple_of_32_and_within_the_budget(h3, w, h):
    """width/height are plain INTs with step 32 on the node, so the app owns
    the geometry and ResolutionSelector is not needed."""
    mh, _base, _cfg = h3
    ow, oh = mh.fit_output_size(w, h)
    assert ow % 32 == 0 and oh % 32 == 0
    assert ow * oh <= mh.MAX_OUTPUT_MP * 1_000_000 * 1.02
    assert abs((ow / oh) / (w / h) - 1) < 0.06


@pytest.mark.parametrize(('requested_aspect', 'expected_ratio'), [
    ('1:1', 1.0),
    ('3:4', 3 / 4),
    ('16:9', 16 / 9),
])
def test_the_catalog_card_decides_the_shape_not_the_reference(
        h3, requested_aspect, expected_ratio):
    """Both callers resolve the card's ratio and pass it; H3 used to drop it and
    copy the REFERENCE's aspect, so a square reference answered a full-body card
    with a square. The pixel budget is unchanged — only the shape moves."""
    mh, _base, _cfg = h3
    ow, oh = mh.fit_output_size(1024, 1024, requested_aspect=requested_aspect)
    assert ow % 32 == 0 and oh % 32 == 0
    assert abs((ow / oh) / expected_ratio - 1) < 0.04


def test_a_card_ratio_never_invents_pixels_the_reference_does_not_have(h3):
    """A packet is sampled per image, so every extra pixel is paid once per
    frame. Reshaping must not become upscaling."""
    mh, _base, _cfg = h3
    ow, oh = mh.fit_output_size(640, 640, requested_aspect='3:4')
    assert ow * oh <= 640 * 640
    assert abs((ow / oh) / (3 / 4) - 1) < 0.04
    # ...and the megapixel budget still wins over a large reference.
    ow, oh = mh.fit_output_size(4000, 4000, requested_aspect='3:4')
    assert ow * oh <= mh.MAX_OUTPUT_MP * 1_000_000


def test_the_enqueue_actually_forwards_the_card_ratio(h3, tmp_path, monkeypatch):
    """The bug this file now guards was NOT in the geometry function: `enqueue`
    accepted `aspect_ratio` from both callers and never read it. Every other test
    here stubs the enqueue, so nothing noticed for a whole engine's lifetime."""
    from PIL import Image
    mh, _base, _cfg = h3
    source = tmp_path / 'square_reference.png'
    Image.new('RGB', (1024, 1024), (7, 7, 7)).save(source)
    staged = tmp_path / 'staged.png'
    Image.new('RGB', (1024, 1024), (7, 7, 7)).save(staged)

    seen = {}
    monkeypatch.setattr(mh, 'preflight', lambda *a, **k: None)
    monkeypatch.setattr(mh, 'resolve_h3_unet', lambda *a, **k: 'unet.safetensors')
    monkeypatch.setattr(mh, 'resolve_h3_text_encoder', lambda: 'clip.safetensors')
    monkeypatch.setattr(mh, 'resolve_h3_video_vae', lambda: 'video.safetensors')
    monkeypatch.setattr(mh, 'resolve_h3_audio_vae', lambda: 'audio.safetensors')
    monkeypatch.setattr(mh, 'resolve_h3_clip_vision', lambda: 'vision.safetensors')
    monkeypatch.setattr(mh, '_comfy_input_dir', lambda: str(tmp_path))
    monkeypatch.setattr(mh.comfy_fs, 'ensure_input_usable', lambda d: str(tmp_path))
    monkeypatch.setattr(mh.comfy_fs, 'stage_input_image',
                        lambda src, name, d: str(staged))
    monkeypatch.setattr(mh, 'available_optional_nodes',
                        lambda: {'speed': False, 'upscale': False})
    monkeypatch.setattr(mh, 'build_workflow',
                        lambda *a, **kw: (seen.update(kw), {})[1])
    monkeypatch.setattr(mh.queue_manager, 'add_job', lambda **kw: None)

    mh.enqueue_minimax_h3(user_id='local', source_filename='square_reference.png',
                          source_path=str(source), edit_prompt='p',
                          aspect_ratio='3:4')

    assert abs((seen['width'] / seen['height']) / (3 / 4) - 1) < 0.04, (
        'a square reference must not decide the shape of a portrait card')


def test_an_unusable_card_ratio_keeps_the_reference_geometry(h3):
    """A custom catalog entry with a typo must not decide the canvas."""
    mh, _base, _cfg = h3
    plain = mh.fit_output_size(1536, 2048)
    for bad in ('not-a-ratio', '0:1', '1:0', '', None, '99:1', 5):
        assert mh.fit_output_size(1536, 2048, requested_aspect=bad) == plain, bad


def test_packet_length_is_clamped_to_what_the_node_accepts(h3):
    """`length` is min 5, step 17 on the node. An out-of-step value is a
    validation error at queue time, i.e. a whole batch of failed tiles."""
    mh, _base, _cfg = h3
    assert mh.clamp_length(1) == 5
    assert mh.clamp_length(5) == 5
    assert mh.clamp_length(22) == 22
    assert mh.clamp_length(20) in (5, 22)
    assert (mh.clamp_length(20) - 5) % 17 == 0
    assert (mh.clamp_length(999) - 5) % 17 == 0
