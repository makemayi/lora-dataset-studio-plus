"""Every H3 swap records what it was actually built from.

WHY THIS EXISTS
---------------
Both swap engines subtract optional stages and optional accelerators before
queueing, and until now `kept` went to the LOG and nowhere else. Two images from
the same dataset, the same prompt and the same seed could therefore differ for a
reason nothing recorded — which is exactly the failure mode of a tuning pass:
you compare two runs, one is better, and the file cannot tell you why.

Three variables in particular:

  * the optional stages (hair removal / LaMa / face detail on the old engine,
    the Ollama head analysis on the new one);
  * the accelerators, which are config AND /object_info — an install without the
    NVIDIA pack silently drops them;
  * on the OLD engine only, `use_rtx_upscale` also sits on the IDENTITY INPUT
    (`NODE_UPSCALE_REF`), so turning it off hands H3 a different reference. The
    new engine feeds the reference through a plain Resize and does not have this.

And on the new engine the head analysis is GENERATED TEXT written into the
prompt — the largest run-to-run variable in that graph, previously logged only,
truncated to 300 characters.

NOTHING here renders anything.
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


def _install(base):
    m = base / 'models'
    _write(m / 'diffusion_models' / 'minimax_h3_ref2va_pruned_int8_convrot.safetensors')
    _write(m / 'diffusion_models' / 'minimax_h3_fl2va_pruned_int8_convrot.safetensors')
    _write(m / 'text_encoders' / 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
    _write(m / 'vae' / 'minimax_h3_video_vae_fp16.safetensors')
    _write(m / 'vae' / 'minimax_h3_audio_vae_fp32.safetensors')
    _write(m / 'clip_vision' / 'CLIP-ViT-H-fp16.safetensors')
    _write(m / 'diffusion_models' / 'klein' / 'flux-2-klein-9b-int8.safetensors')
    _write(m / 'text_encoders' / 'qwen_3_8b_fp8mixed.safetensors')
    _write(m / 'vae' / 'flux2-vae.safetensors')


@pytest.fixture
def swap(monkeypatch, tmp_path):
    """Both swap helpers on a throwaway ComfyUI tree, with a fail-open
    /object_info probe (so a test that turns an accelerator OFF is testing the
    setting, not a missing pack)."""
    config = _fresh_config(monkeypatch, tmp_path)
    base = tmp_path / 'Comfy'
    for sub in ('diffusion_models', 'unet', 'loras', 'text_encoders', 'vae', 'clip_vision'):
        (base / 'models' / sub).mkdir(parents=True, exist_ok=True)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    _install(base)
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    import app.services.minimax_h3_helper as mh
    import app.services.minimax_h3_swap_helper as old
    import app.services.minimax_h3_swap_new_helper as new
    importlib.reload(mh)
    importlib.reload(old)
    importlib.reload(new)
    mh._nodes_ok_until = 0.0
    import app.utils.comfyui as comfyui
    monkeypatch.setattr(comfyui, 'fetch_object_info_classes', lambda: None)
    import flask
    ctx = flask.Flask(__name__).app_context()
    ctx.push()
    try:
        yield old, new, mh, config
    finally:
        ctx.pop()
        comfy_model_paths.clear_cache()


def _set(config, section, **values):
    config.save_config({section: values})


# --- the builder reports what it wired --------------------------------------

def test_both_engines_return_the_accelerators_they_actually_wired(swap):
    old, new, _mh, config = swap
    _set(config, 'minimax_h3', use_rtx_upscale=True, use_speed_nodes=True)
    _, _, accel_old = old.build_swap_workflow(
        't.png', 'r.png', filename_prefix='local_H3Swap_abcd1234')
    _, _, accel_new = new.build_swap_workflow(
        't.png', 'r.png', filename_prefix='local_H3SwapNew_abcd1234', mask_image='m.png')
    assert accel_old == {'speed': True, 'upscale': True}
    assert accel_new == {'speed': True, 'upscale': True}


def test_the_report_follows_the_setting_off(swap):
    old, new, _mh, config = swap
    _set(config, 'minimax_h3', use_rtx_upscale=False, use_speed_nodes=False)
    _, _, accel_old = old.build_swap_workflow(
        't.png', 'r.png', filename_prefix='local_H3Swap_abcd1234')
    _, _, accel_new = new.build_swap_workflow(
        't.png', 'r.png', filename_prefix='local_H3SwapNew_abcd1234', mask_image='m.png')
    assert accel_old == {'speed': False, 'upscale': False}
    assert accel_new == {'speed': False, 'upscale': False}


def test_a_missing_node_pack_is_reported_as_off_not_as_on(swap, monkeypatch):
    """Config asks, /object_info decides. Recording the CONFIG would describe a
    graph that was never built — on an install without the NVIDIA pack the
    upscalers are dropped however the setting reads."""
    old, new, mh, config = swap
    _set(config, 'minimax_h3', use_rtx_upscale=True, use_speed_nodes=True)
    monkeypatch.setattr(mh, 'available_optional_nodes',
                        lambda: {'speed': False, 'upscale': False})
    _, _, accel_old = old.build_swap_workflow(
        't.png', 'r.png', filename_prefix='local_H3Swap_abcd1234')
    assert accel_old == {'speed': False, 'upscale': False}


def test_turning_the_upscaler_off_changes_the_identity_input_on_the_OLD_engine(swap):
    """The reason the report exists at all. One setting drives two upscalers on
    the old graph, and one of them sits on the reference photo — so the same
    dataset and seed feed H3 a DIFFERENT identity depending on a switch that
    reads as an output-quality dial. The new graph resizes the reference and is
    unaffected; that asymmetry is pinned here so a future edit cannot quietly
    introduce it on the new engine too."""
    old, new, _mh, config = swap

    _set(config, 'minimax_h3', use_rtx_upscale=True, use_speed_nodes=False)
    wf_on, _, _ = old.build_swap_workflow(
        't.png', 'r.png', filename_prefix='local_H3Swap_abcd1234')
    _set(config, 'minimax_h3', use_rtx_upscale=False, use_speed_nodes=False)
    wf_off, _, _ = old.build_swap_workflow(
        't.png', 'r.png', filename_prefix='local_H3Swap_abcd1234')

    # The reference upscaler feeds TWO consumers, and the second is worse than the
    # first: the resize that becomes H3's identity input, AND the frame selector's
    # `reference` — the image every frame of the packet is scored against. So the
    # switch changes what H3 is shown AND which frame wins.
    def _source(wf, node_id, key):
        return wf[node_id]['inputs'][key]

    ref_resize, frame_select = '426:311', old.NODE_FRAME_SELECT
    assert _source(wf_on, ref_resize, 'images') == [old.NODE_UPSCALE_REF, 0]
    assert _source(wf_off, ref_resize, 'images') != [old.NODE_UPSCALE_REF, 0]
    assert _source(wf_on, frame_select, 'reference') == [old.NODE_UPSCALE_REF, 0]
    assert _source(wf_off, frame_select, 'reference') != [old.NODE_UPSCALE_REF, 0]

    # The new engine has no reference upscaler at all — only the output one.
    assert new._UPSCALE_NODES == ((new.NODE_UPSCALE_OUT, 'images'),)
    assert old.NODE_UPSCALE_REF in dict(old._UPSCALE_NODES)


# --- what reaches the job ---------------------------------------------------

def _enqueue(monkeypatch, module, owner, helper_name, tmp_path, analysis=None):
    """Drive one enqueue_* to the queue and hand back the metadata it wrote.

    Everything outside the module under test is stubbed: no ComfyUI, no queue, no
    Ollama, no mask model. `owner` is where the SHARED staging helpers live — the
    new engine borrows `_comfy_input_dir` / `mask_prompt` from the old one, so
    patching them on the new module would patch nothing.
    """
    captured = {}

    src = tmp_path / 'src.png'
    src.write_bytes(b'x')

    from app.utils import comfy_fs
    monkeypatch.setattr(owner, '_comfy_input_dir', lambda: str(tmp_path))
    monkeypatch.setattr(comfy_fs, 'ensure_input_usable', lambda d: str(tmp_path))
    monkeypatch.setattr(comfy_fs, 'stage_input_image',
                        lambda src_path, name, d: str(tmp_path / name))
    monkeypatch.setattr(owner, 'mask_source', lambda: 'graph', raising=False)
    from app.services import auto_mask
    monkeypatch.setattr(auto_mask, 'mask_for', lambda p, prompt: str(src))
    if analysis is not None:
        monkeypatch.setattr(module, 'analyse_head', lambda p: analysis)

    from app.job_queue import queue_manager
    monkeypatch.setattr(queue_manager, 'add_job',
                        lambda **kw: captured.update(kw) or 'job')
    getattr(module, helper_name)('u1', str(src), str(src))
    return captured['metadata']


def test_the_old_engine_records_its_build(swap, monkeypatch, tmp_path):
    old, _new, _mh, config = swap
    _set(config, 'minimax_h3', use_rtx_upscale=True, use_speed_nodes=False)
    meta = _enqueue(monkeypatch, old, old, 'enqueue_h3_swap', tmp_path)
    build = meta['swap_build']
    assert build['engine'] == 'h3_old'
    assert build['rtx_upscale'] is True
    assert build['speed_nodes'] is False
    assert isinstance(build['stages'], list)


def test_the_new_engine_records_its_build(swap, monkeypatch, tmp_path):
    _old, new, _mh, config = swap
    _set(config, 'minimax_h3', use_rtx_upscale=False, use_speed_nodes=True)
    meta = _enqueue(monkeypatch, new, _old, 'enqueue_h3_swap_new', tmp_path)
    build = meta['swap_build']
    assert build['engine'] == 'h3_new'
    assert build['rtx_upscale'] is False
    assert build['speed_nodes'] is True


def test_the_head_analysis_is_recorded_in_full_not_as_a_log_preview(swap, monkeypatch, tmp_path):
    """It is generated text written INTO the prompt, so it is the run's biggest
    variable. The log truncated it to 300 characters; the record must not."""
    _old, new, _mh, config = swap
    _set(config, 'face_swap', h3_new_stages={'ollama': True})
    long_analysis = 'the head is turned three-quarters left. ' * 20
    meta = _enqueue(monkeypatch, new, _old, 'enqueue_h3_swap_new', tmp_path,
                    analysis=long_analysis)
    build = meta['swap_build']
    assert build['head_analysis'] == long_analysis[:4000]
    assert len(build['head_analysis']) > 300
    assert build['head_analysis_model']


def test_no_head_analysis_key_when_the_stage_is_off(swap, monkeypatch, tmp_path):
    """Absent, not empty-string: "the stage did not run" and "it ran and said
    nothing" are different runs and must not read the same."""
    _old, new, _mh, config = swap
    _set(config, 'face_swap', h3_new_stages={'ollama': False})
    meta = _enqueue(monkeypatch, new, _old, 'enqueue_h3_swap_new', tmp_path)
    assert 'head_analysis' not in meta['swap_build']
