"""The list behind the model-file pickers, and the refusal that protects it.

Two questions, and the second is the expensive one:

  1. does the picker offer exactly what the RESOLVER can load? A second way of
     listing model files is how a dropdown starts offering a file the graph
     refuses (or hiding one it would happily load, the day an install has an
     extra_model_paths.yaml);
  2. what happens when a pinned file is NOT there? It used to be a log line and
     a silent election — the failure mode that ran a whole training on a
     third-party finetune nobody chose.
"""
import os

import pytest

from app.services import comfy_model_picker as picker


def _write(path, size=2048):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\0' * size)


@pytest.fixture
def comfy(app, tmp_path, monkeypatch):
    """A ComfyUI tree with one file in each folder the pickers scan."""
    from app import config as cfg
    base = tmp_path / 'ComfyUI'
    (base / 'models').mkdir(parents=True)
    with app.app_context():
        cfg.save_config({'comfyui': {'base_dir': str(base)}})
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    picker.clear_cache()
    yield base
    comfy_model_paths.clear_cache()
    picker.clear_cache()


def test_each_slot_lists_its_own_folder_and_says_where_that_is(app, comfy):
    _write(comfy / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-fp8.safetensors')
    _write(comfy / 'models' / 'text_encoders' / 'qwen_3_8b_fp8mixed.safetensors')
    _write(comfy / 'models' / 'vae' / 'flux2-vae.safetensors')
    _write(comfy / 'models' / 'loras' / 'klein' / 'consistency.safetensors')
    with app.app_context():
        unet, hint = picker.list_slot_files('klein_unet')
        assert unet == [os.path.join('klein', 'flux-2-klein-9b-fp8.safetensors')], (
            'the name must be the loader-relative string the field already stored — '
            'anything else would need an alias for every install that typed one')
        assert 'unet' in hint, 'the empty state has to be able to say WHERE to put a file'
        assert picker.list_slot_files('klein_vae')[0] == ['flux2-vae.safetensors']
        assert picker.list_slot_files('klein_text_encoder')[0] == [
            'qwen_3_8b_fp8mixed.safetensors']
        assert picker.list_slot_files('klein_consistency_lora')[0] == [
            os.path.join('klein', 'consistency.safetensors')]


def test_the_krea_base_list_is_the_RESOLVER_s_candidates_not_the_whole_folder(app, comfy):
    """BigLove carries 'krea' in its name and renders pure noise under the
    identity LoRA, so the resolver excludes it. A picker built on its own walk of
    diffusion_models would offer it — a choice that silently does nothing."""
    d = comfy / 'models' / 'diffusion_models' / 'Krea'
    _write(d / 'krea2_turbo_fp8_scaled.safetensors')
    _write(d / 'BigLoveKreaEdit1_fp8mixed.safetensors')
    _write(comfy / 'models' / 'unet' / 'some_other_model.safetensors')
    with app.app_context():
        files, _hint = picker.list_slot_files('krea_base_model')
    assert any('krea2_turbo' in f for f in files)
    assert not any('BigLove' in f for f in files), (
        'the picker offered a checkpoint the resolver refuses to elect')
    assert not any('some_other_model' in f for f in files), (
        'a non-krea folder file is not a Krea base candidate')


def test_an_unknown_slot_and_an_unconfigured_comfyui_degrade_to_free_text(app, tmp_path):
    """Never an error: the field falls back to plain text rather than blocking
    the whole Settings panel on an install that has no ComfyUI at all."""
    with app.app_context():
        assert picker.list_slot_files('not_a_slot') == ([], '')
        files, hint = picker.list_slot_files('klein_vae')
        assert files == []
        assert hint


def test_the_endpoint_answers_the_shape_the_picker_reads(client, app, comfy):
    _write(comfy / 'models' / 'vae' / 'flux2-vae.safetensors')
    with app.app_context():
        picker.clear_cache()
    r = client.get('/api/comfy/model-files?slot=klein_vae')
    assert r.status_code == 200
    body = r.get_json()
    assert body['files'] == ['flux2-vae.safetensors']
    assert body['folder']
    # A junk slot from a stale client is a 200 with nothing, not a 500.
    assert client.get('/api/comfy/model-files?slot=../../etc').status_code == 200


def test_a_klein_pin_that_is_not_on_disk_keeps_the_engine_dark(app, comfy):
    """The Settings badge already existed; the RUN still went ahead on whatever
    auto-detection found. Now the engine refuses and names the file."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    _write(comfy / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-fp8.safetensors')
    with app.app_context():
        cfg.save_config({'klein': {'unet': 'klein/a-model-i-deleted.safetensors'}})
        gaps = keh.klein_pin_gaps()
        assert [g['slot'] for g in gaps] == ['unet']
        assert gaps[0]['configured'] == 'klein/a-model-i-deleted.safetensors'
        assert keh.klein_engine_ready(True) is False
        # Clearing the field is the explicit way back to auto-detection.
        cfg.save_config({'klein': {'unet': ''}})
        assert keh.klein_pin_gaps() == []


def test_the_shipped_consistency_lora_name_is_not_a_user_pin(app, comfy):
    """klein.consistency_lora ships a NON-EMPTY default, so 'set' is not
    'chosen'. Treating it as a pin would brick every install that never
    downloaded that optional LoRA — the whole point of its documented 'skipped'
    behaviour."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        shipped = (cfg.DEFAULTS.get('klein') or {}).get('consistency_lora')
        assert shipped, 'this test is meaningless if the default becomes blank'
        cfg.save_config({'klein': {'consistency_lora': shipped}})
        assert keh.klein_pin_gaps() == []
        cfg.save_config({'klein': {'consistency_lora': 'klein/mine.safetensors'}})
        assert [g['slot'] for g in keh.klein_pin_gaps()] == ['consistency_lora']


def test_the_klein_unet_list_is_the_RESOLVER_s_candidates_too(app, comfy):
    """The Klein UNET slot had the same defect the Krea base was protected from,
    and one consequence more.

    A picker built on its own walk of diffusion_models offers every file in
    there, so it can hand over one the resolver will not build — a choice that
    silently does nothing. But Klein ALSO has a second, older picker: the
    per-dataset one in KleinModelSetting, whose list comes from the capabilities
    probe and has always been filtered to klein-named candidates. Two dropdowns
    driving the SAME UNETLoader with different contents is the drift the repo
    warned about in that component's own header, and it is invisible on a machine
    whose diffusion_models holds nothing else."""
    d = comfy / 'models' / 'diffusion_models' / 'klein'
    _write(d / 'flux-2-klein-9b-fp8.safetensors')
    _write(comfy / 'models' / 'diffusion_models' / 'some_sdxl_unet.safetensors')
    with app.app_context():
        files, hint = picker.list_slot_files('klein_unet')
    assert any('flux-2-klein' in f for f in files)
    assert not any('some_sdxl' in f for f in files), (
        'the picker offered a UNET the Klein resolver refuses to build')
    assert hint


def test_both_unet_pickers_agree_with_what_capabilities_publishes(app, comfy):
    """The two ends, side by side, on one tree. This is the assertion that would
    have caught the divergence at the commit that introduced it: what Settings
    offers globally and what the per-dataset control offers must be the same set,
    because they write to the same loader."""
    d = comfy / 'models' / 'diffusion_models' / 'klein'
    _write(d / 'flux-2-klein-9b-fp8.safetensors')
    _write(d / 'flux-2-klein-9b-bf16.safetensors')
    _write(comfy / 'models' / 'diffusion_models' / 'some_sdxl_unet.safetensors')
    from app import capabilities
    with app.app_context():
        globally, _hint = picker.list_slot_files('klein_unet', force=True)
        per_dataset = capabilities._scan_models().get('klein') or []
    assert sorted(os.path.basename(f) for f in globally) == \
        sorted(os.path.basename(f) for f in per_dataset), (
            'the Settings picker and the per-dataset picker disagree about which '
            'files the same UNETLoader can load')
