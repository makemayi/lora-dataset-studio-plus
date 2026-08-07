"""Krea2-HQ (✨ improve) asset resolution — the folder contract.

The graph in `workflows/krea2 high resolution.json` loads its SeedVR2 halves
through a CORE `UNETLoader` / `VAELoader`, not through the SeedVR2 pack's own
loader nodes. Those core nodes only list ComfyUI's `diffusion_models` and `vae`
folders, so where the resolvers LOOK has to match where the nodes can LOAD from.

It shipped looking in the pack-private `SEEDVR2` folder instead, which broke it
both ways at once: a correctly installed DiT reported "missing" in a 409 that
told the user to move it somewhere the graph could never have used, and had the
scan matched there, ComfyUI would have answered the enqueue with a bare 400.
Same class of mistake as SeedVR2's own destination test — a file that lands
where nothing looks is a multi-gigabyte no-op.
"""
import os
import struct


def _make_comfyui(root):
    base = root / 'ComfyUI'
    (base / 'models').mkdir(parents=True, exist_ok=True)
    (base / 'main.py').write_text('# fake ComfyUI entrypoint', encoding='utf-8')
    return base


def _safetensors(payload_size=1024):
    header = b'{"__metadata__":{"lds":"test"}}'
    return struct.pack('<Q', len(header)) + header + b'\0' * payload_size


def _install(base, folder, *names):
    d = base / 'models' / folder
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        (d / name).write_bytes(_safetensors())
    return d


def _configured(app, base):
    from app import config
    from app.services import comfy_model_paths
    config.save_config({'comfyui': {'base_dir': str(base)}})
    comfy_model_paths.clear_cache()


# --- Where the core loaders can actually read from ---------------------------

def test_the_dit_resolves_from_diffusion_models(app, tmp_path):
    from app.services import krea_hq_helper as khh
    base = _make_comfyui(tmp_path)
    _install(base, 'diffusion_models', khh._SEEDVR2_UNET_CANONICAL)
    _install(base, 'vae', khh._SEEDVR2_VAE_CANONICAL)
    with app.app_context():
        _configured(app, base)
        assert khh.resolve_seedvr2_unet() == khh._SEEDVR2_UNET_CANONICAL
        assert khh.resolve_seedvr2_vae() == khh._SEEDVR2_VAE_CANONICAL
        assert 'seedvr2_model' not in khh.krea_hq_missing_assets()
        assert 'seedvr2_vae' not in khh.krea_hq_missing_assets()


def test_the_pack_private_seedvr2_folder_does_not_count(app, tmp_path):
    """seedvr2_helper's own weights live in `models/SEEDVR2` and its own nodes
    load them from there. A core UNETLoader/VAELoader cannot see that folder, so
    a copy there must still report missing — reporting it PRESENT would enqueue
    a name ComfyUI rejects with a bare 400 whose text is in its console, not the
    app."""
    from app.services import krea_hq_helper as khh
    base = _make_comfyui(tmp_path)
    _install(base, 'SEEDVR2', khh._SEEDVR2_UNET_CANONICAL, khh._SEEDVR2_VAE_CANONICAL)
    with app.app_context():
        _configured(app, base)
        assert khh.resolve_seedvr2_unet() is None
        assert khh.resolve_seedvr2_vae() is None
        missing = khh.krea_hq_missing_assets()
        assert 'seedvr2_model' in missing and 'seedvr2_vae' in missing


def test_every_named_409_path_is_a_folder_its_resolver_scans(app, tmp_path):
    """The 409 names an exact path. Placing the file there must make the SAME
    resolver that raised the 409 find it — otherwise the message sends the user
    on an errand that ends where it started."""
    from app.services import krea_hq_helper as khh
    base = _make_comfyui(tmp_path)
    with app.app_context():
        _configured(app, base)
        for key in khh.KREA_HQ_REQUIRED:
            rel = khh.KREA_HQ_ASSETS[key]['path']
            dest = base.joinpath(*rel.split('/'))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_safetensors())
        from app.services import comfy_model_paths
        comfy_model_paths.clear_cache()
        assert khh.resolve_seedvr2_unet() is not None
        assert khh.resolve_seedvr2_vae() is not None
        assert khh.resolve_quality_lora()[0] is not None


# --- Resolution never guesses ------------------------------------------------

def test_an_unrelated_file_is_never_handed_to_either_loader(app, tmp_path):
    """`diffusion_models` and `vae` are shared folders full of other models. A
    wrong DiT build (different parameter count) or a wrong VAE fails deep inside
    the node, so neither resolver falls back to whatever else is present."""
    from app.services import krea_hq_helper as khh
    base = _make_comfyui(tmp_path)
    _install(base, 'diffusion_models', 'flux-2-klein-9b.safetensors')
    _install(base, 'vae', 'qwen_image_vae.safetensors',
             'vae-ft-mse-840000-ema-pruned.safetensors')
    with app.app_context():
        _configured(app, base)
        assert khh.resolve_seedvr2_unet() is None
        assert khh.resolve_seedvr2_vae() is None


def test_the_base_model_setting_cannot_pull_a_raw_build_into_this_lane(app, tmp_path):
    """`krea.base_model` belongs to the OTHER Krea lanes. This graph's
    BasicScheduler is pinned to 10 steps — a distilled build's step count — so a
    Raw build here does not fail, it under-samples, and the user sees a soft
    noisy improve rather than an error. Turbo wins even when the setting says
    Raw."""
    from app import config
    from app.services import krea_hq_helper as khh, comfy_model_paths
    base = _make_comfyui(tmp_path)
    _install(base, 'diffusion_models', 'Krea2_Raw_convrot_int8mixed.safetensors',
             'Krea2_Turbo_convrot_int8mixed.safetensors')
    with app.app_context():
        config.save_config({
            'comfyui': {'base_dir': str(base)},
            'krea': {'base_model': 'Krea2_Raw_convrot_int8mixed.safetensors'}})
        comfy_model_paths.clear_cache()
        assert khh.krea_unet() == 'Krea2_Turbo_convrot_int8mixed.safetensors'


def test_a_raw_only_install_reports_missing_instead_of_under_sampling(app, tmp_path):
    """Same discipline as require_raw's: a build a pipeline cannot serve must
    report missing — a named 409 the user can act on — never get silently
    handed to it."""
    from app import config
    from app.services import krea_hq_helper as khh, comfy_model_paths
    base = _make_comfyui(tmp_path)
    _install(base, 'diffusion_models', 'Krea2_Raw_convrot_int8mixed.safetensors')
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        assert khh.krea_unet() is None
        assert 'krea_model' in khh.krea_hq_missing_assets()


def test_the_two_build_requirements_are_mutually_exclusive():
    """Asking for both narrows the candidates to nothing and would report every
    install empty-handed — a caller error, not a missing file."""
    import pytest
    from app.services import krea_edit_helper as keh
    with pytest.raises(ValueError):
        keh.resolve_krea_unet(require_raw=True, require_turbo=True)


def test_a_subfoldered_build_keeps_its_prefix(app, tmp_path):
    """The loader widget lists `sub/name`, and ComfyUI's validator is an exact
    string match — a basename alone would 400."""
    from app.services import krea_hq_helper as khh
    base = _make_comfyui(tmp_path)
    _install(base, os.path.join('diffusion_models', 'SeedVR2'),
             khh._SEEDVR2_UNET_CANONICAL)
    with app.app_context():
        _configured(app, base)
        assert khh.resolve_seedvr2_unet() == os.path.join(
            'SeedVR2', khh._SEEDVR2_UNET_CANONICAL)
