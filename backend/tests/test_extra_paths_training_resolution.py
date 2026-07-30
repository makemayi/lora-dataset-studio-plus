"""The last two READ paths that ignored extra_model_paths.yaml.

Deploying a LoRA already follows the yaml (issue #25). Reading a BASE did not, in
two places, and each one broke a feature outright:

  * `lora_training._sdxl_base_path` walked ONLY <base>/models/checkpoints, then
    fell back to returning the bare name. An SDXL base declared in the yaml was
    handed to ai-toolkit as `bigLove_photo5.safetensors` — a relative path against
    ai-toolkit's own cwd — and the run died there, with ai-toolkit's message, about
    ai-toolkit's directory. Nothing pointed at the real cause.
  * `zimage_convert._resolve_merge` looked ONLY under <base>/models/{unet,
    diffusion_models}. A Z-Image merge under a yaml root was simply not convertible.

Every test here fails on the pre-fix code except the two `_without_yaml_` ones,
which are the anti-regression pins: with no yaml, resolution must be byte-for-byte
what it always was.

Casing: the on-disk spelling deliberately DIFFERS from the requested spelling, so
the case-insensitive guarantee is actually exercised on a case-SENSITIVE
filesystem (Linux/cloud), not just tautologically satisfied by Windows.
"""
import os
import struct

import pytest

# Structurally valid safetensors (8-byte LE header length + '{}'), so anything
# that validates the container rather than merely seeing the file is happy.
_ST = struct.pack('<Q', 2) + b'{}'


@pytest.fixture(autouse=True)
def _clear_cmp_cache():
    from app.services import comfy_model_paths as cmp
    from app.utils import comfyui as cu
    cmp.clear_cache()
    cu.clear_model_caches()     # get_checkpoint_models has a 5-min process-global TTL
    yield
    cmp.clear_cache()
    cu.clear_model_caches()


def _comfy(tmp_path, cfg):
    base = tmp_path / 'ComfyUI'
    (base / 'models').mkdir(parents=True)
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


def _write(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_ST)
    return path


def _yaml(base, text):
    (base / 'extra_model_paths.yaml').write_text(text, encoding='utf-8')


# --- Hole 1: the SDXL base ----------------------------------------------------

def test_sdxl_base_resolves_from_an_extra_model_paths_root(app, tmp_path):
    """RED before the fix: returns the bare name, which ai-toolkit cannot open."""
    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        target = _write(shared / 'checkpoints' / 'bigLove_photo5.safetensors')
        _yaml(base, f"other:\n  base_path: {shared.as_posix()}\n  checkpoints: checkpoints\n")
        assert lt._sdxl_base_path('bigLove_photo5.safetensors') == str(target)


def test_sdxl_base_resolves_a_subfoldered_name_from_an_extra_root(app, tmp_path):
    """get_checkpoint_models flattens to a basename, so the resolver must still find
    a file that lives in a SUBFOLDER of an extra root."""
    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        target = _write(shared / 'ckpts' / 'Biglove' / 'bigLove_photo5.safetensors')
        _yaml(base, f"other:\n  checkpoints: {shared.as_posix()}/ckpts\n")
        assert lt._sdxl_base_path('bigLove_photo5.safetensors') == str(target)


def test_sdxl_base_choices_include_extra_roots(app, tmp_path):
    """The launch guard whitelists basenames. If it can't see the extra root, the
    resolver fix is unreachable: every launch 400s with 'unknown SDXL checkpoint'."""
    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        _write(shared / 'checkpoints' / 'bigLove_photo5.safetensors')
        _yaml(base, f"other:\n  checkpoints: {shared.as_posix()}/checkpoints\n")
        assert 'bigLove_photo5.safetensors' in lt._sdxl_base_choices()


def test_sdxl_picker_lists_checkpoints_from_an_extra_root(app, tmp_path):
    """The whitelist fix is useless if the picker never OFFERS the base. This is the
    same hole one layer up: get_checkpoint_models only knew <base>/models/checkpoints."""
    from app import config as cfg
    from app.utils.comfyui import get_checkpoint_models
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        _write(shared / 'checkpoints' / 'Biglove' / 'bigLove_photo5.safetensors')
        _yaml(base, f"other:\n  checkpoints: {(shared / 'checkpoints').as_posix()}\n")
        assert 'bigLove_photo5.safetensors' in (get_checkpoint_models(include_hidden=True) or [])


def test_sdxl_picker_without_yaml_is_unchanged(app, tmp_path):
    """ANTI-REGRESSION for the picker: no yaml -> exactly the historical list."""
    from app import config as cfg
    from app.utils.comfyui import get_checkpoint_models
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        _write(base / 'models' / 'checkpoints' / 'sdxl_base.safetensors')
        _write(base / 'models' / 'checkpoints' / 'Biglove' / 'bigLove_photo5.safetensors')
        assert not (base / 'extra_model_paths.yaml').exists()
        assert sorted(get_checkpoint_models(include_hidden=True) or []) == [
            'bigLove_photo5.safetensors', 'sdxl_base.safetensors']


def test_sdxl_base_without_yaml_is_unchanged(app, tmp_path):
    """ANTI-REGRESSION. No yaml: exactly the historical <base>/models/checkpoints
    answer, root file and subfoldered file alike."""
    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        flat = _write(base / 'models' / 'checkpoints' / 'sdxl_base.safetensors')
        deep = _write(base / 'models' / 'checkpoints' / 'Biglove' / 'bigLove_photo5.safetensors')
        assert not (base / 'extra_model_paths.yaml').exists()
        assert lt._sdxl_base_path('sdxl_base.safetensors') == str(flat)
        assert lt._sdxl_base_path('bigLove_photo5.safetensors') == str(deep)
        assert lt._sdxl_base_path(os.path.join('Biglove', 'bigLove_photo5.safetensors')) == str(deep)


def test_sdxl_base_is_default_root_wins_over_the_base_models_folder(app, tmp_path):
    """Same filename in two roots -> the one a running ComfyUI would load. Getting
    this backwards trains on weights that are NOT the ones generating."""
    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        _write(base / 'models' / 'checkpoints' / 'twin.safetensors')
        winner = _write(shared / 'checkpoints' / 'twin.safetensors')
        _yaml(base, "primary:\n  is_default: true\n"
                    f"  checkpoints: {(shared / 'checkpoints').as_posix()}\n")
        assert lt._sdxl_base_path('twin.safetensors') == str(winner)


def test_sdxl_base_non_default_extra_root_loses_to_the_base_models_folder(app, tmp_path):
    """The mirror of the test above: without is_default, the default root keeps
    priority — same as folder_paths.add_model_folder_path's append."""
    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        winner = _write(base / 'models' / 'checkpoints' / 'twin.safetensors')
        _write(shared / 'checkpoints' / 'twin.safetensors')
        _yaml(base, f"other:\n  checkpoints: {(shared / 'checkpoints').as_posix()}\n")
        assert lt._sdxl_base_path('twin.safetensors') == str(winner)


def test_sdxl_base_tolerates_a_different_casing(app, tmp_path):
    """On disk 'Biglove/bigLove_photo5'; asked for 'biglove/BIGLOVE_PHOTO5'. Must
    hold on a case-SENSITIVE filesystem too."""
    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        target = _write(shared / 'checkpoints' / 'Biglove' / 'bigLove_photo5.safetensors')
        _yaml(base, f"other:\n  checkpoints: {(shared / 'checkpoints').as_posix()}\n")
        got = lt._sdxl_base_path(os.path.join('biglove', 'BIGLOVE_PHOTO5.safetensors'))
        assert os.path.normcase(got) == os.path.normcase(str(target))


def test_sdxl_base_missing_everywhere_names_the_file(app, tmp_path):
    """The real defect behind both holes: a silent bare-name fallback moved the
    failure to ai-toolkit, where the message is incomprehensible. Fail HERE, by name."""
    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        (shared / 'checkpoints').mkdir(parents=True)
        _yaml(base, f"other:\n  checkpoints: {(shared / 'checkpoints').as_posix()}\n")
        with pytest.raises(ValueError) as ei:
            lt._sdxl_base_path('ghost.safetensors')
        assert 'ghost.safetensors' in str(ei.value)


# --- Hole 2: the Z-Image merge to convert -------------------------------------

def test_zimage_merge_resolves_from_an_extra_model_paths_root(app, tmp_path):
    """RED before the fix: None -> 'base model not found on disk', for a file that
    is on disk and that ComfyUI itself loads every day."""
    from app import config as cfg
    from app.services import zimage_convert as zc
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        target = _write(shared / 'unet' / 'z image' / 'bigLove_zt3.safetensors')
        _yaml(base, f"other:\n  unet: {(shared / 'unet').as_posix()}\n")
        assert zc._resolve_merge(os.path.join('z image', 'bigLove_zt3.safetensors')) \
            == os.path.realpath(str(target))


def test_zimage_merge_uses_the_diffusion_models_alias(app, tmp_path):
    """The yaml key may be `unet` OR `diffusion_models`; folder_paths.map_legacy
    folds them together, so both must work."""
    from app import config as cfg
    from app.services import zimage_convert as zc
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        target = _write(shared / 'dm' / 'z image' / 'bigLove_zt3.safetensors')
        _yaml(base, f"other:\n  diffusion_models: {(shared / 'dm').as_posix()}\n")
        assert zc._resolve_merge(os.path.join('z image', 'bigLove_zt3.safetensors')) \
            == os.path.realpath(str(target))


def test_zimage_picker_lists_merges_from_an_extra_root(app, tmp_path):
    """Same reachability point as the SDXL picker: a merge the base picker never
    offers cannot be converted, however well _resolve_merge resolves it."""
    from app import config as cfg
    from app.utils.comfyui import get_zimage_models
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        _write(shared / 'unet' / 'z image' / 'bigLove_zt3.safetensors')
        _yaml(base, f"other:\n  unet: {(shared / 'unet').as_posix()}\n")
        assert os.path.join('z image', 'bigLove_zt3.safetensors') in get_zimage_models()


def test_zimage_picker_without_yaml_is_unchanged(app, tmp_path):
    """ANTI-REGRESSION for the Z-Image picker."""
    from app import config as cfg
    from app.utils.comfyui import get_zimage_models
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        _write(base / 'models' / 'unet' / 'z image' / 'a.safetensors')
        # decoy in a foreign family folder — must not leak into the Z-Image list.
        # (Careful: the folder filter is a SUBSTRING match, so a name like
        # 'not_zimage' would legitimately match. Pre-existing looseness, not ours.)
        _write(base / 'models' / 'unet' / 'krea' / 'b.safetensors')
        assert not (base / 'extra_model_paths.yaml').exists()
        assert get_zimage_models() == [os.path.join('z image', 'a.safetensors')]


def test_zimage_merge_without_yaml_is_unchanged(app, tmp_path):
    """ANTI-REGRESSION: both historical subfolders, both historical shapes (the
    full relative name, and the bare basename found under 'z image/')."""
    from app import config as cfg
    from app.services import zimage_convert as zc
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        u = _write(base / 'models' / 'unet' / 'z image' / 'a.safetensors')
        d = _write(base / 'models' / 'diffusion_models' / 'z image' / 'b.safetensors')
        assert not (base / 'extra_model_paths.yaml').exists()
        assert zc._resolve_merge(os.path.join('z image', 'a.safetensors')) == os.path.realpath(str(u))
        assert zc._resolve_merge(os.path.join('z image', 'b.safetensors')) == os.path.realpath(str(d))
        # bare basename -> the historical 'z image/' fallback
        assert zc._resolve_merge('a.safetensors') == os.path.realpath(str(u))


def test_zimage_merge_is_default_root_wins(app, tmp_path):
    from app import config as cfg
    from app.services import zimage_convert as zc
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        _write(base / 'models' / 'unet' / 'z image' / 'twin.safetensors')
        winner = _write(shared / 'unet' / 'z image' / 'twin.safetensors')
        _yaml(base, "primary:\n  is_default: true\n"
                    f"  unet: {(shared / 'unet').as_posix()}\n")
        assert zc._resolve_merge(os.path.join('z image', 'twin.safetensors')) \
            == os.path.realpath(str(winner))


def test_zimage_merge_tolerates_a_different_casing(app, tmp_path):
    """On disk 'Z Image/'; the stored value says 'z image\\' (the workflow's own
    spelling). bobba84's report, applied to the conversion path."""
    from app import config as cfg
    from app.services import zimage_convert as zc
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        target = _write(shared / 'unet' / 'Z Image' / 'BigLove_ZT3.safetensors')
        _yaml(base, f"other:\n  unet: {(shared / 'unet').as_posix()}\n")
        got = zc._resolve_merge(os.path.join('z image', 'biglove_zt3.safetensors'))
        assert got and os.path.normcase(got) == os.path.normcase(os.path.realpath(str(target)))


def test_zimage_merge_missing_everywhere_is_named(app, tmp_path):
    from app import config as cfg
    from app.services import zimage_convert as zc
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        (shared / 'unet').mkdir(parents=True)
        # convert() touches the ai-toolkit dir before it resolves the merge; give it
        # one so the test fails on the MISSING MODEL, not on an unconfigured toolkit.
        cfg.save_config({'comfyui': {'base_dir': str(base)},
                         'aitoolkit': {'dir': str(tmp_path / 'ait')}})
        _yaml(base, f"other:\n  unet: {(shared / 'unet').as_posix()}\n")
        assert zc._resolve_merge(os.path.join('z image', 'ghost.safetensors')) is None
        with pytest.raises(ValueError) as ei:
            zc.convert(os.path.join('z image', 'ghost.safetensors'))
        assert 'ghost.safetensors' in str(ei.value)


def test_zimage_merge_still_refuses_traversal_across_extra_roots(app, tmp_path):
    """Adding roots must not add escape hatches: absolute values and '..' stay
    refused, whatever the yaml declares."""
    from app import config as cfg
    from app.services import zimage_convert as zc
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        shared = tmp_path / 'Shared'
        outside = _write(tmp_path / 'outside' / 'evil.safetensors')
        (shared / 'unet').mkdir(parents=True)
        _yaml(base, f"other:\n  unet: {(shared / 'unet').as_posix()}\n")
        assert zc._resolve_merge(os.path.join('..', '..', 'outside', 'evil.safetensors')) is None
        assert zc._resolve_merge(str(outside)) is None
        assert zc._resolve_merge('') is None
