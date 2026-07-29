"""Every place that COMPOSES a relative model name, one test each.

WHY ONE TEST PER PRODUCER (2026-07-28)
--------------------------------------
GitHub #21 (1Tomber) arrived quoting two widgets, `unet_name` and `lora_name`.
Fixing the two it quoted would have been the wrong size of fix: the same
hardcoded `.replace("/", "\\")` appeared in SIX listers/resolvers, so the report
would simply have come back through another door — `ckpt_name` from the SDXL
lane, `vae_name` from Klein — and each time it would have read like a new bug.

So this file is an INVENTORY, not a repro. Each producer gets a case, and each
case asserts the same two things:

  * the name is joined with the separator of the tree we actually walked
    (`os.sep`) and never with a hardcoded backslash — on Windows that is
    invisible, on Linux it is the whole bug;
  * once put through `canonical_model_widgets` against a POSIX ComfyUI (the
    reporter's install), no backslash survives.

The second assertion is what makes this file provable on Windows: the LINUX
outcome is driven through the separator seam rather than pretended.
"""
import os

import pytest

from app.utils.comfy_names import canonical_model_widgets, normalise_model_name

pytestmark = pytest.mark.usefixtures('app')

POSIX = '/'


def _posix_enum(cls, field, *names):
    """A distilled /object_info view as a LINUX ComfyUI publishes it."""
    return {cls: {field: {normalise_model_name(n): n for n in names}}}


def _base(tmp_path):
    """A ComfyUI base the config accessors resolve output/models/loras from."""
    from app import config as cfg
    base = tmp_path / 'ComfyUI'
    for sub in ('models/unet', 'models/diffusion_models', 'models/loras',
                'models/checkpoints', 'output', 'input'):
        (base / sub).mkdir(parents=True, exist_ok=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


def _touch(base, *parts):
    p = base.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b'x')
    return p


def _assert_host_joined(name, folder, filename):
    """The producer's own output: segments joined with THIS host's separator, and
    no backslash smuggled in by hand."""
    assert name == os.path.join(folder, filename), name
    if os.sep == '/':                      # a Linux dev box / the CI container
        assert '\\' not in name, name


def _assert_posix_after_canonicalisation(cls, field, name):
    """And the reporter's install accepts it."""
    graph = {'1': {'class_type': cls, 'inputs': {field: name}}}
    published = name.replace('\\', '/')
    out, _ = canonical_model_widgets(graph, _posix_enum(cls, field, published))
    assert out['1']['inputs'][field] == published
    assert '\\' not in out['1']['inputs'][field]


# --- 1. resolve_checkpoint_ckpt_name (SDXL / image_real_HQ lane) -----------

def test_resolve_checkpoint_ckpt_name(tmp_path):
    from app.utils import comfyui
    base = _base(tmp_path)
    _touch(base, 'models', 'checkpoints', 'Biglove', 'photo5.safetensors')
    comfyui.clear_model_caches()
    name = comfyui.resolve_checkpoint_ckpt_name('photo5.safetensors')
    _assert_host_joined(name, 'Biglove', 'photo5.safetensors')
    _assert_posix_after_canonicalisation('CheckpointLoaderSimple', 'ckpt_name', name)


def test_resolve_checkpoint_ckpt_name_keeps_an_already_relative_name(tmp_path):
    """The early-return branch had its own hardcoded backslash — a name arriving
    from a config or a DB row went through it untouched by the walk above."""
    from app.utils import comfyui
    _base(tmp_path)
    assert comfyui.resolve_checkpoint_ckpt_name('Biglove/photo5.safetensors') == \
        os.path.join('Biglove', 'photo5.safetensors')


# --- 2-3. the UNET listers -------------------------------------------------

def test_get_zimage_models(tmp_path):
    from app.utils import comfyui
    base = _base(tmp_path)
    _touch(base, 'models', 'unet', 'z image', 'bigLove_zt3.safetensors')
    comfyui.clear_model_caches()
    names = comfyui.get_zimage_models()
    assert names, 'the lister found nothing - the assertions below would be vacuous'
    _assert_host_joined(names[0], 'z image', 'bigLove_zt3.safetensors')
    _assert_posix_after_canonicalisation('UNETLoader', 'unet_name', names[0])


def test_get_krea_models(tmp_path):
    from app.utils import comfyui
    base = _base(tmp_path)
    _touch(base, 'models', 'unet', 'Krea', 'krea2_turbo_fp8.safetensors')
    comfyui.clear_model_caches()
    names = [n for n in comfyui.get_krea_models() if os.sep in n or '/' in n or '\\' in n]
    assert names
    _assert_host_joined(names[0], 'Krea', 'krea2_turbo_fp8.safetensors')
    _assert_posix_after_canonicalisation('UNETLoader', 'unet_name', names[0])


# --- 4-6. the LoRA listers -------------------------------------------------

@pytest.mark.parametrize('lister,folder', [
    ('get_zimage_loras', 'z image'),
    ('get_sdxl_loras', 'sdxl'),
    ('get_krea_loras', 'krea'),
])
def test_lora_listers(tmp_path, lister, folder):
    from app.utils import comfyui
    base = _base(tmp_path)
    _touch(base, 'models', 'loras', folder, 'mylora.safetensors')
    comfyui.clear_model_caches()
    rows = getattr(comfyui, lister)()
    assert rows, f'{lister} found nothing'
    name = rows[0]['filename']
    _assert_host_joined(name, folder, 'mylora.safetensors')
    _assert_posix_after_canonicalisation('LoraLoaderModelOnly', 'lora_name', name)


# --- 7. comfy_model_paths.list_models — the folder_paths mirror ------------

def test_list_models_mirrors_folder_paths(tmp_path):
    """Already os.sep-joined (it mirrors ComfyUI's own recursive_search), so this
    case exists to keep it that way, not to fix it."""
    from app.services import comfy_model_paths
    base = _base(tmp_path)
    _touch(base, 'models', 'unet', 'klein', 'flux-2-klein-9b-fp8.safetensors')
    comfy_model_paths.clear_cache()
    names = [rel for rel, _ab in comfy_model_paths.list_models('diffusion_models')]
    assert names
    _assert_host_joined(names[0], 'klein', 'flux-2-klein-9b-fp8.safetensors')
    _assert_posix_after_canonicalisation('UNETLoader', 'unet_name', names[0])


# --- 8. the shipped workflow templates -------------------------------------

def test_every_shipped_template_survives_a_posix_comfyui():
    """The templates were captured on a Windows machine and pin names like
    "Krea\\krea2_turbo_fp8.safetensors" — several of which no resolver overwrites
    (the SDXL checkpoint, the DMD2 and subtle LoRAs). They are LDS-internal until
    the queue respells them, and this is where that promise is checked.

    Deliberately NOT "the templates must contain forward slashes": rewriting them
    would put the Windows install's correctness on the canonicaliser's shoulders
    with no fallback, and Windows is where almost every install is today."""
    import json
    from pathlib import Path

    from app.utils.comfy_names import MODEL_FILE_INPUTS
    workflow_dir = Path(__file__).resolve().parents[1] / 'workflows'
    checked = 0
    for path in sorted(workflow_dir.glob('*.json')):
        graph = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(graph, dict):
            continue
        out, _ = canonical_model_widgets(graph, None, sep=POSIX)
        for node_id, node in out.items():
            if not isinstance(node, dict) or not isinstance(node.get('inputs'), dict):
                continue
            for field, value in node['inputs'].items():
                if field in MODEL_FILE_INPUTS and isinstance(value, str) and value:
                    assert '\\' not in value, f'{path.name} node {node_id}: {field}={value!r}'
                    checked += 1
    assert checked >= 8, f'only {checked} model widgets seen across the templates'


# --- 9. the local-filesystem side, which must NOT become POSIX -------------

def test_zimage_convert_resolves_a_subfoldered_model_on_this_filesystem(tmp_path):
    """The same bug one layer down and in the opposite direction: this path is
    OPENED, so it must carry os.sep. On Linux the old hardcoded backslash did not
    raise — it built the name of a single file containing a backslash, which no
    install has, and the conversion just "could not find the model"."""
    from app.services import zimage_convert
    base = _base(tmp_path)
    real = _touch(base, 'models', 'unet', 'z image', 'bigLove_zt3.safetensors')
    for spelling in ('z image/bigLove_zt3.safetensors', 'z image\\bigLove_zt3.safetensors'):
        got = zimage_convert._resolve_merge(spelling)
        assert got and os.path.samefile(got, str(real)), (spelling, got)
