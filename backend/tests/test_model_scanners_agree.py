"""The four family scanners, pinned file by file on ONE tree.

WHY THIS FILE EXISTS
--------------------
Four functions walk the same ComfyUI folders looking for one family's weights:
`comfyui.get_krea_models`, `comfyui.get_zimage_models`,
`krea_edit_helper._krea_unet_folders` and `klein_edit_helper._klein_unet_folders`.
They were four hand-written walks, and their drift has already shipped a bug —
two of them answered differently about the SAME folder, so a file the picker
offered was a file the resolver refused.

These tests are a GOLDEN RECORD taken before the four walks were folded onto two
shared ones. Every list below was produced by the code as it stood, on the tree
built by the `tree` fixture; the refactor is only allowed if they all stay green.
They are deliberately exact-equality assertions on full lists rather than `in`
checks: a consolidation that quietly widens or narrows a list is exactly the
failure mode, and a membership test would sail straight past it.

WHERE THE FOUR STILL DISAGREE, ON PURPOSE OR NOT
------------------------------------------------
One difference is pinned here rather than fixed, because removing it would be a
behaviour change and that pass was a refactor:

  * DEPTH. `get_*_models` walk the tree (`os.walk`), so a model two folders deep
    is listed. `_*_unet_folders` read ONE level, so the same file is invisible to
    the Generate resolvers. Not levelled blind: which of the two matches what
    ComfyUI itself lists is a question that deserves its own measurement.

The other one was levelled, in its own commit and on purpose — see
`test_the_studio_no_longer_offers_a_base_generate_refuses` below.
"""
import importlib
import os

import pytest


@pytest.fixture
def tree(monkeypatch, tmp_path):
    """One ComfyUI tree, seen by BOTH root-discovery paths: `comfyui._out_dir()`
    (which reads `<output>/../models`) and `comfy_model_paths.search_roots`
    (which reads `comfyui.base_dir`). They are two different config routes to the
    same folders — a test that wired only one would compare scanners that were
    not even looking at the same disk."""
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as config
    importlib.reload(config)
    base = tmp_path / 'Comfy'
    (base / 'output').mkdir(parents=True)
    models = base / 'models'
    for sub in ('unet', 'diffusion_models', 'loras', 'text_encoders', 'vae'):
        (models / sub).mkdir(parents=True)
    config.save_config({'comfyui': {'base_dir': str(base)}})

    def put(*parts):
        p = models.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'W')

    # --- unet/ ---------------------------------------------------------------
    put('unet', 'Krea', 'krea2_turbo_fp8.safetensors')
    put('unet', 'Krea', 'BigLoveKreaEdit1_fp8mixed.safetensors')   # noise base
    put('unet', 'Krea', 'notes.txt')                               # not a model
    put('unet', 'Krea', 'nested', 'deep_krea.safetensors')         # depth 2
    put('unet', 'Krea', 'quant.gguf')
    put('unet', 'BigLoveKreaEdit1_fp8mixed.safetensors')           # noise, at root
    put('unet', 'Krea_full_x_fp8.safetensors')                     # root, named krea
    put('unet', 'flux2-klein-9b.safetensors')                      # another family
    put('unet', 'z image', 'zt3.safetensors')
    put('unet', 'z image', 'sub', 'deep_z.safetensors')            # depth 2
    put('unet', 'klein', 'flux2-klein-4b.safetensors')
    # --- diffusion_models/ ---------------------------------------------------
    put('diffusion_models', 'krea2', 'another_krea.safetensors')
    put('diffusion_models', 'zimage', 'z_merge.sft')
    put('diffusion_models', 'Flux2 klein', 'klein9b.safetensors')
    put('diffusion_models', 'klein_root.safetensors')

    from app.services import comfy_model_paths
    from app.utils import comfyui
    comfy_model_paths.clear_cache()
    monkeypatch.setattr(comfyui, '_out_dir', lambda: str(base / 'output'))
    comfyui.clear_model_caches()
    yield base
    comfy_model_paths.clear_cache()
    comfyui.clear_model_caches()


def _j(*parts):
    return os.path.join(*parts)


# --- Krea, the Studio's list --------------------------------------------------

def test_get_krea_models_is_pinned_file_by_file(tree):
    from app.utils.comfyui import get_krea_models
    assert get_krea_models() == [
        _j('Krea', 'krea2_turbo_fp8.safetensors'),
        _j('Krea', 'nested', 'deep_krea.safetensors'),
        _j('Krea', 'quant.gguf'),
        'Krea_full_x_fp8.safetensors',
        _j('krea2', 'another_krea.safetensors'),
    ]


def test_the_studio_no_longer_offers_a_base_generate_refuses(tree):
    """THE ONE BEHAVIOUR CHANGE, and the reason for it.

    `BigLove*` carries 'krea' and renders PURE NOISE under the Krea pipeline —
    measured, and already in `KREA_INCOMPATIBLE_TOKENS`. The Generate resolver
    refuses it everywhere. The Studio's lister refused it only at a ROOT, because
    the check lived inside `_krea_root_candidate`, which by construction never
    runs on a subfolder. So the same file was refused at a root and offered as a
    Test Studio base one folder down — an accident of placement, not a decision.

    Someone picking it got noise and concluded the model, or the app, was broken.
    The exclusion now applies at every depth, on both surfaces."""
    from app.utils.comfyui import get_krea_models
    from app.services import krea_edit_helper as keh
    importlib.reload(keh)
    listed = get_krea_models()
    assert not any('biglove' in m.lower() for m in listed)
    resolver = [n for _sub, group in keh._krea_unet_folders() for n in group]
    assert not any('biglove' in n.lower() for n in resolver)


# --- Z-Image, the Studio's list ----------------------------------------------

def test_get_zimage_models_is_pinned_file_by_file(tree):
    from app.utils.comfyui import get_zimage_models
    assert get_zimage_models() == [
        _j('z image', 'sub', 'deep_z.safetensors'),
        _j('z image', 'zt3.safetensors'),
        _j('zimage', 'z_merge.sft'),
    ]


def test_a_zimage_file_at_a_root_is_not_listed(tree):
    """Unlike Krea, this family has no root-filename rule: the folder is the only
    claim. Pinned because the two listers look alike and are not."""
    from app.utils.comfyui import get_zimage_models
    assert not any(os.sep not in m for m in get_zimage_models())


# --- Krea, the Generate resolver's folders ------------------------------------

def test_krea_unet_folders_is_pinned_group_by_group(tree):
    from app.services import krea_edit_helper as keh
    importlib.reload(keh)
    assert keh._krea_unet_folders() == [
        ('Krea', ['krea2_turbo_fp8.safetensors', 'quant.gguf']),
        ('', ['Krea_full_x_fp8.safetensors']),
        ('krea2', ['another_krea.safetensors']),
    ]


def test_the_resolver_refuses_the_noise_base_in_a_subfolder_too(tree):
    """The other half of the asymmetry above, on the same tree and the same file."""
    from app.services import krea_edit_helper as keh
    importlib.reload(keh)
    names = [n for _sub, group in keh._krea_unet_folders() for n in group]
    assert not any('biglove' in n.lower() for n in names)


def test_the_resolver_never_sees_a_model_two_folders_deep(tree):
    """`deep_krea.safetensors` IS offered by the Studio and IS invisible here."""
    from app.services import krea_edit_helper as keh
    from app.utils.comfyui import get_krea_models
    importlib.reload(keh)
    names = [n for _sub, group in keh._krea_unet_folders() for n in group]
    assert 'deep_krea.safetensors' not in names
    assert _j('Krea', 'nested', 'deep_krea.safetensors') in get_krea_models()


# --- Klein, the Generate resolver's folders -----------------------------------

def test_klein_unet_folders_is_pinned_group_by_group(tree):
    from app.services import klein_edit_helper as kleh
    importlib.reload(kleh)
    assert kleh._klein_unet_folders() == [
        ('klein', ['flux2-klein-4b.safetensors']),
        ('', ['flux2-klein-9b.safetensors']),
        ('Flux2 klein', ['klein9b.safetensors']),
        ('', ['klein_root.safetensors']),
    ]


def test_klein_keeps_every_candidate_it_finds(tree):
    """Klein has no incompatible-base list — pinned so a consolidation cannot
    hand it Krea's exclusion by accident."""
    from app.services import klein_edit_helper as kleh
    importlib.reload(kleh)
    groups = kleh._klein_unet_folders()
    assert sum(len(names) for _sub, names in groups) == 4
