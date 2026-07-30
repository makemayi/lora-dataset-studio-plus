"""Deploying a LoRA must land where ComfyUI actually looks for LoRAs.

Reported by Geekswordsman (GitHub #25): with a `loras:` root declared in
`extra_model_paths.yaml`, both the deploy action and the "open LoRA folder"
button used `<ComfyUI>/models/loras` anyway. The app was inconsistent with
itself — READING went through `comfy_model_paths.search_roots('loras')` (which
mirrors ComfyUI's own root priority), WRITING through `cfg.comfyui_dir('loras')`,
which never reads the yaml.

The contract pinned here:
  * priority for the WRITE root: explicit `comfyui.loras_dir` override >
    yaml roots (ComfyUI's own order, `is_default` first) > `<base>/models/loras`;
  * with no yaml, byte-for-byte the historical folder;
  * the folder button opens exactly the folder deploys write to;
  * a LoRA deployed BEFORE this change (in the old default root) stays listed,
    resolvable and deletable — a path fix must not orphan what is on disk.
"""
import os
import textwrap
import types

import pytest


@pytest.fixture(autouse=True)
def _clear_cmp_cache():
    from app.services import comfy_model_paths as cmp
    cmp.clear_cache()
    cmp._warned.clear()
    yield
    cmp.clear_cache()
    cmp._warned.clear()


def _comfy_base(tmp_path, cfg, **comfy):
    base = tmp_path / 'ComfyUI'
    (base / 'models' / 'loras').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    cfg.save_config({'comfyui': dict({'base_dir': str(base)}, **comfy)})
    return base


def _write_yaml(base, text):
    (base / 'extra_model_paths.yaml').write_text(textwrap.dedent(text), encoding='utf-8')


def _ds(trigger='tata', train_type='zimage', ds_id=1):
    return types.SimpleNamespace(id=ds_id, trigger_word=trigger, train_type=train_type)


def _lora_file(path):
    """A structurally valid (empty-tensor) safetensors, so arch detection reads it
    instead of blowing up on garbage."""
    import json
    import struct
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = json.dumps({'__metadata__': {}}).encode()
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(body)) + body)
    return path


# --- where a deploy lands ---------------------------------------------------

def test_deploy_follows_yaml_default_root(app, tmp_path):
    """#25: an `is_default` loras root in the yaml IS the folder ComfyUI treats as
    primary, so it is the folder a deployed LoRA must land in."""
    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        external = tmp_path / 'external'
        (external / 'loras').mkdir(parents=True)
        _write_yaml(base, f"""
            comfyui:
              base_path: {external}
              is_default: true
              loras: loras
        """)
        assert str(lt._lora_dest_dir_zimage()) == os.path.join(
            os.path.normpath(str(external / 'loras')), 'z image')
        assert str(lt._lora_dest_dir_krea()) == os.path.join(
            os.path.normpath(str(external / 'loras')), 'krea')


def test_explicit_override_beats_the_yaml(app, tmp_path):
    """Someone who filled `comfyui.loras_dir` said exactly where files go; the
    yaml must not take that back."""
    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        chosen = tmp_path / 'chosen'
        chosen.mkdir()
        base = _comfy_base(tmp_path, cfg, loras_dir=str(chosen))
        external = tmp_path / 'external'
        (external / 'loras').mkdir(parents=True)
        _write_yaml(base, f"""
            comfyui:
              base_path: {external}
              is_default: true
              loras: loras
        """)
        assert str(lt._lora_dest_dir_zimage()) == os.path.join(str(chosen), 'z image')


def test_no_yaml_keeps_the_historical_folder(app, tmp_path):
    """Installs without a yaml must be byte-for-byte unchanged."""
    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        assert str(lt._lora_dest_dir_zimage()) == os.path.join(
            str(base), 'models', 'loras', 'z image')
        assert str(lt._lora_dest_dir_sdxl()) == os.path.join(
            str(base), 'models', 'loras', 'sdxl')


def test_yaml_without_is_default_keeps_comfyui_priority(app, tmp_path):
    """A plain extra root is a SECONDARY location for ComfyUI (appended, not
    inserted): it must not silently steal the deploys. `is_default` is the lever."""
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        external = tmp_path / 'external'
        (external / 'loras').mkdir(parents=True)
        _write_yaml(base, f"""
            a111:
              base_path: {external}
              loras: loras
        """)
        assert cmp.write_root('loras') == os.path.normpath(str(base / 'models' / 'loras'))
        assert str(lt._lora_dest_dir_zimage()) == os.path.join(
            str(base / 'models' / 'loras'), 'z image')


def test_several_roots_the_is_default_one_receives(app, tmp_path):
    """Reading may span every root; writing needs ONE. It is the first root of
    ComfyUI's own priority order — which `is_default` puts at the front."""
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        first, second = tmp_path / 'shared', tmp_path / 'primary'
        (first / 'loras').mkdir(parents=True)
        (second / 'loras').mkdir(parents=True)
        _write_yaml(base, f"""
            a111:
              base_path: {first}
              loras: loras
            portable:
              base_path: {second}
              is_default: true
              loras: loras
        """)
        roots = cmp.search_roots('loras')
        assert roots[0] == os.path.normpath(str(second / 'loras'))
        assert cmp.write_root('loras') == roots[0]
        assert str(lt._lora_dest_dir_zimage()).startswith(
            os.path.normpath(str(second / 'loras')))


# --- the folder button and the write path may never diverge -----------------

def test_open_folder_button_opens_the_deploy_folder(app, tmp_path, monkeypatch):
    """Half of #25 is that the button showed a different folder than the one used.
    Both must resolve through the same accessor, whatever the config shape."""
    import sys

    from app import config as cfg
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        external = tmp_path / 'external'
        (external / 'loras').mkdir(parents=True)
        _write_yaml(base, f"""
            comfyui:
              base_path: {external}
              is_default: true
              loras: loras
        """)
        ds = _ds()
        monkeypatch.setattr(lt.fds, 'get_dataset', lambda u, d: ds)
        opened = []
        if os.name == 'nt':
            monkeypatch.setattr(os, 'startfile', opened.append, raising=False)
        else:
            monkeypatch.setattr(lt.subprocess, 'Popen', lambda argv: opened.append(argv[-1]))
        monkeypatch.setattr(sys, 'platform', 'linux', raising=False)
        path = lt.open_training_folder(1, 1, target='loras', family='krea')
        assert opened == [path]
        assert path == lt._lora_dest_dir(ds, 'krea')
        assert path.startswith(os.path.normpath(str(external / 'loras')))


def test_settings_preview_names_the_folder_deploys_use(app, tmp_path):
    """The overrides panel promises "the folder the app uses while this is empty".
    Once deploys follow the yaml, so must that line."""
    from app import capabilities
    from app import config as cfg
    from app.services import comfy_model_paths as cmp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        external = tmp_path / 'external'
        (external / 'loras').mkdir(parents=True)
        _write_yaml(base, f"""
            comfyui:
              base_path: {external}
              is_default: true
              loras: loras
        """)
        info = capabilities.classify_comfyui_folders(str(base))['loras_dir']
        assert info['resolved'] == cmp.write_root('loras')
        assert info['resolved'] == os.path.normpath(str(external / 'loras'))
        assert info['source'] == 'extra_paths'
        # the other three keep their historical shape
        assert capabilities.classify_comfyui_folders(str(base))['models_dir']['source'] == 'derived'


def test_settings_preview_unchanged_without_a_yaml(app, tmp_path):
    from app import capabilities
    from app import config as cfg
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        info = capabilities.classify_comfyui_folders(str(base))['loras_dir']
        assert info['source'] == 'derived'
        assert info['resolved'] == str(base / 'models' / 'loras')


# --- what was deployed BEFORE the fix stays reachable -----------------------

def test_legacy_deployment_stays_listed_and_resolvable(app, tmp_path, monkeypatch):
    """The dangerous half of a path fix: files already deployed under the old
    default root must keep showing up in "IN COMFYUI", keep resolving to their
    real path, and stay deletable."""
    from app import config as cfg
    from app.services import lora_test_studio as lts
    from app.services import lora_training as lt
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        external = tmp_path / 'external'
        (external / 'loras').mkdir(parents=True)
        _write_yaml(base, f"""
            comfyui:
              base_path: {external}
              is_default: true
              loras: loras
        """)
        ds = _ds()
        monkeypatch.setattr(lt.fds, 'get_dataset', lambda u, d: ds)
        legacy = _lora_file(os.path.join(str(base / 'models' / 'loras'),
                                         'z image', 'lora_tata_old.safetensors'))
        fresh = _lora_file(os.path.join(os.path.normpath(str(external / 'loras')),
                                        'z image', 'lora_tata_new.safetensors'))
        names = {c['filename'] for c in lt.list_imported_checkpoints(1, 1)}
        assert names == {os.path.join('z image', 'lora_tata_old.safetensors'),
                         os.path.join('z image', 'lora_tata_new.safetensors')}
        # …and each resolves to the file that really exists, in its own root.
        assert os.path.normpath(lts._resolve_lora_abs_path(
            'z image/lora_tata_old.safetensors')) == os.path.normpath(legacy)
        assert os.path.normpath(lts._resolve_lora_abs_path(
            'z image/lora_tata_new.safetensors')) == os.path.normpath(fresh)
        # deleting the legacy one finds it where it is, instead of "file not found"
        lt.delete_imported_checkpoint(1, 1, os.path.join('z image', 'lora_tata_old.safetensors'))
        assert not os.path.exists(legacy)
