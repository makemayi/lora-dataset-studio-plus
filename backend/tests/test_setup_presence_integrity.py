"""A file that is PRESENT but cannot be loaded is not "already installed" — through
EVERY door, not just the one that was reported.

54e5011 fixed the ``dest`` door: Setup used to see the 9.5 GB truncated Klein UNET,
say "already present", and hand the user back the same dead end they had opened
Setup to escape. Three more doors skip the very same download for the very same
reason — because some OTHER file resolves:

  * ``_variant_already_present``     — an earlier default filename in the app's own tree
  * ``_download_present_in_extra``   — a copy under an extra_model_paths.yaml root
  * ``_krea_asset_already_installed``— a hand-placed Krea asset the resolver finds

None of them looked at the file they were vouching for. These tests are the
scenario, once per door: a corrupted file present, and the download must NOT be
skipped. They fail on the pre-fix code.

Also pins the second half of the same "green without checking" family:
``watermark_klein.is_available()`` used to be laxer than the badge that gates the
button (``caps.watermark_klein``), so the two could disagree about one engine.
"""
import json
import os
import struct

import pytest


# --- Fixtures on disk --------------------------------------------------------

def _valid_weights(path):
    """A structurally valid safetensors container (8-byte LE header length + JSON)."""
    meta = {'w': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]}}
    body = json.dumps(meta).encode('utf-8')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(body)) + body + b'\x00' * 64)
    return path


def _gate_page(path):
    """The real-world corruption: a licence/login HTML page saved under the model's
    name. Loads nowhere, resolves everywhere."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(b'<!doctype html>\n<html><head><title>Access gated</title></head></html>')
    return path


def _comfy_base(tmp_path, cfg):
    base = tmp_path / 'ComfyUI'
    (base / 'models').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


@pytest.fixture(autouse=True)
def _clear_caches():
    from app.services import comfy_model_paths as cmp, model_integrity as mi
    cmp.clear_cache()
    mi.clear_cache()
    yield
    cmp.clear_cache()
    mi.clear_cache()


# --- Door 2: an earlier default filename in the app's own tree ---------------

def test_legacy_variant_that_cannot_load_is_not_already_installed(app, tmp_path):
    """The pre-KV Klein UNET is on disk under its old name, and it is an HTML gate
    page. It resolves by name exactly like a good one, so presence alone said
    "skip the download" — and the engine stayed dark with no way out from inside
    the app."""
    from app import config as cfg, setup_installer as si
    with app.app_context():
        _comfy_base(tmp_path, cfg)
        dest = si._download_dest_path('klein_model')
        legacy = os.path.join(os.path.dirname(dest),
                              si._MODEL_DOWNLOADS['klein_model']['legacy_names'][0])
        _gate_page(legacy)
        assert si._variant_already_present('klein_model') is None
        # Same tree the app installs into, and the resolver may prefer this name
        # over the fresh download -> it is cleared, exactly as at `dest`.
        assert not os.path.exists(legacy)


def test_legacy_variant_that_loads_still_counts_as_installed(app, tmp_path):
    """The no-break half: a REAL earlier build must still suppress a ~10 GB refetch."""
    from app import config as cfg, setup_installer as si
    with app.app_context():
        _comfy_base(tmp_path, cfg)
        dest = si._download_dest_path('klein_model')
        name = si._MODEL_DOWNLOADS['klein_model']['legacy_names'][0]
        legacy = _valid_weights(os.path.join(os.path.dirname(dest), name))
        assert si._variant_already_present('klein_model') == name
        assert os.path.exists(legacy)


# --- Door 3: a copy under an extra_model_paths.yaml root ---------------------

def _yaml(base, text):
    (base / 'extra_model_paths.yaml').write_text(text, encoding='utf-8')


def test_extra_root_copy_that_cannot_load_does_not_skip_the_download(app, tmp_path):
    from app import config as cfg, setup_installer as si
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        ext = tmp_path / 'ext'
        _gate_page(str(ext / 'vae' / 'flux2-vae.safetensors'))
        _yaml(base, f'comfyui:\n  vae: {ext / "vae"}\n')
        assert si._download_present_in_extra('klein_vae') is False
        # NOT deleted: this tree belongs to the user, and the download lands in the
        # app's own dest regardless.
        assert os.path.exists(str(ext / 'vae' / 'flux2-vae.safetensors'))


def test_extra_root_copy_that_loads_still_skips_the_download(app, tmp_path):
    from app import config as cfg, setup_installer as si
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        ext = tmp_path / 'ext'
        _valid_weights(str(ext / 'vae' / 'flux2-vae.safetensors'))
        _yaml(base, f'comfyui:\n  vae: {ext / "vae"}\n')
        assert si._download_present_in_extra('klein_vae') is True


# --- Door 4: a hand-placed Krea asset the resolver finds --------------------

def test_krea_asset_that_cannot_load_is_not_already_installed(app, monkeypatch):
    """The Krea door asks the engine's own resolver ("is this installed?"), which
    answers presence. The resolver's integrity list has to veto it, or the hand-
    placed HTML page keeps certifying itself."""
    from app import setup_installer as si
    from app.services import krea_edit_helper as keh
    action = next(iter(si._KREA_DOWNLOADS))
    monkeypatch.setattr(keh, 'krea_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'krea_invalid_assets', lambda: [
        {'asset': action, 'filename': 'x.safetensors', 'verdict': 'html_or_text',
         'blocking': True, 'reason': 'not a real model'}])
    with app.app_context():
        assert si._krea_asset_already_installed(action) is False


def test_krea_asset_only_advisory_small_still_counts_as_installed(app, monkeypatch):
    """`too_small` is advisory — a small-but-loadable file is the user's, not ours,
    and re-downloading over it would be the app overruling a deliberate choice."""
    from app import setup_installer as si
    from app.services import krea_edit_helper as keh
    action = next(iter(si._KREA_DOWNLOADS))
    monkeypatch.setattr(keh, 'krea_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'krea_invalid_assets', lambda: [
        {'asset': action, 'filename': 'x.safetensors', 'verdict': 'too_small',
         'blocking': False, 'reason': 'suspiciously small'}])
    with app.app_context():
        assert si._krea_asset_already_installed(action) is True


# --- The watermark cleaner reads the SAME verdict as the badge --------------

def test_watermark_klein_availability_is_the_capability_verdict(app, monkeypatch):
    """`caps.watermark_klein` (= klein_engine_ready) gates the button; is_available()
    decides the silent LaMa fallback and words the refusal. A laxer copy of the
    second is how one engine gets two sincere, contradictory answers."""
    from app.services import watermark_klein as wk, klein_edit_helper as keh
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_comfyui', lambda *a, **k: {'ok': True})
    calls = {}

    def _ready(comfy_ok, **kw):
        calls['comfy_ok'] = comfy_ok
        return False

    monkeypatch.setattr(keh, 'klein_engine_ready', _ready)
    with app.app_context():
        assert wk.is_available() is False
    assert calls['comfy_ok'] is True


def test_klein_engine_ready_refuses_a_present_but_unloadable_required_asset():
    from app.services import klein_edit_helper as keh
    broken = [{'asset': 'klein_model', 'filename': 'u.safetensors',
               'verdict': 'truncated_or_garbage', 'blocking': True, 'reason': 'truncated'}]
    assert keh.klein_engine_ready(True, missing=[], invalid=broken,
                                  unsupported_enums=[]) is False
    # ...and an advisory-small one does not gate.
    small = [dict(broken[0], verdict='too_small', blocking=False)]
    assert keh.klein_engine_ready(True, missing=[], invalid=small,
                                  unsupported_enums=[]) is True


def test_klein_engine_ready_refuses_an_unsupported_widget_value():
    from app.services import klein_edit_helper as keh
    gap = [{'node_id': '9', 'class_type': 'KSampler', 'input': 'scheduler',
            'value': 'beta57', 'pack': 'RES4LYF', 'url': None}]
    assert keh.klein_engine_ready(True, missing=[], invalid=[],
                                  unsupported_enums=gap) is False
    assert keh.klein_engine_ready(False, missing=[], invalid=[],
                                  unsupported_enums=[]) is False
    assert keh.klein_engine_ready(True, missing=['klein_vae'], invalid=[],
                                  unsupported_enums=[]) is False
