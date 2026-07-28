"""Replacing a corrupted weight must never leave the user with LESS than they had.

The "four doors" pass taught Setup to DELETE a present-but-unloadable file so the
re-download it suggests is no longer a no-op. It deleted it too early: the removal
happened before the request was even opened, so a 401, an expired token, a
re-gated repo or a dead host turned "you have a broken file" into "you have no
file at all" — and the broken file was at least a 9.5 GB head start and a name
the user could replace by hand.

The order is now: open the request, check the HTTP status, stream to `.part`,
and only then let the fresh copy take the place of the old one. Nothing is
removed until something better exists.
"""
import json
import os
import struct

import pytest


def _valid_weights(path, body=b'\x00' * 64):
    meta = {'w': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]}}
    head = json.dumps(meta).encode('utf-8')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(head)) + head + body)
    return path


def _gate_page(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(b'<!doctype html>\n<html><head><title>Access gated</title></head></html>')
    return path


class _Resp:
    """Just enough of requests' streaming response for _run_model_download."""

    def __init__(self, status, payload=b''):
        self.status_code = status
        self.headers = {'content-length': str(len(payload))}
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size=1):
        yield self._payload


@pytest.fixture(autouse=True)
def _clear_caches():
    from app.services import comfy_model_paths as cmp, model_integrity as mi
    cmp.clear_cache()
    mi.clear_cache()
    yield
    cmp.clear_cache()
    mi.clear_cache()


def _comfy_base(tmp_path, cfg):
    base = tmp_path / 'ComfyUI'
    (base / 'models').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


def _no_other_door(monkeypatch, si):
    """Neutralise the three OTHER skip doors so each test exercises `dest` only."""
    monkeypatch.setattr(si, '_variant_already_present', lambda action, condemned=None: None)
    monkeypatch.setattr(si, '_download_present_in_extra', lambda action: False)
    monkeypatch.setattr(si, '_krea_asset_already_installed', lambda action: False)


def test_a_denied_download_leaves_the_broken_file_in_place(app, tmp_path, monkeypatch):
    """The regression this file exists for: HTTP 401 and the old file is GONE."""
    from app import config as cfg, setup_installer as si
    with app.app_context():
        _comfy_base(tmp_path, cfg)
        dest = si._download_dest_path('klein_model')
        _gate_page(dest)
        before = open(dest, 'rb').read()
        _no_other_door(monkeypatch, si)
        si._runs['klein_model'] = si._new_run()
        monkeypatch.setattr(si.requests, 'get', lambda *a, **k: _Resp(401))

        assert si._run_model_download('klein_model') == 1
        assert os.path.exists(dest), 'the unusable file was deleted before the download proved it could be replaced'
        assert open(dest, 'rb').read() == before
        assert not os.path.exists(dest + '.part')


def test_a_server_error_also_leaves_the_broken_file_in_place(app, tmp_path, monkeypatch):
    """Not just auth: any >=400 status is a download that never happened."""
    from app import config as cfg, setup_installer as si
    with app.app_context():
        _comfy_base(tmp_path, cfg)
        dest = si._download_dest_path('klein_model')
        _gate_page(dest)
        _no_other_door(monkeypatch, si)
        si._runs['klein_model'] = si._new_run()
        monkeypatch.setattr(si.requests, 'get', lambda *a, **k: _Resp(503))

        assert si._run_model_download('klein_model') == 1
        assert os.path.exists(dest)


def test_a_network_error_mid_stream_leaves_the_broken_file_in_place(app, tmp_path, monkeypatch):
    """The status was fine and the host died halfway — still no reason to have
    nothing: the fresh bytes were going to a `.part` file."""
    from app import config as cfg, setup_installer as si
    with app.app_context():
        _comfy_base(tmp_path, cfg)
        dest = si._download_dest_path('klein_model')
        _gate_page(dest)
        _no_other_door(monkeypatch, si)
        si._runs['klein_model'] = si._new_run()

        class _Dies(_Resp):
            def iter_content(self, chunk_size=1):
                yield b'\x00' * 8
                raise si.requests.ConnectionError('host went away')

        monkeypatch.setattr(si.requests, 'get', lambda *a, **k: _Dies(200, b'x' * 4096))

        assert si._run_model_download('klein_model') == 1
        assert os.path.exists(dest)
        assert not os.path.exists(dest + '.part')


def test_a_successful_download_still_replaces_the_broken_file(app, tmp_path, monkeypatch):
    """The no-break half: when the download DOES arrive, the corrupted file is gone
    and the fresh weights are at `dest`. Deleting late must not mean never."""
    from app import config as cfg, setup_installer as si
    with app.app_context():
        _comfy_base(tmp_path, cfg)
        dest = si._download_dest_path('klein_model')
        _gate_page(dest)
        good = _valid_weights(str(tmp_path / 'src.safetensors'))
        payload = open(good, 'rb').read()
        _no_other_door(monkeypatch, si)
        si._runs['klein_model'] = si._new_run()
        monkeypatch.setattr(si.requests, 'get', lambda *a, **k: _Resp(200, payload))

        assert si._run_model_download('klein_model') == 0
        assert open(dest, 'rb').read() == payload


def test_a_loadable_file_is_still_never_re_downloaded(app, tmp_path, monkeypatch):
    """And a good file still short-circuits: no request is opened at all."""
    from app import config as cfg, setup_installer as si
    with app.app_context():
        _comfy_base(tmp_path, cfg)
        dest = si._download_dest_path('klein_model')
        _valid_weights(dest)
        _no_other_door(monkeypatch, si)
        si._runs['klein_model'] = si._new_run()

        def _boom(*a, **k):
            raise AssertionError('a present, loadable weight must not be re-fetched')

        monkeypatch.setattr(si.requests, 'get', _boom)
        assert si._run_model_download('klein_model') == 0


# --- the legacy-name door has the same ordering rule -------------------------

def test_an_unloadable_legacy_variant_is_only_removed_once_the_new_copy_landed(
        app, tmp_path, monkeypatch):
    """`_variant_already_present` deletes a legacy-named file for a real reason (the
    resolver may prefer that name over the fresh download), but it is a SEPARATE
    path — `os.replace` will not overwrite it — so it is the one deletion that has
    to be carried out by hand. It now happens after the download succeeded."""
    from app import config as cfg, setup_installer as si
    with app.app_context():
        _comfy_base(tmp_path, cfg)
        dest = si._download_dest_path('klein_model')
        legacy = os.path.join(os.path.dirname(dest),
                              si._MODEL_DOWNLOADS['klein_model']['legacy_names'][0])
        _gate_page(legacy)
        monkeypatch.setattr(si, '_download_present_in_extra', lambda action: False)
        monkeypatch.setattr(si, '_krea_asset_already_installed', lambda action: False)
        si._runs['klein_model'] = si._new_run()
        monkeypatch.setattr(si.requests, 'get', lambda *a, **k: _Resp(403))

        assert si._run_model_download('klein_model') == 1
        assert os.path.exists(legacy), 'the legacy file was dropped for a download that never happened'

        payload = open(_valid_weights(str(tmp_path / 'src.safetensors')), 'rb').read()
        monkeypatch.setattr(si.requests, 'get', lambda *a, **k: _Resp(200, payload))
        assert si._run_model_download('klein_model') == 0
        assert not os.path.exists(legacy), 'a shadowing unloadable legacy name survived a successful install'
        assert open(dest, 'rb').read() == payload
