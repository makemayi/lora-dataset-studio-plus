"""SeedVR2 — the fidelity upscaler (issue #32, requested by SurpassHR).

Rebuilt 2026-08-16 on the user's own verified workflow
(`utility_seedvr2_7b_int8_upscale_image.json`, shipped as
`workflows/seedvr2 7b int8 upscale.json`): core UNETLoader/VAELoader + the
manual SeedVR2 nodes + the core tiled VAE + one-step KSampler, with the DiT
resolving from `diffusion_models` (the 7B Sharp int8 build the ✨ improve
'klein' lane already requires) and the VAE from `vae`.

What these tests pin is what would silently ruin the feature:

* the NODE CLASS NAMES of the shipped workflow — the pack's README spells them
  differently from its code, and every preflight lies if they drift;
* the WORKFLOW FILE contract: a re-export that renumbers or drops a node this
  lane fills must fail loudly, not silently fill the wrong node;
* the DESTINATION: the VAE download must land in `models/vae` (a core
  VAELoader reads that folder — comfy-loader-folder-rule), and the DiT is NOT
  downloaded at all (the 7B Sharp int8 build has no public URL);
* the RESOLVERS never guessing: a wrong file is worse than a missing one;
* the in-place replace route (tile overwritten, original kept for undo) sharing
  the face swap's pending/restore shape;
* the improve lane staying engine-agnostic ABOVE the dispatch, and the stored
  ids (`derivation_kind`, `action`) not moving when the engine does.
"""
import io
import os
import struct

import pytest
from PIL import Image


def _make_comfyui(root):
    base = root / 'ComfyUI'
    (base / 'models').mkdir(parents=True, exist_ok=True)
    (base / 'input').mkdir(parents=True, exist_ok=True)
    (base / 'output').mkdir(parents=True, exist_ok=True)
    (base / 'main.py').write_text('# fake ComfyUI entrypoint', encoding='utf-8')
    return base


def _safetensors(payload_size=1024):
    header = b'{"__metadata__":{"lds":"test"}}'
    return struct.pack('<Q', len(header)) + header + b'\0' * payload_size


def _png(color=(10, 200, 10)):
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _install(base, *relparts, data=None):
    p = base.joinpath(*relparts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if data is not None else _safetensors())
    return p


def _install_weights(base):
    """The two assets where the shipped pipeline reads them: `diffusion_models`
    for the DiT (the improve lane's canonical build), `vae` for the VAE."""
    _install(base, 'models', 'diffusion_models',
             'seedvr2_7b_sharp_int8_convrot.safetensors')
    _install(base, 'models', 'vae', 'ema_vae_fp16.safetensors')


def _configure(app, tmp_path):
    from app import config
    from app.services import comfy_model_paths
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
    return base


# --- The contract with the shipped workflow ---------------------------------

def test_node_class_names_are_the_manual_pipeline_not_the_one_box_nodes():
    """Since 2026-08-16 the lane runs the manual pipeline, not the ONE-BOX
    nodes (SeedVR2LoadDiTModel / SeedVR2VideoUpscaler) — the whole point of the
    rebuild is that a core UNETLoader can read the improve lane's build."""
    from app.services import seedvr2_helper as svr
    assert svr.SEEDVR2_NODE_CLASSES == ('SeedVR2Preprocess',
                                        'SeedVR2Conditioning',
                                        'SeedVR2PostProcessing')


def test_build_workflow_loads_the_shipped_file_and_fills_the_seven_values(app, tmp_path):
    """The graph is the USER'S file, filled in place — not a hand-built twin
    that could drift. Every value this lane owns lands on the right node."""
    from app.services import seedvr2_helper as svr
    from app.services.seedvr2_helper import WORKFLOW_PATH
    assert WORKFLOW_PATH.exists(), 'shipped workflow file missing'
    with app.app_context():
        wf = svr.build_workflow('s.png', dit='d', vae='v', seed=7, resolution=2.0,
                                color_correct='wavelet', filename_prefix='p')
    # The debug before/after viewer never ships in a job.
    assert svr.NODE_IMAGE_COMPARE not in wf
    assert wf[svr.NODE_LOAD_IMAGE]['inputs']['image'] == 's.png'
    assert wf[svr.NODE_UNET]['inputs']['unet_name'] == 'd'
    assert wf[svr.NODE_VAE]['inputs']['vae_name'] == 'v'
    assert wf[svr.NODE_RESIZE]['inputs']['resize_type'] == 'scale by multiplier'
    assert wf[svr.NODE_RESIZE]['inputs']['resize_type.multiplier'] == 2.0
    assert wf[svr.NODE_SAMPLER]['inputs']['seed'] == 7
    assert wf[svr.NODE_SAMPLER]['inputs']['steps'] == 1      # one-step distill
    assert wf[svr.NODE_POST]['inputs']['color_correction_method'] == 'wavelet'
    assert wf[svr.NODE_SAVE]['inputs']['filename_prefix'] == 'p'


def test_build_workflow_raises_when_a_filled_node_is_missing(app, monkeypatch, tmp_path):
    """A re-export that renumbers a node this lane fills must fail loudly."""
    from app.services import seedvr2_helper as svr
    raw = {'1': {'class_type': 'LoadImage', 'inputs': {}}}
    monkeypatch.setattr(svr, 'load_workflow_local', lambda *a, **k: raw)
    with app.app_context(), pytest.raises(
            ValueError, match='seedvr2 7b int8 upscale.json has changed'):
        svr.build_workflow('s.png', dit='d', vae='v')


def test_build_workflow_is_pure(app, monkeypatch):
    from app.services import seedvr2_helper as svr
    monkeypatch.setattr(svr.cfg, 'get', lambda *a, **k: pytest.fail('config read'))
    with app.app_context():
        svr.build_workflow('s.png', dit='d', vae='v', seed=1)


# --- Resolution never guesses ------------------------------------------------

def test_resolvers_read_the_improve_lanes_folders(app, tmp_path):
    """DiT from `diffusion_models`, VAE from `vae` — the folders a core
    UNETLoader/VAELoader list (comfy-loader-folder-rule). A build sitting in the
    pack-private `SEEDVR2` folder is INVISIBLE to this pipeline."""
    from app.services import seedvr2_helper as svr
    from app.services import comfy_model_paths
    base = _configure(app, tmp_path)
    _install_weights(base)
    comfy_model_paths.clear_cache()
    assert svr.resolve_seedvr2_dit() == svr.SEEDVR2_UNET_CANONICAL
    assert svr.resolve_seedvr2_vae() == svr.SEEDVR2_VAE_CANONICAL
    assert svr.seedvr2_missing_assets() == []


def test_resolvers_fall_back_to_a_narrow_token_match(app, tmp_path):
    """Not the canonical name — a renamed build still resolves, but only one
    whose basename says SeedVR2 7B sharp / ema vae."""
    from app.services import seedvr2_helper as svr
    from app.services import comfy_model_paths
    base = _configure(app, tmp_path)
    _install(base, 'models', 'diffusion_models', 'my_seedvr2_7b_sharp_v3.safetensors')
    _install(base, 'models', 'vae', 'my_ema_vae_v2.safetensors')
    comfy_model_paths.clear_cache()
    assert svr.resolve_seedvr2_dit() == 'my_seedvr2_7b_sharp_v3.safetensors'
    assert svr.resolve_seedvr2_vae() == 'my_ema_vae_v2.safetensors'


def test_a_wrong_file_is_missing_not_a_guess(app, tmp_path):
    """A diffusion_models folder full of OTHER models must resolve to None, not
    to the first file — handing a non-SeedVR2 DiT to the one-step sampler dies
    inside the node with an unreadable error."""
    from app.services import seedvr2_helper as svr
    from app.services import comfy_model_paths
    base = _configure(app, tmp_path)
    _install(base, 'models', 'diffusion_models', 'flux2-klein-9b-fp8.safetensors')
    _install(base, 'models', 'vae', 'flux2-vae.safetensors')
    comfy_model_paths.clear_cache()
    assert svr.resolve_seedvr2_dit() is None
    assert svr.resolve_seedvr2_vae() is None
    assert sorted(svr.seedvr2_missing_assets()) == ['seedvr2_model', 'seedvr2_vae']


def test_an_html_gate_page_saved_as_weights_is_reported_invalid(app, tmp_path):
    """An interrupted/proxied download can save an HTML error page under the
    right name: it passes 'the file is there' and dies inside the node."""
    from app.services import seedvr2_helper as svr
    from app.services import comfy_model_paths
    base = _configure(app, tmp_path)
    _install(base, 'models', 'diffusion_models',
             'seedvr2_7b_sharp_int8_convrot.safetensors',
             data=b'<html>Access Denied</html>')
    _install_weights(base)
    comfy_model_paths.clear_cache()
    invalid = svr.seedvr2_invalid_assets()
    assert any(a['asset'] == 'seedvr2_model' for a in invalid)


# --- Preflight / node probe --------------------------------------------------

def test_preflight_raises_before_anything_is_queued(app, tmp_path, monkeypatch):
    from app import config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    monkeypatch.setattr(svr, 'seedvr2_missing_nodes', lambda: [])
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        with pytest.raises(svr.SeedVR2ModelsMissing) as exc:
            svr.preflight()
        assert sorted(exc.value.missing) == ['seedvr2_model', 'seedvr2_vae']
        assert exc.value.missing_nodes == []


def test_missing_nodes_derives_from_the_shipped_workflow(app, monkeypatch):
    """Every class the workflow needs minus what /object_info exposes — so a new
    node in the workflow is demanded automatically instead of listed by hand."""
    from app.services import seedvr2_helper as svr
    svr.clear_nodes_cache()
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                        lambda: {'LoadImage', 'VAELoader', 'UNETLoader',
                                 'KSampler', 'VAEEncodeTiled', 'VAEDecodeTiled'})
    with app.app_context():
        missing = svr.seedvr2_missing_nodes()
    svr.clear_nodes_cache()
    assert 'SeedVR2Preprocess' in missing
    assert 'SeedVR2Conditioning' in missing
    assert 'ResizeImageMaskNode' in missing


def test_missing_nodes_fails_open_when_comfyui_is_unreachable(app, monkeypatch):
    from app.services import seedvr2_helper as svr
    svr.clear_nodes_cache()
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes', lambda: None)
    with app.app_context():
        assert svr.seedvr2_missing_nodes() == []
    svr.clear_nodes_cache()


def test_node_hints_only_guess_the_pack_for_the_manual_nodes(app):
    """SeedVR2Preprocess et al map to the numz pack; ResizeImageMaskNode and
    friends are named WITHOUT a guessed pack, same as krea_hq_helper."""
    from app.services import seedvr2_helper as svr
    hints = {h['class_type']: h for h in
             svr.seedvr2_node_hints(['SeedVR2Preprocess', 'ResizeImageMaskNode'])}
    assert hints['SeedVR2Preprocess']['pack'] == 'ComfyUI-SeedVR2_VideoUpscaler'
    assert hints['ResizeImageMaskNode']['pack'] is None


def test_a_pack_installed_under_another_folder_name_still_counts(app, tmp_path):
    from app import config
    from app.services import seedvr2_helper as svr
    base = _make_comfyui(tmp_path)
    pack = base / 'custom_nodes' / 'seedvr2_videoupscaler'
    pack.mkdir(parents=True)
    (pack / '__init__.py').write_text('', encoding='utf-8')
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        assert svr.seedvr2_node_pack_installed() is True


# --- Settings ----------------------------------------------------------------

def test_the_multiplier_defaults_to_two_and_clamps(app):
    """2x is the value of the user's verified workflow; 1.0-4.0 is the range the
    ResizeImageMaskNode multiplier is allowed to be."""
    from app import config
    from app.services import seedvr2_helper as svr
    with app.app_context():
        assert svr.target_resolution() == 2.0
        for bad, expected in (('1080', 4.0), (0.1, 1.0), ('junk', 2.0)):
            config.save_config({'seedvr2': {'resolution': bad}})
            assert svr.target_resolution() == expected
        config.save_config({'seedvr2': {'resolution': 2.5}})
        assert svr.target_resolution() == 2.5


def test_color_correction_never_passes_an_invalid_enum(app):
    from app import config
    from app.services import seedvr2_helper as svr
    with app.app_context():
        config.save_config({'seedvr2': {'color_correction': 'wavelet'}})
        assert svr.color_correction() == 'wavelet'
        config.save_config({'seedvr2': {'color_correction': 'nope'}})
        assert svr.color_correction() == 'lab'


# --- Setup: the VAE is downloadable, the DiT is not --------------------------

def test_only_the_vae_is_a_download_action_and_it_lands_in_models_vae(app, tmp_path):
    """The 7B Sharp int8 DiT has no public URL (community re-quantisation, same
    as the improve lane's); the VAE is fetchable and must land in `vae` — the
    folder a core VAELoader reads."""
    from app import setup_installer, config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        assert 'seedvr2_model' not in setup_installer.INSTALL_ACTIONS
        assert 'seedvr2_vae' in setup_installer.INSTALL_ACTIONS
        dest = setup_installer._download_dest_path('seedvr2_vae')
        assert dest.endswith(os.path.join('models', 'vae', 'ema_vae_fp16.safetensors'))
        assert setup_installer._INSTALL_GROUPS['seedvr2'] == ('seedvr2_vae',)
        # The DiT missing is answered with its exact path, never a download.
        assert svr.SEEDVR2_ASSETS['seedvr2_model']['source'].startswith('no public URL')


# --- In-place replace (tile overwritten, original kept for undo) -------------

def _dataset_with_image(svc, tmp_path):
    from app.config import LOCAL_USER
    ds = svc.create_dataset(LOCAL_USER, 'SeedVR2', 'sv2')
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'tile.png'), 'wb') as fh:
        fh.write(_png((10, 200, 10)))
    img = svc.FaceDatasetImage(dataset_id=ds.id, source='generated',
                               status='finished', filename='tile.png',
                               variation_prompt='p', variation_label='x')
    svc.db.session.add(img)
    svc.db.session.commit()
    return ds, img


def test_seedvr2_replace_transitions_the_row_like_a_swap(app, tmp_path, monkeypatch):
    """The tile's CURRENT image goes through the pipeline and the result
    REPLACES it: row set pending, filename cleared, snapshot held for undo —
    the same shape as a face swap, so Stop/failure restore and success trashes
    the original via the existing completion link."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.services import seedvr2_helper as svr
    from app.services import comfy_model_paths
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    base = _make_comfyui(tmp_path)
    _install_weights(base)
    with app.app_context():
        cfg.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        ds, img = _dataset_with_image(svc, tmp_path)
        img_id, ds_id = img.id, ds.id
        captured = {}
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.update(kw), kw['job_id'])[1])
        job_id = svr_seedvr2_replace(svc, LOCAL_USER, img_id)
        assert job_id and job_id == captured['job_id']
        row = svc.db.session.get(svc.FaceDatasetImage, img_id)
        assert row.status == 'pending'
        assert row.filename is None
        assert row.job_id == job_id
        # The snapshot is held for undo, provenance untouched.
        assert row.swap_restore is not None
        assert row.variation_prompt == 'p'
        assert captured['metadata']['replace_kind'] == 'seedvr2_upscale'
        assert captured['metadata']['dataset_id'] == ds_id
        # The old FILE stays on disk while the job runs.
        assert os.path.exists(os.path.join(svc._dataset_dir(ds_id), 'tile.png'))


def svr_seedvr2_replace(svc, user_id, image_id):
    from app.services import dataset_generation_service as dgs
    return dgs.seedvr2_upscale_replace(user_id, image_id)


def test_seedvr2_replace_route_starts_a_job(app, client, tmp_path):
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.services import comfy_model_paths
    from app.job_queue import queue_manager
    base = _make_comfyui(tmp_path)
    _install_weights(base)
    with app.app_context():
        cfg.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        ds, img = _dataset_with_image(svc, tmp_path)
        img_id = img.id
    import unittest.mock as mock
    with mock.patch.object(queue_manager, 'add_job', lambda **kw: kw['job_id']):
        resp = client.post(f'/api/dataset/image/{img_id}/seedvr2-replace')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True and body['job_id']


# --- Improve lane stays engine-agnostic --------------------------------------

def test_improve_engine_resolution_falls_back_instead_of_raising(app):
    from app import config
    from app.services import face_dataset_service as svc
    with app.app_context():
        assert svc.resolve_improve_engine() == 'klein'
        assert svc.resolve_improve_engine('seedvr2') == 'seedvr2'
        assert svc.resolve_improve_engine('nonsense') == 'klein'
        config.save_config({'improve': {'engine': 'seedvr2'}})
        assert svc.resolve_improve_engine() == 'seedvr2'


def test_the_candidates_stored_ids_do_not_move_with_the_engine(app):
    from app.services import face_dataset_service as svc

    class _Src:
        id, dataset_id, source_metadata = 7, 3, None

    for engine in svc.IMPROVE_ENGINES:
        meta = svc._improve_extra_metadata(_Src(), 'label', engine=engine)
        assert meta['derivation_kind'] == 'klein_image_improve'
        assert meta['action'] == 'upscale_improve'
        assert meta['improve_engine'] == engine


# --- Capabilities / endpoints ------------------------------------------------

def test_capabilities_publishes_gaps_without_a_tiling_lane(app, tmp_path, monkeypatch):
    from app import capabilities, config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    monkeypatch.setattr(svr, 'seedvr2_missing_nodes', lambda: [])
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        caps = capabilities.probe(force=True)['comfyui']
    for key in ('seedvr2_missing', 'seedvr2_nodes_missing',
                'seedvr2_nodes_installed', 'seedvr2_invalid', 'seedvr2_ready'):
        assert key in caps
    # The TTP lane is gone — no tile/lane/ceiling keys any more.
    for dead in ('seedvr2_tiling_ready', 'seedvr2_tiling_nodes_missing',
                 'seedvr2_ceiling_mp'):
        assert dead not in caps
    assert sorted(caps['seedvr2_missing']) == ['seedvr2_model', 'seedvr2_vae']
    assert caps['seedvr2_ready'] is False


def test_the_models_endpoint_reports_what_the_loaders_will_read(app, client, tmp_path):
    from app import config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    _install_weights(base)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
    body = client.get('/api/seedvr2/models').get_json()
    assert body['dit'] == svr.SEEDVR2_UNET_CANONICAL
    assert body['vae'] == svr.SEEDVR2_VAE_CANONICAL
    assert body['missing'] == []
