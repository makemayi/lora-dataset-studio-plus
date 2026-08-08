"""SeedVR2 — the fidelity upscaler (issue #32, requested by SurpassHR).

What these tests pin is what would silently ruin the feature:

* the NODE CLASS NAMES. The pack's README spells them `SeedVR2_VideoUpscaler`;
  its code registers `SeedVR2VideoUpscaler`. Reading the README would have made
  every preflight report a missing node on a perfectly good install, and the
  symptom ("install the pack" about a pack that is there) is the least
  debuggable one there is.
* the DESTINATION. A download that lands where the resolver never looks is a
  3.4 GB no-op — the exact failure the Krea install tests exist for.
* the RESOLVERS never guessing: handing the DiT weights to the VAE loader, or a
  name that is not on disk to a loader that DOWNLOADS unknown names, both turn a
  button into a multi-gigabyte surprise.
* the improve lane staying engine-agnostic ABOVE the dispatch, and the stored
  ids (`derivation_kind`, `action`) not moving when the engine does.
"""
import os
import struct

import pytest


def _make_comfyui(root):
    base = root / 'ComfyUI'
    (base / 'models').mkdir(parents=True, exist_ok=True)
    (base / 'main.py').write_text('# fake ComfyUI entrypoint', encoding='utf-8')
    return base


def _safetensors(payload_size=1024):
    header = b'{"__metadata__":{"lds":"test"}}'
    return struct.pack('<Q', len(header)) + header + b'\0' * payload_size


def _install_weights(base, *names):
    folder = base / 'models' / 'SEEDVR2'
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(_safetensors())
    return folder


# --- The contract with the node pack ----------------------------------------

def test_node_class_names_match_the_packs_code_not_its_readme():
    """`SeedVR2VideoUpscaler`, not `SeedVR2_VideoUpscaler`. Read from the pack's
    own define_schema on 2026-08-02. If this ever changes, every preflight lies."""
    from app.services import seedvr2_helper as svr
    assert svr.SEEDVR2_NODE_CLASSES == ('SeedVR2LoadDiTModel',
                                        'SeedVR2LoadVAEModel',
                                        'SeedVR2VideoUpscaler')
    assert all('_' not in c.removeprefix('SeedVR2')
               for c in svr.SEEDVR2_NODE_CLASSES)


def test_workflow_wires_the_three_nodes_and_pins_batch_size_to_one():
    """One image per job. The node's batch_size is a VIDEO window whose frames
    share temporal attention — grouping unrelated dataset photos would let them
    bleed into each other, which is why no batch-size setting is shipped."""
    from app.services import seedvr2_helper as svr
    g = svr.build_workflow('src.png', dit='dit.safetensors', vae='vae.safetensors',
                           seed=42, resolution=1440, max_res=2048,
                           color_correct='wavelet', swap_blocks=12,
                           filename_prefix='pfx')
    classes = {n['class_type'] for n in g.values()}
    assert classes == {'SeedVR2LoadDiTModel', 'SeedVR2LoadVAEModel',
                       'LoadImage', 'SeedVR2VideoUpscaler', 'SaveImage'}
    up = next(n for n in g.values() if n['class_type'] == 'SeedVR2VideoUpscaler')
    assert up['inputs']['batch_size'] == 1
    assert up['inputs']['resolution'] == 1440
    assert up['inputs']['max_resolution'] == 2048
    assert up['inputs']['color_correction'] == 'wavelet'
    # Every loader value is one a resolver produced, and the graph is connected.
    assert up['inputs']['image'] == ['3', 0]
    assert up['inputs']['dit'] == ['1', 0]
    assert up['inputs']['vae'] == ['2', 0]
    dit = next(n for n in g.values() if n['class_type'] == 'SeedVR2LoadDiTModel')
    assert dit['inputs']['model'] == 'dit.safetensors'
    assert dit['inputs']['blocks_to_swap'] == 12
    # Caching would keep several GB resident between jobs on a contended GPU.
    assert dit['inputs']['cache_model'] is False
    vae = next(n for n in g.values() if n['class_type'] == 'SeedVR2LoadVAEModel')
    assert vae['inputs']['cache_model'] is False
    save = next(n for n in g.values() if n['class_type'] == 'SaveImage')
    assert save['inputs']['filename_prefix'] == 'pfx'


def test_build_workflow_is_pure(app, tmp_path, monkeypatch):
    """No config read, no disk access — a graph assertion must not need a
    ComfyUI, and a settings change must not silently rewrite an enqueued job."""
    from app.services import seedvr2_helper as svr
    monkeypatch.setattr(svr.cfg, 'get', lambda *a, **k: pytest.fail('config read'))
    svr.build_workflow('s.png', dit='d', vae='v', seed=1)


# --- Resolution never guesses ------------------------------------------------

def test_weights_land_where_the_resolvers_actually_look(app, tmp_path):
    from app import setup_installer, config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        for action in ('seedvr2_model', 'seedvr2_vae'):
            dest = setup_installer._download_dest_path(action)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, 'wb') as fh:
                fh.write(_safetensors())
        assert svr.resolve_seedvr2_dit() == svr.CANONICAL_DIT
        assert svr.resolve_seedvr2_vae() == svr.CANONICAL_VAE
        assert svr.seedvr2_missing_assets() == []


def test_download_dest_is_the_folder_the_node_pack_registers(app, tmp_path):
    """`models/SEEDVR2` — SEEDVR2_FOLDER_NAME in the pack's constants.py, and the
    same string the helper searches. A one-word mistake with a 3.4 GB cost."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        dest = setup_installer._download_dest_path('seedvr2_model')
    assert dest.endswith(os.path.join('models', 'SEEDVR2',
                                      'seedvr2_ema_3b_fp8_e4m3fn.safetensors'))


def test_the_vae_is_never_offered_as_a_dit_build(app, tmp_path):
    """Both files live in ONE folder. Handing the VAE to the DiT loader fails deep
    inside the node with an unreadable error, so the picker must not list it."""
    from app import config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    _install_weights(base, svr.CANONICAL_DIT, svr.CANONICAL_VAE)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        assert svr.installed_dit_models() == [svr.CANONICAL_DIT]
        assert svr.resolve_seedvr2_vae() == svr.CANONICAL_VAE


def _enqueue_and_capture(app, tmp_path, monkeypatch, settings, source_px=(800, 1200)):
    """Run the REAL enqueue path with `settings` saved, and return the workflow
    it submitted. The point of going through enqueue rather than calling
    build_workflow directly: a setting that is read but never passed on looks
    perfectly fine in a unit test of either half."""
    from PIL import Image
    from app import config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    (base / 'input').mkdir(exist_ok=True)
    (base / 'output').mkdir(exist_ok=True)
    _install_weights(base, svr.CANONICAL_DIT, svr.CANONICAL_VAE)
    src = tmp_path / 'source.png'
    Image.new('RGB', source_px, (32, 64, 96)).save(src)

    captured = {}
    monkeypatch.setattr(svr.queue_manager, 'add_job',
                        lambda **kw: captured.update(kw) or 'job')
    monkeypatch.setattr(svr, 'seedvr2_missing_nodes', lambda: [])
    monkeypatch.setattr(svr, 'tiling_available', lambda *a, **k: True)
    monkeypatch.setattr(svr, 'full_frame_ceiling_mp', lambda *a, **k: 13.2)
    with app.app_context():
        # Start from the shipped defaults every time: save_config MERGES, so a
        # previous call's tile size would otherwise still be in the file and the
        # "default" half of these tests would silently assert nothing.
        config.save_config({'comfyui': {'base_dir': str(base)},
                            'seedvr2': dict(config.DEFAULTS['seedvr2'])})
        config.save_config(settings)
        comfy_model_paths.clear_cache()
        svr.enqueue_seedvr2_upscale('u1', 'source.png', source_path=str(src))
    return captured['workflow_data']


def test_the_tile_size_setting_reaches_the_submitted_workflow(app, tmp_path, monkeypatch):
    """THE propagation proof for issue #32's settings half. A non-default tile
    side must appear in the graph ComfyUI is actually handed — in the TTP tiler
    (what a pass holds) AND in the VAE's tiled encode/decode (the same memory
    decision, one node down)."""
    from app.services import seedvr2_helper as svr
    g = _enqueue_and_capture(app, tmp_path, monkeypatch,
                             {'seedvr2': {'resolution': 2160, 'tile_px': 768}})
    tiler = next(n for n in g.values() if n['class_type'] == 'TTP_Image_Tile_Batch')
    assert tiler['inputs']['tile_width'] == 768
    assert tiler['inputs']['tile_height'] == 768
    vae = next(n for n in g.values() if n['class_type'] == 'SeedVR2LoadVAEModel')
    assert vae['inputs']['encode_tile_size'] == 768
    assert vae['inputs']['decode_tile_size'] == 768
    assert vae['inputs']['decode_tile_overlap'] == 96
    up = next(n for n in g.values() if n['class_type'] == 'SeedVR2VideoUpscaler')
    # A tile is upscaled at the TILE's size, never the frame's.
    assert up['inputs']['resolution'] == 768
    # …and the default still submits the contributed 1024.
    d = _enqueue_and_capture(app, tmp_path, monkeypatch, {'seedvr2': {'resolution': 2160}})
    assert next(n for n in d.values()
                if n['class_type'] == 'TTP_Image_Tile_Batch')['inputs']['tile_width'] == svr.TILE_PX


def test_the_tile_size_also_reaches_the_FULL_frame_lane(app, tmp_path, monkeypatch):
    """The full-frame lane runs a tiled VAE too, so this setting saves memory
    even for someone who never installed the tiling node pack — the person most
    likely to need it. ('never' stands in for that install: same lane, and it is
    also the setting someone picks after seeing a seam.)"""
    g = _enqueue_and_capture(app, tmp_path, monkeypatch,
                             {'seedvr2': {'resolution': 1080, 'tile_px': 512,
                                          'tiling': 'never'}})
    assert not any(n['class_type'].startswith('TTP_') for n in g.values())  # full lane
    vae = next(n for n in g.values() if n['class_type'] == 'SeedVR2LoadVAEModel')
    assert vae['inputs']['encode_tile_size'] == 512
    assert vae['inputs']['encode_tile_overlap'] == 64


def test_the_tiling_threshold_setting_decides_the_lane(app, tmp_path, monkeypatch):
    """A 1080 target is left whole by default; lowering the crossover makes the
    SAME request tile. That is the lane choice moving because of a setting, not
    because of the geometry."""
    full = _enqueue_and_capture(app, tmp_path, monkeypatch,
                                {'seedvr2': {'resolution': 1080}})
    assert not any(n['class_type'].startswith('TTP_') for n in full.values())
    tiled = _enqueue_and_capture(app, tmp_path, monkeypatch,
                                 {'seedvr2': {'resolution': 1080,
                                              'tile_threshold': 640}})
    assert any(n['class_type'] == 'TTP_Image_Tile_Batch' for n in tiled.values())


def test_the_pinned_vae_reaches_the_loader_node(app, tmp_path, monkeypatch):
    from app.services import seedvr2_helper as svr
    g = _enqueue_and_capture(app, tmp_path, monkeypatch, {'seedvr2': {}})
    vae = next(n for n in g.values() if n['class_type'] == 'SeedVR2LoadVAEModel')
    assert vae['inputs']['model'] == svr.CANONICAL_VAE


def test_a_vae_named_nothing_like_one_can_be_pinned(app, tmp_path):
    """The whole reason `seedvr2.vae` exists. The automatic path finds a file
    whose NAME says vae; someone whose file is called something else had no way
    to say so, and 'seedvr2_vae' missing meant the engine simply refused to run.
    A pin is therefore matched against the whole folder, not re-filtered through
    the heuristic that failed."""
    from app import config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    _install_weights(base, svr.CANONICAL_DIT, 'seedvr2_ema_decoder.safetensors')
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        assert svr.resolve_seedvr2_vae() is None          # nothing looks like a VAE
        assert svr.seedvr2_missing_assets() == ['seedvr2_vae']
        config.save_config({'seedvr2': {'vae': 'seedvr2_ema_decoder.safetensors'}})
        assert svr.resolve_seedvr2_vae() == 'seedvr2_ema_decoder.safetensors'
        assert svr.seedvr2_missing_assets() == []
        # …and the picker still offers it, flagged for what it is, so the setting
        # is reachable from the UI and not only from config.json.
        choices = {c['file']: c['likely_vae'] for c in svr.vae_choices()}
        assert choices == {svr.CANONICAL_DIT: False,
                           'seedvr2_ema_decoder.safetensors': False}


def test_a_pinned_vae_that_is_absent_falls_back_instead_of_being_submitted(app, tmp_path):
    """Same rule as the DiT pin: the loader DOWNLOADS an unknown name, so a stale
    pin must degrade to what is on disk, never be passed through."""
    from app import config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    _install_weights(base, svr.CANONICAL_DIT, svr.CANONICAL_VAE)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)},
                            'seedvr2': {'vae': 'a_vae_from_another_install.safetensors'}})
        comfy_model_paths.clear_cache()
        assert svr.resolve_seedvr2_vae() == svr.CANONICAL_VAE
        # An explicit argument still wins over the setting, like the DiT resolver.
        assert svr.resolve_seedvr2_vae(svr.CANONICAL_VAE) == svr.CANONICAL_VAE


def test_only_an_installed_build_is_ever_submitted(app, tmp_path):
    """The loader nodes DOWNLOAD an unknown name on first use. A pin pointing at a
    build that is not on disk must therefore fall back to one that is, never be
    passed through — otherwise a dropdown starts a multi-gigabyte download from a
    button that promised an upscale."""
    from app import config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    _install_weights(base, 'seedvr2_ema_3b_fp16.safetensors', svr.CANONICAL_VAE)
    with app.app_context():
        config.save_config({
            'comfyui': {'base_dir': str(base)},
            'seedvr2': {'model': 'seedvr2_ema_7b_fp16.safetensors'}})
        comfy_model_paths.clear_cache()
        assert svr.resolve_seedvr2_dit() == 'seedvr2_ema_3b_fp16.safetensors'


def test_nothing_on_disk_resolves_to_nothing(app, tmp_path):
    from app import config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        assert svr.resolve_seedvr2_dit() is None
        assert svr.resolve_seedvr2_vae() is None
        assert sorted(svr.seedvr2_missing_assets()) == ['seedvr2_model', 'seedvr2_vae']


def test_a_pinned_build_is_honoured_when_it_is_present(app, tmp_path):
    from app import config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    _install_weights(base, svr.CANONICAL_DIT, 'seedvr2_ema_7b_fp8_e4m3fn.safetensors',
                     svr.CANONICAL_VAE)
    with app.app_context():
        config.save_config({
            'comfyui': {'base_dir': str(base)},
            'seedvr2': {'model': 'seedvr2_ema_7b_fp8_e4m3fn.safetensors'}})
        comfy_model_paths.clear_cache()
        assert svr.resolve_seedvr2_dit() == 'seedvr2_ema_7b_fp8_e4m3fn.safetensors'


def test_a_gguf_build_is_usable_here_even_though_other_engines_refuse_one(app, tmp_path):
    """comfy_model_paths.is_loadable_model excludes .gguf because the app's other
    loaders cannot read one. THIS pack ships its own GGUF loader, and on a small
    card the quantised build is the only one that fits."""
    from app import config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    _install_weights(base, svr.CANONICAL_VAE)
    (base / 'models' / 'SEEDVR2' / 'seedvr2_ema_3b-Q8_0.gguf').write_bytes(b'GGUF' * 64)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        assert svr.installed_dit_models() == ['seedvr2_ema_3b-Q8_0.gguf']
        # ...and it is NOT condemned by the safetensors header check, which a
        # valid .gguf could never satisfy. (The stub VAE trips only the ADVISORY
        # too_small floor — nothing here is blocking.)
        assert [i['asset'] for i in svr.seedvr2_invalid_assets()
                if i['blocking']] == []


def test_an_html_gate_page_saved_as_weights_is_reported_invalid(app, tmp_path):
    """Present, not loadable — the state between 'missing' and 'ready'. Same
    verdict shape as klein_invalid / krea_invalid so one banner covers all."""
    from app import config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    folder = base / 'models' / 'SEEDVR2'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / svr.CANONICAL_DIT).write_bytes(b'<!DOCTYPE html><html>login</html>' * 40)
    (folder / svr.CANONICAL_VAE).write_bytes(_safetensors())
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        bad = svr.seedvr2_invalid_assets()
        blocking = [i for i in bad if i['blocking']]
        assert [i['asset'] for i in blocking] == ['seedvr2_model']
        assert blocking[0]['verdict'] == 'html_or_text'
        assert svr.engine_ready(True, missing=[], invalid=bad, nodes_missing=[]) is False


# --- Preflight ---------------------------------------------------------------

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


def test_missing_nodes_fails_open_when_comfyui_is_unreachable(app, monkeypatch):
    """A transient probe failure must never block a pass — the job would fail
    with ComfyUI's own error, and two red flags for one cause is noise."""
    from app.services import seedvr2_helper as svr
    svr.clear_nodes_cache()
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes', lambda: None)
    with app.app_context():
        assert svr.seedvr2_missing_nodes() == []


def test_missing_nodes_lists_exactly_what_object_info_lacks(app, monkeypatch):
    from app.services import seedvr2_helper as svr
    svr.clear_nodes_cache()
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                        lambda: {'SeedVR2LoadDiTModel', 'LoadImage'})
    with app.app_context():
        assert svr.seedvr2_missing_nodes() == ['SeedVR2LoadVAEModel',
                                               'SeedVR2VideoUpscaler']
    svr.clear_nodes_cache()


def test_a_pack_installed_under_another_folder_name_still_counts(app, tmp_path):
    """ComfyUI-Manager clones the repo name; the registry installs
    `seedvr2_videoupscaler`. Both mean "installed" — the distinction this answers
    is 'install the pack' vs 'restart ComfyUI', and getting it wrong tells someone
    to install what they just installed."""
    from app import config
    from app.services import seedvr2_helper as svr
    base = _make_comfyui(tmp_path)
    pack = base / 'custom_nodes' / 'seedvr2_videoupscaler'
    pack.mkdir(parents=True)
    (pack / '__init__.py').write_text('', encoding='utf-8')
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        assert svr.seedvr2_node_pack_installed() is True


def test_no_comfyui_means_we_do_not_know_rather_than_installed(app, tmp_path):
    from app import config
    from app.services import seedvr2_helper as svr
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(tmp_path / 'nope')}})
        assert svr.seedvr2_node_pack_installed() is False


# --- The node pack is deliberately NOT auto-installed ------------------------

def test_the_node_pack_is_not_an_install_action():
    """It declares thirteen pip dependencies that belong in ComfyUI's
    interpreter, which this app does not own and must never pip into. Cloning it
    alone would land a pack that fails to import."""
    from app import setup_installer
    assert 'seedvr2_nodes' not in setup_installer.INSTALL_ACTIONS
    assert not any('seedvr2' in p for p in setup_installer._NODE_PACKS)
    assert setup_installer._INSTALL_GROUPS['seedvr2'] == ('seedvr2_model',
                                                          'seedvr2_vae')


def test_install_actions_and_workers_cover_the_weights():
    from app import setup_installer
    for a in ('seedvr2_model', 'seedvr2_vae'):
        assert a in setup_installer.INSTALL_ACTIONS
        assert a in setup_installer._WORKERS


def test_seedvr2_stays_out_of_install_everything():
    """"Install everything" runs unattended from a Setup button. SeedVR2 is an
    explicit pick like Krea, not a 3.9 GB surprise on a metered link."""
    from app import setup_installer
    caps = {'comfyui': {'dir_valid': True,
                        'seedvr2_missing': ['seedvr2_model', 'seedvr2_vae']}}
    assert 'seedvr2_model' not in setup_installer.install_all_plan(caps)


def test_the_group_plan_only_queues_what_is_missing():
    from app import setup_installer
    caps = {'comfyui': {'dir_valid': True, 'reachable': True,
                        'seedvr2_missing': ['seedvr2_vae'], 'seedvr2_invalid': []}}
    assert setup_installer.install_group_plan('seedvr2', caps) == ['seedvr2_vae']
    # A file that is present but not loadable is not "installed".
    caps['comfyui']['seedvr2_missing'] = []
    caps['comfyui']['seedvr2_invalid'] = [{'asset': 'seedvr2_model', 'blocking': True}]
    assert setup_installer.install_group_plan('seedvr2', caps) == ['seedvr2_model']
    # Nowhere to install into -> never guess a path.
    assert setup_installer.install_group_plan(
        'seedvr2', {'comfyui': {'dir_valid': False}}) == []


def test_krea_group_planning_is_unchanged_by_the_generalisation():
    """The group table was generalised to carry SeedVR2; Krea's node-pack logic
    (three states: on disk / missing / unreachable) must be byte-identical."""
    from app import setup_installer
    caps = {'comfyui': {'dir_valid': True, 'reachable': True,
                        'krea_missing': ['krea_vae'], 'krea_invalid': [],
                        'krea_nodes_missing': ['Krea2EditModelPatch'],
                        'krea_nodes_installed': False}}
    assert setup_installer.install_group_plan('krea', caps) == ['krea_nodes', 'krea_vae']
    caps['comfyui']['krea_nodes_installed'] = True
    assert setup_installer.install_group_plan('krea', caps) == ['krea_vae']


# --- Settings clamp to something the node accepts ----------------------------

def test_settings_clamp_and_never_pass_an_invalid_enum_through(app):
    from app import config
    from app.services import seedvr2_helper as svr
    with app.app_context():
        config.save_config({'seedvr2': {'resolution': 99999, 'max_resolution': 1081,
                                        'color_correction': 'sepia',
                                        'blocks_to_swap': 999}})
        assert svr.target_resolution() == svr.RESOLUTION_MAX
        assert svr.max_resolution() == 1080          # snapped even
        assert svr.color_correction() == 'lab'       # typo -> the node's default
        assert svr.blocks_to_swap() == svr.BLOCKS_TO_SWAP_MAX
        config.save_config({'seedvr2': {'resolution': 'nonsense',
                                        'color_correction': 'wavelet'}})
        assert svr.target_resolution() == 1080
        assert svr.color_correction() == 'wavelet'


def test_shipped_defaults_are_the_conservative_ones(app):
    from app import config
    from app.services import seedvr2_helper as svr
    with app.app_context():
        assert svr.target_resolution() == 1080
        assert svr.max_resolution() == 0
        assert svr.color_correction() == 'lab'
        assert svr.blocks_to_swap() == 0
        assert config.DEFAULTS['improve']['engine'] == 'klein'


# --- The improve lane -------------------------------------------------------

def test_improve_engine_resolution_falls_back_instead_of_raising(app):
    """A stale tab naming an engine that no longer exists must degrade to the
    historical behaviour, not refuse a 250-image batch."""
    from app import config
    from app.services import face_dataset_service as svc
    with app.app_context():
        assert svc.resolve_improve_engine() == 'klein'
        assert svc.resolve_improve_engine('seedvr2') == 'seedvr2'
        assert svc.resolve_improve_engine('SeedVR2') == 'seedvr2'
        assert svc.resolve_improve_engine('nonsense') == 'klein'
        config.save_config({'improve': {'engine': 'seedvr2'}})
        assert svc.resolve_improve_engine() == 'seedvr2'
        # An explicit request still wins over the setting.
        assert svc.resolve_improve_engine('klein') == 'klein'


def test_the_candidates_stored_ids_do_not_move_with_the_engine(app):
    """`derivation_kind` and `action` are STORED in user databases. Renaming them
    for a second engine would strand every existing improved tile — the engine is
    recorded additively instead."""
    from app.services import face_dataset_service as svc

    class _Src:
        id, dataset_id, source_metadata = 7, 3, None

    for engine in svc.IMPROVE_ENGINES:
        meta = svc._improve_extra_metadata(_Src(), 'label', engine=engine)
        assert meta['derivation_kind'] == 'klein_image_improve'
        assert meta['action'] == 'upscale_improve'
        assert meta['improve_engine'] == engine


def test_preflight_raises_the_engines_own_exception_type(app, tmp_path, monkeypatch):
    """Each engine keeps its own type so the routes can answer 'install the node
    pack' vs 'place the weights' — collapsing them loses that distinction."""
    from app import config
    from app.services import face_dataset_service as svc
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    monkeypatch.setattr(svr, 'seedvr2_missing_nodes', lambda: [])
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        with pytest.raises(svr.SeedVR2ModelsMissing):
            svc._improve_preflight('seedvr2')


def test_the_bulk_batch_refuses_once_not_once_per_image(app, tmp_path, monkeypatch):
    """A missing model must surface ONE actionable 409 before any candidate row
    exists, not 250 broken tiles."""
    from app import config
    from app.services import face_dataset_service as svc
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    monkeypatch.setattr(svr, 'seedvr2_missing_nodes', lambda: [])
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        ds = svc.create_dataset('local', 'seedvr2 batch', 'sv2trigger')
        with pytest.raises(svr.SeedVR2ModelsMissing):
            svc.start_bulk_improve(app, 'local', ds.id, [1, 2, 3], engine='seedvr2')


def test_the_batch_endpoint_answers_a_structured_409(app, client, tmp_path, monkeypatch):
    from app import config
    from app.services import face_dataset_service as svc
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    monkeypatch.setattr(svr, 'seedvr2_missing_nodes', lambda: ['SeedVR2VideoUpscaler'])
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
        ds = svc.create_dataset('local', 'seedvr2 409', 'sv2trigger')
        dataset_id = ds.id
    r = client.post(f'/api/dataset/{dataset_id}/improve/batch',
                    json={'image_ids': [1], 'engine': 'seedvr2'})
    assert r.status_code == 409
    body = r.get_json()
    assert body['ok'] is False
    assert 'seedvr2_missing' in body
    assert body['seedvr2_missing']['nodes'] == ['SeedVR2VideoUpscaler']
    # The node pack is named with a way to install it that also installs its
    # dependencies — a bare clone would land a pack that cannot import.
    assert 'ComfyUI-Manager' in body['error']
    assert [f['path'] for f in body['seedvr2_missing']['files']] == [
        'models/SEEDVR2/seedvr2_ema_3b_fp8_e4m3fn.safetensors',
        'models/SEEDVR2/ema_vae_fp16.safetensors']


def test_capabilities_publishes_every_gap_separately(app, tmp_path, monkeypatch):
    """"Download the weights" and "install the node pack" are different actions
    with different buttons, so they are different keys."""
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
    assert sorted(caps['seedvr2_missing']) == ['seedvr2_model', 'seedvr2_vae']
    assert caps['seedvr2_ready'] is False


def test_the_models_endpoint_only_offers_installed_builds(app, client, tmp_path):
    from app import config
    from app.services import seedvr2_helper as svr, comfy_model_paths
    base = _make_comfyui(tmp_path)
    _install_weights(base, svr.CANONICAL_DIT, svr.CANONICAL_VAE)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        comfy_model_paths.clear_cache()
    body = client.get('/api/seedvr2/models').get_json()
    assert body['installed'] == [svr.CANONICAL_DIT]
    assert body['resolved'] == svr.CANONICAL_DIT
    assert body['vae'] == svr.CANONICAL_VAE
    # The catalog still SHOWS the bigger builds (with size and VRAM guidance) and
    # marks them uninstalled — informing is not fetching.
    by_file = {v['file']: v for v in body['catalog']}
    assert by_file[svr.CANONICAL_DIT]['installed'] is True
    assert by_file['seedvr2_ema_7b_fp16.safetensors']['installed'] is False
    assert by_file['seedvr2_ema_7b_fp16.safetensors']['size_gb'] > 16
