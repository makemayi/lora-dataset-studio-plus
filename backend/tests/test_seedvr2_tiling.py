"""SeedVR2 high-resolution lane — tiling, and the ceiling that warns first.

WHY THIS EXISTS. SurpassHR (GitHub #32) hit a real CUDA OOM upscaling full-frame
on an 11.6 GB card and shipped a tiled workflow on his fork that reaches >4K on
the same machine. Two things came out of that report, and both are tested here:
the tiled lane itself, and the fact that a user must be TOLD where the limit is
before a run dies rather than discovering it in a PyTorch traceback.

WHAT WAS DELIBERATELY NOT PORTED. His graph chains three node packs; two of them
only do arithmetic (normalise a pixel count, divide by 1024 to count tiles), and
one of those two is GPL-3.0 while this repo is MIT and has refused a dependency
over its licence before. The arithmetic lives in `tile_plan` instead, so the
lane needs TTP alone — MIT, two classes. These tests pin that reduction: if
someone re-adds a pack, `test_the_tiled_graph_needs_only_the_two_TTP_nodes`
fails.
"""
import pytest


def _plan(**over):
    from app.services import seedvr2_helper as svr
    base = dict(width=2000, height=3000, short_edge=2160)
    base.update(over)
    return svr.tile_plan(base['width'], base['height'], base['short_edge'])


# --- The geometry his graph computed with two extra node packs --------------

def test_a_single_tile_target_yields_no_grid():
    """A frame that fits inside one tile has no grid to describe. Whether a
    2-tile grid is WORTH cutting is a different question, and it belongs to
    choose_lane, which knows the card — see the 'does not tile what needs no
    tiling' test below."""
    from app.services import seedvr2_helper as svr
    assert svr.tile_plan(1024, 1024, 1024) is None
    assert svr.tile_plan(512, 512, 800) is None
    # ...and just past one tile, the geometry is honest about needing two.
    assert svr.tile_plan(512, 768, 720)['tiles'] == 2


def test_a_big_target_is_cut_into_overlapping_tiles():
    plan = _plan()
    # 2000x3000 asked for a 2160 short edge -> 2160x3240 out.
    assert (plan['output_width'], plan['output_height']) == (2160, 3240)
    assert plan['tile_width'] == plan['tile_height'] == 1024
    # Overlap is SHARED between neighbours, so columns count on the step
    # (1024 * 0.9 = 922), not on the full tile side.
    assert plan['columns'] == 3 and plan['rows'] == 4
    assert plan['tiles'] == 12


def test_more_overlap_means_more_tiles_never_fewer():
    from app.services import seedvr2_helper as svr
    lean = svr.tile_plan(2000, 3000, 2160, overlap_rate=0.0)
    fat = svr.tile_plan(2000, 3000, 2160, overlap_rate=0.4)
    assert fat['tiles'] > lean['tiles']


def test_absurd_overlap_cannot_explode_the_tile_count():
    """A config value of 0.99 would make the step 10 px and the grid enormous —
    a clamp, because a bad setting must degrade the pass, never hang the GPU."""
    from app.services import seedvr2_helper as svr
    plan = svr.tile_plan(2000, 3000, 2160, overlap_rate=0.99)
    assert plan['tiles'] <= svr.tile_plan(2000, 3000, 2160, overlap_rate=0.45)['tiles']


def test_junk_geometry_returns_no_plan_rather_than_raising():
    from app.services import seedvr2_helper as svr
    for args in [(0, 100, 1024), (100, 0, 1024), (100, 100, 0),
                 (None, None, None), ('a', 'b', 'c')]:
        assert svr.tile_plan(*args) is None


# --- The ceiling that speaks before the GPU dies ---------------------------

def test_the_ceiling_scales_with_the_card_and_stays_quiet_when_unknown():
    from app.services import seedvr2_helper as svr
    assert svr.full_frame_ceiling_mp(24) > svr.full_frame_ceiling_mp(12)
    # An unseen card gets NO number: inventing one is the false promise this
    # replaces.
    for unknown in (None, 0, -1, 'nope'):
        assert svr.full_frame_ceiling_mp(unknown) is None


def test_the_reported_case_lands_on_the_right_side_of_the_ceiling():
    """The report that opened this: full-frame OOM at roughly 4K on 11.6 GB."""
    from app.services import seedvr2_helper as svr
    ceiling = svr.full_frame_ceiling_mp(11.6)
    four_k_mp = svr.output_megapixels(2000, 3000, 2160)   # ~7.0 MP
    assert four_k_mp > ceiling, (
        'the ceiling must classify the reported OOM as over budget, or it warns '
        'about nothing')


def test_a_modest_upscale_on_the_same_card_is_NOT_flagged():
    """A ceiling that cried wolf on every run would be turned off on day one."""
    from app.services import seedvr2_helper as svr
    lane = svr.choose_lane(1024, 1024, short_edge=1080, tiling_ok=False,
                           ceiling_mp=svr.full_frame_ceiling_mp(11.6))
    assert lane['capped'] is False and lane['notice'] is None


# --- Which lane runs -------------------------------------------------------

def test_without_the_pack_the_run_still_happens_and_says_what_would_help():
    """The ceiling is guidance, not a gate: real headroom moves with the build
    and block swapping, so refusing would refuse runs that would have worked."""
    from app.services import seedvr2_helper as svr
    lane = svr.choose_lane(2000, 3000, short_edge=2160, tiling_ok=False,
                           ceiling_mp=svr.full_frame_ceiling_mp(11.6))
    assert lane['lane'] == 'full'
    assert lane['capped'] is True
    assert 'Comfyui_TTP_Toolset' in lane['notice']
    assert 'run out of memory' in lane['notice']


def test_with_the_pack_the_same_request_is_tiled_instead():
    from app.services import seedvr2_helper as svr
    lane = svr.choose_lane(2000, 3000, short_edge=2160, tiling_ok=True,
                           ceiling_mp=svr.full_frame_ceiling_mp(11.6))
    assert lane['lane'] == 'tiled'
    assert lane['capped'] is False
    assert lane['plan']['tiles'] == 12
    assert 'tiles' in lane['notice']


def test_the_pack_being_present_does_not_tile_what_needs_no_tiling():
    from app.services import seedvr2_helper as svr
    lane = svr.choose_lane(768, 1024, short_edge=1080, tiling_ok=True,
                           ceiling_mp=svr.full_frame_ceiling_mp(24))
    assert lane['lane'] == 'full' and lane['plan'] is None


# --- The graph itself ------------------------------------------------------

def test_the_tiled_graph_needs_only_the_two_TTP_nodes():
    """The reduction that dropped a GPL-3.0 dependency. If a third pack creeps
    back in, this is where it is caught."""
    from app.services import seedvr2_helper as svr
    g = svr.build_tiled_workflow('src.png', dit='d.safetensors', vae='v.safetensors',
                                 seed=42, plan=_plan(), filename_prefix='pfx')
    classes = {n['class_type'] for n in g.values()}
    foreign = {c for c in classes
               if c.startswith('TTP_') or '+' in c or c.startswith('easy ')}
    assert foreign == {'TTP_Image_Tile_Batch', 'TTP_Image_Assy'}, (
        f'the tiled lane must depend on TTP alone; found {sorted(foreign)}')
    assert classes == {'SeedVR2LoadDiTModel', 'SeedVR2LoadVAEModel', 'LoadImage',
                       'ImageScale', 'TTP_Image_Tile_Batch',
                       'SeedVR2VideoUpscaler', 'TTP_Image_Assy', 'SaveImage'}


def test_the_tiled_graph_is_wired_tiles_in_tiles_out():
    from app.services import seedvr2_helper as svr
    g = svr.build_tiled_workflow('src.png', dit='d', vae='v', seed=42, plan=_plan())
    tiler = next(k for k, n in g.items() if n['class_type'] == 'TTP_Image_Tile_Batch')
    up = next(k for k, n in g.items() if n['class_type'] == 'SeedVR2VideoUpscaler')
    assy = next(k for k, n in g.items() if n['class_type'] == 'TTP_Image_Assy')
    # The upscaler eats the tile BATCH...
    assert g[up]['inputs']['image'] == [tiler, 0]
    # ...and the assembler takes the upscaled tiles plus the tiler's own
    # positions / original size / grid, which is what puts them back in place.
    assert g[assy]['inputs']['tiles'] == [up, 0]
    assert g[assy]['inputs']['positions'] == [tiler, 1]
    assert g[assy]['inputs']['original_size'] == [tiler, 2]
    assert g[assy]['inputs']['grid_size'] == [tiler, 3]
    assert g[next(k for k, n in g.items() if n['class_type'] == 'SaveImage')]['inputs']['images'] == [assy, 0]


def test_the_tiled_lane_asks_for_a_TILE_sized_target_not_the_frames():
    """Each tile is already ~1024 px of source. Passing the frame's 2160 here
    would upscale every tile to the whole frame's size — the OOM this lane
    exists to avoid, only worse."""
    from app.services import seedvr2_helper as svr
    plan = _plan()
    g = svr.build_tiled_workflow('src.png', dit='d', vae='v', seed=42, plan=plan,
                                 resolution=min(2160, plan['tile_width']))
    up = next(n for n in g.values() if n['class_type'] == 'SeedVR2VideoUpscaler')
    assert up['inputs']['resolution'] == 1024
    assert up['inputs']['batch_size'] == 1


def test_both_lanes_offload_and_the_tiled_one_tiles_its_VAE():
    """The two VRAM wins taken from SurpassHR's graph, independent of tiling."""
    from app.services import seedvr2_helper as svr
    full = svr.build_workflow('s.png', dit='d', vae='v', seed=42, tiled_vae=True)
    dit = next(n for n in full.values() if n['class_type'] == 'SeedVR2LoadDiTModel')
    vae = next(n for n in full.values() if n['class_type'] == 'SeedVR2LoadVAEModel')
    assert dit['inputs']['offload_device'] == 'cpu'
    assert vae['inputs']['encode_tiled'] is True and vae['inputs']['decode_tiled'] is True
    assert vae['inputs']['decode_tile_size'] == 1024
    tiled = svr.build_tiled_workflow('s.png', dit='d', vae='v', seed=42, plan=_plan())
    tvae = next(n for n in tiled.values() if n['class_type'] == 'SeedVR2LoadVAEModel')
    assert tvae['inputs']['encode_tiled'] is True


def test_the_tiled_graph_is_pure(monkeypatch):
    from app.services import seedvr2_helper as svr
    monkeypatch.setattr(svr.cfg, 'get', lambda *a, **k: pytest.fail('config read'))
    svr.build_tiled_workflow('s.png', dit='d', vae='v', seed=1, plan=_plan())


# --- The probe -------------------------------------------------------------

def test_the_TTP_names_are_the_ones_the_pack_registers():
    """Read from TTP_toolsets.py NODE_CLASS_MAPPINGS, not the README — in this
    very pack `TTP_Tile_image_size` maps to a class called `Tile_imageSize`, so
    the two spellings do not even agree."""
    from app.services import seedvr2_helper as svr
    assert svr.TTP_NODE_CLASSES == ('TTP_Image_Tile_Batch', 'TTP_Image_Assy')
    assert svr.TTP_NODE_PACK['license'] == 'MIT'


def test_tiling_probe_fails_closed_when_comfyui_is_unreachable(app, monkeypatch):
    """The OPPOSITE of the SeedVR2 asset probe, on purpose. There, failing open
    avoids blocking a run over a transient blip. Here failing open would promise
    a lane whose nodes may not exist — so an unreachable ComfyUI means no tiling."""
    from app.services import seedvr2_helper as svr
    svr.clear_nodes_cache()
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes', lambda: None)
    with app.app_context():
        assert svr.tiling_available(comfy_ok=False) is False


def test_tiling_is_available_only_when_both_classes_are_there(app, monkeypatch):
    from app.services import seedvr2_helper as svr
    svr.clear_nodes_cache()
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                        lambda: {'TTP_Image_Tile_Batch'})
    with app.app_context():
        assert svr.ttp_missing_nodes() == ['TTP_Image_Assy']
        assert svr.tiling_available(True) is False
    svr.clear_nodes_cache()
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                        lambda: set(svr.TTP_NODE_CLASSES) | {'LoadImage'})
    with app.app_context():
        assert svr.tiling_available(True) is True
    svr.clear_nodes_cache()


def test_capabilities_publishes_the_lane_and_the_ceiling(app, monkeypatch):
    from app import capabilities
    from app.services import seedvr2_helper as svr
    monkeypatch.setattr(svr, 'ttp_missing_nodes', lambda: list(svr.TTP_NODE_CLASSES))
    monkeypatch.setattr(svr, 'seedvr2_missing_nodes', lambda: [])
    with app.app_context():
        caps = capabilities.probe(force=True)['comfyui']
    for key in ('seedvr2_tiling_ready', 'seedvr2_tiling_nodes_missing',
                'seedvr2_ceiling_mp'):
        assert key in caps
    assert caps['seedvr2_tiling_ready'] is False


# --- The lane is a CHOICE, not a VRAM fallback ------------------------------
# SurpassHR re-tested the shipped version and posted a side-by-side on his own
# hardware (GitHub #32): the full-frame result lost detail and gained artifacts,
# the tiled one did not. Tiling is therefore a QUALITY decision, and the old
# rule — tile only past the VRAM ceiling — had a perverse consequence: the
# bigger the card, the less often anyone got the better picture. These tests pin
# the new default and the two escape hatches.

BIG = dict(width=2000, height=3000, short_edge=2160)
SMALL_CARD = 6.4      # ~11.6 GB
BIG_CARD = 13.2       # ~24 GB


def test_the_default_tiles_on_a_BIG_card_too():
    """THE regression this feature exists for. On 24 GB the 4K frame fits whole,
    so the old VRAM-only rule kept it whole — and the person with the best
    hardware was the only one never getting the better result."""
    from app.services import seedvr2_helper as svr
    lane = svr.choose_lane(**BIG, tiling_ok=True, ceiling_mp=BIG_CARD)
    assert lane['lane'] == 'tiled', (
        'with the pack installed the default must tile for quality, not wait '
        'for the card to run out of memory')
    assert lane['plan']['tiles'] == 12


def test_never_forces_full_frame_whatever_the_geometry():
    from app.services import seedvr2_helper as svr
    lane = svr.choose_lane(**BIG, tiling_ok=True, ceiling_mp=SMALL_CARD, mode='never')
    assert lane['lane'] == 'full' and lane['plan'] is None
    # ...and the ceiling still does its one honest job.
    assert lane['capped'] is True
    assert 'run out of memory' in lane['notice']


def test_always_is_literal_and_tiles_below_the_crossover_too():
    """'always' means what it says: cut whenever there is a grid to make, even
    at a target the default would leave whole. For whoever wants tiling
    unconditionally — and the only way to get a grid under the crossover."""
    from app.services import seedvr2_helper as svr
    modest = dict(width=768, height=1024, short_edge=1200)
    assert svr.choose_lane(**modest, tiling_ok=True,
                           ceiling_mp=BIG_CARD)['lane'] == 'full'
    assert svr.choose_lane(**modest, tiling_ok=True, ceiling_mp=BIG_CARD,
                           mode='always')['lane'] == 'tiled'


def test_the_refuted_vram_only_rule_is_gone_as_an_option():
    """The pre-#32 default (tile ONLY when it would not fit) is not offered
    under any name: it is the rule the side-by-side refuted, and keeping it
    would be nostalgia for the setting that gave the biggest cards the worst
    pictures."""
    from app.services import seedvr2_helper as svr
    assert svr.TILING_MODES == ('auto', 'always', 'never')
    # On a card that can hold the 4K frame, NO mode reproduces "leave it whole
    # because it fits" except the explicit opt-out.
    tiled = [m for m in svr.TILING_MODES
             if svr.choose_lane(**BIG, tiling_ok=True, ceiling_mp=BIG_CARD,
                                mode=m)['lane'] == 'tiled']
    assert tiled == ['auto', 'always']


def test_no_pack_means_no_tiles_in_any_mode():
    """The setting cannot conjure a node pack. Without TTP every mode runs
    full-frame — and the over-budget case still says what would help."""
    from app.services import seedvr2_helper as svr
    for mode in svr.TILING_MODES:
        lane = svr.choose_lane(**BIG, tiling_ok=False, ceiling_mp=SMALL_CARD, mode=mode)
        assert lane['lane'] == 'full', f'{mode} tiled without the pack'
        assert lane['capped'] is True
        assert 'Comfyui_TTP_Toolset' in lane['notice']


def test_a_frame_inside_one_tile_is_never_tiled_even_on_always():
    """Not even the literal mode can "cut" a picture that fits in one tile:
    there is no grid, and paying seam blending for a single piece is nonsense."""
    from app.services import seedvr2_helper as svr
    # 768x1024 asked for a 768 short edge stays 768x1024 — both sides within one
    # 1024 px tile, so there is no grid to make. (Ask for 1024 instead and the
    # height becomes 1365: that DOES need two tiles, and 'always' rightly cuts
    # it — the literal mode is literal.)
    lane = svr.choose_lane(768, 1024, short_edge=768, tiling_ok=True,
                           ceiling_mp=BIG_CARD, mode='always')
    assert lane['lane'] == 'full' and lane['plan'] is None
    assert svr.choose_lane(768, 1024, short_edge=1024, tiling_ok=True,
                           ceiling_mp=BIG_CARD, mode='always')['lane'] == 'tiled'


def test_the_default_leaves_a_1080_target_whole():
    """The shipped default resolution. The model already runs at a comfortable
    size there, so tiling would buy nothing and still pay for seams and a second
    pass — this is the guard that stopped 'tile for quality' becoming 'tile
    everything'."""
    from app.services import seedvr2_helper as svr
    lane = svr.choose_lane(832, 1216, short_edge=1080, tiling_ok=True,
                           ceiling_mp=BIG_CARD)
    assert lane['lane'] == 'full'


def test_the_shipped_default_is_auto(app):
    from app import config
    from app.services import seedvr2_helper as svr
    with app.app_context():
        assert config.DEFAULTS['seedvr2']['tiling'] == 'auto'
        assert svr.tiling_mode() == 'auto'


def test_an_unknown_mode_degrades_to_the_recommended_one(app):
    """A stale tab or a typo in config must not refuse a batch."""
    from app import config
    from app.services import seedvr2_helper as svr
    with app.app_context():
        assert svr.tiling_mode('nonsense') == 'auto'
        assert svr.tiling_mode('NEVER') == 'never'
        config.save_config({'seedvr2': {'tiling': 'always'}})
        assert svr.tiling_mode() == 'always'
        # an explicit request still wins over the setting
        assert svr.tiling_mode('never') == 'never'
        config.save_config({'seedvr2': {'tiling': 'garbage'}})
        assert svr.tiling_mode() == 'auto'


# --- The two dials the lane is made of --------------------------------------
# Requested with the engine itself (SurpassHR, GitHub #32: "DiT/VAE model
# locations, target resolution, batch size, etc."). The tile SIDE is the honest
# answer to the memory half of that list: batch size is a temporal window and
# shipping it would ship a way to corrupt a batch, while the tile is what a pass
# actually holds — and 1024 was chosen on one card, not on everyone's.

def test_the_tile_size_default_is_the_contributed_value(app):
    from app import config
    from app.services import seedvr2_helper as svr
    with app.app_context():
        assert config.DEFAULTS['seedvr2']['tile_px'] == svr.TILE_PX
        assert svr.tile_size() == 1024


def test_the_tile_size_is_clamped_and_snapped(app):
    """A hand-typed number reaches ComfyUI. Below 512 a tile carries too little
    context to restore anything; a side that is not a multiple of 64 is padded
    inside the VAE, so it is snapped here where it can be explained."""
    from app import config
    from app.services import seedvr2_helper as svr
    with app.app_context():
        for stored, expected in ((768, 768), (100, 512), (99999, 2048),
                                 (700, 640), ('nonsense', 1024), (None, 1024)):
            config.save_config({'seedvr2': {'tile_px': stored}})
            assert svr.tile_size() == expected, f'tile_px={stored!r}'


def test_the_crossover_follows_the_tile_unless_it_is_set(app):
    """0 = derived, so someone who halves the tile to fit an 8 GB card also gets
    tiling half as early, without having to know the second setting exists."""
    from app import config
    from app.services import seedvr2_helper as svr
    with app.app_context():
        assert config.DEFAULTS['seedvr2']['tile_threshold'] == 0
        assert svr.tile_threshold() == svr.TILE_ABOVE_SHORT_EDGE == 1536
        config.save_config({'seedvr2': {'tile_px': 512}})
        assert svr.tile_threshold() == 768
        config.save_config({'seedvr2': {'tile_threshold': 900}})
        assert svr.tile_threshold() == 900          # explicit wins over derived
        config.save_config({'seedvr2': {'tile_threshold': 12}})
        assert svr.tile_threshold() == svr.RESOLUTION_MIN


def test_a_smaller_tile_tiles_sooner():
    """The lane decision, not just the arithmetic: the same 1200 px request that
    stays whole at a 1024 tile is tiled at a 512 one."""
    from app.services import seedvr2_helper as svr
    modest = dict(width=768, height=1024, short_edge=1200)
    assert svr.choose_lane(**modest, tiling_ok=True,
                           ceiling_mp=BIG_CARD)['lane'] == 'full'
    lane = svr.choose_lane(**modest, tiling_ok=True, ceiling_mp=BIG_CARD,
                           tile_px=512)
    assert lane['lane'] == 'tiled'
    assert lane['plan']['tile_width'] == 512
    # An explicit crossover overrides the derived one, in both directions.
    assert svr.choose_lane(**modest, tiling_ok=True, ceiling_mp=BIG_CARD,
                           tile_px=512, tile_above=4096)['lane'] == 'full'


def test_a_target_sitting_EXACTLY_on_the_crossover_stays_full_frame():
    """The boundary, pinned as a DECISION rather than left as an accident.

    The crossover is derived (1.5x the tile side) and both grids are round
    numbers, so they collide on values people really type: 1536 at the default
    1024 tile, 768 at a 512 one. Asking for exactly that runs whole, because
    'above' is what `seedvr2.tile_threshold` promises and what an already-stored
    value means — flipping to >= would redefine numbers users have saved.

    What was wrong was that nothing SAID so; that is fixed in Settings, next to
    the number being chosen (frontend `laneForTarget`), not by moving the
    boundary here. Reported by SurpassHR (GitHub #32)."""
    from app.services import seedvr2_helper as svr
    at = dict(width=1024, height=1536, short_edge=svr.TILE_ABOVE_SHORT_EDGE)
    assert svr.TILE_ABOVE_SHORT_EDGE == 1536
    assert svr.choose_lane(**at, tiling_ok=True, ceiling_mp=BIG_CARD)['lane'] == 'full'
    # ...and one pixel further is tiled, so the boundary really is the boundary
    # and not a wider dead band.
    over = dict(at, short_edge=svr.TILE_ABOVE_SHORT_EDGE + 2)
    assert svr.choose_lane(**over, tiling_ok=True, ceiling_mp=BIG_CARD)['lane'] == 'tiled'
    # The same holds on a halved tile, where the crossover lands on 768 — the
    # small-card user is the one most likely to type its exact value.
    small = dict(width=768, height=1024, short_edge=768)
    assert svr.choose_lane(**small, tiling_ok=True, ceiling_mp=BIG_CARD,
                           tile_px=512)['lane'] == 'full'
    assert svr.choose_lane(**dict(small, short_edge=770), tiling_ok=True,
                           ceiling_mp=BIG_CARD, tile_px=512)['lane'] == 'tiled'


def test_the_shipped_defaults_change_nothing():
    """The whole promise of this settings wave: someone who touches nothing gets
    the exact graph the previous release built."""
    from app.services import seedvr2_helper as svr
    lane = svr.choose_lane(**BIG, tiling_ok=True, ceiling_mp=BIG_CARD,
                           tile_px=svr.TILE_PX, tile_above=svr.TILE_ABOVE_SHORT_EDGE)
    assert lane == svr.choose_lane(**BIG, tiling_ok=True, ceiling_mp=BIG_CARD)
    g = svr.build_workflow('s.png', dit='d', vae='v', seed=42, tiled_vae=True)
    vae = next(n for n in g.values() if n['class_type'] == 'SeedVR2LoadVAEModel')
    assert vae['inputs']['encode_tile_size'] == 1024
    assert vae['inputs']['encode_tile_overlap'] == 128
