"""SeedVR2 — the FIDELITY upscaler, next to Klein's rewriting "improve".

WHY IT EXISTS (issue #32, requested by SurpassHR)
-------------------------------------------------
The app already had one way to make a dataset image bigger and cleaner: the
Klein ✨ Upscale & improve pass. It is a diffusion EDIT — it re-renders skin,
hair and micro-detail from a prompt, so it genuinely improves a soft photo and
genuinely CHANGES it. On a dataset built to teach a likeness that is sometimes
the wrong trade: the crop you picked for its exact skin tone comes back with a
different one.

SeedVR2 is the other half of that choice. It is ByteDance-Seed's one-step
diffusion restoration model: it resolves detail at a higher resolution and
leaves the content where it was. Neither pass replaces the other, and the UI
says which is which in one line rather than leaving people to discover it on a
ruined batch.

THE ECOSYSTEM (measured 2026-08-02, not quoted from a README)
--------------------------------------------------------------
  * Node pack: `ComfyUI-SeedVR2_VideoUpscaler` (numz / adrientoupet),
    Apache-2.0. Node ids are `SeedVR2LoadDiTModel`, `SeedVR2LoadVAEModel` and
    `SeedVR2VideoUpscaler` — read from the pack's own `define_schema`, NOT from
    its README, whose prose names differ (`SeedVR2_VideoUpscaler`) and would
    have made every preflight report a missing node on a correct install.
  * Weights: `numz/SeedVR2_comfyUI` on Hugging Face — anonymous HTTP 200, API
    `gated=false`, Apache-2.0, same licence as ByteDance's originals
    (`ByteDance-Seed/SeedVR2-3B` / `-7B`). Sizes measured from content-length:
    3B fp8 3.39 GB, 3B fp16 6.78 GB, 7B fp8 8.24 GB, 7B fp16 16.48 GB, VAE
    501 MB — all in that one repo. GGUF quantisations live in
    `AInVFX/SeedVR2_comfyUI`.
  * Licence check is the point of that paragraph: the repo is MIT and has
    refused an AGPL dependency before (ultralytics, for watermark detection).
    Apache-2.0 on BOTH the code and the weights is clean.

WHY THE NODE PACK IS NOT AUTO-INSTALLED (unlike comfyui-krea2edit)
-------------------------------------------------------------------
The Krea pack declares `dependencies = []`, so installing it is a clone and the
app does it. This one declares thirteen — diffusers, peft, omegaconf, einops,
rotary_embedding_torch, gguf, opencv-python… — and they belong in ComfyUI's
interpreter, which this app does not own and must never pip into. A clone alone
would land a pack that fails to import, and the user would read "install the
pack" about a pack that is already there. So the pack is DETECTED and explained
(install it through ComfyUI-Manager, restart ComfyUI), and only the WEIGHTS —
which are just files in a folder — are one-click downloadable from Setup.

WHY WE ONLY EVER SUBMIT A MODEL THAT IS ALREADY ON DISK
--------------------------------------------------------
Both loader nodes are "(Down)Load" nodes: their `model` combo lists every known
build and the node downloads it on first use. Handing them a name we have not
verified would start a multi-gigabyte download inside ComfyUI, from a button
that promised an upscale — exactly the "nothing downloads without a click" rule
this app keeps everywhere else. So the resolvers below list what is ON DISK and
`preflight()` refuses when nothing is.

ONE IMAGE PER JOB — and why there is no batch-size setting
------------------------------------------------------------
The node's `batch_size` is a VIDEO window: the frames in one batch share
temporal attention, which is what keeps a clip coherent. Feeding it five
unrelated dataset photos would let them bleed into each other. The requested
"batch size" therefore has no honest meaning for a photo set, and shipping the
knob anyway would ship a way to corrupt a batch. Images go one per job
(`batch_size=1`, which is also the node's required 4n+1 shape), the dataset
batch gets its throughput from the existing MAX_FANOUT queue, and the Settings
card says so instead of leaving a dead dial.
"""
from __future__ import annotations
import logging
import os
import time
import uuid

from .. import config as cfg
from . import comfy_model_paths
from ..utils import comfy_fs
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

ENGINE_ID = 'seedvr2'
ENGINE_LABEL = 'SeedVR2'

# ComfyUI folder type. The pack registers `models/SEEDVR2` under the model type
# 'seedvr2' but the FOLDER on disk is 'SEEDVR2', and comfy_model_paths falls
# back to `<models>/<type>` for a type it has no default mapping for — so this
# name is both the folder and the search-root key, and an extra_model_paths.yaml
# entry for it works like any other.
MODEL_FOLDER = 'SEEDVR2'

# Loadable here, deliberately NOT comfy_model_paths.is_loadable_model: that
# predicate excludes .gguf because the app's OTHER loaders cannot read one. This
# pack ships its own GGUF loader (utils/gguf), so a quantised build IS loadable
# through it — for someone on 8 GB of VRAM it is the only build that fits.
_MODEL_SUFFIXES = ('.safetensors', '.gguf')

SEEDVR2_NODE_CLASSES = ('SeedVR2LoadDiTModel', 'SeedVR2LoadVAEModel',
                        'SeedVR2VideoUpscaler')
SEEDVR2_NODE_PACK = {
    'pack': 'ComfyUI-SeedVR2_VideoUpscaler',
    'url': 'https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler',
    'search': 'SeedVR2',
}

# Where each asset belongs inside a ComfyUI install, for the "place it here"
# message. Display paths only — the real lookup goes through comfy_model_paths,
# so an extra_model_paths.yaml root works exactly the same.
SEEDVR2_ASSETS = {
    'seedvr2_model': {
        'kind': 'SeedVR2 DiT model (3B or 7B)',
        'path': 'models/SEEDVR2/seedvr2_ema_3b_fp8_e4m3fn.safetensors',
        'source': 'https://huggingface.co/numz/SeedVR2_comfyUI',
    },
    'seedvr2_vae': {
        'kind': 'SeedVR2 VAE',
        'path': 'models/SEEDVR2/ema_vae_fp16.safetensors',
        'source': 'https://huggingface.co/numz/SeedVR2_comfyUI',
    },
}
SEEDVR2_REQUIRED = tuple(SEEDVR2_ASSETS)

CANONICAL_DIT = 'seedvr2_ema_3b_fp8_e4m3fn.safetensors'
CANONICAL_VAE = 'ema_vae_fp16.safetensors'

# The builds the app knows how to talk about, smallest first. `vram_gb` is the
# pack's own published guidance, not a measurement of ours — it is shown as
# guidance and never used to gate anything, because real usage moves with
# resolution and block swapping.
DIT_VARIANTS = (
    {'file': 'seedvr2_ema_3b_fp8_e4m3fn.safetensors', 'label': '3B FP8',
     'size_gb': 3.4, 'vram_gb': '8-12', 'recommended': True},
    {'file': 'seedvr2_ema_3b_fp16.safetensors', 'label': '3B FP16',
     'size_gb': 6.8, 'vram_gb': '12-16', 'recommended': False},
    {'file': 'seedvr2_ema_7b_fp8_e4m3fn.safetensors', 'label': '7B FP8',
     'size_gb': 8.2, 'vram_gb': '16-20', 'recommended': False},
    {'file': 'seedvr2_ema_7b_fp16.safetensors', 'label': '7B FP16',
     'size_gb': 16.5, 'vram_gb': '24+', 'recommended': False},
    {'file': 'seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors', 'label': '7B Sharp FP8',
     'size_gb': 8.2, 'vram_gb': '16-20', 'recommended': False},
)


# --- The HIGH-RESOLUTION lane (tiling) --------------------------------------
# Idea, workflow and the measurement behind it: SurpassHR (GitHub #32), who hit
# a real CUDA OOM upscaling full-frame on an 11.6 GB card and shipped a tiled
# graph on his fork that reaches >4K on the same machine.
#
# WHAT WE PORTED, AND WHAT WE DID NOT. His graph chains three node packs: TTP
# (MIT) for the tiling itself, ComfyUI_essentials (MIT) and ComfyUI-Easy-Use
# (GPL-3.0) for what is, in the end, arithmetic — normalising a pixel count,
# dividing by 1024 to count tiles, resizing at the end. This repo is MIT and has
# refused a dependency over its licence before, and none of that arithmetic
# needs to happen inside a graph: it happens here, in Python, where it is also
# testable without a ComfyUI. So the lane depends on TTP alone, two classes.
#
# Node names read from the pack's CODE (TTP_toolsets.py NODE_CLASS_MAPPINGS),
# never its README — in this very pack `TTP_Tile_image_size` maps to a class
# named `Tile_imageSize`, so the two do not even match.
TTP_NODE_CLASSES = ('TTP_Image_Tile_Batch', 'TTP_Image_Assy')
TTP_NODE_PACK = {
    'pack': 'Comfyui_TTP_Toolset',
    'url': 'https://github.com/TTPlanetPig/Comfyui_TTP_Toolset',
    'search': 'TTP Toolset',
    'license': 'MIT',
}

# Tiles are square and this is their side. 1024 is what SurpassHR's graph uses
# and what the SeedVR2 VAE's own tiled encode/decode is sized for. It is the
# DEFAULT of `seedvr2.tile_px`, not a constant any more: the side is the single
# biggest VRAM lever on the tiled lane, and 1024 was chosen on a card that is
# not everyone's (see `tile_size`).
TILE_PX = 1024
# Bounds for that setting. Below 512 a tile carries too little context for the
# model to restore anything convincingly and the seam count explodes; above 2048
# a tile is no longer a tile and the lane loses its reason to exist.
TILE_PX_MIN, TILE_PX_MAX = 512, 2048
# Fraction of a tile shared with its neighbour. TTP_Image_Assy blends the seam
# across that band, so too little shows a grid and too much wastes GPU time on
# pixels computed twice. 0.1 is his value.
TILE_OVERLAP_RATE = 0.1

# How many megapixels the FULL-FRAME lane can hold, per GB of VRAM on the card.
# Deliberately conservative and deliberately a single number: this exists to say
# "past here you want tiles" BEFORE a run dies, not to predict VRAM use, which
# moves with the build, the batch and block swapping. Calibrated on the report
# that opened this: full-frame OOM at ~4K (8.3 MP) on 11.6 GB.
#
# KNOWN TO BE FALSELY PESSIMISTIC, and left that way ON PURPOSE. That OOM was
# measured BEFORE this lane gained the two memory savings taken from the same
# contribution — the DiT offloading to system RAM between phases and the tiled
# VAE encode/decode, both of which now apply to the full-frame lane too. The
# real headroom is therefore higher than 0.55 MP/GB implies, and this number
# will warn about frames that would in fact have fitted. Raising it would need a
# measurement nobody has taken; inventing a better-looking constant would trade
# a documented, harmless pessimism for an undocumented, harmful optimism. The
# failure mode of being too cautious is an unnecessary suggestion to install a
# node pack; the failure mode of being too bold is the CUDA out-of-memory this
# exists to prevent. Measure before you touch it.
MP_PER_VRAM_GB = 0.55
# Below this we never claim a ceiling at all — an unknown or tiny card gets the
# honest "we cannot tell", not a made-up number.
MIN_CEILING_MP = 1.0
# With no ceiling to compare against (unknown card), tile only past this. A frame
# under it fits essentially anywhere, and tiling it would spend seam-blending on
# a picture that never needed it.
TILE_WORTH_IT_MP = 6.0
# Above this OUTPUT SHORT EDGE, 'always' tiles. The mechanism decides the shape
# of this rule: SeedVR2's `resolution` is the size the model actually works at,
# so tiling helps exactly when full-frame would push it well past the tile size
# it is comfortable with. At a 1080 target the model runs at 1080 either way —
# tiling would buy nothing and still pay for seams and a second pass; at 2160 it
# runs at 2160 full-frame versus 1024 per tile, which is the gap SurpassHR's
# side-by-side shows (GitHub #32).
# The 1.5x factor itself is a JUDGEMENT, not a measurement: it is placed so the
# shipped 1080 default keeps its single fast pass while 4K work tiles. If anyone
# measures the crossover properly, this is the number to move.
#
# It is a FACTOR of the tile side rather than a bare pixel count because the two
# move together: the crossover exists where full-frame starts working well past
# the size a tile is upscaled at, so halving the tile side must halve the
# crossover too. `seedvr2.tile_threshold` overrides it with a literal number for
# whoever wants to place the crossover by hand (see `tile_threshold`).
TILE_ABOVE_FACTOR = 1.5
TILE_ABOVE_SHORT_EDGE = int(TILE_PX * TILE_ABOVE_FACTOR)


_PROBE_VRAM = object()   # "not supplied" — distinct from an explicit unknown


def full_frame_ceiling_mp(vram_gb=_PROBE_VRAM):
    """Megapixels the full-frame lane is willing to promise on this machine, or
    None when the VRAM is unknown (no nvidia-smi, CPU-only, a remote ComfyUI).

    None means "say nothing", never "no limit": a number invented for a card we
    cannot see would be exactly the false promise this function exists to
    replace. Note the sentinel — passing None explicitly means "I looked and
    could not tell", and must NOT be mistaken for "go and probe"."""
    if vram_gb is _PROBE_VRAM:
        from .. import capabilities
        vram_gb = capabilities.gpu_vram_gb()
    try:
        vram = float(vram_gb)
    except (TypeError, ValueError):
        return None
    if not (vram > 0):
        return None
    ceiling = vram * MP_PER_VRAM_GB
    return round(ceiling, 1) if ceiling >= MIN_CEILING_MP else None


def output_megapixels(width, height, short_edge):
    """MP the upscaler will actually produce for a source of `width`x`height`
    asked to reach `short_edge` on its short side, aspect ratio preserved."""
    try:
        w, h, target = float(width), float(height), float(short_edge)
    except (TypeError, ValueError):
        return 0.0
    if not (w > 0 and h > 0 and target > 0):
        return 0.0
    scale = target / min(w, h)
    return (w * scale) * (h * scale) / 1_000_000


def tile_plan(width, height, short_edge, tile_px=TILE_PX,
              overlap_rate=TILE_OVERLAP_RATE):
    """The tiling this source needs to reach `short_edge`, or None when one tile
    already covers it (there is nothing to gain from tiling a small image, and
    a 1x1 grid pays the seam-blending cost for nothing).

    Returns ``{tile_width, tile_height, columns, rows, tiles, output_width,
    output_height}``. Pure arithmetic — this is the part SurpassHR's graph did
    with three extra node packs.

    It answers "what WOULD the grid be", never "is tiling worth it": that second
    question needs the card's ceiling, so it lives in `choose_lane`. Keeping them
    apart is what stopped a 0.8 MP thumbnail being cut into two tiles merely
    because the tile side is 1024."""
    try:
        w, h, target = int(width), int(height), int(short_edge)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0 or target <= 0:
        return None
    scale = target / min(w, h)
    out_w, out_h = max(1, round(w * scale)), max(1, round(h * scale))
    side = max(64, int(tile_px))
    # The overlap is shared between neighbours, so the NEW ground each tile
    # covers is a step, not the full side. Columns are counted on that step.
    try:
        rate = min(0.45, max(0.0, float(overlap_rate)))
    except (TypeError, ValueError):
        rate = TILE_OVERLAP_RATE
    step = max(1, int(round(side * (1 - rate))))
    columns = max(1, -(-max(0, out_w - side) // step) + 1)
    rows = max(1, -(-max(0, out_h - side) // step) + 1)
    if columns * rows <= 1:
        return None
    return {'tile_width': side, 'tile_height': side,
            'columns': columns, 'rows': rows, 'tiles': columns * rows,
            'output_width': out_w, 'output_height': out_h}


class SeedVR2ModelsMissing(Exception):
    """A SeedVR2 asset is not on disk and/or the custom-node pack is absent, so
    no valid job can be built. Raised BEFORE any row or job is created, so a
    batch answers ONE actionable 409 instead of failing image by image.

    Same attribute shape as KreaModelsMissing so the routes and the preflight
    banner need no second format: `.missing` = asset keys (subset of
    SEEDVR2_REQUIRED), `.missing_nodes` = class_types this ComfyUI lacks."""

    def __init__(self, missing, missing_nodes=None):
        self.missing = list(missing or [])
        self.missing_nodes = list(missing_nodes or [])
        super().__init__('SeedVR2 assets missing: '
                         + ', '.join(self.missing + self.missing_nodes))


# --- Resolution -------------------------------------------------------------

def _listings():
    out = []
    for folder in comfy_model_paths.search_roots(MODEL_FOLDER):
        try:
            out.append((folder, sorted(n for n in os.listdir(folder)
                                       if n.lower().endswith(_MODEL_SUFFIXES))))
        except OSError:
            continue
    return out


def installed_dit_models():
    """Bare filenames of every SeedVR2 DiT build present in any SEEDVR2 search
    root, de-duplicated, sorted. The VAE lives in the same folder and is NOT a
    DiT, so it is filtered out — offering it in the model picker would produce a
    job that dies at load time."""
    seen, out = set(), []
    for _root, names in _listings():
        for n in names:
            if n == CANONICAL_VAE or _is_vae_name(n):
                continue
            if n not in seen:
                seen.add(n)
                out.append(n)
    return sorted(out)


def _is_vae_name(name):
    return 'vae' in (name or '').lower()


def resolve_seedvr2_dit(selected=None):
    """The `model` value for SeedVR2LoadDiTModel, or None when no DiT build is on
    disk.

    Preference: the explicit pick (`selected`, or the `seedvr2.model` setting,
    matched on its BASENAME so a value copied from a listing still resolves),
    then the canonical 3B FP8, then the first installed build in name order.
    Deterministic — the same install always resolves the same file, which is what
    makes a re-run reproduce its result."""
    installed = installed_dit_models()
    if not installed:
        return None
    pick = selected or cfg.get('seedvr2.model') or ''
    bare = os.path.basename(str(pick).replace('/', os.sep).replace('\\', os.sep))
    if bare:
        if bare in installed:
            return bare
        logger.warning('seedvr2.model %r is not in the SEEDVR2 folder — '
                       'falling back to automatic resolution', pick)
    if CANONICAL_DIT in installed:
        return CANONICAL_DIT
    return installed[0]


def installed_files():
    """Every loadable file present in any SEEDVR2 search root, de-duplicated and
    sorted — DiT builds and VAEs together. The raw material of the pins."""
    seen, out = set(), []
    for _root, names in _listings():
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return sorted(out)


def vae_choices():
    """What the VAE pin may point at: ``[{file, likely_vae}]`` over everything in
    the folder, sorted, VAE-named files first.

    Both halves matter. The heuristic ('vae' in the name) is what makes the
    dropdown safe to use blind — handing the DiT weights to the VAE loader dies
    deep inside the node with an unreadable error. The rest of the folder is
    still offered, flagged, because the pin exists precisely for the person
    whose VAE is called something the heuristic cannot see; hiding those files
    would leave that person with a picker that cannot express their install."""
    files = installed_files()
    likely = [f for f in files if _is_vae_name(f)]
    others = [f for f in files if not _is_vae_name(f)]
    return ([{'file': f, 'likely_vae': True} for f in likely]
            + [{'file': f, 'likely_vae': False} for f in others])


def resolve_seedvr2_vae(selected=None):
    """The `model` value for SeedVR2LoadVAEModel, or None when no VAE is on disk.

    Preference: the explicit pick (`selected`, or the `seedvr2.vae` setting,
    matched on its BASENAME like the DiT pin), then the canonical
    ema_vae_fp16.safetensors, then the first file in the folder whose name says
    VAE. Never a blind first-file guess: handing the DiT weights to the VAE
    loader fails deep inside the node with an unreadable error.

    A PIN IS HONOURED AGAINST THE WHOLE FOLDER, not only against VAE-named
    files. That is the entire point of having one: the automatic path already
    covers every install where the file is named like a VAE, so the only person
    who needs the setting is the person whose file is not — and re-applying the
    heuristic to their explicit choice would silently ignore it."""
    listings = _listings()
    pick = selected or cfg.get('seedvr2.vae') or ''
    bare = os.path.basename(str(pick).replace('/', os.sep).replace('\\', os.sep))
    if bare:
        if any(bare in names for _root, names in listings):
            return bare
        logger.warning('seedvr2.vae %r is not in the SEEDVR2 folder — '
                       'falling back to automatic resolution', pick)
    if any(CANONICAL_VAE in names for _root, names in listings):
        return CANONICAL_VAE
    for _root, names in listings:
        for n in names:
            if _is_vae_name(n):
                return n
    return None


def _abs_under_roots(rel_name):
    if not rel_name:
        return None
    for root in comfy_model_paths.search_roots(MODEL_FOLDER):
        cand = os.path.join(root, rel_name)
        if os.path.exists(cand):
            return cand
    return None


def seedvr2_missing_assets():
    """Which SeedVR2 assets are NOT on disk, as SEEDVR2_ASSETS keys. Disk-only,
    network-free — safe for the readiness probe."""
    missing = []
    if not resolve_seedvr2_dit():
        missing.append('seedvr2_model')
    if not resolve_seedvr2_vae():
        missing.append('seedvr2_vae')
    return missing


# Advisory floors, deliberately far under the real sizes (3.4 GB / 501 MB) so a
# legitimate file can never trip them. Same reason as Klein's and Krea's: an
# interrupted or proxied download saves an HTML error page or a half file under
# the right name, passes "the file is there", and then dies inside the node.
SEEDVR2_MIN_BYTES = {
    'seedvr2_model': 512 * 1024 ** 2,   # 512 MB (real >= 3.4 GB)
    'seedvr2_vae': 32 * 1024 ** 2,      # 32 MB  (real ~= 500 MB)
}


def seedvr2_invalid_assets():
    """SeedVR2 assets present under the resolved name but NOT real, loadable
    weights — the state between 'missing' and 'ready'. Same
    [{asset, filename, verdict, blocking, reason}] shape as klein_invalid /
    krea_invalid, so one banner covers every engine.

    GGUF builds are skipped: model_integrity reads a safetensors header, and a
    valid .gguf has none — validating it there would condemn a working file."""
    from . import model_integrity
    out = []
    for key, rel in (('seedvr2_model', resolve_seedvr2_dit()),
                     ('seedvr2_vae', resolve_seedvr2_vae())):
        path = _abs_under_roots(rel)
        if not path or str(rel).lower().endswith('.gguf'):
            continue
        res = model_integrity.validate_model_file(
            path, min_bytes=SEEDVR2_MIN_BYTES.get(key))
        if res['ok']:
            continue
        out.append({'asset': key, 'filename': res['filename'],
                    'verdict': res['verdict'], 'blocking': res['blocking'],
                    'reason': res['reason']})
    return out


# --- Custom-node preflight ---------------------------------------------------
# Success-only TTL cache, the same contract as krea_missing_nodes: /object_info
# is the heaviest probe in the app, node packs do not uninstall mid-session, and
# a MISS is never cached so "install the pack, restart ComfyUI, retry" re-probes
# at once. FAIL-OPEN when ComfyUI cannot be reached — a transient probe failure
# must never block a pass.
_NODES_OK_TTL_S = 300
_nodes_ok_until = 0.0
_ttp_ok_until = 0.0


def seedvr2_missing_nodes():
    """[class_type] of the SeedVR2 nodes the target ComfyUI does not expose.
    [] when they are all present OR when /object_info is unreachable."""
    global _nodes_ok_until
    if time.time() < _nodes_ok_until:
        return []
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = sorted(c for c in SEEDVR2_NODE_CLASSES if c not in available)
    if not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def ttp_missing_nodes():
    """[class_type] of the TTP tiling nodes this ComfyUI does not expose.

    Same contract as seedvr2_missing_nodes and for the same reasons — success
    cached, misses never, FAIL-OPEN on an unreachable ComfyUI. One difference
    that matters: an absent TTP pack is NOT an error. The high-resolution lane
    is optional; without it the default lane still works, it is only capped."""
    global _ttp_ok_until
    if time.time() < _ttp_ok_until:
        return []
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = sorted(c for c in TTP_NODE_CLASSES if c not in available)
    if not out:
        _ttp_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def tiling_available(comfy_ok=True):
    """Can the high-resolution lane run here? Requires a reachable ComfyUI (the
    probe cannot fail open into a promise) AND both TTP classes."""
    return bool(comfy_ok) and not ttp_missing_nodes()


def clear_nodes_cache():
    """Drop the success TTL so the next probe re-asks /object_info."""
    global _nodes_ok_until, _ttp_ok_until
    _nodes_ok_until = 0.0
    _ttp_ok_until = 0.0


def seedvr2_node_pack_installed():
    """Is the pack's folder present in this ComfyUI's custom_nodes? Disk-only.

    This is what separates "install the pack" from "the pack is installed,
    ComfyUI just hasn't been restarted yet" — ComfyUI registers nodes at STARTUP
    only. Folder names vary (ComfyUI-Manager clones the repo name, the registry
    installs `seedvr2_videoupscaler`), so any custom_nodes entry whose name
    contains 'seedvr2' counts. False whenever ComfyUI's folder isn't
    configured/valid: we then genuinely do not know."""
    from .. import capabilities
    r = capabilities.resolve_comfyui_base(cfg.get('comfyui.base_dir') or '')
    if not r['valid']:
        return False
    root = os.path.join(r['resolved'], 'custom_nodes')
    try:
        for entry in os.scandir(root):
            if entry.is_dir() and 'seedvr2' in entry.name.lower():
                return any(os.scandir(entry.path))
    except OSError:
        return False
    return False


def seedvr2_node_hints(nodes):
    """[{class_type, pack, url, search}] for each missing node — the shape the
    preflight banner already renders."""
    return [{'class_type': ct, **SEEDVR2_NODE_PACK} for ct in (nodes or [])]


def missing_file_entries(missing):
    """[{path, kind, source}] for each missing asset key — again the shape the
    banner already renders."""
    out = []
    for key in missing or []:
        meta = SEEDVR2_ASSETS.get(key)
        if meta:
            out.append({'path': meta['path'], 'kind': meta['kind'],
                        'source': meta['source']})
    return out


def engine_ready(comfy_ok, missing=None, invalid=None, nodes_missing=None):
    """THE readiness verdict, so every caller reads the same four conditions
    instead of its own laxer copy. Ingredients are passed in because
    capabilities.probe() has already computed them."""
    if not comfy_ok:
        return False
    if missing is None:
        missing = seedvr2_missing_assets()
    if nodes_missing is None:
        nodes_missing = seedvr2_missing_nodes()
    if invalid is None:
        invalid = seedvr2_invalid_assets()
    return not missing and not nodes_missing and not any(
        i.get('blocking') for i in invalid)


def preflight():
    """Raise SeedVR2ModelsMissing when the engine cannot run."""
    missing = seedvr2_missing_assets()
    nodes = seedvr2_missing_nodes()
    if missing or nodes:
        raise SeedVR2ModelsMissing(missing, nodes)


# --- Settings ----------------------------------------------------------------

RESOLUTION_MIN, RESOLUTION_MAX = 256, 4096
MAX_RESOLUTION_MAX = 8192
COLOR_CORRECTIONS = ('lab', 'wavelet', 'wavelet_adaptive', 'hsv', 'adain', 'none')
BLOCKS_TO_SWAP_MAX = 36


def _clamp_int(value, lo, hi, default):
    try:
        return int(max(lo, min(hi, float(value))))
    except (TypeError, ValueError):
        return default


TILING_MODES = ('auto', 'always', 'never')


def tiling_mode(requested=None):
    """`seedvr2.tiling`, clamped to a value the lane understands.

    'auto' is the DEFAULT and the recommended one: tile when tiling actually
    buys something — past the size the model is comfortable at, or when the
    frame would not fit at all. Tiling is a QUALITY decision before it is a
    memory one (SurpassHR's A/B, GitHub #32), but below the crossover the model
    already works at a good size and a grid would only add seams.

    'always' is the literal reading: cut whenever there is more than one tile to
    make, whatever the size. For whoever wants tiling unconditionally.

    'never' stays full-frame whatever the geometry, for whoever sees a seam and
    prefers the softer image. The VRAM warning still applies there.

    The pre-#32 rule — tile ONLY when the frame would not fit — is deliberately
    gone: it was the default SurpassHR's side-by-side refuted, and keeping it
    under a name would just be nostalgia for a setting that made the biggest
    cards get the worst pictures.

    Unknown values fall back to the default rather than raising: a stale tab or
    a typo in config must degrade to the recommended behaviour, never refuse a
    batch."""
    for candidate in (requested, cfg.get('seedvr2.tiling')):
        name = str(candidate or '').strip().lower()
        if name in TILING_MODES:
            return name
        if name:
            logger.warning('unknown seedvr2.tiling %r — using auto', candidate)
    return 'auto'


def tile_size():
    """Side of one tile, in pixels — `seedvr2.tile_px`, clamped and snapped to a
    multiple of 64 (the VAE's own stride; an odd side just gets padded).

    THE VRAM LEVER OF THIS ENGINE, and the reason it is a setting at all. 1024
    is SurpassHR's value and it is a good one on the cards this was built on,
    but the tile is what a run has to hold: on 8 GB, 768 or 512 is the
    difference between a 4K upscale and an out-of-memory, at the cost of more
    seams and more passes. Bigger tiles on a 24 GB card go the other way — fewer
    seams, more context per tile, more VRAM.

    It also sizes the VAE's tiled encode/decode, which applies on the FULL-FRAME
    lane too, so lowering it helps even without the tiling node pack."""
    v = _clamp_int(cfg.get('seedvr2.tile_px'), TILE_PX_MIN, TILE_PX_MAX, TILE_PX)
    return v - (v % 64)


def tile_threshold(tile_px=None):
    """Output short edge above which 'auto' tiles — `seedvr2.tile_threshold`.

    0 (the default) means DERIVED: `TILE_ABOVE_FACTOR` x the tile side, which is
    the shipped 1536 at the default 1024 tile and keeps following the tile size
    when that is changed. A positive value places the crossover by hand, for
    whoever measures their own — the constant it replaces is a judgement, not a
    measurement, and says so."""
    side = tile_size() if tile_px is None else max(64, int(tile_px))
    v = _clamp_int(cfg.get('seedvr2.tile_threshold'), 0, MAX_RESOLUTION_MAX, 0)
    if v <= 0:
        return int(side * TILE_ABOVE_FACTOR)
    return max(RESOLUTION_MIN, v)


def target_resolution():
    """Short-edge target in pixels. The node scales the SHORT edge to this and
    keeps the aspect ratio, so 1080 on a 3:2 photo gives 1620x1080. Snapped to an
    even number — the node requires it."""
    v = _clamp_int(cfg.get('seedvr2.resolution'), RESOLUTION_MIN, RESOLUTION_MAX, 1080)
    return v - (v % 2)


def max_resolution():
    """Hard cap on the LONG edge, 0 = no cap. This is the VRAM safety valve on a
    panorama: without it a 4:1 crop at 1080 short edge becomes 4320 px wide."""
    v = _clamp_int(cfg.get('seedvr2.max_resolution'), 0, MAX_RESOLUTION_MAX, 0)
    return v - (v % 2)


def color_correction():
    """How the output is graded back onto the input's colours. Unknown values
    fall back to the node's own default rather than being passed through — a
    typo in config must not reach ComfyUI as an invalid enum."""
    v = str(cfg.get('seedvr2.color_correction') or '').strip().lower()
    return v if v in COLOR_CORRECTIONS else 'lab'


def blocks_to_swap():
    """Transformer blocks offloaded to system RAM during inference. 0 = none
    (fastest); higher trades speed for VRAM headroom, which is what lets the 7B
    builds run on a card that cannot hold them."""
    return _clamp_int(cfg.get('seedvr2.blocks_to_swap'), 0, BLOCKS_TO_SWAP_MAX, 0)


# --- Graph -------------------------------------------------------------------

def _dit_loader(dit, swap_blocks):
    """The DiT loader node, shared by both lanes.

    `offload_device: cpu` (SurpassHR's value, GitHub #32) parks the model in
    system RAM between phases instead of holding it on the card — a free VRAM
    win the full-frame lane was leaving on the table."""
    return {'class_type': 'SeedVR2LoadDiTModel',
            'inputs': {'model': dit, 'device': 'cuda:0',
                       'offload_device': 'cpu', 'cache_model': False,
                       'blocks_to_swap': int(swap_blocks),
                       'swap_io_components': False,
                       'attention_mode': 'sdpa'},
            '_meta': {'title': 'SeedVR2 DiT model'}}


def _vae_loader(vae, tiled, tile_px=TILE_PX):
    """The VAE loader node. `tiled` runs encode AND decode in `tile_px` tiles
    with a 1/8th overlap (128 px at the default 1024) — the other half of
    SurpassHR's VRAM saving, and the reason the tiled lane can assemble a >4K
    frame at all. The size follows `seedvr2.tile_px` so ONE setting moves the
    whole engine's memory appetite, on both lanes."""
    side = max(64, int(tile_px))
    inputs = {'model': vae, 'device': 'cuda:0',
              'offload_device': 'cpu', 'cache_model': False,
              'encode_tiled': bool(tiled), 'decode_tiled': bool(tiled)}
    if tiled:
        overlap = max(8, side // 8)
        inputs.update({'encode_tile_size': side, 'encode_tile_overlap': overlap,
                       'decode_tile_size': side, 'decode_tile_overlap': overlap})
    return {'class_type': 'SeedVR2LoadVAEModel', 'inputs': inputs,
            '_meta': {'title': 'SeedVR2 VAE'}}


def build_workflow(source_image, *, dit, vae, seed, resolution=1080,
                   max_res=0, color_correct='lab', swap_blocks=0,
                   tiled_vae=False, vae_tile_px=TILE_PX,
                   filename_prefix='seedvr2_upscale'):
    """The ComfyUI API-format graph. Pure function of its arguments — no config
    read, no disk access — so a test can assert the exact wiring without a
    ComfyUI, and every loader value is one a resolver produced.

    `batch_size` is pinned to 1: see the module docstring — it is a video frame
    window, and one dataset image is one job.

    `cache_model` stays False on both loaders. Caching would keep several GB
    resident in ComfyUI between jobs, which on a single-GPU machine is exactly
    the memory the next generation (or a training run) needs; the app already
    treats the GPU as a contended resource everywhere else."""
    return {
        '1': _dit_loader(dit, swap_blocks),
        '2': _vae_loader(vae, tiled_vae, vae_tile_px),
        '3': {'class_type': 'LoadImage', 'inputs': {'image': source_image}},
        '4': {'class_type': 'SeedVR2VideoUpscaler',
              'inputs': {'image': ['3', 0], 'dit': ['1', 0], 'vae': ['2', 0],
                         'seed': int(seed), 'resolution': int(resolution),
                         'max_resolution': int(max_res), 'batch_size': 1,
                         'uniform_batch_size': False, 'temporal_overlap': 0,
                         'prepend_frames': 0,
                         'color_correction': color_correct,
                         'input_noise_scale': 0.0, 'latent_noise_scale': 0.0,
                         'offload_device': 'cpu', 'enable_debug': False},
              '_meta': {'title': 'SeedVR2 upscale'}},
        '5': {'class_type': 'SaveImage',
              'inputs': {'filename_prefix': filename_prefix, 'images': ['4', 0]}},
    }


def build_tiled_workflow(source_image, *, dit, vae, seed, plan,
                         resolution=1080, color_correct='lab', swap_blocks=0,
                         padding=64, filename_prefix='seedvr2_upscale'):
    """The HIGH-RESOLUTION graph: cut the source into overlapping tiles, upscale
    each, blend them back. Pure, like its full-frame sibling.

    PORTED FROM SurpassHR's fork (GitHub #32) — and deliberately NOT identical
    to it, so here is the difference in one place rather than in a chat log. His
    graph chains three node packs: TTP for the tiling, plus ComfyUI_essentials
    (MIT) and ComfyUI-Easy-Use (GPL-3.0) for arithmetic — normalising a pixel
    count, dividing by 1024 to count tiles, resizing at the end. This repo is MIT
    and has refused a dependency over its licence before, and that arithmetic
    does not need to run inside a graph, so `tile_plan` does it in Python and
    hands the result in as `plan`. Net effect: one node pack instead of three,
    MIT instead of GPL-3.0, two classes to probe instead of six, and geometry
    that a test can check without a ComfyUI at all.

    The upscaler runs on the tile BATCH: `TTP_Image_Tile_Batch` emits every tile
    as one image batch, so a single SeedVR2 pass covers them all and never holds
    more than one tile's worth of activations. `batch_size` stays 1 for the same
    reason it does full-frame — it is a temporal window, and tiles of one still
    image are not frames of a video.

    `resolution` is the SHORT EDGE OF A TILE, not of the picture: each tile is
    already `plan['tile_width']` px of source, and asking for the frame's target
    here would upscale every tile to the whole frame's size."""
    return {
        '1': _dit_loader(dit, swap_blocks),
        # The VAE tiles at the SAME side as the picture: the plan's tile is what
        # a pass actually holds, so a second, different size here would undo the
        # memory decision the user made in Settings.
        '2': _vae_loader(vae, True, plan['tile_width']),
        '3': {'class_type': 'LoadImage', 'inputs': {'image': source_image}},
        # Scale the SOURCE to the target frame size first, then cut: tiling a
        # small image and enlarging each tile would ask the model to invent the
        # same detail with less context each time.
        '4': {'class_type': 'ImageScale',
              'inputs': {'image': ['3', 0], 'upscale_method': 'lanczos',
                         'width': int(plan['output_width']),
                         'height': int(plan['output_height']), 'crop': 'disabled'},
              '_meta': {'title': 'Target frame size'}},
        '5': {'class_type': 'TTP_Image_Tile_Batch',
              'inputs': {'image': ['4', 0],
                         'tile_width': int(plan['tile_width']),
                         'tile_height': int(plan['tile_height'])},
              '_meta': {'title': f"Cut into {plan['tiles']} tiles"}},
        '6': {'class_type': 'SeedVR2VideoUpscaler',
              'inputs': {'image': ['5', 0], 'dit': ['1', 0], 'vae': ['2', 0],
                         'seed': int(seed), 'resolution': int(resolution),
                         'max_resolution': 0, 'batch_size': 1,
                         'uniform_batch_size': False, 'temporal_overlap': 0,
                         'prepend_frames': 0,
                         'color_correction': color_correct,
                         'input_noise_scale': 0.0, 'latent_noise_scale': 0.0,
                         'offload_device': 'cpu', 'enable_debug': False},
              '_meta': {'title': 'SeedVR2 upscale (per tile)'}},
        '7': {'class_type': 'TTP_Image_Assy',
              'inputs': {'tiles': ['6', 0], 'positions': ['5', 1],
                         'original_size': ['5', 2], 'grid_size': ['5', 3],
                         'padding': int(padding)},
              '_meta': {'title': 'Blend the seams back together'}},
        '8': {'class_type': 'SaveImage',
              'inputs': {'filename_prefix': filename_prefix, 'images': ['7', 0]}},
    }


def _source_size(path):
    """(width, height) of the staged source, or a neutral square when it cannot
    be measured — an unreadable header must pick a lane, not crash the enqueue."""
    try:
        from PIL import Image
        from . import image_encoding
        with Image.open(path) as im:
            return image_encoding.visual_size_from_header(im)
    except Exception:
        return (1024, 1024)


def choose_lane(width, height, *, short_edge, tiling_ok, ceiling_mp=None,
               tile_px=TILE_PX, overlap_rate=TILE_OVERLAP_RATE, mode=None,
               tile_above=None):
    """Which lane runs, and what the user must be told BEFORE the GPU starts.

    Returns ``{lane, plan, output_mp, ceiling_mp, capped, notice}``:
      * ``lane`` — 'tiled' when tiling is available AND worth it, else 'full'.
      * ``plan`` — the tile geometry for the tiled lane, None otherwise.
      * ``capped`` — True when the request exceeds what full-frame can promise
        on this card and no tiled lane will run (pack absent, or 'never').

    ``mode`` is `seedvr2.tiling`: 'auto' (default — tile when it helps: past the
    model's comfortable size, or when the frame would not fit), 'always' (tile
    whenever there is more than one tile to make) or 'never' (full-frame always;
    the ceiling still warns). The caller still runs (the
        ceiling is guidance, not a gate — see below), but ``notice`` says so.

    ``tile_px`` is the tile side (`seedvr2.tile_px`) and ``tile_above`` the
    'auto' crossover (`seedvr2.tile_threshold`); left None the crossover follows
    the tile side, which is the shipped 1536 at the default 1024 tile.

    WHY THE CEILING NEVER REFUSES. It is arithmetic over a single constant, on a
    card whose real headroom moves with the build, block swapping and whatever
    else holds VRAM. Turning that into a hard stop would refuse runs that would
    have worked. What it must not do is stay SILENT: the report that opened this
    (SurpassHR, GitHub #32) is someone discovering the limit as a CUDA OOM in a
    log. So the honest contract is: always run, always say."""
    mode = tiling_mode(mode)
    # The crossover follows the tile side unless a caller places it by hand
    # (`seedvr2.tile_threshold`): a smaller tile has to start tiling sooner.
    crossover = int(tile_above) if tile_above else int(
        max(64, int(tile_px)) * TILE_ABOVE_FACTOR)
    out_mp = output_megapixels(width, height, short_edge)
    # WHY TILING IS NOT JUST A VRAM WORKAROUND — the comment that used to sit
    # here said "a picture this card can hold whole must stay whole", and it was
    # WRONG. SurpassHR posted a side-by-side on his own hardware (GitHub #32):
    # the full-frame result lost detail and gained artifacts, the tiled one did
    # not. The reason is in SeedVR2's `resolution` argument — it sets the size
    # the model actually works at, so a whole 4K frame spreads the model's
    # capacity across four times the surface, while a tile is upscaled in the
    # range the model is good at. Tiling therefore preserves HIGH-FREQUENCY
    # DETAIL, and framing it as a memory trick had a perverse consequence: the
    # threshold scaled with VRAM, so the bigger the card, the less often anyone
    # got the better picture. Someone on 24 GB essentially never did.
    #
    # So the lane is now a CHOICE (seedvr2.tiling), defaulting to 'auto', which
    # tiles whenever tiling helps rather than only when memory forces it. The
    # VRAM ceiling keeps its one honest job — warning when the pack is absent
    # and a frame will not fit.
    budget = ceiling_mp if ceiling_mp else TILE_WORTH_IT_MP
    over_budget = out_mp > budget
    if mode == 'never':
        wants_tiles = False
    elif mode == 'always':
        # Literal: cut whenever there is a grid to make. `tile_plan` still
        # returns None for anything that fits in a single tile, so this cannot
        # "cut" a picture into one piece.
        wants_tiles = True
    else:                              # 'auto' — the default
        # Tile when it actually buys something: past the size the model is
        # comfortable at, or when the frame would not fit at all. Below that,
        # the model already runs at a good size and a grid is pure cost.
        #
        # THE COMPARISON IS STRICT, AND THAT IS A DECISION. A target sitting
        # EXACTLY on the crossover runs full-frame — 1536 px at the default 1024
        # tile, 768 px at a 512 one. That is not an edge case: the crossover is
        # derived (1.5x the tile side) and both grids are round numbers, so they
        # collide on values people really type, and someone who asks for exactly
        # 1536 gets no tiles. It stays strict because 'above' is what the
        # `seedvr2.tile_threshold` setting says and what an already-stored value
        # means; moving it to >= would silently redefine numbers users have saved.
        # What was wrong was the SILENCE around it, and that is fixed where the
        # number is chosen: Settings ▸ Engines now names the lane the configured
        # target will take (`laneForTarget`). Reported by SurpassHR (GitHub #32).
        wants_tiles = short_edge > crossover or over_budget
    plan = (tile_plan(width, height, short_edge, tile_px, overlap_rate)
            if (tiling_ok and wants_tiles) else None)
    if plan:
        return {'lane': 'tiled', 'plan': plan, 'output_mp': round(out_mp, 1),
                'ceiling_mp': ceiling_mp, 'capped': False,
                'notice': (f"Tiling this one: {plan['columns']}x{plan['rows']} tiles "
                           f"blended back together, so the card never holds the whole "
                           f"{round(out_mp, 1)} MP frame at once.")}
    over = bool(ceiling_mp) and out_mp > ceiling_mp
    notice = None
    if over:
        notice = (f"This asks for about {round(out_mp, 1)} MP in one pass, and this "
                  f"GPU is only good for roughly {ceiling_mp} MP full-frame — it may "
                  f"run out of memory. Install the {TTP_NODE_PACK['pack']} node pack "
                  f"to upscale it in tiles instead, or lower the target resolution.")
    return {'lane': 'full', 'plan': None, 'output_mp': round(out_mp, 1),
            'ceiling_mp': ceiling_mp, 'capped': over, 'notice': notice}


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def enqueue_seedvr2_upscale(user_id, source_filename, source_path=None,
                            extra_metadata=None, model=None, seed=42):
    """Copy the source into ComfyUI's input folder, build the SeedVR2 graph
    against what is ACTUALLY installed, and enqueue it. Returns the app job_id.

    Raises SeedVR2ModelsMissing when an asset or a node is absent (checked BEFORE
    anything is copied or queued), ValueError on a missing source, RuntimeError
    when ComfyUI isn't configured. Same contract, in the same order, as
    krea_edit_helper.enqueue_krea_edit — the callers treat the two engines
    interchangeably and must not need two error paths.

    The seed is FIXED (42, the node's own default) rather than random: this is a
    restoration, not a generation. A user who re-runs it expects the same file
    back, not a lottery."""
    if source_path is None:
        out_dir = cfg.comfyui_dir('output')
        if not out_dir:
            raise RuntimeError('ComfyUI is not configured')
        source_path = os.path.join(str(out_dir), source_filename)
    if not os.path.exists(source_path):
        raise ValueError(f'source image not found: {source_filename}')

    preflight()
    dit = resolve_seedvr2_dit(model)
    vae = resolve_seedvr2_vae()

    comfy_input_dir = comfy_fs.ensure_input_usable(_comfy_input_dir())
    uid = uuid.uuid4().hex[:8]
    source_stem = os.path.splitext(os.path.basename(str(source_filename)))[0] or 'source'
    staged_source = comfy_fs.stage_input_image(
        source_path, f'seedvr2_source_{uid}_{source_stem}.png', comfy_input_dir)
    comfy_input = os.path.basename(staged_source)

    # UNIQUE prefix per job: SaveImage numbers from what is currently in the
    # output folder and the app moves each result out right after completion,
    # so a shared prefix makes the counter re-issue the same name.
    prefix = f'{user_id}_DatasetSeedVR2_{uid}'
    short_edge = target_resolution()
    side = tile_size()
    src_w, src_h = _source_size(staged_source)
    lane = choose_lane(src_w, src_h, short_edge=short_edge,
                       tiling_ok=tiling_available(), ceiling_mp=full_frame_ceiling_mp(),
                       mode=tiling_mode(), tile_px=side,
                       tile_above=tile_threshold(side))
    if lane['notice']:
        logger.info('seedvr2: %s', lane['notice'])
    if lane['lane'] == 'tiled':
        workflow = build_tiled_workflow(
            comfy_input, dit=dit, vae=vae, seed=seed, plan=lane['plan'],
            resolution=min(short_edge, lane['plan']['tile_width']),
            color_correct=color_correction(), swap_blocks=blocks_to_swap(),
            filename_prefix=prefix)
    else:
        workflow = build_workflow(
            comfy_input, dit=dit, vae=vae, seed=seed,
            resolution=short_edge, max_res=max_resolution(),
            color_correct=color_correction(), swap_blocks=blocks_to_swap(),
            tiled_vae=True, vae_tile_px=side, filename_prefix=prefix)

    job_id = str(uuid.uuid4())
    meta = {'model_name': 'seedvr2_upscale', 'seedvr2_lane': lane['lane']}
    if extra_metadata:
        meta.update(extra_metadata)
    meta['staged_inputs'] = [comfy_input]   # dropped again when the job ends
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt='', job_id=job_id,
                          metadata=meta)
    return job_id
