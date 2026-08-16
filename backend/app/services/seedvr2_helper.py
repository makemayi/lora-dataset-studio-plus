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

HOW THIS ENGINE RUNS (2026-08-16: rebuilt on the same manual pipeline the
✨ improve 'klein' lane ships)
----------------------------------------------
This lane used to run the pack's ONE-BOX nodes (`SeedVR2LoadDiTModel` +
`SeedVR2VideoUpscaler`), which load a standard fp8 build from the pack-private
`SEEDVR2` folder and cannot read the community convrot/INT8 quantisations. The
improve lane's SeedVR2 half (krea_hq_helper) always ran the OTHER shape: core
`UNETLoader`/`VAELoader` + `SeedVR2Preprocess`/`SeedVR2Conditioning`/
`SeedVR2PostProcessing` + core `VAEEncodeTiled`/`VAEDecodeTiled` + core
`KSampler` (one step). That shape reads ANY build a core loader can read —
which is the build most installs already have, because the improve lane
requires it in `models/diffusion_models` (`seedvr2_7b_sharp_int8_convrot`).

The two lanes now share that shape, so:
  * the DiT resolves from `diffusion_models` (canonical 7B Sharp int8 first,
    then a narrow token match) — the same folder and the same file the improve
    lane uses, no second download;
  * the VAE resolves from `vae` (canonical ema_vae_fp16);
  * tiling is the CORE `VAEEncodeTiled`/`VAEDecodeTiled`, always on — no
    `Comfyui_TTP_Toolset`, no lane choice, no VRAM ceiling: a big frame is cut
    by the tiled VAE instead of being refused;
  * the node pack is still `seedvr2_videoupscaler` — it provides the three
    manual SeedVR2 nodes — but the heavy ONE-BOX loaders are gone.

WEIGHTS ARE NOT AUTO-DOWNLOADED
-------------------------------
The 7B Sharp int8 build has NO verified public URL (it is a community
re-quantisation, same as the improve lane's — see krea_hq_helper), so this
lane adds no download action. The VAE's canonical file is `ema_vae_fp16`,
which the improve lane already needs in `models/vae`; missing assets are named
with their exact path and the user places them, exactly like the improve lane.

ONE IMAGE PER JOB — and why there is no batch-size setting
------------------------------------------------------------
SeedVR2 is a video model: frames in one batch share temporal attention, which
is what keeps a clip coherent. Feeding it five unrelated dataset photos would
let them bleed into each other. Images go one per job, and the dataset batch
gets its throughput from the existing MAX_FANOUT queue. The Settings card says
so instead of leaving a dead dial.
"""
from __future__ import annotations
import logging
import os
import time
import uuid

from .. import config as cfg
from . import comfy_model_paths
from ..utils import comfy_fs
from ..utils.comfyui import load_workflow_local
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

ENGINE_ID = 'seedvr2'
ENGINE_LABEL = 'SeedVR2'

# The shipped workflow — the user's own verified `utility_seedvr2_7b_int8_upscale_image.json`
# (7B Sharp int8, one-step restore), kept AS-IS except for the values this lane
# fills. Same contract as every other shipped workflow: loader values are
# re-resolved against what is actually installed, never left as the author's.
WORKFLOW_PATH = cfg.BACKEND_DIR / 'workflows' / 'seedvr2 7b int8 upscale.json'

# Where the DiT and VAE resolve from. NOT the pack-private `SEEDVR2` folder:
# this graph loads them through core `UNETLoader`/`VAELoader`, whose widget
# lists come from `diffusion_models` and `vae` — the same folders the Klein
# improve lane reads, so one install serves both lanes
# (comfy-loader-folder-rule).
DIT_FOLDER = 'diffusion_models'
VAE_FOLDER = 'vae'

_MODEL_SUFFIXES = ('.safetensors', '.gguf')

# The three MANUAL pipeline nodes this graph uses, all from the
# seedvr2_videoupscaler pack. The ONE-BOX nodes (SeedVR2LoadDiTModel /
# SeedVR2LoadVAEModel / SeedVR2VideoUpscaler) are NOT required any more.
SEEDVR2_NODE_CLASSES = ('SeedVR2Preprocess', 'SeedVR2Conditioning',
                        'SeedVR2PostProcessing')
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
        'kind': 'SeedVR2 DiT model (7B Sharp int8 build)',
        'path': 'models/diffusion_models/seedvr2_7b_sharp_int8_convrot.safetensors',
        'source': 'no public URL — the same community build the ✨ improve '
                  "'klein' engine loads; see krea_hq_helper",
    },
    'seedvr2_vae': {
        'kind': 'SeedVR2 VAE',
        'path': 'models/vae/ema_vae_fp16.safetensors',
        'source': 'https://huggingface.co/numz/SeedVR2_comfyUI',
    },
}
SEEDVR2_REQUIRED = tuple(SEEDVR2_ASSETS)

# The builds this lane loads, canonical first then a narrow token match — the
# same discipline as krea_hq_helper.resolve_seedvr2_unet, so both lanes agree
# on the file. The canonical is the 7B Sharp int8 the improve lane requires.
SEEDVR2_UNET_CANONICAL = 'seedvr2_7b_sharp_int8_convrot.safetensors'
SEEDVR2_UNET_TOKENS = ('7b_sharp', 'sharp_int8')
SEEDVR2_VAE_CANONICAL = 'ema_vae_fp16.safetensors'
SEEDVR2_VAE_TOKENS = ('ema_vae',)

# The model's working short edge, in pixels — what the input is scaled to
# before the one-step restore (the value the shipped Klein improve graph pins
# on ResizeImageMaskNode). Exposed as `seedvr2.resolution`, same clamp range as
# before. A bigger number restores at higher resolution at more VRAM/time.
RESOLUTION_MIN, RESOLUTION_MAX = 1.0, 4.0

# color-correction enums the node accepts (its own order).
COLOR_CORRECTIONS = ('lab', 'wavelet', 'wavelet_adaptive', 'hsv', 'adain', 'none')

# Tiled VAE geometry, from the user's verified `utility_seedvr2_7b_int8_upscale_image.json`
# — measured values, not ours.
VAE_TILE_SIZE = 512
VAE_TILE_OVERLAP = 128
VAE_TEMPORAL_SIZE = 4096
VAE_TEMPORAL_OVERLAP = 8

# The one-step sampler this graph runs: SeedVR2 is a single-step distill, so
# steps=1 / cfg=1 / euler / simple is the shipped pipeline, not a guess.
SAMPLE_STEPS = 1
SAMPLE_CFG = 1.0
SAMPLE_SAMPLER = 'euler'
SAMPLE_SCHEDULER = 'simple'
SAMPLE_DENOISE = 1.0
SAMPLE_SEED = 42   # fixed: a restoration must come back identical on re-run

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

# `(rel_name, abs_path)` per loadable file across the search roots of a folder
# type — the faithful mirror of what the loader node's widget lists, exactly
# like krea_hq_helper's SeedVR2 resolvers. DiT from `diffusion_models`, VAE from
# `vae`: the same folders the improve lane reads.

def _resolve_in_folder(folder, canonical, tokens):
    """`(rel_name, abs_path)` of `canonical` under any search root of `folder`,
    else the first loadable file whose basename carries one of `tokens`, else
    (None, None). Relative — WITH its subfolder prefix, which is exactly what
    the loader widget lists. Never a blind first-file guess: a WRONG file is
    worse than a missing one (different parameter count for the DiT; a
    non-SeedVR2 VAE fails deep inside the node), and `vae` in particular is
    full of unrelated VAEs."""
    listings = [(rel, ab) for rel, ab
                in sorted(comfy_model_paths.list_models(folder))
                if comfy_model_paths.is_loadable_model(os.path.basename(rel))]
    canon = [item for item in listings
             if os.path.basename(item[0]).lower() == canonical.lower()]
    if canon:
        return canon[0]
    for rel, ab in listings:
        low = os.path.basename(rel).lower()
        if any(tok in low for tok in tokens):
            return rel, ab
    return None, None


def resolve_seedvr2_dit():
    """The `unet_name` for this graph's core UNETLoader — ComfyUI-relative, from
    the `diffusion_models` folder, or None when no SeedVR2 DiT build is on disk.

    Canonical 7B Sharp int8 first, else a narrow token match — the same
    discipline and the same file as krea_hq_helper.resolve_seedvr2_unet, so the
    two lanes can never disagree about which build is "the" SeedVR2 model."""
    rel, _ab = _resolve_in_folder(DIT_FOLDER, SEEDVR2_UNET_CANONICAL,
                                  SEEDVR2_UNET_TOKENS)
    return rel


def resolve_seedvr2_vae():
    """The `vae_name` for this graph's core VAELoader — ComfyUI-relative, from
    the `vae` folder, or None when no VAE is on disk. Canonical ema_vae_fp16
    first, else a narrow token match."""
    rel, _ab = _resolve_in_folder(VAE_FOLDER, SEEDVR2_VAE_CANONICAL,
                                  SEEDVR2_VAE_TOKENS)
    return rel


def _abs_under_roots(rel_name):
    if not rel_name:
        return None
    for root in comfy_model_paths.search_roots(DIT_FOLDER):
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


def _workflow_class_types(workflow):
    return {n.get('class_type') for n in (workflow or {}).values()
            if isinstance(n, dict) and n.get('class_type')}


def seedvr2_missing_nodes():
    """[class_type] of the nodes the SHIPPED WORKFLOW needs that the target
    ComfyUI does not expose. [] when they are all present OR when /object_info
    is unreachable — FAIL-OPEN, same contract as every sibling helper."""
    global _nodes_ok_until
    if time.time() < _nodes_ok_until:
        return []
    workflow = load_workflow_local(str(WORKFLOW_PATH)) or {}
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = sorted(_workflow_class_types(workflow) - available)
    if not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def clear_nodes_cache():
    """Drop the success TTL so the next probe re-asks /object_info."""
    global _nodes_ok_until
    _nodes_ok_until = 0.0


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
    preflight banner already renders. The three manual SeedVR2 nodes map to the
    numz pack; anything else the workflow needs (ResizeImageMaskNode,
    JoinImageWithAlpha) is named WITHOUT a guessed pack, same as krea_hq_helper."""
    out = []
    for ct in (nodes or []):
        if ct in SEEDVR2_NODE_CLASSES:
            out.append({'class_type': ct, **SEEDVR2_NODE_PACK})
        else:
            out.append({'class_type': ct, 'pack': None, 'url': None,
                        'search': ct})
    return out


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

COLOR_CORRECTIONS = ('lab', 'wavelet', 'wavelet_adaptive', 'hsv', 'adain', 'none')


def _clamp_int(value, lo, hi, default):
    try:
        return int(max(lo, min(hi, float(value))))
    except (TypeError, ValueError):
        return default


def target_resolution():
    """The UPSIZE MULTIPLIER — `seedvr2.resolution`. 2.0 (the default, and the
    value of the user's verified workflow) doubles the short edge: a 1024px
    photo is restored at 2048px. Clamped to 1.0-4.0. The tiled VAE cuts the
    scaled frame into 512 px tiles, so any size runs on a small card; a bigger
    multiplier means the model restores at proportionally more pixels, at more
    VRAM and time per tile.

    NOTE the semantic change on 2026-08-16: this key used to be a SHORT-EDGE
    pixel target for the one-box upscaler; the manual pipeline's ResizeImageMaskNode
    has no equivalent, so the key now carries the multiplier. A stored old-style
    value (e.g. 1080) clamps to the top of the range rather than being read as
    pixels."""
    try:
        v = float(cfg.get('seedvr2.resolution') or 2.0)
    except (TypeError, ValueError):
        v = 2.0
    return round(max(RESOLUTION_MIN, min(RESOLUTION_MAX, v)), 2)


def color_correction():
    """How the output is graded back onto the input's colours. Unknown values
    fall back to the node's own default rather than being passed through — a
    typo in config must not reach ComfyUI as an invalid enum."""
    v = str(cfg.get('seedvr2.color_correction') or '').strip().lower()
    return v if v in COLOR_CORRECTIONS else 'lab'


# --- Graph -------------------------------------------------------------------

# The workflow's node ids, read from the shipped file (user's own
# `utility_seedvr2_7b_int8_upscale_image.json`). Kept as constants so a
# re-export that renumbers nodes fails the shape test loudly instead of
# silently filling the wrong node.
NODE_LOAD_IMAGE = '1'
NODE_VAE = '66:51'
NODE_UNET = '66:52'
NODE_SAMPLER = '66:54'
NODE_RESIZE = '66:57'
NODE_POST = '66:59'
NODE_SAVE = '9'
NODE_IMAGE_COMPARE = '65'   # the debug before/after viewer — dropped from jobs


def build_workflow(source_image, *, dit, vae, seed=SAMPLE_SEED, resolution=2.0,
                   color_correct='lab', filename_prefix='seedvr2_upscale'):
    """Load the shipped SeedVR2 workflow (the user's verified
    `utility_seedvr2_7b_int8_upscale_image.json` — 7B Sharp int8, one-step
    restore) and fill the values this job owns. Everything else stays exactly
    as the author validated it.

    `resolution` is the UPSIZE MULTIPLIER (2.0 doubles the short edge): the
    input is scaled by it before the one-step restore, so a 1024px photo
    becomes a 2048px restoration. The core tiled VAE (512 px tiles, temporal
    4096) cuts the scaled frame, so a big upscale never has to fit the card in
    one piece.

    The debug `ImageCompare` node (original vs output) is DROPPED from jobs:
    it is a front-end viewer with a socketless `__value__` input, not a pass
    this engine ships."""
    raw = load_workflow_local(str(WORKFLOW_PATH))
    if not raw:
        raise ValueError('failed to load the SeedVR2 upscale workflow')
    workflow = dict(raw)
    for node in (NODE_LOAD_IMAGE, NODE_VAE, NODE_UNET, NODE_SAMPLER,
                 NODE_RESIZE, NODE_POST, NODE_SAVE):
        if node not in workflow:
            raise ValueError(f'workflow node {node} missing — '
                             'seedvr2 7b int8 upscale.json has changed')
    workflow[NODE_LOAD_IMAGE]['inputs']['image'] = source_image
    workflow[NODE_UNET]['inputs']['unet_name'] = dit
    workflow[NODE_VAE]['inputs']['vae_name'] = vae
    workflow[NODE_RESIZE]['inputs']['resize_type.multiplier'] = float(resolution)
    workflow[NODE_POST]['inputs']['color_correction_method'] = color_correct
    workflow[NODE_SAMPLER]['inputs']['seed'] = int(seed)
    workflow[NODE_SAVE]['inputs']['filename_prefix'] = filename_prefix
    workflow.pop(NODE_IMAGE_COMPARE, None)
    return workflow


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def enqueue_seedvr2_upscale(user_id, source_filename, source_path=None,
                            extra_metadata=None, seed=SAMPLE_SEED):
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
    dit = resolve_seedvr2_dit()
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
    workflow = build_workflow(
        comfy_input, dit=dit, vae=vae, seed=seed,
        resolution=target_resolution(),
        color_correct=color_correction(),
        filename_prefix=prefix)

    job_id = str(uuid.uuid4())
    meta = {'model_name': 'seedvr2_upscale'}
    if extra_metadata:
        meta.update(extra_metadata)
    meta['staged_inputs'] = [comfy_input]   # dropped again when the job ends
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt='', job_id=job_id,
                          metadata=meta)
    return job_id
