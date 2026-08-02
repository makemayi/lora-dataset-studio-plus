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


def resolve_seedvr2_vae():
    """The `model` value for SeedVR2LoadVAEModel — the canonical
    ema_vae_fp16.safetensors when present, else the first file in the folder
    whose name says VAE. Never a blind first-file guess: handing the DiT weights
    to the VAE loader fails deep inside the node with an unreadable error."""
    listings = _listings()
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

def build_workflow(source_image, *, dit, vae, seed, resolution=1080,
                   max_res=0, color_correct='lab', swap_blocks=0,
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
        '1': {'class_type': 'SeedVR2LoadDiTModel',
              'inputs': {'model': dit, 'device': 'cuda:0',
                         'offload_device': 'none', 'cache_model': False,
                         'blocks_to_swap': int(swap_blocks),
                         'swap_io_components': False,
                         'attention_mode': 'sdpa'},
              '_meta': {'title': 'SeedVR2 DiT model'}},
        '2': {'class_type': 'SeedVR2LoadVAEModel',
              'inputs': {'model': vae, 'device': 'cuda:0',
                         'offload_device': 'none', 'cache_model': False,
                         'encode_tiled': False, 'decode_tiled': False},
              '_meta': {'title': 'SeedVR2 VAE'}},
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

    workflow = build_workflow(
        comfy_input, dit=dit, vae=vae, seed=seed,
        resolution=target_resolution(), max_res=max_resolution(),
        color_correct=color_correction(), swap_blocks=blocks_to_swap(),
        # UNIQUE prefix per job: SaveImage numbers from what is currently in the
        # output folder and the app moves each result out right after completion,
        # so a shared prefix makes the counter re-issue the same name.
        filename_prefix=f'{user_id}_DatasetSeedVR2_{uid}')

    job_id = str(uuid.uuid4())
    meta = {'model_name': 'seedvr2_upscale'}
    if extra_metadata:
        meta.update(extra_metadata)
    meta['staged_inputs'] = [comfy_input]   # dropped again when the job ends
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt='', job_id=job_id,
                          metadata=meta)
    return job_id
