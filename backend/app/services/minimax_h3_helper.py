"""MiniMax H3 — the THIRD local generation engine, next to Klein and Krea 2 Edit.

WHAT IT IS
----------
An identity engine like Krea 2 Edit: one reference photo in, the same person
re-staged into the shot you asked for, with no character LoRA. It gets there by a
different route — H3 is a VIDEO model. `MiniMaxH3ReferenceToVideo` samples a short
packet of frames and `H3FrameSelect` keeps the single best one as the still. That
detour is the whole engine, and it is why every number below is larger than the
Krea lane's.

WHY A HAND-BUILT GRAPH (no shipped workflow JSON)
-------------------------------------------------
Same reason as krea_edit_helper: Klein loads a JSON lifted from a developer's own
ComfyUI whose loader nodes hardcode filenames that exist on no fresh install, and
klein_edit_helper then has to re-resolve every one of them. Building the graph in
code removes that class of bug — every loader value comes from a resolver here.

THE GRAPH (validated against /object_info on a live install, 2026-08-08)

    UNETLoader ──[opt]SpectrumApplyMiniMaxH3 ──[opt]PathchSageAttentionKJ
                                              ├─> BasicGuider(positive)
                                              └─> BasicScheduler(sigmas)
    CLIPLoader(qwen3-vl 32B, type='minimax') ┐
    VAELoader(video) ───────────────────────┤
    VAELoader(audio) ───────────────────────┼─> MiniMaxH3ReferenceToVideo
    LoadImage -> ResizeImagesByLongerEdge ──┘      -> CONDITIONING + LATENT
    RandomNoise + KSamplerSelect + BasicScheduler + LATENT
                       -> SamplerCustomAdvanced -> VAEDecode(video vae)
                       -> H3FrameSelect(select_count=1, reference, clip_vision)
                       -> [opt]RTXVideoSuperResolution -> SaveImage

`MiniMaxH3ReferenceToVideo`, `ResolutionSelector` and every loader are ComfyUI
CORE (`comfy_extras.nodes_minimax_h3` / `nodes_resolution` / `nodes`). Exactly ONE
custom pack is mandatory: `MinimaxH3-Image`, for `H3FrameSelect`. The three speed
nodes and the RTX upscaler are attached when present and skipped when absent — an
install carrying only core plus that one pack must generate, slower, same image.

`ResolutionSelector` is dropped on purpose: the node's `width`/`height` are plain
INTs with step 32, so `fit_output_size` owns the geometry and one more node-name
dependency disappears.

MEASURED, 2026-08-08, RTX 3090 24 GB (do not "simplify" these away)
-------------------------------------------------------------------
  * **ComfyUI must run with `--disable-dynamic-vram`.** The weights total 40.2 GB
    (text encoder 14.6 + UNET 19.5 + video VAE 4.9 + CLIP-Vision 1.2) against
    24 GB of card. With dynamic staging on, two of them stay staged and Windows
    pages VRAM to system RAM at random: identical runs measured anywhere from
    69 s to 397 s. With the flag, sequential loading peaks at max(model) = 19.5 GB
    and the SAME prompt change costs 77.5 s. Nothing errors without it — it just
    randomly takes 6x longer, which is why `preflight` says so out loud.
  * New prompt 77.5 s · same prompt, new seed 37-38 s · first run after a restart
    257 s. The encode is ~40 s of that, not the 300 s a naive subtraction across
    a thrashing install suggests.
  * **The seed does not invalidate the encode**: `RandomNoise` feeds the sampler
    only, never `MiniMaxH3ReferenceToVideo`. A batch that emits one card's copies
    consecutively pays the encode ONCE — 75 min against 130 min for 100 images
    over 20 cards. `dataset_generation_service` owns that ordering; a test pins
    the wiring here.
  * `length` is min 5, step 17, node default 124, trained range ~124-362. We run
    the FLOOR (5) on purpose: we want stills, not motion, and every extra frame
    is sampled at full cost.
  * `weight_reference` on `H3FrameSelect` ships non-zero (a dataset wants the
    frame that looks most like the reference), but that is UNVERIFIED: raising it
    0 -> 1.5 on a 5-frame packet produced a pixel-identical image. Re-test at
    `length: 22`, where the candidates actually differ.

A PRECONDITION THIS MODULE CANNOT ENFORCE
-----------------------------------------
H3 copies whatever the reference carries. A reference cropped from a broadcast
still reproduced its title card and channel logo verbatim in the output, and
those would be trained into the LoRA. The app's watermark tools can clean it
afterwards; cropping the reference first is cheaper. Same for every angle slot.
"""
from __future__ import annotations
import logging
import math
import os
import random
import time
import uuid

from .. import config as cfg
from . import comfy_model_paths
from ..utils import comfy_fs
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

ENGINE_ID = 'minimax_h3'
ENGINE_LABEL = 'MiniMax H3'

_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')

# Mandatory: the core node that turns a reference into a frame packet, and the
# selector that turns the packet back into one still. Without the selector this
# is a video engine writing five tiles per shot, not a stills engine.
H3_REQUIRED_NODE_CLASSES = ('MiniMaxH3ReferenceToVideo', 'H3FrameSelect')

H3_FRAME_SELECT_PACK = {
    'pack': 'MinimaxH3-Image',
    'url': 'https://github.com/Merserk/MinimaxH3-Image',
    'search': 'MinimaxH3-Image',
}

# Attached when present, skipped when absent. NONE of these changes the image:
# Spectrum forecasts sampler steps, SageAttention swaps the attention kernel.
H3_SPEED_NODE_CLASSES = ('SpectrumApplyMiniMaxH3', 'PathchSageAttentionKJ')
# NVIDIA-only 2x upscale. Default ON, but a non-RTX card loses the 2x, not the
# engine — the graph drops the node and saves the selected frame directly.
H3_UPSCALE_NODE_CLASS = 'RTXVideoSuperResolution'

# Where each asset belongs inside a ComfyUI install, for the "place it here"
# message. Display paths only — the real lookup goes through comfy_model_paths,
# so an extra_model_paths.yaml root works exactly the same.
_H3_REPO = 'https://huggingface.co/Comfy-Org/MiniMax-H3'
H3_ASSETS = {
    'h3_unet': {
        'kind': 'MiniMax H3 Ref2VA model (any quantisation)',
        'path': 'models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors',
        'source': f'{_H3_REPO}/tree/main/diffusion_models',
    },
    'h3_text_encoder': {
        'kind': 'Qwen3-VL 32B MiniMax H3 text encoder',
        'path': 'models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
        'source': f'{_H3_REPO}/tree/main/text_encoders',
    },
    'h3_video_vae': {
        'kind': 'MiniMax H3 video VAE',
        'path': 'models/vae/minimax_h3_video_vae_fp16.safetensors',
        'source': f'{_H3_REPO}/tree/main/vae',
    },
    'h3_audio_vae': {
        'kind': 'MiniMax H3 audio VAE (required even for a still)',
        'path': 'models/vae/minimax_h3_audio_vae_fp32.safetensors',
        'source': f'{_H3_REPO}/tree/main/vae',
    },
    'h3_clip_vision': {
        'kind': 'CLIP-ViT-H vision tower (scores frame selection)',
        'path': 'models/clip_vision/CLIP-ViT-H-fp16.safetensors',
        'source': 'https://huggingface.co/Comfy-Org/sigclip_vision_patch14_384',
    },
}
H3_REQUIRED = tuple(H3_ASSETS)


class MinimaxH3ModelsMissing(Exception):
    """A MiniMax H3 asset is not on disk and/or `MinimaxH3-Image` is absent, so
    no valid job can be built. Raised BEFORE any row or job is created, so the
    route answers ONE actionable 409 instead of a grid of tiles each dying on
    ComfyUI validation.

    `.missing` = asset keys (subset of H3_REQUIRED); `.missing_nodes` =
    class_type strings the target ComfyUI does not expose."""

    def __init__(self, missing, missing_nodes=None):
        self.missing = list(missing or [])
        self.missing_nodes = list(missing_nodes or [])
        super().__init__('MiniMax H3 assets missing: '
                         + ', '.join(self.missing + self.missing_nodes))


# --- Resolution -------------------------------------------------------------
# Canonical-name-first with NARROW token fallbacks, scanning roots in ComfyUI's
# own priority order — the same discipline as klein/krea. A WRONG model is worse
# than a missing one here in a specific way worth naming: three OTHER Qwen3-VL
# encoders ship with this app's other engines, they all LOAD under type='minimax',
# and they produce garbage rather than an error.

def _listings(comfy_type):
    """(root, [rel_name]) per search root, recursive — ComfyUI's loaders accept a
    subfolder-qualified name, and real installs use them."""
    out = []
    for root in comfy_model_paths.search_roots(comfy_type):
        names = []
        if os.path.isdir(root):
            for dirpath, _sub, filenames in os.walk(root, followlinks=True):
                for fn in filenames:
                    if fn.lower().endswith(_MODEL_SUFFIXES):
                        rel = os.path.relpath(os.path.join(dirpath, fn), root)
                        names.append(rel)
        out.append((root, sorted(names)))
    return out


def _find_model_file(comfy_type, canonical, tokens, exclude=()):
    """Bare (or subfolder-qualified) name for a ComfyUI folder type: the canonical
    name if present in ANY search root, else the first (sorted) name containing a
    NARROW token and none of `exclude`. None when nothing matches — never a blind
    first-file guess."""
    listings = _listings(comfy_type)
    for _root, names in listings:
        if canonical in names:
            return canonical
    for _root, names in listings:
        for n in names:
            low = os.path.basename(n).lower()
            if any(bad in low for bad in exclude):
                continue
            if any(tok in low for tok in tokens):
                return n
    return None


# 'fl2va' is the FIRST/LAST-FRAME sibling of our model. It carries every token
# 'minimax_h3' would match, it loads without complaint, and then it does a
# different job. Excluding it by name is what stops a token match from taking it.
H3_WRONG_TASK_TOKENS = ('fl2va',)


def resolve_h3_unet(selected=None):
    """The Ref2VA diffusion model. `selected` (a Settings value) wins when it is
    still on disk; a stale one degrades to auto-resolution rather than failing."""
    if selected:
        for _root, names in _listings('diffusion_models'):
            if selected in names:
                return selected
        logger.info('minimax_h3: configured base %r is gone, auto-resolving', selected)
    return _find_model_file(
        'diffusion_models', H3_ASSETS['h3_unet']['path'].rsplit('/', 1)[-1],
        ('ref2va',), exclude=H3_WRONG_TASK_TOKENS)


def resolve_h3_text_encoder():
    """The 32B Qwen3-VL H3 encoder — NOT any of the other qwen3vl files a stock
    install carries for Krea/Qwen, which load and then produce garbage."""
    return _find_model_file(
        'text_encoders', H3_ASSETS['h3_text_encoder']['path'].rsplit('/', 1)[-1],
        ('minimax_h3', 'minimax-h3'))


def resolve_h3_video_vae():
    return _find_model_file(
        'vae', H3_ASSETS['h3_video_vae']['path'].rsplit('/', 1)[-1],
        ('h3_video_vae', 'h3-video-vae'))


def resolve_h3_audio_vae():
    """The node REQUIRES an audio VAE even for a still. Handing it the video VAE
    twice loads fine and dies at sample time on a shape mismatch."""
    return _find_model_file(
        'vae', H3_ASSETS['h3_audio_vae']['path'].rsplit('/', 1)[-1],
        ('h3_audio_vae', 'h3-audio-vae'))


def resolve_h3_clip_vision():
    return _find_model_file(
        'clip_vision', H3_ASSETS['h3_clip_vision']['path'].rsplit('/', 1)[-1],
        ('vit-h', 'vit_h'))


_RESOLVERS = {
    'h3_unet': resolve_h3_unet,
    'h3_text_encoder': resolve_h3_text_encoder,
    'h3_video_vae': resolve_h3_video_vae,
    'h3_audio_vae': resolve_h3_audio_vae,
    'h3_clip_vision': resolve_h3_clip_vision,
}


def h3_missing_assets():
    """Asset keys that do not resolve. Every gap at once — an install with
    nothing must produce a complete shopping list, not the first failure."""
    return [key for key in H3_REQUIRED if not _RESOLVERS[key]()]


# --- Nodes ------------------------------------------------------------------

_NODES_OK_TTL_S = 300
_nodes_ok_until = 0.0


def h3_missing_nodes():
    """[class_type] of the MANDATORY H3 nodes the target ComfyUI does not expose.
    [] when they are present OR when /object_info is unreachable — an unreachable
    ComfyUI is not evidence of a missing pack, and blocking there would refuse to
    generate whenever the probe times out.

    The speed nodes and the upscaler are deliberately NOT checked: they are
    optional, and `available_optional_nodes` reports them separately."""
    global _nodes_ok_until
    if time.time() < _nodes_ok_until:
        return []
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = sorted(c for c in H3_REQUIRED_NODE_CLASSES if c not in available)
    if not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def clear_nodes_cache():
    """Drop the success-TTL so the next probe re-asks /object_info. The cache only
    ever holds a POSITIVE result, but a stale positive would hide a pack the user
    removed, and clearing costs one probe."""
    global _nodes_ok_until
    _nodes_ok_until = 0.0


def available_optional_nodes():
    """{'speed': bool, 'upscale': bool} — which optional accelerators this
    ComfyUI actually exposes, so the graph attaches only what exists. Both True
    when /object_info is unreachable: the graph is then built as the user
    configured it and ComfyUI validates it, which beats silently degrading a
    working install because one probe timed out."""
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return {'speed': True, 'upscale': True}
    return {
        'speed': all(c in available for c in H3_SPEED_NODE_CLASSES),
        'upscale': H3_UPSCALE_NODE_CLASS in available,
    }


def h3_node_hints(nodes):
    """[{class_type, pack, url, search}] for each missing node — the shape the
    Studio preflight banner already renders, so the front needs no second one."""
    return [{'class_type': ct, **H3_FRAME_SELECT_PACK} for ct in (nodes or [])]


def missing_file_entries(missing):
    """[{path, kind, source}] for each missing asset key — again the Studio
    `files` shape, so one banner covers every engine."""
    out = []
    for key in missing or []:
        meta = H3_ASSETS.get(key)
        if meta:
            out.append({'path': meta['path'], 'kind': meta['kind'],
                        'source': meta['source']})
    return out


def dynamic_vram_warning():
    """A one-line warning when ComfyUI was NOT launched with
    `--disable-dynamic-vram`, or None.

    This is the only place in the app where HOW ComfyUI was started changes
    whether a feature is usable. Measured: the same prompt change costs 397 s
    without the flag and 77.5 s with it, and nothing errors — the run just
    randomly takes up to 6x longer, because 40.2 GB of weights against 24 GB of
    card makes the driver page VRAM. A warning, never a block: a bigger card
    does not need the flag at all."""
    from ..utils.comfyui import fetch_system_argv
    argv = fetch_system_argv()
    if argv is None or '--disable-dynamic-vram' in argv:
        return None
    return ('ComfyUI is running without --disable-dynamic-vram. MiniMax H3 loads '
            'more weights than a 24 GB card holds, so generations can take '
            'several times longer and vary run to run. Add the flag to ComfyUI\'s '
            'launch command and restart it.')


def preflight():
    """Raise MinimaxH3ModelsMissing when the engine cannot run. No auto-download:
    the honest answer is a named gap, not a fake installer."""
    missing = h3_missing_assets()
    nodes = h3_missing_nodes()
    if missing or nodes:
        raise MinimaxH3ModelsMissing(missing, nodes)


# --- Output geometry and packet length --------------------------------------
# Measured at 1 MP. Larger canvases were not tested, and every extra pixel is
# paid five times over because the model samples a packet, not an image.
MAX_OUTPUT_MP = 1.0
_SIZE_MULTIPLE = 32          # the node's own step for width/height

# `length` is min 5, step 17 on the node — an off-step value is a validation
# error at queue time, which means a whole batch of failed tiles.
LENGTH_MIN = 5
LENGTH_STEP = 17
LENGTH_MAX = 124             # our ceiling, not the node's (3600): stills, not film


def clamp_length(value):
    """Snap a requested packet length onto the node's own min/step grid."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = LENGTH_MIN
    n = max(LENGTH_MIN, min(LENGTH_MAX, n))
    return LENGTH_MIN + ((n - LENGTH_MIN) // LENGTH_STEP) * LENGTH_STEP


def _aspect_ratio(requested_aspect):
    """Return a positive finite ``W:H`` ratio, or None for an unusable request."""
    if not isinstance(requested_aspect, str):
        return None
    try:
        aw, ah = (float(part.strip()) for part in requested_aspect.split(':', 1))
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(aw) and math.isfinite(ah) and aw > 0 and ah > 0):
        return None
    ratio = aw / ah
    # The widest shipped catalog card is 16:9. Past this it is a typo in a custom
    # entry rather than a shot, and it would form a one-cell canvas.
    if not math.isfinite(ratio) or not 1 / 32 <= ratio <= 32:
        return None
    return ratio


def _requested_canvas(ratio, budget):
    """Largest near-ratio 32-grid canvas that does not exceed ``budget``."""
    cells = max(1, int(budget // (_SIZE_MULTIPLE ** 2)))
    height_cells = max(1, int((cells / ratio) ** 0.5))
    width_cells = max(1, int(round(ratio * height_cells)))
    while width_cells * height_cells > cells:
        # Step down whichever side currently sits furthest from the requested
        # ratio. This only runs on the 32 grid, so it cannot drift above budget.
        if width_cells / height_cells > ratio:
            width_cells = max(1, width_cells - 1)
        else:
            height_cells = max(1, height_cells - 1)
    return width_cells * _SIZE_MULTIPLE, height_cells * _SIZE_MULTIPLE


def fit_output_size(width, height, max_mp=None, requested_aspect=None):
    """A width/height inside the megapixel budget, both multiples of 32. The node
    takes plain INTs, so this replaces `ResolutionSelector` and its ratio enum
    entirely.

    A valid ``requested_aspect`` (``W:H``, the catalog card's own ratio) decides
    the SHAPE; without one, the source's aspect is kept. Either way the pixel
    count stays inside the budget AND inside what the reference already carries:
    a portrait card must not turn a 1024² reference into four times the pixels,
    because H3 pays every one of them once per frame of the packet.

    Before this, the shape came from the reference alone — so a square reference
    answered a full-body card with a square, whatever the card asked for."""
    budget = (MAX_OUTPUT_MP if max_mp is None else float(max_mp)) * 1_000_000
    w = max(1.0, float(width or 0))
    h = max(1.0, float(height or 0))
    ratio = _aspect_ratio(requested_aspect)
    if ratio is not None:
        return _requested_canvas(ratio, min(budget, w * h))
    scale = math.sqrt(budget / (w * h))
    if scale < 1.0:
        w, h = w * scale, h * scale
    out_w = max(_SIZE_MULTIPLE, int(round(w / _SIZE_MULTIPLE)) * _SIZE_MULTIPLE)
    out_h = max(_SIZE_MULTIPLE, int(round(h / _SIZE_MULTIPLE)) * _SIZE_MULTIPLE)
    # Rounding UP on both axes can cross the budget; step the long side back.
    while out_w * out_h > budget and max(out_w, out_h) > _SIZE_MULTIPLE:
        if out_w >= out_h:
            out_w -= _SIZE_MULTIPLE
        else:
            out_h -= _SIZE_MULTIPLE
    return out_w, out_h


def _source_size(path):
    """(width, height) of an image, or (1024, 1024) when it cannot be read."""
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return int(im.width), int(im.height)
    except Exception:
        logger.debug('minimax_h3: could not size %s', path, exc_info=True)
        return 1024, 1024


# --- The graph ---------------------------------------------------------------

DEFAULT_STEPS = 25
DEFAULT_SAMPLER = 'res_multistep'
DEFAULT_SCHEDULER = 'simple'
DEFAULT_REF_IMAGE_SIZE = 'match'      # 'max' is 2048px short edge, several times slower
DEFAULT_REF_LONGER_EDGE = 1024
DEFAULT_FRAME_WEIGHT_REFERENCE = 1.0
DEFAULT_RTX_SCALE = 2


def build_workflow(source_image, prompt, *, unet, clip, video_vae, audio_vae,
                   clip_vision, width, height, seed, length=LENGTH_MIN,
                   steps=DEFAULT_STEPS, sampler=DEFAULT_SAMPLER,
                   scheduler=DEFAULT_SCHEDULER,
                   ref_image_size=DEFAULT_REF_IMAGE_SIZE,
                   ref_longer_edge=DEFAULT_REF_LONGER_EDGE,
                   frame_weight_reference=DEFAULT_FRAME_WEIGHT_REFERENCE,
                   use_speed_nodes=True, use_rtx_upscale=True,
                   rtx_scale=DEFAULT_RTX_SCALE,
                   filename_prefix='minimax_h3'):
    """The ComfyUI API-format graph. Pure function of its arguments — no config
    read, no disk access — so a test can assert the exact wiring without a
    ComfyUI, and every loader value is one a resolver produced.

    `use_speed_nodes` / `use_rtx_upscale` are DEGRADATION switches, not features:
    False removes the node and rewires around it, so the same call produces a
    valid graph on an install that lacks the pack.

    The seed reaches `RandomNoise` and nothing else. That is load-bearing:
    `MiniMaxH3ReferenceToVideo` is then identical across a card's copies and
    ComfyUI serves the 40-second encode from cache."""
    length = clamp_length(length)
    steps = max(1, int(steps))
    g = {
        '1': {'class_type': 'UNETLoader',
              'inputs': {'unet_name': unet, 'weight_dtype': 'default'},
              '_meta': {'title': 'MiniMax H3 Ref2VA'}},
        '2': {'class_type': 'CLIPLoader',
              'inputs': {'clip_name': clip, 'type': 'minimax', 'device': 'default'},
              '_meta': {'title': 'Qwen3-VL 32B (H3)'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': video_vae},
              '_meta': {'title': 'H3 video VAE'}},
        '4': {'class_type': 'VAELoader', 'inputs': {'vae_name': audio_vae},
              '_meta': {'title': 'H3 audio VAE (required input)'}},
        '5': {'class_type': 'LoadImage', 'inputs': {'image': source_image}},
        '6': {'class_type': 'ResizeImagesByLongerEdge',
              'inputs': {'longer_edge': int(ref_longer_edge), 'images': ['5', 0]},
              '_meta': {'title': 'Reference, downscaled before the encoder'}},
    }

    # Model chain. Each accelerator is a pass-through patch, so dropping one only
    # shortens the chain — the sampler still sees a model either way.
    model_out = ['1', 0]
    if use_speed_nodes:
        g['7'] = {'class_type': 'SpectrumApplyMiniMaxH3',
                  'inputs': {'enabled': True, 'blend_weight': 0.5, 'degree': 1,
                             'ridge_lambda': 0.1, 'window_size': 2,
                             'flex_window': 0.75, 'warmup_steps': 1,
                             'tail_actual_steps': 1, 'max_history': 8,
                             # debug prints a summary per run; never ship it on.
                             'debug': False, 'history_storage': 'system_ram',
                             'bootstrap_first_forecast': True,
                             'model': model_out},
                  '_meta': {'title': 'Spectrum step forecasting (optional)'}}
        model_out = ['7', 0]
        g['8'] = {'class_type': 'PathchSageAttentionKJ',
                  'inputs': {'sage_attention': 'auto', 'allow_compile': False,
                             'model': model_out},
                  '_meta': {'title': 'Sage attention (optional)'}}
        model_out = ['8', 0]

    g['9'] = {'class_type': 'MiniMaxH3ReferenceToVideo',
              'inputs': {'prompt': prompt,
                         'width': int(width), 'height': int(height),
                         'length': length,
                         'ref_image_size': ref_image_size,
                         'clip': ['2', 0], 'vae': ['3', 0], 'audio_vae': ['4', 0],
                         'ref_images.ref_image_0': ['6', 0]},
              '_meta': {'title': 'H3 reference -> frame packet'}}
    g['10'] = {'class_type': 'RandomNoise',
               'inputs': {'noise_seed': int(seed)},
               '_meta': {'title': 'Seed (sampler only — never the encode)'}}
    g['11'] = {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': sampler}}
    g['12'] = {'class_type': 'BasicScheduler',
               'inputs': {'scheduler': scheduler, 'steps': steps, 'denoise': 1,
                          'model': model_out}}
    g['13'] = {'class_type': 'BasicGuider',
               'inputs': {'model': model_out, 'conditioning': ['9', 0]}}
    g['14'] = {'class_type': 'SamplerCustomAdvanced',
               'inputs': {'noise': ['10', 0], 'guider': ['13', 0],
                          'sampler': ['11', 0], 'sigmas': ['12', 0],
                          'latent_image': ['9', 1]}}
    g['15'] = {'class_type': 'VAEDecode',
               'inputs': {'samples': ['14', 0], 'vae': ['3', 0]}}
    g['16'] = {'class_type': 'CLIPVisionLoader', 'inputs': {'clip_name': clip_vision}}
    g['17'] = {'class_type': 'H3FrameSelect',
               'inputs': {'images': ['15', 0],
                          'select_count': 1,
                          'weight_sharpness': 1.0,
                          'weight_exposure': 0.5,
                          # Non-zero on purpose: a dataset wants the frame that
                          # looks most like the reference, not merely the
                          # sharpest one. See the module docstring - UNVERIFIED.
                          'weight_reference': float(frame_weight_reference),
                          'target_brightness': 0.5,
                          'brightness_tolerance': 0.25,
                          'min_score': 0.0,
                          'dedup_threshold': 0.98,
                          # Scored against the ORIGINAL reference, not the
                          # downscaled copy the encoder saw.
                          'reference': ['5', 0],
                          'clip_vision': ['16', 0]},
               '_meta': {'title': 'Keep the best single frame'}}

    images_out = ['17', 0]
    if use_rtx_upscale:
        g['18'] = {'class_type': 'RTXVideoSuperResolution',
                   'inputs': {'resize_type': 'scale by multiplier',
                              'resize_type.scale': int(rtx_scale),
                              'quality': 'ULTRA', 'images': images_out},
                   '_meta': {'title': 'RTX super resolution (optional, NVIDIA only)'}}
        images_out = ['18', 0]

    g['19'] = {'class_type': 'SaveImage',
               'inputs': {'filename_prefix': filename_prefix, 'images': images_out}}
    return g


# --- Enqueue -----------------------------------------------------------------

def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def _cfg_int(key, default):
    try:
        return int(cfg.get(key))
    except (TypeError, ValueError):
        return default


def _cfg_float(key, default):
    try:
        return float(cfg.get(key))
    except (TypeError, ValueError):
        return default


def enqueue_minimax_h3(user_id, source_filename, edit_prompt, source_path=None,
                       extra_metadata=None, h3_model=None, framing=None,
                       aspect_ratio=None):
    """Copy the reference into ComfyUI's input folder, build the H3 graph against
    what is ACTUALLY installed, and enqueue it. Returns the app job_id.

    Raises MinimaxH3ModelsMissing when an asset or a mandatory node is absent
    (checked BEFORE anything is copied or queued), ValueError on a missing
    source, RuntimeError when ComfyUI isn't configured.

    `aspect_ratio` is the catalog card's own ``W:H`` (both callers resolve it
    with `aspect_for_label`). It decides the output SHAPE; the reference still
    decides the pixel count. Absent or unusable keeps the reference's aspect.

    `framing` is accepted for signature parity with the Krea lane and is not read
    yet: H3 has no per-framing dial measured, and inventing one would be guessing.
    """
    if source_path is None:
        out_dir = cfg.comfyui_dir('output')
        if not out_dir:
            raise RuntimeError('ComfyUI is not configured')
        source_path = os.path.join(str(out_dir), source_filename)
    if not os.path.exists(source_path):
        raise ValueError(f'source image not found: {source_filename}')

    preflight()
    unet = resolve_h3_unet(h3_model or cfg.get('minimax_h3.base_model'))
    clip = resolve_h3_text_encoder()
    video_vae = resolve_h3_video_vae()
    audio_vae = resolve_h3_audio_vae()
    clip_vision = resolve_h3_clip_vision()

    # The URL being up says nothing about ComfyUI's input folder being reachable
    # from here — the same guard the Klein and Krea lanes use.
    comfy_input_dir = comfy_fs.ensure_input_usable(_comfy_input_dir())
    uid = uuid.uuid4().hex[:8]
    source_stem = os.path.splitext(os.path.basename(str(source_filename)))[0] or 'source'
    staged_source = comfy_fs.stage_input_image(
        source_path, f'h3_source_{uid}_{source_stem}.png', comfy_input_dir)
    comfy_input = os.path.basename(staged_source)

    src_w, src_h = _source_size(staged_source)
    # The catalog card decides the SHAPE, the reference decides how many pixels
    # (see fit_output_size). Both callers already resolve the card's ratio; H3
    # used to drop it and copy the reference's own aspect, so a square reference
    # answered a full-body card with a square.
    width, height = fit_output_size(src_w, src_h,
                                    max_mp=_cfg_float('minimax_h3.max_output_mp',
                                                      MAX_OUTPUT_MP),
                                    requested_aspect=aspect_ratio)
    optional = available_optional_nodes()
    workflow = build_workflow(
        comfy_input, edit_prompt, unet=unet, clip=clip, video_vae=video_vae,
        audio_vae=audio_vae, clip_vision=clip_vision, width=width, height=height,
        seed=random.randint(0, 2 ** 64 - 1),
        length=clamp_length(_cfg_int('minimax_h3.length', LENGTH_MIN)),
        steps=_cfg_int('minimax_h3.steps', DEFAULT_STEPS),
        ref_image_size=(cfg.get('minimax_h3.ref_image_size') or DEFAULT_REF_IMAGE_SIZE),
        ref_longer_edge=_cfg_int('minimax_h3.ref_longer_edge', DEFAULT_REF_LONGER_EDGE),
        frame_weight_reference=_cfg_float('minimax_h3.frame_weight_reference',
                                          DEFAULT_FRAME_WEIGHT_REFERENCE),
        # Config asks; /object_info decides. A user who leaves the toggle on
        # without the pack installed gets the image, not a validation error.
        use_speed_nodes=bool(cfg.get('minimax_h3.use_speed_nodes')) and optional['speed'],
        use_rtx_upscale=bool(cfg.get('minimax_h3.use_rtx_upscale')) and optional['upscale'],
        # UNIQUE prefix per job: SaveImage numbers from what is currently in the
        # output folder and the app moves each result out right after completion,
        # so a shared prefix makes the counter re-issue the same name.
        filename_prefix=f'{user_id}_DatasetH3_{uid}')

    job_id = str(uuid.uuid4())
    meta = {'model_name': 'minimax_h3_dataset'}
    if extra_metadata:
        meta.update(extra_metadata)
    meta['staged_inputs'] = [comfy_input]   # dropped again when the job ends
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt=edit_prompt,
                          job_id=job_id, metadata=meta)
    return job_id


# The PROMPT side of this engine (the `<Picture 1>` reference token, the outfit
# palette, the wrapper) lives in face_variations — the pure, DB-free, Flask-free
# module that owns every prompt in the app. Nothing prompt-shaped belongs here.
