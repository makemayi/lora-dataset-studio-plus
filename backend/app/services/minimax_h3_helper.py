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
dependency disappears. What it must NOT drop with it is the model's own canvas
rule — see the geometry section below; the first version did, and cut the top of
people's heads off for it.

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
    a thrashing install suggests. Taken at the OLD 992² canvas, so the sampling
    half of those numbers is now an upper bound — 768² is 40% fewer pixels.
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
# they swap the attention kernel and how much of it is held at once.
#
# `SpectrumApplyMiniMaxH3` was here until 2026-08-16 and forecast sampler steps
# instead. It left with the maintainer's graph redesign: its wins came from the
# middle of a long schedule, and both graphs now run a short one.
H3_SPEED_NODE_CLASSES = ('ModelAttentionBackend',
                         'MiniMaxH3MemoryEfficientSageAttentionPatch',
                         'PathchSageAttentionKJ')
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


# The Fl2VA sibling, as an asset in its own right. Everything above treats
# 'fl2va' as the file you must NOT pick — for the generation lane and for the
# original swap graph that is exactly right, because there it would load and
# then do a different job. `MiniMaxH3HybridLoader` (the new swap graph) is the
# one node that wants BOTH: Fl2VA as the base and Ref2VA laid over its last
# blocks. So this is a separate resolver rather than a relaxation of
# H3_WRONG_TASK_TOKENS — the exclusion still holds everywhere else, and only a
# caller that asks for Fl2VA by name can get it.
H3_FL2VA_ASSET = {
    'kind': 'MiniMax H3 Fl2VA model — base half of the hybrid loader '
            '(the new head-swap graph only)',
    'path': 'models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors',
    'source': f'{_H3_REPO}/tree/main/diffusion_models',
}


def resolve_h3_fl2va():
    """The Fl2VA diffusion model, or None. Not part of H3_REQUIRED: the
    generation engine and the old swap must not start demanding a file they
    never load."""
    return _find_model_file(
        'diffusion_models', H3_FL2VA_ASSET['path'].rsplit('/', 1)[-1], ('fl2va',))


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


#: The T1 image VAE the 2026-08-16 swap graph decodes through. NOT in
#: `H3_ASSETS`, on purpose: that dict IS `H3_REQUIRED`, so listing it there
#: would refuse H3 outright on every install that has only the video VAE — which
#: is every install that predates this file existing.
H3_IMAGE_VAE_FILE = 'minimax_h3_t1_image_vae_step1597.safetensors'


def resolve_h3_image_vae(selected=None):
    """The VAE both H3 lanes decode through: `selected` if it is still on disk,
    else the T1 image VAE, else the video VAE.

    Three levels because the three answers have different lifetimes. A picked
    file is a decision and wins. Absent one, the T1 IMAGE VAE is preferred —
    the maintainer's 2026-08-16 graphs decode a one-frame packet through it
    rather than the video VAE, which is a quality choice and not a rename. And
    the video VAE is the floor, because it is what every earlier graph used and
    it decodes the same latents: making the new file mandatory would turn an
    upgrade into a 409 for everyone who has not downloaded it.

    A STALE pick degrades to auto rather than failing, matching
    `resolve_h3_unet` — a filename that left the disk is a settings value gone
    out of date, not a reason to refuse the job."""
    if selected:
        picked = _find_model_file('vae', str(selected), ())
        if picked:
            return picked
        logger.warning('configured H3 VAE %r is not on disk — auto-resolving',
                       selected)
    return _find_model_file(
        'vae', H3_IMAGE_VAE_FILE, ('h3_t1_image_vae', 'h3-t1-image-vae'),
        exclude=('video', 'audio')) or resolve_h3_video_vae()


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


def h3_missing_assets(*, need_clip_vision=True):
    """Asset keys that do not resolve. Every gap at once — an install with
    nothing must produce a complete shopping list, not the first failure.

    `need_clip_vision` is the CALLER's answer, not a config read, because the
    three H3 graphs disagree about it and only the caller knows which one it is
    about to build. The vision tower exists to score a frame packet against the
    reference: the old swap always scores, the generation lane scores only when
    `length` > 1, and the 2026-08-16 swap takes frame 0 with a plain
    `ImageFromBatch` and never loads it. Reading `minimax_h3.length` here
    instead would have let a generation setting excuse an asset the OLD swap
    still needs — which is exactly the bug this parameter replaced."""
    keys = H3_REQUIRED if need_clip_vision else tuple(
        k for k in H3_REQUIRED if k != 'h3_clip_vision')
    return [key for key in keys if not _RESOLVERS[key]()]


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
    `files` shape, so one banner covers every engine.

    `h3_fl2va` is looked up here even though it is NOT in H3_REQUIRED: only the
    new swap graph loads it, but when that graph reports it missing the banner
    still has to name a file. A key with no entry would render an error with
    nothing to act on."""
    known = {**H3_ASSETS, 'h3_fl2va': H3_FL2VA_ASSET}
    out = []
    for key in missing or []:
        meta = known.get(key)
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
    the honest answer is a named gap, not a fake installer.

    The vision tower is only asked for when this lane is about to build a frame
    selector, i.e. when a packet is sampled — see `h3_missing_assets`."""
    packet = clamp_length(_cfg_int('minimax_h3.length', LENGTH_MIN)) > 1
    missing = h3_missing_assets(need_clip_vision=packet)
    nodes = h3_missing_nodes()
    if missing or nodes:
        raise MinimaxH3ModelsMissing(missing, nodes)


# --- Output geometry and packet length --------------------------------------
# THE CANVAS IS THE MODEL'S, NOT OURS.
#
# `comfy_extras/nodes_minimax_h3.py` defines `BASE_SHORT_EDGE = 768`,
# `MAX_PIXELS = 768 * 1344` and an `adapt_canvas(w, h)` that turns a ratio into
# the canvas H3 was trained on. The ref2va node does NOT call it — only the
# reference-VIDEO branch does — so the `width`/`height` we send land on the empty
# latent unchanged. Dropping `ResolutionSelector` therefore dropped the
# constraint along with the node, and the first version of this function put a
# flat 1 MP budget in its place:
#
#   card 1:1, 1024² reference -> 992x992   native 768x768   short edge +29%
#   card 3:4, 1024² reference -> 864x1152  native 768x1024  short edge +12%
#
# Off its trained canvas the DiT enlarges the subject rather than showing more of
# it, so the composition grows past the frame. On a head-and-shoulders card that
# reads exactly as reported: the top of the hair is cut off, worst on 1:1, which
# is also the framing that overshoots most. Reported 2026-08-09.
#
# So: the ratio still comes from the catalog card, and the SIZE now comes from
# the model. `max_output_mp` stays a cap on top, defaulting to the model's own.
BASE_SHORT_EDGE = 768             # nodes_minimax_h3.BASE_SHORT_EDGE
MAX_CANVAS_PIXELS = 768 * 1344    # nodes_minimax_h3.MAX_PIXELS
MAX_OUTPUT_MP = MAX_CANVAS_PIXELS / 1_000_000
_SIZE_MULTIPLE = 32          # the node's own step for width/height

# `length` is step 17 on the node — an off-grid value is a validation error at
# queue time, which means a whole batch of failed tiles.
#
# TWO legal shapes, not one range. The packet grid starts at 5 (5, 22, 39 ...),
# and 1 is the single-frame path: stock ComfyUI refuses it (`min=5`), a PATCHED
# `comfy_extras/nodes_minimax_h3.py` accepts it. We keep one frame anyway, so a
# packet of 5 samples four frames nobody reads — 1 is the cheap shape, and it
# also avoids the grid artefacts that pulling frame 0 out of a packet produces
# with the single-image VAE.
#
# THE PATCH IS THE PRECONDITION. On an unpatched ComfyUI `length=1` fails
# validation and takes the whole batch with it, and a ComfyUI update silently
# reverts the patch. That is why `caps` reports the H3 assets: if tiles start
# dying at queue time after an update, re-apply the patch or set length to 5.
LENGTH_MIN = 1               # the patched single-frame path
LENGTH_GRID_MIN = 5          # the packet grid's own floor
LENGTH_STEP = 17
LENGTH_MAX = 124             # our ceiling, not the node's (3600): stills, not film


def clamp_length(value):
    """Snap a requested packet length onto one of the node's two legal shapes."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = LENGTH_MIN
    n = max(LENGTH_MIN, min(LENGTH_MAX, n))
    if n < LENGTH_GRID_MIN:
        # Below the packet grid there is exactly one legal value, and rounding
        # DOWN keeps the old flooring behaviour rather than buying frames the
        # caller did not ask for.
        return LENGTH_MIN
    return LENGTH_GRID_MIN + ((n - LENGTH_GRID_MIN) // LENGTH_STEP) * LENGTH_STEP


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


def _snap(value):
    """Round one axis onto the node's 32 grid, never below one cell."""
    return max(_SIZE_MULTIPLE,
               int(round(float(value) / _SIZE_MULTIPLE)) * _SIZE_MULTIPLE)


def _model_canvas(ratio):
    """The canvas H3 was trained on for a ``W:H`` ratio.

    A port of `adapt_canvas` in `comfy_extras/nodes_minimax_h3.py`, deliberately
    kept numerically identical line for line: short edge 768, total area capped
    at 768*1344, each axis rounded to 32. The ref2va node never runs it on the
    generation size, so this is the only place it happens.

        1:1 -> 768x768    3:4 -> 768x1024    16:9 -> 1344x768
    """
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, float(BASE_SHORT_EDGE)
    else:
        nom_w, nom_h = float(BASE_SHORT_EDGE), BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_CANVAS_PIXELS:
        s = math.sqrt(MAX_CANVAS_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return _snap(nom_w), _snap(nom_h)


def fit_output_size(width, height, max_mp=None, requested_aspect=None):
    """The model's own canvas for this shot: a width/height, both multiples of
    32. The node takes plain INTs, so this replaces `ResolutionSelector` and its
    ratio enum entirely.

    A valid ``requested_aspect`` (``W:H``, the catalog card's own ratio) decides
    the SHAPE; without one, the source's aspect is kept. The SIZE is then the
    trained canvas for that shape (see `_model_canvas`), optionally capped by
    ``max_mp``. ``width``/``height`` are the reference's — they are read for
    their ratio only.

    Two rules used to live here and are gone on purpose:

    * a flat megapixel budget instead of the canvas. It overshot every ratio,
      worst on 1:1, and that overshoot is what crops the top of a head.
    * "never invent pixels the reference does not have". It sounds thrifty and
      it re-creates the same bug pointing down: a 640² reference would be
      answered at 640x832 for a 3:4 card, off-canvas again, and no pixel is
      copied from the reference anyway — H3 re-synthesises the whole frame from
      a latent, so a small reference does not make a small shot cheaper to get
      right. Cost is bounded by the canvas itself, which is smaller than the old
      budget for every ratio except the widest.
    """
    ratio = _aspect_ratio(requested_aspect)
    if ratio is None:
        ratio = max(1.0, float(width or 0)) / max(1.0, float(height or 0))
        if not (math.isfinite(ratio) and ratio > 0):
            ratio = 1.0
    out_w, out_h = _model_canvas(ratio)
    if max_mp is None:
        return out_w, out_h
    budget = float(max_mp) * 1_000_000
    if out_w * out_h <= budget:
        return out_w, out_h
    # A user cap below the trained canvas is honoured, off-canvas and all: it is
    # a config-only escape hatch for a card that cannot hold 768 short edge.
    scale = math.sqrt(budget / (out_w * out_h))
    out_w, out_h = _snap(out_w * scale), _snap(out_h * scale)
    # Rounding UP on both axes can cross the cap; step the long side back.
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

#: Both H3 lanes cap their LoRA list here. Not a technical limit — a stack that
#: deep is already unreadable, and the picker has to end somewhere.
MAX_H3_LORAS = 4


def sanitize_lora_rows(raw):
    """Ordered [{file, strength}] from a settings list: blank rows dropped,
    strengths clamped to [0, 1.5] (junk -> 1.0), capped at `MAX_H3_LORAS`.

    Shared by the generation lane and the swap so one list cannot accept a value
    the other rejects. Deliberately NOT capped at one row despite being called
    an "accelerator" slot: a step-distill and a subject LoRA stack legitimately,
    and a list that refuses the second one just moves the problem into a text
    field somewhere else."""
    rows = []
    for entry in (raw if isinstance(raw, list) else []):
        if not isinstance(entry, dict):
            continue
        name = entry.get('file')
        name = name.strip() if isinstance(name, str) else ''
        if not name:
            continue
        strength = entry.get('strength')
        strength = float(strength) if isinstance(strength, (int, float)) else 1.0
        rows.append({'file': name, 'strength': max(0.0, min(1.5, strength))})
        if len(rows) >= MAX_H3_LORAS:
            break
    return rows


def configured_h3_loras():
    """`minimax_h3.loras`, sanitized, with rows whose file is not on disk DROPPED.

    ComfyUI answers a validation 400 for the WHOLE job on one bad `lora_name`,
    so a stale row in a settings list would cost every tile of a batch. Skipping
    it costs that row's effect and says so in the log — the same trade the swap
    lane already made."""
    rows = []
    for row in sanitize_lora_rows(cfg.get('minimax_h3.loras')):
        found = _find_model_file('loras', row['file'], ())
        if not found:
            logger.warning('minimax_h3: LoRA %r is not on disk — skipping it',
                           row['file'])
            continue
        rows.append({'file': found, 'strength': row['strength']})
    return rows


def build_workflow(source_image, prompt, *, unet, clip, video_vae, audio_vae,
                   clip_vision, width, height, seed, length=LENGTH_MIN,
                   steps=DEFAULT_STEPS, sampler=DEFAULT_SAMPLER,
                   scheduler=DEFAULT_SCHEDULER,
                   ref_image_size=DEFAULT_REF_IMAGE_SIZE,
                   ref_longer_edge=DEFAULT_REF_LONGER_EDGE,
                   frame_weight_reference=DEFAULT_FRAME_WEIGHT_REFERENCE,
                   use_speed_nodes=True, use_rtx_upscale=True,
                   rtx_scale=DEFAULT_RTX_SCALE, fl2va=None, loras=(),
                   filename_prefix='minimax_h3'):
    """The ComfyUI API-format graph. Pure function of its arguments — no config
    read, no disk access — so a test can assert the exact wiring without a
    ComfyUI, and every loader value is one a resolver produced.

    `use_speed_nodes` / `use_rtx_upscale` are DEGRADATION switches, not features:
    False removes the node and rewires around it, so the same call produces a
    valid graph on an install that lacks the pack.

    `fl2va` selects the LOADER. With it, the graph runs the hybrid loader the
    maintainer's 2026-08-16 graphs use — Ref2VA laid over Fl2VA's last twenty
    blocks — which is a different model, not a faster one. Without it the plain
    `UNETLoader` path is built, because Fl2VA is a 66 GB download and an install
    that has only Ref2VA still generates perfectly well; making it mandatory
    would take the engine away from everyone who has not fetched it.

    `loras` is [{file, strength}] chained AFTER the speed patches, same rule and
    the same reason as the swap lane: the accelerator LoRAs this graph wants are
    community re-quantisations that differ per install, so no graph here can
    ship one and this is the only place one may come from.

    The seed reaches `RandomNoise` and nothing else. That is load-bearing:
    `MiniMaxH3ReferenceToVideo` is then identical across a card's copies and
    ComfyUI serves the 40-second encode from cache."""
    length = clamp_length(length)
    steps = max(1, int(steps))
    if fl2va:
        loader = {'class_type': 'MiniMaxH3HybridLoader',
                  'inputs': {'base_model': fl2va, 'overlay_model': unet,
                             # The maintainer's own preset. The block range is
                             # the last 20 of 50: earlier ranges were measured
                             # to drift identity, later ones to do nothing.
                             'overlay_preset': 'block_range_adaln',
                             'block_range_start': 30, 'block_range_end': 49,
                             'final_adaln_from_overlay': False,
                             'custom_overlays': '', 'custom_base': '',
                             'weight_dtype': 'default'},
                  '_meta': {'title': 'MiniMax H3 hybrid (Ref2VA over Fl2VA)'}}
    else:
        loader = {'class_type': 'UNETLoader',
                  'inputs': {'unet_name': unet, 'weight_dtype': 'default'},
                  '_meta': {'title': 'MiniMax H3 Ref2VA'}}
    g = {
        '1': loader,
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
    #
    # 2026-08-16: `SpectrumApplyMiniMaxH3` left this chain and the attention
    # backend + memory-efficient patch took its place, matching both of the
    # maintainer's graphs and the swap lane, which had already moved. Spectrum
    # forecast SAMPLER STEPS; these two change the attention kernel. Running the
    # forecaster on top of an 8-step distill predicts a curve that is nearly all
    # warmup, which is where its wins came from.
    model_out = ['1', 0]
    if use_speed_nodes:
        g['7'] = {'class_type': 'ModelAttentionBackend',
                  'inputs': {'attention': 'comfy kitchen attention',
                             'model': model_out},
                  '_meta': {'title': 'Attention backend (optional)'}}
        model_out = ['7', 0]
        g['8'] = {'class_type': 'MiniMaxH3MemoryEfficientSageAttentionPatch',
                  'inputs': {'model': model_out},
                  '_meta': {'title': 'H3 memory-efficient attention (optional)'}}
        model_out = ['8', 0]
        g['20'] = {'class_type': 'PathchSageAttentionKJ',
                   'inputs': {'sage_attention': 'auto', 'allow_compile': False,
                              'model': model_out},
                   '_meta': {'title': 'Sage attention (optional)'}}
        model_out = ['20', 0]

    # Accelerator / subject LoRAs, after the patches so switching `use_speed_nodes`
    # off does not move where they attach. A blank file or a zero strength is a
    # row that is off, and it is skipped rather than sent as an empty lora_name —
    # ComfyUI answers a validation 400 for the WHOLE job on one of those.
    for i, row in enumerate(loras or ()):
        name = (row.get('file') or '').strip() if isinstance(row, dict) else ''
        strength = row.get('strength', 1.0) if isinstance(row, dict) else 0
        if not name or not strength:
            continue
        node_id = f'lora_{i}'
        g[node_id] = {'class_type': 'LoraLoaderModelOnly',
                      'inputs': {'lora_name': name,
                                 'strength_model': float(strength),
                                 'model': model_out},
                      '_meta': {'title': f'H3 LoRA {i + 1} (Settings)'}}
        model_out = [node_id, 0]

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
    # ONE frame comes back either way, but by two different routes, and which
    # one is right is decided by how many frames there are to choose between.
    #
    # A packet (`length` > 1) is a choice, and `H3FrameSelect` scores it against
    # the reference. A single frame is not a choice: scoring it costs the
    # MinimaxH3-Image pack, a CLIP-ViT-H download and a vision pass to rank a
    # list of one. `length` defaults to 1, so the common install now needs
    # neither — which is what the maintainer's 2026-08-16 graph does too.
    if length > 1:
        g['16'] = {'class_type': 'CLIPVisionLoader',
                   'inputs': {'clip_name': clip_vision}}
        g['17'] = {'class_type': 'H3FrameSelect',
                   'inputs': {'images': ['15', 0],
                              'select_count': 1,
                              'weight_sharpness': 1.0,
                              'weight_exposure': 0.5,
                              # Non-zero on purpose: a dataset wants the frame
                              # that looks most like the reference, not merely
                              # the sharpest one. See the module docstring -
                              # UNVERIFIED.
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
    else:
        g['17'] = {'class_type': 'ImageFromBatch',
                   'inputs': {'batch_index': 0, 'length': 1,
                              'image': ['15', 0]},
                   '_meta': {'title': 'The single frame'}}

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
    with `aspect_for_label`). It decides the output SHAPE; the model's trained
    canvas for that shape decides the size. Absent or unusable keeps the
    reference's aspect.

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
    # `minimax_h3.vae`: blank auto-resolves (T1 image VAE if present, else the
    # video VAE). The generation lane samples a one-frame packet exactly like
    # the swap does, so both go through the same picker rather than one lane
    # quietly decoding through a different VAE than the other.
    video_vae = resolve_h3_image_vae(cfg.get('minimax_h3.vae'))
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
    # The catalog card decides the SHAPE, the model's trained canvas decides the
    # size (see fit_output_size). The reference is read for its ratio only, and
    # only when the card has none. Both callers already resolve the card's ratio;
    # H3 used to drop it and copy the reference's own aspect, so a square
    # reference answered a full-body card with a square.
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
        # The hybrid loader when Fl2VA is on disk, the plain UNETLoader when it
        # is not — see `build_workflow`. Resolved here rather than inside it so
        # the builder stays a pure function of its arguments.
        fl2va=resolve_h3_fl2va(),
        loras=configured_h3_loras(),
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
