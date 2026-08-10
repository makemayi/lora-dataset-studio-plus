"""Enqueue a SINGLE fixed MiniMax H3 head-swap ComfyUI job — the SECOND engine
behind the 🎭↔ button, next to the Klein one in face_swap_helper.

Node 313 = the tile's current image (the TARGET, whose head is masked out and
repainted), node 114 = the dataset's reference photo (the identity SOURCE).
Ships 'minimax h3 swap.json'.

WHAT THE GRAPH DOES (and how it differs from the Klein swap)
------------------------------------------------------------
Klein repaints the head with a diffusion edit driven by a swap LoRA. This one
has no swap LoRA at all: it masks the head on the target, hands H3 TWO
references — the identity photo and the masked target — and lets the video
model re-stage the head into the hole, then stitches the crop back:

    313 target -> Resize(1536) -> [hair removal] -> PersonMaskUltra(face+hair+
        body) -> InpaintCropImproved -> GrowMaskWithBlur -> [LaMa] ->
        AILab_MaskOverlay
    114 reference -> RTXVideoSuperResolution -> Resize(1024)
        -> MiniMaxH3ReferenceToVideo(ref_image_0 = identity,
           ref_image_1 = the masked crop) -> SamplerCustomAdvanced -> VAEDecode
        -> H3FrameSelect(1) -> [face detail] -> InpaintStitchImproved
        -> RTXVideoSuperResolution -> SaveImage

So it inherits H3's own economics wholesale — see minimax_h3_helper's measured
notes, including `--disable-dynamic-vram`, which matters MORE here because the
graph also loads the segmentation and inpaint stacks alongside 40 GB of H3.

THE THREE OPTIONAL STAGES
-------------------------
The three bracketed steps above are the nodes the maintainer runs BYPASSED and
un-bypasses to compare. They ship WIRED, and each setting under
`face_swap.h3_stages` removes its stage when off — the code only ever subtracts,
which is the only direction that cannot invent wiring nobody measured:

  hair_removal  a full Klein 9B edit pass ("remove the hair, change nothing
                else") on the target BEFORE the head is masked. Costs a second
                diffusion model in the same job, and needs the Klein assets —
                so with it on, a missing Klein file blocks the swap.
  lama          LaMa inpainting (`LayerUtility: LaMa`, zits) wipes the masked
                head region before H3 sees it, so the model is not looking at
                the old head through a translucent mask.
  face_detail   a Z-Image Turbo detailer pass over the eyes and mouth of the
                selected frame, before it is stitched back.

`face_detail` is the one place this module does NOT re-resolve loader
filenames: it names a Z-Image checkpoint, a lumina2 text encoder and a private
LoRA stack, and this app has no Z-Image resolver to route them through. Its
files are whatever the graph says, so switching it on with a different ComfyUI
answers a validation error naming the file. Off by default for that reason.

WHAT IS RESOLVED, WHAT IS LEFT ALONE
------------------------------------
* Every H3 loader value is re-resolved through minimax_h3_helper's resolvers,
  and the hair-removal stage's Klein loaders through klein_edit_helper's — the
  committed graph names the maintainer's own files, which exist on no other
  install (the trap klein_edit_helper spends half its length on).
* Both `OTUNetLoaderW8A8` nodes are kept when an INT8/W8A8 build is on disk and
  degraded to core `UNETLoader` when not. Handing a W8A8 loader an fp16
  checkpoint is not a slow success, it is a failure.
* The two RTX upscalers and the two speed nodes follow the H3 engine's own
  `use_rtx_upscale` / `use_speed_nodes` settings AND /object_info: an install
  without the NVIDIA pack loses the 2x, not the swap.
* The H3 PROMPT comes from the graph unless `minimax_h3.swap_prompt` overrides
  it, which is the key to A/B a wording without editing a file an update
  replaces.

THE PROMPT, AND WHAT IT IS TALKING ABOUT
----------------------------------------
`<Picture 1>` is the identity photo. `<Picture 2>` is NOT the tile: it is the
inpaint CROP with the mask painted solid white over it (AILab_MaskOverlay). The
white area is therefore exactly the region being repainted — the HEAD, face and
hair, per the maintainer's own mask (2026-08-10). Which is why every clause is
about a head sitting on a neck and shoulders that must not change: the mask
does not cover them, so anything the prompt makes the model reconsider down
there comes back as a seam.

HOW MUCH OF THE SHOT <Picture 2> SHOWS
---------------------------------------
`face_swap.h3_context_factor` (see `context_factor`). The crop is grown from the
mask box and clamped to the frame, so one number adapts to the photo: a
full-body shot gets head plus chest, a portrait clamps to its own edges and is
not cropped at all. It shipped at 1.3 — head only, everywhere — which is the
most pixels per head and also leaves the model nothing to size the head
against, since the prompt asks it to match shoulders that were cropped away.

PersonMaskUltra is therefore set to face + hair and NOTHING else. It shipped
with `body` on as well, which is a person-shaped mask: InpaintCropImproved
crops to whatever the mask covers, so that made the repainted region grow down
the torso — a much larger area for H3 to re-invent, and a prompt about a head
describing the wrong region. Turned off 2026-08-10 at the maintainer's word
that the painted area is the head.

The wording was generalised on 2026-08-10. It used to name one subject ("black
short hair woman"), which is a fact about the reference it was tuned on: any
other subject put the prompt in a fight with the picture, and the model split
the difference. What it now spells out instead is the four things that fail
silently — keep the identity of <Picture 1>, change nothing outside the white
area, match <Picture 2>'s head angle and lighting, and leave no seam at the
neck and hairline.
"""
from __future__ import annotations
import logging
import os
import random
import uuid

from .. import config as cfg
from . import minimax_h3_helper as mh
from ..utils import comfy_fs
from ..utils.comfyui import load_workflow_local
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

WORKFLOW_H3_SWAP_PATH = cfg.BACKEND_DIR / 'workflows' / 'minimax h3 swap.json'

# Nodes this helper rewires — fail LOUDLY if the workflow file changes shape.
# The helper writes into nodes BY ID, so a graph swap nobody notices does not
# crash: it silently produces the wrong picture, or drops the seed, or saves
# under ComfyUI's own prefix so every tile displays the same file.
NODE_TARGET_IMAGE = '313'        # LoadImage — the tile being repainted
NODE_REF_IMAGE = '114'           # LoadImage — the identity to graft on
NODE_H3_UNET = '426:312'         # OTUNetLoaderW8A8 (may degrade to UNETLoader)
NODE_H3_CLIP = '426:130'         # CLIPLoader, type='minimax'
NODE_VIDEO_VAE = '426:121'
NODE_AUDIO_VAE = '426:172'       # required by the node even for a still
NODE_CLIP_VISION = '426:305'     # scores which frame of the packet to keep
NODE_H3 = '426:170'              # MiniMaxH3ReferenceToVideo — the prompt lives here
NODE_LENGTH = '426:139'          # PrimitiveInt -> frame packet length
NODE_SEED = '426:131'            # RandomNoise -> 'noise_seed'
NODE_CROP = '426:402'            # InpaintCropImproved — how much shot H3 sees
NODE_MASK_OVERLAY = '426:413'    # AILab_MaskOverlay — paints the hole H3 fills
NODE_FRAME_SELECT = '426:304'    # H3FrameSelect — which frame of the packet wins
NODE_TARGET_RESIZE = '426:401'   # ResizeImagesByLongerEdge — what the mask must match
NODE_PERSON_MASK = '426:340'     # LayerMask: PersonMaskUltra — the in-graph masker
NODE_SAVE = '412'

# Ids for the nodes that carry an APP-SIDE mask into the graph. Named rather
# than numbered because they do not come from the maintainer's export.
NODE_APP_MASK_LOAD = 'app_mask_load'
NODE_APP_MASK_RESIZE = 'app_mask_resize'
NODE_APP_MASK_TO_MASK = 'app_mask_to_mask'

_REQUIRED_NODES = (NODE_TARGET_IMAGE, NODE_REF_IMAGE, NODE_H3_UNET, NODE_H3_CLIP,
                   NODE_VIDEO_VAE, NODE_AUDIO_VAE, NODE_CLIP_VISION, NODE_H3,
                   NODE_LENGTH, NODE_SEED, NODE_CROP, NODE_MASK_OVERLAY,
                   NODE_FRAME_SELECT, NODE_SAVE)

# How opaque the placeholder painted over the head is, in `face_swap.h3_mask_opacity`.
# AILab_MaskOverlay composites a flat colour with `mask * opacity` as its alpha,
# so 1.0 replaces the head with a structureless white slab — and a generative
# model asked to fill a white slab sometimes just draws the slab back. That is
# the "white face" result. Below 1.0 a ghost of the original head shows through
# and gives the model geometry to work with; too low and the OLD identity starts
# coming back, which is the thing the swap exists to remove.
DEFAULT_MASK_OPACITY = 1.0


def mask_opacity():
    """Opacity of the placeholder over the head, clamped to [0, 1]. Junk falls
    back to the default: this tunes a result, it cannot make one invalid."""
    raw = cfg.get('face_swap.h3_mask_opacity')
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_MASK_OPACITY
    if value != value:                          # NaN
        return DEFAULT_MASK_OPACITY
    return max(0.0, min(1.0, value))

# `context_from_mask_extend_factor` bounds. The node itself allows up to 100,
# which is meaningless here: past the point where the crop covers the frame the
# number stops doing anything, and every value in between only costs head
# pixels. 1.0 is "crop to the mask box exactly".
CONTEXT_FACTOR_MIN = 1.0
CONTEXT_FACTOR_MAX = 8.0
DEFAULT_CONTEXT_FACTOR = 3.0


def context_factor():
    """How far the crop reaches around the head, from `face_swap.h3_context_factor`.

    The crop is grown from the MASK box and then clamped to the image, so one
    number adapts to the shot by itself: at 3.0 a full-body photo crops to head
    and chest, while a bust or a head-and-shoulders — where the head already
    fills the frame — clamps to the edges and is not cropped at all. That is the
    whole reason this is a factor and not a pixel size.

    Junk falls back to the default rather than raising: this decides framing, not
    correctness, and a bad value in config must not refuse the swap."""
    raw = cfg.get('face_swap.h3_context_factor')
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_CONTEXT_FACTOR
    if not (value == value and value not in (float('inf'), float('-inf'))):
        return DEFAULT_CONTEXT_FACTOR
    return max(CONTEXT_FACTOR_MIN, min(CONTEXT_FACTOR_MAX, value))

# Optional accelerators: (node id, the input carrying its pass-through).
NODE_SPEED_SPECTRUM = '426:310'  # SpectrumApplyMiniMaxH3   -> 'model'
NODE_SPEED_SAGE = '426:187'      # PathchSageAttentionKJ    -> 'model'
NODE_UPSCALE_REF = '426:314'     # RTXVideoSuperResolution  -> 'images' (reference)
NODE_UPSCALE_OUT = '426:309'     # RTXVideoSuperResolution  -> 'images' (result)

_SPEED_NODES = ((NODE_SPEED_SPECTRUM, 'model'), (NODE_SPEED_SAGE, 'model'))
_UPSCALE_NODES = ((NODE_UPSCALE_REF, 'images'), (NODE_UPSCALE_OUT, 'images'))

# --- The three optional stages ----------------------------------------------
# Each is declared by its TAIL (the node whose output the rest of the graph
# consumes) and by what that output falls back to when the stage is off. Those
# fallbacks are not a reconstruction: they are what the maintainer's own
# BYPASSED export wired, read off it node by node.
#
#   hair_removal  Klein edit branch, tail VAEDecode -> the resized target
#   lama          LaMa inpaint,      tail its own   -> the inpaint crop
#   face_detail   Z-Image detailer,  tail Detailer  -> the selected frame
#
# Switching a stage off repoints its consumers and lets `prune_to_outputs` drop
# whatever is now unreachable — so the stage's loaders, its samplers and its
# custom-node requirements all leave the job together.
NODE_HAIR_REMOVAL_TAIL = '426:423:65'
NODE_HAIR_REMOVAL_KLEIN_UNET = '426:423:424'   # OTUNetLoaderW8A8 (Klein INT8)
NODE_HAIR_REMOVAL_CLIP = '426:423:71'
NODE_HAIR_REMOVAL_VAE = '426:423:72'
NODE_HAIR_REMOVAL_SEED = '426:423:73'          # RandomNoise -> 'noise_seed'
NODE_LAMA_TAIL = '426:411'          # LayerUtility: LaMa — also its own model pick
NODE_FACE_DETAIL_TAIL = '426:396:370'          # DetailerForEach -> 'seed'

STAGES = {
    'hair_removal': (NODE_HAIR_REMOVAL_TAIL, ['426:401', 0]),
    'lama': (NODE_LAMA_TAIL, ['426:402', 1]),
    'face_detail': (NODE_FACE_DETAIL_TAIL, ['426:304', 0]),
}
STAGE_LABELS = {
    'hair_removal': 'Klein hair removal',
    'lama': 'LaMa head cleanup',
    'face_detail': 'Z-Image face detail',
}

# Tokens that mark a build the W8A8 loader can actually take.
_INT8_TOKENS = ('int8', 'w8a8')

# Pack attribution for the custom nodes THIS graph needs, on top of the H3
# frame-selector pack minimax_h3_helper already knows about. Same
# {pack, url, search} shape as H3_FRAME_SELECT_PACK, so the existing preflight
# banner renders these without a second code path.
H3_SWAP_NODE_PACKS = {
    'LayerMask: PersonMaskUltra': {
        'pack': 'ComfyUI_LayerStyle',
        'url': 'https://github.com/chflame163/ComfyUI_LayerStyle',
        'search': 'LayerStyle'},
    'LayerUtility: LaMa': {
        'pack': 'ComfyUI_LayerStyle',
        'url': 'https://github.com/chflame163/ComfyUI_LayerStyle',
        'search': 'LayerStyle'},
    'InpaintCropImproved': {
        'pack': 'ComfyUI-Inpaint-CropAndStitch',
        'url': 'https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch',
        'search': 'Inpaint CropAndStitch'},
    'InpaintStitchImproved': {
        'pack': 'ComfyUI-Inpaint-CropAndStitch',
        'url': 'https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch',
        'search': 'Inpaint CropAndStitch'},
    'GrowMaskWithBlur': {
        'pack': 'ComfyUI-KJNodes',
        'url': 'https://github.com/kijai/ComfyUI-KJNodes',
        'search': 'KJNodes'},
    'PathchSageAttentionKJ': {
        'pack': 'ComfyUI-KJNodes',
        'url': 'https://github.com/kijai/ComfyUI-KJNodes',
        'search': 'KJNodes'},
    'AILab_MaskOverlay': {
        'pack': 'ComfyUI-RMBG',
        'url': 'https://github.com/1038lab/ComfyUI-RMBG',
        'search': 'RMBG'},
    'FaceSegment': {
        'pack': 'ComfyUI-RMBG',
        'url': 'https://github.com/1038lab/ComfyUI-RMBG',
        'search': 'RMBG'},
    'MaskToSEGS': {
        'pack': 'ComfyUI-Impact-Pack',
        'url': 'https://github.com/ltdrdata/ComfyUI-Impact-Pack',
        'search': 'Impact Pack'},
    'DetailerForEach': {
        'pack': 'ComfyUI-Impact-Pack',
        'url': 'https://github.com/ltdrdata/ComfyUI-Impact-Pack',
        'search': 'Impact Pack'},
    'Power Lora Loader (rgthree)': {
        'pack': 'rgthree-comfy',
        'url': 'https://github.com/rgthree/rgthree-comfy',
        'search': 'rgthree'},
    'RTXVideoSuperResolution': {
        'pack': 'Nvidia_RTX_Nodes_ComfyUI',
        'url': 'https://github.com/NVIDIA/Nvidia_RTX_Nodes_ComfyUI',
        'search': 'Nvidia RTX Nodes'},
    'H3ImageResolutionPreset': dict(mh.H3_FRAME_SELECT_PACK),
    'H3FrameSelect': dict(mh.H3_FRAME_SELECT_PACK),
    'OTUNetLoaderW8A8': {
        'pack': 'unknown pack — without an INT8 build the swap drops to the '
                'core UNETLoader, so this node is only needed for one',
        'url': None, 'search': 'OTUNetLoaderW8A8'},
    'ZImageTurboLoraLoader': {
        'pack': 'unknown pack — only the optional Z-Image face-detail stage '
                'needs it; switch that stage off to run without it',
        'url': None, 'search': 'ZImageTurboLoraLoader'},
    'SpectrumApplyMiniMaxH3': {
        'pack': 'unknown pack — optional speed node; switch off '
                'minimax_h3.use_speed_nodes to run without it',
        'url': None, 'search': 'SpectrumApplyMiniMaxH3'},
}


def node_hints(nodes):
    """[{class_type, pack, url, search}] for each missing node — the shape the
    Studio preflight banner already renders. An unknown class still gets a row,
    with the class name as the search term: a named gap beats a silent one."""
    out = []
    for ct in (nodes or []):
        meta = H3_SWAP_NODE_PACKS.get(ct, {'pack': None, 'url': None, 'search': ct})
        out.append({'class_type': ct, **meta})
    return out


class H3SwapNodesMissing(mh.MinimaxH3ModelsMissing):
    """The target ComfyUI lacks a custom node THIS graph needs.

    A subclass rather than a new type: every caller and route branch that
    already handles MinimaxH3ModelsMissing keeps working, and the only thing
    added is `node_packs` — the swap graph pulls nodes from six packs the
    generation lane never touches, so "install MinimaxH3-Image" would be the
    wrong instruction most of the time."""

    def __init__(self, nodes):
        super().__init__([], nodes)
        self.node_packs = node_hints(nodes)


MASK_SOURCES = ('graph', 'app')
# A HEAD IS NOT ONE OBJECT.
# To an open-vocabulary segmenter, 'head' is the head and the glasses on it are
# something else — so a one-word head mask comes back with a hole exactly where
# the accessories are, and the swap paints a new head around the old spectacles.
# Everything that sits ON the head and would go with it belongs in the list;
# their masks are unioned. A phrase that matches nothing in a given photo costs
# one cheap grounding pass and contributes nothing, which is why naming the
# accessories that are usually absent is still the right default.
DEFAULT_MASK_PROMPT = 'head, glasses, sunglasses, hat, headband, earrings'

# iopaint's own default and the only one measured to survive an arbitrarily
# sized crop here. The node offers lama / ldm / zits / mat / fcf / manga /
# spread; they differ in look, not in that constraint.
DEFAULT_LAMA_MODEL = 'lama'


def mask_source():
    """Where the head mask comes from: 'graph' (PersonMaskUltra inside the
    workflow) or 'app' (services/auto_mask, SAM 3 in the app's own interpreter).

    Fail-SAFE like every other engine pick here: an unknown value falls back to
    the graph rather than refusing the swap."""
    name = str(cfg.get('face_swap.h3_mask_source') or '').strip().lower()
    if name in MASK_SOURCES:
        return name
    if name:
        logger.warning('unknown h3 mask source %r — using the graph', name)
    return 'graph'


def mask_prompt():
    """The phrases the app-side masker is given, comma-separated. Open-vocabulary,
    so this is the one place the masked REGION is decided: face plus hair cut at
    the jaw, PLUS whatever is worn on the head, which is what the swap's prompt
    and its stitch both assume."""
    value = cfg.get('face_swap.h3_mask_prompt')
    value = value.strip() if isinstance(value, str) else ''
    return value or DEFAULT_MASK_PROMPT


def attach_app_mask(workflow, mask_image):
    """Feed an app-produced mask PNG into the graph in place of PersonMaskUltra.

    The mask arrives at the TILE's own resolution while the graph masks a copy
    resized to `longer_edge`, so it is put through the SAME
    `ResizeImagesByLongerEdge` node with the SAME value rather than resized
    app-side: that node truncates (`new_h = int(h * (edge / w))`), and
    reproducing its arithmetic somewhere else is how an off-by-one size mismatch
    gets discovered by ComfyUI at queue time instead of here.

    PersonMaskUltra is left unwired and disappears in `prune_to_outputs` — with
    it goes the ComfyUI_LayerStyle dependency, unless the LaMa stage is on."""
    longer_edge = ((workflow.get(NODE_TARGET_RESIZE) or {}).get('inputs', {})
                   .get('longer_edge', 1536))
    workflow[NODE_APP_MASK_LOAD] = {
        'class_type': 'LoadImage', 'inputs': {'image': mask_image},
        '_meta': {'title': 'Mask (from the app)'}}
    workflow[NODE_APP_MASK_RESIZE] = {
        'class_type': 'ResizeImagesByLongerEdge',
        'inputs': {'longer_edge': longer_edge,
                   'images': [NODE_APP_MASK_LOAD, 0]},
        '_meta': {'title': 'Mask, to the same geometry as the target'}}
    workflow[NODE_APP_MASK_TO_MASK] = {
        'class_type': 'ImageToMask',
        'inputs': {'image': [NODE_APP_MASK_RESIZE, 0], 'channel': 'red'},
        '_meta': {'title': 'Mask'}}
    workflow[NODE_CROP]['inputs']['mask'] = [NODE_APP_MASK_TO_MASK, 0]
    return NODE_APP_MASK_TO_MASK


def enabled_stages():
    """{stage: bool} from `face_swap.h3_stages`, defaulting to OFF.

    Off is the honest default for all three: each adds a second model family to
    a job that already loads 40 GB of H3, and the face-detail one names files
    this app cannot resolve. They are comparison switches, and the maintainer
    runs them bypassed most of the time."""
    raw = cfg.get('face_swap.h3_stages')
    raw = raw if isinstance(raw, dict) else {}
    return {name: bool(raw.get(name)) for name in STAGES}


def prune_to_outputs(workflow):
    """Keep SaveImage and everything it depends on; drop everything else.

    Not a tidy-up: ComfyUI executes every OUTPUT node in a prompt, so a
    PreviewImage left in a headless job costs its pack for a picture nobody
    sees, and an unreachable branch can still fail validation for the whole
    prompt. It is also how a switched-off stage disappears completely — its
    loaders and its custom-node requirements leave with it. Returns a NEW dict."""
    keep, stack = set(), [nid for nid, n in (workflow or {}).items()
                          if isinstance(n, dict) and n.get('class_type') == 'SaveImage']
    while stack:
        nid = stack.pop()
        if nid in keep or nid not in workflow:
            continue
        keep.add(nid)
        for value in (workflow[nid].get('inputs') or {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                stack.append(value[0])
    return {nid: node for nid, node in workflow.items() if nid in keep}


def repoint(workflow, from_ref, to_ref):
    """Every input reading `from_ref` now reads `to_ref`. Returns the count.

    `from_ref` is matched on the NODE only ([id, slot] with any slot), because a
    stage tail has exactly one output and matching the slot too would silently
    skip a consumer wired to a second one."""
    moved = 0
    for node in workflow.values():
        for key, value in (node.get('inputs') or {}).items():
            if isinstance(value, list) and len(value) == 2 and value[0] == from_ref:
                node['inputs'][key] = list(to_ref)
                moved += 1
    return moved


def apply_stages(workflow, stages):
    """Bypass every stage that is off, by repointing its consumers at what the
    stage's own bypassed export fell back to. Returns the ids left in place."""
    kept = []
    for name, (tail, fallback) in STAGES.items():
        if stages.get(name):
            kept.append(name)
            continue
        if tail in workflow and not repoint(workflow, tail, fallback):
            logger.debug('h3 swap: stage %s has no consumers to repoint', name)
    return kept


def drop_passthrough(workflow, node_id, input_key):
    """Remove a pass-through node and rewire every consumer to its own source.
    Returns True when the node was removed. Same helper the Klein swap uses for
    its optional style LoRA — copied rather than imported, because that module's
    version defaults to 'model' and half the nodes here pass IMAGE."""
    node = workflow.get(node_id)
    if node is None:
        return False
    upstream = (node.get('inputs') or {}).get(input_key)
    if not (isinstance(upstream, list) and len(upstream) == 2):
        return False
    repoint(workflow, node_id, upstream)
    workflow.pop(node_id, None)
    return True


def workflow_class_types(workflow):
    return {n.get('class_type') for n in (workflow or {}).values()
            if isinstance(n, dict) and n.get('class_type')}


def missing_nodes(workflow):
    """[class_type] this graph needs that the target ComfyUI does not expose.

    FAIL-OPEN, like every other node probe in the app: [] when /object_info is
    unreachable, because a probe timeout is not evidence of a missing pack and
    blocking there would refuse a swap on a working install.

    Call it AFTER the optional stages and accelerators have been removed — an
    install without the Impact Pack must not be told to install the Impact Pack
    for a stage the job no longer contains."""
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    return sorted(workflow_class_types(workflow) - available)


def _int8(name):
    return any(tok in os.path.basename(str(name or '')).lower() for tok in _INT8_TOKENS)


def _degrade_w8a8(workflow, node_id, unet_name, title):
    """Point a W8A8 loader at `unet_name`, or replace it with the core loader
    when that file is not an INT8 build. Not a performance choice: the W8A8
    loader FAILS on an fp8/fp16 checkpoint."""
    if node_id not in workflow:
        return
    if not _int8(unet_name):
        workflow[node_id] = {'class_type': 'UNETLoader',
                             'inputs': {'weight_dtype': 'default'},
                             '_meta': {'title': title}}
    workflow[node_id]['inputs']['unet_name'] = unet_name


def _cfg_float(key, default):
    try:
        return float(cfg.get(key))
    except (TypeError, ValueError):
        return default


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def build_swap_workflow(target_image, ref_image, *, filename_prefix, stages=None,
                        mask_image=None):
    """Load the shipped graph, apply the stage switches, resolve every loader
    against what is actually installed, and return (workflow, stages_kept).

    Split out of `enqueue_h3_swap` so a test can assert the exact wiring without
    a ComfyUI or a queue. It still reads config and disk — the resolvers are the
    whole point — but it enqueues nothing."""
    raw = load_workflow_local(str(WORKFLOW_H3_SWAP_PATH))
    if not raw:
        raise ValueError('failed to load the MiniMax H3 swap workflow')
    workflow = dict(raw)
    for node in _REQUIRED_NODES:
        if node not in workflow:
            raise ValueError(f'workflow node {node} missing — '
                             'minimax h3 swap.json has changed')

    stages = enabled_stages() if stages is None else dict(stages)
    kept = apply_stages(workflow, stages)
    # An app-produced mask replaces PersonMaskUltra BEFORE the node probe, so an
    # install without ComfyUI_LayerStyle is never told to install a pack for a
    # node this job no longer contains.
    if mask_image:
        attach_app_mask(workflow, mask_image)

    # Config asks; /object_info decides. A user who leaves an accelerator on
    # without the pack installed gets the image, not a validation 400.
    optional = mh.available_optional_nodes()
    want_speed = bool(cfg.get('minimax_h3.use_speed_nodes')) and optional['speed']
    want_upscale = bool(cfg.get('minimax_h3.use_rtx_upscale')) and optional['upscale']
    for group, wanted in ((_SPEED_NODES, want_speed), (_UPSCALE_NODES, want_upscale)):
        if not wanted:
            for node_id, input_key in group:
                drop_passthrough(workflow, node_id, input_key)

    # H3's five assets, resolved through the generation lane's own resolvers: an
    # install that can generate with H3 can swap with it, and a pinned
    # `minimax_h3.base_model` applies to both.
    missing = mh.h3_missing_assets()
    if missing:
        raise mh.MinimaxH3ModelsMissing(missing)
    h3_unet = mh.resolve_h3_unet(cfg.get('minimax_h3.base_model'))
    _degrade_w8a8(workflow, NODE_H3_UNET, h3_unet,
                  'MiniMax H3 (no INT8 build found)')
    workflow[NODE_H3_CLIP]['inputs']['clip_name'] = mh.resolve_h3_text_encoder()
    workflow[NODE_VIDEO_VAE]['inputs']['vae_name'] = mh.resolve_h3_video_vae()
    workflow[NODE_AUDIO_VAE]['inputs']['vae_name'] = mh.resolve_h3_audio_vae()
    workflow[NODE_CLIP_VISION]['inputs']['clip_name'] = mh.resolve_h3_clip_vision()

    # The hair-removal stage is a full Klein edit, so with it ON the Klein
    # assets become required — and are resolved the same way every other Klein
    # call site resolves them, never from the filenames in the graph.
    if 'hair_removal' in kept:
        from .klein_edit_helper import (KLEIN_REQUIRED, KleinModelsMissing,
                                        klein_missing_assets,
                                        resolve_klein_text_encoder,
                                        resolve_klein_vae)
        from .face_swap_helper import resolve_swap_unet
        klein_gaps = klein_missing_assets()
        if any(a in klein_gaps for a in KLEIN_REQUIRED):
            raise KleinModelsMissing(klein_gaps)
        klein_unet, _int8_found = resolve_swap_unet()
        _degrade_w8a8(workflow, NODE_HAIR_REMOVAL_KLEIN_UNET, klein_unet,
                      'Klein (no INT8 build found)')
        workflow[NODE_HAIR_REMOVAL_CLIP]['inputs']['clip_name'] = \
            resolve_klein_text_encoder()
        workflow[NODE_HAIR_REMOVAL_VAE]['inputs']['vae_name'] = resolve_klein_vae()
        workflow[NODE_HAIR_REMOVAL_SEED]['inputs']['noise_seed'] = \
            random.randint(0, 2 ** 64 - 1)
    # The face-detail stage's Z-Image files are deliberately NOT resolved — see
    # the module docstring. Its seed still has to move, or every tile of a batch
    # gets the identical detailer pass.
    # The LaMa stage's inpainting model. The graph asks for 'zits', which CANNOT
    # run on this lane: ZITS pads to a multiple of 32 and drives a 256->512
    # structure upsampler, and the inpaint crop here is an arbitrary size
    # (output_resize_to_target_size is off), so it dies inside TorchScript at
    # `upsample_bilinear2d`. Reported as "enabling LaMa errors", 2026-08-11.
    if 'lama' in kept and NODE_LAMA_TAIL in workflow:
        model = cfg.get('face_swap.h3_lama_model')
        model = model.strip() if isinstance(model, str) else ''
        workflow[NODE_LAMA_TAIL]['inputs']['lama_model'] = model or DEFAULT_LAMA_MODEL

    if 'face_detail' in kept and NODE_FACE_DETAIL_TAIL in workflow:
        workflow[NODE_FACE_DETAIL_TAIL]['inputs']['seed'] = \
            random.randint(0, 2 ** 64 - 1)

    # Everything a switched-off stage or accelerator left behind goes now, so
    # the node probe below asks about the job that will actually run.
    workflow = prune_to_outputs(workflow)
    gaps = missing_nodes(workflow)
    if gaps:
        raise H3SwapNodesMissing(gaps)

    workflow[NODE_TARGET_IMAGE]['inputs']['image'] = target_image
    workflow[NODE_REF_IMAGE]['inputs']['image'] = ref_image
    # How much of the shot travels with the head. See `context_factor`: the crop
    # grows from the mask and clamps to the frame, so this one number gives a
    # full-body shot its chest and leaves a portrait uncropped.
    workflow[NODE_CROP]['inputs']['context_from_mask_extend_factor'] = context_factor()
    # How solid the hole H3 is asked to fill is. See `mask_opacity`: at 1.0 the
    # head is a structureless slab of colour, and a model asked to fill a slab
    # sometimes paints the slab back — the "white face" result.
    workflow[NODE_MASK_OVERLAY]['inputs']['mask_opacity'] = mask_opacity()
    # Which frame of the packet is kept. The graph shipped this at 0, i.e. the
    # selector judged sharpness and exposure and never asked whether the face
    # looks like the person — so a blank or wrong face competed on equal terms
    # with a good one. The generation lane has defaulted it to 1.0 all along;
    # this reads the SAME setting rather than inventing a second one.
    workflow[NODE_FRAME_SELECT]['inputs']['weight_reference'] = _cfg_float(
        'minimax_h3.frame_weight_reference', mh.DEFAULT_FRAME_WEIGHT_REFERENCE)
    # Frames sampled per shot. Snapped onto the node's own min/step grid — an
    # off-step value is a validation error, i.e. a whole batch of dead tiles.
    length = cfg.get('minimax_h3.length')
    if length is None:
        length = workflow[NODE_LENGTH]['inputs'].get('value', mh.LENGTH_MIN)
    workflow[NODE_LENGTH]['inputs']['value'] = mh.clamp_length(length)
    workflow[NODE_SEED]['inputs']['noise_seed'] = random.randint(0, 2 ** 64 - 1)
    prompt_override = cfg.get('minimax_h3.swap_prompt')
    if isinstance(prompt_override, str) and prompt_override.strip():
        workflow[NODE_H3]['inputs']['prompt'] = prompt_override.strip()
    ref_size = cfg.get('minimax_h3.ref_image_size')
    if ref_size:
        workflow[NODE_H3]['inputs']['ref_image_size'] = ref_size
    # UNIQUE prefix per job: SaveImage numbers from what is in the output folder
    # and the app moves each result out right after completion, so a shared
    # prefix makes ComfyUI's counter re-issue the same name and every tile of a
    # batch ends up displaying the SAME image.
    workflow[NODE_SAVE]['inputs']['filename_prefix'] = filename_prefix
    return workflow, kept


def enqueue_h3_swap(user_id, target_path, ref_path, extra_metadata=None):
    """Copy the tile's current image (the target) and the dataset's reference
    photo (the identity) into ComfyUI input, configure the fixed H3 swap
    workflow, and enqueue it. Returns the app job_id.

    Signature-identical to face_swap_helper.enqueue_face_swap, because
    dataset_generation_service picks between the two by config alone.

    Raises ValueError on a missing source image / unloadable workflow / missing
    required node, MinimaxH3ModelsMissing (or its H3SwapNodesMissing subclass)
    when an H3 asset or a custom node the graph needs is absent,
    KleinModelsMissing when the hair-removal stage is on and a Klein asset is
    not, RuntimeError when ComfyUI isn't configured."""
    if not target_path or not os.path.exists(target_path):
        raise ValueError(f'target image not found: {target_path}')
    if not ref_path or not os.path.exists(ref_path):
        raise ValueError(f'reference image not found: {ref_path}')

    # The mask is computed FIRST, on the tile as it is on disk: it is the one
    # step that can still refuse the whole job for a reason the user can act on
    # ("nothing matched that phrase"), and doing it before anything is staged or
    # queued keeps that refusal free.
    mask_path = None
    if mask_source() == 'app':
        from . import auto_mask
        mask_path = auto_mask.mask_for(target_path, mask_prompt())

    comfy_input_dir = comfy_fs.ensure_input_usable(_comfy_input_dir())
    uid = uuid.uuid4().hex[:8]
    target_stem = os.path.splitext(os.path.basename(str(target_path)))[0] or 'target'
    ref_stem = os.path.splitext(os.path.basename(str(ref_path)))[0] or 'ref'
    staged_target = comfy_fs.stage_input_image(
        target_path, f'h3swap_target_{uid}_{target_stem}.png', comfy_input_dir)
    staged_ref = comfy_fs.stage_input_image(
        ref_path, f'h3swap_ref_{uid}_{ref_stem}.png', comfy_input_dir)
    staged_inputs = [os.path.basename(staged_target), os.path.basename(staged_ref)]
    staged_mask = None
    if mask_path:
        staged_mask = os.path.basename(comfy_fs.stage_input_image(
            mask_path, f'h3swap_mask_{uid}_{target_stem}.png', comfy_input_dir))
        staged_inputs.append(staged_mask)

    workflow, kept = build_swap_workflow(
        staged_inputs[0], staged_inputs[1],
        filename_prefix=f'{user_id}_H3Swap_{uid}', mask_image=staged_mask)
    if kept:
        logger.info('h3 swap: optional stages on — %s',
                    ', '.join(STAGE_LABELS[k] for k in kept))

    job_id = str(uuid.uuid4())
    meta = {'model_name': 'minimax_h3_face_swap_dataset'}
    if extra_metadata:
        meta.update(extra_metadata)
    meta['staged_inputs'] = staged_inputs
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow,
                          prompt='Head swap (MiniMax H3, reference identity)',
                          job_id=job_id, metadata=meta)
    return job_id
