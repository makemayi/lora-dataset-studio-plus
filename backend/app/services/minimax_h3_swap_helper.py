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
* The H3 PROMPT stays as tuned in the graph unless `minimax_h3.swap_prompt`
  overrides it. It names a subject because that is what it was measured on; a
  different subject degrades the result rather than failing, and inventing a
  substitution would change results nobody has looked at. Same rule as the
  Klein swap's SAM3 prompts.
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
NODE_SAVE = '412'

_REQUIRED_NODES = (NODE_TARGET_IMAGE, NODE_REF_IMAGE, NODE_H3_UNET, NODE_H3_CLIP,
                   NODE_VIDEO_VAE, NODE_AUDIO_VAE, NODE_CLIP_VISION, NODE_H3,
                   NODE_LENGTH, NODE_SEED, NODE_SAVE)

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
NODE_LAMA_TAIL = '426:411'
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


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def build_swap_workflow(target_image, ref_image, *, filename_prefix, stages=None):
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

    comfy_input_dir = comfy_fs.ensure_input_usable(_comfy_input_dir())
    uid = uuid.uuid4().hex[:8]
    target_stem = os.path.splitext(os.path.basename(str(target_path)))[0] or 'target'
    ref_stem = os.path.splitext(os.path.basename(str(ref_path)))[0] or 'ref'
    staged_target = comfy_fs.stage_input_image(
        target_path, f'h3swap_target_{uid}_{target_stem}.png', comfy_input_dir)
    staged_ref = comfy_fs.stage_input_image(
        ref_path, f'h3swap_ref_{uid}_{ref_stem}.png', comfy_input_dir)
    staged_inputs = [os.path.basename(staged_target), os.path.basename(staged_ref)]

    workflow, kept = build_swap_workflow(
        staged_inputs[0], staged_inputs[1],
        filename_prefix=f'{user_id}_H3Swap_{uid}')
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
