"""The NEW MiniMax H3 head swap — the maintainer's 2026-08-14 redesign of the
graph, shipped as 'minimax h3 swap new.json' and selected by
`face_swap.engine = 'minimax_h3'`.

The one it replaces is still here, unchanged, under `minimax_h3_old` and
`minimax_h3_swap_helper` — two engines, two graphs, two sets of node ids. They
share their graph ALGEBRA (repoint / prune / drop_passthrough / the node-gap
exception) by importing it from that module rather than by copy, so a fix to how
a stage is subtracted reaches both.

WHAT CHANGED, AND WHY IT IS A DIFFERENT ENGINE RATHER THAN A NEW VERSION
-----------------------------------------------------------------------
The old graph worked on a CROP: it masked the head, cut a box around it, sent
only that box through H3, and composited the result back — every pixel outside
the mask survived the swap untouched. This one has no crop and no stitch. The
whole photo goes through H3 and the whole photo comes back, so the body,
clothing and background are RE-RENDERED, not preserved. That is a deliberate
trade (H3 sees the entire shot, and there is no seam to hide because there is no
join), but it is also why `face_swap.h3_context_factor` and
`h3_blend_pixels` — both crop/stitch parameters — do nothing here, and why the
old engine is kept rather than retired.

The pipeline:

    1003 target -> Resize(longer edge 1024)
        |-> ClothesSegment (RMBG, Hair+Face)  -> head mask -> GrowMask(20)
        |-> ImageScaleToTotalPixels(1 MP) -> Klein 9B + LanPaint KSampler
        |     "remove the whole head, keep the neck, render what was removed
        |      as a face depth map" -> VAEDecode
        |     -> [Mask Overlay (RMBG), blue]            <- optional stage
        |-> H3ImageResolutionPreset (native detail)
        |-> [Ollama API: describe how this head sits]   <- optional stage
              -> Text Concatenate(fixed instruction + description)
    114 reference -> Resize(1024)
        -> MiniMaxH3ReferenceToVideo(ref_image_0 = identity,
           ref_image_1 = the head-removed target, prompt = the text above)
        -> SamplerCustomAdvanced -> VAEDecode -> H3FrameSelect(1)
        -> RTXVideoSuperResolution -> SaveImage

Three consequences worth naming before anyone debugs this graph:

* KLEIN IS NO LONGER OPTIONAL. In the old graph a Klein pass was one of three
  stages you could switch on; here the Klein edit IS how the head is removed, so
  every swap loads Klein 9B + its text encoder + its VAE alongside 40 GB of H3.
  A missing Klein asset blocks the swap, and it is reported the same way the old
  engine reported it for its hair-removal stage.
* THE PROMPT IS NOT A STRING ON THE H3 NODE. It arrives through
  `Text Concatenate`, so `minimax_h3.swap_prompt` and the pose hint are written
  into the `Text Multiline` node (990) instead — writing to the H3 node's own
  `prompt` input would be overwritten by the link at execution time, silently.
* THE HYBRID LOADER WANTS TWO MODELS. `MiniMaxH3HybridLoader` lays Ref2VA over
  Fl2VA's last blocks, and Fl2VA is the file every other H3 path here goes out
  of its way NOT to pick (minimax_h3_helper.H3_WRONG_TASK_TOKENS). It is
  resolved through `resolve_h3_fl2va`, which is the only caller allowed to ask
  for it by name.

THE TWO OPTIONAL STAGES (`face_swap.h3_new_stages`, both OFF by default)
-----------------------------------------------------------------------
These are the two nodes the maintainer runs BYPASSED in the source graph. They
ship wired and each switch SUBTRACTS, exactly like the old engine's stages —
the only direction that cannot invent wiring nobody measured.

  mask_overlay  `Mask Overlay (RMBG)` paints the head region a flat blue over
                the head-removed image before H3 sees it. Off, H3 receives the
                Klein output as it is: the head gone and a depth map of where it
                was, which is the point of that pass. `h3_mask_opacity` applies
                to this node when it is on.
  ollama        an `OllamaAPI` call analyses the target and writes a paragraph
                about how the head sits (position, angle, occlusion, lighting),
                appended to the fixed instruction. The MODEL is the app's own
                configured vision model, never the graph's — a graph naming one
                specific Ollama tag fails on any machine that does not have it.
                The instruction text stays as the maintainer tuned it.

Switching `ollama` ON disables the pose hint (`face_swap.h3_pose_hint`) for this
job. Both describe the same thing — how this particular head is turned — but
the hint is INFERRED from the catalog prompt that made the tile while Ollama is
LOOKING at the actual picture. Sending both means sending two descriptions that
can disagree, and the model then splits the difference.
"""
from __future__ import annotations
import logging
import os
import random
import uuid

from .. import config as cfg
from . import minimax_h3_helper as mh
from . import minimax_h3_swap_helper as old
from ..utils import comfy_fs
from ..utils.comfyui import load_workflow_local
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

WORKFLOW_H3_SWAP_NEW_PATH = cfg.BACKEND_DIR / 'workflows' / 'minimax h3 swap new.json'

# Nodes this helper rewires. Same rule as the old engine: the helper writes BY
# ID, so a graph swap nobody notices does not crash — it silently produces the
# wrong picture. Every one of these is asserted present before anything is
# written.
NODE_TARGET_IMAGE = '1003'       # LoadImage — the tile being repainted
NODE_REF_IMAGE = '114'           # LoadImage — the identity to graft on
NODE_TARGET_RESIZE = '928'       # ResizeImagesByLongerEdge — what the mask must match
NODE_REF_RESIZE = '311'          # ResizeImagesByLongerEdge — the identity photo
NODE_SEGMENT = '957'             # ClothesSegment (RMBG) — the in-graph head masker
NODE_H3 = '170'                  # MiniMaxH3ReferenceToVideo
NODE_PROMPT_TEXT = '990'         # Text Multiline — THE prompt lives here
NODE_PROMPT_CONCAT = '991'       # Text Concatenate — instruction + description
NODE_OLLAMA = '988'              # OllamaAPI — optional, off by default
NODE_SEED = '131'                # RandomNoise -> 'noise_seed'
NODE_LENGTH = '139'              # PrimitiveInt -> frame packet length
NODE_FRAME_SELECT = '304'        # H3FrameSelect — which frame wins
NODE_CLIP_VISION = '305'
NODE_H3_HYBRID = '925:427'       # MiniMaxH3HybridLoader — base + overlay
NODE_H3_CLIP = '925:922'
NODE_VIDEO_VAE = '925:921'
NODE_AUDIO_VAE = '925:926'       # required by the node even for a still
NODE_MASK_OVERLAY = '983:1002'   # AILab_MaskOverlay — optional, off by default
NODE_KLEIN_DECODE = '983:973'    # VAEDecode — the head-removed target
NODE_KLEIN_UNET = '983:962'      # OTUNetLoaderW8A8 (may degrade to UNETLoader)
NODE_KLEIN_CLIP = '983:963'
NODE_KLEIN_VAE = '983:966'
NODE_KLEIN_SAMPLER = '983:969'   # LanPaint_KSampler -> 'seed'
NODE_SAVE = '165'

# Optional accelerators: (node id, the input carrying its pass-through).
NODE_SPEED_SAGE = '925:315'      # PathchSageAttentionKJ                  -> 'model'
NODE_SPEED_MEMEFF = '925:428'    # MiniMaxH3MemEffSageAttentionPatch      -> 'model'
NODE_UPSCALE_OUT = '309'         # RTXVideoSuperResolution                -> 'images'

_SPEED_NODES = ((NODE_SPEED_SAGE, 'model'), (NODE_SPEED_MEMEFF, 'model'))
_UPSCALE_NODES = ((NODE_UPSCALE_OUT, 'images'),)

_REQUIRED_NODES = (NODE_TARGET_IMAGE, NODE_REF_IMAGE, NODE_TARGET_RESIZE,
                   NODE_SEGMENT, NODE_H3, NODE_PROMPT_TEXT, NODE_PROMPT_CONCAT,
                   NODE_OLLAMA, NODE_SEED, NODE_LENGTH, NODE_FRAME_SELECT,
                   NODE_CLIP_VISION, NODE_H3_HYBRID, NODE_H3_CLIP,
                   NODE_VIDEO_VAE, NODE_AUDIO_VAE, NODE_MASK_OVERLAY,
                   NODE_KLEIN_DECODE, NODE_KLEIN_UNET, NODE_KLEIN_CLIP,
                   NODE_KLEIN_VAE, NODE_KLEIN_SAMPLER, NODE_SAVE)

# stage -> (tail node, what its consumers read when the stage is off).
# Both fallbacks are the node's OWN pass-through input, i.e. exactly what
# ComfyUI does to a bypassed node — read off the maintainer's export, not
# reconstructed.
STAGES = {
    'mask_overlay': (NODE_MASK_OVERLAY, [NODE_KLEIN_DECODE, 0]),
    'ollama': (NODE_PROMPT_CONCAT, [NODE_PROMPT_TEXT, 0]),
}
STAGE_LABELS = {
    'mask_overlay': 'Blue mask overlay',
    'ollama': 'Ollama head analysis',
}

# Packs for the classes THIS graph adds on top of the ones the old swap already
# names. Same {pack, url, search} shape, so the preflight banner renders them
# with no second code path.
H3_SWAP_NEW_NODE_PACKS = dict(old.H3_SWAP_NODE_PACKS)
H3_SWAP_NEW_NODE_PACKS.update({
    'ClothesSegment': {
        'pack': 'ComfyUI-RMBG',
        'url': 'https://github.com/1038lab/ComfyUI-RMBG',
        'search': 'RMBG'},
    'LanPaint_KSampler': {
        # No URL on purpose: the pack NAME is what the ComfyUI Manager search
        # box takes, and a link guessed from a node name is worse than none.
        'pack': 'LanPaint', 'url': None, 'search': 'LanPaint'},
    'Text Multiline': {
        'pack': 'was-node-suite-comfyui',
        'url': 'https://github.com/WASasquatch/was-node-suite-comfyui',
        'search': 'WAS Node Suite'},
    'Text Concatenate': {
        'pack': 'was-node-suite-comfyui',
        'url': 'https://github.com/WASasquatch/was-node-suite-comfyui',
        'search': 'WAS Node Suite'},
    'OllamaAPI': {
        'pack': 'unknown pack — only the optional Ollama stage needs it; '
                'switch that stage off to run without it',
        'url': None, 'search': 'OllamaAPI'},
    'MiniMaxH3HybridLoader': dict(mh.H3_FRAME_SELECT_PACK),
    'MiniMaxH3MemoryEfficientSageAttentionPatch': {
        'pack': 'unknown pack — optional speed node; switch off '
                'minimax_h3.use_speed_nodes to run without it',
        'url': None, 'search': 'MiniMaxH3MemoryEfficientSageAttentionPatch'},
})


def node_hints(nodes):
    """[{class_type, pack, url, search}] for each missing node."""
    out = []
    for ct in (nodes or []):
        meta = H3_SWAP_NEW_NODE_PACKS.get(ct, {'pack': None, 'url': None, 'search': ct})
        out.append({'class_type': ct, **meta})
    return out


class H3SwapNewNodesMissing(old.H3SwapNodesMissing):
    """The target ComfyUI lacks a custom node THIS graph needs. Subclasses the
    old engine's exception so every caller and route branch that already handles
    it keeps working — only the pack attribution differs."""

    def __init__(self, nodes):
        super().__init__(nodes)
        self.node_packs = node_hints(nodes)


def enabled_stages():
    """{stage: bool} from `face_swap.h3_new_stages`, defaulting to OFF."""
    raw = cfg.get('face_swap.h3_new_stages')
    raw = raw if isinstance(raw, dict) else {}
    return {name: bool(raw.get(name)) for name in STAGES}


def apply_stages(workflow, stages):
    """Subtract every stage that is off by repointing its consumers at the
    stage's own bypass fallback. Returns the stage names left in place."""
    kept = []
    for name, (tail, fallback) in STAGES.items():
        if stages.get(name):
            kept.append(name)
            continue
        if tail in workflow and not old.repoint(workflow, tail, fallback):
            logger.debug('h3 swap (new): stage %s has no consumers to repoint', name)
    return kept


def attach_app_mask(workflow, mask_image):
    """Feed an app-produced mask PNG into the graph in place of ClothesSegment.

    Same discipline as the old engine: the mask arrives at the TILE's own
    resolution while the graph works on a copy resized to `longer_edge`, so it
    goes through the SAME `ResizeImagesByLongerEdge` value rather than being
    resized app-side — that node truncates, and reproducing its arithmetic
    elsewhere is how an off-by-one lands as a ComfyUI error instead of here.

    ClothesSegment is left unwired and disappears in `prune_to_outputs`."""
    longer_edge = ((workflow.get(NODE_TARGET_RESIZE) or {}).get('inputs', {})
                   .get('longer_edge', 1024))
    workflow[old.NODE_APP_MASK_LOAD] = {
        'class_type': 'LoadImage', 'inputs': {'image': mask_image},
        '_meta': {'title': 'Mask (from the app)'}}
    workflow[old.NODE_APP_MASK_RESIZE] = {
        'class_type': 'ResizeImagesByLongerEdge',
        'inputs': {'longer_edge': longer_edge,
                   'images': [old.NODE_APP_MASK_LOAD, 0]},
        '_meta': {'title': 'Mask, to the same geometry as the target'}}
    workflow[old.NODE_APP_MASK_TO_MASK] = {
        'class_type': 'ImageToMask',
        'inputs': {'image': [old.NODE_APP_MASK_RESIZE, 0], 'channel': 'red'},
        '_meta': {'title': 'Mask'}}
    # Every reader of the segmenter's mask output now reads the app's mask.
    old.repoint(workflow, NODE_SEGMENT, [old.NODE_APP_MASK_TO_MASK, 0])
    return old.NODE_APP_MASK_TO_MASK


def missing_nodes(workflow):
    """[class_type] this graph needs that the target ComfyUI does not expose.
    Fail-OPEN — see the old engine's version, same contract."""
    return old.missing_nodes(workflow)


def build_swap_workflow(target_image, ref_image, *, filename_prefix, stages=None,
                        mask_image=None, pose_hint=None):
    """Load the shipped graph, subtract the optional stages, resolve every loader
    against what is actually installed, and return (workflow, stages_kept).

    Split out of `enqueue_h3_swap_new` so a test can assert the exact wiring
    without a ComfyUI or a queue."""
    raw = load_workflow_local(str(WORKFLOW_H3_SWAP_NEW_PATH))
    if not raw:
        raise ValueError('failed to load the new MiniMax H3 swap workflow')
    workflow = dict(raw)
    for node in _REQUIRED_NODES:
        if node not in workflow:
            raise ValueError(f'workflow node {node} missing — '
                             'minimax h3 swap new.json has changed')

    stages = enabled_stages() if stages is None else dict(stages)
    kept = apply_stages(workflow, stages)
    # An app-produced mask replaces ClothesSegment BEFORE the node probe, so an
    # install is never told to install a pack for a node this job dropped.
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
                old.drop_passthrough(workflow, node_id, input_key)

    # H3's assets, through the generation lane's own resolvers — an install that
    # can generate with H3 can swap with it. PLUS Fl2VA, which only this graph
    # loads: the hybrid loader takes it as the base and lays Ref2VA over it.
    missing = mh.h3_missing_assets()
    if missing:
        raise mh.MinimaxH3ModelsMissing(missing)
    fl2va = mh.resolve_h3_fl2va()
    if not fl2va:
        raise mh.MinimaxH3ModelsMissing(['h3_fl2va'])
    workflow[NODE_H3_HYBRID]['inputs']['base_model'] = fl2va
    workflow[NODE_H3_HYBRID]['inputs']['overlay_model'] = mh.resolve_h3_unet(
        cfg.get('minimax_h3.base_model'))
    workflow[NODE_H3_CLIP]['inputs']['clip_name'] = mh.resolve_h3_text_encoder()
    workflow[NODE_VIDEO_VAE]['inputs']['vae_name'] = mh.resolve_h3_video_vae()
    workflow[NODE_AUDIO_VAE]['inputs']['vae_name'] = mh.resolve_h3_audio_vae()
    workflow[NODE_CLIP_VISION]['inputs']['clip_name'] = mh.resolve_h3_clip_vision()

    # Klein removes the head, so its assets are REQUIRED here — resolved the way
    # every other Klein call site resolves them, never from the graph's own
    # filenames (those name the maintainer's disk).
    from .klein_edit_helper import (KLEIN_REQUIRED, KleinModelsMissing,
                                    klein_missing_assets,
                                    resolve_klein_text_encoder,
                                    resolve_klein_vae)
    from .face_swap_helper import resolve_swap_unet
    klein_gaps = klein_missing_assets()
    if any(a in klein_gaps for a in KLEIN_REQUIRED):
        raise KleinModelsMissing(klein_gaps)
    klein_unet, _int8 = resolve_swap_unet()
    old._degrade_w8a8(workflow, NODE_KLEIN_UNET, klein_unet,
                      'Klein (no INT8 build found)')
    workflow[NODE_KLEIN_CLIP]['inputs']['clip_name'] = resolve_klein_text_encoder()
    workflow[NODE_KLEIN_VAE]['inputs']['vae_name'] = resolve_klein_vae()
    workflow[NODE_KLEIN_SAMPLER]['inputs']['seed'] = random.randint(0, 2 ** 64 - 1)

    # The Ollama stage runs on the model the APP is configured with, not the tag
    # the graph carries: a hard-coded Ollama tag is a file nobody else has.
    if 'ollama' in kept and NODE_OLLAMA in workflow:
        from .vision_ollama import get_vision_model
        workflow[NODE_OLLAMA]['inputs']['ollama_model'] = get_vision_model()
        workflow[NODE_OLLAMA]['inputs']['seed'] = random.randint(0, 2 ** 64 - 1)

    # Everything a switched-off stage or accelerator left behind goes now, so
    # the node probe below asks about the job that will actually run.
    workflow = old.prune_to_outputs(workflow)
    gaps = missing_nodes(workflow)
    if gaps:
        raise H3SwapNewNodesMissing(gaps)

    workflow[NODE_TARGET_IMAGE]['inputs']['image'] = target_image
    workflow[NODE_REF_IMAGE]['inputs']['image'] = ref_image
    # Only meaningful with the overlay stage on — with it off the node is gone.
    if NODE_MASK_OVERLAY in workflow:
        workflow[NODE_MASK_OVERLAY]['inputs']['mask_opacity'] = old.mask_opacity()
    workflow[NODE_FRAME_SELECT]['inputs']['weight_reference'] = old._cfg_float(
        'minimax_h3.frame_weight_reference', mh.DEFAULT_FRAME_WEIGHT_REFERENCE)
    length = cfg.get('minimax_h3.length')
    if length is None:
        length = workflow[NODE_LENGTH]['inputs'].get('value', mh.LENGTH_MIN)
    workflow[NODE_LENGTH]['inputs']['value'] = mh.clamp_length(length)
    workflow[NODE_SEED]['inputs']['noise_seed'] = random.randint(0, 2 ** 64 - 1)

    # THE PROMPT LIVES IN THE TEXT NODE. On this graph the H3 node's `prompt` is
    # a LINK, so anything written to it is discarded at execution time.
    prompt_override = cfg.get('minimax_h3.swap_prompt')
    if isinstance(prompt_override, str) and prompt_override.strip():
        workflow[NODE_PROMPT_TEXT]['inputs']['text'] = prompt_override.strip()
    # The pose hint is APPENDED, never substituted — and never sent at all when
    # Ollama is analysing the same picture (see the module docstring).
    if pose_hint and 'ollama' not in kept:
        workflow[NODE_PROMPT_TEXT]['inputs']['text'] = (
            workflow[NODE_PROMPT_TEXT]['inputs']['text'].rstrip()
            + ' ' + pose_hint.strip())
    ref_size = cfg.get('minimax_h3.ref_image_size')
    if ref_size:
        workflow[NODE_H3]['inputs']['ref_image_size'] = ref_size
    # UNIQUE prefix per job — a shared one makes ComfyUI's counter re-issue the
    # same name and every tile of a batch ends up displaying the SAME image.
    workflow[NODE_SAVE]['inputs']['filename_prefix'] = filename_prefix
    return workflow, kept


def enqueue_h3_swap_new(user_id, target_path, ref_path, extra_metadata=None):
    """Stage both images into ComfyUI input, configure the new H3 swap workflow,
    and enqueue it. Returns the app job_id.

    Signature-identical to face_swap_helper.enqueue_face_swap and to the old
    engine's enqueue_h3_swap, because dataset_generation_service picks between
    the three by config alone."""
    if not target_path or not os.path.exists(target_path):
        raise ValueError(f'target image not found: {target_path}')
    if not ref_path or not os.path.exists(ref_path):
        raise ValueError(f'reference image not found: {ref_path}')

    # The mask is computed FIRST, on the tile as it is on disk: it is the one
    # step that can still refuse the whole job for a reason the user can act on.
    mask_path = None
    if old.mask_source() == 'app':
        from . import auto_mask
        mask_path = auto_mask.mask_for(target_path, old.mask_prompt())

    comfy_input_dir = comfy_fs.ensure_input_usable(old._comfy_input_dir())
    uid = uuid.uuid4().hex[:8]
    target_stem = os.path.splitext(os.path.basename(str(target_path)))[0] or 'target'
    ref_stem = os.path.splitext(os.path.basename(str(ref_path)))[0] or 'ref'
    staged_target = comfy_fs.stage_input_image(
        target_path, f'h3swapnew_target_{uid}_{target_stem}.png', comfy_input_dir)
    staged_ref = comfy_fs.stage_input_image(
        ref_path, f'h3swapnew_ref_{uid}_{ref_stem}.png', comfy_input_dir)
    staged_inputs = [os.path.basename(staged_target), os.path.basename(staged_ref)]
    staged_mask = None
    if mask_path:
        staged_mask = os.path.basename(comfy_fs.stage_input_image(
            mask_path, f'h3swapnew_mask_{uid}_{target_stem}.png', comfy_input_dir))
        staged_inputs.append(staged_mask)

    hint = None
    if cfg.get('face_swap.h3_pose_hint') is not False:
        from .face_swap_pose import pose_hint as _pose_hint
        meta = extra_metadata or {}
        hint = _pose_hint(meta.get('variation_prompt'), meta.get('framing'),
                          meta.get('variation_label'))
    workflow, kept = build_swap_workflow(
        staged_inputs[0], staged_inputs[1],
        filename_prefix=f'{user_id}_H3SwapNew_{uid}', mask_image=staged_mask,
        pose_hint=hint)
    if hint and 'ollama' not in kept:
        logger.info('h3 swap (new): pose hint — %s', hint)
    if kept:
        logger.info('h3 swap (new): optional stages on — %s',
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
