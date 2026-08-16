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

Re-pointed at the maintainer's 2026-08-16 export. Same shape, four changes that
matter to anything reading this file: the H3 loader group moved from the `925:*`
subgraph to `1009:*`, the head is masked by SAM3 rather than RMBG's
ClothesSegment, `H3FrameSelect` became a plain `ImageFromBatch`, and the graph
now ships 8 steps rather than 25 because the maintainer runs a turbo distill.

    1010 target -> Resize(longer edge 1024)
        |-> [SAM3_Detect: hair+face] -> head mask -> GrowMask(26)
        |     ALWAYS substituted: the app runs the same model, so this branch is
        |     repointed at the app's mask and prunes away. See `attach_app_mask`.
        |-> ImageScaleToTotalPixels(1 MP) -> Klein 9B + LanPaint KSampler
        |     "strip the hair, replace the head with a grey SKULL stand-in a
        |      fifth smaller than the haired head" -> VAEDecode
        |-> H3ImageResolutionPreset (native detail)
        |-> [Ollama, in the APP: describe how this head sits]  <- optional stage
              -> Text Concatenate(fixed instruction + description)
    114 reference -> Resize(1024)
        -> MiniMaxH3ReferenceToVideo(ref_image_0 = identity,
           ref_image_1 = the head-removed target, prompt = the text above)
        -> SamplerCustomAdvanced -> VAEDecode -> ImageFromBatch(0, 1)
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

THE OPTIONAL STAGE (`face_swap.h3_new_stages`, OFF by default)
--------------------------------------------------------------
It ships wired and the switch SUBTRACTS, exactly like the old engine's stages —
the only direction that cannot invent wiring nobody measured.

`mask_overlay` was the second one and is GONE, not disabled: the 2026-08-16
graph has no AILab_MaskOverlay node, so the blue slab it painted cannot be
produced at all. A Settings switch that silently does nothing is worse than one
that is absent, so the key left with the node. `face_swap.h3_mask_opacity` was
its only parameter here and now applies to the old engine alone.

  ollama        a vision model analyses the target and writes a paragraph about
                how the head sits (position, angle, occlusion, lighting), which
                is appended to the fixed instruction.

                THIS ONE NO LONGER RUNS IN THE GRAPH. The `OllamaAPI` node is
                subtracted like a switched-off stage even when the stage is ON;
                the app makes the call itself and writes the answer into the
                text node. Four things were wrong with letting the node do it,
                and all four are structural rather than bad luck:

                  * it talks to `127.0.0.1:11434` and takes no URL input, so
                    `ollama.url` could not reach it and a remote Ollama was
                    simply unusable;
                  * a stopped Ollama failed INSIDE ComfyUI, after the 40 GB H3
                    stack had loaded, consuming the tile for nothing;
                  * it bypassed this app's Ollama GPU fence entirely, so the
                    vision model landed on the same card as H3 — on a 24 GB card
                    that is the difference between a render and an OOM;
                  * its prompt asked the model to "重点参考深度图" while the node
                    was wired to ONE image input carrying the plain target, so
                    that whole section was answered from nothing.

                Running it here fixes all four: the app's URL and model apply,
                failure is refused before anything is queued, the call finishes
                and unloads BEFORE the render is queued (so the card holds one
                model at a time instead of two), and the prompt now asks only
                for what the model is actually shown.

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
NODE_TARGET_IMAGE = '1010'       # LoadImage — the tile being repainted
NODE_REF_IMAGE = '114'           # LoadImage — the identity to graft on
NODE_TARGET_RESIZE = '928'       # ResizeImagesByLongerEdge — what the mask must match
NODE_REF_RESIZE = '311'          # ResizeImagesByLongerEdge — the identity photo
# SAM3_Detect. Kept as a constant because it is the ANCHOR the app's own mask is
# repointed onto — never because the job runs it. This engine always masks in the
# app (see `attach_app_mask`), so the node, its CheckpointLoaderSimple and its
# CLIPTextEncode are left unwired and leave in `prune_to_outputs`. That is what
# spares an install the ComfyUI-SAM3 pack and the sam3.1 checkpoint entirely.
NODE_SEGMENT = '1001:75'
NODE_H3 = '170'                  # MiniMaxH3ReferenceToVideo
NODE_PROMPT_TEXT = '990'         # Text Multiline — THE prompt lives here
NODE_PROMPT_CONCAT = '991'       # Text Concatenate — instruction + description
NODE_SEED = '131'                # RandomNoise -> 'noise_seed'
NODE_FRAME_SELECT = '999'        # ImageFromBatch — which frame wins
NODE_H3_HYBRID = '1009:1005'     # MiniMaxH3HybridLoader — base + overlay
NODE_H3_CLIP = '1009:1002'
NODE_VIDEO_VAE = '1009:1008'
NODE_AUDIO_VAE = '1009:1007'     # required by the node even for a still
NODE_KLEIN_POSITIVE = '983:964'  # CLIPTextEncode — what the Klein pass is told to do
NODE_KLEIN_NEGATIVE = '983:965'  # CLIPTextEncode — shipped empty by the maintainer
# The far end of H3's model chain. Whatever feeds THIS node's `model` is the
# tail extra LoRAs chain onto — found, never hardcoded, so the speed patches
# above it can be dropped without moving the insertion point.
NODE_H3_MODEL_SINK = '128'       # BasicGuider — first consumer of the H3 model
NODE_H3_STEPS = '126'            # BasicScheduler — the graph ships 8 steps
NODE_KLEIN_DECODE = '983:973'    # VAEDecode — the head-removed target
NODE_KLEIN_UNET = '983:962'      # OTUNetLoaderW8A8 (may degrade to UNETLoader)
NODE_KLEIN_CLIP = '983:963'
NODE_KLEIN_VAE = '983:966'
NODE_KLEIN_SAMPLER = '983:969'   # LanPaint_KSampler -> 'seed'
NODE_SAVE = '165'

# Optional accelerators: (node id, the input carrying its pass-through).
NODE_SPEED_SAGE = '1009:1004'    # PathchSageAttentionKJ                  -> 'model'
NODE_SPEED_MEMEFF = '1009:1003'  # MiniMaxH3MemEffSageAttentionPatch      -> 'model'
# New in the 2026-08-16 graph, and grouped with the speed patches for the same
# reason: it is a backend switch on a model that runs without it. It sits BELOW
# the two patches in the chain, so dropping it does not move where they attach.
NODE_ATTENTION_BACKEND = '1009:937'  # ModelAttentionBackend              -> 'model'
NODE_UPSCALE_OUT = '309'         # RTXVideoSuperResolution                -> 'images'

# The two accelerator LoRAs the maintainer's export carries baked in. They are
# named for files on that disk (`minimax H3\...`), which is the exact class of
# breakage this helper exists to prevent, so they are never sent as shipped —
# `minimax_h3.swap_loras` fills them, or they leave the graph entirely.
NODE_GRAPH_LORAS = ('1009:938', '1009:939')

_SPEED_NODES = ((NODE_SPEED_SAGE, 'model'), (NODE_SPEED_MEMEFF, 'model'),
                (NODE_ATTENTION_BACKEND, 'model'))
_UPSCALE_NODES = ((NODE_UPSCALE_OUT, 'images'),)

_REQUIRED_NODES = (NODE_TARGET_IMAGE, NODE_REF_IMAGE, NODE_TARGET_RESIZE,
                   NODE_SEGMENT,
                   NODE_H3, NODE_PROMPT_TEXT, NODE_PROMPT_CONCAT,
                   NODE_SEED, NODE_FRAME_SELECT,
                   NODE_H3_HYBRID, NODE_H3_CLIP,
                   NODE_VIDEO_VAE, NODE_AUDIO_VAE,
                   NODE_KLEIN_DECODE, NODE_KLEIN_UNET, NODE_KLEIN_CLIP,
                   NODE_KLEIN_VAE, NODE_KLEIN_SAMPLER, NODE_SAVE,
                   NODE_KLEIN_POSITIVE, NODE_KLEIN_NEGATIVE,
                   NODE_H3_MODEL_SINK, NODE_H3_STEPS)

# stage -> (tail node, what its consumers read when the stage is off).
# The fallback is the node's OWN pass-through input, i.e. exactly what ComfyUI
# does to a bypassed node — read off the maintainer's export, not reconstructed.
#
# `mask_overlay` left with the 2026-08-16 graph: that redesign has no
# AILab_MaskOverlay node, so the blue slab it painted cannot be produced at all.
# The stage is gone rather than kept-but-inert, because a Settings switch that
# does nothing is worse than one that is absent.
STAGES = {
    'ollama': (NODE_PROMPT_CONCAT, [NODE_PROMPT_TEXT, 0]),
}
STAGE_LABELS = {
    'ollama': 'Ollama head analysis',
}

# Stages the APP performs. Their node is subtracted from the graph whether the
# stage is on or off — being "kept" here means the app runs it, not that the
# graph does. Everything else about a kept stage is unchanged: it still
# suppresses the pose hint, still reports itself in the job log.
APP_SIDE_STAGES = frozenset({'ollama'})

# The sentence that introduces the head analysis. It used to be the last line of
# the shipped instruction (990), where it was sent on EVERY swap — including the
# default one, with the stage off and nothing following it. "严格按照下面的说明
# 进行" pointing at an empty tail is not a harmless leftover: it tells the model
# to obey instructions it cannot see. It now travels WITH the analysis, so it
# exists exactly when the thing it introduces does.
HEAD_ANALYSIS_LEAD_IN = '下面是图片头部的相关信息，请严格按照下面的说明进行。'

# The instruction the head analysis runs on. It came off the maintainer's
# `OllamaAPI` node and is kept almost verbatim — it is tuned, and the shape of
# its answer is what the swap prompt was written to receive. TWO deliberate
# edits, both because the node's own wiring never matched what the text claimed:
#   * "请同时参考提供的RGB图片和深度图" -> the node had ONE image input, fed the
#     plain target. No depth map was ever sent, so asking for depth-derived
#     findings invited the model to invent them.
#   * item 6 kept the two questions that survive without a depth map (occlusion,
#     volume/perspective) and dropped the two that do not (front-to-back
#     position, the neck/shoulder depth join).
# It lives here rather than in the JSON because this is where it now runs; a
# prompt stored on a node that is always subtracted would read as live wiring.
HEAD_ANALYSIS_PROMPT = '''你是一个专业的图像头部分析专家，只负责精准描述图片中人物的头部信息，用于后续换头操作。

请仔细观察提供的图片，严格按照以下要求输出，不要描述身体、服装、背景、发型、五官细节或任何其他内容：

输出必须以「图片中头部位于」作为开头。

1. 头部位置：在画面中的大致位置（左上/中上/右上/左中/正中/右中/左下/中下/右下），以及占画面的大致比例。
2. 头部朝向：正面、左侧脸（角度）、右侧脸（角度）、微侧、仰视、俯视、低头等，尽量给出具体角度。
3. 头部姿态与动作：是否歪头、转头、抬头、低头、点头等，描述自然动作。
4. 表情：详细描述眉毛、眼睛、嘴部状态和整体情绪（中性、微笑、严肃、惊讶、愤怒等）。
5. 光照与阴影：头部受光方向、脸部明暗分布，是否有硬阴影或柔光。
6. 空间关系与透视：
   - 头部是否存在明显遮挡
   - 头部的大致体积感与透视关系
7. 自身部位相对位置：描述头部与自身其他部位（如肩膀、脖子、躯干）的相对位置关系。
8. 与画面最醒目部位的相对位置：描述头部与图片中最醒目/最突出部位之间的相对位置关系。
9. 脸部遮挡情况：如果脸部存在遮挡，必须明确说明被遮挡的具体部位（如眼睛、鼻子、嘴巴、半边脸等），以及被什么物体遮挡（如手、头发、口罩、物体、另一人等）。
10. 其他影响换头的关键细节：眼镜、帽子、耳环、面部遮挡物、等。

输出要求：
- 必须以「图片中头部位于」开头
- 只输出头部相关信息
- 语言简洁精准，适合直接用于换头提示词参考
- 如果图片中有多个人，分别编号描述
- 不确定的地方标注“不确定”

请参考下面的范例：
图片中头部位于中上偏右区域，占画面比例约8%。

* **朝向与姿态**：右侧脸微侧（约45度），呈趴卧低头、向后下方窥视的姿态。
* **表情**：眼神向下注视，带有中性、慵懒情绪，嘴部及下半脸不可见。
* **光照**：整体为柔和散射光，面部受光均匀，无强硬阴影。
* **空间与相对位置**：处于画面最远端，与前景呈极强透视对比；位于自身肩膀后方，处于画面最醒目部位（臀部）的右上方远端。
* **脸部遮挡**：脸部下半部分（鼻子下方、嘴巴、下巴）被自身身体及臀部线条完全遮挡，仅露出额头、左眼及部分左脸颊。
* **关键细节**：左耳戴有一颗小耳环，换头时需严格处理下脸部被身体遮挡的边缘。'''

# Packs for the classes THIS graph adds on top of the ones the old swap already
# names. Same {pack, url, search} shape, so the preflight banner renders them
# with no second code path.
H3_SWAP_NEW_NODE_PACKS = dict(old.H3_SWAP_NODE_PACKS)
H3_SWAP_NEW_NODE_PACKS.update({
    # No SAM3_Detect entry, for the same reason there is no OllamaAPI one: the
    # 2026-08-16 graph masks with SAM3, but this engine always substitutes the
    # app's own mask, so that node never survives to reach the probe. Naming a
    # pack for it would send users to install ComfyUI-SAM3 and a 3 GB checkpoint
    # for a node no job runs.
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
    # No OllamaAPI entry: the head analysis runs in the app now, so that node is
    # subtracted from every job and can never reach the node probe. Installing a
    # pack for it would buy nothing — which is the whole point of moving the call.
    'MiniMaxH3HybridLoader': dict(mh.H3_FRAME_SELECT_PACK),
    'MiniMaxH3MemoryEfficientSageAttentionPatch': {
        'pack': 'unknown pack — optional speed node; switch off '
                'minimax_h3.use_speed_nodes to run without it',
        'url': None, 'search': 'MiniMaxH3MemoryEfficientSageAttentionPatch'},
    'ModelAttentionBackend': {
        'pack': 'unknown pack — optional speed node; switch off '
                'minimax_h3.use_speed_nodes to run without it',
        'url': None, 'search': 'ModelAttentionBackend'},
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


def swap_ollama_model():
    """The Ollama tag the head-analysis stage runs on.

    `face_swap.h3_new_ollama_model` when the user picked one in Settings, else
    the app's captioning vision model — which is what this stage ran on before
    the key existed, so an untouched install sees no change. Never the tag the
    graph was exported with: that names the maintainer's own pulled models."""
    from .vision_ollama import get_vision_model
    chosen = cfg.get('face_swap.h3_new_ollama_model')
    chosen = chosen.strip() if isinstance(chosen, str) else ''
    return chosen or get_vision_model()


MAX_H3_SWAP_LORAS = mh.MAX_H3_LORAS


def configured_h3_swap_loras():
    """Sanitized `minimax_h3.swap_loras`: ordered [{file, strength}], blank rows
    dropped, strengths clamped to [0, 1.5] (junk -> 1.0), capped.

    Its reason for existing is speed: H3 samples a packet of frames through a
    40 GB stack, and the accelerator LoRAs that make that bearable are files
    nobody can ship — they are re-quantisations and community distills that
    differ per install. So no graph here carries a LoRA of its own (the
    2026-08-16 export arrived with two, and this helper drops both) and this is
    the only place one can come from.

    The rules live in `mh.sanitize_lora_rows` since the generation lane grew the
    same list — one sanitizer, so a value one lane accepts cannot be a value the
    other silently rejects."""
    return mh.sanitize_lora_rows(cfg.get('minimax_h3.swap_loras'))


def analyse_head(target_path):
    """Describe how the head sits in `target_path`, for the swap instruction.

    Raises ValueError — refusing the whole swap — rather than returning empty on
    failure. An empty description would silently produce a DIFFERENT render from
    the one the user switched this stage on for, and the tile is overwritten in
    place, so "it ran but without the analysis" is not a recoverable outcome.

    Runs on the GPU, like every other vision call here. It does NOT overlap with
    the render: this happens before the job is queued and `keep_alive=0` drops
    the model as soon as it answers, so the card holds one model at a time. That
    ordering is the whole benefit of moving the call out of the graph — inside
    it, the node fired while H3 was resident. Pinning this to the CPU was tried
    and reverted: a 9B vision model on 8 cores is a minute or more per tile,
    which is not a trade worth making for a card the call no longer shares."""
    from .vision_ollama import describe_image_ollama
    model = swap_ollama_model()
    try:
        with open(target_path, 'rb') as fh:
            image_bytes = fh.read()
    except OSError as e:
        raise ValueError(f'could not read the tile for the head analysis: {e}') from e
    try:
        text = describe_image_ollama(
            image_bytes, HEAD_ANALYSIS_PROMPT,
            model=model,
            # Enough for the ten numbered findings; a thinking trace does not
            # fit, which is deliberate — see the empty-answer message below.
            num_predict=1200,
            think=False,           # asked, not guaranteed — some checkpoints ignore it
            keep_alive=0,          # off the card before the render is queued
            auto_start_local=True,  # start a stopped LOCAL server, then raise if it fails
            # A CPU-bound vision pass on a ~1 MP image is minutes, not seconds.
            timeout=(10, 600))
    except RuntimeError as e:
        # describe_image_ollama already words the cause (unreachable, refused,
        # model missing). Name the STAGE in front of it: the same sentence
        # reaches the user from captioning too, and "which of my features just
        # failed" is not answerable from Ollama's half alone.
        raise ValueError(
            f'The H3 swap\'s head-analysis stage could not run: {e} '
            'Turn the stage off in Settings ▸ Engines ▸ Face / head swap engine '
            'to swap without it.') from e
    text = (text or '').strip()
    if not text:
        raise ValueError(
            f'The H3 swap\'s head analysis returned nothing from "{model}". A model that '
            'emits a reasoning trace instead of an answer is the usual cause — pick a '
            'non-thinking vision model in Settings ▸ Engines ▸ Face / head swap engine, '
            'or turn the stage off.')
    return text


def apply_stages(workflow, stages):
    """Subtract every stage the GRAPH will not run, by repointing its consumers
    at the stage's own bypass fallback. Returns the stage names left in place.

    An APP_SIDE_STAGES entry is subtracted from the graph even when it is on —
    the app performs it — but it still counts as kept, because everything else
    that keys off a kept stage (the pose-hint suppression, the job log) is about
    whether the stage HAPPENS, not about where."""
    kept = []
    for name, (tail, fallback) in STAGES.items():
        on = bool(stages.get(name))
        if on:
            kept.append(name)
            if name not in APP_SIDE_STAGES:
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
                        mask_image=None, pose_hint=None, head_analysis=None):
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
    # The app's mask is REQUIRED here, not one of two sources: the graph's own
    # SAM3 branch is always substituted away, so there is nothing left to mask
    # with if it is absent. It is attached BEFORE the node probe, so an install
    # is never told to fetch a pack for a node this job just dropped.
    if not mask_image:
        raise ValueError(
            'the new MiniMax H3 swap masks in the app — no mask was supplied')
    attach_app_mask(workflow, mask_image)

    # The export ships two accelerator LoRAs loaded by name off the maintainer's
    # disk (`minimax H3\...`). Those paths are exactly what this helper exists to
    # keep out of a job, and the app already has ONE place a LoRA may come from
    # (`minimax_h3.swap_loras`, appended below). So both nodes always leave —
    # never "kept if the file happens to exist", which would make the render
    # depend on a folder layout nobody else has.
    for node_id in NODE_GRAPH_LORAS:
        old.drop_passthrough(workflow, node_id, 'model')

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
    # No vision tower on this graph: `ImageFromBatch` takes frame 0, so nothing
    # scores a packet and CLIP-ViT-H is never loaded.
    missing = mh.h3_missing_assets(need_clip_vision=False)
    if missing:
        raise mh.MinimaxH3ModelsMissing(missing)
    fl2va = mh.resolve_h3_fl2va()
    if not fl2va:
        raise mh.MinimaxH3ModelsMissing(['h3_fl2va'])
    workflow[NODE_H3_HYBRID]['inputs']['base_model'] = fl2va
    workflow[NODE_H3_HYBRID]['inputs']['overlay_model'] = mh.resolve_h3_unet(
        cfg.get('minimax_h3.base_model'))
    workflow[NODE_H3_CLIP]['inputs']['clip_name'] = mh.resolve_h3_text_encoder()
    # `minimax_h3.vae` if it is set and on disk, else the T1 image VAE, else the
    # video VAE — see `mh.resolve_h3_image_vae`. One key for both lanes: this
    # node feeds BOTH the H3 node and the final decode, so nothing here can
    # disagree with the generation lane about which VAE made the latents.
    workflow[NODE_VIDEO_VAE]['inputs']['vae_name'] = mh.resolve_h3_image_vae(
        cfg.get('minimax_h3.vae'))
    workflow[NODE_AUDIO_VAE]['inputs']['vae_name'] = mh.resolve_h3_audio_vae()
    # No CLIPVisionLoader line any more: the 2026-08-16 graph still carries the
    # node but nothing reads it, so it leaves in `prune_to_outputs`. Resolving a
    # loader that is about to be pruned would only make an install without that
    # file look broken for a node the job never runs.

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

    # What Klein is TOLD to do decides whether the swap comes back in
    # proportion: it replaces the head with a grey mannequin, and that stand-in
    # is the only thing left saying how big the head was and which way it faced.
    # Config is authoritative (the graph's own text is kept in sync for anyone
    # opening it in ComfyUI, and a test pins the two together); a blank or
    # whitespace-only override falls back to the shipped default rather than
    # sending Klein an empty instruction, which would erase nothing at all.
    for node_id, key in ((NODE_KLEIN_POSITIVE, 'face_swap.h3_head_removal_prompt'),
                         (NODE_KLEIN_NEGATIVE, 'face_swap.h3_head_removal_negative')):
        text = cfg.get(key)
        if not isinstance(text, str) or not text.strip():
            text = cfg.defaults()['face_swap'][key.split('.', 1)[1]]
        workflow[node_id]['inputs']['text'] = text.strip()

    # Extra LoRAs on H3's model, chained onto whatever currently feeds the
    # guider — AFTER the speed patches, so switching `use_speed_nodes` off does
    # not move where they land. Borrowed from the Klein swap engine rather than
    # reimplemented: every rule it enforces (a file that is not on disk is
    # SKIPPED rather than failing the whole batch at ComfyUI's validator, a
    # strength of 0 means the row is off, and a LoRA the graph already loads is
    # refused because chaining it twice sums both deltas into visible
    # macro-blocking) was paid for once already.
    # Sampler steps. The graph ships 25, which is right for the stock model and
    # wrong for every reason someone adds a LoRA above: an accelerator is a
    # step-distill, and running a 4-step distill for 25 steps is both slower
    # than the stock model and visibly worse. The two settings travel together
    # for that reason — the picker only offers this field once a LoRA is there.
    # 0 keeps whatever the graph carries, so an untouched install is untouched.
    steps = cfg.get('minimax_h3.swap_steps')
    steps = int(steps) if isinstance(steps, (int, float)) else 0
    if steps > 0:
        workflow[NODE_H3_STEPS]['inputs']['steps'] = max(1, min(100, steps))

    lora_rows = configured_h3_swap_loras()
    if lora_rows:
        from .face_swap_helper import append_model_loras
        added = append_model_loras(
            workflow, lora_rows,
            sink_node=NODE_H3_MODEL_SINK, prefix='h3swap_lora_',
            title='H3 LoRA {i} (Settings)')
        if added:
            logger.info('h3 swap (new): %d extra LoRA(s) on the H3 model', len(added))

    # Everything a switched-off stage or accelerator left behind goes now, so
    # the node probe below asks about the job that will actually run.
    workflow = old.prune_to_outputs(workflow)
    gaps = missing_nodes(workflow)
    if gaps:
        raise H3SwapNewNodesMissing(gaps)

    workflow[NODE_TARGET_IMAGE]['inputs']['image'] = target_image
    workflow[NODE_REF_IMAGE]['inputs']['image'] = ref_image
    # The frame packet length used to live on its own PrimitiveInt; the
    # 2026-08-16 graph writes it straight onto the H3 node. Same setting, same
    # clamp — only the address changed.
    length = cfg.get('minimax_h3.length')
    if length is None:
        length = workflow[NODE_H3]['inputs'].get('length', mh.LENGTH_MIN)
    workflow[NODE_H3]['inputs']['length'] = mh.clamp_length(length)
    # `ImageFromBatch` takes the frame at an INDEX; `H3FrameSelect` scored the
    # packet and picked a winner. With length pinned to 1 for a still there is
    # one frame to take, so index 0 is the whole choice and
    # `minimax_h3.frame_weight_reference` has nothing left to weigh here.
    workflow[NODE_FRAME_SELECT]['inputs']['batch_index'] = 0
    workflow[NODE_FRAME_SELECT]['inputs']['length'] = 1
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
    # The head analysis lands in the SAME node, for the same reason: the H3
    # node's `prompt` is a link. `Text Concatenate` used to join the two — with
    # the call moved app-side there is one string and one writer, and the concat
    # node leaves with the rest of the stage's branch.
    if head_analysis and 'ollama' in kept:
        workflow[NODE_PROMPT_TEXT]['inputs']['text'] = (
            workflow[NODE_PROMPT_TEXT]['inputs']['text'].rstrip()
            + '\n' + HEAD_ANALYSIS_LEAD_IN
            + '\n\n' + str(head_analysis).strip())
    # Which reference pipeline the H3 node runs on. Its OWN key
    # (`face_swap.h3_ref_image_size`, blank = the shipped graph's value — 'max'
    # on this graph, 'match' on the old): the swap used to inherit the
    # generation lane's `minimax_h3.ref_image_size`, whose default 'match'
    # silently overwrote this graph's shipped 'max', i.e. the lower-likeness
    # pipeline on the one job where likeness is the product.
    ref_size = old.swap_ref_image_size()
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
    #
    # Unconditional since the 2026-08-16 graph. `face_swap.h3_mask_source` used
    # to choose between this and a segmenter inside the workflow; that graph's
    # segmenter is SAM3, which is the SAME model the app already runs, so the
    # in-graph one only added a node pack and a checkpoint to install. The key
    # still steers the OLD engine, where the in-graph masker is a different
    # model (PersonMaskUltra) rather than a second copy of this one.
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
    staged_mask = os.path.basename(comfy_fs.stage_input_image(
        mask_path, f'h3swapnew_mask_{uid}_{target_stem}.png', comfy_input_dir))
    staged_inputs.append(staged_mask)

    hint = None
    if cfg.get('face_swap.h3_pose_hint') is not False:
        from .face_swap_pose import pose_hint as _pose_hint
        meta = extra_metadata or {}
        hint = _pose_hint(meta.get('variation_prompt'), meta.get('framing'),
                          meta.get('variation_label'))
    # The stages are read ONCE here and handed down, because the head analysis
    # has to run before the workflow is built and both must agree about whether
    # the stage is on. It reads the tile from DISK, not the staged copy — same
    # bytes, and it keeps the analysis independent of ComfyUI being usable.
    stages = enabled_stages()
    analysis = analyse_head(target_path) if stages.get('ollama') else None
    workflow, kept = build_swap_workflow(
        staged_inputs[0], staged_inputs[1],
        filename_prefix=f'{user_id}_H3SwapNew_{uid}', mask_image=staged_mask,
        stages=stages, pose_hint=hint, head_analysis=analysis)
    if hint and 'ollama' not in kept:
        logger.info('h3 swap (new): pose hint — %s', hint)
    if analysis:
        logger.info('h3 swap (new): head analysis (%s, CPU) — %s',
                    swap_ollama_model(), analysis.replace('\n', ' ')[:300])
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
