"""Enqueue a SINGLE fixed face-swap ComfyUI job: node 423 = a tile's current
image (the TARGET, whose head is masked out and repainted), node 221 = the
dataset's reference photo (the identity SOURCE, which rides in as a reference
latent). Ships 'face swap.json' — the maintainer's 'klein swap.json' with its
display-only leaves dropped (see test_face_swap_workflow_shape.py).

THE 2026-08-09 GRAPH REPLACEMENT
--------------------------------
The workflow was swapped for a rebuilt one and NOTHING about its shape carried
over: 28 nodes became 52, every id this helper rewires changed, and the method
changed with them. The old graph masked with `LayerMask: PersonMaskUltra V2`
and sampled once; the new one segments the head with three `SAM3Segment` passes
(hair / face / head, unioned), crops to the mask, and runs TWO samplers — a
`SamplerCustomAdvanced` stage that builds the swapped head from the reference,
then `LanPaint_KSampler` inpainting it back into the target. So there are TWO
seeds to randomise, not one; missing the second would make every tile of a
batch identical in its first stage.

Klein 9B + Qwen3-8B TE + Flux.2 VAE are the SAME shared assets
klein_edit_helper.py already requires for every other Klein job on this
dataset; only the swap LoRA itself
('klein/Klein2-9B-SmartCharacterSwap.safetensors') is specific to this
feature, so it is checked separately and never auto-downloaded.

TWO THINGS THE MAINTAINER'S GRAPH HARDCODES, AND WHAT WE DO WITH THEM
---------------------------------------------------------------------
* `OTUNetLoaderW8A8` loading a specific INT8 build. The loader is kept — it is
  materially faster — but `unet_name` is RESOLVED, and when no INT8/W8A8 Klein
  build is on disk the node degrades to core `UNETLoader`, because handing a
  W8A8 loader an fp8 checkpoint is not a slow success, it is a failure.
* A second `LoraLoaderModelOnly` (a style LoRA, node 424:253). Attached when
  the file is present, rewired around when it is not — an install without that
  private LoRA must still swap faces, just without its look.

The three `SAM3Segment` prompts stay as tuned ("woman's hair/face/head",
maintainer's call, 2026-08-09). They are what the graph was measured with; a
different subject degrades the mask rather than failing, and inventing a
substitution would change results nobody has looked at."""
from __future__ import annotations
import logging
import os
import random
import uuid

from .. import config as cfg
from . import comfy_model_paths
from .klein_edit_helper import (
    KLEIN_REQUIRED, KleinModelsMissing, klein_missing_assets, klein_missing_nodes,
    resolve_klein_text_encoder, resolve_klein_unet, resolve_klein_vae,
)
from ..utils import comfy_fs
from ..utils.comfyui import load_workflow_local
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

WORKFLOW_FACE_SWAP_PATH = cfg.BACKEND_DIR / 'workflows' / 'face swap.json'

# Nodes this helper rewires — fail LOUDLY if the workflow file changes shape.
# Every one of these moved when the graph was replaced on 2026-08-09; the check
# is what turns "silently wrong image" into one named error.
NODE_TARGET_IMAGE = '423'        # LoadImage — the tile being repainted
NODE_REF_IMAGE = '221'           # LoadImage — the identity to graft on
NODE_UNET = '424:251'            # OTUNetLoaderW8A8 (may degrade to UNETLoader)
NODE_VAE = '424:104'
NODE_CLIP = '424:175'
NODE_SWAP_LORA = '424:246'       # the LoRA that IS the effect
NODE_STYLE_LORA = '424:253'      # optional; dropped when the file is absent
NODE_SEED_INPAINT = '261'        # LanPaint_KSampler   -> 'seed'
NODE_SEED_NOISE = '425:211'      # RandomNoise         -> 'noise_seed'
NODE_SAVE = '156'

_REQUIRED_NODES = (NODE_TARGET_IMAGE, NODE_REF_IMAGE, NODE_UNET, NODE_VAE,
                   NODE_CLIP, NODE_SWAP_LORA, NODE_STYLE_LORA,
                   NODE_SEED_INPAINT, NODE_SEED_NOISE, NODE_SAVE)

# Tokens that mark a Klein build the W8A8 loader can actually take. Narrow on
# purpose: a wrong match here loads a checkpoint the loader cannot quantise.
_INT8_TOKENS = ('int8', 'w8a8')
# ...and among those, a FEW-STEP build wins. This graph schedules 4 steps
# (Flux2Scheduler) and 6 (LanPaint_KSampler): a full-step Klein at 4 steps is
# not a slightly worse face, it is mush. Measured the hard way — the first
# version of this resolver picked INT8 builds alphabetically and chose a
# non-turbo one over the turbo build sitting beside it.
_FEW_STEP_TOKENS = ('turbo', 'lightning', 'hyper', 'schnell')

# The LoRA that IS the face-swap effect (not the optional consistency LoRA
# klein_edit_helper injects elsewhere) — required for this workflow alone.
FACE_SWAP_LORA_NAME = os.path.join('klein', 'Klein2-9B-SmartCharacterSwap.safetensors')


class FaceSwapLoraMissing(Exception):
    """The face-swap LoRA itself is not on disk — a valid job can't be built.
    Unlike the shared Klein assets, this one has no auto-download action, so
    the caller must be told exactly where to place the file."""

    def __init__(self, name=FACE_SWAP_LORA_NAME):
        self.name = name
        super().__init__(
            f'the face-swap LoRA is not on disk: {name} — place it under '
            'ComfyUI/models/loras/klein/ and retry')


def _face_swap_lora_abs(name=FACE_SWAP_LORA_NAME):
    """Absolute path of a LoRA under the first loras search root that holds it,
    else None. Mirrors klein_edit_helper's own private lora-lookup technique
    (comfy_model_paths.search_roots) rather than reaching into that module's
    underscore-prefixed helper.

    Takes a name so the OPTIONAL style LoRA can be probed the same way; the
    default stays the required swap LoRA."""
    for root in comfy_model_paths.search_roots('loras'):
        cand = os.path.join(root, str(name).replace('\\', os.sep))
        if os.path.exists(cand):
            return cand
    return None


def resolve_swap_unet(workflow=None):
    """(unet_name, int8) for the face-swap graph's diffusion loader.

    `int8` is False when no INT8/W8A8 Klein build is on disk, and the caller
    then degrades `OTUNetLoaderW8A8` to core `UNETLoader`. Handing a W8A8
    loader an fp8 checkpoint does not run slower — it fails — so this is a
    correctness switch, not a performance one.

    Preference among INT8 builds: a FEW-STEP one first. The graph's own
    schedule is 4 and 6 steps, so a full-step checkpoint here returns mush
    rather than a worse face — see `_FEW_STEP_TOKENS`. The graph's committed
    `unet_name` is deliberately NOT consulted: it named the maintainer's own
    checkpoint, and a file that happens to share that name on someone else's
    disk is not the same weights."""
    candidates = []
    for root in comfy_model_paths.search_roots('diffusion_models'):
        if not os.path.isdir(root):
            continue
        for dirpath, _sub, filenames in os.walk(root, followlinks=True):
            for fn in sorted(filenames):
                low = fn.lower()
                if not low.endswith(('.safetensors', '.gguf', '.sft')):
                    continue
                if 'klein' in low and any(t in low for t in _INT8_TOKENS):
                    rel = os.path.relpath(os.path.join(dirpath, fn), root)
                    few = any(t in low for t in _FEW_STEP_TOKENS)
                    candidates.append((0 if few else 1, rel))
    if candidates:
        return sorted(candidates)[0][1], True
    return resolve_klein_unet(), False


MAX_FACE_SWAP_LORAS = 8          # mirrors klein_edit_helper.MAX_GENERATION_LORAS
NODE_MODEL_SINK = '262'          # DifferentialDiffusion — first consumer of the
                                 # model chain; used to FIND the chain's tail
_LORA_NODE_PREFIX = 'faceswap_lora_'


def configured_face_swap_loras():
    """Sanitized `klein.face_swap_loras`: ordered [{file, strength}] with blank
    rows dropped, strengths clamped to [0, 1.5] (junk -> 1.0) and the list
    capped at MAX_FACE_SWAP_LORAS.

    A FLAT list, not the named presets the Klein generation lane uses. Presets
    exist there because the workspace offers a per-run picker; face swap is one
    fixed action with no picker, so a preset would be a name nobody ever
    selects. Same fail-closed discipline though: config is the only place files
    and order can come from."""
    raw = cfg.get('klein.face_swap_loras')
    rows = []
    for e in (raw if isinstance(raw, list) else []):
        if not isinstance(e, dict):
            continue
        f = e.get('file')
        f = f.strip() if isinstance(f, str) else ''
        if not f:
            continue
        s = e.get('strength')
        s = float(s) if isinstance(s, (int, float)) else 1.0
        rows.append({'file': f, 'strength': max(0.0, min(1.5, s))})
        if len(rows) >= MAX_FACE_SWAP_LORAS:
            break
    return rows


def append_model_loras(workflow, rows):
    """Chain extra LoraLoaderModelOnly nodes onto the END of the model chain,
    after the swap LoRA and the optional style LoRA. Returns the ids added.

    The tail is FOUND, not hardcoded: it is whatever currently feeds
    NODE_MODEL_SINK's `model`, so this keeps working when the style LoRA was
    dropped for being absent — and would keep working if the graph gains
    another patch node before the sink.

    A row naming a file that is not on disk is SKIPPED with a warning rather
    than queued: ComfyUI would answer a validation 400 for the whole job, and
    losing every tile of a batch to one stale filename in a settings list is a
    bad trade for a LoRA the user can re-add in one click."""
    sink = workflow.get(NODE_MODEL_SINK, {}).get('inputs', {}).get('model')
    if not (rows and isinstance(sink, list) and len(sink) == 2):
        return []
    usable = []
    for row in rows:
        if _face_swap_lora_abs(row['file']):
            usable.append(row)
        else:
            logger.warning('face swap: LoRA %r is not on disk — skipped',
                           row['file'])
    if not usable:
        return []

    anchor = list(sink)
    added, new_nodes = [], {}
    for i, row in enumerate(usable, 1):
        nid = f'{_LORA_NODE_PREFIX}{i}'
        new_nodes[nid] = {
            'class_type': 'LoraLoaderModelOnly',
            'inputs': {'lora_name': row['file'],
                       'strength_model': row['strength'],
                       'model': anchor},
            '_meta': {'title': f'Extra LoRA {i} (Settings)'},
        }
        anchor = [nid, 0]
        added.append(nid)
    # Repoint the EXISTING consumers of the old tail before the new nodes land,
    # so the chain cannot be spliced onto itself.
    old_tail = list(sink)
    for node in workflow.values():
        for key, value in (node.get('inputs') or {}).items():
            if key == 'model' and isinstance(value, list) and value == old_tail:
                node['inputs'][key] = list(anchor)
    workflow.update(new_nodes)
    return added


def drop_passthrough(workflow, node_id, input_key='model'):
    """Remove a pass-through node and rewire every consumer to its own source.

    Used for the optional style LoRA: an install without that private file must
    still swap faces. Returns True when the node was removed."""
    node = workflow.get(node_id)
    if node is None:
        return False
    upstream = (node.get('inputs') or {}).get(input_key)
    if not (isinstance(upstream, list) and len(upstream) == 2):
        return False
    for other in workflow.values():
        for key, value in (other.get('inputs') or {}).items():
            if isinstance(value, list) and len(value) == 2 and value[0] == node_id:
                other['inputs'][key] = list(upstream)
    workflow.pop(node_id, None)
    return True


def _comfy_input_dir() -> str:
    """Same tiny lookup klein_edit_helper/krea_edit_helper each keep their own
    copy of, rather than importing one another's private helper."""
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def enqueue_face_swap(user_id, target_path, ref_path, extra_metadata=None):
    """Copy the tile's current image (target_path, image1) and the dataset's
    reference photo (ref_path, image2) into ComfyUI input, configure the
    fixed face-swap workflow, and enqueue it. Returns the app job_id.

    Raises ValueError on a missing source image / unloadable workflow /
    missing required node, FaceSwapLoraMissing if the swap LoRA itself is
    absent, KleinModelsMissing if a SHARED Klein asset (model/VAE/text
    encoder) is absent, RuntimeError if ComfyUI isn't configured, and
    KleinNodesMissing (face_dataset_service) itself if the target ComfyUI
    lacks a custom node this graph needs — unlike enqueue_klein_edit, that
    check is NOT the caller's job here; this function raises it directly."""
    if not target_path or not os.path.exists(target_path):
        raise ValueError(f"target image not found: {target_path}")
    if not ref_path or not os.path.exists(ref_path):
        raise ValueError(f"reference image not found: {ref_path}")
    workflow = load_workflow_local(str(WORKFLOW_FACE_SWAP_PATH))
    if not workflow:
        raise ValueError("failed to load face swap workflow")
    for node in _REQUIRED_NODES:
        if node not in workflow:
            raise ValueError(f"workflow node {node} missing — face swap.json has changed")

    unet_ref, unet_is_int8 = resolve_swap_unet(workflow)
    vae_ref = resolve_klein_vae()
    te_ref = resolve_klein_text_encoder()
    missing = klein_missing_assets()
    if any(a in missing for a in KLEIN_REQUIRED):
        raise KleinModelsMissing(missing)
    if not _face_swap_lora_abs():
        raise FaceSwapLoraMissing()

    # The graph's diffusion loader is INT8-only. Without an INT8/W8A8 build on
    # disk it is swapped for the core loader BEFORE the node-availability probe
    # below, so an install that lacks the OTU pack is not told to install a pack
    # it will not use.
    if not unet_is_int8:
        workflow[NODE_UNET] = {
            'class_type': 'UNETLoader',
            'inputs': {'unet_name': unet_ref, 'weight_dtype': 'default'},
            '_meta': {'title': 'Klein (no INT8 build found)'},
        }
    # Optional style LoRA: attach when present, rewire around it when not.
    style_name = (workflow.get(NODE_STYLE_LORA, {}).get('inputs', {})
                  .get('lora_name'))
    if style_name and not _face_swap_lora_abs(style_name):
        drop_passthrough(workflow, NODE_STYLE_LORA, 'model')
    # ...then the user's own, from Settings, at the END of the chain.
    append_model_loras(workflow, configured_face_swap_loras())

    missing_nodes = klein_missing_nodes(workflow=workflow)
    if missing_nodes:
        from .face_dataset_service import KleinNodesMissing
        raise KleinNodesMissing([], missing_nodes)

    comfy_input_dir = comfy_fs.ensure_input_usable(_comfy_input_dir())
    uid = uuid.uuid4().hex[:8]
    target_stem = os.path.splitext(os.path.basename(str(target_path)))[0] or 'target'
    ref_stem = os.path.splitext(os.path.basename(str(ref_path)))[0] or 'ref'
    staged_target = comfy_fs.stage_input_image(
        target_path, f"faceswap_target_{uid}_{target_stem}.png", comfy_input_dir)
    staged_ref = comfy_fs.stage_input_image(
        ref_path, f"faceswap_ref_{uid}_{ref_stem}.png", comfy_input_dir)
    staged_inputs = [os.path.basename(staged_target), os.path.basename(staged_ref)]

    workflow[NODE_TARGET_IMAGE]["inputs"]["image"] = staged_inputs[0]
    workflow[NODE_REF_IMAGE]["inputs"]["image"] = staged_inputs[1]
    workflow[NODE_UNET]["inputs"]["unet_name"] = unet_ref
    workflow[NODE_VAE]["inputs"]["vae_name"] = vae_ref
    workflow[NODE_CLIP]["inputs"]["clip_name"] = te_ref
    # BOTH seeds. The graph samples twice — the head is built from the reference
    # by SamplerCustomAdvanced (RandomNoise), then inpainted back into the
    # target by LanPaint_KSampler. Randomising only one leaves every tile of a
    # batch sharing a stage, which reads as "the swap ignores my seed".
    workflow[NODE_SEED_NOISE]["inputs"]["noise_seed"] = random.randint(0, 2 ** 64 - 1)
    workflow[NODE_SEED_INPAINT]["inputs"]["seed"] = random.randint(0, 2 ** 64 - 1)
    # Unique prefix per job — see klein_edit_helper's node-9 comment: a shared
    # prefix makes ComfyUI's own output counter re-issue the same filename,
    # so every tile would end up displaying the SAME result image.
    workflow[NODE_SAVE]["inputs"]["filename_prefix"] = f"{user_id}_FaceSwap_{uid}"

    job_id = str(uuid.uuid4())
    meta = {"model_name": "klein_face_swap_dataset"}
    if extra_metadata:
        meta.update(extra_metadata)
    meta["staged_inputs"] = staged_inputs
    queue_manager.add_job(job_type="image", user_id=str(user_id), workflow_data=workflow,
                          prompt="Face swap (reference identity)", job_id=job_id, metadata=meta)
    return job_id
