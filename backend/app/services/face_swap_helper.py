"""Enqueue a SINGLE fixed face-swap ComfyUI job: image1 = a tile's current
image (the target), image2 = the dataset's reference photo (the identity
source). Ships 'face swap.json' — the user-provided 'klein swap.json' with
its display-only rgthree Image Comparer node dropped (see
test_face_swap_workflow_shape.py). Klein 9B + Qwen3-8B TE + Flux.2 VAE are the
SAME shared assets klein_edit_helper.py already requires for every other
Klein job on this dataset; only the swap LoRA itself
('klein/Klein2-9B-SmartCharacterSwap.safetensors') is specific to this
feature, so it is checked separately and never auto-downloaded."""
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
_REQUIRED_NODES = ('151', '121', '165:126', '165:102', '165:146',
                   '165:161', '165:156', '9')

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


def _face_swap_lora_abs():
    """Absolute path of the fixed face-swap LoRA under the first loras search
    root that holds it, else None. Mirrors klein_edit_helper's own private
    lora-lookup technique (comfy_model_paths.search_roots) rather than
    reaching into that module's underscore-prefixed helper."""
    for root in comfy_model_paths.search_roots('loras'):
        cand = os.path.join(root, FACE_SWAP_LORA_NAME)
        if os.path.exists(cand):
            return cand
    return None


def _comfy_input_dir() -> str:
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
    encoder) is absent, RuntimeError if ComfyUI isn't configured. A
    custom-node gap surfaces via klein_missing_nodes — callers check that
    themselves via KleinNodesMissing (face_dataset_service), same as the
    existing Klein regenerate path."""
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

    unet_ref = resolve_klein_unet()
    vae_ref = resolve_klein_vae()
    te_ref = resolve_klein_text_encoder()
    missing = klein_missing_assets()
    if any(a in missing for a in KLEIN_REQUIRED):
        raise KleinModelsMissing(missing)
    if not _face_swap_lora_abs():
        raise FaceSwapLoraMissing()

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

    workflow["151"]["inputs"]["image"] = staged_inputs[0]
    workflow["121"]["inputs"]["image"] = staged_inputs[1]
    workflow["165:126"]["inputs"]["unet_name"] = unet_ref
    workflow["165:102"]["inputs"]["vae_name"] = vae_ref
    workflow["165:146"]["inputs"]["clip_name"] = te_ref
    workflow["165:156"]["inputs"]["seed"] = random.randint(0, 2 ** 64 - 1)
    # Unique prefix per job — see klein_edit_helper's node-9 comment: a shared
    # prefix makes ComfyUI's own output counter re-issue the same filename,
    # so every tile would end up displaying the SAME result image.
    workflow["9"]["inputs"]["filename_prefix"] = f"{user_id}_FaceSwap_{uid}"

    job_id = str(uuid.uuid4())
    meta = {"model_name": "klein_face_swap_dataset"}
    if extra_metadata:
        meta.update(extra_metadata)
    meta["staged_inputs"] = staged_inputs
    queue_manager.add_job(job_type="image", user_id=str(user_id), workflow_data=workflow,
                          prompt="Face swap (reference identity)", job_id=job_id, metadata=meta)
    return job_id
