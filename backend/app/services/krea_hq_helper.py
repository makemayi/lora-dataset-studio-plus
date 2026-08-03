"""Krea 2 Ostris Edit + SeedVR2 restore — the ✨ improve pass's 'klein' engine
implementation as of the user's own tested workflow (issue: Klein's edit
changed too much detail/color; this pipeline instead does a small edit pass
for tone/sharpness intent, colour-corrects it back to the source, then lets
SeedVR2 do the actual detail restoration/upscale).

Ships `workflows/krea2 high resolution.json` — the user's own tested
ComfyUI export, with one dead disconnected node (`ImageColorMatch+`, no
consumer anywhere in the graph) dropped. The graph, in order:

    LoadImage -> resize to 512 longer edge -> Krea2 Ostris Edit (small,
    tone/sharpness pass, prompt-driven) -> ColorTransfer (recolor the Krea
    edit back to the ORIGINAL's tone — reinhard_lab) -> SeedVR2 (tiled
    VAE encode/KSampler/decode restore+upscale to 1536px shorter edge, with
    its own LAB colour-correction pass) -> SaveImage.

Two DIFFERENT node families than this app already integrates elsewhere:
this is NOT krea_edit_helper's Krea2 pack (`Krea2EditModelPatch` /
`Krea2EditGroundedEncode`) — it is an 'Ostris' variant
(`Krea2OstrisEditModelPatch` / `TextEncodeKrea2OstrisEdit`). And it is NOT
seedvr2_helper's SeedVR2 pack (`SeedVR2LoadDiTModel` / `SeedVR2VideoUpscaler`,
models under the `SEEDVR2` folder type) — it is a lower-level tiled pipeline
(`SeedVR2Preprocess` / `SeedVR2Conditioning` / `SeedVR2PostProcessing` +
plain `UNETLoader`/`VAELoader` + core `VAEEncodeTiled`/`VAEDecodeTiled`),
though it shares the SAME `SEEDVR2` model folder and even the same VAE
filename (`ema_vae_fp16.safetensors`) as that pack — so its weights search
reuses `seedvr2_helper.MODEL_FOLDER`. The three `SeedVR2Xxx` node classes
here are INFERRED to ship in the same `ComfyUI-SeedVR2_VideoUpscaler` pack
(same author ecosystem, same folder convention) — not measured live against
that pack's current /object_info, unlike the rest of this app's node-pack
hints. `FluxKontextImageScale` / `FluxKontextMultiReferenceLatentMethod` /
`VAEEncodeTiled` / `VAEDecodeTiled` are treated as ComfyUI-core (no pack
hint needed) — also inferred, not measured here.

NO auto-download for any of the six models below (unlike Klein/Krea/SeedVR2's
OWN 409s): the Krea2 Turbo/CLIP/VAE triple happens to already be
auto-installable through krea_edit_helper's mechanism for the OTHER Krea
engine and this module reuses those SAME resolvers, but this module itself
starts nothing — the quality LoRA and the specific 7B-sharp-int8-convrot
SeedVR2 build are the user's own curated files with no verified public URL,
so a missing weight here is always a named 409 telling the user exactly
where to place it, never a fake installer."""
from __future__ import annotations
import os
import random
import time
import uuid

from .. import config as cfg
from . import comfy_model_paths
from ..utils import comfy_fs
from ..utils.comfyui import load_workflow_local
from ..job_queue import queue_manager

WORKFLOW_PATH = cfg.BACKEND_DIR / 'workflows' / 'krea2 high resolution.json'

# Nodes this helper rewires — fail LOUDLY if the workflow file changes shape
# instead of silently enqueuing a job with the wrong source/model/prompt.
_REQUIRED_NODES = ('41771', '41730', '41727:41604', '41727:41600', '41729',
                   '41770:41760', '41770:41761', '41727:41605',
                   '41727:41734', '41770:41766', '41735')

DEFAULT_PROMPT = 'photorealistic'  # the tested default TextEncodeKrea2OstrisEdit prompt


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


# --- Krea2 assets: reuse krea_edit_helper's OWN resolvers (canonical-first,
# narrow-token fallback) — the Turbo build / Qwen3-VL clip / Qwen Image VAE
# this graph wants are the SAME assets that engine already resolves. -------

def krea_unet():
    from . import krea_edit_helper as keh
    return keh.resolve_krea_unet()


def krea_clip():
    from . import krea_edit_helper as keh
    return keh.resolve_krea_text_encoder()


def krea_vae():
    from . import krea_edit_helper as keh
    return keh.resolve_krea_vae()


# --- The one NEW Krea-side asset: the quality LoRA, not krea_edit_helper's
# identity-edit LoRA — different purpose, different file. ------------------
_QUALITY_LORA_CANONICAL = 'high quality.safetensors'
_QUALITY_LORA_TOKENS = ('high_quality', 'high-quality', 'highquality', 'high quality')


def resolve_quality_lora():
    """(relative_name, absolute_path) of the Krea 2 'high quality' LoRA, or
    (None, None) when it isn't on disk. Canonical basename first (preferring
    a 'krea2' folder when more than one file matches it), else a narrow
    token scan of every loras root — never a blind first-file guess."""
    listings = sorted(comfy_model_paths.list_models('loras'))
    canon = [(rel, ab) for rel, ab in listings
             if os.path.basename(rel).lower() == _QUALITY_LORA_CANONICAL]
    if canon:
        for rel, ab in canon:
            if 'krea2' in rel.lower():
                return rel, ab
        return canon[0]
    for rel, ab in listings:
        low = os.path.basename(rel).lower()
        if any(tok in low for tok in _QUALITY_LORA_TOKENS):
            return rel, ab
    return None, None


# --- SeedVR2 assets: same `SEEDVR2` model folder as seedvr2_helper (its
# canonical VAE filename is byte-identical to what this graph wants), but
# THIS graph's UNETLoader wants a specific 7B-sharp-int8 build seedvr2_helper
# has no concept of, so the DiT resolver is its own narrow scan. -----------
_SEEDVR2_UNET_CANONICAL = 'seedvr2_7b_sharp_int8_convrot.safetensors'
_SEEDVR2_UNET_TOKENS = ('7b_sharp', 'sharp_int8')


def resolve_seedvr2_unet():
    """Bare filename for the SeedVR2 UNETLoader, or None. Canonical name
    first, else a narrow '7B sharp' token match across every SEEDVR2 search
    root — a WRONG DiT build is worse than a missing one (different
    parameter count, different weights), so this never falls back to
    whatever else happens to be in the folder (unlike seedvr2_helper's own
    DiT resolver, which serves a picker UI where any build is a valid
    choice)."""
    from . import seedvr2_helper
    listings = []
    for root in comfy_model_paths.search_roots(seedvr2_helper.MODEL_FOLDER):
        try:
            names = sorted(n for n in os.listdir(root)
                           if n.lower().endswith(('.safetensors', '.gguf')))
        except OSError:
            continue
        listings.append(names)
    if any(_SEEDVR2_UNET_CANONICAL in names for names in listings):
        return _SEEDVR2_UNET_CANONICAL
    for names in listings:
        for n in names:
            if seedvr2_helper._is_vae_name(n):
                continue
            if any(tok in n.lower() for tok in _SEEDVR2_UNET_TOKENS):
                return n
    return None


def resolve_seedvr2_vae():
    """SeedVR2 VAE — byte-identical canonical filename to seedvr2_helper's
    own VAE (`ema_vae_fp16.safetensors`), same folder, so this just IS that
    resolver."""
    from . import seedvr2_helper
    return seedvr2_helper.resolve_seedvr2_vae()


KREA_HQ_ASSETS = {
    'krea_model': {
        'kind': 'Krea 2 Turbo base model',
        'path': 'models/diffusion_models/Krea/Krea2_Turbo_convrot_int8mixed.safetensors',
        'source': 'https://huggingface.co/Comfy-Org/Krea-2/tree/main/diffusion_models '
                  '(or your own converted Turbo build)',
    },
    'krea_text_encoder': {
        'kind': 'Qwen3-VL 4B text encoder',
        'path': 'models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors',
        'source': 'https://huggingface.co/Comfy-Org/Krea-2/tree/main/text_encoders',
    },
    'krea_vae': {
        'kind': 'Qwen Image VAE',
        'path': 'models/vae/qwen_image_vae.safetensors',
        'source': 'https://huggingface.co/Comfy-Org/Krea-2/tree/main/vae',
    },
    'krea_quality_lora': {
        'kind': "Krea 2 'high quality' LoRA",
        'path': 'models/loras/krea2/high quality.safetensors',
        'source': 'wherever you obtained this LoRA — not auto-fetched',
    },
    'seedvr2_model': {
        'kind': 'SeedVR2 7B Sharp (int8, convrot) DiT model',
        'path': 'models/SEEDVR2/seedvr2_7b_sharp_int8_convrot.safetensors',
        'source': 'wherever you obtained this specific build — not auto-fetched '
                  '(the SeedVR2 setup card installs a different variant/node interface)',
    },
    'seedvr2_vae': {
        'kind': 'SeedVR2 VAE',
        'path': 'models/SEEDVR2/ema_vae_fp16.safetensors',
        'source': 'https://huggingface.co/numz/SeedVR2_comfyUI',
    },
}
KREA_HQ_REQUIRED = tuple(KREA_HQ_ASSETS)


class KreaHQModelsMissing(Exception):
    """A Krea2-HQ asset is not on disk and/or the target ComfyUI is missing a
    node class this graph needs, so no valid job can be built. Raised BEFORE
    any row or job is created. `.missing` = asset keys (subset of
    KREA_HQ_REQUIRED); `.missing_nodes` = class_type strings absent from
    /object_info."""

    def __init__(self, missing, missing_nodes=None):
        self.missing = list(missing or [])
        self.missing_nodes = list(missing_nodes or [])
        super().__init__('Krea2 HQ restore assets missing: '
                         + ', '.join(self.missing + self.missing_nodes))


def krea_hq_missing_assets():
    """Which KREA_HQ_ASSETS keys are NOT on disk. Disk-only, network-free —
    safe for the readiness probe."""
    missing = []
    if not krea_unet():
        missing.append('krea_model')
    if not krea_clip():
        missing.append('krea_text_encoder')
    if not krea_vae():
        missing.append('krea_vae')
    if not resolve_quality_lora()[0]:
        missing.append('krea_quality_lora')
    if not resolve_seedvr2_unet():
        missing.append('seedvr2_model')
    if not resolve_seedvr2_vae():
        missing.append('seedvr2_vae')
    return missing


def missing_file_entries(missing):
    """[{path, kind, source}] for each missing asset key — the same shape
    krea_edit_helper/seedvr2_helper already publish, so one banner covers
    every engine."""
    out = []
    for key in missing or []:
        meta = KREA_HQ_ASSETS.get(key)
        if meta:
            out.append({'path': meta['path'], 'kind': meta['kind'], 'source': meta['source']})
    return out


# --- Custom-node preflight. Pack hints are given ONLY where reasonably
# confident (see module docstring) — everything else falls back to the
# generic "class X is missing, find its pack yourself" shape every other
# engine's preflight already degrades to for an unmapped class. ------------
KREA_HQ_NODE_PACK = {
    'pack': 'ComfyUI-SeedVR2_VideoUpscaler',
    'url': 'https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler',
    'search': 'SeedVR2',
}
_SEEDVR2_TILED_CLASSES = ('SeedVR2Preprocess', 'SeedVR2Conditioning', 'SeedVR2PostProcessing')

_NODES_OK_TTL_S = 300
_nodes_ok_until = 0.0


def _workflow_class_types(workflow):
    return {n.get('class_type') for n in (workflow or {}).values()
            if isinstance(n, dict) and n.get('class_type')}


def krea_hq_missing_nodes(workflow=None):
    """[{class_type, pack, url}] for every node class the shipped workflow
    needs that the target ComfyUI does NOT expose. FAIL-OPEN when
    /object_info can't be fetched, same contract as every sibling helper."""
    global _nodes_ok_until
    shipped = workflow is None
    if shipped:
        if time.time() < _nodes_ok_until:
            return []
        workflow = load_workflow_local(str(WORKFLOW_PATH)) or {}
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = []
    for ct in sorted(_workflow_class_types(workflow) - available):
        if ct in _SEEDVR2_TILED_CLASSES:
            out.append({'class_type': ct, 'pack': KREA_HQ_NODE_PACK['pack'],
                        'url': KREA_HQ_NODE_PACK['url']})
        else:
            out.append({'class_type': ct, 'pack': None, 'url': None})
    if shipped and not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def node_hints(missing_nodes):
    """[{class_type, pack, url}] — `krea_hq_missing_nodes` already returns
    this shape; kept as a thin alias so the route imports one consistent
    name across every engine helper."""
    return list(missing_nodes or [])


def clear_nodes_cache():
    global _nodes_ok_until
    _nodes_ok_until = 0.0


def preflight():
    """Raise KreaHQModelsMissing when the pipeline cannot run."""
    missing = krea_hq_missing_assets()
    nodes = krea_hq_missing_nodes()
    if missing or nodes:
        raise KreaHQModelsMissing(missing, nodes)


def enqueue_krea_hq_improve(user_id, source_filename, edit_prompt='', source_path=None,
                            extra_metadata=None):
    """Copy the source into ComfyUI input, resolve every loader against
    what's ACTUALLY installed, wire the shipped Krea2-HQ workflow and
    enqueue it. Returns the app job_id.

    `edit_prompt` feeds the Krea2 Ostris Edit text encode (the SAME editable
    klein_improve instruction the Klein engine used) — blank falls back to
    the workflow's own tested default ('photorealistic') rather than an
    empty string, since this node's positive branch was never tested empty.

    Raises ValueError on a missing source / unloadable workflow / missing
    required node, KreaHQModelsMissing when an asset or a node is absent,
    RuntimeError if ComfyUI isn't configured."""
    if source_path is None:
        out_dir = cfg.comfyui_dir('output')
        if not out_dir:
            raise RuntimeError('ComfyUI is not configured')
        source_path = os.path.join(str(out_dir), source_filename)
    if not os.path.exists(source_path):
        raise ValueError(f"source image not found: {source_filename}")
    workflow = load_workflow_local(str(WORKFLOW_PATH))
    if not workflow:
        raise ValueError("failed to load Krea2 HQ restore workflow")
    for node in _REQUIRED_NODES:
        if node not in workflow:
            raise ValueError(
                f"workflow node {node} missing — krea2 high resolution.json has changed")

    preflight()
    krea_unet_ref = krea_unet()
    krea_clip_ref = krea_clip()
    krea_vae_ref = krea_vae()
    quality_lora_ref, _quality_lora_path = resolve_quality_lora()
    seedvr2_unet_ref = resolve_seedvr2_unet()
    seedvr2_vae_ref = resolve_seedvr2_vae()

    comfy_input_dir = comfy_fs.ensure_input_usable(_comfy_input_dir())
    uid = uuid.uuid4().hex[:8]
    source_stem = os.path.splitext(os.path.basename(str(source_filename)))[0] or 'source'
    staged_source = comfy_fs.stage_input_image(
        source_path, f"krea_hq_{uid}_{source_stem}.png", comfy_input_dir)
    comfy_input = os.path.basename(staged_source)

    prompt_text = edit_prompt.strip() if isinstance(edit_prompt, str) and edit_prompt.strip() \
        else DEFAULT_PROMPT

    workflow["41771"]["inputs"]["image"] = comfy_input
    workflow["41730"]["inputs"]["unet_name"] = krea_unet_ref
    workflow["41727:41604"]["inputs"]["clip_name"] = krea_clip_ref
    workflow["41727:41600"]["inputs"]["vae_name"] = krea_vae_ref
    workflow["41729"]["inputs"]["lora_name"] = quality_lora_ref
    workflow["41727:41605"]["inputs"]["prompt"] = prompt_text
    workflow["41727:41734"]["inputs"]["noise_seed"] = random.randint(0, 2 ** 64 - 1)
    workflow["41770:41760"]["inputs"]["unet_name"] = seedvr2_unet_ref
    workflow["41770:41761"]["inputs"]["vae_name"] = seedvr2_vae_ref
    workflow["41770:41766"]["inputs"]["seed"] = random.randint(0, 2 ** 64 - 1)
    # Unique prefix per job — see klein_edit_helper's node-9 comment: a shared
    # prefix makes ComfyUI's own output counter re-issue the same filename.
    workflow["41735"]["inputs"]["filename_prefix"] = f"{user_id}_KreaHQ_{uid}"

    job_id = str(uuid.uuid4())
    meta = {"model_name": "krea2_hq_restore_dataset"}
    if extra_metadata:
        meta.update(extra_metadata)
    meta["staged_inputs"] = [comfy_input]
    queue_manager.add_job(job_type="image", user_id=str(user_id), workflow_data=workflow,
                          prompt=prompt_text, job_id=job_id, metadata=meta)
    return job_id
