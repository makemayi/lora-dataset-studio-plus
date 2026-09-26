"""Qwen-Image 2.1 — the FOURTH local generation engine, next to Klein, Krea and
MiniMax H3.

WHAT IT IS
----------
Qwen-Image 2.1 is Qwen's image-editing base model, supported NATIVELY by a
recent ComfyUI: the whole edit conditioning lives in the core node
``TextEncodeQwenImage21``. There is NO custom-node pack to install, and NO
identity-edit LoRA — the reference transfer IS the model's native skill. Given
a reference photo it re-stages the subject while keeping identity, driven by a
plain instruction, which is exactly the bootstrap case the dataset builder
needs.

THE GRAPH (validated against this ComfyUI's core source and a working local
export that loads the very files this resolver elects)
-------------------------------------------------------
    UNETLoader -> [LoraLoaderModelOnly chain] -> KSampler
    LoadImage -> TextEncodeQwenImage21.image_1 (more refs: image_2..)
    CLIPLoader(qwen3-vl 8B, type='qwen_image') ─┐
    VAELoader(qwen image 2.1 vae) ──────────────┴─> TextEncodeQwenImage21
        outputs positive / negative / empty latent (first ref's size)
    TextEncodeQwenImage21 -> KSampler(pos, neg, latent) -> VAEDecode -> SaveImage

WHY THIS IS SIMPLER THAN KREA — and where its dials went
--------------------------------------------------------
``TextEncodeQwenImage21`` replaces Krea's whole encode side: it resizes each
reference, VAE-encodes it into ``reference_latents``, shows it to the vision
encoder, and emits the output-sized empty latent. Consequences this module is
built on:

  * The consistency dial is the node's ``resolution`` input (the pixel budget
    every reference is resized to, and so the OUTPUT size too — the node
    preserves the reference's aspect). LOW = faster, softer likeness;
    HIGH = stronger likeness, more VRAM. It is ``qwen21.ref_resolution``,
    capped by ``qwen21.max_output_mp`` — Krea's grounding_px/ref_boost pair
    has no analogue here, because the reference pull is not a knob the graph
    owns.
  * cfg 4.0 with a REAL negative branch: the node takes ``negative_prompt``
    alongside ``prompt`` and encodes both. Unlike Krea (guidance-distilled,
    cfg pinned 1.0), 2.1 wants the guidance. The negative text is
    ``qwen21.negative_prompt`` (blank default); cfg itself stays pinned in
    code at the measured 4.0.
  * Output geometry comes from the reference: the caller does NOT pick
    width/height. The one size lever is ``resolution`` (rounded to /32 by the
    node itself), so Krea's fit_output_size machinery has no job here.

MEASURED SAMPLING PROFILE (local working export, 2026-09-23 — do not guess)
---------------------------------------------------------------------------
    KSampler: euler / simple, cfg 4.0, 30 steps, denoise 1.0
(a second pass at 12 steps / denoise 0.7 through the same conditioning is the
HQ lane's shape — Wave 2, see krea_hq_helper when that lands).

TEXT ENCODER — the 8B, not the 4B
----------------------------------
2.1's clip target is the FULL Qwen3-VL-8B (comfy picks it by checkpoint size
under CLIPLoader type='qwen_image'). The 4B file Krea uses produces the
classic Qwen-Image target instead — wrong model family. The resolver below
accepts only 8B-named files and never falls back to a 4B: a missing encoder
must surface as a named gap, not as subtly wrong embeddings.

VAE — the 2.1 one, not the Qwen-Image one
-----------------------------------------
The 2.1 latent format differs from Qwen-Image 1 (comfy.latent_formats has
separate classes). ``qwen_image_vae.safetensors`` — the file Krea ships — is
therefore NOT a fallback: decoding 2.1 latents through it is noise. Only
2.1-named VAE files resolve.

BASE MODEL — safetensors only
-----------------------------
``UNETLoader`` loads .safetensors; the GGUF builds of 2.1 need ComfyUI-GGUF's
own loader class and are deliberately NOT resolved (a resolver that returns a
file the graph cannot open is the exact failure the narrow-token discipline
exists to prevent).
"""
from __future__ import annotations
import logging
import os
import random
import time
import uuid

from .. import config as cfg
from . import comfy_model_paths
from ..utils import comfy_fs
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

ENGINE_ID = 'qwen21'
ENGINE_LABEL = 'Qwen Image 2.1'

# The dataset-variation job stamp below MUST stay a member of
# job_queue.DATASET_IMAGE_JOB_NAMES — the dispatch routes finished jobs by it,
# and a stamp missing from that set renders fine in ComfyUI and then NEVER
# attaches to its row (the Krea incident, twice, and this lane's first run).
# test_qwen21.py pins stamp and membership together; keep them a string literal
# so the harvest guard's AST walk keeps seeing it.

_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')

# The one core node class the graph cannot be built without, and the minimum
# that must be true of an install lacking it: ComfyUI is simply too old for
# Qwen-Image 2.1, and the fix is an update — there is no pack to clone.
QWEN21_NODE_CLASSES = ('TextEncodeQwenImage21',)
QWEN21_NODE_HINT = {
    'class_type': 'TextEncodeQwenImage21',
    'pack': '',
    'url': 'https://github.com/comfyanonymous/ComfyUI',
    'search': 'update ComfyUI — Qwen-Image 2.1 nodes ship in the core',
}

# Where each asset belongs inside a ComfyUI install, for the "place it here"
# message. Display paths only — the real lookup goes through comfy_model_paths.
QWEN21_ASSETS = {
    'qwen21_model': {
        'kind': 'Qwen-Image 2.1 base model (safetensors)',
        'path': 'models/diffusion_models/',
        'source': 'https://huggingface.co/Comfy-Org/Qwen-Image-2.1 (or your quantizer of choice)',
    },
    'qwen21_text_encoder': {
        'kind': 'Qwen3-VL 8B text encoder',
        'path': 'models/text_encoders/',
        'source': 'https://huggingface.co/Comfy-Org/Qwen-Image-2.1',
    },
    'qwen21_vae': {
        'kind': 'Qwen Image 2.1 VAE (not the Qwen-Image 1 one)',
        'path': 'models/vae/',
        'source': 'https://huggingface.co/Comfy-Org/Qwen-Image-2.1',
    },
}
QWEN21_REQUIRED = tuple(QWEN21_ASSETS)


class Qwen21ModelsMissing(Exception):
    """A Qwen-Image 2.1 asset (base / text encoder / VAE) is not on disk and/or
    the core node class is absent from the target ComfyUI. Raised BEFORE any
    row or job is created, so the route answers ONE actionable 409.
    `.missing` = asset keys (subset of QWEN21_REQUIRED); `.missing_nodes` =
    class_type strings the target ComfyUI does not expose."""

    def __init__(self, missing, missing_nodes=None):
        self.missing = list(missing or [])
        self.missing_nodes = list(missing_nodes or [])
        super().__init__('Qwen-Image 2.1 assets missing: '
                         + ', '.join(self.missing + self.missing_nodes))


class Qwen21PinnedModelMissing(Exception):
    """A model file the user PINNED in Settings is not on disk. A pin that
    cannot be resolved is a STOP naming the file — never a silent fallback to
    automatic resolution (see krea_edit_helper.KreaPinnedModelMissing for the
    full story; the failure mode is shared)."""

    def __init__(self, gaps):
        self.gaps = list(gaps or [])
        super().__init__(
            'Qwen-Image 2.1: pinned model file not found — '
            + '; '.join(f"{g['key']} = {g['configured']!r}" for g in self.gaps)
            + '. Pick a file that exists, or clear the field to use auto-detection.')


# --- Resolution -------------------------------------------------------------
# Same discipline as klein/krea: canonical name first, then NARROW token
# matches across every search root. A WRONG model is worse than a missing one.

def _listings(comfy_type):
    out = []
    for folder in comfy_model_paths.search_roots(comfy_type):
        try:
            out.append((folder, sorted(n for n in os.listdir(folder)
                                       if n.lower().endswith(_MODEL_SUFFIXES))))
        except OSError:
            continue
    return out


def _find_model_file(comfy_type, canonical, tokens, *,
                     suffixes=_MODEL_SUFFIXES, reject=()):
    """Bare filename for a ComfyUI folder type: canonical name if present in
    ANY search root, else the best narrow-token match. None when nothing
    matches — never a blind first-file guess. `suffixes` narrows the loader
    format (the 2.1 UNET takes .safetensors only — see the module docstring);
    `reject` drops names that carry a family-foreign token."""
    listings = _listings(comfy_type)
    if any(canonical in names for _root, names in listings):
        return canonical
    matches = [n for _root, names in listings for n in names
               if n.lower().endswith(tuple(suffixes))
               and any(tok in n.lower() for tok in tokens)
               and not any(bad in n.lower() for bad in reject)]
    if not matches:
        return None
    lead = [n for n in matches if n.lower().startswith(tokens[0])]
    pool = lead or matches
    suffix = canonical.lower().rsplit('.', 1)[0]
    for tok in tokens:
        if suffix.startswith(tok):
            suffix = suffix[len(tok):].strip('_-')
            break
    if suffix:
        same_precision = [n for n in pool if suffix in n.lower()]
        if same_precision:
            return same_precision[0]
    return pool[0]


def _qwen21_unet_candidates():
    """(prefix, [model files]) for the 2.1 base across every diffusion-model
    search root — 'qwen'-named folders and flat roots alike, matching Krea's
    discipline (flat installs load as os.path.join('', name))."""
    return comfy_model_paths.scan_family_folders(
        comfy_model_paths.search_roots('diffusion_models'), ('qwen',),
        accept=_qwen21_base_compatible, suffixes=('.safetensors',))


# Files that mention 2.1's neighbours but are NOT a 2.1 base: the Layered
# variant is a different architecture (its own latent class in comfy core).
QWEN21_INCOMPATIBLE_TOKENS = ('layered',)


def _qwen21_base_compatible(name):
    low = (name or '').lower()
    return not any(tok in low for tok in QWEN21_INCOMPATIBLE_TOKENS)


# The 2.1 token set. Everything else qwen-shaped in the same folders — 2509,
# 2511, 2512, plain qwen_image, Layered — fails one of these tokens, which is
# what keeps the resolver from electing a sibling generation. UNETLoader wants
# .safetensors; the GGUF builds are a different loader class entirely.
_QWEN21_UNET_TOKENS = ('qwen_image_2.1', 'qwen-image-2.1', 'qwen_image21',
                       'qwen21', 'qwen_2.1')
QWEN21_CANONICAL_UNET = 'qwen_image_2.1_bf16.safetensors'


def elect_qwen21_base(candidates):
    """The 2.1 base to use as a default out of `candidates` — an ordered list
    of ComfyUI-relative names. None when nothing loadable survives. ONE
    ranking for every surface (Generate, Studio), same contract as
    krea_edit_helper.elect_krea_base: loadable safetensors only, then
    deterministic name order (the app's convrot/int8 exports and stock builds
    share one family; precision preference is a Settings pin's job, not a
    guess the resolver makes)."""
    ranked = [name for name in (candidates or [])
              if name and comfy_model_paths.is_loadable_model(name)
              and _qwen21_base_compatible(os.path.basename(str(name).replace('\\', '/')))]
    return ranked[0] if ranked else None


def resolve_qwen21_unet(selected=None):
    """ComfyUI-relative `unet_name` for the UNETLoader (with subfolder prefix),
    or None when no compatible 2.1 base is on disk. Preference: the explicit
    pick (`selected`, or `qwen21.base_model` — matched on basename), then the
    deterministic election. A pin that resolves nowhere is a hard None (see
    Qwen21PinnedModelMissing), never a fallback."""
    folders = _qwen21_unet_candidates()
    if not folders:
        return None
    pick = selected or cfg.get('qwen21.base_model') or ''
    bare_pick = os.path.basename(str(pick).replace('/', os.sep).replace('\\', os.sep))
    if bare_pick:
        for sub, names in folders:
            if bare_pick in names:
                return os.path.join(sub, bare_pick)
        logger.warning('qwen21.base_model %r not found under any qwen folder', pick)
        return None
    folders = [(sub, [n for n in names if n.lower().endswith('.safetensors')
                      and any(tok in n.lower() for tok in _QWEN21_UNET_TOKENS)])
               for sub, names in folders]
    folders = [(sub, names) for sub, names in folders if names]
    if not folders:
        return None
    return elect_qwen21_base([os.path.join(sub, n) for sub, names in folders
                              for n in names])


def resolve_qwen21_text_encoder():
    """`clip_name` for the CLIPLoader (type='qwen_image'). Only 8B files: the
    4B Krea uses produces the WRONG clip target (classic Qwen-Image), and the
    32B is MiniMax H3's encoder. Never a bare 'qwen' match."""
    configured = (cfg.get('qwen21.text_encoder') or '').strip()
    if configured:
        if _abs_under_roots('text_encoders', configured):
            return configured
        if _is_user_pin('qwen21.text_encoder'):
            logger.warning('qwen21.text_encoder %r is not on disk — refusing to '
                           'scan for a different one', configured)
            return None
    return _find_model_file(
        'text_encoders', 'qwen3vl_8b_fp8_scaled.safetensors',
        ('qwen3vl_8b', 'qwen3_vl_8b', 'qwen3-vl-8b'))


def resolve_qwen21_vae():
    """`vae_name` for the VAELoader — 2.1-named files only. The Qwen-Image 1
    VAE (Krea's `qwen_image_vae.safetensors`) carries a DIFFERENT latent
    format and decodes 2.1 latents as noise; it must never fall back to it."""
    configured = (cfg.get('qwen21.vae') or '').strip()
    if configured:
        if _abs_under_roots('vae', configured):
            return configured
        if _is_user_pin('qwen21.vae'):
            logger.warning('qwen21.vae %r is not on disk — refusing to scan for '
                           'a different one', configured)
            return None
    return _find_model_file(
        'vae', 'qwen_image_2.1_vae_bf16.safetensors',
        ('qwen_image_2.1_vae', 'qwen-image-2.1-vae', 'qwen_image21_vae'))


def qwen21_missing_assets():
    """Which 2.1 assets are NOT on disk, as QWEN21_ASSETS keys. Disk-only,
    network-free — safe for the readiness probe."""
    missing = []
    if not resolve_qwen21_unet():
        missing.append('qwen21_model')
    if not resolve_qwen21_text_encoder():
        missing.append('qwen21_text_encoder')
    if not resolve_qwen21_vae():
        missing.append('qwen21_vae')
    return missing


# --- Present-but-INVALID assets ---------------------------------------------
# Same mechanism (and same reason) as klein/krea: an interrupted or proxied
# download saves an HTML page or a half file as .safetensors — present, passes
# the missing check, then dies at generate time. Header-only + cached.
QWEN21_MIN_BYTES = {
    'qwen21_model': 1024 ** 3,               # 1 GB   (real ≈ 10-20 GB)
    'qwen21_text_encoder': 256 * 1024 ** 2,  # 256 MB (real ≈ 8-10 GB at 8B)
    'qwen21_vae': 8 * 1024 ** 2,             # 8 MB   (real ≈ 250 MB)
}


def _abs_under_roots(comfy_type, rel_name):
    if not rel_name:
        return None
    for root in comfy_model_paths.search_roots(comfy_type):
        cand = os.path.join(root, rel_name)
        if os.path.exists(cand):
            return cand
    return None


def _qwen21_asset_paths():
    paths = {}
    for key, comfy_type, rel in (
            ('qwen21_model', 'diffusion_models', resolve_qwen21_unet()),
            ('qwen21_text_encoder', 'text_encoders', resolve_qwen21_text_encoder()),
            ('qwen21_vae', 'vae', resolve_qwen21_vae())):
        p = _abs_under_roots(comfy_type, rel)
        if p:
            paths[key] = p
    return paths


def qwen21_invalid_assets():
    """2.1 assets on disk under the resolved name that are NOT loadable
    weights — the state BETWEEN 'missing' and 'ready'. Same shape
    klein_invalid_assets publishes."""
    from . import model_integrity
    out = []
    for asset, path in _qwen21_asset_paths().items():
        res = model_integrity.validate_model_file(path, min_bytes=QWEN21_MIN_BYTES.get(asset))
        if res['ok']:
            continue
        out.append({'asset': asset, 'filename': res['filename'],
                    'verdict': res['verdict'], 'blocking': res['blocking'],
                    'reason': res['reason']})
    return out


# --- Core-node preflight ---------------------------------------------------
# /object_info probe, success-only TTL, fail-OPEN on an unreachable ComfyUI —
# identical contract to krea_missing_nodes (a transient probe failure must
# never block a generation that ComfyUI itself could still serve).
_NODES_OK_TTL_S = 300
_nodes_ok_until = 0.0


def qwen21_missing_nodes():
    """[class_type] of the 2.1 core nodes this ComfyUI does not expose — [] when
    present OR when /object_info is unreachable. An install missing the class
    needs a ComfyUI UPDATE, not a pack install."""
    global _nodes_ok_until
    if time.time() < _nodes_ok_until:
        return []
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = sorted(c for c in QWEN21_NODE_CLASSES if c not in available)
    if not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def clear_nodes_cache():
    global _nodes_ok_until
    _nodes_ok_until = 0.0


def qwen21_node_hints(nodes):
    """[{class_type, pack, url, search}] — the StudioPreflightBanner shape."""
    return [dict(QWEN21_NODE_HINT) for _ in (nodes or [])]


def missing_file_entries(missing):
    """[{path, kind, source}] — the Studio `files` shape."""
    out = []
    for key in missing or []:
        meta = QWEN21_ASSETS.get(key)
        if meta:
            out.append({'path': meta['path'], 'kind': meta['kind'],
                        'source': meta['source']})
    return out


def _is_user_pin(key: str) -> bool:
    """Is this config value something the USER chose, not the shipped default?
    (krea_edit_helper._is_user_pin verbatim — the distinction between a hard
    stop and a renamed-download recovery applies to every engine.)"""
    raw_value = str(cfg.get(key) or '').strip()
    if not raw_value:
        return False
    section, _, field = key.partition('.')
    shipped = str((cfg.DEFAULTS.get(section) or {}).get(field) or '').strip()
    return raw_value != shipped


def qwen21_pin_gaps():
    """``[{'slot', 'key', 'configured'}]`` for every pinned 2.1 file that is
    not on disk. One function, so Settings, the engine card and the refusal
    can never disagree."""
    gaps = []
    for slot, key, resolve in (
            ('base_model', 'qwen21.base_model', resolve_qwen21_unet),
            ('text_encoder', 'qwen21.text_encoder', resolve_qwen21_text_encoder),
            ('vae', 'qwen21.vae', resolve_qwen21_vae)):
        raw = (cfg.get(key) or '').strip()
        if raw and _is_user_pin(key) and resolve() is None:
            gaps.append({'slot': slot, 'key': key, 'configured': raw})
    return gaps


def preflight():
    """Raise Qwen21ModelsMissing when the engine cannot run. No auto-download:
    a named gap, never a fake installer."""
    gaps = qwen21_pin_gaps()
    if gaps:
        raise Qwen21PinnedModelMissing(gaps)
    missing = qwen21_missing_assets()
    nodes = qwen21_missing_nodes()
    if missing or nodes:
        raise Qwen21ModelsMissing(missing, nodes)


# --- Settings ----------------------------------------------------------------

def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def ref_resolution():
    """THE likeness <-> speed dial, in pixels — TextEncodeQwenImage21's
    `resolution` input. Every reference is resized into this square budget
    (aspect preserved), which also sets the OUTPUT size. Snapped to /32 like
    the node's own rounding; capped by max_output_mp so a big dial cannot
    quietly exceed the VRAM contract."""
    v = _clamp(cfg.get('qwen21.ref_resolution'), 256, 4096, 1024.0)
    cap = _max_output_resolution()
    return int(min(v, cap) // 32 * 32)


def _max_output_resolution():
    """Pixel side at which a square render equals the max_output_mp budget."""
    mp = _clamp(cfg.get('qwen21.max_output_mp'), 0.1, 16.0, MAX_OUTPUT_MP)
    return int((mp * 1_000_000) ** 0.5) // 32 * 32


def _steps():
    return int(_clamp(cfg.get('qwen21.steps'), 4, 60, 30.0))


def _negative_prompt():
    return str(cfg.get('qwen21.negative_prompt') or '')


# cfg is pinned at 4.0 in code (the measured working profile for 2.1) and is
# deliberately NOT a setting — same philosophy as Krea's pinned cfg 1.0, this
# model just wants the guidance Krea was distilled away from.
QWEN21_CFG = 4.0
QWEN21_SAMPLER = 'euler'
QWEN21_SCHEDULER = 'simple'

MAX_OUTPUT_MP = 2.0

# Optional always-on generation LoRAs — same shape, same caps as the Krea lane
# (krea_edit_helper keeps the numbers; the two lanes are deliberate copies).
MAX_GENERATION_LORAS = 8
MAX_GENERATION_LORA_PRESETS = 12
LORA_STRENGTH_MAX = 20.0
DEFAULT_ROW_STRENGTH = 1.0

CHARACTER_LORA_SLOTS = 5   # mirrors the Settings UI's fixed 5 rows


def configured_generation_lora_presets():
    """Sanitized `qwen21.generation_lora_presets` — THE single source of truth
    for which files may chain (a request only NAMES a preset)."""
    raw = cfg.get('qwen21.generation_lora_presets')
    out, seen = [], set()
    for p in (raw if isinstance(raw, list) else []):
        if not isinstance(p, dict):
            continue
        name = p.get('name')
        name = name.strip() if isinstance(name, str) else ''
        if not name or name in seen:
            continue
        rows = []
        for e in (p.get('loras') if isinstance(p.get('loras'), list) else []):
            if not isinstance(e, dict):
                continue
            f = e.get('file')
            f = f.strip() if isinstance(f, str) else ''
            if not f:
                continue
            s = e.get('strength')
            s = float(s) if isinstance(s, (int, float)) else DEFAULT_ROW_STRENGTH
            rows.append({'file': f, 'strength': max(0.0, min(LORA_STRENGTH_MAX, s))})
            if len(rows) >= MAX_GENERATION_LORAS:
                break
        seen.add(name)
        out.append({'name': name, 'loras': rows})
        if len(out) >= MAX_GENERATION_LORA_PRESETS:
            break
    return out


def resolve_generation_lora_preset(preset_name):
    """Ordered [{file, strength}] of the named preset; fail-closed to [] (a
    stale UI degrades to 'no extra LoRAs', never blocks a run)."""
    name = preset_name.strip() if isinstance(preset_name, str) else ''
    if not name:
        return []
    for p in configured_generation_lora_presets():
        if p['name'] == name:
            return [dict(e) for e in p['loras']]
    logger.warning('qwen21 generation-LoRA preset %r is not configured — '
                   'running with no extra LoRAs', name)
    return []


def _lora_abs(rel_name):
    if not rel_name:
        return None
    for root in comfy_model_paths.search_roots('loras'):
        cand = os.path.join(root, rel_name)
        if os.path.exists(cand):
            return cand
    return None


def _existing_generation_lora_rows(rows):
    """The subset of `rows` that can chain, capped and per-row degraded — same
    contract as krea's (a dataset run must not die over an optional LoRA; a
    dropped row says why in the log)."""
    out = []
    for entry in (rows or []):
        if len(out) >= MAX_GENERATION_LORAS:
            break
        if not isinstance(entry, dict):
            continue
        name = (entry.get('file') or '')
        name = name.replace('/', os.sep) if isinstance(name, str) else ''
        if not name:
            continue
        try:
            strength = max(0.0, min(LORA_STRENGTH_MAX, float(entry.get('strength'))))
        except (TypeError, ValueError):
            strength = 0.0
        if strength <= 0:
            logger.info('qwen21 generation LoRA %r at strength 0 — skipped (row off)',
                        name)
            continue
        if not _lora_abs(name):
            logger.warning('qwen21 generation LoRA %r not found under any loras '
                           'root — skipped', name)
            continue
        out.append({'file': name, 'strength': strength})
    return out


def _character_loras() -> list:
    """Up to CHARACTER_LORA_SLOTS (file, strength) pairs, user-configured order,
    blank rows dropped — each names ONE file the user trained; no substitution."""
    raw = cfg.get('qwen21.character_loras')
    if not isinstance(raw, list):
        return []
    out = []
    for row in raw[:CHARACTER_LORA_SLOTS]:
        if not isinstance(row, dict):
            continue
        f = row.get('file')
        f = f.strip() if isinstance(f, str) else ''
        if not f:
            continue
        # ComfyUI enumerates loras with the OS-native separator; same
        # normalization every other engine's loader chain applies.
        f = f.replace('/', os.sep)
        out.append((f, _clamp(row.get('strength'), 0.0, 1.5, 0.8)))
    return out


# --- Graph -------------------------------------------------------------------
# How many reference slots the graph wires beyond the primary. The node takes
# up to 16; the edit dialog's contract (LOCAL_EDIT_REF_SUPPORT 'modal_one') is
# the primary + ONE compositional second subject, exactly like Krea's `_b`.
MAX_REFERENCE_SLOTS = 2


def build_workflow(source_image, prompt, *, unet, clip, vae,
                   seed, steps=None, resolution=None, negative_prompt=None,
                   character_loras=None, generation_loras=None,
                   extra_source_image=None, filename_prefix='qwen21_edit'):
    """The ComfyUI API-format graph. Pure function of its arguments — no
    config read, no disk access — so a test can assert the exact wiring
    without a ComfyUI, and every loader value is one a resolver produced.

    The output latent comes from the NODE (first reference's size at
    `resolution`), so unlike Krea's builder there is no width/height here:
    geometry is the reference's own, scaled by the resolution dial.

    `character_loras`: OPTIONAL (name, strength) pairs chained after the base;
    `generation_loras`: ordered [{file, strength}] rows chained AFTER those —
    both arrive pre-checked from enqueue_qwen21_edit, so this stays pure.
    `extra_source_image`: OPTIONAL ComfyUI input filename of a second
    reference (a DIFFERENT subject — same contract as Krea's `_b` slot)."""
    steps = 30 if steps is None else max(4, int(steps))
    resolution = 1024 if resolution is None else int(resolution)
    negative_prompt = '' if negative_prompt is None else str(negative_prompt)
    character_loras = [(str(n).strip(), float(s)) for n, s in (character_loras or [])
                       if str(n).strip()]
    g = {
        '1': {'class_type': 'UNETLoader',
              'inputs': {'unet_name': unet, 'weight_dtype': 'default'},
              '_meta': {'title': 'Qwen-Image 2.1 base model'}},
        '2': {'class_type': 'CLIPLoader',
              'inputs': {'clip_name': clip, 'type': 'qwen_image', 'device': 'default'},
              '_meta': {'title': 'Qwen3-VL 8B text encoder'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': vae}},
        '5': {'class_type': 'LoadImage', 'inputs': {'image': source_image}},
    }
    model_out = ['1', 0]
    for i, (name, strength) in enumerate(character_loras):
        node_id = f'4c{i}'
        g[node_id] = {'class_type': 'LoraLoaderModelOnly',
                      'inputs': {'lora_name': name, 'strength_model': strength,
                                 'model': model_out},
                      '_meta': {'title': f'Character LoRA {i + 1} (extra consistency)'}}
        model_out = [node_id, 0]
    for i, entry in enumerate((generation_loras or [])[:MAX_GENERATION_LORAS], start=1):
        node_id = f'gen_lora_{i}'
        g[node_id] = {'class_type': 'LoraLoaderModelOnly',
                      'inputs': {'lora_name': entry['file'],
                                 'strength_model': float(entry['strength']),
                                 'model': model_out},
                      '_meta': {'title': f"Generation LoRA {i}: "
                                         f"{os.path.basename(entry['file'])}"}}
        model_out = [node_id, 0]
    # The encode node owns the whole edit side: pos + neg + output latent.
    # Autogrow references go as DOTTED API keys — `images.image_1` — measured
    # against this exact ComfyUI build: the flat `image_1` form reaches execute
    # as a stray kwarg (TypeError), and a nested `images: {…}` dict is silently
    # dropped (input never validated → empty refs → PURE TEXT-TO-IMAGE, which
    # looked like the engine ignoring the reference photo entirely). The dotted
    # form is the only one whose dynamic-path expansion actually nests the
    # resolved tensor into the node's `images` dict.
    encode_inputs = {'clip': ['2', 0], 'prompt': prompt,
                     'negative_prompt': negative_prompt, 'vae': ['3', 0],
                     'resolution': resolution,
                     'images.image_1': ['5', 0]}
    if extra_source_image:
        g['ref_b'] = {'class_type': 'LoadImage',
                      'inputs': {'image': extra_source_image},
                      '_meta': {'title': 'Second reference (a different subject / scene)'}}
        encode_inputs['images.image_2'] = ['ref_b', 0]
    g['6'] = {'class_type': 'TextEncodeQwenImage21',
              'inputs': encode_inputs,
              '_meta': {'title': 'Edit conditioning (positive / negative / latent)'}}
    g.update({
        '7': {'class_type': 'KSampler',
              'inputs': {'seed': seed, 'steps': steps, 'cfg': QWEN21_CFG,
                         'sampler_name': QWEN21_SAMPLER,
                         'scheduler': QWEN21_SCHEDULER, 'denoise': 1.0,
                         'model': model_out, 'positive': ['6', 0],
                         'negative': ['6', 1], 'latent_image': ['6', 2]}},
        '8': {'class_type': 'VAEDecode', 'inputs': {'samples': ['7', 0],
                                                    'vae': ['3', 0]}},
        '9': {'class_type': 'SaveImage',
              'inputs': {'filename_prefix': filename_prefix, 'images': ['8', 0]}},
    })
    return g


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def enqueue_qwen21_edit(user_id, source_filename, edit_prompt, source_path=None,
                        extra_metadata=None, qwen21_model=None,
                        generation_loras=None, extra_ref_paths=None):
    """Copy the reference into ComfyUI's input folder, build the 2.1 graph
    against what is ACTUALLY installed, and enqueue it. Returns the app job_id.

    Raises Qwen21ModelsMissing / Qwen21PinnedModelMissing when an asset or the
    core node is absent (checked BEFORE anything is copied or queued),
    ValueError on a missing source, RuntimeError when ComfyUI isn't configured.

    `extra_ref_paths`: up to MAX_REFERENCE_SLOTS-1 further COMPOSITIONAL
    subjects (a second person, a scene) — never another angle of the same
    face, same reasoning as Krea's `_b` (see its docstring; the duplicated-
    subject failure is a property of reference conditioning in general).
    Beyond the cap, or not on disk, is skipped with a log line."""
    if source_path is None:
        out_dir = cfg.comfyui_dir('output')
        if not out_dir:
            raise RuntimeError('ComfyUI is not configured')
        source_path = os.path.join(str(out_dir), source_filename)
    if not os.path.exists(source_path):
        raise ValueError(f'source image not found: {source_filename}')

    preflight()
    unet = resolve_qwen21_unet(qwen21_model)
    clip = resolve_qwen21_text_encoder()
    vae = resolve_qwen21_vae()

    comfy_input_dir = comfy_fs.ensure_input_usable(_comfy_input_dir())
    uid = uuid.uuid4().hex[:8]
    source_stem = os.path.splitext(os.path.basename(str(source_filename)))[0] or 'source'
    staged_source = comfy_fs.stage_input_image(
        source_path, f'qwen21_source_{uid}_{source_stem}.png', comfy_input_dir)
    comfy_input = os.path.basename(staged_source)

    extra_inputs = []
    for candidate in (extra_ref_paths or [])[:MAX_REFERENCE_SLOTS - 1]:
        if not candidate or not os.path.exists(candidate):
            logger.warning('qwen21: extra reference is not on disk — skipped: %s',
                           os.path.basename(str(candidate or '')))
            continue
        staged_extra = comfy_fs.stage_input_image(
            candidate, f'qwen21_ref_b_{uid}_{len(extra_inputs)}.png', comfy_input_dir)
        extra_inputs.append(os.path.basename(staged_extra))

    workflow = build_workflow(
        comfy_input, edit_prompt, unet=unet, clip=clip, vae=vae,
        seed=random.randint(0, 2 ** 64 - 1), steps=_steps(),
        resolution=ref_resolution(), negative_prompt=_negative_prompt(),
        character_loras=_character_loras(),
        generation_loras=_existing_generation_lora_rows(generation_loras),
        extra_source_image=extra_inputs[0] if extra_inputs else None,
        # UNIQUE prefix per job: SaveImage numbers from what is currently in
        # the output folder (the Klein bug — every tile the same file).
        filename_prefix=f'{user_id}_DatasetQwen21_{uid}')

    job_id = str(uuid.uuid4())
    # The literal stays a literal: test_dataset_job_harvest's AST walk discovers
    # engine stamps by string constant — a module constant here would make the
    # guard blind (the exact class of miss that stranded this lane's first run).
    meta = {'model_name': 'qwen21_dataset'}
    if extra_metadata:
        meta.update(extra_metadata)
    meta['staged_inputs'] = [comfy_input] + extra_inputs
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt=edit_prompt,
                          job_id=job_id, metadata=meta)
    return job_id
