"""Qwen Image (Alibaba DashScope) variation generator for the face Dataset Maker.

Sends reference photo(s) + variation prompt to Qwen Image model via DashScope API
and returns generated image bytes. No GPU, no ComfyUI — runs fully off-device.
SFW capable (no explicit policy like Gemini/OpenAI but handles general content).

CHOOSING THE MODEL
------------------
`engines.qwen_model` is free text (Settings ▸ Image engines). Resolution order, read
at CALL time so a change in Settings applies without restart:

    engines.qwen_model  >  QWEN_MODEL (env)  >  DEFAULT_MODEL

Use the ALIAS name (no date suffix), e.g. 'qwen-image-2.0-pro' — that is what
DashScope's own request examples show in the model field. A dated snapshot name
(e.g. 'qwen-image-2.0-pro-2026-06-25') is a SPECIFIC pinned version, not the
generally-documented form, and will 400 with "model not exist" once DashScope
retires that snapshot. See AVAILABLE_MODELS for the current catalog. Models are
NOT region-restricted — the same names work on every DashScope endpoint.

REGIONAL CONFIGURATION
----------------------
DashScope endpoints AND API keys are region-locked to each other — a key issued
for 华北2 (Beijing/cn) only works against the cn endpoint; using it against sg/us
(or vice versa) fails authentication. Configure the region to match where your
key was issued:

    backends.qwen_region: 'cn' | 'sg' | 'us' (default 'sg')

FAILING LOUDLY
--------------
Failures raise a NAMED cause (see engine_errors) instead of returning None. Model/key
failures that would repeat on every row are FATAL and stop the batch."""

from __future__ import annotations
import base64
import json
import logging
import os

import requests

from .. import config as cfg
from .engine_errors import EngineError, EngineFatal

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'qwen-image-2.0-pro'
_ENV_VAR = 'QWEN_MODEL'

# Regional endpoints. DashScope API keys are region-specific.
_ENDPOINTS = {
    'cn': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation',
    'sg': 'https://dashscope-sg.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation',
    'us': 'https://dashscope-us.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation',
}

# Models are NOT region-restricted — the same model catalog is available on every
# DashScope endpoint (cn/sg/us alike). What IS region-locked is the API key: a
# 华北2 (Beijing/cn) key only works against the cn endpoint, and a key from one
# region is rejected outright on another (see _ENDPOINTS). Use the ALIAS name
# (no date suffix) — DashScope resolves it to whichever dated snapshot is
# "recommended" today; a hardcoded dated snapshot goes stale and 400s with
# "model not exist" once DashScope retires it.
# From https://bailian.console.aliyun.com/cn-beijing#/api/?type=model&url=2975126
AVAILABLE_MODELS = [
    'qwen-image-2.0-pro',  # recommended — best text rendering / realism
    'qwen-image-2.0',      # recommended — faster, cheaper
    'qwen-image-max',
    'qwen-image-plus',
    'qwen-image',
]

_NO_KEY = ('no DashScope API key saved — add QWEN_API_KEY in '
           'Settings > Image engines')

# Fragments Qwen uses for model/request errors rather than per-prompt issues.
_MODEL_FAULT_HINTS = (
    'model_not_found', 'invalid model', 'not found', 'not supported',
    'unsupported', 'does not support', 'invalid parameter',
)


class QwenImageError(EngineError):
    """A named Qwen Image failure. User-facing text, never carries the key."""


class QwenImageFatal(QwenImageError, EngineFatal):
    """A failure that would repeat on every remaining row (missing key, unknown
    model, configuration error)."""


def _api_key():
    return cfg.secret('QWEN_API_KEY')


def _region():
    region = (cfg.get('engines.qwen_region') or '').strip().lower()
    return region if region in _ENDPOINTS else 'sg'


def get_available_models(region: str | None = None) -> list[str]:
    """Models DashScope offers. `region` is accepted for call-site compatibility
    but ignored — the model catalog is the same on every endpoint; only the API
    key is region-locked (see AVAILABLE_MODELS)."""
    return list(AVAILABLE_MODELS)


def _endpoint():
    region = _region()
    return _ENDPOINTS.get(region, _ENDPOINTS['sg'])


def get_image_model() -> str:
    """The image model to request: setting > env var > built-in default. Read
    fresh on every call so a change in Settings applies without restart."""
    return ((cfg.get('engines.qwen_model') or '').strip()
            or (os.environ.get(_ENV_VAR) or '').strip()
            or DEFAULT_MODEL)


def _error_message(resp) -> str:
    """Qwen's error explanation, trimmed. Standard envelope is
    {"code": "...", "message": "..."}."""
    try:
        body = resp.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        msg = str(body.get('message') or '').strip()
        if msg:
            return msg[:300]
    try:
        return (resp.text or '').strip()[:300]
    except Exception:
        return ''


def _blames_the_model(resp, detail: str) -> bool:
    """Does this error blame the model rather than the prompt/auth?"""
    try:
        err = (resp.json() or {}).get('code') or ''
    except Exception:
        err = ''
    if isinstance(err, str):
        err_low = err.lower()
        if any(h in err_low for h in _MODEL_FAULT_HINTS):
            return True
    low = detail.lower()
    return any(h in low for h in _MODEL_FAULT_HINTS)


def _raise_for_api_status(resp, *, model: str) -> None:
    """Turn non-200 from DashScope into actionable errors."""
    status = resp.status_code
    if status == 200:
        return
    detail = _error_message(resp)
    suffix = f': {detail}' if detail else ''
    region = _region()

    if status == 401:
        raise QwenImageFatal(f'DashScope rejected the API key (HTTP 401){suffix}')
    if status == 403:
        raise QwenImageFatal(
            f'DashScope access denied (HTTP 403){suffix} — '
            'check API key and region configuration')
    if status == 404:
        raise QwenImageFatal(
            f'DashScope model not found: "{model}" (HTTP 404){suffix}')
    if status == 429:
        raise QwenImageError(f'DashScope rate-limited (HTTP 429){suffix}')
    if status == 400:
        if 'model not exist' in detail.lower():
            raise QwenImageFatal(
                f'DashScope does not recognize model "{model}" (HTTP 400){suffix}. '
                f'Use one of: {", ".join(get_available_models())} — '
                f'change it in Settings > Image engines > Qwen Image model. '
                f'(Region "{region}" only affects which endpoint/key pair is used, '
                f'not which model names exist.)')
        if _blames_the_model(resp, detail):
            raise QwenImageFatal(
                f'DashScope rejected model "{model}" (HTTP 400){suffix} — '
                'check the model in Settings > Image engines')
        return
    raise QwenImageError(f'DashScope returned HTTP {status}{suffix}')


def size_for_aspect(aspect_ratio: str) -> str:
    """Map dataset aspect strings onto a DashScope `size` value. DashScope wants
    'width*height' (asterisk, e.g. '1024*1024') — NOT 'widthxheight'; sending an
    'x' separator is a plain HTTP 400 InvalidParameter ('Invalid size format').
    Supported range is 512*512 to 2048*2048 per model."""
    try:
        w, h = (int(x) for x in str(aspect_ratio).split(':', 1))
        if w > 0 and h > 0:
            if h > w:
                return '1024*1536'
            if w > h:
                return '1536*1024'
    except (ValueError, TypeError):
        pass
    return '1024*1024'


def _extract_image_ref(data) -> str | None:
    """Pull the raw `image` value out of a DashScope multimodal-generation
    response. The observed (and only confirmed-working) shape is the OpenAI-style
    chat envelope:

        {"output": {"choices": [{"message": {"content": [{"image": "https://…"}]}}]}}

    — NOT the flat `output.results[].image` shape some DashScope docs/older SDKs
    describe, which this endpoint does not actually return. Both are checked so a
    future/alternate response shape doesn't silently break this."""
    try:
        output = data.get('output') or {}
        choices = output.get('choices') or []
        if choices:
            content = ((choices[0].get('message') or {}).get('content') or [])
            for item in content:
                if isinstance(item, dict) and item.get('image'):
                    return item['image']
        results = output.get('results') or []
        if results and results[0].get('image'):
            return results[0]['image']
    except (TypeError, AttributeError, IndexError, KeyError):
        pass
    return None


def parse_image_response(data) -> bytes | None:
    """Extract the first generated image from a DashScope response as bytes.
    The `image` field DashScope returns is a temporary OSS download URL (not
    inline base64) — fetch it. Falls back to base64-decoding if a value ever
    arrives inline instead (defensive; not observed in practice)."""
    ref = _extract_image_ref(data)
    if not ref:
        return None
    if ref.startswith('http://') or ref.startswith('https://'):
        try:
            r = requests.get(ref, timeout=(10, 60))
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            logger.warning(f"qwen_image: could not download result image: {e}")
            return None
    try:
        return base64.b64decode(ref)
    except (TypeError, ValueError):
        return None


def generate_variation(ref_bytes: bytes | list[bytes], prompt: str, model: str | None = None,
                       aspect_ratio: str = '1:1') -> bytes | None:
    """Reference photo(s) + variation prompt -> generated image bytes, or None.
    Multimodal-generation endpoint with Base64-encoded reference images."""
    key = _api_key()
    if not key:
        raise QwenImageFatal(_NO_KEY)
    mdl = (model or '').strip() or get_image_model()
    refs = list(ref_bytes) if isinstance(ref_bytes, (list, tuple)) else [ref_bytes]
    refs = refs[:3]

    # Build message with references + prompt
    content = []
    for i, ref_b in enumerate(refs):
        b64_img = base64.b64encode(ref_b).decode('ascii')
        content.append({
            'type': 'image',
            'image': f'data:image/webp;base64,{b64_img}'
        })
    content.append({'type': 'text', 'text': prompt})

    body = {
        'model': mdl,
        'input': {
            'messages': [{'role': 'user', 'content': content}]
        },
        'parameters': {'size': size_for_aspect(aspect_ratio)}
    }

    try:
        r = requests.post(
            _endpoint(),
            headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
            json=body,
            timeout=(10, 120)
        )
    except requests.RequestException as e:
        raise QwenImageError(f'could not reach DashScope: {e}')
    if r.status_code != 200:
        logger.warning(f"qwen_image: HTTP {r.status_code}: {r.text[:300]}")
        _raise_for_api_status(r, model=mdl)
        return None
    img = parse_image_response(r.json())
    if img is None:
        logger.warning("qwen_image: no image in response")
    return img
