"""llama.cpp (llama-server) vision captioning — the SECOND local vision engine.

llama-server speaks an OpenAI-compatible API (``/v1/chat/completions``); it has
NO Ollama endpoints (no /api/generate, /api/tags, /api/pull), so this module
mirrors vision_ollama's contract over that API instead of reusing it. Signatures
match the Ollama trio on purpose: services/vision_ollama dispatches on
``vision.backend`` at its public seams, so callers never import this directly.

DELIBERATE v1 LIMITS (stated in Settings copy and docs/guide):
  - no lifecycle control: llama-server needs its model file at launch, so the
    user runs it themselves — no install / start / pull here (Setup's Ollama
    cards stay Ollama-only);
  - no GPU fence: the fence speaks Ollama's /api/ps + /api/generate, which
    llama-server does not implement, and there is no remote unload to enforce a
    release with — VRAM contention is the user's launch decision
    (# ponytail: route through ollama_gpu_fence when llama.cpp gains an unload API);
  - ``keep_alive`` / ``num_gpu`` / ``think`` are accepted and ignored: residency,
    GPU layers and reasoning format are llama-server LAUNCH flags, not per-request
    options (OpenAI-compatible API has no such fields).
"""
from __future__ import annotations

import logging
import os as _os

import requests

from .. import config as cfg

logger = logging.getLogger(__name__)


def _llamacpp_url() -> str:
    # Total accessor: cfg.get() can return None (missing/corrupted config
    # section); callers rstrip('/') unconditionally. Same guard as vision_ollama.
    return cfg.get('llamacpp.url') or 'http://127.0.0.1:8080'


def get_vision_model() -> str:
    """Configured llama.cpp vision model name. Blank (the default) means
    "whatever this server has loaded" — llama-server runs ONE model, chosen at
    launch, and classic builds ignore the request's model field outright."""
    env = (_os.environ.get('VISION_LLAMACPP_MODEL') or '').strip()
    if env:
        return env
    return cfg.get('llamacpp.vision_model') or ''


def _reject_message(exc: Exception, engine: str = 'llama.cpp') -> str:
    """'llama.cpp rejected the request (HTTP 400): <body>' when the server answered
    with a 4xx/5xx; '' on a connection/timeout error (no response). Mirrors
    vision_ollama._ollama_reject_message."""
    resp = getattr(exc, 'response', None)
    status = getattr(resp, 'status_code', None)
    if status is None:
        return ''
    detail = ''
    try:
        body = resp.json()
        if isinstance(body, dict):
            detail = (body.get('error') or body.get('message') or '').strip()
    except Exception:  # noqa: BLE001 - non-JSON body: fall through to raw text
        pass
    if not detail:
        try:
            detail = (resp.text or '').strip()[:300]
        except Exception:  # noqa: BLE001
            detail = ''
    return (f'{engine} rejected the request (HTTP {status}): {detail}' if detail
            else f'{engine} rejected the request (HTTP {status})')


def _extract_text(data: dict, *, prefer_json: bool = False) -> str:
    """The answer out of an OpenAI-shaped chat completion. llama.cpp builds with
    reasoning support (--reasoning-format) put the thinking trace in
    ``reasoning_content`` and the answer in ``content``; older or mis-set builds
    can leave content empty with everything in reasoning_content — the same
    non-determinism vision_ollama handles for Ollama's ``thinking`` field, so the
    same fallback shape applies: prefer a field holding an object when
    ``prefer_json`` (the caller's JSON extractor pulls it out intact), else
    content, else the reasoning trace."""
    choices = data.get('choices') or []
    msg = (choices[0].get('message') or {}) if choices else {}
    content = (msg.get('content') or '').strip()
    reasoning = (msg.get('reasoning_content') or '').strip()
    if prefer_json:
        for cand in (content, reasoning):
            if '{' in cand:
                return cand
        return content or reasoning
    return content or reasoning


def _chat(url: str, payload: dict, timeout) -> dict:
    resp = requests.post(f'{url}/v1/chat/completions', json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _unreachable_error(url: str) -> str:
    return (f'llama.cpp server unreachable at {url} — llama-server is started by '
            f'you, not by this app: launch it yourself with a vision-capable model '
            f'(and its mmproj), then check Settings › Local tools.')


def describe_image_llamacpp(image_bytes: bytes, prompt: str, *,
                            ollama_url: str | None = None,
                            model: str | None = None,
                            num_predict: int = 800,
                            num_ctx: int = 8192,
                            repeat_penalty: float = 1.1,
                            prefer_json: bool = False,
                            fmt: str | None = None,
                            keep_alive: str | int = 0,
                            num_gpu: int | None = None,
                            think: bool | None = None,
                            auto_start_local: bool = False,
                            timeout: tuple[float, float] | float = (10, 120)) -> str:
    """Describe an image via llama-server's OpenAI-compatible chat API. Same
    contract as describe_image_ollama: '' best-effort on failure, a
    user-facing RuntimeError on a persistent failure with ``auto_start_local``
    (which here can only ever VERIFY — there is nothing to start). The Ollama
    URL parameter is accepted for signature parity and IGNORED: the endpoint is
    ``llamacpp.url`` from config."""
    from .vision_ollama import _ensure_ollama_decodable
    prepared = _ensure_ollama_decodable(image_bytes)
    if prepared is None:
        logger.warning('vision_llamacpp: describe skipped: image is unsafe or unreadable')
        return ''
    import base64
    url = _llamacpp_url().rstrip('/')
    model_name = model or get_vision_model()
    b64 = base64.b64encode(prepared).decode('ascii')
    content = [
        {'type': 'text', 'text': prompt},
        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
    ]
    payload = {
        # Sent even when blank: classic single-model builds ignore it, router
        # builds need it to pick the slot.
        'model': model_name,
        'messages': [{'role': 'user', 'content': content}],
        'stream': False,
        'temperature': 0.3,
        'max_tokens': int(num_predict),
        'repeat_penalty': float(repeat_penalty),
    }
    if fmt:
        # 'json' (Ollama's value) → the OpenAI JSON mode llama.cpp compiles to a
        # grammar. Any other value is passed as-is and the server rejects it with
        # its own reason.
        payload['response_format'] = (
            {'type': 'json_object'} if fmt == 'json' else {'type': fmt})
    try:
        data = _chat(url, payload, timeout)
        return _extract_text(data, prefer_json=prefer_json)
    except Exception as e:  # noqa: BLE001
        reject = _reject_message(e)
        if auto_start_local:
            if reject:
                logger.warning('vision_llamacpp: %s', reject)
                raise RuntimeError(reject) from e
            raise RuntimeError(_unreachable_error(url)) from e
        logger.warning('vision_llamacpp: describe skipped: %s', reject or e)
        return ''


def generate_text_llamacpp(prompt: str, *,
                           ollama_url: str | None = None,
                           model: str | None = None,
                           num_predict: int = 400,
                           num_ctx: int = 4096,
                           repeat_penalty: float = 1.1,
                           keep_alive: str | int = 0,
                           strict: bool = False,
                           timeout: tuple[float, float] | float = (10, 120)) -> str:
    """Text-only generation via the same llama-server. Same contract as
    generate_text_ollama: '' best-effort, or RuntimeError with the concrete
    reason when ``strict`` (no fence exception can occur here — llama.cpp has
    no fence, see the module docstring)."""
    url = _llamacpp_url().rstrip('/')
    payload = {
        'model': model or get_vision_model(),
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'temperature': 0.3,
        'max_tokens': int(num_predict),
        'repeat_penalty': float(repeat_penalty),
    }
    try:
        return _extract_text(_chat(url, payload, timeout))
    except Exception as e:  # noqa: BLE001
        reason = _reject_message(e) or str(e)
        logger.warning('vision_llamacpp: text generate skipped: %s', reason)
        if strict:
            raise RuntimeError(reason) from e
        return ''


def unload_vision_model(*, ollama_url: str | None = None, model: str | None = None) -> bool:
    """No-op that reports success: llama-server has NO remote unload — residency
    is a launch-flag decision (how it was started is when it stops). Keeping the
    Ollama seam's return contract lets batch code that unloads in its finally
    stay engine-agnostic instead of growing a backend branch."""
    return True


def list_models() -> dict:
    """{ok, reachable, models:[...]} for the model picker, from GET /v1/models.
    llama.cpp lists only what is LOADED (usually one entry) — there is no library
    to pull from. Never raises; unreachable → reachable=False, empty list."""
    url = _llamacpp_url().rstrip('/')
    try:
        resp = requests.get(f'{url}/v1/models', timeout=3)
        if resp.status_code >= 400:
            return {'ok': False, 'reachable': False, 'models': []}
        ids = []
        for m in (resp.json().get('data') or []):
            if isinstance(m, dict):
                mid = (m.get('id') or '').strip()
                if mid and mid not in ids:
                    ids.append(mid)
        return {'ok': True, 'reachable': True, 'models': ids}
    except Exception:  # noqa: BLE001
        return {'ok': False, 'reachable': False, 'models': []}
