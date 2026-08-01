"""Shared Ollama vision captioning helper.

Single responsibility: run one robust vision pass on an image via Ollama and
return the caption text. Ordinary best-effort calls return "" on failure;
caption batches that request local auto-start receive a clear exception if the
server still cannot caption. Lifted from the parent project's seedance_routes extraction so both
the classify/caption passes of the face-dataset service can reuse it without
duplicating the Qwen3-VL quirks.
"""
from __future__ import annotations

import base64
import logging
import os as _os
import warnings

import requests

from .. import config as cfg
from . import image_encoding

logger = logging.getLogger(__name__)


def _ollama_url() -> str:
    # Total accessor: cfg.get() can return None (missing/corrupted config
    # section) and callers rstrip('/') the result unconditionally -- this
    # must never hand back None, or the never-raise contract below breaks.
    return cfg.get('ollama.url') or 'http://127.0.0.1:11434'


def _ollama_error_detail(exc: Exception) -> str:
    """Pull Ollama's own explanation out of a failed request. Ollama always
    answers a rejected /api/generate with a JSON body ({"error": "..."}) — e.g.
    a model with no image support, an architecture an older Ollama can't load, or
    a model that isn't pulled — but requests' HTTPError carries only the status
    line, so the reason is on the attached Response, not the exception string.

    Returns '' when there is no HTTP response at all (a connection/timeout error,
    where `exc.response` is None) so the caller can tell "Ollama rejected this"
    (has a body) from "Ollama was unreachable" (no response). Never raises."""
    resp = getattr(exc, 'response', None)
    if resp is None:
        return ''
    try:
        body = resp.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        msg = (body.get('error') or '').strip()
        if msg:
            return msg
    # Non-JSON body, or JSON without an 'error' key: fall back to the raw text so
    # the user still sees *something* actionable rather than a bare status code.
    try:
        text = (resp.text or '').strip()
    except Exception:
        text = ''
    return text[:300]


def _ollama_reject_message(exc: Exception) -> str:
    """User-facing one-liner when Ollama actively REJECTED a request (the server
    answered with a 4xx/5xx). '' when the failure wasn't an HTTP rejection (e.g. a
    connection error, which has no response/status) — the caller then keeps its
    unreachable/restart handling. Shape: 'Ollama rejected the request (HTTP 400):
    <exact error>' so the user can act without opening the log."""
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    if status is None:
        return ''
    detail = _ollama_error_detail(exc)
    return (f'Ollama rejected the request (HTTP {status}): {detail}' if detail
            else f'Ollama rejected the request (HTTP {status})')


# Ollama decodes the image bytes SERVER-SIDE before handing pixels to the model.
# Which decoder runs depends on the model's runtime: the llama.cpp runner (used by the
# GGUF vision models most users pull, e.g. huihui_ai/qwen3-vl-abliterated) loads images
# with stb_image, which handles JPEG/PNG/BMP/GIF/TGA but NOT WebP; Ollama's native Go
# engine only decodes the formats registered in image.Decode (gif/jpeg/png) unless the
# build blank-imports x/image/webp. Dataset masters may now stay JPEG/PNG/WebP/BMP and
# the Studio "Describe" pass can still produce WebP, so on those builds a WebP request
# fails with HTTP 400 "Failed to load image or audio file" (the exact llama.cpp reject) —
# issue #6, theotherbox122 on Ollama 0.32.0. Every decodable image is re-encoded at this
# single seam: it guarantees a JPEG all Ollama runners can read, bounds payload size, bakes
# EXIF orientation, and prevents camera EXIF/XMP/GPS from leaving the machine for the
# configured Ollama endpoint. The dataset master is only read, never rewritten.
_OLLAMA_MAX_SIDE = 1536


def _ensure_ollama_decodable(image_bytes: bytes) -> bytes | None:
    """Return image bytes Ollama's server-side decoder can definitely read.

    Every decodable format becomes a fresh, metadata-free JPEG (alpha flattened
    onto white, EXIF orientation baked, longest side bounded, quality 90). That
    makes remote captioning an explicit pixel-only disclosure even for source
    JPEG/PNG files. Fail closed: an undecodable or unsafe source returns ``None``
    so raw camera bytes can never be sent to the configured (possibly remote)
    Ollama endpoint."""
    try:
        import io

        from PIL import Image, ImageOps
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as source:
                width, height = source.size
                if (not isinstance(width, int) or not isinstance(height, int)
                        or width <= 0 or height <= 0
                        or width > image_encoding.INPUT_MAX_SIDE
                        or height > image_encoding.INPUT_MAX_SIDE
                        or width * height > image_encoding.INPUT_MAX_PIXELS):
                    raise ValueError(
                        f'image exceeds {image_encoding.INPUT_MAX_SIDE} px per side or '
                        f'{image_encoding.INPUT_MAX_PIXELS} pixels')
                source.load()
                im = ImageOps.exif_transpose(source)
        if im.mode in ('RGBA', 'LA', 'PA') or (im.mode == 'P' and 'transparency' in im.info):
            im = im.convert('RGBA')
            bg = Image.new('RGB', im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != 'RGB':
            im = im.convert('RGB')
        # A fresh image has no inherited `info`; Pillow otherwise may carry
        # source metadata in surprising format-specific save paths.
        clean = Image.new('RGB', im.size)
        clean.paste(im)
        im = clean
        if max(im.size) > _OLLAMA_MAX_SIDE:
            im.thumbnail((_OLLAMA_MAX_SIDE, _OLLAMA_MAX_SIDE), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, 'JPEG', quality=90)
        return out.getvalue()
    except Exception as e:  # noqa: BLE001 - never disclose raw bytes after a decode failure
        logger.warning('vision_ollama: refusing unsafe/unreadable image before Ollama (%s)', e)
        return None


def get_vision_model() -> str:
    """Resolve the Ollama vision model: env ``VISION_OLLAMA_MODEL`` > config
    ``ollama.vision_model`` (defaults to 'huihui_ai/qwen3-vl-abliterated:8b-instruct', see
    config.DEFAULTS — the ABLITERATED/uncensored Qwen3-VL, needed because the vanilla
    'qwen3-vl:8b' refuses to describe the NSFW concept datasets this app captions).
    CRITICAL: use the '-instruct' tag, NOT plain ':8b' (which resolves to the THINKING
    variant). The Thinking model reasons out loud in the response on caption/omission tasks
    ("So the shot type is... Wait, is that the shared element?") - benchmarked 2/8 usable vs
    8/8 for -instruct, and ~8x slower (13s vs 1.6s/image). The 30b-a3b-instruct ties -instruct
    on quality at 3x the VRAM, so -instruct is the default; upgrade via config without code."""
    env = (_os.environ.get('VISION_OLLAMA_MODEL') or '').strip()
    if env:
        return env
    return cfg.get('ollama.vision_model') or 'huihui_ai/qwen3-vl-abliterated:8b-instruct'


class LocalOllamaFenceError(RuntimeError):
    """A local inference lost its verified Vision GPU ownership."""


def _admit_local_ollama(url, model) -> None:
    """Record a local request and renew its outer Vision ownership when present."""
    from . import ollama_gpu_fence
    scope = ollama_gpu_fence.mark_before_generate(url, model)
    if scope == 'blocked':
        raise LocalOllamaFenceError(
            'A local Ollama model is already in use outside LDS. LDS will not change it; '
            'unload it first or configure a dedicated Ollama endpoint for LDS.')
    if scope != 'local':
        return
    from ..gpu_window import renew_gpu_exclusive_vision_window, vision_window_is_owned
    if vision_window_is_owned() and not renew_gpu_exclusive_vision_window():
        raise LocalOllamaFenceError(
            'The Vision GPU window expired before Ollama could start safely.')


def describe_image_ollama(image_bytes: bytes, prompt: str, *,
                          ollama_url: str | None = None,
                          model: str | None = None,
                          num_predict: int = 800,
                          num_ctx: int = 8192,
                          repeat_penalty: float = 1.1,
                          prefer_json: bool = False,
                          fmt: str | None = None,
                          keep_alive: str | int = 0,
                          auto_start_local: bool = False,
                          timeout: tuple[float, float] | float = (10, 120)) -> str:
    """Describe an image via Ollama vision. Returns the caption text, or "" on
    failure for ordinary best-effort calls. With ``auto_start_local=True``, a
    stopped local server is started once and a persistent failure raises a
    user-facing RuntimeError.

    `timeout` is a (connect, read) tuple by default: fail fast (10s) when Ollama
    is unreachable so a caller never hangs, but allow a long read (120s) for a
    cold model load + inference. Pass a single float to use it for both phases.

    Model variant matters: the default is now the `-instruct` tag (NON-thinking) — it
    answers directly, no reasoning trace, so a modest `num_predict` suffices. The
    `-thinking` / plain `:8b` variant instead ALWAYS emits a `thinking` trace (~900-1400
    tokens) that can't be skipped (think:false / `/no_think` are ignored by that
    checkpoint); with it, `num_predict` must be large enough to cover the thinking AND the
    answer (>=5000) or the response comes back empty with `done_reason=length`. We still
    fall back to the tail of `thinking` when `response` is empty (harmless with instruct —
    that field is empty — and correct for the thinking variant). `num_ctx` defaults to 8192
    so a long answer (plus any thinking trace) fits in context.

    `keep_alive` (défaut 0) : 0 décharge le modèle après CET appel (VRAM-safe,
    bon pour les appels isolés) ; un batch (caption/classify de N images) doit
    passer une durée (ex. '5m') pour garder le modèle chaud entre les images, PUIS
    appeler unload_vision_model() en fin de batch pour rendre la VRAM à ComfyUI.
    """
    prepared = _ensure_ollama_decodable(image_bytes)
    if prepared is None:
        # This deliberately happens before base64 / requests.post. A malformed
        # Bank master must behave like an unavailable vision result, not become a
        # raw-byte egress to a remote endpoint or provoke an auto-start retry.
        logger.warning('vision_ollama: describe skipped: image is unsafe or unreadable')
        return ''
    try:
        url = (ollama_url or _ollama_url()).rstrip('/')
        model_name = model or get_vision_model()
        # Every valid source was converted above to a bounded, metadata-free JPEG.
        # Without this, WebP dataset bytes hit HTTP 400 "Failed to load image or
        # audio file" on llama.cpp-backed runners (issue #6).
        b64 = base64.b64encode(prepared).decode('ascii')
        payload = {
            'model': model_name,
            'prompt': prompt,
            'images': [b64],
            'stream': False,
            'options': {'temperature': 0.3, 'num_ctx': int(num_ctx),
                        'num_predict': int(num_predict),
                        'repeat_penalty': float(repeat_penalty)},
            'keep_alive': keep_alive,
        }
        # `format='json'` constrains the response to valid JSON (Ollama grammar) —
        # stops the abliterated model from rambling prose instead of the object.
        if fmt:
            payload['format'] = fmt
        _admit_local_ollama(url, model_name)
        resp = requests.post(f'{url}/api/generate', json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        caption = (data.get('response') or '').strip()
        thinking = (data.get('thinking') or '').strip()
        # JSON callers: the structured object may land in EITHER `response` or
        # `thinking` (this checkpoint is non-deterministic about it). Return the
        # FULL field that contains an object so the caller's JSON extractor can
        # pull it out — the last-paragraph heuristic below would truncate it.
        if prefer_json:
            for cand in (caption, thinking):
                if '{' in cand:
                    return cand
            return caption or thinking
        if caption:
            return caption
        if thinking:
            done_reason = data.get('done_reason')
            logger.info('vision_ollama: response empty, falling back to thinking (done_reason=%s)',
                        done_reason)
            parts = [p.strip() for p in thinking.split('\n\n') if p.strip()]
            if not parts:
                return thinking
            # Graceful fallback: when the answer was truncated (done_reason='length')
            # the model never emitted a clean `response`, and the usable description is
            # the tail of a LONG thinking trace. Returning only the final paragraph
            # there collapses the scene to one sentence, so when the trace is genuinely
            # long we keep the last few paragraphs instead. A short trace's last
            # paragraph is already the whole answer, so we leave that case unchanged.
            if done_reason == 'length' and len(parts) > 3:
                return '\n\n'.join(parts[-3:])
            return parts[-1]
        return ''
    except LocalOllamaFenceError:
        raise
    except Exception as e:
        # If Ollama answered with a 4xx/5xx it told us WHY in the body — carry that
        # exact reason into both the log and the user-facing error. '' when the
        # failure had no HTTP response (connection/timeout), leaving the existing
        # unreachable/restart handling untouched.
        reject = _ollama_reject_message(e)
        if auto_start_local:
            # A rejection means the server DID answer, so starting a stopped server
            # can't fix it — surface Ollama's own reason now instead of retrying
            # into the same wall and reporting a generic "no caption after restart".
            if reject:
                logger.warning('vision_ollama: %s', reject)
                raise RuntimeError(reject) from e
            from . import ollama_control
            ready = ollama_control.ensure_captioning_ready(model=model_name)
            if not ready.get('ok'):
                raise RuntimeError(ready.get('error') or 'Ollama is unavailable') from e
            retried = describe_image_ollama(
                image_bytes, prompt, ollama_url=ollama_url, model=model,
                num_predict=num_predict, num_ctx=num_ctx,
                repeat_penalty=repeat_penalty, prefer_json=prefer_json, fmt=fmt,
                keep_alive=keep_alive, auto_start_local=False, timeout=timeout)
            if not retried:
                raise RuntimeError(
                    'Ollama did not return a caption after restart — check the configured '
                    'vision model and the application log.') from e
            return retried
        # Best-effort call: contract is to return "" — but still log the concrete
        # reason (previously only the opaque status code reached the log).
        logger.warning('vision_ollama: describe skipped: %s', reject or e)
        return ''


def generate_text_ollama(prompt: str, *,
                         ollama_url: str | None = None,
                         model: str | None = None,
                         num_predict: int = 400,
                         num_ctx: int = 4096,
                         repeat_penalty: float = 1.1,
                         keep_alive: str | int = 0,
                         timeout: tuple[float, float] | float = (10, 120)) -> str:
    """Text-only generation via the SAME Ollama model as the vision seam (no image
    attached). Used to derive a SHORT caption from an already-stored long one — a pure
    text transform, so no GPU-heavy vision decode and no reason to pull in a second model.
    Reusing the abliterated Qwen3-VL matters: a vanilla text model would refuse to
    shorten the NSFW captions this app produces. Returns the text, or "" best-effort on
    any failure (the caller degrades to keeping the long caption). Same response/thinking
    extraction as describe_image_ollama so the -instruct answer is read correctly."""
    try:
        url = (ollama_url or _ollama_url()).rstrip('/')
        model_name = model or get_vision_model()
        payload = {
            'model': model_name,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.3, 'num_ctx': int(num_ctx),
                        'num_predict': int(num_predict),
                        'repeat_penalty': float(repeat_penalty)},
            'keep_alive': keep_alive,
        }
        _admit_local_ollama(url, model_name)
        resp = requests.post(f'{url}/api/generate', json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        caption = (data.get('response') or '').strip()
        if caption:
            return caption
        thinking = (data.get('thinking') or '').strip()
        if thinking:
            parts = [p.strip() for p in thinking.split('\n\n') if p.strip()]
            return parts[-1] if parts else thinking
        return ''
    except Exception as e:
        logger.warning('vision_ollama: text generate skipped: %s', _ollama_reject_message(e) or e)
        return ''


def unload_vision_model(*, ollama_url: str | None = None, model: str | None = None) -> bool:
    """Release all LDS-owned local Vision models and verify their absence.

    For a configured remote Ollama endpoint, no local GPU is involved, so this
    keeps the old targeted best-effort semantics but still requires a strict
    successful HTTP acknowledgement. Local ownership is delegated to the fence:
    a bare call releases every tracked custom model, not just today's default.
    """
    try:
        url = ollama_url or _ollama_url()
        if not isinstance(url, str) or not url.strip():
            raise ValueError('invalid Ollama URL')
        url = url.rstrip('/')
    except Exception as exc:
        logger.warning('vision_ollama: unload url/model resolution failed: %s', exc)
        return False

    from . import ollama_gpu_fence
    # With no explicit endpoint a batch's finally must release every local LDS
    # model it used, including a custom model that differs from current Settings.
    # A RESOLVED remote endpoint must still be passed to the fence so it returns
    # None and lets the targeted remote keep_alive=0 POST below run. For a resolved
    # local endpoint, keep the bare-call "all LDS-owned local models" behavior.
    scope, _ = ollama_gpu_fence._endpoint_scope(url)
    fence_url = url if ollama_url is not None or scope == 'remote' else None
    released = ollama_gpu_fence.release_owned_models(
        ollama_url=fence_url,
        model=model)
    if released is not None:
        if released:
            from .vision_keepalive import forget_lease
            forget_lease()
        return released

    # Remote Ollama: it cannot contend with this machine's ComfyUI GPU. Keep the
    # request narrowly targeted and never use the local residency registry.
    payload = {'model': model or get_vision_model(), 'keep_alive': 0}
    for attempt in (1, 2):
        try:
            response = requests.post(f'{url}/api/generate', json=payload, timeout=(10, 30))
            status = getattr(response, 'status_code', None)
            if type(status) is int and 200 <= status < 300:
                from .vision_keepalive import forget_lease
                forget_lease()
                return True
            logger.warning('vision_ollama: remote unload attempt %d returned invalid/non-2xx status %r',
                           attempt, status)
        except Exception as exc:
            logger.warning('vision_ollama: remote unload attempt %d failed: %s', attempt, exc)
    return False
