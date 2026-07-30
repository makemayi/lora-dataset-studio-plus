"""Fail-closed ownership fence for LDS-managed local Ollama models.

Ollama is a separate process: a successful ``keep_alive: 0`` request is not a
proof that its CUDA allocation has already disappeared.  LDS therefore records
every local model it asks to run, then verifies ``/api/ps`` is empty before a
ComfyUI workflow may load its own models.  Remote Ollama endpoints do not share
this machine's GPU and are deliberately never probed or unloaded here.
"""
from __future__ import annotations

import errno
import ipaddress
import logging
import threading
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

_lock = threading.RLock()
# Normalised local endpoint -> exact models LDS has admitted in this process.
# ComfyUI probes before every new prompt: a manual Ollama load between cells
# must block rather than overlap with ComfyUI.
_owned_models: dict[str, set[str]] = {}
# Endpoint seen with a pre-existing / otherwise unowned model. It may be a
# user's own Ollama session, so it is never automatically unloaded.
_foreign_local_endpoints: set[str] = set()


def _exception_chain(exc):
    seen, pending = set(), [exc]
    while pending:
        current = pending.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ('__cause__', '__context__', 'reason'):
            nested = getattr(current, attr, None)
            if isinstance(nested, BaseException):
                pending.append(nested)
        pending.extend(arg for arg in getattr(current, 'args', ())
                       if isinstance(arg, BaseException))


def _connection_refused(exc) -> bool:
    return any(
        isinstance(item, ConnectionRefusedError)
        or (isinstance(item, OSError) and item.errno in (errno.ECONNREFUSED, 10061))
        for item in _exception_chain(exc)
    )


def _endpoint_scope(url) -> tuple[str, str | None]:
    """Return ``local``/``remote``/``unknown`` and a stable local endpoint."""
    if not isinstance(url, str) or not url.strip():
        return 'unknown', None
    try:
        parsed = urlsplit(url.strip())
    except Exception:
        return 'unknown', None
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        return 'unknown', None
    # The vision client preserves a configured URL path while this fence speaks
    # directly to /api/ps and /api/generate. Treat a proxy base-path as unknown
    # rather than proving the wrong daemon empty.
    if parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        return 'unknown', None
    host = parsed.hostname.lower().rstrip('.')
    if host == 'localhost':
        local = True
    else:
        try:
            # ``127.evil.example`` is a hostname, not loopback. Never use a
            # string prefix here: this fence must never probe or unload a
            # remote Ollama endpoint merely because its DNS name looks local.
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
    if not local:
        return 'remote', None
    try:
        port = parsed.port
    except ValueError:
        return 'unknown', None
    authority_host = f'[{host}]' if ':' in host else host
    authority = authority_host if port is None else f'{authority_host}:{port}'
    return 'local', f'{parsed.scheme}://{authority}'


def mark_before_generate(url, model) -> str:
    """Return ``local``, ``remote`` or ``blocked`` before an Ollama request.

    A shared Ollama daemon exposes no client owner id. LDS probes before its
    local admission: if another model is already resident, LDS must not
    alter its ``keep_alive`` or load a competing model. The request is blocked
    until that runner is empty (or a dedicated LDS endpoint is configured).
    """
    scope, endpoint = _endpoint_scope(url)
    if scope == 'unknown':
        return 'blocked'
    if scope != 'local':
        return scope
    if not isinstance(model, str) or not model.strip():
        return 'blocked'

    model = model.strip()
    with _lock:
        owned = set(_owned_models.get(endpoint, set()))

    state, loaded = _probe(endpoint)
    with _lock:
        # Re-read process ownership after the probe: concurrent LDS requests may
        # have claimed another model, but they never make an external model safe
        # to change or unload.
        owned = set(_owned_models.get(endpoint, set()))
        if state == 'unknown':
            return 'blocked'
        if state == 'empty' or loaded.issubset(owned):
            _owned_models.setdefault(endpoint, set()).add(model)
            _foreign_local_endpoints.discard(endpoint)
            return 'local'

        _foreign_local_endpoints.add(endpoint)
    logger.info('ollama GPU fence: preserving a pre-existing local model; LDS inference is blocked')
    return 'blocked'


def _probe(endpoint):
    """Return (``empty`` | ``models`` | ``unknown``, model names)."""
    try:
        response = requests.get(f'{endpoint}/api/ps', timeout=(3, 5), allow_redirects=False)
        status = getattr(response, 'status_code', None)
        if type(status) is not int or not 200 <= status < 300:
            return 'unknown', set()
        data = response.json()
        models = data.get('models') if isinstance(data, dict) else None
        if not isinstance(models, list):
            return 'unknown', set()
        names = set()
        for item in models:
            if not isinstance(item, dict):
                return 'unknown', set()
            name = item.get('name') or item.get('model')
            if not isinstance(name, str) or not name.strip():
                return 'unknown', set()
            names.add(name.strip())
        return ('empty' if not names else 'models'), names
    except (requests.RequestException, OSError) as exc:
        if _connection_refused(exc):
            return 'empty', set()
        return 'unknown', set()
    except Exception:
        return 'unknown', set()


def _post_unload(endpoint, model) -> bool:
    try:
        response = requests.post(f'{endpoint}/api/generate',
                                 json={'model': model, 'keep_alive': 0},
                                 timeout=(10, 30), allow_redirects=False)
        status = getattr(response, 'status_code', None)
        return type(status) is int and 200 <= status < 300
    except Exception as exc:
        logger.warning('ollama GPU fence: unload request failed for %s (%s)', model, exc)
        return False


def _release_endpoint(endpoint, expected_models) -> bool:
    """Unload only LDS-owned models and prove the runner is empty afterwards."""
    state, loaded = _probe(endpoint)
    if state == 'unknown':
        return False
    if state == 'empty':
        with _lock:
            _owned_models.pop(endpoint, None)
            _foreign_local_endpoints.discard(endpoint)
        return True

    with _lock:
        foreign = endpoint in _foreign_local_endpoints
    if foreign:
        # A later /api/ps empty response is the only way this endpoint becomes
        # safe again. Never infer that a same-named resident model is still ours.
        logger.warning('ollama GPU fence: preserving a pre-existing local model; ComfyUI stays blocked')
        return False

    unknown = loaded - expected_models
    if unknown:
        with _lock:
            _foreign_local_endpoints.add(endpoint)
        logger.warning('ollama GPU fence: local runner has an unowned model; ComfyUI stays blocked')
        return False
    for model in loaded:
        if not _post_unload(endpoint, model):
            return False
    state, remaining = _probe(endpoint)
    if state != 'empty' or remaining:
        return False
    with _lock:
        _owned_models.pop(endpoint, None)
        _foreign_local_endpoints.discard(endpoint)
    return True


def release_owned_models(*, ollama_url=None, model=None) -> bool | None:
    """Release all tracked local LDS models, or ``None`` for a remote endpoint.

    ``model`` is intentionally only a filter for ownership registration, not an
    instruction to unload an arbitrary user model. Bare ``unload_vision_model``
    calls release every tracked custom model from a caption batch.
    """
    scope, endpoint = _endpoint_scope(ollama_url) if ollama_url is not None else ('all', None)
    if scope == 'remote':
        return None
    with _lock:
        foreign = set(_foreign_local_endpoints)
        if scope == 'local':
            candidates = {endpoint: set(_owned_models.get(endpoint, set()))}
            if endpoint in foreign:
                candidates.setdefault(endpoint, set())
        elif scope == 'all':
            candidates = {key: set(value) for key, value in _owned_models.items()}
            for foreign_endpoint in foreign:
                candidates.setdefault(foreign_endpoint, set())
        else:
            return False
    if model is not None:
        if not isinstance(model, str) or not model.strip():
            return False
        wanted = model.strip()
        candidates = {key: values for key, values in candidates.items()
                      if wanted in values}

    # No local model has been admitted by LDS and no endpoint is known foreign:
    # do not poke or unload a user's manually loaded Ollama model.
    if not candidates:
        return True
    for local_endpoint, expected in candidates.items():
        if not _release_endpoint(local_endpoint, expected):
            return False
    return True


def _configured_local_endpoint() -> tuple[str, str | None]:
    """Resolve the current local runner without importing Vision request code."""
    try:
        from .. import config as cfg
        url = cfg.get('ollama.url') or 'http://127.0.0.1:11434'
    except Exception:
        return 'unknown', None
    return _endpoint_scope(url)


def ensure_released_for_comfy() -> bool:
    """Prove local Ollama is empty before every new ComfyUI prompt admission.

    The read-only ``/api/ps`` probe is intentionally repeated between cells: a
    user can start Ollama after the previous image. It never unloads ComfyUI
    models; it either verifies the runner is empty, releases a model LDS proved
    it owns, or blocks safely without touching a pre-existing user model.
    """
    with _lock:
        candidates = {key: set(value) for key, value in _owned_models.items()}
        for foreign_endpoint in _foreign_local_endpoints:
            candidates.setdefault(foreign_endpoint, set())

    scope, endpoint = _configured_local_endpoint()
    if scope == 'unknown':
        return False
    if scope == 'local':
        candidates.setdefault(endpoint, set())
    for local_endpoint, expected in candidates.items():
        if not _release_endpoint(local_endpoint, expected):
            return False
    return True


def reset_for_tests() -> None:
    """Forget process-local bookkeeping only; never call a local service."""
    with _lock:
        _owned_models.clear()
        _foreign_local_endpoints.clear()
