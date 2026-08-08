"""Engine-aware semantic embedding caches for Image Bank.

CLIP remains the historical semantic space produced by the Score pass.  Its
``score_cache.npz`` format is deliberately only *read* here and is never
rewritten.  SigLIP2 has a separate, provenance-bound cache whose contract is
created from :mod:`bank_semantic_models`, the single source of truth in the app.

This module contains no ML imports.  It is safe to use from request/capability
paths in the lightweight Flask interpreter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable
import zipfile

from .. import config as cfg
from . import bank_semantic_models as siglip2_models


logger = logging.getLogger(__name__)

CLIP_ENGINE = 'clip'
SIGLIP2_ENGINE = 'siglip2'
SEMANTIC_CACHE_VERSION = 1
SIGLIP2_CACHE_FILENAME = 'semantic_siglip2_cache.npz'
CLIP_CACHE_FILENAME = 'score_cache.npz'

CLIP_MODEL_KEY = 'clip-vit-l-14-openai'
CLIP_MODEL_LABEL = 'CLIP ViT-L/14'
SIGLIP2_MODEL_LABEL = 'SigLIP 2 Base'
CLIP_DIMENSION = 768

_ENGINE_ALIASES = {
    'clip': CLIP_ENGINE,
    'open-clip': CLIP_ENGINE,
    'open_clip': CLIP_ENGINE,
    'score': CLIP_ENGINE,
    CLIP_MODEL_KEY: CLIP_ENGINE,
    'siglip': SIGLIP2_ENGINE,
    'siglip-2': SIGLIP2_ENGINE,
    'siglip_2': SIGLIP2_ENGINE,
    'siglip2': SIGLIP2_ENGINE,
    siglip2_models.MODEL_KEY.lower(): SIGLIP2_ENGINE,
}

_SIGLIP2_KEYS = frozenset({
    'version', 'engine', 'model_id', 'revision', 'model_key', 'dimension',
    'paths', 'states', 'embs', 'sigs', 'hashes',
})


@dataclass(frozen=True)
class _ParsedCache:
    paths: tuple[str, ...]
    states: tuple[str, ...]
    embs: Any
    sigs: tuple[str, ...]
    hashes: Any
    legacy_hashes: bool = False


@dataclass(frozen=True)
class _CountsCache:
    paths: tuple[str, ...]
    states: tuple[str, ...]
    sigs: tuple[str, ...]
    hashes: Any
    legacy_hashes: bool = False


# One parsed NPZ at a time.  Arrays dominate memory (N x 768 float32), so an
# unbounded per-Bank cache would retain hundreds of MiB while switching Banks.
_memo_lock = threading.RLock()
_memo: tuple[tuple[Any, ...], _ParsedCache] | None = None
_VALIDATED_MEMO_TTL = 60.0
_validated_memo: tuple[
    tuple[Any, ...], float, dict[str, Any], dict, dict[str, str]
] | None = None
_counts_memo: tuple[tuple[Any, ...], float, dict] | None = None
_fingerprints: dict[str, dict[str, str]] = {
    CLIP_ENGINE: {},
    SIGLIP2_ENGINE: {},
}


def normalize_engine(engine: Any) -> str:
    """Return ``clip`` or ``siglip2``; reject an unknown embedding space.

    ``None``/blank maps to CLIP for existing Banks whose column predates the
    engine selector.
    """
    raw = str(engine or CLIP_ENGINE).strip().lower()
    normalized = _ENGINE_ALIASES.get(raw)
    if normalized is None:
        raise ValueError('semantic engine must be clip or siglip2')
    return normalized


def engine_model_key(engine: Any) -> str:
    engine = normalize_engine(engine)
    return (CLIP_MODEL_KEY if engine == CLIP_ENGINE
            else siglip2_models.MODEL_KEY)


def engine_model_label(engine: Any) -> str:
    engine = normalize_engine(engine)
    return CLIP_MODEL_LABEL if engine == CLIP_ENGINE else SIGLIP2_MODEL_LABEL


def engine_dimension(engine: Any) -> int:
    engine = normalize_engine(engine)
    return (CLIP_DIMENSION if engine == CLIP_ENGINE
            else int(siglip2_models.DIMENSION))


def semantic_contract(engine: Any = SIGLIP2_ENGINE) -> dict:
    """Immutable worker handshake assembled from the app's pinned model source.

    Only SigLIP2 has a standalone worker.  CLIP's image vectors continue to be
    produced by ``bank_score_infer.py`` and its text worker has its own historical
    model-pair contract.
    """
    engine = normalize_engine(engine)
    if engine != SIGLIP2_ENGINE:
        raise ValueError('the standalone semantic worker only supports siglip2')
    kwargs = siglip2_models.model_kwargs()
    return {
        'cache_version': SEMANTIC_CACHE_VERSION,
        'engine': engine,
        'model_id': kwargs['pretrained_model_name_or_path'],
        'revision': kwargs['revision'],
        'model_key': siglip2_models.MODEL_KEY,
        'dimension': int(siglip2_models.DIMENSION),
        'models_root': kwargs['cache_dir'],
        'local_files_only': True,
    }


def _bank_id(bank_or_id: Any) -> int:
    value = getattr(bank_or_id, 'id', bank_or_id)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('invalid bank id')
    return int(value)


def semantic_cache_path(bank_or_id: Any, engine: Any = CLIP_ENGINE) -> Path:
    """Cache path for one Bank/engine without creating or mutating it."""
    engine = normalize_engine(engine)
    filename = (CLIP_CACHE_FILENAME if engine == CLIP_ENGINE
                else SIGLIP2_CACHE_FILENAME)
    return cfg.banks_root() / str(_bank_id(bank_or_id)) / filename


def cache_generation(bank_or_id: Any, engine: Any = CLIP_ENGINE) -> tuple | None:
    """``(size, mtime_ns)`` fence for a semantic rebuild, or ``None``."""
    path = semantic_cache_path(bank_or_id, engine)
    try:
        st = path.stat()
        return st.st_size, st.st_mtime_ns
    except OSError:
        return None


def image_worker_payload(images: Iterable[Any], bank_or_id: Any, *,
                         engine: Any = SIGLIP2_ENGINE, device: str | None = None,
                         rescore: bool = False, rescan: bool = False) -> dict:
    """Build the one-shot SigLIP2 image-worker payload used by the service."""
    engine = normalize_engine(engine)
    contract = semantic_contract(engine)
    cache = semantic_cache_path(bank_or_id, engine)
    selected_device = str(
        device if device is not None else cfg.get('bank_semantic.device') or 'auto'
    ).strip().lower()
    if selected_device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('bank semantic device must be auto, cpu or cuda')
    return {
        **contract,
        'images': [str(path) for path in images],
        'cache': str(cache),
        'cancel_file': str(cache) + '.cancel',
        'device': selected_device,
        'rescore': bool(rescore or rescan),
    }


def text_worker_handshake(engine: Any = SIGLIP2_ENGINE) -> dict:
    """Warm SigLIP2 text-worker handshake, with exact image-space provenance."""
    return semantic_contract(engine)


def reset_memo() -> None:
    """Forget parsed cache arrays and path fingerprints (tests/deletion)."""
    global _memo, _validated_memo, _counts_memo
    with _memo_lock:
        _memo = None
        _validated_memo = None
        _counts_memo = None
        _fingerprints[CLIP_ENGINE] = {}
        _fingerprints[SIGLIP2_ENGINE] = {}


def _scalar(array: Any, key: str) -> Any:
    import numpy as np
    value = np.asarray(array)
    if value.shape != (1,):
        raise ValueError(f'{key} metadata must have shape (1,)')
    return value[0].item()


def _string_vector(value: Any, key: str, length: int | None = None) -> tuple[str, ...]:
    import numpy as np
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in ('U', 'S'):
        raise ValueError(f'{key} must be a string vector')
    if length is not None and len(array) != length:
        raise ValueError(f'{key} length mismatch')
    return tuple(str(item) for item in array)


def _parse_siglip2_npz(z: Any) -> _ParsedCache:
    import numpy as np
    if set(z.files) != _SIGLIP2_KEYS:
        raise ValueError('SigLIP2 cache keys do not match version 1')
    version = np.asarray(z['version'])
    dimension = np.asarray(z['dimension'])
    if (version.dtype != np.dtype('int32')
            or int(_scalar(version, 'version')) != SEMANTIC_CACHE_VERSION):
        raise ValueError('unsupported SigLIP2 cache version')
    if (dimension.dtype != np.dtype('int32')
            or int(_scalar(dimension, 'dimension')) != siglip2_models.DIMENSION):
        raise ValueError('SigLIP2 cache dimension mismatch')
    provenance = {
        'engine': SIGLIP2_ENGINE,
        'model_id': siglip2_models.MODEL_ID,
        'revision': siglip2_models.REVISION,
        'model_key': siglip2_models.MODEL_KEY,
    }
    for key, expected in provenance.items():
        if str(_scalar(z[key], key)) != expected:
            raise ValueError(f'SigLIP2 cache {key} mismatch')

    paths = _string_vector(z['paths'], 'paths')
    if len(set(paths)) != len(paths):
        raise ValueError('SigLIP2 cache contains duplicate paths')
    states = _string_vector(z['states'], 'states', len(paths))
    if any(state not in ('ok', 'error') for state in states):
        raise ValueError('SigLIP2 cache contains an invalid state')
    sigs = _string_vector(z['sigs'], 'sigs', len(paths))
    embs = np.asarray(z['embs'])
    hashes = np.asarray(z['hashes'])
    if embs.dtype != np.dtype('float32') or embs.shape != (
            len(paths), siglip2_models.DIMENSION):
        raise ValueError('SigLIP2 cache embedding shape or dtype mismatch')
    if hashes.dtype != np.dtype('uint8') or hashes.shape != (len(paths), 32):
        raise ValueError('SigLIP2 cache hash shape or dtype mismatch')
    if not np.isfinite(embs).all():
        raise ValueError('SigLIP2 cache contains non-finite embeddings')
    ok_indexes = [i for i, state in enumerate(states) if state == 'ok']
    if ok_indexes:
        ok_embs = embs[ok_indexes]
        norms = np.linalg.norm(ok_embs, axis=1)
        if (not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-4)
                or any(not sigs[i] for i in ok_indexes)
                or any(not hashes[i].any() for i in ok_indexes)):
            raise ValueError('SigLIP2 cache has unbound or unnormalised embeddings')
    return _ParsedCache(paths, states, embs, sigs, hashes)


def _parse_clip_npz(z: Any) -> _ParsedCache:
    """Parse the additive historical Score format without imposing provenance.

    Old Score caches have no engine/model metadata and early caches have no
    ``sigs``/``hashes``.  They remain readable here; ``image_bank_service`` is
    still the authoritative compatibility loader used by Bank operations.
    """
    import numpy as np
    required = {'paths', 'states', 'embs'}
    if not required.issubset(z.files):
        raise ValueError('CLIP score cache is missing embedding arrays')
    if 'engine' in z.files and str(_scalar(z['engine'], 'engine')) != CLIP_ENGINE:
        raise ValueError('score cache belongs to another engine')
    if ('dimension' in z.files
            and int(_scalar(z['dimension'], 'dimension')) != CLIP_DIMENSION):
        raise ValueError('CLIP score cache dimension mismatch')
    if ('model_key' in z.files
            and str(_scalar(z['model_key'], 'model_key')) != CLIP_MODEL_KEY):
        raise ValueError('CLIP score cache model mismatch')
    paths = _string_vector(z['paths'], 'paths')
    states = _string_vector(z['states'], 'states', len(paths))
    embs = np.asarray(z['embs'])
    if embs.ndim != 2 or embs.shape != (len(paths), CLIP_DIMENSION):
        raise ValueError('CLIP score cache embedding dimension mismatch')
    if embs.dtype.kind != 'f' or not np.isfinite(embs).all():
        raise ValueError('CLIP score cache embeddings are invalid')
    embs = np.asarray(embs, dtype='float32')
    sigs = (_string_vector(z['sigs'], 'sigs', len(paths))
            if 'sigs' in z.files else ('',) * len(paths))
    if 'hashes' in z.files:
        hashes = np.asarray(z['hashes'])
        if hashes.dtype != np.dtype('uint8') or hashes.shape != (len(paths), 32):
            raise ValueError('CLIP score cache hash shape or dtype mismatch')
        legacy_hashes = False
    else:
        hashes = np.zeros((len(paths), 32), dtype='uint8')
        legacy_hashes = True
    return _ParsedCache(paths, states, embs, sigs, hashes, legacy_hashes)


def _embedding_header(path: Path) -> tuple[tuple[int, ...], Any]:
    """Read only ``embs.npy``'s NPY header, never its matrix payload."""
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names.count('embs.npy') != 1:
            raise ValueError('semantic cache has no unique embedding array')
        info = archive.getinfo('embs.npy')
        with archive.open(info) as source:
            version = np.lib.format.read_magic(source)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
            else:
                raise ValueError('unsupported embedding NPY version')
            payload_offset = source.tell()
        if fortran or payload_offset + int(np.prod(shape)) * dtype.itemsize != info.file_size:
            raise ValueError('invalid embedding NPY layout')
    return tuple(int(part) for part in shape), np.dtype(dtype)


def _parse_counts_npz(path: Path, engine: str) -> _CountsCache:
    """Validate cache provenance/index arrays without materialising ``embs``."""
    import numpy as np
    with np.load(str(path), allow_pickle=False) as z:
        if engine == SIGLIP2_ENGINE:
            if set(z.files) != _SIGLIP2_KEYS:
                raise ValueError('SigLIP2 cache keys do not match version 1')
            version = np.asarray(z['version'])
            dimension = np.asarray(z['dimension'])
            if (version.dtype != np.dtype('int32')
                    or int(_scalar(version, 'version')) != SEMANTIC_CACHE_VERSION):
                raise ValueError('unsupported SigLIP2 cache version')
            if (dimension.dtype != np.dtype('int32')
                    or int(_scalar(dimension, 'dimension'))
                    != siglip2_models.DIMENSION):
                raise ValueError('SigLIP2 cache dimension mismatch')
            provenance = {
                'engine': SIGLIP2_ENGINE,
                'model_id': siglip2_models.MODEL_ID,
                'revision': siglip2_models.REVISION,
                'model_key': siglip2_models.MODEL_KEY,
            }
            for key, expected in provenance.items():
                if str(_scalar(z[key], key)) != expected:
                    raise ValueError(f'SigLIP2 cache {key} mismatch')
            paths = _string_vector(z['paths'], 'paths')
            if len(set(paths)) != len(paths):
                raise ValueError('SigLIP2 cache contains duplicate paths')
            states = _string_vector(z['states'], 'states', len(paths))
            if any(state not in ('ok', 'error') for state in states):
                raise ValueError('SigLIP2 cache contains an invalid state')
            sigs = _string_vector(z['sigs'], 'sigs', len(paths))
            hashes = np.asarray(z['hashes'])
            if hashes.dtype != np.dtype('uint8') or hashes.shape != (len(paths), 32):
                raise ValueError('SigLIP2 cache hash shape or dtype mismatch')
            legacy_hashes = False
            dimension_expected = siglip2_models.DIMENSION
        else:
            required = {'paths', 'states', 'embs'}
            if not required.issubset(z.files):
                raise ValueError('CLIP score cache is missing embedding arrays')
            if ('engine' in z.files
                    and str(_scalar(z['engine'], 'engine')) != CLIP_ENGINE):
                raise ValueError('score cache belongs to another engine')
            if ('dimension' in z.files
                    and int(_scalar(z['dimension'], 'dimension')) != CLIP_DIMENSION):
                raise ValueError('CLIP score cache dimension mismatch')
            if ('model_key' in z.files
                    and str(_scalar(z['model_key'], 'model_key')) != CLIP_MODEL_KEY):
                raise ValueError('CLIP score cache model mismatch')
            paths = _string_vector(z['paths'], 'paths')
            states = _string_vector(z['states'], 'states', len(paths))
            sigs = (_string_vector(z['sigs'], 'sigs', len(paths))
                    if 'sigs' in z.files else ('',) * len(paths))
            if 'hashes' in z.files:
                hashes = np.asarray(z['hashes'])
                if (hashes.dtype != np.dtype('uint8')
                        or hashes.shape != (len(paths), 32)):
                    raise ValueError('CLIP score cache hash shape or dtype mismatch')
                legacy_hashes = False
            else:
                hashes = np.zeros((len(paths), 32), dtype='uint8')
                legacy_hashes = True
            dimension_expected = CLIP_DIMENSION
    emb_shape, emb_dtype = _embedding_header(path)
    if (emb_shape != (len(paths), dimension_expected)
            or emb_dtype.kind != 'f'
            or engine == SIGLIP2_ENGINE and emb_dtype != np.dtype('float32')):
        raise ValueError('semantic cache embedding header mismatch')
    return _CountsCache(paths, states, sigs, hashes, legacy_hashes)


def _parsed_cache(bank_or_id: Any, engine: str) -> tuple[_ParsedCache | None, str | None]:
    global _memo
    import numpy as np
    path = semantic_cache_path(bank_or_id, engine)
    try:
        st = path.stat()
    except OSError:
        return None, None
    key = (engine, str(path), st.st_size, st.st_mtime_ns)
    with _memo_lock:
        if _memo is not None and _memo[0] == key:
            return _memo[1], None
    try:
        with np.load(str(path), allow_pickle=False) as z:
            parsed = (_parse_clip_npz(z) if engine == CLIP_ENGINE
                      else _parse_siglip2_npz(z))
    except Exception as exc:  # corrupt/wrong provenance = unavailable, never fatal
        logger.warning('semantic cache %s rejected: %s', path, exc)
        return None, str(exc)
    with _memo_lock:
        _memo = (key, parsed)
    return parsed, None


def _file_sig(path: str) -> str:
    try:
        st = os.stat(path)
        return f'{st.st_size}:{st.st_mtime_ns}'
    except OSError:
        return ''


def _sha256_path(path: str, expected_sig: str | None = None) -> bytes:
    """SHA-256 only when the path stays on the same stat generation."""
    before = _file_sig(path)
    if not before or (expected_sig and before != expected_sig):
        return b''
    digest = hashlib.sha256()
    try:
        with open(path, 'rb') as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return b''
    return digest.digest() if _file_sig(path) == before else b''


def _entry_fingerprint(path: str, sig: str, stored_hash: bytes, *,
                       legacy: bool) -> tuple[bool, str | None]:
    current_sig = _file_sig(path)
    if not current_sig:
        return False, None
    if sig and current_sig != sig:
        return False, None
    if stored_hash and stored_hash != b'\0' * 32:
        current_hash = _sha256_path(path, current_sig)
        return (current_hash == stored_hash,
                stored_hash.hex() if current_hash == stored_hash else None)
    if not legacy:
        return False, None
    # Pre-hash CLIP cache: stay readable but bind the in-memory result to the
    # exact current bytes.  The app's historical loader remains the authority
    # for deciding whether a production operation may use such an old row.
    current_hash = _sha256_path(path, current_sig)
    return bool(current_hash), current_hash.hex() if current_hash else None


def _inspect(bank_or_id: Any, engine: str) -> tuple[dict[str, Any], dict]:
    global _validated_memo
    import numpy as np
    cache_path = semantic_cache_path(bank_or_id, engine)
    try:
        cache_stat = cache_path.stat()
        validated_key = (
            engine, str(cache_path), cache_stat.st_size, cache_stat.st_mtime_ns)
    except OSError:
        validated_key = None
    if validated_key is not None:
        with _memo_lock:
            if _validated_memo is not None:
                memo_key, memo_at, memo_embs, memo_metrics, memo_fingerprints = \
                    _validated_memo
                if (memo_key == validated_key
                        and time.time() - memo_at < _VALIDATED_MEMO_TTL):
                    _fingerprints[engine] = memo_fingerprints
                    return memo_embs, dict(memo_metrics)
    parsed, error = _parsed_cache(bank_or_id, engine)
    metrics = {'cached': 0, 'ok': 0, 'stale': 0, 'error': error}
    if parsed is None:
        with _memo_lock:
            _fingerprints[engine] = {}
        return {}, metrics
    metrics['cached'] = len(parsed.paths)
    embeddings: dict[str, Any] = {}
    fingerprints: dict[str, str] = {}
    for index, path in enumerate(parsed.paths):
        digest = parsed.hashes[index].tobytes()
        current, fingerprint = _entry_fingerprint(
            path, parsed.sigs[index], digest, legacy=parsed.legacy_hashes)
        if not current:
            metrics['stale'] += 1
            continue
        if parsed.states[index] != 'ok':
            continue
        emb = np.asarray(parsed.embs[index], dtype='float32')
        norm = float(np.linalg.norm(emb))
        if not np.isfinite(norm) or norm <= 0:
            metrics['stale'] += 1
            continue
        # Historical CLIP caches are expected to be normalised, but normalising
        # on read keeps the public helper's dot-product contract explicit.
        if engine == CLIP_ENGINE and not np.isclose(norm, 1.0, rtol=1e-3, atol=1e-4):
            emb = emb / norm
        embeddings[path] = emb
        if fingerprint:
            fingerprints[path] = fingerprint
    metrics['ok'] = len(embeddings)
    with _memo_lock:
        _fingerprints[engine] = fingerprints
        if validated_key is not None:
            _validated_memo = (
                validated_key, time.time(), embeddings, dict(metrics), fingerprints)
    return embeddings, metrics


def _inspect_counts(bank_or_id: Any, engine: str,
                    eligible_paths: frozenset[str] | None = None) -> dict:
    """Fast poll metrics: NPZ + stat signatures, never image byte reads.

    The stored SHA must be non-zero, but comparing it to the live payload is the
    authority-only work of :func:`load_semantic_embeddings` and transport.  A
    Bank workspace polls frequently; hashing the whole Bank on every poll would
    turn a cheap status request into gigabytes of I/O.
    """
    global _counts_memo
    cache_path = semantic_cache_path(bank_or_id, engine)
    try:
        cache_stat = cache_path.stat()
        # The Bank can reject/delete rows without rewriting either embedding
        # cache.  Coverage memoization therefore includes the exact live target
        # set; otherwise a previous ``65 ok`` result can survive after the Bank
        # has become ``0 eligible`` and leak impossible ``65 / 0`` status.
        counts_key = (
            engine, str(cache_path), cache_stat.st_size, cache_stat.st_mtime_ns,
            eligible_paths)
    except OSError:
        counts_key = None
    if counts_key is not None:
        with _memo_lock:
            if _counts_memo is not None:
                memo_key, memo_at, memo_metrics = _counts_memo
                if (memo_key == counts_key
                        and time.time() - memo_at < _VALIDATED_MEMO_TTL):
                    return dict(memo_metrics)
    if counts_key is None:
        return {'cached': 0, 'ok': 0, 'stale': 0, 'error': None}
    try:
        parsed = _parse_counts_npz(cache_path, engine)
        error = None
    except Exception as exc:
        logger.warning('semantic cache %s rejected for counts: %s', cache_path, exc)
        parsed = None
        error = str(exc)
    metrics = {'cached': 0, 'ok': 0, 'stale': 0, 'error': error}
    if parsed is None:
        if counts_key is not None:
            with _memo_lock:
                _counts_memo = (counts_key, time.time(), dict(metrics))
        return metrics
    for index, path in enumerate(parsed.paths):
        if eligible_paths is not None and path not in eligible_paths:
            continue
        metrics['cached'] += 1
        digest = parsed.hashes[index].tobytes()
        current_sig = _file_sig(path)
        stored_sig = parsed.sigs[index]
        if (not current_sig
                or stored_sig and stored_sig != current_sig
                or not digest or digest == b'\0' * 32):
            metrics['stale'] += 1
            continue
        if parsed.states[index] != 'ok':
            continue
        metrics['ok'] += 1
    if counts_key is not None:
        with _memo_lock:
            _counts_memo = (counts_key, time.time(), dict(metrics))
    return metrics


def load_semantic_embeddings(bank: Any, engine: Any = None) -> dict:
    """Return current, SHA-bound ``{absolute_path: L2 float32 vector}``.

    A SigLIP2 cache with another model/revision/key/dimension is rejected as a
    whole.  Individual entries whose stat or SHA changed are omitted.  Passing
    an ImageBank object is preferred; an id remains accepted for diagnostics.
    """
    selected = normalize_engine(
        engine if engine is not None else getattr(bank, 'semantic_engine', None))
    embeddings, _metrics = _inspect(bank, selected)
    return embeddings


def embedding_fingerprint(path: Any, engine: Any = None) -> str | None:
    """Exact SHA-256 hex associated with the most recent cache load."""
    key = str(path)
    with _memo_lock:
        if engine is not None:
            return _fingerprints[normalize_engine(engine)].get(key)
        # Selected-engine callers normally pass it.  The fallback is useful to
        # legacy callers and refuses ambiguity instead of guessing cross-space.
        found = {mapping[key] for mapping in _fingerprints.values() if key in mapping}
        return next(iter(found)) if len(found) == 1 else None


def _infer_total(bank_or_id: Any, total: int | None, engine: str) -> int | None:
    if total is not None:
        return max(0, int(total))
    bank_id = getattr(bank_or_id, 'id', None)
    if bank_id is None:
        return None
    try:
        from ..models import BankImage
        query = (BankImage.query.filter_by(bank_id=bank_id)
                 .filter(BankImage.status != 'reject'))
        return int(query.count())
    except Exception:  # no Flask app/query context (pure helpers/tests)
        return None


def _sidecar_count(path: Path) -> int:
    try:
        return max(0, int(Path(str(path) + '.count').read_text(
            encoding='utf-8').strip()))
    except (OSError, TypeError, ValueError):
        return 0


def semantic_counts(bank_or_id: Any, engine: Any = None,
                    total: int | None = None, *,
                    eligible_paths: Iterable[Any] | None = None) -> dict:
    """Cache coverage/provenance summary consumed by Bank service payloads."""
    selected = normalize_engine(
        engine if engine is not None
        else getattr(bank_or_id, 'semantic_engine', None))
    path = semantic_cache_path(bank_or_id, selected)
    generation = cache_generation(bank_or_id, selected)
    eligible = (None if eligible_paths is None else frozenset(
        str(candidate) for candidate in eligible_paths if candidate))
    metrics = _inspect_counts(bank_or_id, selected, eligible)
    total_count = (len(eligible) if total is None and eligible is not None
                   else _infer_total(bank_or_id, total, selected))
    cached = int(metrics['cached'])
    if not cached and generation is None and eligible is None:
        # During a first interrupted write the sidecar can be the only durable
        # progress signal the lightweight parent has.
        cached = _sidecar_count(path)
    valid = generation is not None and metrics['error'] is None
    stale = int(metrics['stale'])
    ok = int(metrics['ok'])
    if total_count is None:
        complete = bool(valid and cached > 0 and stale == 0)
        needs_index = not complete
    elif total_count == 0:
        complete = True
        needs_index = False
    else:
        complete = bool(valid and ok >= total_count and stale == 0)
        needs_index = not complete
    return {
        'engine': selected,
        'model_key': engine_model_key(selected),
        'model_label': engine_model_label(selected),
        'source': 'score' if selected == CLIP_ENGINE else 'semantic_cache',
        'cache_exists': generation is not None,
        'cache_path': str(path),
        'cache_generation': generation,
        'total': total_count,
        'cached': cached,
        'ok': ok,
        'stale': stale,
        'ready': ok > 0,
        'complete': complete,
        'needs_index': needs_index,
        'dimension': engine_dimension(selected),
        'error': metrics['error'],
    }


__all__ = (
    'CLIP_ENGINE', 'SIGLIP2_ENGINE', 'SIGLIP2_CACHE_FILENAME',
    'SEMANTIC_CACHE_VERSION', 'normalize_engine', 'engine_model_key',
    'engine_model_label', 'engine_dimension', 'semantic_contract',
    'semantic_cache_path', 'cache_generation', 'semantic_counts',
    'load_semantic_embeddings', 'embedding_fingerprint', 'reset_memo',
    'image_worker_payload', 'text_worker_handshake',
)
