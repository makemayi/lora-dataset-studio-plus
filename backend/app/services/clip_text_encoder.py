"""Turn a written query into a CLIP vector — and pay for it as rarely as possible.

The bank already holds a CLIP embedding for every ✨ Score-d image, so ranking a
bank against a sentence is one dot product per image: numpy does it in-process in
milliseconds. The ONE expensive step is producing the SENTENCE's vector, because
torch/open_clip are not in the Flask venv.

Measured on the reference machine (ai-toolkit venv, torch 2.9.1, CPU forced):

    import torch + open_clip .................. 5.4 s
    build model + load weights ................ 2.4 s
    encode_text, first call ................... 24-48 ms
    encode_text, subsequent calls ............. 20-25 ms
    whole one-shot subprocess, launch → exit ... 8.5-8.8 s
    peak RSS .................................. 2.43 GB

Those numbers decide the design. 8.5 s to encode three words, of which 7.8 s is
pure start-up, makes "one subprocess per search" untenable — and search is
iterative by nature, so the user pays it on every refinement. But the flip side
is 2.43 GB held for as long as a model stays resident, which on a modest machine
is a lot to spend on a feature used in bursts.

So the arbitration is three-layered, cheapest first:

1. **A persistent query cache.** A CLIP text vector depends only on the
   checkpoint, so it is written next to the banks and re-read forever, across
   restarts, app-wide. A re-typed or favourite phrase costs nothing and does not
   even wake the worker. This is the only layer that survives a restart.
2. **A lazily-started WARM worker.** The first uncached query of a session starts
   the child and pays the ~8.5 s load; every later one costs ~20 ms. Nothing is
   started at boot and nothing is started for a cache hit — an install that never
   searches never pays a byte.
3. **An idle reaper.** The worker exits after ``bank_scoring.text_search_idle_minutes``
   without a query (default 10). Ten minutes covers a realistic refine-and-retry
   session while guaranteeing the 2.43 GB is never held overnight or through an
   unattended training run. Setting it to 0 disables the warm layer entirely —
   the honest choice on a memory-tight machine, where paying 8.5 s per distinct
   phrase beats surrendering 2.4 GB. ``release()`` reaps it immediately when the
   user closes the search panel; the timer is the backstop for a closed tab.

**CPU, always.** The text tower is small and the cost is the load, not the maths,
so the GPU would buy nothing measurable and would make a search collide with a
training run. The child hides CUDA from itself, no GPU window is ever taken, and
a search works fine WHILE a LoRA trains — it competes for RAM, never for VRAM.

**No silent download.** open_clip fetches ~1.6 GB of ViT-L/14 weights on first
use. Text search never triggers that on its own: it refuses outright unless the
bank already HAS ✨ Score embeddings, which means this machine already downloaded
this exact checkpoint to produce them. The one residual case — the user re-pointed
``bank_scoring.python``/``models_root`` at an environment with a different cache
since scoring — is reported by ``weights_warning()`` so the UI can warn BEFORE the
click rather than stall for ten minutes.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time

from .. import config as cfg

_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'clip_text_infer.py')

# A cold `import torch` alone can run tens of seconds on a fresh machine (native
# DLLs + antivirus), and CLIP ViT-L/14 has ~1.6 GB of weights to read after that
# — more if they still have to be downloaded. Generous on purpose: a timeout here
# would read as "text search is broken" about an interpreter that was merely slow.
START_TIMEOUT = 900
# Once warm, a query is ~20 ms. Anything past this means the worker is wedged.
QUERY_TIMEOUT = 120
DEFAULT_IDLE_MINUTES = 10

_lock = threading.RLock()
_memory: dict = {}          # normalised query -> list[float]
_loaded = False             # has the on-disk cache been read into _memory yet?

_proc = None                # the warm worker (subprocess.Popen) or None
_last_used = 0.0
_reaper = None


class TextEncodeError(RuntimeError):
    """The encoder ran but could not produce a vector. Carries the child's own
    words so the UI can show WHY instead of a generic failure."""


# --- the cache key ------------------------------------------------------------
_WS = re.compile(r'\s+')


def normalize_query(text: str) -> str:
    """The cache key for a query: lowercased, whitespace collapsed, trimmed.

    Deliberately nothing more. Stripping punctuation or stop-words would change
    the vector CLIP produces (its tokenizer sees the raw string), so the key must
    only fold differences the ENCODER itself ignores. Kept in one named place
    because widening it later would silently invalidate every cached vector on
    every install."""
    return _WS.sub(' ', str(text or '').strip()).lower()


def idle_minutes() -> float:
    """Configured warm window, clamped to something sane. 0 = never stay warm."""
    try:
        v = float(cfg.get('bank_scoring.text_search_idle_minutes'))
    except (TypeError, ValueError):
        return DEFAULT_IDLE_MINUTES
    if v != v or v < 0:           # NaN / negative -> the default, never a trap
        return DEFAULT_IDLE_MINUTES
    return min(v, 120.0)


# --- persistent cache ---------------------------------------------------------
def _cache_path():
    """App-wide, next to the banks: a text vector depends only on the CLIP
    checkpoint, never on which bank is being searched."""
    return cfg.banks_root() / 'text_query_cache.npz'


def forget_memory_cache() -> None:
    """Drop the in-memory layer (tests use it to simulate a restart). The file is
    untouched — that is the point."""
    global _loaded
    with _lock:
        _memory.clear()
        _loaded = False


def _load_disk_cache() -> None:
    """Read the .npz into memory once. A missing or corrupt file is simply an
    empty cache — never an error; the worst case is re-encoding."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    import numpy as np
    path = _cache_path()
    try:
        if not path.is_file():
            return
        with np.load(str(path), allow_pickle=False) as z:
            queries = [str(q) for q in z['queries']]
            vecs = z['vecs']
        for i, q in enumerate(queries):
            _memory[q] = [float(x) for x in vecs[i]]
    except Exception:  # noqa: BLE001 — a corrupt cache = recompute, never fatal
        _memory.clear()


def _save_disk_cache() -> None:
    import numpy as np
    if not _memory:
        return
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(_memory)
        tmp = str(path) + '.tmp.npz'
        np.savez_compressed(tmp,
                            queries=np.array(keys),
                            vecs=np.array([_memory[k] for k in keys], dtype='float32'))
        os.replace(tmp, str(path))
    except Exception:  # noqa: BLE001 — losing the cache costs time, never data
        pass


def cached_queries() -> int:
    """How many phrases are already encoded — lets the UI say "instant" honestly."""
    with _lock:
        _load_disk_cache()
        return len(_memory)


def is_cached(text: str) -> bool:
    with _lock:
        _load_disk_cache()
        return normalize_query(text) in _memory


# --- availability -------------------------------------------------------------
def unavailable_reason():
    """None when a text query CAN be encoded here, else a sentence explaining why
    not. Text search reuses the ✨ Score interpreter — the very one that produced
    the embeddings it ranks — so if that cannot run, neither can this, and the
    honest answer is to say so rather than to fail mid-request.

    Never raises: a probe that itself explodes reports "could not be verified",
    which is still an answer the UI can render."""
    try:
        from ..capabilities import probe_bank_scoring
        if probe_bank_scoring().get('ok'):
            return None
    except Exception as e:  # noqa: BLE001
        return (f'the ✨ Score environment could not be verified '
                f'({type(e).__name__}) — text search needs it')
    return ('text search needs the same environment as ✨ Score '
            '(torch + open_clip). Install the Quality tools step in Setup, or '
            'point ✨ Score at a Python that already has them.')


def weights_warning():
    """A warning string when the FIRST search might have to download the ~1.6 GB
    ViT-L/14 checkpoint, else None.

    Normally it cannot: text search requires ✨ Score embeddings to exist, and
    producing those downloaded this very checkpoint. The exception is a
    models_root that has been re-pointed since — we can see the configured folder
    is empty of an open_clip cache and say so, instead of letting the user stare
    at a spinner for ten minutes. Best effort by design: an unreadable folder
    returns None (we do not cry wolf about a layout we could not inspect)."""
    root = (cfg.get('bank_scoring.models_root') or '').strip()
    if not root:
        return None            # default HF cache — the Score pass filled it
    try:
        d = os.path.join(root, 'open_clip')
        if os.path.isdir(d) and any(os.scandir(d)):
            return None
    except OSError:
        return None
    return ('the CLIP weights are not in the models folder currently configured '
            'for ✨ Score — the first search may download ~1.6 GB')


# --- warm worker --------------------------------------------------------------
def _reap_if_idle():
    """Reaper loop: exit the worker once it has been unused for the idle window."""
    global _reaper
    while True:
        with _lock:
            if _proc is None:
                _reaper = None
                return
            window = idle_minutes() * 60.0
            quiet = time.time() - _last_used
            if window <= 0 or quiet >= window:
                _stop_worker_locked()
                _reaper = None
                return
            sleep_for = max(1.0, min(30.0, window - quiet))
        time.sleep(sleep_for)


def _stop_worker_locked():
    """Terminate the worker and give its ~2.4 GB back. Caller holds the lock."""
    global _proc
    proc, _proc = _proc, None
    if proc is None:
        return
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()           # EOF = the child's clean exit path
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def release() -> bool:
    """Reap the warm worker now (the search panel was closed). True when there
    was one to reap — the UI turns that into an honest "memory released"."""
    with _lock:
        had = _proc is not None
        _stop_worker_locked()
    return had


def status() -> dict:
    """What the UI needs to set expectations BEFORE the click: is the model
    already warm (next search instant), how many phrases are cached, is the
    feature available at all, and would a download be needed."""
    with _lock:
        warm = _proc is not None and _proc.poll() is None
    reason = unavailable_reason()
    return {'available': reason is None, 'reason': reason, 'warm': warm,
            'idle_minutes': idle_minutes(), 'cached_queries': cached_queries(),
            'weights_warning': weights_warning()}


def _start_worker_locked():
    """Spawn the child and block until it says it is ready. Caller holds the lock."""
    global _proc, _reaper
    python = cfg.get('bank_scoring.python') or sys.executable
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ''          # belt and braces with the child
    env['PYTHONUTF8'] = '1'
    try:
        proc = subprocess.Popen(
            [python, _SCRIPT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
            errors='replace', bufsize=1, env=env,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except Exception as e:  # noqa: BLE001
        raise TextEncodeError(f'could not start the text encoder: '
                              f'{type(e).__name__}: {e}') from None
    try:
        proc.stdin.write(json.dumps({
            'models_root': cfg.get('bank_scoring.models_root') or None}) + '\n')
        proc.stdin.flush()
        line = _readline_with_timeout(proc, START_TIMEOUT)
        data = json.loads(line)
    except TextEncodeError:
        _kill(proc)
        raise
    except Exception:  # noqa: BLE001
        _kill(proc)
        raise TextEncodeError('the text encoder produced no result — check the '
                              '✨ Score interpreter') from None
    if not data.get('ok') or not data.get('ready'):
        _kill(proc)
        raise TextEncodeError(str(data.get('error') or 'unknown encoder error'))
    _proc = proc
    if _reaper is None:
        _reaper = threading.Thread(target=_reap_if_idle, daemon=True,
                                   name='clip-text-reaper')
        _reaper.start()
    return proc


def _kill(proc):
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


def _readline_with_timeout(proc, timeout):
    """One stdout line, or TextEncodeError. subprocess pipes have no timed read,
    so the wait happens on a helper thread — a wedged child must never hang a
    Flask worker forever."""
    box = {}

    def _read():
        try:
            box['line'] = proc.stdout.readline()
        except Exception as e:  # noqa: BLE001
            box['err'] = e

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or 'err' in box:
        raise TextEncodeError(
            f'the text encoder did not answer within {int(timeout // 60)} minutes '
            '— loading CLIP on this machine is unusually slow')
    line = box.get('line') or ''
    if not line.strip():
        raise TextEncodeError('the text encoder exited before answering — check '
                              'the ✨ Score interpreter')
    return line


def _encode_uncached(texts, models_root=None):
    """[vector] for `texts`, through the warm worker (started on demand). Raises
    TextEncodeError with the child's message on any failure. Monkeypatched
    wholesale in tests — no test ever loads a real model."""
    global _last_used
    import numpy as np
    out = []
    with _lock:
        proc = _proc if (_proc is not None and _proc.poll() is None) else None
        if proc is None:
            proc = _start_worker_locked()
        for text in texts:
            try:
                proc.stdin.write(json.dumps({'text': text}) + '\n')
                proc.stdin.flush()
                data = json.loads(_readline_with_timeout(proc, QUERY_TIMEOUT))
            except TextEncodeError:
                _stop_worker_locked()      # a wedged worker must not be reused
                raise
            except Exception:  # noqa: BLE001
                _stop_worker_locked()
                raise TextEncodeError('the text encoder stopped responding') from None
            if not data.get('ok'):
                raise TextEncodeError(str(data.get('error') or 'unknown encoder error'))
            out.append(np.asarray(data.get('vector') or [], dtype='float32'))
        _last_used = time.time()
        # idle_minutes == 0 means "never stay warm" — reap as soon as the answer
        # is in hand, so a memory-tight machine never carries the 2.4 GB.
        if idle_minutes() <= 0:
            _stop_worker_locked()
    return out


def encode_query(text: str):
    """(vector, from_cache) for ONE query, L2-normed float32.

    Raises TextEncodeError when the encoder is unavailable or failed — the caller
    turns that into an announced 503, never a 500."""
    import numpy as np
    key = normalize_query(text)
    if not key:
        raise TextEncodeError('empty query')
    with _lock:
        _load_disk_cache()
        hit = _memory.get(key)
    if hit is not None:
        return np.asarray(hit, dtype='float32'), True
    reason = unavailable_reason()
    if reason:
        raise TextEncodeError(reason)
    vec = _encode_uncached([key])[0]
    vec = np.asarray(vec, dtype='float32')
    if vec.size == 0:
        raise TextEncodeError('the text encoder returned an empty vector')
    vec /= (float(np.linalg.norm(vec)) + 1e-8)
    with _lock:
        _memory[key] = [float(x) for x in vec]
        _save_disk_cache()
    return vec, False
