"""Remembering that an install was already checked out fine, so coming back to
the app does not mean sitting through Setup again.

The Setup wizard is a FIRST-RUN experience, but nothing recorded that the first
run had happened: the onboarding redirect keyed off a per-tab sessionStorage
flag, so every new tab / browser session re-offered Setup on a machine that had
been working for weeks. That is the redundancy this module removes.

Two ideas, kept apart on purpose:

  verified  — this install was once observed working (configured + at least one
              image engine ready). Persisted in the data dir, so it survives a
              new tab, a new browser, a server restart, and a second machine on
              the LAN. Never expires: an install does not become un-set-up.

  checks    — a HIGH-WATER MARK of what this install has proven it can do
              (`TRACKED` below). A re-check compares the live probe against it
              and reports what STOPPED working — a regression is a real failure
              worth interrupting for, whereas "not everything is installed" is
              the normal state of almost every install and must never nag.

Only DURABLE signals are tracked. Whether ComfyUI or Ollama happens to be
running right now is not a setup failure — those are started on demand and go
dark a dozen times a day; treating them as regressions would turn this into the
exact nag it exists to remove. See TRACKED / the module docstring of
capabilities.py for the full published surface.

Nothing here is a user SETTING: it is app state the user never edits, so it
lives in its own small file rather than in config.json.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from . import config

_LOCK = threading.RLock()

# Durable capability checks, keyed by their dotted path in the /api/capabilities
# payload. The KEYS are persisted (never rename one without an alias — CLAUDE.md
# rule 7); the labels are read at response time and are free to change.
TRACKED = (
    ('engines.nanobanana', 'Nano Banana (Gemini)'),
    ('engines.chatgpt', 'ChatGPT (gpt-image-2)'),
    ('engines.openrouter', 'OpenRouter'),
    ('comfyui.dir_valid', 'ComfyUI folder'),
    ('ollama.installed', 'Ollama'),
    ('aitoolkit.valid', 'ai-toolkit'),
    ('captioners.joycaption', 'JoyCaption captioning'),
    ('face_scoring', 'Face-similarity scoring'),
    ('masks', 'Person masks'),
    ('watermark_inpaint', 'Watermark inpainting'),
    ('training_visible', 'LoRA training'),
)

TRACKED_KEYS = tuple(k for k, _ in TRACKED)
_LABELS = dict(TRACKED)

# Engines that make an install "known-good". Deliberately the same set the nav
# rail already calls "recommended" (useSetupSteps.recommendedMet): one working
# way to generate an image is what separates a set-up machine from a fresh one.
_RECOMMENDED_ENGINES = ('nanobanana', 'chatgpt', 'openrouter', 'klein')


def _state_path():
    return config.data_dir() / 'setup_state.json'


def _dig(caps: dict, dotted: str):
    """Read a dotted path out of the capabilities payload; missing -> None, so a
    key this build does not publish is 'unknown', never a false regression."""
    cur = caps
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def snapshot(caps: dict) -> dict:
    """The tracked checks as they stand in `caps`. Unknown keys are dropped
    rather than recorded as False — an older/newer payload must not manufacture
    a regression out of a field it simply does not have."""
    out = {}
    for key in TRACKED_KEYS:
        val = _dig(caps or {}, key)
        if val is not None:
            out[key] = bool(val)
    return out


def install_works(caps: dict) -> bool:
    """Is this a set-up machine? Configured AND able to generate an image."""
    c = caps or {}
    if not c.get('configured'):
        return False
    engines = c.get('engines') or {}
    return any(bool(engines.get(name)) for name in _RECOMMENDED_ENGINES)


def read() -> dict:
    """Stored state, or the empty shape. Never raises: a corrupt/unreadable file
    means 'we have not verified this install', which degrades to the classic
    first-run wizard rather than to an error screen."""
    with _LOCK:
        try:
            raw = json.loads(_state_path().read_text(encoding='utf-8'))
        except Exception:
            return {'verified': False, 'verified_at': None, 'checks': {}}
        if not isinstance(raw, dict):
            return {'verified': False, 'verified_at': None, 'checks': {}}
        checks = raw.get('checks')
        if not isinstance(checks, dict):
            checks = {}
        return {
            'verified': bool(raw.get('verified')),
            'verified_at': raw.get('verified_at') or None,
            # Unknown keys are dropped on read, so a key retired in a later
            # version cannot resurface as a phantom regression.
            'checks': {k: bool(v) for k, v in checks.items() if k in TRACKED_KEYS},
        }


def _write(state: dict) -> dict:
    with _LOCK:
        path = _state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
            tmp.replace(path)
        except OSError:
            # A read-only data dir must not break the app: the feature simply
            # stops remembering and the user gets today's first-run behaviour.
            pass
        return state


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def observe(caps: dict) -> dict:
    """Record what this install can currently do, and stamp it verified the first
    time it is seen working.

    The stored checks are a HIGH-WATER MARK: a capability that has ever worked
    stays True here even when it is currently failing, which is precisely what
    lets `compare` keep reporting a regression until it is fixed (or explicitly
    dismissed) instead of quietly re-baselining the broken state.

    Called on every read/re-check, so an install set up BEFORE this feature
    shipped is retrofitted on its first page load — no re-run of the wizard.
    """
    with _LOCK:
        state = read()
        checks = dict(state['checks'])
        for key, val in snapshot(caps).items():
            if val:
                checks[key] = True
            else:
                checks.setdefault(key, False)
        verified = bool(state['verified'] or install_works(caps))
        nxt = {
            'verified': verified,
            'verified_at': state['verified_at'] or (_now() if verified else None),
            'checks': checks,
        }
        # This runs on every page load; a steady install would otherwise rewrite
        # a byte-identical file each time for nothing.
        return nxt if nxt == state else _write(nxt)


def compare(caps: dict, state: dict | None = None) -> list:
    """Capabilities this install has proven it can do and that are failing NOW.

    Returns [{'key', 'label'}] — empty means the background re-check found
    nothing to interrupt for. A key that is missing from the live payload is
    skipped (unknown != broken)."""
    stored = (state or read())['checks']
    live = snapshot(caps)
    return [{'key': k, 'label': _LABELS[k]}
            for k in TRACKED_KEYS
            if stored.get(k) and k in live and not live[k]]


def dismiss(keys) -> dict:
    """Forget that a capability used to work — the user's "yes, I removed that
    on purpose". Without this, an intentional uninstall would be reported as a
    regression forever, and the only escape would be deleting a state file the
    user has no reason to know exists."""
    with _LOCK:
        state = read()
        checks = dict(state['checks'])
        for key in keys or ():
            if key in TRACKED_KEYS:
                checks[key] = False
        return _write({**state, 'checks': checks})
