"""OneTrainer — second LOCAL training backend (Krea 2 first slice).

Everything here is additive and isolated: nothing in the existing
lora_training.py / cloud_training.py imports this module, and this module
is only reached when a training launch explicitly asks for
trainer='onetrainer'. See docs/superpowers/specs/2026-07-30-onetrainer-backend-design.md
(local-only file, not in version control) for the full design.

Reuses the EXISTING shared safety/tracking machinery from lora_training.py
and checkpoint_registry.py rather than duplicating it — a OneTrainer run is
still a `source='local'` TrainingRunRecord, still governed by the same
single-training-in-progress guard, still watched by the same kind of
process-exit thread. The `trainer` column is an orthogonal tag, not a new
state machine.
"""
from __future__ import annotations

import os
from pathlib import Path

from .. import config as cfg

MODEL_TYPE_KREA_2 = 'KREA_2'
TRAINING_METHOD_LORA = 'LORA'


def _derived_python(root: Path) -> Path:
    win = root / 'venv' / 'Scripts' / 'python.exe'
    return win if os.name == 'nt' else root / 'venv' / 'bin' / 'python'


def onetrainer_path(kind: str):
    """Mirrors `cfg.aitoolkit_path` — same blank-means-unconfigured contract."""
    root = cfg.get('onetrainer.dir') or ''
    if not root:
        return None
    root = Path(root)
    if kind == 'dir':
        return root
    if kind == 'venv_python':
        explicit = (cfg.get('onetrainer.python') or '').strip()
        if explicit:
            return Path(explicit)
        return _derived_python(root)
    raise ValueError(f'unknown onetrainer_path kind: {kind}')


def is_installed() -> bool:
    """OneTrainer usable (venv python present)?"""
    p = onetrainer_path('venv_python')
    return bool(p) and p.is_file()
