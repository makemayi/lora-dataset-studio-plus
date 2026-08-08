"""Pinned, explicitly-installed model assets for the Bank SigLIP2 engine.

This module is intentionally dependency-free. Capability polling runs on every
Bank page and must be able to answer from the filesystem without importing torch,
transformers or huggingface_hub in the Flask interpreter.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .. import config as cfg

MODEL_ID = 'google/siglip2-base-patch16-224'
REVISION = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
MODEL_KEY = 'siglip2-base-p16-224@75de2d55'
DIMENSION = 768
DOWNLOAD_MB = 1540

# Exact allow-list, rather than snapshot_download: no training artefacts and no
# future repository addition can silently enlarge the Setup action.
FILES = (
    'config.json',
    'model.safetensors',
    'preprocessor_config.json',
    'special_tokens_map.json',
    'tokenizer.json',
    'tokenizer.model',
    'tokenizer_config.json',
)


def semantic_python() -> str:
    """Interpreter used by every SigLIP2 probe and worker.

    ``bank_scoring.python`` is retained only as a read-time compatibility
    fallback for configs written before SigLIP2 gained its own managed runtime.
    New installs persist ``bank_semantic.python`` and never repoint Score.
    """
    return (str(cfg.get('bank_semantic.python') or '').strip()
            or str(cfg.get('bank_scoring.python') or '').strip()
            or sys.executable)


def models_root() -> Path:
    configured = str(cfg.get('bank_semantic.models_root') or '').strip()
    return Path(configured) if configured else cfg.data_dir() / 'models' / 'bank_semantic'


def snapshot_dir(root=None) -> Path:
    root = Path(root) if root else models_root()
    repo = 'models--' + MODEL_ID.replace('/', '--')
    return root / repo / 'snapshots' / REVISION


def weights_present(root=None) -> bool:
    """True only when every pinned file exists and is non-empty."""
    snap = snapshot_dir(root)
    try:
        return all((snap / name).is_file() and (snap / name).stat().st_size > 0
                   for name in FILES)
    except OSError:
        return False


def model_kwargs() -> dict:
    """The immutable local-only contract shared by image and text workers."""
    return {
        'pretrained_model_name_or_path': MODEL_ID,
        'revision': REVISION,
        'cache_dir': str(models_root()),
        'local_files_only': True,
    }
