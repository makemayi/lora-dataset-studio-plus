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

import math
import os
from pathlib import Path

from .. import config as cfg

MODEL_TYPE_KREA_2 = 'KREA_2'
TRAINING_METHOD_LORA = 'LORA'

# The shipped preset this app builds ON TOP OF, never duplicates. Verified
# against Nerogar/OneTrainer's own repo (training_presets/Krea 2/), not
# guessed — see the spec's "Verified facts" section. train.py's
# --preset_path merges this UNDER our --config_path overrides below, so
# every knob this app doesn't explicitly own (model_type, training_method,
# base_model_name, transformer/text_encoder/vae dtypes, attention_mechanism,
# ...) stays exactly whatever OneTrainer's own maintainers tuned it to.
KREA2_PRESET_RELATIVE_PATH = 'training_presets/Krea 2/#krea2 LoRA 16GB.json'


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


def build_job_config(trigger: str, dataset_folder: str, training_folder: str,
                     steps: int, num_images: int, rank: int) -> dict:
    """The OVERRIDE config this app writes to --config_path, merged by
    OneTrainer OVER its own shipped Krea 2 preset (--preset_path). Contains
    ONLY the fields this app's own UI/dataset state actually owns — never a
    field the shipped preset already decided (see the ownership-boundary
    test above).

    `epochs` is an approximation: OneTrainer trains by epoch count, this
    app's UI/recommended_steps() thinks in step count. One epoch here means
    "one pass over the dataset at batch_size 1" — a documented approximation
    (see spec's Open Questions), not a verified equivalence."""
    epochs = max(1, math.ceil(steps / max(1, num_images)))
    training_folder = Path(training_folder)
    return {
        'workspace_dir': str(training_folder),
        'cache_dir': str(training_folder / 'cache'),
        'output_model_destination': str(training_folder / f'{trigger}.safetensors'),
        'epochs': epochs,
        'lora_rank': int(rank),
    }


def build_concepts(trigger: str, dataset_folder: str) -> list[dict]:
    """The concepts.json content — one concept pointing at the already-
    exported dataset folder. `prompt_source` is deliberately OMITTED: its
    default ("sample" — a per-image .txt sidecar matching the image
    filename) is already this app's export format, so there is nothing to
    override."""
    return [{'name': trigger, 'path': dataset_folder, 'enabled': True}]
