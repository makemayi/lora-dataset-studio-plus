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

import json
import logging
import math
import os
import subprocess
from pathlib import Path

from .. import config as cfg

logger = logging.getLogger(__name__)

MODEL_TYPE_KREA_2 = 'KREA_2'
TRAINING_METHOD_LORA = 'LORA'

# PEFT adapter choices this app's own `onetrainer.peft_type` setting accepts.
# LORA is what this app's own Krea 2 Edit inference graph loads via
# LoraLoaderModelOnly; OFT_2 (Orthogonal Finetuning) is a different adapter
# algorithm, user-confirmed to train and load correctly (2026-07-31). An
# unrecognised setting value falls back to LORA rather than failing a launch.
PEFT_TYPE_LORA = 'LORA'
PEFT_TYPE_OFT_2 = 'OFT_2'
PEFT_TYPES = (PEFT_TYPE_LORA, PEFT_TYPE_OFT_2)

# The shipped preset this app builds ON TOP OF, never duplicates. Verified
# against Nerogar/OneTrainer's own repo (training_presets/Krea 2/), not
# guessed — see the spec's "Verified facts" section. train.py's
# --preset-path merges this UNDER our --config-path overrides below, so
# every knob this app doesn't explicitly own (model_type, training_method,
# base_model_name, transformer/text_encoder/vae dtypes, attention_mechanism,
# ...) stays exactly whatever OneTrainer's own maintainers tuned it to.
KREA2_PRESET_RELATIVE_PATH = 'training_presets/Krea 2/#krea2 LoRA 16GB.json'
KREA2_PRESET_24GB_RELATIVE_PATH = 'training_presets/Krea 2/#krea2 LoRA 24GB.json'
# Below this much VRAM, keep the 16 GB preset. The two shipped presets differ in
# EXACTLY ONE field (diffed 2026-08-09):
#
#   16GB  transformer: {train, weight_dtype: INT_W8A8, offload_fraction: 0.3}
#   24GB  transformer: {train, weight_dtype: INT_W8A8}
#
# `offload_fraction: 0.3` keeps 30% of the transformer in system RAM and swaps
# it per step. On a 24 GB card that is a pure, unnecessary slowdown — the same
# class of problem ("Moving transformer to CPU") that was chased for hours on
# the ai-toolkit lane the same day. The app pinned the 16 GB preset
# unconditionally, so every 24 GB machine paid for offload it did not need.
#
# 20 GB, not 24: a "24 GB" card reports ~23.99 GiB and a 20 GB card (RTX 4000
# Ada) also has room to skip the offload. Detection is ADVISORY — an unknown
# VRAM keeps the conservative 16 GB preset, because guessing wrong upward turns
# a slow run into an OOM.
KREA2_PRESET_24GB_MIN_VRAM_GB = 20


def krea2_preset_relative_path(vram_gb=None) -> str:
    """Which shipped Krea 2 LoRA preset to merge under this app's overrides."""
    if vram_gb is None:
        try:
            from .run_environment import local_vram_gb
            vram_gb = local_vram_gb()
        except Exception:                                # noqa: BLE001 — advisory
            vram_gb = None
    if isinstance(vram_gb, (int, float)) and vram_gb >= KREA2_PRESET_24GB_MIN_VRAM_GB:
        return KREA2_PRESET_24GB_RELATIVE_PATH
    return KREA2_PRESET_RELATIVE_PATH


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


# The shipped preset's own resolution (512) is tuned for its 16GB-VRAM budget,
# not for this app's own recipe — 1024 is what this app's own launches want,
# so it moves into the fields we own rather than a one-off edit of the
# install's shipped preset file (which a reinstall/preset update would lose).
KREA2_RESOLUTION = 1024


def build_job_config(trigger: str, dataset_folder: str, training_folder: str,
                     steps: int, num_images: int, rank: int,
                     peft_type: str = PEFT_TYPE_LORA,
                     learning_rate: float | None = None,
                     resolution: int | None = None) -> dict:
    """The OVERRIDE config this app writes to --config-path, merged by
    OneTrainer OVER its own shipped Krea 2 preset (--preset-path). Contains
    ONLY the fields this app's own UI/dataset state actually owns — never a
    field the shipped preset already decided (see the ownership-boundary
    test above).

    `epochs` is an approximation: OneTrainer trains by epoch count, this
    app's UI/recommended_steps() thinks in step count. One epoch here means
    "one pass over the dataset at batch_size 1" — a documented approximation
    (see spec's Open Questions), not a verified equivalence. `batch_size` is
    therefore pinned to 1 here too: leaving it at the shipped preset's own
    value would silently change how many optimizer steps `epochs` actually
    buys, without this function's epoch math ever knowing.

    `lora_alpha` is pinned to equal `rank` (scale factor 1.0, the standard
    "no-op" LoRA convention) for the same reason `lora_rank` is owned here:
    a LoRA's effective output strength scales by alpha/rank, and the shipped
    preset's own `lora_alpha` was tuned for ITS rank, not whatever rank this
    app's UI passes in. MEASURED (2026-07-31): a run left at the preset's
    lora_alpha=1 with this app's rank=32 trained a LoRA scaled to ~1/32 of
    its intended strength — indistinguishable from "training never
    converged," when the weights themselves were fine.

    `peft_type` (default LORA, the `onetrainer.peft_type` setting) is owned
    here rather than left to the shipped preset for the same reason: it is a
    user choice this app's own UI now exposes, not a preset-tuning decision.
    An unrecognised value degrades to LORA."""
    epochs = max(1, math.ceil(steps / max(1, num_images)))
    training_folder = Path(training_folder)
    if peft_type not in PEFT_TYPES:
        peft_type = PEFT_TYPE_LORA
    return {
        'workspace_dir': str(training_folder),
        'cache_dir': str(training_folder / 'cache'),
        'output_model_destination': str(training_folder / f'{trigger}.safetensors'),
        'epochs': epochs,
        'lora_rank': int(rank),
        'lora_alpha': float(rank),
        'batch_size': 1,
        # OneTrainer takes resolution as a STRING (its shipped preset says
        # "512"), unlike ai-toolkit's list of ints. Passing an int here is not a
        # type nit — it is how the two lanes end up training the same dataset at
        # different sizes without anything on screen saying so.
        'resolution': str(int(resolution or KREA2_RESOLUTION)),
        'peft_type': peft_type,
        # The app owns the learning rate: it is a per-dataset setting the UI
        # exposes and the ai-toolkit lane already honours. Left unset, this run
        # silently used the shipped preset's 0.0003 while the SAME dataset
        # trained at 0.0001 on ai-toolkit — a 3x divergence with no indication
        # anywhere. Only written when the caller resolved one, so the preset
        # still decides for any path that has no opinion.
        **({'learning_rate': float(learning_rate)} if learning_rate else {}),
    }


def build_concepts(trigger: str, dataset_folder: str) -> list[dict]:
    """The concepts.json content — one concept pointing at the already-
    exported dataset folder. `prompt_source` is deliberately OMITTED: its
    default ("sample" — a per-image .txt sidecar matching the image
    filename) is already this app's export format, so there is nothing to
    override."""
    return [{'name': trigger, 'path': dataset_folder, 'enabled': True}]


def checkpoint_ready(output_model_destination: str) -> bool:
    """Success is exit-code-0 AND the file existing — see spec's Error
    handling: a process that exits 0 but produced nothing is treated as a
    failure, never a silent no-op success."""
    return os.path.isfile(output_model_destination)


def launch(trigger: str, dataset_folder: str, training_folder: str,
          steps: int, num_images: int, rank: int,
          peft_type: str = PEFT_TYPE_LORA,
          learning_rate: float | None = None,
          resolution: int | None = None) -> dict:
    """Write concepts.json + config.json under `training_folder` and spawn
    `scripts/train.py --preset-path <shipped Krea 2 preset> --config-path
    <our config.json>`. Returns {'pid': int, 'config_path': str,
    'concepts_path': str}. Raises RuntimeError if OneTrainer isn't
    installed/configured — same contract as lora_training.launch_training's
    own ai-toolkit check, so the route can map it to a 409 the same way."""
    if not is_installed():
        raise RuntimeError('OneTrainer is not configured')
    venv_python = onetrainer_path('venv_python')
    root = onetrainer_path('dir')

    training_folder_p = Path(training_folder)
    training_folder_p.mkdir(parents=True, exist_ok=True)

    config = build_job_config(trigger=trigger, dataset_folder=dataset_folder,
                              training_folder=training_folder, steps=steps,
                              num_images=num_images, rank=rank, peft_type=peft_type,
                              learning_rate=learning_rate, resolution=resolution)
    concepts = build_concepts(trigger=trigger, dataset_folder=dataset_folder)

    concepts_path = training_folder_p / 'concepts.json'
    config_path = training_folder_p / 'config.json'
    concepts_path.write_text(json.dumps(concepts, indent=2), encoding='utf-8')
    config_with_concepts = {**config, 'concept_file_name': str(concepts_path)}
    config_path.write_text(json.dumps(config_with_concepts, indent=2), encoding='utf-8')

    preset_rel = krea2_preset_relative_path()
    preset_path = root / preset_rel
    logger.info('onetrainer: using shipped preset %s', preset_rel)
    log_path = training_folder_p / 'onetrainer.log'
    logf = open(log_path, 'w', encoding='utf-8')
    proc = subprocess.Popen(
        # OneTrainer's actual argparse flags are HYPHENATED (--preset-path,
        # --config-path), confirmed against train.py's own usage output —
        # the underscored form silently fails argparse and the process
        # exits before doing anything (issue found running Krea 2 for real).
        [str(venv_python), 'scripts/train.py',
         '--preset-path', str(preset_path), '--config-path', str(config_path)],
        cwd=str(root), stdout=logf, stderr=subprocess.STDOUT, shell=False,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return {'pid': proc.pid, 'config_path': str(config_path),
            'concepts_path': str(concepts_path), 'log_path': str(log_path),
            'output_model_destination': config['output_model_destination'],
            '_proc': proc}


def _training_folder_for(ds) -> Path:
    """OneTrainer's own run root, isolated from ai-toolkit's `_output_dir()`
    on purpose: a user may have ONLY OneTrainer configured (no ai-toolkit at
    all — see routes' `dataset_train_status`), so this must never call
    anything that raises when ai-toolkit is unconfigured. Rooted under this
    app's own data dir, named after the dataset the same way ai-toolkit's
    `_run_name` keys on user+trigger."""
    from .lora_training import _safe_trigger
    return cfg.data_dir() / 'onetrainer_runs' / f'u{ds.user_id}_{_safe_trigger(ds)}'


def launch_training(user_id, dataset_id, steps: int | None = None,
                    check_captions: bool = True) -> dict:
    """The OneTrainer counterpart of lora_training.launch_training, scoped to
    Krea 2 only for this slice. Reuses the SAME disk-space guard, caption
    check, checkpoint_registry.register_launch (source='local') and
    training-in-progress system-state keys ai-toolkit runs already use — a
    OneTrainer run is tracked identically, only the config/launch step
    differs (see spec's Hard constraint section)."""
    if not is_installed():
        raise RuntimeError('OneTrainer is not configured')
    from . import face_dataset_service as fds
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if (ds.train_type or 'zimage') != 'krea':
        raise ValueError('OneTrainer only supports the Krea 2 family in this build')

    from .lora_training import (_TRAIN_STATE_TTL, _pid_alive, assert_free_disk,
                                assert_trainable, export_dataset_to_aitoolkit,
                                recommended_steps)
    from ..job_queue import queue_manager

    assert_free_disk(cfg.data_dir(), 5, 'a training run')
    if (queue_manager._get_system_state('training_in_progress', False)
            and _pid_alive(queue_manager._get_system_state('training_pid', None))):
        raise ValueError('a training is already in progress - wait for it to finish or queue this dataset')
    # THE GPU, same three questions the ai-toolkit lane asks. This lane asked
    # none of them: it checked only that no OTHER training was running, so it
    # would start on a card ComfyUI was rendering on, or one an idle ComfyUI was
    # still holding.
    #
    # That second case is not theoretical — it killed an ai-toolkit run on
    # 2026-08-09. An idle ComfyUI sat on 4.4 GB (6005 MiB, 1581 MiB after
    # /free), the run started with ~18 GB of a 24 GB card instead of ~22 GB,
    # and WDDM paged VRAM to system RAM until the step time went 8.4 s -> 78 s
    # -> 104 s and the process died at step 3 with no error. OneTrainer trains
    # the same 12B Krea model on the same card and would fail the same way.
    from ..gpu_window import GpuBusyError
    from ..utils.comfyui import ComfyVramFreeVerdict, free_comfyui_vram
    from .lora_training import _assert_no_vision_pass_on_gpu
    _assert_no_vision_pass_on_gpu()
    if queue_manager.has_comfyui_work():
        raise GpuBusyError(
            'ComfyUI has queued or active work, so local training cannot take the GPU. '
            'Wait for it to finish or cancel it safely first.')
    try:
        verdict = free_comfyui_vram()
    except Exception:
        logger.exception('onetrainer: ComfyUI /free raised unexpectedly')
        verdict = None
    if verdict not in (ComfyVramFreeVerdict.FREED,
                       ComfyVramFreeVerdict.COMFYUI_OFFLINE):
        raise GpuBusyError(
            'ComfyUI did not confirm that its GPU models were released, so '
            'training would start on a card it is still holding. Wait for '
            'ComfyUI to recover (or stop it) and start the run again.')
    logger.info('onetrainer: ComfyUI VRAM release verdict=%s',
                getattr(verdict, 'value', verdict))
    if check_captions:
        assert_trainable(dataset_id, train_type='krea')

    trigger = (ds.trigger_word or '').strip() or f'ds{dataset_id}'
    from .face_dataset_service import FaceDatasetImage as _FDI, db as _db
    num_images = (_db.session.query(_FDI)
                  .filter_by(dataset_id=dataset_id, status='keep').count())
    steps = int(steps) if steps else recommended_steps(dataset_id)
    training_folder = _training_folder_for(ds)
    # Despite the name, export_dataset_to_aitoolkit is generic: `dest_dir` is
    # exactly the "cloud seam" this app's own cloud-training path already uses
    # to export WITHOUT ai-toolkit configured locally — writes kept images as
    # .png/.txt pairs (OneTrainer's 'sample' prompt_source default already
    # matches this layout, see build_concepts). masked=False: this slice's
    # concepts.json carries no mask wiring yet (see spec's Non-goals).
    dataset_folder = export_dataset_to_aitoolkit(
        user_id, dataset_id, masked=False, dest_dir=str(training_folder / 'dataset'))

    peft_type = cfg.get('onetrainer.peft_type') or PEFT_TYPE_LORA
    # The learning rate and the resolution are the app's, not the preset's, and
    # they come from the SAME resolvers the ai-toolkit lane uses — not a second
    # constant that can drift. Without this, the same dataset trained at
    # lr 0.0003 / 1024 here and lr 0.0001 / 768 there, with nothing on screen
    # saying the two lanes disagreed.
    from .lora_training import _effective_resolution, _lr_eff
    try:
        lr = _lr_eff(ds)
    except Exception:                                    # noqa: BLE001
        logger.exception('onetrainer: could not resolve the learning rate; '
                         'leaving it to the shipped preset')
        lr = None
    try:
        res_list = _effective_resolution(ds) or []
        resolution = max(int(r) for r in res_list) if res_list else None
    except Exception:                                    # noqa: BLE001
        logger.exception('onetrainer: could not resolve the resolution; '
                         'falling back to the module default')
        resolution = None
    launched = launch(trigger=trigger, dataset_folder=dataset_folder,
                      training_folder=str(training_folder), steps=steps,
                      num_images=max(1, num_images), rank=32, peft_type=peft_type,
                      learning_rate=lr, resolution=resolution)

    from . import checkpoint_registry
    checkpoint_registry.register_launch(
        user_id, dataset_id, family='krea', source='local', variant='raw',
        masked=False, steps=steps, trainer='onetrainer')
    queue_manager._set_system_state('training_in_progress', True, ttl_seconds=_TRAIN_STATE_TTL)
    queue_manager._set_system_state('training_pid', launched['pid'], ttl_seconds=_TRAIN_STATE_TTL)
    queue_manager._set_system_state('training_dataset_id', int(dataset_id), ttl_seconds=_TRAIN_STATE_TTL)
    queue_manager._set_system_state('training_train_type', 'krea', ttl_seconds=_TRAIN_STATE_TTL)

    from flask import current_app
    import threading
    threading.Thread(target=_watch_onetrainer,
                     args=(current_app._get_current_object(), launched, dataset_id),
                     daemon=True).start()

    return {'started': True, 'pid': launched['pid'], 'config_path': launched['config_path'],
           'steps': steps, 'dataset_folder': dataset_folder,
           'log_path': launched['log_path']}


def _watch_onetrainer(app, launched, dataset_id) -> None:
    """Minimal watcher (see spec's Non-goals: no queue-chaining for
    OneTrainer in this slice, unlike ai-toolkit's _watch_training which
    calls process_training_queue()). Waits for exit, clears the shared
    in-progress flags, reuses _crash_payload for a consistent failure UX."""
    proc = launched['_proc']
    try:
        proc.wait()
        rc = proc.returncode
    except Exception:
        return
    from .lora_training import _crash_payload
    from ..job_queue import queue_manager
    try:
        with app.app_context():
            ok = rc == 0 and checkpoint_ready(launched['output_model_destination'])
            if not ok:
                payload = _crash_payload(launched['log_path'], dataset_id, rc)
                queue_manager._set_system_state('training_error', payload, ttl_seconds=3600)
            queue_manager._set_system_state('training_in_progress', False, ttl_seconds=1)
            queue_manager._set_system_state('training_pid', None, ttl_seconds=1)
    except Exception:
        pass
