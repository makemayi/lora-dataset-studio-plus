"""Fizgig — third LOCAL training backend (MOSS-Image first slice).

Same shape as onetrainer_service: additive and isolated, reached only when a
launch explicitly asks for trainer='fizgig'. A Fizgig run is still a
`source='local'` TrainingRunRecord governed by the same single-training guard,
GPU admission checks and Stop button; only the config/launch step differs.

The recipe is Fizgig's own MOSS-Image **Standard** preset, as run by hand on
joyce001 (2026-10-20: 40 images, 2800 steps, ~3.1 s/step, peak VRAM 10.5 GB on
a 3090) — see docs/guide/moss-lora-training.md:

  - train on the Base (int8 convrot), preview on the SFT;
  - LoRA rank 32 / alpha 32 on every Linear of all 42 blocks, block 22
    INCLUDED (Fizgig measured a likeness loss when it was left out);
  - AdamW, flat lr 1e-4, EMA 0.98, 70 epochs, 1 MP buckets (992²), batch 1;
  - clean-end share 30% (`lownoise_pct=30`, Z mean -0.41 instead of the
    model's own -1.09);
  - frozen text encoder; previews res_multistep 30 steps, CFG 4, Fizgig's
    negative prompt.

What the panel may change on this lane: rank, alpha, learning rate, epochs
(see training_settings_map). Everything else is pinned to the preset.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from .. import config as cfg

logger = logging.getLogger(__name__)

TRAINER = 'fizgig'
FAMILY = 'moss'

# --- the Standard preset ------------------------------------------------------
DEFAULT_RANK = 32
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 70
EPOCHS_RANGE = (1, 400)
EMA_DECAY = 0.98
LOWNOISE_PCT = 30
RESOLUTION = 992                 # 1 MP buckets, what the Standard preset trains at
SEED = 42
# Checkpoints and previews every 5 epochs: every-epoch saves left 50 LoRAs
# (7.4 GB) on the joyce001 run, and the picks (e55/e65/e70) all sat on a
# multiple of 5 anyway.
SAVE_EVERY_EPOCHS = 5
SAMPLE_EVERY_EPOCHS = 5
MAX_SAMPLE_PROMPTS = 3
SAMPLE_SIZE = 1024
SAMPLE_STEPS = 30
SAMPLE_CFG = 4.0
SAMPLE_NEGATIVE = 'worst quality, bad anatomy, watermark, jpeg artifact'

# --- model files, looked up under ComfyUI's models folder ----------------------
# First existing candidate wins. int8 convrot first: it is what the measured
# run used and what fits a 24 GB card with room for previews.
DIT_CANDIDATES = ('diffusion_models/moss_image_base_int8_convrot.safetensors',
                  'diffusion_models/moss_image_base_bf16.safetensors')
PREVIEW_CANDIDATES = ('diffusion_models/moss_image_sft_int8_convrot.safetensors',
                      'diffusion_models/moss_image_sft_bf16.safetensors')
TE_CANDIDATES = ('clip/qwen3.5_9b_int8_convrot.safetensors',
                 'text_encoders/qwen3.5_9b_int8_convrot.safetensors',
                 'clip/qwen3.5_9b_bf16.safetensors',
                 'text_encoders/qwen3.5_9b_bf16.safetensors')
# The small VAE shares the full VAE's encoder, so either one caches the same
# latents; the small one decodes previews faster.
VAE_CANDIDATES = ('vae/flux2-vae_small.safetensors',
                  'vae/flux2-vae.safetensors')

_RUNNER_PATH = Path(__file__).resolve().parent / 'fizgig_runner.py'
LOG_NAME = 'fizgig.log'
META_NAME = 'lds_meta.json'


# --- install ------------------------------------------------------------------

def _derived_python(root: Path) -> Path:
    for env_dir in ('venv', '.venv'):
        p = (root / env_dir / 'Scripts' / 'python.exe' if os.name == 'nt'
             else root / env_dir / 'bin' / 'python')
        if p.is_file():
            return p
    return (root / 'venv' / 'Scripts' / 'python.exe' if os.name == 'nt'
            else root / 'venv' / 'bin' / 'python')


def fizgig_path(kind: str):
    """Same blank-means-unconfigured contract as onetrainer_path."""
    root = (cfg.get('fizgig.dir') or '').strip()
    if not root:
        return None
    root = Path(root)
    if kind == 'dir':
        return root
    if kind == 'venv_python':
        explicit = (cfg.get('fizgig.python') or '').strip()
        return Path(explicit) if explicit else _derived_python(root)
    raise ValueError(f'unknown fizgig_path kind: {kind}')


def is_installed() -> bool:
    """Fizgig usable: venv python present AND the trainer script where it ships."""
    py = fizgig_path('venv_python')
    root = fizgig_path('dir')
    return bool(py and py.is_file() and root
                and (root / 'src' / 'fizgig' / 'families' / 'train.py').is_file())


def _models_dir() -> Path | None:
    explicit = (cfg.get('fizgig.models_dir') or '').strip()
    if explicit:
        return Path(explicit)
    try:
        d = cfg.comfyui_dir('models')
    except Exception:                                    # noqa: BLE001 — advisory
        d = None
    return Path(d) if d else None


def _first_existing(models: Path | None, candidates) -> Path | None:
    if not models:
        return None
    for rel in candidates:
        p = models / rel
        if p.is_file():
            return p
    return None


def resolve_models() -> dict:
    """{'dit','te','vae','preview'} -> Path or None. `preview` is optional."""
    models = _models_dir()
    return {'dit': _first_existing(models, DIT_CANDIDATES),
            'te': _first_existing(models, TE_CANDIDATES),
            'vae': _first_existing(models, VAE_CANDIDATES),
            'preview': _first_existing(models, PREVIEW_CANDIDATES)}


def assert_models_ready() -> dict:
    found = resolve_models()
    missing = [k for k in ('dit', 'te', 'vae') if not found[k]]
    if missing:
        where = _models_dir() or '(ComfyUI models folder not configured)'
        names = {'dit': DIT_CANDIDATES[0], 'te': TE_CANDIDATES[0], 'vae': VAE_CANDIDATES[0]}
        raise ValueError('Fizgig MOSS-Image training needs '
                         + ', '.join(names[k] for k in missing) + f' under {where}')
    return found


# --- settings -----------------------------------------------------------------

def _num(value, cast, lo=None, hi=None):
    try:
        v = cast(value)
    except (TypeError, ValueError):
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return None
    return v


def effective_settings(ds) -> dict:
    """The four panel settings this lane reads, falling back to the preset."""
    from .lora_training import _train_settings
    s = _train_settings(ds) or {}
    rank = _num(s.get('rank'), int, 1, 256) or DEFAULT_RANK
    alpha = _num(s.get('alpha'), int, 1, 256) or rank
    lr = _num(s.get('learning_rate'), float, 1e-7, 1e-2) or DEFAULT_LR
    epochs = _num(s.get('epochs'), int, *EPOCHS_RANGE) or DEFAULT_EPOCHS
    return {'rank': rank, 'alpha': alpha, 'learning_rate': lr, 'epochs': epochs}


# --- run folder ----------------------------------------------------------------

def _runs_root() -> Path:
    explicit = (cfg.get('fizgig.output_dir') or '').strip()
    if explicit:
        return Path(explicit)
    root = fizgig_path('dir')
    return (root / 'output_loras' / 'lds') if root else cfg.data_dir() / 'fizgig_runs'


def training_folder_for(ds) -> Path:
    from .lora_training import _safe_trigger
    return _runs_root() / f'u{ds.user_id}_{_safe_trigger(ds)}'


def output_name_for(ds) -> str:
    from .lora_training import _safe_trigger
    return f'{_safe_trigger(ds)}_fizgig'


def _archive(folder: Path) -> str | None:
    """Every launch trains from zero: move the previous run aside (never delete)."""
    if not folder.exists():
        return None
    dest = folder.with_name(f'{folder.name}_archived_{datetime.now():%Y%m%d-%H%M%S}')
    try:
        os.rename(folder, dest)
    except OSError as e:
        raise ValueError(f'could not archive the previous Fizgig run ({e}) - close '
                         f'anything using "{folder}" and retry')
    return str(dest)


def _toml_str(p) -> str:
    return json.dumps(str(p).replace('\\', '/'))


def build_dataset_toml(image_dir, cache_dir) -> str:
    return (
        '[general]\n'
        f'resolution = [{RESOLUTION}, {RESOLUTION}]\n'
        'caption_extension = ".txt"\n'
        'batch_size = 1\n'
        'num_repeats = 1\n'
        'enable_bucket = true\n'
        'bucket_no_upscale = true\n'
        '\n'
        '[[datasets]]\n'
        f'image_directory = {_toml_str(image_dir)}\n'
        f'cache_directory = {_toml_str(cache_dir)}\n')


def build_plan(*, fizgig_dir, python, toml_path, output_dir, output_name, models,
               settings, prompts_path=None) -> dict:
    """The three stages, as Fizgig's CLI takes them (docs/CLI.md, MOSS-Image).
    Pure — unit-testable without Fizgig installed."""
    py = str(python)
    cache = 'src/fizgig/families/cache.py'
    train = [
        py, 'src/fizgig/families/train.py',
        '--family', 'moss_image',
        '--dataset_config', str(toml_path),
        '--dit', str(models['dit']),
        '--output_dir', str(output_dir),
        '--output_name', output_name,
        '--network_dim', str(settings['rank']),
        '--network_alpha', str(settings['alpha']),
        '--learning_rate', f"{settings['learning_rate']:g}",
        '--max_train_epochs', str(settings['epochs']),
        '--save_every_n_epochs', str(SAVE_EVERY_EPOCHS),
        '--seed', str(SEED),
        '--optimizer_type', 'adamw',
        '--max_grad_norm', '1.0',
        '--precision', 'auto', '--compile_blocks', 'auto',
        '--ema_decay', f'{EMA_DECAY:g}',
        '--family_option', f'lownoise_pct={LOWNOISE_PCT}',
        '--save_state_on_train_end',
        '--vae', str(models['vae']),
        '--text_encoder', str(models['te']),
    ]
    if prompts_path:
        if models.get('preview'):
            train += ['--preview_checkpoint', str(models['preview'])]
        train += ['--sample_prompts', str(prompts_path),
                  '--sample_every_n_epochs', str(SAMPLE_EVERY_EPOCHS),
                  '--sample_width', str(SAMPLE_SIZE), '--sample_height', str(SAMPLE_SIZE),
                  '--sample_steps', str(SAMPLE_STEPS),
                  '--sample_cfg_scale', f'{SAMPLE_CFG:g}',
                  '--sample_negative', SAMPLE_NEGATIVE]
    return {
        'cwd': str(fizgig_dir),
        'env': {'PYTHONPATH': 'src', 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'},
        'stages': [
            {'name': 'cache latents',
             'args': [py, cache, '--family', 'moss_image', '--stage', 'latents',
                      '--dataset_config', str(toml_path), '--model', str(models['vae'])]},
            {'name': 'cache text',
             'args': [py, cache, '--family', 'moss_image', '--stage', 'text',
                      '--dataset_config', str(toml_path), '--model', str(models['te'])]},
            {'name': 'train', 'args': train},
        ],
    }


# --- launch -------------------------------------------------------------------

_ALLOW_FLAGS = ('allow_caption_mismatch', 'allow_uncaptioned',
                'allow_caption_quality', 'allow_not_ready')

def launch_training(user_id, dataset_id, steps=None, check_captions: bool = True,
                    **allow) -> dict:
    """Fizgig counterpart of lora_training.launch_training (MOSS-Image only).

    `steps` is ignored: this lane counts EPOCHS (panel setting, default 70);
    the returned `steps` is the resulting step count (images × epochs).
    `allow` = the confirmable caption/readiness overrides assert_trainable
    takes (allow_uncaptioned, allow_caption_mismatch, ...)."""
    if not is_installed():
        raise RuntimeError('Fizgig is not configured')
    from . import face_dataset_service as fds
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if (ds.train_type or 'zimage') != FAMILY:
        raise ValueError('Fizgig only supports the MOSS-Image family in this build')
    models = assert_models_ready()

    from .lora_training import (_TRAIN_STATE_TTL, _assert_no_vision_pass_on_gpu,
                                _pid_alive, _record_training_process_identity,
                                _safe_trigger, _sample_prompts, assert_free_disk,
                                assert_trainable, export_dataset_to_aitoolkit)
    from ..job_queue import queue_manager
    from ..gpu_window import GpuBusyError
    from ..utils.comfyui import ComfyVramFreeVerdict, free_comfyui_vram

    root = _runs_root()
    root.mkdir(parents=True, exist_ok=True)
    assert_free_disk(root, 10, 'a training run')
    if (queue_manager._get_system_state('training_in_progress', False)
            and _pid_alive(queue_manager._get_system_state('training_pid', None))):
        raise ValueError('a training is already in progress - wait for it to finish '
                         'or queue this dataset')
    # The GPU, same questions the other two lanes ask before taking the card.
    _assert_no_vision_pass_on_gpu()
    if queue_manager.has_comfyui_work():
        raise GpuBusyError(
            'ComfyUI has queued or active work, so local training cannot take the GPU. '
            'Wait for it to finish or cancel it safely first.')
    try:
        verdict = free_comfyui_vram()
    except Exception:                                    # noqa: BLE001
        logger.exception('fizgig: ComfyUI /free raised unexpectedly')
        verdict = None
    if verdict not in (ComfyVramFreeVerdict.FREED, ComfyVramFreeVerdict.COMFYUI_OFFLINE):
        raise GpuBusyError(
            'ComfyUI did not confirm that its GPU models were released, so '
            'training would start on a card it is still holding. Wait for '
            'ComfyUI to recover (or stop it) and start the run again.')
    if check_captions:
        assert_trainable(dataset_id, train_type=FAMILY,
                         **{k: bool(v) for k, v in allow.items() if k in _ALLOW_FLAGS})

    settings = effective_settings(ds)
    folder = training_folder_for(ds)
    archived = _archive(folder)
    folder.mkdir(parents=True)
    image_dir = Path(export_dataset_to_aitoolkit(
        user_id, dataset_id, masked=False, dest_dir=str(folder / 'dataset')))
    num_images = sum(1 for f in image_dir.iterdir()
                     if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))
    if num_images < 1:
        raise ValueError('the exported dataset has no images')
    total_steps = num_images * settings['epochs']

    toml_path = folder / 'dataset.toml'
    toml_path.write_text(build_dataset_toml(image_dir, folder / 'cache'), encoding='utf-8')
    # Same token the exported captions start with (export_dataset_to_aitoolkit).
    prompts = [p for p in (_sample_prompts(ds, _safe_trigger(ds)) or []) if p.strip()]
    prompts = [' '.join(p.split()) for p in prompts[:MAX_SAMPLE_PROMPTS]]
    prompts_path = None
    if prompts:
        prompts_path = folder / 'sample_prompts.txt'
        prompts_path.write_text('\n'.join(prompts) + '\n', encoding='utf-8')
    output_dir = folder / 'output'
    output_name = output_name_for(ds)
    python = fizgig_path('venv_python')
    plan = build_plan(fizgig_dir=fizgig_path('dir'), python=python, toml_path=toml_path,
                      output_dir=output_dir, output_name=output_name, models=models,
                      settings=settings, prompts_path=prompts_path)
    plan_path = folder / 'plan.json'
    plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    (folder / META_NAME).write_text(json.dumps({
        'num_images': num_images, 'epochs': settings['epochs'], 'total_steps': total_steps,
        'output_name': output_name, 'settings': settings,
        'models': {k: (str(v) if v else None) for k, v in models.items()},
        'launched_at': datetime.now().isoformat(timespec='seconds')}, indent=2),
        encoding='utf-8')

    log_path = folder / LOG_NAME
    logf = open(log_path, 'w', encoding='utf-8')
    try:
        proc = subprocess.Popen(
            [str(python), str(_RUNNER_PATH), str(plan_path)],
            cwd=str(fizgig_path('dir')), stdout=logf, stderr=subprocess.STDOUT,
            shell=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except OSError as e:
        logf.close()
        raise ValueError(f'could not start Fizgig: {e}') from e

    from . import checkpoint_registry
    rec = checkpoint_registry.register_launch(
        user_id, dataset_id, family=FAMILY, source='local', variant='',
        masked=False, steps=total_steps, settings={**settings, 'trainer': TRAINER},
        trainer=TRAINER)
    queue_manager._set_system_state('training_error', None, ttl_seconds=1)
    queue_manager._set_system_state('training_in_progress', True, ttl_seconds=_TRAIN_STATE_TTL)
    _record_training_process_identity(proc.pid)
    for key, value in (('training_dataset_id', int(dataset_id)),
                       ('training_train_type', FAMILY),
                       ('training_target_step', int(total_steps)),
                       ('training_variant', ''),
                       ('training_base_model', ''),
                       # Stop kills the process tree (taskkill /T), exactly like
                       # ai-toolkit: Fizgig keeps every saved epoch on disk.
                       ('training_trainer', TRAINER)):
        queue_manager._set_system_state(key, value, ttl_seconds=_TRAIN_STATE_TTL)

    try:
        from flask import current_app
        threading.Thread(target=_watch, daemon=True,
                         args=(current_app._get_current_object(), proc, logf, log_path,
                               output_dir / f'{output_name}.safetensors',
                               user_id, int(dataset_id), getattr(rec, 'id', None))).start()
    except Exception as e:                               # noqa: BLE001
        logger.warning('fizgig watcher not started: %s', e)
    return {'started': True, 'pid': proc.pid, 'config_path': str(plan_path),
            'steps': total_steps, 'epochs': settings['epochs'], 'num_images': num_images,
            'dataset_folder': str(image_dir), 'log_path': str(log_path),
            'archived_run': archived, 'trainer': TRAINER}


def _watch(app, proc, logf, log_path, final_lora: Path, user_id, dataset_id, record_id):
    """Wait for exit; on success deploy the final LoRA into ComfyUI's loras/moss,
    on failure surface the log tail. Then release the training fence."""
    try:
        proc.wait()
        rc = proc.returncode
    except Exception:                                    # noqa: BLE001
        return
    finally:
        try:
            logf.close()
        except OSError:
            pass
    from .lora_training import _clear_training_identity, _crash_payload, import_checkpoint
    from ..job_queue import queue_manager
    try:
        with app.app_context():
            if rc == 0 and final_lora.is_file():
                try:
                    import_checkpoint(user_id, dataset_id, filename=final_lora.name,
                                      src_dir=str(final_lora.parent),
                                      run_id=record_id, run_source='local')
                except Exception:                        # noqa: BLE001
                    logger.exception('fizgig: auto-deploy of %s failed; the run itself '
                                     'is unaffected', final_lora)
                logger.info('Fizgig training dataset %s finished.', dataset_id)
            else:
                payload = _crash_payload(str(log_path), dataset_id, rc)
                queue_manager._set_system_state('training_error', payload, ttl_seconds=3600)
                logger.error('Fizgig training dataset %s ended with rc=%s', dataset_id, rc)
            _clear_training_identity(ttl_seconds=1)
    except Exception:                                    # noqa: BLE001
        logger.exception('fizgig watcher: post-processing failed')


# --- live progress ------------------------------------------------------------

_LOG_MAX_BYTES = 4 * 1024 * 1024
# `<output_name>_e000055_00_<timestamp>_<seed>.png` (Fizgig's sampler)
_SAMPLE_RE = re.compile(r'_e(\d+)_(\d+)_\d+_\d+\.(?:png|jpg|jpeg|webp)$', re.IGNORECASE)
_STAGE_RE = re.compile(r'\[lds-fizgig\] \[(\d+)/(\d+)\] ([^—\r\n]+)')
_MAX_SAMPLES = 12


# Fizgig's tqdm updates land in the log with NO separator between them (no CR,
# no LF - measured on the joyce001 run), so one "line" holds thousands of bars
# and a per-line parser pairs the LAST step with the FIRST loss. Split them.
_BAR_START_RE = re.compile(r'(?<=\])(?=steps:\s)')


def split_bar_updates(text: str) -> str:
    return _BAR_START_RE.sub(chr(10), text or '')


def _read_meta(folder: Path) -> dict:
    try:
        return json.loads((folder / META_NAME).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def list_samples(folder: Path, steps_per_epoch: int) -> list[dict]:
    out = []
    try:
        for f in (folder / 'output' / 'sample').iterdir():
            m = _SAMPLE_RE.search(f.name)
            if m and f.is_file():
                epoch = int(m.group(1))
                out.append({'filename': f.name, 'prompt_idx': int(m.group(2)),
                            'epoch': epoch, 'step': epoch * max(1, steps_per_epoch)})
    except OSError:
        return []
    out.sort(key=lambda s: (s['step'], -s['prompt_idx']), reverse=True)
    return out[:_MAX_SAMPLES]


def sample_path(user_id, dataset_id, filename: str) -> str | None:
    from .face_dataset_service import get_dataset
    ds = get_dataset(user_id, dataset_id)
    if not ds or filename != os.path.basename(filename) or not _SAMPLE_RE.search(filename):
        return None
    p = training_folder_for(ds) / 'output' / 'sample' / filename
    return str(p) if p.is_file() else None


def progress(user_id, dataset_id) -> dict:
    """Live view in the shape the TrainingPanel renders. Never raises on a
    missing log — the first seconds of every run have none."""
    from .face_dataset_service import get_dataset
    from .lora_training import _parse_training_log
    from ..job_queue import queue_manager
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    cur_id = queue_manager._get_system_state('training_dataset_id', None)
    active = (bool(queue_manager._get_system_state('training_in_progress', False))
              and cur_id is not None and int(cur_id) == int(dataset_id))
    folder = training_folder_for(ds)
    meta = _read_meta(folder)
    total = meta.get('total_steps')
    per_epoch = meta.get('num_images') or 1
    parsed = {'step': None, 'total': None, 'loss': None, 'speed': None, 'eta': None,
              'loss_curve': []}
    stage = None
    log_path = folder / LOG_NAME
    log_exists = log_path.is_file()
    if log_exists:
        try:
            size = log_path.stat().st_size
            with open(log_path, encoding='utf-8', errors='replace') as fh:
                if size > _LOG_MAX_BYTES:
                    fh.seek(size - _LOG_MAX_BYTES)
                text = fh.read()
            parsed = _parse_training_log(split_bar_updates(text), expected_total=total)
            for m in _STAGE_RE.finditer(text):
                stage = m.group(3).strip()
        except OSError:
            log_exists = False
    step = parsed.get('step')
    return {'active': active, 'log_exists': log_exists, 'trainer': TRAINER,
            'stage': stage,
            'epoch': (step // per_epoch) if step else None,
            'epochs': meta.get('epochs'),
            'samples': list_samples(folder, per_epoch),
            'download': None, 'cache_pending': None,
            **parsed}
