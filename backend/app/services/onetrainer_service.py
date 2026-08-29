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
import re
import signal
import subprocess
import time
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


# The default this lane trains Krea 2 at when the user has not picked one.
#
# It was 1024, on the reasoning that the shipped preset's 512 was a 16GB-VRAM
# budget rather than a recipe. MEASURED on 2026-08-29 on a 24 GB 3090: 1024 at
# the preset's batch 2 ran ~6.5 s/step with the card at 98.5% full (24208 of
# 24576 MiB) — ~6.5 h for a 90-image character set, with no headroom for
# anything else on the machine. The preset's own 512 is the default again, and
# an explicit resolution from the dataset's settings still wins (see
# launch_training, which forwards ONLY a chosen one).
KREA2_RESOLUTION = 512

# The batch this lane defaults to, which is the app's choice and not the
# preset's (both shipped Krea 2 presets say 2). At 512 the step is ~4x cheaper
# than the 1024 it replaced, so the card has room for a bigger batch, and a
# batch of 4 is what turns that room into throughput instead of idle VRAM.
# `epochs = ceil(steps * batch / images)` reads this too, so a run launched with
# no opinion on either still gets an epoch count that matches its batch.
KREA2_DEFAULT_BATCH_SIZE = 4

# The launcher runs OneTrainer's train.py through this, so that Stop can save.
_SHIM_PATH = Path(__file__).resolve().parent / 'onetrainer_train_shim.py'

# What the shipped Krea 2 presets ask for. Read here rather than parsed at run
# time so the epoch arithmetic has a number even when the preset file is not
# reachable — and asserted against the real file by a test, so it cannot drift
# into a comfortable fiction. Both shipped presets say 2 (diffed 2026-08-17).
KREA2_PRESET_BATCH_SIZE = 2

# This app's scheduler vocabulary is lower-case and ai-toolkit's; OneTrainer's
# is an upper-case enum of its own (modules/util/enum/LearningRateScheduler.py).
# They are NOT the same list, so the value is mapped rather than passed through
# — an unmapped string is not an error in OneTrainer, it is a printed line and a
# run that keeps the default.
#
# `constant_with_warmup` has no OneTrainer member: OneTrainer expresses warmup
# as its own `learning_rate_warmup_steps` field alongside a CONSTANT schedule,
# so that choice maps to the pair rather than to a name.
_SCHEDULER_TO_ONETRAINER = {
    'constant': 'CONSTANT',
    'linear': 'LINEAR',
    'cosine': 'COSINE',
    'cosine_with_restarts': 'COSINE_WITH_RESTARTS',
    'constant_with_warmup': 'CONSTANT',
}
_SCHEDULER_NEEDS_WARMUP = ('constant_with_warmup',)

# The per-lane declaration moved to `training_settings_map`, which names BOTH
# lanes. A one-sided list could not group the panel: "ai-toolkit only" is not
# the complement of "OneTrainer reads it" — several stored settings have no
# control on either lane. These two functions stay as this module's door onto
# it, so existing callers and tests keep working.
from . import training_settings_map as _tsm       # noqa: E402

SETTING_APPLIES = _tsm.APPLIES
SETTING_PINNED = _tsm.PINNED
SETTING_PRESET = _tsm.ABSENT      # this lane's old word for "not read here"

# The rank `launch_training` passes, named so the contract test can check the
# 'pinned' claim in the map against the CALL rather than against a comment.
PINNED_RANK = 32


def setting_status(setting_key):
    """``(state, why)`` for one Advanced-options setting on the OneTrainer
    lane. An unknown key is ABSENT rather than applying: a setting the map has
    never heard of is certainly not one this lane sends."""
    return _tsm.status(setting_key, _tsm.LANE_ONETRAINER)


def settings_status():
    """This lane's slice of the declaration, shaped for the UI."""
    return _tsm.for_lane(_tsm.LANE_ONETRAINER)


def build_job_config(trigger: str, dataset_folder: str, training_folder: str,
                     steps: int, num_images: int, rank: int,
                     peft_type: str = PEFT_TYPE_LORA,
                     learning_rate: float | None = None,
                     resolution: int | None = None,
                     epochs: int | None = None,
                     batch_size: int | None = None,
                     te1_lr: float | None = None,
                     te2_lr: float | None = None,
                     lr_scheduler: str | None = None,
                     warmup_steps: int | None = None,
                     min_snr_gamma: float | None = None,
                     grad_accum: int | None = None,
                     dropout: float | None = None,
                     ema: float | None = None,
                     save_every: int | None = None,
                     save_every_unit: str = 'STEP',
                     sample_every: int | None = None,
                     sample_prompts: list | None = None,
                     base_model_name: str | None = None) -> dict:
    """The OVERRIDE config this app writes to --config-path, merged by
    OneTrainer OVER its own shipped Krea 2 preset (--preset-path). Contains
    ONLY the fields this app's own UI/dataset state actually owns — never a
    field the shipped preset already decided (see the ownership-boundary
    test above).

    `epochs` and `batch_size` are the caller's when it has an opinion, and
    derived when it does not. OneTrainer trains by EPOCH; this app's UI and
    recommended_steps() think in STEPS, and for a long time the gap was papered
    over by deriving `epochs = ceil(steps / images)` and pinning `batch_size`
    to 1 so that arithmetic held. The panel now asks for epochs and batch
    directly on this lane and shows the resulting step count as a label, so the
    approximation is gone wherever the UI is involved.

    The derivation survives for callers with no opinion, and it is now correct
    for a batch above 1: one epoch is `images / batch` optimizer steps, so
    `epochs = ceil(steps * batch / images)`. The old form silently assumed
    batch 1 and would have bought a third of the training it promised at
    batch 3.

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
    # The batch the ARITHMETIC below assumes. When the user has not chosen one
    # this is the shipped preset's own (2 for Krea 2) — writing a 1 over it, as
    # this function used to do unconditionally, silently halved the throughput
    # the preset was tuned around. Same rule as the learning rate: the preset
    # decides until the user does.
    batch = max(1, int(batch_size)) if batch_size else KREA2_DEFAULT_BATCH_SIZE
    epochs_eff = (max(1, int(epochs)) if epochs
                  else max(1, math.ceil(steps * batch / max(1, num_images))))
    training_folder = Path(training_folder)
    if peft_type not in PEFT_TYPES:
        peft_type = PEFT_TYPE_LORA
    out = {
        'workspace_dir': str(training_folder),
        'cache_dir': str(training_folder / 'cache'),
        'output_model_destination': str(training_folder / f'{trigger}.safetensors'),
        # When the shared cache holds the SAME model as a local diffusers
        # directory, point OneTrainer at the LOCAL path instead of the gated
        # repo id (see _local_krea_snapshot for why). Absent -> the preset's
        # own `krea/Krea-2-Raw` stands.
        **({'base_model_name': base_model_name} if base_model_name else {}),
        'epochs': epochs_eff,
        'lora_rank': int(rank),
        'lora_alpha': float(rank),
        # Always written now: the app's default (4) is NOT the preset's (2), so
        # staying silent would train a batch the epoch arithmetic above did not
        # assume — the two must never disagree.
        'batch_size': batch,
        # OneTrainer takes resolution as a STRING (its shipped preset says
        # "512"), unlike ai-toolkit's list of ints. Passing an int here is not a
        # type nit — it is how the two lanes end up training the same dataset at
        # different sizes without anything on screen saying so.
        'resolution': str(int(resolution or KREA2_RESOLUTION)),
        'peft_type': peft_type,
        # The text encoders are NESTED in OneTrainer's schema, and a partial
        # nested dict is safe: BaseConfig.from_dict walks its OWN field table
        # and skips what the incoming dict does not mention (verified in
        # OneTrainer's modules/util/config/BaseConfig.py — the miss lands in an
        # `except` whose else-branch is a bare `pass`). So `train` and the rate
        # arrive without disturbing the preset's weight_dtype or dropout.
        #
        # `train` is written WITH the rate on purpose: the shipped Krea 2 preset
        # freezes the first text encoder (`"text_encoder": {"train": false}`),
        # so a learning rate on its own would be a number attached to a
        # component that never learns — set, stored, and inert.
        #
        # That same `except` is why the KEY NAMES here are pinned by a test: a
        # misspelling is not an error in OneTrainer, it is a line in its log and
        # a run that quietly ignores the setting.
        **({'text_encoder': {'train': True, 'learning_rate': float(te1_lr)}}
           if te1_lr else {}),
        **({'text_encoder_2': {'train': True, 'learning_rate': float(te2_lr)}}
           if te2_lr else {}),
        **({'learning_rate_scheduler': _SCHEDULER_TO_ONETRAINER[lr_scheduler]}
           if lr_scheduler in _SCHEDULER_TO_ONETRAINER else {}),
        # Only for the one choice that MEANS warmup. Attaching it to every
        # schedule would hand OneTrainer a warmup the user never asked for.
        **({'learning_rate_warmup_steps': float(warmup_steps)}
           if (lr_scheduler in _SCHEDULER_NEEDS_WARMUP and warmup_steps) else {}),
        # Min-SNR gamma, in OneTrainer's MODERN shape. Writing the field called
        # `min_snr_gamma` would be silently ignored on this path: over there it
        # is a legacy name that __migration_2 rewrites into the pair below, and
        # migrations only run when `migrate=True` — which train.py sets to
        # `preset_path is None`, and this app always passes a preset. Right
        # name, right value, no effect.
        **({'loss_weight_fn': 'MIN_SNR_GAMMA',
            'loss_weight_strength': float(min_snr_gamma)}
           if min_snr_gamma else {}),
        # grad_accum / dropout / ema, in OneTrainer's OWN top-level fields.
        # `dropout_probability` here is the TOP-LEVEL field (TrainConfig :447,
        # "this is LoRA dropout!") — never the text-encoder caption-dropout
        # field (:266), which the panel does not control. ema is CPU by design:
        # the EMA weights are a rank-32 LoRA, tiny, and this app keeps VRAM off
        # an already-tight 12B run. Each is written only when the user chose a
        # non-default value — 1 (grad accum) / 0 (dropout) / off (ema) means
        # the preset or OneTrainer's own default decides.
        **({'gradient_accumulation_steps': max(1, int(grad_accum))}
           if grad_accum and int(grad_accum) > 1 else {}),
        **({'dropout_probability': float(dropout)} if dropout else {}),
        **({'ema': 'CPU', 'ema_decay': float(ema)}
           if ema in (0.99, 0.999) else {}),
        # Checkpoint / sample frequency. OneTrainer pairs each with a TimeUnit,
        # and the unit MUST be pinned to STEP: its shipped defaults are minutes
        # (and NEVER), so a bare number would be read as epochs or as a unit the
        # user never chose. The `sample_after` key (OneTrainer's name) is what
        # this app's `sample_every` maps to — `save_after` was renamed to
        # `save_every` upstream, `sample_after` never was.
        # The unit is pinned rather than inherited: OneTrainer's shipped defaults
        # are minutes (and NEVER), so a bare number would be read as something
        # the user never chose. STEP unless the caller asked for EPOCH — which
        # this lane trains in, so "save every 20 epochs" is a thing users ask
        # for in exactly those words.
        **({'save_every': int(save_every),
            'save_every_unit': ('EPOCH' if str(save_every_unit).upper() == 'EPOCH'
                                else 'STEP')}
           if save_every else {}),
        # NEVER is written EXPLICITLY when previews are off. Leaving the keys out
        # would inherit the preset's own cadence (10 MINUTE), which is how this
        # lane spent a run sampling on a schedule nobody chose; and an empty
        # prompt list is not the same statement — that is what it did for months
        # while claiming to sample.
        **({'sample_after': float(sample_every), 'sample_after_unit': 'STEP'}
           if sample_every else {'sample_after_unit': 'NEVER'}),
        # WHERE the preview prompts live. OneTrainer's own default points at
        # `training_samples/samples.json` inside its install — a file that ships
        # containing `[]`. So the cadence above was honoured against an EMPTY
        # list: the sampler woke up on schedule, found nothing to render, and
        # every run produced zero previews while the config said it was
        # sampling. Each run now carries its own definitions next to its
        # concepts (see launch), which is also what keeps two datasets from
        # sharing one file.
        # ALWAYS this run's own file, even when previews are off — OneTrainer
        # opens `sample_definition_file_name` unconditionally at startup
        # (TrainConfig.to_pack_dict), so a path that does not exist is a crash
        # before step 1, and an omitted key falls back to the shared file inside
        # the install. `launch` therefore writes it every time, with an empty
        # list when previews are off; `sample_after_unit: NEVER` above is what
        # actually turns them off.
        'sample_definition_file_name': str(training_folder / 'samples.json'),
        # The app owns the learning rate: it is a per-dataset setting the UI
        # exposes and the ai-toolkit lane already honours. Left unset, this run
        # silently used the shipped preset's 0.0003 while the SAME dataset
        # trained at 0.0001 on ai-toolkit — a 3x divergence with no indication
        # anywhere. Only written when the caller resolved one, so the preset
        # still decides for any path that has no opinion.
        **({'learning_rate': float(learning_rate)} if learning_rate else {}),
    }
    return out


def build_samples(prompts, resolution: int | None = None) -> list[dict]:
    """OneTrainer's `samples.json`: one entry per preview prompt.

    Only the fields this app has an opinion about are written — the rest
    (diffusion_steps, cfg_scale, scheduler, seed) come from OneTrainer's own
    per-model defaults, the same ownership boundary build_job_config keeps.

    The exception is the SIZE. Krea 2's sample default is 1024x1024 while this
    lane now trains at 512, and a preview is rendered with the training weights
    resident: sampling two octaves above the training size is the one field
    where an inherited default can cost the run an OOM three hours in. Previews
    are therefore rendered at the size the run trains at.
    """
    px = int(resolution or KREA2_RESOLUTION)
    out = []
    for line in prompts or []:
        if not isinstance(line, str) or not line.strip():
            continue
        out.append({'enabled': True, 'prompt': line.strip(),
                    'width': px, 'height': px})
    return out


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


def deploy_onetrainer_checkpoint(user_id, dataset_id, record_id, output_path):
    """On a successful run, deploy the output into the dataset's ComfyUI loras
    folder so the Studio / Canvas picker sees it, via the SAME import_checkpoint
    every other local/cloud checkpoint deploys with (family subfolder + a run
    tag that links the deployed name back to this TrainingRunRecord).

    ``output_path`` is the app-side model destination (the onetrainer_runs file);
    its directory is passed as ``src_dir`` so no ai-toolkit install is needed.
    FAIL-SAFE and deliberately non-blocking: a deploy problem (ComfyUI
    unconfigured, an arch-guard refusal, a full disk) is logged and swallowed —
    it must never turn an already-finished run into an error."""
    from .lora_training import import_checkpoint
    name = os.path.basename(output_path)
    run_dir = os.path.dirname(output_path)
    try:
        import_checkpoint(
            user_id, dataset_id, filename=name, src_dir=run_dir,
            run_id=record_id, run_source='local')
    except Exception:                                    # noqa: BLE001
        logger.exception('onetrainer: auto-deploy of %s failed; the run itself '
                         'is unaffected', name)


def _local_krea_snapshot() -> str | None:
    """The shared cache's Krea-2-Raw snapshot as a LOCAL diffusers directory.

    OneTrainer's preset names the model `krea/Krea-2-Raw`, which makes
    huggingface_hub resolve it — and a gated repo answers the HEAD that
    resolution performs with 401 EVEN when every component is already on disk
    (measured repeatedly on this machine). Pointing `base_model_name` at the
    local snapshot instead makes `from_pretrained(<local dir>, subfolder=...)`
    purely local: no HEAD, no network, no gated check. Returns None when the
    snapshot is not present, in which case the preset's name stands (and the
    run behaves exactly as it always has)."""
    try:
        hub = Path(cfg.aitoolkit_path('hf_home')) / 'hub'
    except Exception:
        return None
    repo = hub / 'models--krea--Krea-2-Raw'
    try:
        rev = (repo / 'refs' / 'main').read_text(encoding='utf-8').strip()
    except OSError:
        return None
    snap = repo / 'snapshots' / rev if rev else None
    if not snap or not snap.is_dir():
        return None
    # A diffusers layout is present when the familiar subfolders are.
    for sub in ('tokenizer', 'text_encoder', 'transformer', 'vae'):
        if not (snap / sub).is_dir():
            return None
    return str(snap)


def launch(trigger: str, dataset_folder: str, training_folder: str,
          steps: int, num_images: int, rank: int,
          peft_type: str = PEFT_TYPE_LORA,
          learning_rate: float | None = None,
          resolution: int | None = None,
          epochs: int | None = None,
          batch_size: int | None = None,
          te1_lr: float | None = None,
          te2_lr: float | None = None,
          lr_scheduler: str | None = None,
          warmup_steps: int | None = None,
          min_snr_gamma: float | None = None,
          grad_accum: int | None = None,
          dropout: float | None = None,
          ema: float | None = None,
          save_every: int | None = None,
          save_every_unit: str = 'STEP',
          sample_every: int | None = None,
          sample_prompts: list | None = None) -> dict:
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

    local_krea = _local_krea_snapshot()
    if local_krea:
        logger.info('onetrainer: base model resolved to the local snapshot %s',
                    local_krea)
    config = build_job_config(trigger=trigger, dataset_folder=dataset_folder,
                              training_folder=training_folder, steps=steps,
                              num_images=num_images, rank=rank, peft_type=peft_type,
                              base_model_name=local_krea,
                              learning_rate=learning_rate, resolution=resolution,
                              epochs=epochs, batch_size=batch_size,
                              te1_lr=te1_lr, te2_lr=te2_lr,
                              lr_scheduler=lr_scheduler, warmup_steps=warmup_steps,
                              min_snr_gamma=min_snr_gamma,
                              grad_accum=grad_accum, dropout=dropout, ema=ema,
                              save_every=save_every, save_every_unit=save_every_unit,
                              sample_every=sample_every,
                              sample_prompts=sample_prompts)
    concepts = build_concepts(trigger=trigger, dataset_folder=dataset_folder)

    concepts_path = training_folder_p / 'concepts.json'
    config_path = training_folder_p / 'config.json'
    concepts_path.write_text(json.dumps(concepts, indent=2), encoding='utf-8')
    # Written unconditionally: see build_job_config's note on why the config
    # always names this file. Empty list = previews off, stated in the run's own
    # folder rather than inherited from whatever the install happens to hold.
    samples = build_samples(sample_prompts, resolution=resolution) if sample_every else []
    (training_folder_p / 'samples.json').write_text(
        json.dumps(samples, indent=2), encoding='utf-8')
    config_with_concepts = {**config, 'concept_file_name': str(concepts_path)}
    config_path.write_text(json.dumps(config_with_concepts, indent=2), encoding='utf-8')

    preset_rel = krea2_preset_relative_path()
    preset_path = root / preset_rel
    logger.info('onetrainer: using shipped preset %s', preset_rel)
    log_path = training_folder_p / 'onetrainer.log'
    logf = open(log_path, 'w', encoding='utf-8')
    # The Krea 2 base model is a HF GATED repo: downloading it without a
    # granted token answers 401 (measured on this machine, 2026-08-29). The
    # copy already on disk lives in the SHARED HF cache the ai-toolkit lane
    # uses, so the child gets the same HF_HOME — from_pretrained(
    # 'krea/Krea-2-Raw') then resolves from that cache instead of the network,
    # granted or not. When ai-toolkit is unconfigured the child inherits the
    # parent environment, so a OneTrainer-only install behaves exactly as
    # before. PYTHONUNBUFFERED matters here for the same reason it does on the
    # ai-toolkit lane: stdout goes to a FILE, and block-buffering would make
    # "is it moving?" unanswerable for an hour.
    env = dict(os.environ)
    hf_home = cfg.aitoolkit_path('hf_home')
    if hf_home:
        env['HF_HOME'] = str(hf_home)
        env['HF_HUB_DOWNLOAD_TIMEOUT'] = os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT', '30')
        # The model is ALREADY in the shared cache (measured: the Krea-2-Raw
        # snapshot is complete and 57 GB). huggingface_hub still does a HEAD
        # round-trip per file to verify etags, and a gated repo answers that
        # HEAD with 401 even when every byte is local — so the run fails
        # before it ever reads disk. Pin OFFLINE: from_pretrained then
        # resolves straight from the cache and never touches the network. A
        # cache that is missing something fails loudly rather than silently
        # downloading, which is the honest behaviour here.
        env.setdefault('HF_HUB_OFFLINE', '1')
        env.setdefault('TRANSFORMERS_OFFLINE', '1')
    token = (cfg.secret('HF_TOKEN') or '').strip()
    if token:
        env['HF_TOKEN'] = token
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    proc = subprocess.Popen(
        # OneTrainer's actual argparse flags are HYPHENATED (--preset-path,
        # --config-path), confirmed against train.py's own usage output —
        # the underscored form silently fails argparse and the process
        # exits before doing anything (issue found running Krea 2 for real).
        #
        # train.py is reached THROUGH the shim so that a stop can be graceful:
        # OneTrainer saves the LoRA on a KeyboardInterrupt and never sees one
        # from a taskkill. See onetrainer_train_shim for the whole reason.
        [str(venv_python), str(_SHIM_PATH),
         '--preset-path', str(preset_path), '--config-path', str(config_path)],
        cwd=str(root), stdout=logf, stderr=subprocess.STDOUT, shell=False,
        env=env,
        # CREATE_NEW_PROCESS_GROUP is what makes the child addressable by a
        # console control event at all; without it CTRL_BREAK would be
        # broadcast to this app's own process group — i.e. to the server.
        creationflags=(getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                       | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)))
    return {'pid': proc.pid, 'config_path': str(config_path),
            'concepts_path': str(concepts_path), 'log_path': str(log_path),
            'output_model_destination': config['output_model_destination'],
            '_proc': proc}


# How long the STOP call itself waits for the trainer to finish saving before
# it answers. The save is a backup (training state) followed by the LoRA, and
# `end()` also evicts the model from VRAM first — tens of seconds on a 12B base.
# The caller is an HTTP request, so this is deliberately short: past it the run
# is still exiting, the watcher still finalises it, and the answer says so
# rather than pretending the run is gone.
GRACEFUL_STOP_WAIT_SECONDS = 25.0
_GRACEFUL_POLL_SECONDS = 0.5


def request_graceful_stop(pid) -> bool:
    """Ask a running OneTrainer child to stop the way its own cancel path
    expects: a console control event the shim turns into a KeyboardInterrupt.

    Returns whether the event was DELIVERED, not whether the run has ended —
    the trainer then writes its backup and saves the LoRA, which takes as long
    as it takes. False means the caller should fall back to killing.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    sig = getattr(signal, 'CTRL_BREAK_EVENT', None)
    if sig is None:                                  # POSIX: SIGINT is the same ask
        sig = signal.SIGINT
    try:
        os.kill(pid, sig)
        return True
    except (OSError, ValueError, ProcessLookupError) as exc:
        logger.warning('onetrainer: could not ask pid %s to stop gracefully: %s',
                       pid, exc)
        return False


def wait_for_exit(pid, timeout_seconds: float = GRACEFUL_STOP_WAIT_SECONDS) -> bool:
    """True once the process is gone. Never treats an unreadable state as death:
    the GPU fence is released on this answer."""
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        try:
            import psutil
            if not psutil.pid_exists(int(pid)):
                return True
            if psutil.Process(int(pid)).status() == psutil.STATUS_ZOMBIE:
                return True
        except Exception as exc:                     # noqa: BLE001
            try:
                import psutil
                if isinstance(exc, psutil.NoSuchProcess):
                    return True
            except Exception:                        # noqa: BLE001
                pass
            logger.warning('onetrainer: could not probe pid %s while stopping: %s',
                           pid, exc)
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(_GRACEFUL_POLL_SECONDS)


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
    from .lora_training import (_effective_resolution, _resolution_is_explicit,
                               _train_settings, _lora_rank, _sample_every,
                               _sample_prompts)
    _s = _train_settings(ds) or {}
    # ONLY when the user chose one. `_lr_eff` never returns None — it falls back
    # to the family-fixed 1e-4 — so calling it here wrote `learning_rate` on
    # EVERY run and silently overrode the shipped preset's own 0.0003. Three
    # times off, chosen by OneTrainer's maintainers for this model, replaced by
    # a default this app picked for a different trainer, with nothing on screen
    # saying so. The ownership rule at the top of build_job_config already said
    # not to touch a field the preset decided; this is that rule applied.
    try:
        lr = _s.get('learning_rate')
        lr = float(lr) if isinstance(lr, (int, float)) and lr > 0 else None
    except Exception:                                    # noqa: BLE001
        logger.exception('onetrainer: could not resolve the learning rate; '
                         'leaving it to the shipped preset')
        lr = None
    # ONLY a resolution the user actually picked. `_effective_resolution` never
    # returns nothing — it falls back to the ai-toolkit family default
    # (768+1024), whose LARGEST value used to become this lane's resolution on
    # every run that had never touched the setting. That is how a Krea 2 run
    # nobody configured ended up at 1024. Unset now means KREA2_RESOLUTION.
    try:
        if _resolution_is_explicit(ds):
            res_list = _effective_resolution(ds) or []
            resolution = max(int(r) for r in res_list) if res_list else None
        else:
            resolution = None
    except Exception:                                    # noqa: BLE001
        logger.exception('onetrainer: could not resolve the resolution; '
                         'falling back to the module default')
        resolution = None
    # Epochs and batch are OneTrainer's OWN vocabulary and the panel now asks
    # for them in it, so they come straight from the dataset's settings. Unset
    # means "no opinion" and build_job_config derives them from `steps`, which
    # is how a run launched from anywhere but that panel still works.
    ot_epochs = _s.get('epochs')
    ot_batch = _s.get('batch_size')
    launched = launch(trigger=trigger, dataset_folder=dataset_folder,
                      training_folder=str(training_folder), steps=steps,
                      num_images=max(1, num_images),
                      rank=_lora_rank(ds, 'krea'), peft_type=peft_type,
                      learning_rate=lr, resolution=resolution,
                      epochs=ot_epochs, batch_size=ot_batch,
                      te1_lr=_s.get('te1_lr'), te2_lr=_s.get('te2_lr'),
                      lr_scheduler=_s.get('lr_scheduler'),
                      warmup_steps=_s.get('warmup'),
                      min_snr_gamma=_s.get('min_snr_gamma'),
                      grad_accum=_s.get('grad_accum'),
                      dropout=_s.get('dropout'),
                      ema=_s.get('ema'),
                      save_every=(_s.get('save_epochs') or _s.get('save_every')),
                      save_every_unit=('EPOCH' if _s.get('save_epochs') else 'STEP'),
                      # RESOLVED, not raw: `_sample_every` falls back to the same
                      # 250 steps the ai-toolkit lane uses, and `_sample_prompts`
                      # to the kind's own defaults with the trigger injected. Read
                      # raw, both were None on every dataset whose panel had never
                      # been touched — which is every dataset — so the lane asked
                      # for no previews at all.
                      # `sample_every` 0 means OFF — an explicit choice this lane
                      # can express because it is the one that renders previews
                      # with the training weights resident, and on a 24 GB card a
                      # run at 1024 does not always have room for one.
                      sample_every=(None if _s.get('sample_every') == 0
                                    else _sample_every(ds)),
                      sample_prompts=_sample_prompts(ds, trigger))

    from . import checkpoint_registry
    rec = checkpoint_registry.register_launch(
        user_id, dataset_id, family='krea', source='local', variant='raw',
        masked=False, steps=steps, trainer='onetrainer')
    queue_manager._set_system_state('training_in_progress', True, ttl_seconds=_TRAIN_STATE_TTL)
    # PID **plus birth time**, the same identity the ai-toolkit lane records.
    # Writing the bare pid was enough to SHOW a run, and not enough to STOP one:
    # `stop_training` probes `_pid_alive`, which needs `training_pid_create_time`
    # to tell "still the training child" from "Windows reused that pid". With the
    # birth time absent the probe answers None, and the fail-closed rule turns
    # that into a refusal — so the Stop button could never kill a OneTrainer run
    # (observed 2026-08-29: 'training pid ... lacks a durable birth-time
    # identity', the run had to be killed with taskkill by hand).
    from .lora_training import _record_training_process_identity
    _record_training_process_identity(launched['pid'])
    queue_manager._set_system_state('training_dataset_id', int(dataset_id), ttl_seconds=_TRAIN_STATE_TTL)
    queue_manager._set_system_state('training_train_type', 'krea', ttl_seconds=_TRAIN_STATE_TTL)
    # WHOSE run this is. Stop asks, because the two lanes are stopped
    # differently: ai-toolkit is killed, OneTrainer is ASKED — it saves the LoRA
    # on the way out and a kill throws that away.
    queue_manager._set_system_state('training_trainer', 'onetrainer',
                                    ttl_seconds=_TRAIN_STATE_TTL)

    from flask import current_app
    import threading
    threading.Thread(target=_watch_onetrainer,
                     args=(current_app._get_current_object(), launched, dataset_id,
                           user_id, rec.id),
                     daemon=True).start()

    return {'started': True, 'pid': launched['pid'], 'config_path': launched['config_path'],
           'steps': steps, 'dataset_folder': dataset_folder,
           'log_path': launched['log_path']}


def _watch_onetrainer(app, launched, dataset_id, user_id, record_id) -> None:
    """Minimal watcher (see spec's Non-goals: no queue-chaining for
    OneTrainer in this slice, unlike ai-toolkit's _watch_training which
    calls process_training_queue()). Waits for exit, clears the shared
    in-progress flags, reuses _crash_payload for a consistent failure UX, and
    on success auto-deploys the checkpoint into the ComfyUI loras root so the
    Studio / Canvas picker can generate from it (see
    deploy_onetrainer_checkpoint)."""
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
            if ok:
                deploy_onetrainer_checkpoint(
                    user_id, dataset_id, record_id,
                    launched['output_model_destination'])
            else:
                payload = _crash_payload(launched['log_path'], dataset_id, rc)
                queue_manager._set_system_state('training_error', payload, ttl_seconds=3600)
            queue_manager._set_system_state('training_in_progress', False, ttl_seconds=1)
            queue_manager._set_system_state('training_pid', None, ttl_seconds=1)
            # The birth time is half of that identity; leaving it behind would
            # pair a stale creation time with the NEXT run's pid.
            queue_manager._set_system_state('training_pid_create_time', None,
                                            ttl_seconds=1)
    except Exception:
        pass


# --- live progress ------------------------------------------------------------
#
# WHY THIS LANE NEEDS ITS OWN PARSER, rather than reusing
# `lora_training._parse_training_log`:
#
#   . that parser takes the run's CONFIGURED step count and accepts ONLY a tqdm
#     bar counting to it - the rule that stopped a quantization bar ("28/28")
#     being shown as training progress;
#   . OneTrainer prints TWO bars and neither counts to that number. `epoch:`
#     counts epochs (0/156) and `step:` restarts every epoch (13/22). Fed to the
#     step parser they are correctly rejected, which is why this lane showed
#     "Starting up..." from the first second of a run to the last.
#
# So the two bars are read as the pair they are: the global step is
# `epoch * steps_per_epoch + step`, and the total is `epochs * steps_per_epoch`.
# `steps_per_epoch` comes from the step bar itself rather than from
# images/batch - the trainer's own count already accounts for a dropped last
# batch (90 images at batch 4 prints 22, not 22.5).
#
# The log is opened 'w' at every launch, so a tail is always THIS run.
_OT_EPOCH_RE = re.compile(r'\bepoch:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^<\]]*)<([^,\]]*)')
_OT_STEP_RE = re.compile(r'\bstep:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^<\]]*)<([^,\]]*),\s*([^,\]]+)')
_OT_LOSS_RE = re.compile(r'\bloss=([0-9.eE+-]+)')
_OT_LOG_MAX_BYTES = 4 * 1024 * 1024
_OT_CURVE_MAX_POINTS = 200


def _ot_hms(seconds: float) -> str:
    """tqdm's own remaining-time format, so a derived ETA reads like a printed
    one: H:MM:SS above an hour, MM:SS below it."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f'{h}:{m:02d}:{sec:02d}' if h else f'{m:02d}:{sec:02d}'


def parse_progress(text: str, expected_epochs: int | None = None) -> dict:
    """Pure: the two OneTrainer bars -> the same shape the panel already reads
    from the ai-toolkit lane ({step, total, loss, speed, eta, loss_curve}) plus
    the epoch pair it can show alongside.

    `expected_epochs` is the config's own `epochs`. Given, an epoch bar counting
    to anything else is not this run's and is ignored - the same defence the
    step parser has, expressed in this lane's vocabulary."""
    out = {'step': None, 'total': None, 'loss': None, 'speed': None, 'eta': None,
           'loss_curve': [], 'epoch': None, 'epochs': None}
    epoch = epochs = None
    epoch_eta = None
    step = steps_per_epoch = None
    curve = []
    for seg in re.split(r'[\r\n]+', text or ''):
        em = None
        for em in _OT_EPOCH_RE.finditer(seg):
            pass
        if em:
            cur, tot = int(em.group(1)), int(em.group(2))
            if tot > 0 and cur <= tot and not (expected_epochs and tot != expected_epochs):
                epoch, epochs = cur, tot
                remaining = (em.group(4) or '').strip()
                # tqdm prints '?' until it has timed one epoch.
                epoch_eta = remaining if remaining and '?' not in remaining else None
        sm = None
        for sm in _OT_STEP_RE.finditer(seg):
            pass
        if not sm:
            continue
        cur, tot = int(sm.group(1)), int(sm.group(2))
        if tot <= 0 or cur > tot:
            continue
        step, steps_per_epoch = cur, tot
        out['speed'] = (sm.group(5) or '').strip() or None
        lm = _OT_LOSS_RE.search(seg)
        if lm:
            try:
                out['loss'] = float(lm.group(1))
            except ValueError:
                pass
            else:
                pos = (epoch or 0) * tot + cur
                if not curve or curve[-1][0] != pos:
                    curve.append([pos, out['loss']])
    if step is not None and steps_per_epoch:
        out['step'] = (epoch or 0) * steps_per_epoch + step
        if epochs:
            out['total'] = epochs * steps_per_epoch
    out['epoch'], out['epochs'] = epoch, epochs
    # The epoch bar's own remaining time is the whole run's, so it is preferred
    # over anything derived. Early on it is '?' - then the step rate answers the
    # same question, and saying nothing until tqdm has timed a full epoch would
    # be a blank ETA for the first several minutes of every run.
    out['eta'] = epoch_eta
    if not out['eta'] and out['speed'] and out['step'] and out['total']:
        m = re.match(r'([0-9.]+)\s*(s/it|it/s)', out['speed'])
        if m:
            try:
                v = float(m.group(1))
            except ValueError:
                v = 0.0
            per_step = v if m.group(2) == 's/it' else (1.0 / v if v else 0.0)
            if per_step:
                out['eta'] = _ot_hms((out['total'] - out['step']) * per_step)
    if len(curve) > _OT_CURVE_MAX_POINTS:
        stride = len(curve) / _OT_CURVE_MAX_POINTS
        curve = [curve[int(i * stride)] for i in range(_OT_CURVE_MAX_POINTS - 1)] + [curve[-1]]
    out['loss_curve'] = curve
    return out


# `<workspace>/samples/<i> - <safe prompt>/<prefix><timestamp>-training-sample-
#  <global step>-<epoch>-<epoch step>.jpg` — read from OneTrainer's own
# GenericTrainer, not guessed. The step is the number the panel labels the
# thumbnail with; the folder index is which prompt it came from.
_OT_SAMPLE_FILE_RE = re.compile(r'-training-sample-(\d+)-(\d+)-(\d+)$')
_OT_SAMPLE_DIR_RE = re.compile(r'^(\d+) - ')
_OT_SAMPLE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')
_OT_MAX_SAMPLES = 12


def list_samples(training_folder: Path) -> list[dict]:
    """The preview images this run has written, newest first.

    Flat `filename` on purpose: OneTrainer nests one folder per prompt, but the
    panel addresses a sample by basename through a route that refuses
    separators. The basename carries the step and is unique per prompt+step, so
    flattening loses nothing and keeps ONE sample URL shape for both lanes."""
    root = training_folder / 'samples'
    out = []
    try:
        for sub in root.iterdir():
            if not sub.is_dir() or sub.name == 'custom':
                continue
            dm = _OT_SAMPLE_DIR_RE.match(sub.name)
            prompt_idx = int(dm.group(1)) if dm else 0
            for f in sub.iterdir():
                if f.suffix.lower() not in _OT_SAMPLE_EXTS or not f.is_file():
                    continue
                fm = _OT_SAMPLE_FILE_RE.search(f.stem)
                if not fm:
                    continue
                out.append({'filename': f.name, 'prompt_idx': prompt_idx,
                            'step': int(fm.group(1))})
    except OSError:
        return []
    out.sort(key=lambda s: (s['step'], s['prompt_idx']), reverse=True)
    return out[:_OT_MAX_SAMPLES]


def sample_path(user_id, dataset_id, filename: str) -> str | None:
    """Resolve one preview image by BASENAME inside this run's samples tree.

    The caller has already refused separators, so the name cannot escape; this
    only has to find which prompt folder holds it."""
    from .face_dataset_service import get_dataset
    ds = get_dataset(user_id, dataset_id)
    if not ds or filename != os.path.basename(filename):
        return None
    root = _training_folder_for(ds) / 'samples'
    try:
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            cand = sub / filename
            if cand.is_file():
                return str(cand)
    except OSError:
        return None
    return None


def _configured_epochs(training_folder: Path) -> int | None:
    try:
        with open(training_folder / 'config.json', encoding='utf-8') as fh:
            v = json.load(fh).get('epochs')
        return int(v) if v else None
    except (OSError, ValueError, TypeError):
        return None


def progress(user_id, dataset_id) -> dict:
    """The live view of a OneTrainer run, in the shape the TrainingPanel already
    renders. Never raises on a missing/unreadable log: a run that has not
    written yet is the normal first seconds of every launch."""
    from .face_dataset_service import get_dataset
    from ..job_queue import queue_manager
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    cur_id = queue_manager._get_system_state('training_dataset_id', None)
    active = (bool(queue_manager._get_system_state('training_in_progress', False))
              and cur_id is not None and int(cur_id) == int(dataset_id))
    training_folder = _training_folder_for(ds)
    log_path = training_folder / 'onetrainer.log'
    parsed = {'step': None, 'total': None, 'loss': None, 'speed': None,
              'eta': None, 'loss_curve': [], 'epoch': None, 'epochs': None}
    log_exists = log_path.is_file()
    if log_exists:
        try:
            size = log_path.stat().st_size
            with open(log_path, encoding='utf-8', errors='replace') as fh:
                if size > _OT_LOG_MAX_BYTES:
                    fh.seek(size - _OT_LOG_MAX_BYTES)
                text = fh.read()
            parsed = parse_progress(
                text, expected_epochs=_configured_epochs(training_folder))
        except OSError:
            log_exists = False
    return {'active': active, 'log_exists': log_exists, 'trainer': 'onetrainer',
            'samples': list_samples(training_folder),
            # Nothing is pulled from Hugging Face at run time (the base resolves
            # from the shared cache, see launch), so these two are stated as
            # empty rather than left out — the panel reads them on every payload.
            'download': None, 'cache_pending': None,
            **parsed}
