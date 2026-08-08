"""Deliver a full-model (dense) run to the LOCAL disk, not only to Hugging Face.

WHY THIS EXISTS
---------------
A dense Krea 2 run used to have exactly one delivery address: a PRIVATE Hugging
Face repository the pod pushes to while it trains. That address has a ceiling
nobody controls — run #146 died 250 steps from the end on
``403 Private repository storage limit reached`` and had to be rescued by
deleting 50 GB by hand. A model the user paid eight hours of GPU for must not
depend on somebody else's free-tier quota.

The pod already knows how to hand its files back: that is exactly how every LoRA
checkpoint reaches the user (``/api/jobs/<id>/files`` to list,
``/api/files/<path>`` to stream, with Range resume). Dense was excluded from that
path by a handful of ``if _is_full_transformer_run(run): return`` guards, not by
anything technical. This module is the dense half of the same harvest.

WHAT IT GUARANTEES
------------------
* **Room is checked BEFORE anything starts.** 26 GB (or the size a past run
  really delivered) + the fp8 twin + a margin, against the free space of the
  volume holding the checkpoint store. A refusal is confirmable, exactly like
  the Hugging Face storage one — the numbers are estimates and an estimate must
  never lock someone out of their own paid GPU.
* **The pod is destroyed only after the local file is PROVEN.** Proven means:
  the byte count matches what the pod advertised, and the safetensors header
  re-reads and declares tensors. A rename that lands a 0-byte file, a proxy that
  ends a stream early, a disk that filled up mid-transfer — all three answer the
  same way here, and none of them may cost the pod.
* **Nothing about existing runs changes.** A run with no stamped delivery mode
  is a Hugging Face run and keeps every behaviour it had.
"""
import logging
import os
import re

logger = logging.getLogger(__name__)

# Delivery modes. STORED in train_params and in config.json — never rename these
# strings without an alias table (a stamped run reads its own value forever).
LOCAL = 'local'
HUB = 'hub'
BOTH = 'both'
MODES = (LOCAL, HUB, BOTH)
# Both, and in THAT order: the local copy is harvested and proven first, then
# the Hub backup is attempted from the pod. The Hub copy is what makes a run
# resumable later, so dropping it by default would take away a capability the
# user only discovers missing when they try to continue; and doing it SECOND is
# what stops a full private quota from ever costing a run again.
DEFAULT_MODE = BOTH

# A run stamped before this feature existed. Its artifact only ever lived in the
# private Hugging Face repository, so that is what it must keep meaning.
LEGACY_MODE = HUB

# Confirmable-refusal marker (frontend: utils/trainingRefusals.js). The
# window.confirm IS the answer and the retry carries `allow_local_disk`.
DISK_REFUSAL_MARKER = 'LOCAL_DISK_FULL: '

# Free space to leave on the volume on top of the delivery itself. A drive that
# ends the run at 0 bytes free is a broken machine, not a successful download.
DEFAULT_DISK_MARGIN_GB = 15

# The fp8 twin is written next to the master with this suffix (fp8_export
# .fp8_name_for). Telling them apart matters: only the master can be trained
# again, and only the twin is worth loading in ComfyUI.
FP8_SUFFIX = '_fp8.safetensors'


def normalize_mode(value, default=DEFAULT_MODE) -> str:
    """One of MODES. Anything unrecognised falls back to ``default`` — a typo in
    config.json must not silently disable a delivery."""
    text = str(value or '').strip().lower()
    if text in MODES:
        return text
    # Historical/likely spellings, mapped rather than refused.
    if text in ('disk', 'local_disk', 'computer'):
        return LOCAL
    if text in ('huggingface', 'hf'):
        return HUB
    return default


def delivers_local(mode) -> bool:
    return normalize_mode(mode) in (LOCAL, BOTH)


def delivers_hub(mode) -> bool:
    return normalize_mode(mode) in (HUB, BOTH)


def configured_mode(cloud_cfg=None) -> str:
    """The delivery a NEW dense launch gets, from config."""
    if cloud_cfg is None:
        from .. import config as cfg
        cloud_cfg = cfg.get('cloud') or {}
    dense = (cloud_cfg or {}).get('full_transformer') or {}
    return normalize_mode(dense.get('delivery'), DEFAULT_MODE)


def run_mode(params) -> str:
    """The delivery mode of ONE run, from its stamped params.

    An unstamped run predates this feature: it delivered to Hugging Face and
    nothing else, and every surface must keep reading it that way."""
    source = params if isinstance(params, dict) else {}
    return normalize_mode(source.get('dense_delivery'), LEGACY_MODE)


# --- local disk accounting ------------------------------------------------------

def disk_margin_bytes(cloud_cfg=None) -> int:
    from .hf_storage import GB
    if cloud_cfg is None:
        from .. import config as cfg
        cloud_cfg = cfg.get('cloud') or {}
    dense = (cloud_cfg or {}).get('full_transformer') or {}
    try:
        gb = float(dense.get('local_disk_margin_gb', DEFAULT_DISK_MARGIN_GB))
    except (TypeError, ValueError):
        gb = DEFAULT_DISK_MARGIN_GB
    return int(max(0, gb) * GB)


def local_delivery_forecast(keeps=1, fp8_export=True, keep_bf16=True,
                            _free=None, _checkpoint=None) -> dict:
    """What ONE dense delivery will occupy locally, and whether it fits.

    Same shape and the same tri-state ``fits`` as the Hugging Face forecast
    (``None`` = could not be measured, which NEVER blocks): the two guards read
    identically on purpose, because they answer the same question about two
    different disks.
    """
    from .. import config as cfg
    from . import hf_storage, storage_locations
    from .fp8_export import estimate_fp8_bytes

    keeps = max(1, int(keeps or 1))
    if _checkpoint is None:
        checkpoint, size_source = hf_storage.dense_checkpoint_bytes()
    else:
        checkpoint, size_source = int(_checkpoint), 'provided'
    fp8 = estimate_fp8_bytes(checkpoint) if fp8_export else 0
    # keep_bf16=False means the pod DROPS the master once the twin exists, so
    # only the twin is ever harvested. The master is still downloaded first in
    # the keep case, hence keeps × checkpoint.
    master = checkpoint * keeps if keep_bf16 else 0
    margin = disk_margin_bytes()
    needed = master + fp8 + margin
    root = str(cfg.checkpoints_root(create=False))
    info = _free if _free is not None else (storage_locations.free_space(root) or {})
    free = info.get('free_bytes')
    out = {'needed_bytes': needed, 'checkpoint_bytes': checkpoint,
           'checkpoint_source': size_source, 'keeps': keeps,
           'master_bytes': master, 'fp8_bytes': fp8, 'fp8_export': bool(fp8),
           'keep_bf16': bool(keep_bf16), 'margin_bytes': margin,
           'path': root, 'free_bytes': None, 'total_bytes': info.get('total_bytes'),
           'shortfall_bytes': None, 'fits': None}
    if not isinstance(free, int) or isinstance(free, bool):
        return out
    out['free_bytes'] = free
    out['fits'] = needed <= free
    if not out['fits']:
        out['shortfall_bytes'] = needed - free
    return out


def disk_refusal_message(forecast) -> str:
    """The actionable sentence: how much is needed, what is free, where it goes."""
    from .hf_storage import fmt_bytes
    parts = [
        f"{DISK_REFUSAL_MARKER}This full-model run needs about "
        f"{fmt_bytes(forecast['needed_bytes'])} on this computer "
        f"(one {fmt_bytes(forecast['checkpoint_bytes'])} checkpoint"
        + (f" × {forecast['keeps']} kept" if forecast.get('keeps', 1) > 1 else '')
        + (f" + a {fmt_bytes(forecast.get('fp8_bytes'))} fp8 export"
           if forecast.get('fp8_bytes') else '')
        + f" + {fmt_bytes(forecast['margin_bytes'])} margin), but the checkpoint "
        f"folder has {fmt_bytes(forecast.get('free_bytes'))} free — about "
        f"{fmt_bytes(forecast['shortfall_bytes'])} short.",
        # Deliberately no machine path: this sentence is a refusal users paste
        # into a bug report, and the setting names say where to go just as well.
        'Free space in the checkpoint folder, point it at a bigger drive in '
        'Settings ▸ Storage, or deliver to Hugging Face only '
        '(Settings ▸ Storage ▸ Full-model delivery).',
        'This estimate is generous on purpose; you can start the run anyway.',
    ]
    return '\n'.join(parts)


def hub_backup_warning(forecast) -> str | None:
    """Say BEFORE the run that the Hugging Face backup probably will not fit.

    In ``both`` mode the Hub copy is best-effort — it can no longer fail a run —
    so this is deliberately not a refusal. But "you will not be able to continue
    this model later" is something to learn before eight hours of GPU, not after.
    Returns None when the forecast fits or could not be measured.
    """
    if not forecast or forecast.get('fits') is not False:
        return None
    from .hf_storage import fmt_bytes
    return (
        'The Hugging Face backup copy of this full model probably will not fit: '
        f"it needs about {fmt_bytes(forecast.get('needed_bytes'))} of private "
        f"storage and roughly {fmt_bytes(forecast.get('free_bytes'))} is free "
        f"({fmt_bytes(forecast.get('shortfall_bytes'))} short). The run itself is "
        'unaffected — the model is delivered to this computer first and the Hub '
        'copy is attempted afterwards. Without that copy the run cannot be '
        'continued later: free space in Settings ▸ Training ▸ Hugging Face '
        'storage to keep it resumable.')


def assert_local_disk_headroom(*, keeps=1, fp8_export=True, keep_bf16=True,
                               allow_override=False, _forecast=None) -> dict:
    """Refuse a dense launch that plainly will not fit on this disk.

    Called BEFORE a pod is rented, for the same reason the Hugging Face check is:
    a refusal costs nothing, and a disk that fills up at the very end costs the
    whole run. Overridable, and an unmeasurable volume never blocks.
    """
    forecast = _forecast if _forecast is not None else local_delivery_forecast(
        keeps=keeps, fp8_export=fp8_export, keep_bf16=keep_bf16)
    if forecast.get('fits') is False and not allow_override:
        raise ValueError(disk_refusal_message(forecast))
    return forecast


# --- what the pod is holding ----------------------------------------------------

def is_fp8_name(name) -> bool:
    return str(name or '').endswith(FP8_SUFFIX)


def step_of(name, default=None):
    """Step number encoded in an ai-toolkit save name, or ``default``."""
    m = re.search(r'_(\d{6,})(?:_fp8)?\.safetensors$', str(name or ''))
    return int(m.group(1)) if m else default


def select_pod_artifacts(files, keep_bf16=True) -> dict:
    """Which of the pod's files to bring home: ``{'master', 'fp8'}``.

    ``files`` is what ``RemoteAiToolkit.list_files`` returns ({'path','size'}).
    ai-toolkit zero-pads step numbers, so lexicographic order IS step order —
    the same rule ``_newest_remote_checkpoint`` already relies on. The newest
    master is the run's result; the fp8 twin is taken when the exporter produced
    one, and it is the ONLY thing taken when the master was dropped.
    """
    masters, twins = [], []
    for entry in files or []:
        path = str((entry or {}).get('path') or '')
        if not path.endswith('.safetensors'):
            continue
        name = os.path.basename(path.replace('\\', '/'))
        (twins if is_fp8_name(name) else masters).append(entry)
    masters.sort(key=lambda f: f['path'])
    twins.sort(key=lambda f: f['path'])
    out = {'master': masters[-1] if (masters and keep_bf16) else None,
           'fp8': twins[-1] if twins else None}
    if out['master'] is None and not keep_bf16 and masters and not twins:
        # The twin never appeared (export off or failed) and the master is the
        # only artifact there is. Dropping it would leave the run empty-handed.
        out['master'] = masters[-1]
    return out


# --- proving what landed --------------------------------------------------------

class LocalDeliveryError(RuntimeError):
    """The local copy is not trustworthy. NEVER destroy a pod on this."""


def verify_local_file(path, expected_size=None) -> dict:
    """Prove one harvested file: size, then a real safetensors header read.

    The size check catches a stream that ended early; the header read catches
    the rest — a file of exactly the right length made of the proxy's error
    page, a truncated rename, a disk that returned zeros. Reading the header
    touches the first few kilobytes only: proving a 26 GB file must not cost
    26 GB of I/O.
    """
    from .fp8_export import Fp8ExportError, read_header

    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise LocalDeliveryError(f'the downloaded file is not readable ({e})') from e
    want = int(expected_size or 0)
    if want and size != want:
        raise LocalDeliveryError(
            f'the downloaded file is {size} bytes, the pod advertised {want}')
    if size <= 0:
        raise LocalDeliveryError('the downloaded file is empty')
    try:
        header = read_header(path)
    except Fp8ExportError as e:
        raise LocalDeliveryError(
            f'the downloaded file is not a readable safetensors checkpoint ({e})') from e
    tensors = [k for k in header if k != '__metadata__']
    if not tensors:
        raise LocalDeliveryError(
            'the downloaded file declares no tensors — it is not a checkpoint')
    return {'path': str(path), 'size_bytes': size, 'tensors': len(tensors),
            'name': os.path.basename(str(path))}
