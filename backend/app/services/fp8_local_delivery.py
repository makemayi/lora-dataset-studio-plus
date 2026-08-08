"""One click: get the fp8 file, where ComfyUI will find it.

THE PROBLEM THIS SOLVES
-----------------------
The quantizer worked and nobody could use it on the model they actually had.
``fp8_quantize`` needs an absolute path to a ``.safetensors`` on this machine —
but a dense run's 26 GB master lives ONLY in a private Hugging Face repository
(that is the whole point of the dense lane: the file is never downloaded), so
there was no path to paste. The tool was reachable from two doors and neither
one could open the door the user was standing at.

So this module is the normal path and the tool's text field becomes the exotic
one. It chains, in one job:

    resolve the master → download it (resumable) → quantize it →
    leave the fp8 in the ComfyUI folder that loads it

and it answers, BEFORE anything starts, the three questions that decide whether
it should start at all: where the file will land, whether the disk can take it,
and whether this machine can convert at all.

WHAT IS REUSED, AND WHY NOTHING IS REWRITTEN
--------------------------------------------
* the conversion is ``fp8_quantize.quantize`` — which runs ``fp8_export.py`` in
  an interpreter that has torch, the one implementation, unit tested against
  real torch and accepted by ComfyUI's own loader;
* every refusal is ``fp8_quantize.plan`` — already quantized, LoRA/adapter,
  unreadable file, no room, no torch — run on the downloaded master before a
  byte is written, and the read-back proof comes back with the worker's result;
* which file in the repository IS the master is ``dense_weights.pick_master``,
  the same rule the delivery verifier stamps on the run;
* where a model file goes comes from ``comfy_model_paths``, the same resolver
  the LoRA deploy uses (an ``extra_model_paths.yaml`` root wins there, so it
  wins here).

WHAT IS NEW HERE, AND WHY IT HAD TO BE
--------------------------------------
The download. ``hf_hub_download`` resumes but reports nothing, and a 26 GB
transfer that shows no progress for forty minutes is indistinguishable from a
hung app. This streams the resolved Hub URL itself with a ``Range`` header, so
it reports bytes as they land, resumes across app restarts from the ``.part``
left on disk, and can be cancelled — the three things a transfer of this size
must have. It is the same shape as ``aitoolkit_remote._download``, which already
survives cut streams in production.

DISK, BEFORE — NOT DURING
-------------------------
A 26 GB pull plus a ~14 GB write on a drive with 20 GB free is not a failure to
report at 90 %: it is a refusal to make at 0 %. ``plan`` computes what the whole
chain will claim and says no with the two numbers in it.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time

from .. import config as cfg
from ..job_queue import queue_manager
from . import comfy_model_paths, dense_weights, fp8_export, fp8_quantize

logger = logging.getLogger(__name__)

_STATE_KEY = 'fp8_local_delivery'
_STATE_TTL = 24 * 3600
_lock = threading.Lock()
_cancel = threading.Event()

# Working headroom, shared with the manual lane so one number explains both.
# Everything else in the budget is DERIVED from this job: the bytes still to
# download plus the fp8 file's own planning ceiling. A flat threshold refused a
# conversion that fit twice over (12.8 GB of output, 17.6 GB free, "~30 GB
# needed"), which is the failure mode a disk guard is supposed to be the
# opposite of.
MARGIN_BYTES = fp8_quantize.WRITE_HEADROOM_BYTES

# ComfyUI's folder for the kind of file each family delivers. A dense transformer
# is loaded by "Load Diffusion Model" out of diffusion_models/; an SDXL full
# checkpoint is a one-file bundle read from checkpoints/. Getting this wrong does
# not corrupt anything — it just leaves the file invisible to the node that wants
# it, which is exactly the problem this feature exists to end.
FOLDER_TYPE_BY_FAMILY = {'sdxl': 'checkpoints'}
DEFAULT_FOLDER_TYPE = 'diffusion_models'

_CHUNK = 8 * 1024 * 1024
_DOWNLOAD_ATTEMPTS = 6
_CONNECT_TIMEOUT = 20
_READ_TIMEOUT = 300


class DeliveryError(ValueError):
    """A refusal with a sentence for the user. Never a stack trace."""


class _Cancelled(BaseException):
    """Cancellation, raised from inside the exporter's progress callback.

    A ``BaseException`` on purpose: ``export_scaled_fp8`` swallows anything its
    progress callback raises (``except Exception: progress = None``) so a noisy
    UI callback can never break a conversion — correct, and it also means a
    plain exception cannot stop one. This one passes through, and the partial
    output it leaves behind is removed by the handler below, which knows the
    destination path.
    """


# --- resolution ------------------------------------------------------------------

def folder_type_for(family) -> str:
    return FOLDER_TYPE_BY_FAMILY.get(str(family or '').strip().lower(),
                                     DEFAULT_FOLDER_TYPE)


def _comfy_write_dir(folder_type):
    """The ComfyUI folder to WRITE this kind of model into, or None.

    ``comfy_model_paths.search_roots`` gives ComfyUI's own priority order, and
    that order is where the subtlety is: for a diffusion model it is
    ``[models/unet, models/diffusion_models, …extras]``, because ComfyUI reads
    both and lists ``unet`` first for history. Writing into ``roots[0]`` blindly
    would CREATE a ``models/unet/`` folder on the very many installs that only
    have ``models/diffusion_models/`` — a file that loads fine, in a directory
    the user has never seen. So: the first root that already EXISTS wins (which
    keeps an ``is_default`` yaml root ahead of everything, exactly as ComfyUI
    treats it); with none of them on disk yet, the modern canonical name is
    created rather than the legacy alias.
    """
    try:
        roots = comfy_model_paths.search_roots(folder_type) or []
    except Exception:                                   # noqa: BLE001 — never fatal
        return None
    for root in roots:
        if os.path.isdir(root):
            return root
    for root in roots:
        if os.path.basename(root).lower() == folder_type:
            return root
    return roots[0] if roots else None


def destination_folder(family) -> dict:
    """Where the fp8 file will land, and whether that is really ComfyUI's folder.

    Never silent: the caller shows this to the user BEFORE the job starts and
    again when it ends. A fallback that looked like a ComfyUI folder would send
    someone hunting through Load Diffusion Model for a file that is not there.
    """
    folder_type = folder_type_for(family)
    root = _comfy_write_dir(folder_type)
    if root:
        return {'path': os.path.normpath(str(root)), 'kind': 'comfyui',
                'folder_type': folder_type,
                'note': f'ComfyUI ▸ models/{folder_type}'}
    fallback = os.path.join(str(cfg.data_dir()), 'models', folder_type)
    return {'path': os.path.normpath(fallback), 'kind': 'fallback',
            'folder_type': folder_type,
            'note': ('ComfyUI is not configured in Settings, so the file goes to '
                     "the app's own models folder — move it into ComfyUI yourself, "
                     'or set the ComfyUI folder and run it again')}


def _hf_token():
    return cfg.secret('HF_CLOUD_TOKEN') or cfg.secret('HF_TOKEN') or None


def _repo_files(repo_id, token) -> list[tuple[str, int]]:
    """``(rfilename, size)`` for every file in the repo — metadata only, no download."""
    from .hf_publish import _make_api
    info = _make_api(token).repo_info(repo_id=repo_id, repo_type='model',
                                      files_metadata=True)
    out = []
    for sibling in (getattr(info, 'siblings', None) or []):
        name = str(getattr(sibling, 'rfilename', '') or '')
        if not name:
            continue
        size = getattr(sibling, 'size', None)
        lfs = getattr(sibling, 'lfs', None)
        if size is None and lfs is not None:
            size = lfs.get('size') if isinstance(lfs, dict) else getattr(lfs, 'size', None)
        out.append((name, int(size) if isinstance(size, int) and not isinstance(size, bool) else 0))
    return out


def _free_bytes(path) -> int | None:
    """Free space on the volume that REALLY holds this path.

    ``realpath`` first: a ComfyUI models folder is very often a junction onto
    another drive, and asking about the apparent path answers for the wrong
    volume — measured on exactly that layout (``C:\\…\\models\\unet`` →
    ``A:\\ComfyUI\\models\\unet``, 113 GB free vs 17.6).
    """
    probe = os.path.realpath(path)
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return None
        probe = parent
    try:
        return shutil.disk_usage(probe).free
    except Exception:                                   # noqa: BLE001 — unknown, not fatal
        return None


# --- planning ---------------------------------------------------------------------

def plan(*, repo_id=None, filename=None, path=None, family=None,
         keep_master=True, destination_dir=None, _files=None) -> dict:
    """Everything the user must be told before clicking, or a refusal saying why.

    Two source kinds, one answer shape: a Hugging Face repository (the master is
    downloaded first) or a full-precision file already on this machine (nothing
    to download). The caller never types a path in either case — it passes the
    run's repository, or the path the app already holds.
    """
    keep_master = bool(keep_master)
    # An explicit folder wins: the resolved one may sit on a volume that is
    # simply too full, and refusing an operation that would succeed one drive
    # over is not a safety feature.
    override = str(destination_dir or '').strip().strip('"')
    if override:
        if not os.path.isabs(override):
            raise DeliveryError('give the full path of the folder to write into')
        dest = {'path': os.path.normpath(override), 'kind': 'chosen',
                'folder_type': folder_type_for(family),
                'note': 'the folder you chose — ComfyUI only lists it if it is one '
                        'of its model folders'}
    else:
        dest = destination_folder(family)
    source_kind = 'local' if path else 'huggingface'

    if source_kind == 'local':
        local = str(path or '').strip().strip('"')
        if not os.path.isabs(local) or not os.path.isfile(local):
            raise DeliveryError('this model file is no longer where the app '
                                'recorded it — check the path in Settings')
        # Every refusal (already quantized, adapter, unreadable, no room, output
        # already there) is the shared engine's, run here against the folder this
        # job will really write into — not the one next to the source.
        try:
            checked = fp8_quantize.plan(local, destination=os.path.join(
                dest['path'], fp8_export.fp8_name_for(os.path.basename(local))))
        except (fp8_quantize.QuantizeError, fp8_export.Fp8ExportError) as e:
            raise DeliveryError(str(e)) from e
        weight_name = checked['source_name']
        source_bytes = checked['source_bytes']
        master_path = local
        estimated = checked['estimated_bytes']
        # A file the user already had is never deleted by this job.
        keep_master = True
        download_bytes = 0
    else:
        repo = str(repo_id or '').strip()
        if not repo:
            raise DeliveryError('this run did not record a Hugging Face repository')
        token = _hf_token()
        if not token:
            raise DeliveryError(
                'no Hugging Face token is configured — add HF_CLOUD_TOKEN (or '
                'HF_TOKEN) in Settings ▸ Local tools so the master can be downloaded')
        files = _files if _files is not None else _repo_files(repo, token)
        chosen = describe_repo_choice(files, filename=filename)
        if not chosen['name']:
            raise DeliveryError(
                f'{repo} holds no full-precision checkpoint to quantize — only an '
                'fp8 export, or no weights at all')
        weight_name = chosen['name']
        source_bytes = chosen['size']
        master_path = os.path.join(dest['path'], os.path.basename(weight_name))
        estimated = fp8_export.typical_fp8_bytes(source_bytes)
        have = os.path.getsize(master_path) if os.path.isfile(master_path) else 0
        part = f'{master_path}.part'
        resumed = os.path.getsize(part) if os.path.isfile(part) else 0
        download_bytes = max(0, source_bytes - (have if have else resumed))

    # Can this machine convert AT ALL? Asked for BOTH source kinds, here, before
    # a 26 GB download rather than after it: the conversion runs in an
    # interpreter that has torch, and the app's own usually does not.
    worker = fp8_quantize.interpreter()
    if not worker['ready']:
        raise DeliveryError(worker['reason'])

    destination = os.path.join(dest['path'], fp8_export.fp8_name_for(
        os.path.basename(weight_name)))
    if os.path.exists(destination):
        raise DeliveryError(
            f'{os.path.basename(destination)} is already in '
            f'{dest["path"]} — delete or rename it first. This never overwrites '
            'a file you already have.')

    # Derived from THIS job, never a flat threshold: the bytes still to come
    # down, plus the fp8 file's planning ceiling (which rounds up on purpose —
    # a forecast that under-states is the failure a disk guard exists to
    # prevent), plus the same working headroom the manual lane uses.
    output_ceiling = fp8_export.estimate_fp8_bytes(source_bytes)
    needed = download_bytes + output_ceiling + MARGIN_BYTES
    free = _free_bytes(dest['path'])
    enough = free is None or free >= needed
    out = {
        'source_kind': source_kind,
        'repo_id': repo_id or None,
        'weight_name': weight_name,
        'weight_basename': os.path.basename(weight_name),
        'source_bytes': source_bytes,
        'master_path': master_path,
        'keep_master': keep_master,
        'download_bytes': download_bytes,
        'destination_dir': dest['path'],
        'destination_dir_kind': dest['kind'],
        'destination_dir_note': dest['note'],
        'folder_type': dest['folder_type'],
        'destination': destination,
        'destination_name': os.path.basename(destination),
        'estimated_bytes': estimated,
        'free_bytes': free,
        'required_bytes': needed,
        'enough_space': enough,
        'python': worker['python'],
        'family': str(family or '').strip().lower() or None,
    }
    if source_kind == 'huggingface':
        out['choice'] = chosen
    if not enough:
        # Every term is named, so the number can be checked and acted on. "~30 GB
        # needed" next to a 12.8 GB output was neither.
        parts = []
        if download_bytes:
            parts.append(f'{download_bytes / 1000 ** 3:.1f} GB still to download')
        parts.append(f'up to {output_ceiling / 1000 ** 3:.1f} GB for the fp8 file')
        parts.append(f'{MARGIN_BYTES / 1000 ** 3:.0f} GB of working headroom')
        out['space_error'] = (
            f'not enough disk space in {dest["path"]}: {free / 1000 ** 3:.1f} GB free, '
            f'and this needs {needed / 1000 ** 3:.1f} GB — {", ".join(parts)}. '
            'Free up space, or write it to another folder on a bigger drive.')
    return out


def describe_repo_choice(files, filename=None) -> dict:
    """Which file the job will take, and what it took it over.

    ``filename`` pins an explicit choice (the run's recorded weight); without one
    the shared rule decides. Either way the answer is REPORTED, because a
    repository routinely holds a final save and several step snapshots of the
    same 26 GB model and picking one silently is how a card ends up naming a
    different file from the button underneath it.
    """
    eligible = dense_weights.candidates(files)
    sizes = dict(eligible)
    chosen, pinned = None, False
    wanted = os.path.basename(str(filename or '').strip().replace('\\', '/'))
    if wanted:
        chosen = next((name for name, _s in eligible
                       if os.path.basename(name) == wanted), None)
        pinned = chosen is not None
    if chosen is None:
        chosen = dense_weights.pick_master(eligible)
    if chosen is None:
        return {'name': None, 'size': 0, 'pinned': False, 'total': 0,
                'step': None, 'is_final': False, 'others': []}
    step = dense_weights.split_step(chosen)[1]
    return {'name': chosen, 'size': sizes.get(chosen, 0), 'pinned': pinned,
            'total': len(eligible), 'step': step, 'is_final': step is None,
            'others': [name for name, _s in eligible if name != chosen]}


def describe(**kwargs) -> dict:
    """``plan`` as a payload the UI can always render — refusal included."""
    try:
        return {'ok': True, **plan(**kwargs)}
    except (DeliveryError, fp8_quantize.QuantizeError,
            fp8_export.Fp8ExportError) as e:
        return {'ok': False, 'error': str(e)}
    except Exception as e:                              # noqa: BLE001 — reported, not raised
        logger.warning('fp8 delivery plan unavailable: %s', e)
        return {'ok': False, 'error': f'could not read the model: {e}'[:300]}


# --- state ------------------------------------------------------------------------

def status() -> dict:
    return queue_manager._get_system_state(_STATE_KEY, {}) or {}


def _set(state, info, **extra):
    queue_manager._set_system_state(_STATE_KEY, {
        'status': state,
        'source_kind': info['source_kind'],
        'repo_id': info.get('repo_id'),
        'weight_name': info['weight_basename'],
        'source_bytes': info['source_bytes'],
        'destination_dir': info['destination_dir'],
        'destination_dir_kind': info['destination_dir_kind'],
        'destination_name': info['destination_name'],
        'destination': info['destination'],
        'keep_master': info['keep_master'],
        'master_path': info['master_path'],
        'estimated_bytes': info['estimated_bytes'],
        **extra,
    }, ttl_seconds=_STATE_TTL)


def cancel() -> bool:
    """Ask the running job to stop. Returns False when nothing is running.

    A cancelled download keeps its ``.part``: the next start resumes from it, so
    stopping is never the same as throwing away the gigabytes already pulled.
    """
    if status().get('status') not in ('downloading', 'quantizing'):
        return False
    _cancel.set()
    return True


# --- the job ----------------------------------------------------------------------

def _refuse_if_busy():
    """Raise when this machine is already converting something.

    One sentence, one place: ``start`` asks it twice (once before planning for
    the message the user should get, once under the lock for the exclusion) and
    two copies of the wording would drift apart."""
    if status().get('status') in ('downloading', 'quantizing'):
        raise DeliveryError('a model is already being prepared — wait for it to finish')
    if fp8_quantize.status().get('status') == 'running':
        raise DeliveryError('a quantization is already running — wait for it to finish')


def start(app, *, repo_id=None, filename=None, path=None, family=None,
          keep_master=True, destination_dir=None, _files=None, _download=None) -> dict:
    """Validate, then run the chain in a daemon thread. Refuses on the click.

    Nothing is refused here that ``plan`` would have accepted — the panel's
    button is disabled by the very same verdict it shows.

    ORDER MATTERS, and this is why "a job is already running" is asked FIRST. It
    is a condition that is true RIGHT NOW and that the user can act on in one
    gesture (wait, or stop it), while "not enough disk space" is a FORECAST about
    work that could not have started anyway. Announcing the forecast first sent
    people off to free gigabytes for a run that was never blocked on disk.
    Cheaper too: ``plan`` lists a Hugging Face repository and probes interpreters,
    and none of that has to be paid for to say "something else is already
    running". ``plan`` is read-only, so nothing downstream depends on having
    called it before the refusal.
    """
    _refuse_if_busy()
    info = plan(repo_id=repo_id, filename=filename, path=path, family=family,
                keep_master=keep_master, destination_dir=destination_dir,
                _files=_files)
    if not info['enough_space']:
        raise DeliveryError(info['space_error'])
    with _lock:
        # Asked AGAIN, and this one is the real mutual exclusion: ``plan`` above
        # can spend seconds on the network, so the answer from before it is a
        # message-ordering courtesy, not a guarantee. Only a check that shares
        # the lock with ``_set`` below can keep two clicks from both starting.
        _refuse_if_busy()
        _cancel.clear()
        _set('downloading' if info['download_bytes'] else 'quantizing', info,
             downloaded_bytes=0, download_total_bytes=info['download_bytes'],
             done=0, total=0, started_at=time.time())

    def _run():
        with app.app_context():
            _execute(info, download=_download)

    threading.Thread(target=_run, daemon=True, name='fp8-local-delivery').start()
    return info


def _execute(info, download=None):
    destination = info['destination']
    downloaded_here = False
    try:
        if info['source_kind'] == 'huggingface' and not _master_complete(info):
            _set('downloading', info, downloaded_bytes=0,
                 download_total_bytes=info['source_bytes'])
            (download or _download_master)(info, _on_bytes(info))
            downloaded_here = True
        if _cancel.is_set():
            raise _Cancelled()

        # The master is on disk now: run the REAL guards on the REAL file. A repo
        # can hold something the Hub metadata could not reveal (an already
        # quantized export renamed, an adapter) and refusing here still costs
        # nothing but the download the user asked for.
        checked = fp8_quantize.plan(info['master_path'], destination=destination)
        _set('quantizing', info, done=0, total=checked['quantized_tensors'],
             downloaded_bytes=info['source_bytes'],
             download_total_bytes=info['source_bytes'])

        def on_progress(done, total):
            _set('quantizing', info, done=done, total=total,
                 downloaded_bytes=info['source_bytes'],
                 download_total_bytes=info['source_bytes'])

        # The fallback folder (and a ComfyUI root that only exists in config)
        # may not be on disk yet; a local source never went through the
        # downloader that would have created it.
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        # In a SUBPROCESS: the conversion needs torch, the app does not have it,
        # and Stop has to work here too — the child is killed, so the partial
        # output goes with it.
        summary = fp8_quantize.quantize(
            info['master_path'], destination=destination,
            progress=on_progress, cancelled=_cancel.is_set)
        verified = {'verified': summary.get('verified'),
                    'verify_error': summary.get('verify_error'),
                    'scaled_tensors': summary.get('scaled_tensors')}
        if _cancel.is_set():
            raise _Cancelled()
        removed = False
        if not info['keep_master'] and downloaded_here and not verified.get('verify_error'):
            # Only after the output is proven readable, and only a file this job
            # brought down itself.
            try:
                os.remove(info['master_path'])
                removed = True
            except OSError as e:
                logger.warning('fp8 delivery could not remove the master: %s', e)
        result = {**summary, **verified, 'master_removed': removed,
                  'master_kept': not removed}
        _set('error' if verified.get('verify_error') else 'done', info,
             result=result, error=verified.get('verify_error'),
             done=summary.get('tensors'), total=summary.get('tensors'))
        logger.info('fp8 delivery finished: %s', info['destination_name'])
    except _Cancelled:
        # The killed child leaves its `<dst>.part` behind; the resumable master
        # `.part` is deliberately NOT removed — that is the point of stopping.
        _unlink(f'{destination}.part')
        _set('cancelled', info,
             error='Stopped. The part of the master already downloaded is kept, '
                   'so starting again resumes instead of restarting.')
    except Exception as e:                              # noqa: BLE001 — reported
        if _cancel.is_set():
            # Stopping during the conversion reaches us as the worker's own
            # "stopped" refusal. It is a cancellation, not a failure, and must
            # read like one — including the promise that the download is kept.
            _unlink(f'{destination}.part')
            _set('cancelled', info,
                 error='Stopped. The part of the master already downloaded is kept, '
                       'so starting again resumes instead of restarting.')
        else:
            _set('error', info, error=str(e)[:400])
            logger.warning('fp8 delivery failed (%s): %s', info['weight_basename'], e)
    finally:
        _cancel.clear()


def _on_bytes(info):
    last = [0.0]

    def report(done, total):
        if _cancel.is_set():
            raise _Cancelled()
        now = time.monotonic()
        if now - last[0] < 1.0 and done < total:
            return                                      # a 26 GB pull, once a second
        last[0] = now
        _set('downloading', info, downloaded_bytes=done, download_total_bytes=total)
    return report


def _master_complete(info) -> bool:
    path = info['master_path']
    return os.path.isfile(path) and os.path.getsize(path) == info['source_bytes']


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _download_master(info, on_progress):
    """Stream the master out of the Hub into ``<master>.part``, then rename.

    Resumable on two timescales: within the job (a cut stream continues from the
    current offset with a ``Range`` header) and across app restarts (the ``.part``
    is left in place by a cancel or a crash, and the next start continues it).
    """
    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id=info['repo_id'], filename=info['weight_name'],
                     repo_type='model')
    token = _hf_token()
    tmp = f'{info["master_path"]}.part'
    os.makedirs(os.path.dirname(info['master_path']), exist_ok=True)
    want = int(info['source_bytes'] or 0)
    got = os.path.getsize(tmp) if os.path.isfile(tmp) else 0

    for _attempt in range(_DOWNLOAD_ATTEMPTS):
        if _cancel.is_set():
            raise _Cancelled()
        before = got
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        if got:
            headers['Range'] = f'bytes={got}-'
        try:
            with requests.get(url, stream=True, headers=headers,
                              timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT)) as r:
                if got and r.status_code == 416:
                    pass                                # nothing left to serve
                else:
                    if r.status_code not in (200, 206):
                        raise DeliveryError(
                            f'Hugging Face refused the download (HTTP {r.status_code}) — '
                            'check that HF_CLOUD_TOKEN can read this private repository')
                    if got and r.status_code == 200:
                        got = 0                         # Range ignored -> restart
                    written = got
                    with open(tmp, 'ab' if got else 'wb') as fh:
                        for chunk in r.iter_content(chunk_size=_CHUNK):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            written += len(chunk)
                            on_progress(written, want or written)
        except requests.RequestException:
            pass                                        # cut mid-stream -> resume
        got = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
        if want and got == want:
            os.replace(tmp, info['master_path'])
            return info['master_path']
        if not want and got:
            os.replace(tmp, info['master_path'])
            return info['master_path']
        if got <= before:
            break                                       # no progress: stop retrying
    raise DeliveryError(
        f'the download stopped at {got / 1000 ** 3:.1f} GB of '
        f'{want / 1000 ** 3:.1f} GB. Nothing is lost — start it again and it '
        'resumes from there.')
