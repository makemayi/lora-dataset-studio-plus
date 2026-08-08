"""Plan and run a LoRA-into-base merge: the orchestrator around ``lora_merge``.

WHY THE SPLIT
-------------
Exactly the split the fp8 lane already uses, for exactly the same reason. The
merge itself needs ``torch``; the app's own environment does not have it and must
not (it is gigabytes, and LDS installs and runs without it). So ``lora_merge`` is
a self-contained CLI a configured interpreter runs as a subprocess, and THIS
module — which the server imports — never touches torch. It reads safetensors
HEADERS, which is pure ``struct`` and ``json``.

THE PLAN MUST REPRESENT WHAT THE RUN REQUIRES
---------------------------------------------
The fp8 button failed three times in one week — not enough disk, a fixed
threshold that did not fit, dependencies missing from the environment that would
have done the work — and all three were ONE defect: planning did not model the
conditions execution would meet. So every condition that can stop this merge is
evaluated in ``plan``, before the button is enabled:

  * the base is readable, is not already quantized, is not a LoRA;
  * every LoRA fits it — key by key, shape by shape (``lora_merge.plan_merge``);
  * the output does not already exist;
  * the volume that will REALLY hold the output has room for it, measured
    against the base's actual size rather than a constant somebody guessed;
  * the interpreter that would run the merge can import torch (and only torch —
    ``lora_merge`` reads and writes the safetensors format itself).

None of that reads a weight byte: it is all header arithmetic, so a complete
plan on a 26 GB file returns in milliseconds. And the plan carries what happens
if it dies half way, because that is a question the user is entitled to have
answered before starting something that writes 26 GB.

ONE MERGE AT A TIME
-------------------
Two merges at once would each pass their own free-space check and then race for
the same bytes — the exact way a disk guard turns into a coin toss. The job state
is app-wide, in ``system_state``, so it also survives a page reload: a user who
closes the tab mid-merge finds the progress bar again, instead of being offered a
second merge on top of the first.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading

from ..job_queue import queue_manager
from . import (comfy_model_paths, fp8_quantize, lora_merge, lora_training,
               model_integrity)

logger = logging.getLogger(__name__)

_STATE_KEY = 'lora_merge'
_STATE_TTL = 6 * 3600
_lock = threading.Lock()

# Set to ask a running merge to stop. A merge can legitimately run for half an
# hour on a 26 GB base; leaving the user with no way out but killing the app —
# which would strand the .part file — is not an option a job this long can ship
# without. The worker is killed and its partial file removed by the writer's own
# cleanup, so a cancelled merge changes nothing on disk.
_cancel = threading.Event()

# The thread doing the work, IN THIS PROCESS. Kept so we can tell a merge that is
# really running from a 'running' state left behind by a process that died — see
# `reconcile`.
_worker_thread = None

_ACCEPTED_EXT = ('.safetensors', '.sft')

# ComfyUI folder types a base checkpoint may live under, in the order the app
# already searches them elsewhere.
_BASE_FOLDER_TYPES = ('diffusion_models', 'unet', 'checkpoints')

# Working headroom on top of the file we are about to write. It is NOT a second
# copy: the worker streams into `<dst>.part` and renames, so the only bytes
# claimed are the output's. This covers the filesystem being unhappy near zero
# and whatever else the app writes while a long merge runs. Same reasoning — and
# same number — as the fp8 lane, which learned it by refusing a conversion that
# fit.
WRITE_HEADROOM_BYTES = 2 * 1000 ** 3

# The merge is bounded so a wedged worker cannot hold the single job slot for
# ever. Generous: a 26 GB base on a slow external drive is legitimately long.
DEFAULT_BUDGET_SECONDS = 3 * 3600

# Weight bounds. Beyond 1 a LoRA is being applied harder than it was trained,
# which is a real (if blunt) technique; past 2 it is almost always a typo, and a
# merge is far too expensive to run on a typo. Negative subtracts the LoRA, which
# is legitimate and deliberately allowed.
WEIGHT_MIN, WEIGHT_MAX = -2.0, 2.0

# The base sniff and the LoRA sniff spell this family differently ('krea2' for a
# full checkpoint, 'krea' for an adapter). One vocabulary here.
_BASE_ARCH_TO_FAMILY = {'krea2': 'krea', 'sdxl': 'sdxl', 'sd15': 'sd15',
                        'flux': 'flux'}


class MergeJobError(ValueError):
    """Refusal with a sentence for the user. Never a stack trace."""


def status() -> dict:
    return queue_manager._get_system_state(_STATE_KEY, {}) or {}


def _running_here() -> bool:
    """True when THIS process still has the merge thread alive."""
    return _worker_thread is not None and _worker_thread.is_alive()


def reconcile() -> bool:
    """Turn a 'running' state left by a dead process into an honest outcome.

    The job state lives in ``system_state`` with a 6 h TTL, on purpose: a merge
    outlives the tab that started it, and someone who closes the page must find
    the progress bar again. The cost of that is a state which says "running"
    after the app is restarted mid-merge — and then a progress bar that can never
    move, and every new merge refused for six hours because one is "already
    running". Nobody would connect that to the restart.

    Only one process can be merging (the thread is here or it is nowhere), so
    "the state says running and no thread of ours is alive" is not a guess: the
    merge is gone. Its ``.part`` file was already removed, or is orphaned and
    named ``.part`` precisely so it can never be mistaken for a model.

    Returns True when it cleaned something up. Called by the status route, which
    is the first thing the panel asks after a reload.
    """
    current = status()
    if current.get('status') != 'running' or _running_here():
        return False
    queue_manager._set_system_state(_STATE_KEY, {
        **current,
        'status': 'error',
        'error': ('the app was restarted while this merge was running, so it did '
                  'not finish. Nothing was overwritten — start it again.'),
    }, ttl_seconds=_STATE_TTL)
    logger.info('lora merge state reconciled after a restart: %s',
                current.get('destination_name'))
    return True


def _free_bytes(path):
    """Free space on the volume that REALLY holds this path.

    ``realpath`` first, on purpose: a ComfyUI models folder is very often a
    junction onto another drive, and asking about the apparent path can answer
    for the wrong volume — which turns a disk guard into a coin toss.
    """
    try:
        import shutil
        return shutil.disk_usage(os.path.dirname(os.path.realpath(path))).free
    except Exception:                          # noqa: BLE001
        return None


def space_error(free_bytes, output_bytes) -> str | None:
    """The refusal sentence when the output does not fit, or None.

    States its own arithmetic, so the user can check it and act on it. "~30 GB
    needed" next to a 12.8 GB output was a number that meant nothing to anybody.
    """
    if free_bytes is None:
        return None                            # unmeasurable never blocks
    need = int(output_bytes or 0) + WRITE_HEADROOM_BYTES
    if free_bytes >= need:
        return None
    return (f'not enough disk space where the merged model would go: '
            f'{free_bytes / 1000 ** 3:.1f} GB free, and this needs '
            f'{need / 1000 ** 3:.1f} GB — the {int(output_bytes) / 1000 ** 3:.1f} GB '
            f'model plus {WRITE_HEADROOM_BYTES / 1000 ** 3:.0f} GB of working '
            'headroom. Free up space, or write it to another folder.')


def resolve_ref(ref, folder_types):
    """A user-supplied model reference as an absolute path, or None.

    Accepts BOTH shapes the UI can produce: an absolute path (someone pasted
    where they downloaded the Turbo LoRA) and a ComfyUI-relative name (someone
    picked a row out of the LoRA list, which stores exactly what a loader
    stores). The relative form goes through ``resolve_model_file``, which walks
    the search roots in ComfyUI's own priority order and refuses ``..`` — the
    resolver is not the place to widen what a path may reach.
    """
    ref = str(ref or '').strip().strip('"')
    if not ref:
        return None
    if os.path.isabs(ref):
        return ref
    for folder_type in folder_types:
        try:
            hit = comfy_model_paths.resolve_model_file(folder_type, ref)
        except Exception:                      # noqa: BLE001 — unconfigured ComfyUI
            hit = None
        if hit:
            return hit
    return None


def _readable(ref, what, folder_types=()) -> str:
    raw = str(ref or '').strip().strip('"')
    if not raw:
        raise MergeJobError(f'choose {what}')
    path = resolve_ref(raw, folder_types)
    if not path:
        raise MergeJobError(
            f'{os.path.basename(raw)} is not on this machine — give the full path '
            'to it, or pick it from the list.')
    if not os.path.isfile(path):
        raise MergeJobError(f'no file at {os.path.basename(path)} — check the path')
    if not path.lower().endswith(_ACCEPTED_EXT):
        raise MergeJobError(f'{os.path.basename(path)} is not a .safetensors file')
    return path


def _base_family(path, header):
    """The family of a full checkpoint, or None when we cannot tell."""
    try:
        keys = set(lora_merge.tensor_entries(header))
        return _BASE_ARCH_TO_FAMILY.get(lora_training._detect_safetensors_arch(keys))
    except Exception:                          # noqa: BLE001 — a sniff never blocks
        return None


def _check_weight(value, label) -> float:
    # A decimal COMMA, named. The app's own field is <input type="number">, whose
    # `.value` is a dot-decimal string by specification even where the browser
    # DISPLAYS "0,8" — verified on an fr-FR machine: the field showed 0,8 and the
    # request body carried 0.8. So this cannot arrive from our UI. It can arrive
    # from a script, from curl, or from a future text field, and "the weight is
    # not a number" would be a baffling thing to read after typing 0,8.
    if isinstance(value, str) and ',' in value:
        raise MergeJobError(
            f'the weight for {label} is written "{value.strip()}" — use a dot for '
            'the decimal point (0.8, not 0,8).')
    try:
        weight = float(value)
    except (TypeError, ValueError):
        raise MergeJobError(
            f'the weight for {label} is not a number. 1.0 applies the LoRA exactly '
            'as trained.') from None
    if weight != weight:                       # NaN
        raise MergeJobError(f'the weight for {label} is not a number')
    if not (WEIGHT_MIN <= weight <= WEIGHT_MAX):
        raise MergeJobError(
            f'the weight for {label} is {weight:g} — merge weights run from '
            f'{WEIGHT_MIN:g} to {WEIGHT_MAX:g}. 1.0 applies the LoRA exactly as '
            'trained; the Turbo re-distillation LoRA is normally merged at 0.8-1.0.')
    if weight == 0:
        raise MergeJobError(
            f'{label} is set to 0, so it would contribute nothing — the merge would '
            'spend an hour writing a copy of the base. Remove it from the list, or '
            'give it a weight.')
    return weight


def plan(base, loras, *, destination=None, destination_dir=None,
         overwrite=False, when=None) -> dict:
    """Validate the whole merge and describe exactly what it would produce.

    ``loras`` is a list of ``{'path', 'weight'}``. Raises ``MergeJobError`` with
    a sentence the UI can show instead of enabling the button.
    """
    base = _readable(base, 'a base .safetensors checkpoint to merge into',
                     _BASE_FOLDER_TYPES)
    if not isinstance(loras, (list, tuple)) or not loras:
        raise MergeJobError('choose at least one LoRA to merge into the base')
    if len(loras) > 8:
        raise MergeJobError(
            f'{len(loras)} LoRAs at once is more than this merges in one pass — '
            'stack up to 8, or merge in two rounds (the output of one merge is a '
            'valid base for the next).')

    integrity = model_integrity.validate_model_file(base)
    if integrity.get('blocking'):
        raise MergeJobError(integrity.get('reason') or 'this base is not a readable model')

    # WHY BOTH QUANTIZED FORMS ARE REFUSED, INCLUDING THE ONE THAT IS POSSIBLE.
    #
    # We do not refuse because it cannot be done. We refuse because there is a
    # strictly better route the app already does end to end: merge into the bf16,
    # then quantize. That gives the SAME final file without the double loss. A
    # button whose best case is "worse than the other button" is not an option,
    # it is a trap with a label on it.
    #
    # The two forms hit different walls, and saying which one costs nothing:
    # a structured export stores W/scale next to a separate scale tensor, so
    # there is no full-precision weight in the file to add a delta to; a bare
    # cast is a plain low-precision copy, where adding and re-saving would round
    # twice. Refusing stays refusing — explaining which wall you are at is the
    # difference between a prohibition and an instruction.
    #
    # This reads `quantized`, which is the broad "not full precision" answer and
    # is deliberately unchanged by the training-base work that added `form`.
    # `plan_merge` refuses these files a second time on dtype, independently of
    # this module, so the guard that speaks well can evolve without the safety
    # depending on it.
    report = model_integrity.quantization_report(base)
    if report.get('quantized'):
        signals = ', '.join(report.get('signals') or []) or 'quantized dtypes'
        if report.get('form') == model_integrity.FORM_STRUCTURED:
            wall = ('its weights are stored alongside separate scale tensors, so '
                    'there is no full-precision weight in it to add a LoRA to')
        else:
            wall = ('its weights are a low-precision copy — adding to them and '
                    'saving again would round the numbers twice, and the loss '
                    'compounds every time somebody does it')
        raise MergeJobError(
            f'{os.path.basename(base)} is already a quantized export ({signals}): '
            f'{wall}. Merge into the full-precision (bf16/fp16) version instead, '
            'then quantize the merged model with the fp8 tool — that gives you the '
            'same final file without the double loss.')

    base_header = lora_merge.read_header(base)          # raises MergeError -> mapped below
    family = _base_family(base, base_header)

    prepared, lora_headers = [], []
    for index, item in enumerate(loras):
        item = item if isinstance(item, dict) else {'path': item}
        path = _readable(item.get('path'), f'LoRA #{index + 1}', ('loras',))
        label = os.path.basename(path)
        weight = _check_weight(item.get('weight', 1.0), label)
        header = lora_merge.read_header(path)
        # A structural check would catch a foreign LoRA too ("targets N weights
        # this base does not have"), but naming the family it WAS trained for is
        # the difference between a puzzle and an instruction.
        detected = lora_training.detect_lora_arch(path)
        if family and detected and lora_training.lora_arch_conflicts(detected, family):
            raise MergeJobError(
                f'{label} is a {lora_merge.FAMILY_LABELS.get(detected, detected)} '
                f'LoRA and this base is '
                f'{lora_merge.FAMILY_LABELS.get(family, family)}. They do not share '
                'a single weight, so there is nothing to merge.')
        prepared.append({'path': path, 'weight': weight, 'name': label})
        lora_headers.append((label, header))

    seen = set()
    for item in prepared:
        key = os.path.normcase(os.path.abspath(item['path']))
        if key in seen:
            raise MergeJobError(
                f'{item["name"]} is in the list twice. Merging a LoRA into itself '
                'twice just doubles its weight — set the weight you want instead.')
        seen.add(key)

    shape = lora_merge.plan_merge(base_header, lora_headers, family=family)

    # Written NEXT TO the base by default, never over it: the base is the only
    # file that can be merged again, and a user who picked the wrong one must be
    # able to just delete the output.
    if destination:
        destination = str(destination).strip().strip('"')
        if not os.path.isabs(destination):
            raise MergeJobError('give the full path for the merged model')
    else:
        folder = str(destination_dir or '').strip().strip('"') or os.path.dirname(base)
        if not os.path.isdir(folder):
            raise MergeJobError(f'{folder} is not a folder on this machine')
        destination = os.path.join(folder, lora_merge.merged_name_for(base, when=when))

    exists = os.path.isfile(destination)
    if exists and not overwrite:
        raise MergeJobError(
            f'{os.path.basename(destination)} already exists — delete it first, or '
            're-run with overwrite.')

    free = _free_bytes(destination)
    refusal = space_error(free, shape['output_bytes'])
    if refusal:
        raise MergeJobError(refusal)

    # Deliberately the same probe and the same setting as the fp8 tool: someone
    # who already told LDS which Python has torch should not have to say it
    # twice. Both lanes read and write safetensors by hand, so both need exactly
    # torch — the probe asks for that and nothing more.
    worker = fp8_quantize.interpreter()
    if not worker['ready']:
        raise MergeJobError(
            (worker.get('reason') or 'the Python that would do the merge is missing '
                                     'torch')
            .replace('Quantizing needs them', 'Merging needs them'))

    return {
        'python': worker['python'],
        'base': base,
        'base_name': os.path.basename(base),
        'base_bytes': os.path.getsize(base),
        'family': family,
        'family_label': lora_merge.FAMILY_LABELS.get(family) if family else None,
        'loras': [{'name': i['name'], 'path': i['path'], 'weight': i['weight'],
                   **{k: v for k, v in row.items() if k != 'label'}}
                  for i, row in zip(prepared, shape['loras'])],
        'destination': destination,
        'destination_name': os.path.basename(destination),
        'destination_dir': os.path.dirname(destination),
        'destination_exists': exists,
        'merged_tensors': shape['touched_tensors'],
        'base_tensors': shape['base_tensors'],
        'carried_over': shape['carried_over'],
        'carried_over_bytes': shape['carried_over_bytes'],
        'output_bytes': shape['output_bytes'],
        'required_bytes': shape['output_bytes'] + WRITE_HEADROOM_BYTES,
        'free_bytes': free,
        'estimated_seconds': lora_merge.estimate_seconds(
            os.path.getsize(base), shape['output_bytes']),
        # Answered here because the user is about to start something that writes
        # ~26 GB and is entitled to know before, not after.
        'on_failure': ('The merge writes to a .part file and only renames it when it '
                       'finishes. If it fails or is stopped, that partial file is '
                       'deleted and nothing else changes — the base and the LoRAs are '
                       'never modified.'),
    }


def describe(base, loras, **kwargs) -> dict:
    """``plan`` as a payload the UI can render, refusal included. Never raises —
    a disabled button with a reason beats an error toast after the click."""
    try:
        return {'ok': True, **plan(base, loras, **kwargs)}
    except (MergeJobError, lora_merge.MergeError) as e:
        return {'ok': False, 'error': str(e)}


# --- running it -------------------------------------------------------------------

def worker_command(python, spec_path, *, budget_seconds=DEFAULT_BUDGET_SECONDS) -> list:
    """The exact argv. Exposed so a test can assert it without running torch."""
    return [str(python), os.path.abspath(lora_merge.__file__),
            '--spec', str(spec_path), '--progress',
            '--budget-seconds', str(int(budget_seconds or 0))]


def write_spec(info, *, when=None) -> str:
    """Serialise the merge for the worker and return the temp file's path."""
    spec = {
        'base': info['base'],
        'destination': info['destination'],
        'family': info.get('family'),
        'loras': [{'path': item['path'], 'weight': item['weight']}
                  for item in info['loras']],
        'metadata': lora_merge.merge_metadata(
            info['base'], info['loras'], when=when),
    }
    handle, path = tempfile.mkstemp(prefix='lds-merge-', suffix='.json')
    with os.fdopen(handle, 'w', encoding='utf-8') as fh:
        json.dump(spec, fh)
    return path


def run_worker(info, *, progress=None, cancelled=None,
               budget_seconds=DEFAULT_BUDGET_SECONDS) -> dict:
    """Run the merge in the configured interpreter and stream its progress back.

    The child is ``lora_merge.py`` itself, so there is exactly one merge and one
    verification in the product. It prints one ``LDS_MERGE_PROGRESS done total``
    line per tensor and finishes with the ``LDS_MERGE_RESULT`` JSON, which
    already carries the read-back verification.
    """
    spec_path = write_spec(info)
    command = worker_command(info['python'], spec_path, budget_seconds=budget_seconds)
    try:
        try:
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1,
                cwd=os.path.dirname(os.path.abspath(lora_merge.__file__)),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as e:
            raise MergeJobError(
                f'the Python that would do the merge could not be started: '
                f'{info["python"]} ({e}). Check "quantize": {{"python": "…"}} in '
                'config.json, or clear it to fall back on the environment '
                '✨ Score uses.') from e

        result, tail = None, []
        try:
            for line in proc.stdout:
                line = line.rstrip('\n')
                if line.startswith(lora_merge.PROGRESS_PREFIX):
                    parts = line.split()
                    if progress and len(parts) == 3:
                        try:
                            progress(int(parts[1]), int(parts[2]))
                        except Exception:      # noqa: BLE001 — never fatal
                            pass
                elif line.startswith(lora_merge.RESULT_PREFIX):
                    try:
                        result = json.loads(line[len(lora_merge.RESULT_PREFIX):].strip())
                    except ValueError:
                        result = None
                elif line.strip():
                    tail = (tail + [line])[-12:]       # kept for the error message
                if cancelled is not None and cancelled():
                    proc.kill()
                    raise MergeJobError('the merge was stopped')
        finally:
            try:
                proc.wait(timeout=30)
            except Exception:                  # noqa: BLE001
                proc.kill()
    finally:
        try:
            os.remove(spec_path)
        except OSError:
            pass

    if not isinstance(result, dict):
        raise MergeJobError(
            'the merge produced no result. Its last output was: '
            + (' | '.join(tail) or '(nothing)')[:400])
    if not result.get('ok'):
        raise MergeJobError(str(result.get('error') or 'the merge failed'))
    return result


def merge(base, loras, *, progress=None, cancelled=None, **kwargs) -> dict:
    """Do it (BLOCKING, minutes on a 26 GB base). Returns the verified summary.

    Every refusal lives in ``plan``; the space check is repeated here only
    because a download or another job can eat the drive between the two.
    """
    info = plan(base, loras, **kwargs)
    refusal = space_error(_free_bytes(info['destination']), info['output_bytes'])
    if refusal:
        raise MergeJobError(refusal)
    return {**info, **run_worker(info, progress=progress, cancelled=cancelled)}


def start_async(app, base, loras, **kwargs) -> dict:
    """Run it in a daemon thread; progress and outcome live in system_state.

    Refuses immediately — before the thread — when another merge is running or
    the inputs are not usable, so the user sees the rejection on the click
    rather than in a status poll thirty seconds later.
    """
    global _worker_thread
    info = plan(base, loras, **kwargs)
    with _lock:
        reconcile()                            # never refuse for a ghost
        current = status()
        if current.get('status') == 'running' and _running_here():
            raise MergeJobError(
                f'a merge is already running ({current.get("destination_name") or "…"})'
                ' — wait for it to finish')
        _cancel.clear()
        _set('running', info, done=0, total=0)

    def _run():
        with app.app_context():
            def on_progress(done, total):
                _set('running', info, done=done, total=total)
            try:
                result = run_worker(info, progress=on_progress,
                                    cancelled=_cancel.is_set)
                _set('error' if result.get('verify_error') else 'done', info,
                     result=result, error=result.get('verify_error'))
                logger.info('lora merge finished: %s', info['destination_name'])
            except Exception as e:             # noqa: BLE001 — surfaced in state
                # A cancellation is not a failure: the user asked, the .part file
                # is gone and nothing on disk changed. Saying "error" for it would
                # send them looking for a problem that does not exist.
                cancelled = _cancel.is_set()
                _set('cancelled' if cancelled else 'error', info,
                     error=None if cancelled else str(e)[:400])
                if not cancelled:
                    logger.warning('lora merge failed (%s): %s',
                                   info['destination_name'], e)

    _worker_thread = threading.Thread(target=_run, daemon=True)
    _worker_thread.start()
    return info


def cancel() -> dict:
    """Ask a running merge to stop. Idempotent, and safe at any point.

    The worker writes to ``<dst>.part`` and only renames on success, so stopping
    it leaves the base, the LoRAs and any earlier merge exactly as they were.
    """
    if status().get('status') != 'running':
        return {'cancelled': False, 'reason': 'no merge is running'}
    _cancel.set()
    return {'cancelled': True}


def _set(state, info, **extra):
    queue_manager._set_system_state(_STATE_KEY, {
        'status': state,
        'base_name': info['base_name'],
        'destination_name': info['destination_name'],
        'destination_dir': info['destination_dir'],
        'output_bytes': info['output_bytes'],
        'loras': [{'name': i['name'], 'weight': i['weight']} for i in info['loras']],
        **extra,
    }, ttl_seconds=_STATE_TTL)
