"""What a harvested video LoRA is made of, written where the LoRA is.

WHY NOT THE CHECKPOINT REGISTRY
-------------------------------
The image lane files every launch as a `TrainingRunRecord`: a fingerprint over
the face dataset's IMAGES and their caption hashes, hung off a `face_dataset.id`
and versioned per dataset. None of those three things exists for a video run —
its dataset is a folder of clips, and its `dataset_id` names the OTHER table. A
video run filed there would sit in some face dataset's lineage graph forever,
under a version number that is not its. That is the same collision
`cloud_run_dataset` exists to prevent, one table further along.

So the video lane records its provenance where its output actually is: one JSON
file in the run's checkpoint store, beside the `.safetensors` it describes. That
placement is not a fallback, it is the property worth having — a folder of
weights found on a disk months later can still say which dataset made it, at
which target, from which run, and which files belong together. A database row
cannot travel with the file it describes; this can.

Genealogy stays inside the lane too: a continuation stamps `parent_run_id` — the
CloudTrainingRun id it grew from — so three continuations read as one 3000-step
lineage rather than three unrelated runs.

PASTE-SAFE BY CONSTRUCTION
--------------------------
This file sits next to weights people share. It carries basenames and never a
directory: a machine path in a shared artefact is the leak the repo's privacy
rule is about, and here it would be published by the act of sharing the LoRA.
"""
import json
import logging
import os

from . import cloud_run_dataset as crd

logger = logging.getLogger(__name__)

MANIFEST_NAME = 'lds_video_run.json'


def lineage_dir(run) -> str | None:
    """Where the manifest goes: this run's durable checkpoint store, falling
    back to its staging dir on an install that predates the store. None when the
    run has neither — there is nothing to describe, and inventing a folder for a
    manifest with no weights beside it would only make an orphan.

    Creates nothing. `read` calls this too, and a reader that makes a directory
    on disk as a side effect of answering "is there a manifest?" is a surprise
    nobody goes looking for; `record` does the one mkdir, where it belongs."""
    from . import cloud_training as ct
    return ct.checkpoint_store_dir(run, create=False) or run.staging_dir or None


def build(run) -> dict:
    """The manifest. Names only — no path, no user, no machine."""
    from . import cloud_training as ct
    params = {}
    try:
        params = json.loads(run.train_params or '{}')
    except ValueError:
        params = {}
    if not isinstance(params, dict):
        params = {}
    row = crd.dataset_row(run)
    return {
        'schema': 1,
        'run_id': run.id,
        'dataset_table': crd.table_of(run),
        'dataset_id': run.dataset_id,
        # Best-effort, like every other describer of an old run: a deleted
        # dataset degrades to null, never to another dataset's name.
        'dataset_name': getattr(row, 'name', None),
        'target_profile': (params.get('target_profile')
                           or getattr(row, 'target_profile', None)),
        'frames': params.get('frames') or getattr(row, 'frames', None),
        'base_model': params.get('base_model') or '',
        'steps': params.get('steps'),
        'parent_run_id': params.get('parent_run_id'),
        'resumed_from_step': params.get('resume_step'),
        'gpu': run.gpu_name,
        'status': run.status,
        'finished_at': run.finished_at.isoformat() if run.finished_at else None,
        'files': sorted(ct.run_checkpoint_files(run)),
    }


def record(run) -> str | None:
    """Write the manifest beside this run's weights; return its path or None.

    Best-effort by contract, exactly like the image lane's registry call: a run
    that trained for two hours must not be marked failed because a bookkeeping
    file could not be written."""
    try:
        if not crd.is_video(run):
            return None
        folder = lineage_dir(run)
        if not folder:
            return None
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, MANIFEST_NAME)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(build(run), fh, indent=2, sort_keys=True)
        return path
    except Exception:
        logger.exception('video run lineage not written (the run is unaffected)')
        return None


def read(run) -> dict | None:
    """The manifest of a run that has one, or None."""
    folder = lineage_dir(run)
    if not folder:
        return None
    try:
        with open(os.path.join(folder, MANIFEST_NAME), encoding='utf-8') as fh:
            parsed = json.load(fh)
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
