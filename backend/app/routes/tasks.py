"""🗂️ Task Center — the unified view over every running/queued background job.

One endpoint (GET /api/tasks/overview) feeds both the Tasks page and the nav
badge. It synthesizes three kinds of rows:

  * image    — real ImageGenerationQueue rows (datasets, studio, test studio…)
  * topaz    — TopazJob rows (one per upscale, batches carry k/N)
  * training — durable SystemState flags, rendered as one row
  * vision   — durable SystemState flags, rendered as one row
  * batch    — dataset_activity entries: captioning, watermark passes, face
               analysis, generation fan-outs, exports, backups, subject trim…
  * bank     — bank_jobs entries: scan, faces, embed, promote, dedup…

The last two were the hole this page had while claiming to list everything: the
app runs long work through THREE registries and the Task Center knew one and a
half of them. A caption pass over 400 images, a bank scan over thousands of
files, an hour-long backup — all of it was live, cancellable in several cases,
and visible only on whichever page happened to own it. Both registries are
IN-MEMORY and die with the process, which is exactly why they belong on a page
that only ever claims to show what is running NOW.

Resource linkage (spec §2.5): prefer job_metadata.source_image_id (stamped by
in-place engines), fall back to a FaceDatasetImage.job_id reverse lookup so
legacy jobs still answer "which image?".

Retry and cancel are thin routes over the queue manager's existing, lock-safe
outcomes — no second scheduling authority grows here.
"""
import json
from flask import Blueprint, jsonify

from ..config import LOCAL_USER
from ..extensions import db
from ..job_queue import (AWAITING_COMFYUI, GPU_ARBITER_LOCK, queue_manager)
from ..models import FaceDatasetImage, ImageGenerationQueue, SystemState
from .system import _comfyui_connection, _dataset_name

bp = Blueprint('tasks', __name__, url_prefix='/api/tasks')


def _md(row) -> dict:
    try:
        return json.loads(row.job_metadata or '{}') or {}
    except (TypeError, ValueError):
        return {}


def _resource(row, md) -> dict:
    """Normalized resource link: {'type':'image'|'dataset', dataset_id, image_id}."""
    dataset_id = md.get('dataset_id')
    image_id = md.get('source_image_id')
    if image_id is not None:
        return {'type': 'image', 'dataset_id': dataset_id, 'image_id': image_id}
    if dataset_id is not None:
        linked = (FaceDatasetImage.query
                  .filter_by(job_id=row.job_id).first())
        if linked is not None:
            return {'type': 'image', 'dataset_id': linked.dataset_id,
                    'image_id': linked.id}
        return {'type': 'dataset', 'dataset_id': dataset_id}
    return {'type': 'dataset', 'dataset_id': None}


def _image_row(row) -> dict:
    md = _md(row)
    status = row.status
    return {
        'job_id': row.job_id,
        'kind': 'image',
        'title': md.get('model_name') or 'image job',
        'source': _dataset_name(md.get('dataset_id')),
        'status': status,
        'created_at': row.created_at.isoformat() if row.created_at else None,
        'progress': None,
        'error': row.error_message,
        'resource': _resource(row, md),
        'actions': (['cancel'] if status in ('pending', AWAITING_COMFYUI,
                                             'processing', 'sent_to_comfy')
                    else []) + (['retry'] if status == 'failed' else []),
    }


def _flag_row(kind, title) -> dict:
    return {
        'job_id': f'{kind}-in-progress',
        'kind': kind,
        'title': title,
        'source': None,
        'status': 'running',
        'created_at': None,
        'progress': None,
        'error': None,
        'resource': {'type': 'dataset', 'dataset_id': None},
        'actions': [],
    }


def _topaz_row(row) -> dict:
    """One TopazJob rendered for the Task Center list. A batch carries its
    image count in the title and a k/N progress once it starts."""
    total = row.total_images or 1
    title = f'Topaz upscale ({total})' if total > 1 else 'Topaz upscale'
    progress = None
    if total > 1 and row.status in ('running', 'completed', 'failed', 'cancelled'):
        progress = f'{row.done_images or 0}/{total}'
    return {
        'job_id': row.job_id,
        'kind': 'topaz',
        'title': title,
        'source': _dataset_name(row.dataset_id),
        'status': row.status,
        'created_at': row.created_at.isoformat() if row.created_at else None,
        'progress': progress,
        'error': row.error_message,
        'resource': {'type': 'image', 'dataset_id': row.dataset_id,
                     'image_id': row.image_id or 0},
        'actions': (['cancel'] if row.status in ('queued', 'running')
                    else []) + (['retry'] if row.status == 'failed' else []),
    }


# Human titles for `dataset_activity.KINDS`. Explicit rather than humanised from
# the slug, because several would read wrong: "Analyze faces" is a scoring pass,
# "Classify" is framing, "Trim" is the subject crop, "Generate" is a whole
# fan-out. `test_task_center_registries.py` fails when a KIND has no entry, so
# adding a kind cannot silently ship a row titled after its variable name.
ACTIVITY_LABEL = {
    'watermark_detect': 'Finding watermarks',
    'watermark_clean': 'Cleaning watermarks',
    'caption': 'Captioning',
    'recaption': 'Re-captioning',
    'analyze_faces': 'Face scoring',
    'classify': 'Framing classification',
    'generate': 'Generating variations',
    'improve': 'Improve & upscale',
    'edit_reference': 'Reference edit',
    'bank_export': 'Export to bank',
    'bank_import': 'Import from bank',
    'training_export': 'Training export',
    'backup': 'Backup',
    'trim': 'Subject trim',
}

# A few bank kinds whose slug is not a sentence. Everything else is humanised, so
# a bank lane added later still renders as words instead of vanishing — there is
# no allow-list on that side for a label map to be held to.
BANK_LABEL = {
    'faces': 'Group by person',
    'embed': 'Semantic embedding',
    'scan': 'Folder scan',
    'promote': 'Promote to dataset',
    'bank_promote': 'Promote to bank',
    'semantic_dedup': 'Semantic de-duplication',
    'semantic_engine': 'Semantic engine switch',
    'delete_rejected': 'Deleting rejected',
    'watermark_inpaint': 'Watermark inpaint',
    'watermark_crop': 'Watermark crop',
    'medium': 'Sort by medium',
    'framing': 'Sort by framing',
    'score': 'Face scoring',
}


def _humanise(kind):
    return str(kind or 'job').replace('_', ' ').capitalize()


def _progress_text(done, total):
    """`k/N`, and only when N is known. A bare "0/0" reads as a stalled pass and
    a lone count reads as a total — both are worse than saying nothing."""
    try:
        total = int(total or 0)
        done = int(done or 0)
    except (TypeError, ValueError):
        return None
    return f'{done}/{total}' if total > 0 else None


def _bank_name(bank_id):
    from ..models import ImageBank
    try:
        row = ImageBank.query.filter_by(id=int(bank_id)).first()
    except (TypeError, ValueError):
        return None
    return row.name if row is not None else None


def _activity_row(entry) -> dict:
    """One dataset_activity batch as a Task Center row.

    Cancel is offered ONLY for kinds that have a cooperative stop, and the scope
    test below mirrors dataset_activity's own arming scopes exactly. A Cancel
    button on a pass with no seam to stop at would be a promise the user only
    discovers is empty by clicking it.
    """
    from ..services import dataset_activity as da
    kind = entry.get('kind')
    stoppable = (kind in da.CANCELLABLE_KINDS or kind in da.IMPROVE_KINDS
                 or kind in da.WATERMARK_KINDS or kind in da.TRIM_KINDS
                 or kind == 'generate')
    dataset_id = entry.get('dataset_id')
    return {
        'job_id': f"activity-{entry.get('token')}",
        'kind': 'batch',
        'title': ACTIVITY_LABEL.get(kind) or _humanise(kind),
        'source': _dataset_name(dataset_id),
        # `cancelling` means the stop is armed and the worker has not reached the
        # next item boundary yet. It is still RUNNING, so it keeps the running
        # status and says so in the progress text instead of chipping as cancelled.
        'status': 'running',
        'created_at': None,
        'progress': ('Stopping...' if entry.get('cancelling')
                     else _progress_text(entry.get('done'), entry.get('total'))),
        'error': None,
        'resource': {'type': 'dataset', 'dataset_id': dataset_id},
        'actions': ['cancel'] if (stoppable and not entry.get('cancelling')) else [],
    }


def _bank_row(bank_id, snap) -> dict:
    """One bank_jobs pass as a Task Center row.

    Bank jobs carry an ETA of their own, and it is shown beside the count because
    these are the longest passes in the app by an order of magnitude — a bare
    `120/8000` on a folder scan tells you nothing you wanted to know.
    """
    error = snap.get('error')
    if error:
        status = 'failed'
    elif snap.get('cancelled'):
        status = 'cancelled'
    elif snap.get('finished'):
        status = 'completed'
    else:
        status = 'running'
    progress = _progress_text(snap.get('done'), snap.get('total'))
    eta = snap.get('eta_seconds')
    if status == 'running' and eta:
        mins = max(1, int(eta) // 60)
        progress = f'{progress} - ~{mins} min left' if progress else f'~{mins} min left'
    return {
        'job_id': f'bank-{bank_id}',
        'kind': 'bank',
        'title': BANK_LABEL.get(snap.get('kind')) or _humanise(snap.get('kind')),
        'source': _bank_name(bank_id),
        'status': status,
        'created_at': None,
        'progress': progress,
        'error': error or None,
        # A bank is not a dataset, and sending its id down the dataset link would
        # open the wrong page. The row carries no resource rather than a wrong one.
        'resource': {'type': 'dataset', 'dataset_id': None},
        'actions': ['cancel'] if status == 'running' else [],
    }


@bp.get('/overview')
def tasks_overview():
    rows = (ImageGenerationQueue.query
            .order_by(ImageGenerationQueue.created_at.desc())
            .limit(200).all())
    tasks = [_image_row(r) for r in rows]

    if queue_manager._get_system_state('training_in_progress', False):
        tasks.insert(0, _flag_row('training', 'Training (local)'))
    if queue_manager._get_system_state('vision_in_progress', False):
        tasks.insert(0, _flag_row('vision', 'Vision inference (Ollama)'))

    from ..services.topaz_job_queue import topaz_queue
    for row in topaz_queue.recent(limit=50):
        tasks.append(_topaz_row(row))

    # The two in-memory registries. Both are live-only by design, so they are
    # PREPENDED: a running caption pass matters more than yesterday's completed
    # image job, and this list is ordered newest-first for durable rows only.
    from ..services import bank_jobs, dataset_activity
    live = [_activity_row(e) for e in dataset_activity.list_every()]
    live += [_bank_row(bank_id, snap) for bank_id, snap in bank_jobs.list_every()]
    tasks = live + tasks

    by = {}
    for r in rows:
        by[r.status] = by.get(r.status, 0) + 1
    # The strip counts what the LIST shows. Counting only queue rows here is how
    # "0 running" sat above a visibly running caption pass.
    extra_running = sum(1 for t in live if t['status'] == 'running')
    extra_failed = sum(1 for t in live if t['status'] == 'failed')
    summary = {
        'queued': by.get('pending', 0),
        'paused': by.get(AWAITING_COMFYUI, 0),
        'running': (by.get('processing', 0) + by.get('sent_to_comfy', 0)
                    + extra_running),
        'today_done': by.get('completed', 0),
        'today_failed': by.get('failed', 0) + extra_failed,
    }
    return jsonify({
        'status': {
            'comfyui': _comfyui_connection(None),
            'gpu': {
                'training': bool(queue_manager._get_system_state(
                    'training_in_progress', False)),
                'vision': bool(queue_manager._get_system_state(
                    'vision_in_progress', False)),
            },
            'summary': summary,
        },
        'tasks': tasks,
    })


@bp.post('/<job_id>/retry')
def task_retry(job_id):
    """Re-enqueue one failed job (ComfyUI image job, or a topaz- job).

    `activity-` and `bank-` rows are NOT retryable and say so. Their registries
    are in-memory and hold no recipe — re-running a caption pass means calling the
    screen that owns it, with its engine, its pile and its settings. Answering
    404 here would read as "that job is gone" for a pass that is very much there.
    """
    if str(job_id).startswith(('activity-', 'bank-')):
        return jsonify({'error': 'this pass cannot be retried from here — '
                                 'start it again from the page that owns it'}), 409
    if str(job_id).startswith('topaz-'):
        from ..services.topaz_job_queue import topaz_queue
        if not topaz_queue.retry(job_id):
            return jsonify({'error': 'job not found or not failed'}), 404
        return jsonify({'ok': True, 'job_id': job_id})
    job = (ImageGenerationQueue.query
           .filter_by(job_id=str(job_id), user_id=LOCAL_USER).first())
    if job is None:
        return jsonify({'error': 'job not found'}), 404
    if job.status != 'failed':
        return jsonify({'error': f'job is {job.status}, not failed'}), 409
    with GPU_ARBITER_LOCK:
        job.update_status('pending')
        job.error_message = None
        job.result_filename = None
        job.comfyui_prompt_id = None
        job.retry_count = (job.retry_count or 0) + 1
        db.session.commit()
    return jsonify({'ok': True, 'job_id': job_id})


def _cancel_activity(job_id):
    """Stop one dataset_activity batch. The row id is `activity-<token>` and the
    token is `<dataset_id>:<kind>:<n>`, so the target is recoverable without a
    second lookup.

    Routes to the SAME call the owning screen's own Stop button uses, never to a
    thread kill: these workers stop at an item boundary and keep everything they
    have already written. `generate` is the exception — it has no per-item seam
    of its own and is stopped by cancelling its pending queue rows, which is what
    the dataset page's stop button does too.

    A stop that finds nothing live answers 404 rather than pretending: by the time
    a click lands, a short pass may already be over.
    """
    from ..services import dataset_activity as da
    token = str(job_id)[len('activity-'):]
    parts = token.split(':')
    if len(parts) < 2:
        return jsonify({'error': 'job not found'}), 404
    try:
        dataset_id = da.normalize_dataset_id(parts[0])
    except ValueError:
        return jsonify({'error': 'job not found'}), 404
    kind = parts[1]
    if kind == 'generate':
        from ..services import face_dataset_service as svc
        result = svc.cancel_pending(LOCAL_USER, dataset_id)
        return jsonify({'ok': True, 'job_id': job_id, 'outcome': result})
    if kind not in (da.CANCELLABLE_KINDS + da.IMPROVE_KINDS
                    + da.WATERMARK_KINDS + da.TRIM_KINDS):
        return jsonify({'error': f'{kind} cannot be stopped once started'}), 409
    if not da.request_cancel(dataset_id, (kind,)):
        return jsonify({'error': 'job not found or already finished'}), 404
    return jsonify({'ok': True, 'job_id': job_id, 'outcome': 'stopping'})


@bp.post('/<job_id>/cancel')
def task_cancel(job_id):
    """Cancel one job (queued, paused, or running). Four lanes route here and are
    dispatched on the id prefix: ComfyUI image jobs (bare uuid), `topaz-`,
    `activity-` (dataset batches) and `bank-` (bank passes)."""
    if str(job_id).startswith('topaz-'):
        from ..services.topaz_job_queue import topaz_queue
        if not topaz_queue.cancel(job_id):
            return jsonify({'error': 'job not found or already finished'}), 404
        return jsonify({'ok': True, 'job_id': job_id})
    if str(job_id).startswith('activity-'):
        return _cancel_activity(job_id)
    if str(job_id).startswith('bank-'):
        from ..services import bank_jobs
        try:
            bank_id = int(str(job_id)[len('bank-'):])
        except (TypeError, ValueError):
            return jsonify({'error': 'job not found'}), 404
        if not bank_jobs.cancel(bank_id):
            return jsonify({'error': 'job not found or already finished'}), 404
        return jsonify({'ok': True, 'job_id': job_id, 'outcome': 'stopping'})
    outcome = queue_manager.cancel_job_outcome(str(job_id), user_id=LOCAL_USER)
    if outcome == 'missing':
        return jsonify({'error': 'job not found'}), 404
    if outcome in ('restart_required', 'barrier_corrupt'):
        return jsonify({'error': outcome}), 409
    return jsonify({'ok': True, 'job_id': job_id, 'outcome': outcome})
