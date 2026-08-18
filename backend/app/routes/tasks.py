"""🗂️ Task Center — the unified view over every running/queued background job.

One endpoint (GET /api/tasks/overview) feeds both the Tasks page and the nav
badge. It synthesizes three kinds of rows:

  * image    — real ImageGenerationQueue rows (datasets, studio, test studio…)
  * training — durable SystemState flags, rendered as one row
  * vision   — durable SystemState flags, rendered as one row

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

    by = {}
    for r in rows:
        by[r.status] = by.get(r.status, 0) + 1
    summary = {
        'queued': by.get('pending', 0),
        'paused': by.get(AWAITING_COMFYUI, 0),
        'running': by.get('processing', 0) + by.get('sent_to_comfy', 0),
        'today_done': by.get('completed', 0),
        'today_failed': by.get('failed', 0),
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
    """Re-enqueue one failed job (ComfyUI image job, or a topaz- job)."""
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


@bp.post('/<job_id>/cancel')
def task_cancel(job_id):
    """Cancel one job (queued, paused, or running) — ComfyUI image jobs and
    topaz- jobs both route here, dispatched on the id prefix."""
    if str(job_id).startswith('topaz-'):
        from ..services.topaz_job_queue import topaz_queue
        if not topaz_queue.cancel(job_id):
            return jsonify({'error': 'job not found or already finished'}), 404
        return jsonify({'ok': True, 'job_id': job_id})
    outcome = queue_manager.cancel_job_outcome(str(job_id), user_id=LOCAL_USER)
    if outcome == 'missing':
        return jsonify({'error': 'job not found'}), 404
    if outcome in ('restart_required', 'barrier_corrupt'):
        return jsonify({'error': outcome}), 409
    return jsonify({'ok': True, 'job_id': job_id, 'outcome': outcome})
