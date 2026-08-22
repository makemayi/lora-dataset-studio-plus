"""Topaz Photo AI job queue — one worker thread, one job at a time.

Mirrors JobQueueManager's shape but owns a DIFFERENT GPU consumer: Topaz runs
on its own GPU outside ComfyUI. The worker therefore has its own gate — wait
while training, vision or any ComfyUI work owns the GPU — and claims the GPU
for the duration of a run through the SAME `gpu_exclusive_vision_window` the
vision lane uses, which is what makes the ComfyUI worker pause while Topaz
renders (its `_claim` refuses while a window is active).

The completion link (swap-restore semantics) lives in
`dataset_generation_service.link_topaz_completed`, routed lazily here to avoid
a service->queue import cycle.
"""
from __future__ import annotations
import json
import logging
import os
import pathlib
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime

from ..extensions import db
from ..models import TopazJob
from . import topaz_helper

logger = logging.getLogger(__name__)

POLL_SECONDS = 2.0
JOB_ID_PREFIX = 'topaz-'


def collect_output(tmp_dir, dataset_id, job_id, staged_name=None):
    """Move the Topaz output into the dataset folder, returning the app-side
    path (or None when nothing usable was written).

    Topaz writes `<input-stem>.<ext>` next to the input's basename into the
    output folder; with ``staged_name`` (batch) the EXACT staged basename is
    looked up so outputs map back to the image they came from. This is a
    SEPARATE function so the worker's scheduling can be tested without a real
    file, and the import path owns the naming.
    """
    from . import face_dataset_service as fds
    from .dataset_storage import ensure_dataset_dir

    ds = fds.get_dataset('local', dataset_id)
    if ds is None:
        return None
    out_dir = ensure_dataset_dir(dataset_id)
    candidates = []
    try:
        for name in sorted(p.name for p in tmp_dir.iterdir()):
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                candidates.append(name)
    except OSError:
        return None
    if staged_name:
        matches = [n for n in candidates if n == staged_name]
        if not matches:
            return None
        src = tmp_dir / matches[0]
    else:
        if not candidates:
            return None
        # The first candidate by name is deterministic (Topaz writes one output).
        src = tmp_dir / candidates[0]
    dst_name = f'topaz_{job_id[-8:]}_{src.name}'
    dst = os.path.join(out_dir, dst_name)
    shutil.copy2(str(src), dst)
    return dst


def stage_inputs(inputs, tmp_dir):
    """Copy every source into ONE staging folder with unique names so a single
    tpai folder invocation can process the whole batch. Returns
    {image_id: staged_name} (missing sources are simply left out)."""
    from pathlib import Path as _P
    _P(tmp_dir).mkdir(parents=True, exist_ok=True)
    mapping = {}
    for image_id, input_path in inputs:
        staged = f'img_{image_id}.png'
        try:
            shutil.copy2(input_path, _P(tmp_dir) / staged)
            mapping[image_id] = staged
        except OSError:
            continue        # unreadable source -> left out of the batch
    return mapping


def _run_loop(app, manager):
    while manager._running:
        try:
            with app.app_context():
                worked = manager.process_one()
        except Exception:
            logger.exception('topaz: worker loop error')
            worked = False
        if not worked:
            manager._wake.wait(POLL_SECONDS)
            manager._wake.clear()


class TopazJobManager:
    """One background thread owns all Topaz runs: enqueue/cancel/retry are the
    public surface; process_one() is the synchronous single-step test seam."""

    def __init__(self):
        self._app = None
        self._thread = None
        self._running = False
        self._wake = threading.Event()
        self._lock = threading.RLock()

    def init_app(self, app):
        self._app = app

    # -- public API ---------------------------------------------------------
    def enqueue(self, *, user_id, dataset_id, image_id, input_filename,
                enhancements=None, commit=True):
        job_id = f'{JOB_ID_PREFIX}{uuid.uuid4().hex}'
        row = TopazJob(
            job_id=job_id, user_id=str(user_id),
            dataset_id=int(dataset_id), image_id=int(image_id),
            status='queued', input_filename=input_filename,
            enhancements=json.dumps(enhancements or {'upscale': True}))
        db.session.add(row)
        if commit:
            db.session.commit()
        self._wake.set()
        return job_id

    def enqueue_batch(self, *, user_id, dataset_id, inputs,
                      enhancements=None, commit=True):
        """One job row for MANY images. ``inputs`` is [{image_id, input}] where
        input is the absolute source path (captured before swap-restore clears
        the row's filename). The worker stages them into ONE folder so tpai
        loads its models once for the whole batch."""
        job_id = f'{JOB_ID_PREFIX}{uuid.uuid4().hex}'
        ids = [int(i['image_id']) for i in inputs]
        row = TopazJob(
            job_id=job_id, user_id=str(user_id),
            dataset_id=int(dataset_id),
            image_id=ids[0] if ids else None,   # legacy single-image column
            input_filename=(ids and inputs[0]['input']) or None,
            status='queued',
            image_ids=json.dumps(ids),
            image_inputs=json.dumps({str(i['image_id']): i['input']
                                     for i in inputs}),
            total_images=len(ids),
            enhancements=json.dumps(enhancements or {'upscale': True}))
        db.session.add(row)
        if commit:
            db.session.commit()
        self._wake.set()
        return job_id

    def cancel(self, job_id):
        """Cancel a queued job (or flag a running one). True when cancelled."""
        with self._lock:
            row = TopazJob.query.filter_by(job_id=str(job_id)).first()
            if row is None or row.status in ('completed', 'cancelled'):
                return False
            row.status = 'cancelled'
            row.completed_at = datetime.utcnow()
            db.session.commit()
            self._wake.set()
            return True

    def retry(self, job_id):
        """Requeue a failed job. For a BATCH, only the FAILED images are
        collected into a NEW batch job — successful ones are never re-run."""
        with self._lock:
            row = TopazJob.query.filter_by(job_id=str(job_id)).first()
            if row is None or row.status != 'failed':
                return False
            results = {}
            try:
                results = json.loads(row.image_results or '{}')
            except (TypeError, ValueError):
                pass
            failed = [int(i) for i, r in results.items()
                      if r.get('status') == 'failed']
            if failed:
                inputs = []
                try:
                    inputs_p = json.loads(row.image_inputs or '{}')
                except (TypeError, ValueError):
                    inputs_p = {}
                for img_id in failed:
                    p = inputs_p.get(str(img_id))
                    if p:
                        inputs.append({'image_id': img_id, 'input': p})
                if inputs:
                    self.enqueue_batch(
                        user_id=row.user_id, dataset_id=row.dataset_id,
                        inputs=inputs)
                    self._wake.set()
                    return True
            row.status = 'queued'
            row.error_message = None
            row.image_results = None
            row.done_images = 0
            row.retry_count = (row.retry_count or 0) + 1
            row.completed_at = None
            db.session.commit()
            self._wake.set()
            return True

    def has_work(self):
        return (TopazJob.query.filter(TopazJob.status.in_(
            ('queued', 'running'))).first() is not None)

    def recent(self, limit=50):
        return (TopazJob.query.order_by(TopazJob.created_at.desc())
                .limit(limit).all())

    def link_completed(self, row):
        """Route a finished job to the swap-restore linker (lazy import avoids
        a service->queue import cycle). Test seam: monkeypatch this method."""
        try:
            from .dataset_generation_service import link_topaz_completed
            link_topaz_completed(row)
        except Exception:
            logger.exception('topaz: completion link failed for %s', row.job_id)

    # -- worker --------------------------------------------------------------
    def _gpu_available(self):
        """True only when Topaz may claim the GPU (fail closed)."""
        from ..job_queue import queue_manager
        try:
            from ..gpu_window import vision_gpu_window_blocks_gpu
        except Exception:
            vision_gpu_window_blocks_gpu = lambda: True
        if vision_gpu_window_blocks_gpu():
            return False
        if queue_manager._get_system_state('vision_in_progress', False):
            return False
        if queue_manager._get_system_state('training_in_progress', False):
            return False
        if queue_manager.has_comfyui_work():
            return False
        return True

    def process_one(self):
        """Claim and run one queued Topaz job, or False when gated/idle."""
        from ..gpu_window import gpu_exclusive_vision_window
        from . import topaz_helper as th

        with self._lock:
            if not self._gpu_available():
                return False
            row = (TopazJob.query.filter_by(status='queued')
                   .order_by(TopazJob.created_at.asc()).first())
            if row is None:
                return False
            row.status = 'running'
            row.started_at = datetime.utcnow()
            db.session.commit()
            job_id = row.job_id
            dataset_id = row.dataset_id
            input_filename = row.input_filename
            toggles = json.loads(row.enhancements or '{"upscale": true}')

        output_path = None
        try:
            with gpu_exclusive_vision_window(flag_ttl=300):
                exe = th.preflight()
                batch = self._batch_inputs(row)
                if batch:
                    status, message, results = self._run_batch(
                        exe, row, batch, toggles)
                else:
                    with tempfile.TemporaryDirectory(prefix='lds-topaz-') as tmp:
                        status, message = th.run_tpai(
                            exe, input_filename, tmp, **toggles)
                        if status == 'ok':
                            output_path = collect_output(
                                pathlib.Path(tmp), dataset_id, job_id)
                            if output_path is None:
                                status, message = 'unknown', (
                                    'Topaz finished but wrote nothing readable')
                        results = {str(row.image_id):
                                   {'status': 'completed' if status == 'ok'
                                    else 'failed',
                                    'output_filename': output_path,
                                    'error': None if status == 'ok' else message}}
        except Exception as e:
            status, message = 'unknown', f'{type(e).__name__}: {e}'
            results = {}

        with self._lock:
            row = TopazJob.query.filter_by(job_id=job_id).first()
            if row is None:
                return True
            if row.status == 'cancelled':
                self._restore_unfinished(row, results)
                row.completed_at = datetime.utcnow()
                db.session.commit()
                return True
            row.output_filename = output_path
            row.image_results = json.dumps(results or {})
            done = sum(1 for r in (results or {}).values()
                       if r.get('status') == 'completed')
            row.done_images = done
            total = row.total_images or len(batch) or 1
            if done == total:
                row.status = 'completed'
            elif done == 0:
                row.status = 'failed'
                row.error_message = message or 'Topaz batch failed'
            else:
                row.status = 'failed'
                row.error_message = (f'{done} of {total} succeeded; '
                                     f'{total - done} failed — retry reruns '
                                     f'only the failed ones.')
            row.completed_at = datetime.utcnow()
            db.session.commit()
            self.link_completed(row)
        return True

    def _batch_inputs(self, row):
        """[(image_id, input_path)] for a batch row, or [] for a single image."""
        if not row.image_ids:
            return []
        try:
            ids = json.loads(row.image_ids or '[]')
            inputs = json.loads(row.image_inputs or '{}')
        except (TypeError, ValueError):
            return []
        return [(str(i), inputs.get(str(i))) for i in ids
                if inputs.get(str(i))]

    def _run_batch(self, exe, row, batch, toggles):
        """Stage all inputs into ONE folder, run tpai ONCE, map outputs back,
        and link every finished tile. Returns (status, message, results)."""
        from . import topaz_helper as th
        from .dataset_generation_service import link_topaz_image
        results = {}
        try:
            with tempfile.TemporaryDirectory(prefix='lds-topaz-') as tmp_in, \
                 tempfile.TemporaryDirectory(prefix='lds-topaz-out-') as tmp_out:
                staged = stage_inputs(batch, tmp_in)
                status, message = th.run_tpai(
                    exe, str(pathlib.Path(tmp_in)), str(pathlib.Path(tmp_out)),
                    **toggles)
                if status != 'ok':
                    for image_id, _ in batch:
                        results[str(image_id)] = {
                            'status': 'failed', 'output_filename': None,
                            'error': message or 'Topaz batch failed'}
                    return status, message, results
                for image_id, _ in batch:
                    staged_name = staged.get(image_id)
                    if not staged_name:
                        results[str(image_id)] = {
                            'status': 'failed', 'output_filename': None,
                            'error': 'source could not be staged'}
                        continue
                    out_path = collect_output(
                        pathlib.Path(tmp_out), row.dataset_id, row.job_id,
                        staged_name=staged_name)
                    if out_path is None:
                        results[str(image_id)] = {
                            'status': 'failed', 'output_filename': None,
                            'error': 'Topaz wrote nothing for this image'}
                        continue
                    results[str(image_id)] = {
                        'status': 'completed', 'output_filename': out_path,
                        'error': None}
                    try:
                        link_topaz_image(row.dataset_id, int(image_id),
                                         out_path, 'completed')
                    except Exception as e:                       # noqa: BLE001
                        results[str(image_id)] = {
                            'status': 'failed', 'output_filename': None,
                            'error': f'could not attach the result: {e}'}
                    with self._lock:
                        row.done_images = sum(
                            1 for r in results.values()
                            if r.get('status') == 'completed')
                        db.session.commit()
            return status, message, results
        except Exception as e:
            return 'unknown', f'{type(e).__name__}: {e}', results

    def _restore_unfinished(self, row, results):
        """Cancelled: restore every image that did not finish. Successful ones
        stay — their tiles are already linked and the swap is settled."""
        from .dataset_generation_service import restore_swapped_original
        from ..models import FaceDatasetImage
        finished = {int(i) for i, r in (results or {}).items()
                    if r.get('status') == 'completed'}
        for img_id in (json.loads(row.image_ids or '[]') if row.image_ids else []):
            if int(img_id) in finished:
                continue
            img = db.session.get(FaceDatasetImage, int(img_id))
            if img is not None and img.status == 'pending' and img.job_id == row.job_id:
                try:
                    restore_swapped_original(img)
                except Exception:
                    logger.exception('topaz: cancel restore failed for img %s', img_id)

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=_run_loop, args=(self._app, self),
            name='topaz-job-worker', daemon=True)
        self._thread.start()

    def stop(self, timeout=5):
        self._running = False
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None


topaz_queue = TopazJobManager()
