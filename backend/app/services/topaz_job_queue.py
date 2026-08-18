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


def collect_output(tmp_dir, dataset_id, job_id):
    """Move the Topaz output into the dataset folder, returning the app-side
    path (or None when nothing usable was written).

    Topaz writes `<input-stem>.<ext>` next to the input's basename into the
    output folder; this is a SEPARATE function so the worker's scheduling can
    be tested without a real file, and the import path owns the naming.
    """
    from .. import config as cfg
    from . import face_dataset_service as fds

    ds = fds.get_dataset('local', dataset_id)
    if ds is None:
        return None
    out_dir = cfg.dataset_dir(dataset_id)
    candidates = []
    try:
        for name in sorted(p.name for p in tmp_dir.iterdir()):
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                candidates.append(name)
    except OSError:
        return None
    if not candidates:
        return None
    # The first candidate by name is deterministic (Topaz writes one output).
    src = tmp_dir / candidates[0]
    dst_name = f'topaz_{job_id[-8:]}_{candidates[0]}'
    dst = os.path.join(out_dir, dst_name)
    shutil.copy2(str(src), dst)
    return dst


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
        """Requeue one failed job. True when requeued."""
        with self._lock:
            row = TopazJob.query.filter_by(job_id=str(job_id)).first()
            if row is None or row.status != 'failed':
                return False
            row.status = 'queued'
            row.error_message = None
            row.output_filename = None
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

        try:
            with gpu_exclusive_vision_window(flag_ttl=300):
                exe = th.preflight()
                with tempfile.TemporaryDirectory(prefix='lds-topaz-') as tmp:
                    status, message = th.run_tpai(
                        exe, input_filename, tmp, **toggles)
                    output_path = None
                    if status == 'ok':
                        output_path = collect_output(
                            pathlib.Path(tmp), dataset_id, job_id)
                        if output_path is None:
                            status, message = 'unknown', (
                                'Topaz finished but wrote nothing readable')
        except Exception as e:
            status, message = 'unknown', f'{type(e).__name__}: {e}'
            output_path = None

        with self._lock:
            row = TopazJob.query.filter_by(job_id=job_id).first()
            if row is None:
                return True
            if row.status == 'cancelled':
                row.completed_at = datetime.utcnow()
                db.session.commit()
                return True
            row.output_filename = output_path
            if status == 'ok':
                row.status = 'completed'
                row.completed_at = datetime.utcnow()
                db.session.commit()
            else:
                row.status = 'failed'
                row.error_message = message or 'Topaz upscale failed'
                row.completed_at = datetime.utcnow()
                db.session.commit()
            self.link_completed(row)
        return True

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
