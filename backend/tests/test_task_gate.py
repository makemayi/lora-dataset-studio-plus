"""Task-center gate: ComfyUI health gate pauses jobs instead of wedging.

The three behaviours this file pins are the core of the Task Center dispatch
contract (spec §4.1-4.4):

  * ComfyUI offline -> pending jobs become `awaiting_comfyui`, NO recovery
    barrier is created, nothing is claimed.
  * ComfyUI back   -> paused jobs flip to pending and submit in strict FIFO.
  * A refused TCP connect mid-flight -> the job DEFERS back to pending (the
    gate pauses it next tick) instead of erecting an unknown_submit barrier.
  * A restarted ComfyUI that /queue PROVES lost the prompt -> that one job
    fails deterministically and the queue keeps moving.
"""
import json
from datetime import datetime
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_host_ollama_gpu_fence(monkeypatch):
    """Same fence isolation the queue unit tests rely on."""
    monkeypatch.setattr(
        'app.services.vision_keepalive.ensure_released_for_comfy',
        lambda: True,
    )


def test_offline_gate_marks_pending_as_awaiting_comfyui_without_barrier(app, monkeypatch):
    """ComfyUI offline => process_one must NOT claim, must mark pending jobs
    awaiting_comfyui, and must NOT create a recovery barrier."""
    from app.job_queue import AWAITING_COMFYUI, queue_manager
    from app.models import ImageGenerationQueue, SystemState

    monkeypatch.setattr('app.job_queue._comfyui_ready_for_submit', lambda: False)
    with app.app_context():
        queue_manager.add_job(workflow_data={'1': {}})
        queue_manager.add_job(workflow_data={'1': {}})

        assert queue_manager.process_one() is False

        statuses = [r.status for r in ImageGenerationQueue.query.all()]
        assert statuses == [AWAITING_COMFYUI, AWAITING_COMFYUI]
        assert SystemState.query.filter_by(
            key='comfyui_stalled_barrier').first() is None


def test_online_gate_resumes_paused_jobs_and_submits_fifo(app, monkeypatch):
    """ComfyUI back => awaiting_comfyui flips to pending, oldest submitted first."""
    from app.job_queue import AWAITING_COMFYUI, queue_manager
    from app.models import ImageGenerationQueue

    monkeypatch.setattr('app.job_queue._comfyui_ready_for_submit', lambda: False)
    with app.app_context():
        j1 = queue_manager.add_job(workflow_data={'1': {}})
        j2 = queue_manager.add_job(workflow_data={'1': {}})
        queue_manager.process_one()          # offline tick: both paused
        assert ImageGenerationQueue.query.filter_by(status=AWAITING_COMFYUI).count() == 2

        monkeypatch.setattr('app.job_queue._comfyui_ready_for_submit', lambda: True)
        submitted = []
        with patch('app.job_queue._submit', side_effect=lambda w, c: submitted.append(c) or 'p1'), \
             patch('app.job_queue._poll_outputs', return_value=('out.png', False)):
            assert queue_manager.process_one() is True
        assert submitted == [j1]              # strict FIFO: the OLDER job went first
        # The resume pass flipped j2 back to pending before the pick; only j1
        # was claimed and completed. j2 sits resumed, waiting its turn.
        assert ImageGenerationQueue.query.filter_by(job_id=j1).one().status == 'completed'
        assert ImageGenerationQueue.query.filter_by(job_id=j2).one().status == 'pending'


def test_cancel_paused_job_treats_it_like_pending(app, monkeypatch):
    """cancel_job_outcome must cancel an awaiting_comfyui job exactly like a
    pending one (spec §4.5)."""
    from app.job_queue import AWAITING_COMFYUI, queue_manager
    from app.models import ImageGenerationQueue

    monkeypatch.setattr('app.job_queue._comfyui_ready_for_submit', lambda: False)
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
        queue_manager.process_one()
        assert ImageGenerationQueue.query.filter_by(job_id=jid).one().status == AWAITING_COMFYUI

        assert queue_manager.cancel_job_outcome(jid) == 'cancelled'
        assert ImageGenerationQueue.query.filter_by(job_id=jid).one().status == 'cancelled'


def test_connection_refused_defers_job_back_to_pending_not_barrier(app, monkeypatch):
    """A refused TCP connect means NOTHING was sent: the job must go back to
    pending (the gate pauses it next tick), never to a stalled barrier."""
    from app.job_queue import AWAITING_COMFYUI, queue_manager
    from app.models import ImageGenerationQueue, SystemState

    jid = None

    def refused_submit(workflow, client_id):
        from app.job_queue import _ComfySubmitRefused
        raise _ComfySubmitRefused('COMFYUI_REFUSED: connection refused')

    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})

        with patch('app.job_queue._submit', side_effect=refused_submit):
            assert queue_manager.process_one() is True

        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        # Refused => deferred AND immediately paused (the handler flips the rest
        # to awaiting_comfyui so the worker does not retry a dead endpoint every
        # tick). The gate resumes it FIFO when ComfyUI answers. Never failed,
        # never stalled, never a barrier.
        assert row.status == AWAITING_COMFYUI
        assert SystemState.query.filter_by(key='comfyui_stalled_barrier').first() is None


def test_refused_tag_on_submit_error(app, monkeypatch):
    """queue_prompt_to_comfyui must prefix explicit connection refusals with
    COMFYUI_REFUSED so the worker defers instead of stalling."""
    import app.utils.comfyui as cu

    class FakeRequests:
        exceptions = type('X', (), {'RequestException': OSError})

        def post(self, *a, **k):
            # ConnectionRefusedError: the exact class _is_explicit_connection_refused
            # matches (a bare OSError has no errno and would NOT be tagged).
            raise ConnectionRefusedError('Connection refused')

    monkeypatch.setattr(cu, 'requests', FakeRequests())
    monkeypatch.setattr(cu, '_ensure_comfyui_before_generation', lambda: None)
    monkeypatch.setattr(cu, 'fetch_object_info_model_files', lambda *a: None)
    monkeypatch.setattr(cu, 'comfy_names', type('N', (), {
        'canonical_model_widgets': lambda w, l: (w, False)}))
    monkeypatch.setattr(cu, 'unsupported_enum_values', lambda w: [])
    monkeypatch.setattr(cu, 'unavailable_model_files', lambda w: [])
    monkeypatch.setattr(cu, 'ensure_ollama_running', lambda: True)
    monkeypatch.setattr(cu, 'urljoin', lambda base, p: base + p)

    result, error = cu.queue_prompt_to_comfyui({'1': {}}, 'client-x')
    assert result is None
    assert error.startswith('COMFYUI_REFUSED')


def test_poll_fails_job_when_queue_proves_prompt_absent_after_grace(app, monkeypatch):
    """ComfyUI restarted: /history empty long enough AND /queue healthily proves
    the prompt gone => the job fails deterministically and the queue moves on."""
    import time
    from unittest.mock import patch
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    from app.utils.comfyui import ComfyHistoryHealth, ComfyHistoryProbe

    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})

        monkeypatch.setattr('app.job_queue.RESTART_ABSENT_GRACE_SECONDS', 2)
        monkeypatch.setattr('app.job_queue.POLL_INTERVAL_SECONDS', 0.1)

        t0 = time.monotonic()
        clock = [t0]

        def fake_clock():
            # _poll_outputs reads the clock every loop; each read must advance so
            # the absent-grace window eventually elapses.
            clock[0] += 3
            return clock[0]

        monkeypatch.setattr('app.job_queue.time.monotonic', fake_clock)

        def absent_probe(prompt_id):
            # healthy /queue that proves the id is gone (restart)
            return True

        def history_probe(prompt_id):
            return ComfyHistoryProbe(ComfyHistoryHealth.READY, history={})

        # _poll_outputs imports these INSIDE the function, so the patch target is
        # the module they are imported from, not job_queue's namespace.
        monkeypatch.setattr('app.utils.comfyui.get_comfyui_history_probe', history_probe)
        monkeypatch.setattr('app.utils.comfyui.comfyui_prompt_is_absent', absent_probe)

        # End to end: the real worker claims the pending job, submits (patched),
        # polls (real loop, patched probes), proves the restart and fails it.
        with patch('app.job_queue._submit', return_value='p-restart'):
            assert queue_manager.process_one() is True

        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        assert row.status == 'failed'
        assert 'restarted' in (row.error_message or '')
