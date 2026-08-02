"""Automatic + app-wide recovery from a stalled ComfyUI prompt.

The lived bug: a generation job died with the machine, leaving a durable
recovery barrier. That barrier blocks EVERY local generation, but its only
resolution button lived in the workspace of the dataset that owned the job — so
a user working on any other dataset met "a previous ComfyUI job has an
unresolved remote state" forever, even after doing exactly what the message
asked (restarting ComfyUI).

Two answers, and each refusal below is the reason the first one is safe:
  * provable cases resolve themselves (ComfyUI answers, and knows nothing of the
    prompt id -> the remote job is gone);
  * everything else surfaces app-wide, with one confirmed action.
"""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_host_ollama_gpu_fence(monkeypatch):
    monkeypatch.setattr(
        'app.services.vision_keepalive.ensure_released_for_comfy', lambda: True)


@pytest.fixture(autouse=True)
def _no_leaked_auto_recovery_notice():
    from app.job_queue import clear_auto_recovery_notice
    clear_auto_recovery_notice()
    yield
    clear_auto_recovery_notice()


def _stalled_prompt_barrier(app, *, prompt_id='p-stalled', metadata=None):
    """A known-prompt barrier, exactly as boot recovery would leave one."""
    from app.extensions import db
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    job_id = queue_manager.add_job(workflow_data={'1': {}}, metadata=metadata or {})
    row = ImageGenerationQueue.query.filter_by(job_id=job_id).one()
    row.update_status('sent_to_comfy', comfyui_prompt_id=prompt_id)
    db.session.commit()
    assert queue_manager._stall_comfy_job(
        job_id, prompt_id, allowed_statuses=('sent_to_comfy',))
    return job_id


def _stalled_unknown_submit_barrier(app, metadata=None):
    """A barrier with no prompt id at all: a `/prompt` whose outcome was lost."""
    from app.extensions import db
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    job_id = queue_manager.add_job(workflow_data={'1': {}}, metadata=metadata or {})
    row = ImageGenerationQueue.query.filter_by(job_id=job_id).one()
    row.update_status('processing')
    db.session.commit()
    assert queue_manager._stall_unknown_comfy_job(
        job_id, allowed_statuses=('processing',))
    return job_id


@contextmanager
def _comfyui_forgot_the_prompt():
    """ComfyUI answers, and has never heard of this prompt: queue empty, no
    history entry. That is what a restarted ComfyUI looks like."""
    from app.utils.comfyui import ComfyHistoryHealth, ComfyHistoryProbe, ComfyPromptState
    with ExitStack() as stack:
        for target, value in (
                ('comfyui_prompt_is_absent', True),
                ('cancel_comfyui_prompt_state', ComfyPromptState.ABSENT),
                ('get_comfyui_history_probe',
                 ComfyHistoryProbe(ComfyHistoryHealth.NOT_READY))):
            stack.enter_context(
                patch(f'app.utils.comfyui.{target}', return_value=value))
        yield


# --- the four cases ---------------------------------------------------------

def test_auto_resolves_when_comfyui_no_longer_knows_the_prompt(app):
    """Provable: a restarted ComfyUI knows neither the queue entry nor the id."""
    from app.job_queue import (auto_resolve_comfyui_barrier, peek_auto_recovery_notice,
                               queue_manager)
    from app.models import ImageGenerationQueue
    with app.app_context():
        job_id = _stalled_prompt_barrier(app)
        with _comfyui_forgot_the_prompt():
            resolved = auto_resolve_comfyui_barrier()
        assert resolved and resolved['job_id'] == job_id
        assert queue_manager.get_comfyui_stalled_barrier() is None
        assert not queue_manager.has_comfyui_stalled_barrier()
        assert ImageGenerationQueue.query.filter_by(job_id=job_id).one().status == 'cancelled'
        notice = peek_auto_recovery_notice()
        assert notice and 'cleared automatically' in notice['message']


def test_never_auto_resolves_while_comfyui_is_unreachable(app):
    """No answer is not an answer: an unreachable ComfyUI proves nothing."""
    from app.job_queue import (auto_resolve_comfyui_barrier, peek_auto_recovery_notice,
                               queue_manager)
    with app.app_context():
        job_id = _stalled_prompt_barrier(app)
        with patch('app.utils.comfyui.comfyui_prompt_is_absent',
                   return_value=None) as absent, \
             patch('app.utils.comfyui.cancel_comfyui_prompt_state') as cancel:
            assert auto_resolve_comfyui_barrier() is None
        assert absent.called and not cancel.called
        barrier = queue_manager.get_comfyui_stalled_barrier()
        assert barrier and barrier['job_id'] == job_id
        assert peek_auto_recovery_notice() is None


def test_never_auto_resolves_a_prompt_comfyui_is_still_running(app):
    """The job is ALIVE. Clearing it would strand real GPU work mid-flight."""
    from app.job_queue import auto_resolve_comfyui_barrier, queue_manager
    from app.models import ImageGenerationQueue
    with app.app_context():
        job_id = _stalled_prompt_barrier(app)
        with patch('app.utils.comfyui.comfyui_prompt_is_absent', return_value=False), \
             patch('app.utils.comfyui.cancel_comfyui_prompt_state') as cancel:
            assert auto_resolve_comfyui_barrier() is None
        # Not even a delete attempt: a live prompt is never touched automatically.
        assert not cancel.called
        assert queue_manager.get_comfyui_stalled_barrier()['job_id'] == job_id
        assert ImageGenerationQueue.query.filter_by(job_id=job_id).one().status == 'stalled'


def test_never_auto_resolves_an_unknown_submit_barrier(app):
    """No prompt id to ask about: a timed-out POST can still land later, so this
    one stays a human decision even when ComfyUI looks completely empty."""
    from app.job_queue import auto_resolve_comfyui_barrier, queue_manager
    with app.app_context():
        job_id = _stalled_unknown_submit_barrier(app)
        with patch('app.utils.comfyui.comfyui_prompt_is_absent',
                   return_value=True) as absent:
            assert auto_resolve_comfyui_barrier() is None
        assert not absent.called
        barrier = queue_manager.get_comfyui_stalled_barrier()
        assert barrier and barrier['job_id'] == job_id and barrier['kind'] == 'unknown_submit'


def test_corrupt_barrier_is_never_auto_resolved(app):
    """An unreadable record still blocks — and no code may guess its way out."""
    from app.extensions import db
    from app.job_queue import COMFYUI_STALLED_BARRIER_KEY, auto_resolve_comfyui_barrier
    from app.job_queue import queue_manager
    from app.models import SystemState
    with app.app_context():
        db.session.add(SystemState(key=COMFYUI_STALLED_BARRIER_KEY, value='{invalid-json'))
        db.session.commit()
        assert auto_resolve_comfyui_barrier() is None
        assert queue_manager.has_comfyui_stalled_barrier()


# --- the guard heals the very request it would have refused ------------------

def test_route_guard_clears_a_provable_barrier_and_lets_the_request_through(app):
    from app.routes._common import _require_no_stalled_comfyui
    with app.app_context():
        _stalled_prompt_barrier(app)
        with _comfyui_forgot_the_prompt():
            assert _require_no_stalled_comfyui() is None


def test_route_guard_still_refuses_what_it_cannot_prove(app):
    from app.routes._common import _require_no_stalled_comfyui
    with app.app_context():
        _stalled_unknown_submit_barrier(app)
        gate = _require_no_stalled_comfyui()
        assert gate is not None
        body, status = gate
        assert status == 409 and body.get_json()['code'] == 'comfyui_recovery_required'


# --- app-wide surface: the exact case that trapped a real user ---------------

def _dataset(app, name='Alpha'):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as fds
    return fds.create_dataset(LOCAL_USER, name, 'trg')


def test_recovery_state_is_visible_from_anywhere_not_just_the_owning_dataset(app, client):
    """Dataset A owns the stalled job; the user is on dataset B. The state
    endpoint is global, so B sees WHAT is stuck and WHERE it lives."""
    with app.app_context():
        ds_a = _dataset(app, 'Owner dataset')
        _stalled_prompt_barrier(app, metadata={'dataset_id': ds_a.id,
                                               'variation_label': 'portrait'})
        owner_id, owner_name = ds_a.id, ds_a.name
    with patch('app.utils.comfyui.comfyui_prompt_is_absent', return_value=None):
        payload = client.get('/api/system/comfyui-recovery').get_json()
    recovery = payload['recovery']
    assert recovery['kind'] == 'prompt'
    assert recovery['dataset_id'] == owner_id and recovery['dataset_name'] == owner_name
    assert recovery['variation_label'] == 'portrait'
    assert recovery['stalled_since'] and recovery['can_confirm_restart'] is True


def test_recovery_state_reports_nothing_when_clear(app, client):
    payload = client.get('/api/system/comfyui-recovery').get_json()
    assert payload['ok'] is True and payload['recovery'] is None


def test_recovery_state_poll_auto_clears_and_announces_it(app, client):
    from app.job_queue import queue_manager
    with app.app_context():
        _stalled_prompt_barrier(app)
    with _comfyui_forgot_the_prompt():
        payload = client.get('/api/system/comfyui-recovery').get_json()
    assert payload['recovery'] is None
    assert 'cleared automatically' in payload['auto_cleared']['message']
    with app.app_context():
        assert not queue_manager.has_comfyui_stalled_barrier()


def test_global_resolve_button_clears_an_unknown_submit_from_another_dataset(app, client):
    """The banner's one-click action, pressed while standing on dataset B.

    An unknown-submit card is exactly the case the human confirmation exists
    for, and it used to be reachable only through the owning dataset's Stop
    button. Here the global endpoint routes it to the same service, naming the
    owning dataset it read off the barrier.
    """
    with app.app_context():
        owner = _dataset(app, 'Owner dataset')
        owner_id = owner.id
        _stalled_unknown_submit_barrier(app, metadata={'dataset_id': owner_id})
    seen = {}

    def _confirm(user_id, dataset_id, *, restart_confirmed=False):
        seen.update(dataset_id=dataset_id, restart_confirmed=restart_confirmed)
        return 1

    with patch('app.routes._common.capabilities.probe',
               return_value={'comfyui': {'reachable': True}}), \
         patch('app.services.face_dataset_service.confirm_unknown_generation_restart',
               side_effect=_confirm):
        res = client.post('/api/system/comfyui-recovery/resolve',
                          json={'confirmed_comfyui_restart': True})
    assert res.status_code == 200 and res.get_json()['cleared'] == 1
    assert seen == {'dataset_id': owner_id, 'restart_confirmed': True}


def test_global_resolve_refuses_without_the_explicit_confirmation(app, client):
    with app.app_context():
        _stalled_unknown_submit_barrier(app)
    res = client.post('/api/system/comfyui-recovery/resolve', json={})
    assert res.status_code == 400


def test_global_resolve_refuses_while_comfyui_is_unreachable(app, client):
    """"I restarted it" is only meaningful if the replacement answers now."""
    from app.job_queue import queue_manager
    with app.app_context():
        _stalled_unknown_submit_barrier(app)
    with patch('app.routes._common.capabilities.probe',
               return_value={'comfyui': {'reachable': False, 'hint': 'Check the URL'}}):
        res = client.post('/api/system/comfyui-recovery/resolve',
                          json={'confirmed_comfyui_restart': True})
    assert res.status_code == 409
    with app.app_context():
        assert queue_manager.has_comfyui_stalled_barrier()


def test_global_resolve_refuses_a_prompt_comfyui_still_reports(app, client):
    """Confirmation cannot override evidence: the prompt is still queued."""
    from app.job_queue import queue_manager
    with app.app_context():
        _stalled_prompt_barrier(app)
    with patch('app.routes._common.capabilities.probe',
               return_value={'comfyui': {'reachable': True}}), \
         patch('app.utils.comfyui.comfyui_prompt_is_absent', return_value=False):
        res = client.post('/api/system/comfyui-recovery/resolve',
                          json={'confirmed_comfyui_restart': True})
    assert res.status_code == 409
    assert 'still reports' in res.get_json()['error']
    with app.app_context():
        assert queue_manager.has_comfyui_stalled_barrier()
