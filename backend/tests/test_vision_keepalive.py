"""Vision model keep-warm: the lease is contention-driven AND revocable.

The two halves have to hold together. Keeping 7.5 GB warm is only defensible if
it is handed back the moment something else wants the card — a lease that can't
be revoked is worse than the always-unload behaviour it replaces, because it
swaps a predictable 12.8 s reload for an unpredictable silent eviction (WDDM
pages instead of raising OOM, so nothing would even report it).
"""
from unittest.mock import patch

import pytest

from app.services import vision_keepalive as vk


@pytest.fixture(autouse=True)
def _clean_lease():
    vk.forget_lease()
    yield
    vk.forget_lease()


# -- the knob ---------------------------------------------------------------

@pytest.mark.parametrize('raw,expected', [
    (None, vk.DEFAULT_WARM_SECONDS),
    ('', vk.DEFAULT_WARM_SECONDS),
    ('   ', vk.DEFAULT_WARM_SECONDS),
    ('not a number', vk.DEFAULT_WARM_SECONDS),
    (-5, 0),
    (0, 0),
    ('60', 60),
    (10 ** 6, vk.MAX_WARM_SECONDS),
])
def test_warm_seconds_is_total(raw, expected):
    """Whatever the settings form stored, we get a usable int."""
    assert vk.warm_seconds(override=raw) == expected


def test_zero_disables_the_feature_entirely(app):
    with app.app_context(), patch.object(vk, 'warm_seconds', return_value=0), \
            patch.object(vk, 'gpu_is_contended') as contended:
        assert vk.keep_alive_for_isolated_call() == vk.UNLOAD
        # Not even worth asking who wants the GPU.
        contended.assert_not_called()
    assert vk.lease_is_live() is False


# -- the decision -----------------------------------------------------------

def test_idle_card_leases_the_model_warm(app):
    with app.app_context():
        assert vk.keep_alive_for_isolated_call() == f'{vk.DEFAULT_WARM_SECONDS}s'
    assert vk.lease_is_live() is True


def test_a_training_run_forces_an_immediate_unload(app):
    from app.job_queue import queue_manager
    with app.app_context():
        queue_manager._set_system_state('training_in_progress', True)
        try:
            assert vk.gpu_is_contended() is True
            assert vk.keep_alive_for_isolated_call() == vk.UNLOAD
        finally:
            queue_manager._set_system_state('training_in_progress', None)
    assert vk.lease_is_live() is False


@pytest.mark.parametrize('status', ['pending', 'processing', 'sent_to_comfy'])
def test_an_unfinished_comfyui_job_forces_an_immediate_unload(app, status):
    from app.extensions import db
    from app.models import ImageGenerationQueue
    with app.app_context():
        db.session.add(ImageGenerationQueue(
            job_id=f'ka-{status}', user_id='local',
            status=status, workflow_data='{}'))
        db.session.commit()
        assert vk.gpu_is_contended() is True
        assert vk.keep_alive_for_isolated_call() == vk.UNLOAD


def test_a_finished_job_does_not_count_as_contention(app):
    from app.extensions import db
    from app.models import ImageGenerationQueue
    with app.app_context():
        db.session.add(ImageGenerationQueue(
            job_id='ka-done', user_id='local',
            status='completed', workflow_data='{}'))
        db.session.commit()
        assert vk.gpu_is_contended() is False


def test_an_unreadable_signal_degrades_to_the_old_behaviour():
    """No app context at all: the DB read raises, and we must NOT gamble 7.5 GB
    on an optimistic guess. Unloading is exactly what the code did before."""
    assert vk.gpu_is_contended() is True
    assert vk.keep_alive_for_isolated_call() == vk.UNLOAD
    assert vk.lease_is_live() is False


# -- the revocation ---------------------------------------------------------

def test_revoke_unloads_a_live_lease(app):
    with app.app_context():
        vk.keep_alive_for_isolated_call()
    assert vk.lease_is_live() is True
    with patch('app.services.vision_ollama.unload_vision_model',
               return_value=True) as unload:
        assert vk.revoke('test') is True
    unload.assert_called_once()
    assert vk.lease_is_live() is False


def test_revoke_is_free_when_nothing_is_leased():
    """This runs on the job queue's hot path — it must not fire an HTTP call per
    job just to discover there is nothing to unload."""
    with patch('app.services.vision_ollama.unload_vision_model') as unload:
        assert vk.revoke() is False
    unload.assert_not_called()


def test_unloading_clears_the_lease(app):
    """A batch pass ends with its own unload_vision_model(); after that, a later
    revoke() must not fire a second, pointless unload."""
    from app.services import vision_ollama
    with app.app_context():
        vk.keep_alive_for_isolated_call()
    with patch('app.services.vision_ollama.requests.post'):
        vision_ollama.unload_vision_model()
    assert vk.lease_is_live() is False


# -- the wiring: the paths that take the GPU actually revoke -----------------

def test_the_comfyui_queue_proves_ollama_released_before_submitting(app):
    """The load-bearing handoff runs before the workflow reaches ComfyUI.

    `ensure_released_for_comfy` subsumes an optional warm-lease revoke and the
    mandatory `/api/ps` ownership check. It must finish before `/prompt`.
    """
    from app.job_queue import queue_manager
    with app.app_context():
        queue_manager.add_job(workflow_data={'1': {}}, prompt='p')
    order = []
    with patch('app.services.vision_keepalive.ensure_released_for_comfy',
               side_effect=lambda *a, **k: (order.append('fence'), True)[1]), \
         patch('app.job_queue._submit',
               side_effect=lambda *a, **k: (order.append('submit'), 'pid-1')[1]), \
         patch('app.job_queue._poll_outputs', return_value=('out.png', False)), \
         patch('app.job_queue._dispatch_completion'):
        with app.app_context():
            assert queue_manager.process_one() is True
    assert order == ['fence', 'submit']


def test_head_crop_asks_the_policy_instead_of_hardcoding_zero(app):
    """`detect_head_bbox` is the burst case: five reference crops in a row used
    to pay the 12.8 s cold load five times."""
    from app.services import face_dataset_service as fds
    seen = {}

    def fake_describe(image_bytes, prompt, **kw):
        seen['keep_alive'] = kw.get('keep_alive')
        return ''

    with app.app_context(), \
            patch('app.services.vision_ollama.describe_image_ollama', fake_describe):
        fds.detect_head_bbox(b'not-an-image')
    assert seen['keep_alive'] == f'{vk.DEFAULT_WARM_SECONDS}s'
