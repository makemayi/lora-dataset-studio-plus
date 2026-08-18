"""Topaz job queue — scheduling, GPU arbitration, state machine."""
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_real_topaz_subprocess(monkeypatch):
    """The worker enters the REAL GPU window (conftest already stubs
    free_comfyui_vram) — stubbing the window itself turned out to leak state
    into later tests. Only the output collector is faked so tests never touch
    the real filesystem."""
    monkeypatch.setattr('app.services.topaz_job_queue.collect_output',
                        lambda tmp_dir, dataset_id, job_id: f'DS/{job_id}.png')


def _enqueue(app, **over):
    from app.services.topaz_job_queue import topaz_queue
    kw = dict(user_id='local', dataset_id=1, image_id=2, input_filename='C:/nope/a.png')
    kw.update(over)
    with app.app_context():
        return topaz_queue.enqueue(**kw)


def test_enqueue_creates_queued_row(app):
    from app.models import TopazJob
    jid = _enqueue(app)
    with app.app_context():
        row = TopazJob.query.filter_by(job_id=jid).one()
    assert row.status == 'queued'
    assert jid.startswith('topaz-')
    assert row.retry_count == 0


def test_worker_waits_while_training_runs(app, monkeypatch):
    from app.services.topaz_job_queue import topaz_queue
    from app.job_queue import queue_manager
    from app.models import TopazJob

    jid = _enqueue(app)
    with app.app_context():
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=60)
        with patch('app.services.topaz_helper.run_tpai',
                   return_value=('ok', '')) as fake:
            assert topaz_queue.process_one() is False     # gated, not run
            fake.assert_not_called()
        assert TopazJob.query.filter_by(job_id=jid).one().status == 'queued'
        queue_manager._set_system_state('training_in_progress', None)


def test_worker_runs_and_completes(app, monkeypatch):
    from app.services.topaz_job_queue import topaz_queue
    from app.models import TopazJob

    linked = {}
    monkeypatch.setattr(topaz_queue, 'link_completed',
                        lambda row: linked.update(status=row.status,
                                                  out=row.output_filename))

    jid = _enqueue(app)
    with app.app_context():
        with patch('app.services.topaz_helper.run_tpai',
                   return_value=('ok', '')):
            assert topaz_queue.process_one() is True
        row = TopazJob.query.filter_by(job_id=jid).one()
        assert row.status == 'completed'
    assert linked['status'] == 'completed'
    assert linked['out'].endswith('.png')


def test_license_failure_marks_failed_with_friendly_message(app, monkeypatch):
    from app.services.topaz_job_queue import topaz_queue
    from app.models import TopazJob

    jid = _enqueue(app)
    with app.app_context():
        with patch('app.services.topaz_helper.run_tpai',
                   return_value=('license', 'open Topaz once')):
            topaz_queue.process_one()
        row = TopazJob.query.filter_by(job_id=jid).one()
        assert row.status == 'failed'
        assert 'open Topaz' in (row.error_message or '')


def test_cancel_queued(app):
    from app.services.topaz_job_queue import topaz_queue
    from app.models import TopazJob

    jid = _enqueue(app)
    with app.app_context():
        assert topaz_queue.cancel(jid) is True
        assert TopazJob.query.filter_by(job_id=jid).one().status == 'cancelled'


def test_retry_requeues_failed(app):
    from app.services.topaz_job_queue import topaz_queue
    from app.models import TopazJob
    from app.extensions import db

    jid = _enqueue(app)
    with app.app_context():
        row = TopazJob.query.filter_by(job_id=jid).one()
        row.status = 'failed'
        row.error_message = 'boom'
        db.session.commit()
        assert topaz_queue.retry(jid) is True
        row = TopazJob.query.filter_by(job_id=jid).one()
        assert row.status == 'queued' and row.retry_count == 1
        assert row.error_message is None
