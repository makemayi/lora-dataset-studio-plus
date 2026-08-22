"""The Topaz worker thread must be BOOted — not merely enqueue-able.

Every other topaz test drives ``topaz_queue.process_one()`` directly (the
synchronous seam), so a worker that never starts is invisible to the suite:
the job row lands in SQLite, the tile flips to pending, nothing ever moves
(the 2026-08-19 "batch upscale does nothing" defect — _start_workers started
every other lane except topaz). This file asserts the one property the seam
cannot prove: that create_app() really puts a topaz-job-worker thread out
there, and that stop() joins it back in.
"""
import threading

import pytest

from app import create_app
from app.services.topaz_job_queue import topaz_queue


@pytest.fixture()
def booted_app(tmp_path, monkeypatch):
    """The same isolation as tests/conftest's ``app`` fixture, minus TESTING:
    ``create_app`` only runs ``_start_workers`` for non-TESTING apps, and that
    _start_workers is exactly what used to forget this lane."""
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as _cfg
    monkeypatch.setattr(_cfg, 'ENV_PATH', tmp_path / '.env')
    monkeypatch.setattr(_cfg, '_cache', None)
    import app.capabilities as _caps
    monkeypatch.setattr(_caps, '_cache', None)
    monkeypatch.setattr(_caps, '_cache_ts', 0.0)
    _caps._import_cache.clear()
    application = create_app(
        {'WTF_CSRF_ENABLED': False, 'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})
    yield application
    # Be a good citizen: the manager is a process-global singleton, so join
    # its daemon thread back instead of leaving the suite carrying it.
    topaz_queue.stop()


def test_topaz_worker_thread_boots(booted_app):
    assert topaz_queue._app is booted_app, 'init_app was not called at boot'
    assert topaz_queue._running is True
    assert topaz_queue._thread is not None
    assert topaz_queue._thread.is_alive()
    assert 'topaz-job-worker' in [t.name for t in threading.enumerate()]


def test_topaz_worker_stops_cleanly(booted_app):
    topaz_queue.stop()
    assert topaz_queue._running is False
    assert topaz_queue._thread is None
    assert 'topaz-job-worker' not in [t.name for t in threading.enumerate()]
