"""Stop lane detection — a stale `training_trainer` must not hijack an ai-toolkit stop.

The bug (observed 2026-10-18): only OneTrainer ever stamped `training_trainer`, so
the key kept 'onetrainer' from the last OneTrainer run. The NEXT ai-toolkit run's
stop took the OneTrainer branch, returned "OneTrainer is saving this run…", and
the ai-toolkit process kept training on the GPU until it was taskkilled by hand.
"""
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, 'backend')

from app.services import lora_training as lt          # noqa: E402


@pytest.fixture()
def state(monkeypatch):
    """In-memory system_state so the stop lane can be driven without the app."""
    store = {}

    def _get(key, default=None):
        return store.get(key, default)

    def _set(key, value, ttl_seconds=None):
        store[key] = value

    monkeypatch.setattr(lt.queue_manager, '_get_system_state', _get)
    monkeypatch.setattr(lt.queue_manager, '_set_system_state', _set)
    return store


def _running(monkeypatch, cmdline, pid=4242):
    monkeypatch.setattr(lt, '_pid_alive', lambda p: True)
    monkeypatch.setattr(lt, '_training_process_cmdline', lambda p: cmdline)
    return pid


def test_record_identity_stamps_the_ai_toolkit_lane(state):
    """A launch through _record_training_process_identity says whose run it is."""
    lt._record_training_process_identity(4242)
    assert state['training_trainer'] == 'ai_toolkit'
    assert state['training_pid'] == 4242


def test_onetrainer_lane_still_asks(state, monkeypatch):
    """The real OneTrainer path must keep its graceful save-before-exit."""
    asked = []
    lt._record_training_process_identity(_running(monkeypatch, 'python E:\\OneTrainer\\jobs\\run.py'))
    state['training_trainer'] = 'onetrainer'
    state['training_in_progress'] = True
    monkeypatch.setattr(
        'app.services.onetrainer_service.request_graceful_stop',
        lambda pid: asked.append(pid) or True)
    monkeypatch.setattr(
        'app.services.onetrainer_service.wait_for_exit', lambda pid: True)
    monkeypatch.setattr(lt, '_save_queue', lambda q: None)
    assert lt.stop_training() is True
    assert asked == [4242]


def test_stale_onetrainer_marker_does_not_hijack_an_ai_toolkit_run(state, monkeypatch):
    """THE bug: marker says onetrainer, process is ai-toolkit → kill, don't ask."""
    state.update({'training_pid': 4242, 'training_in_progress': True,
                  'training_trainer': 'onetrainer', 'training_dataset_id': 51})
    _running(monkeypatch, 'F:\\ai-toolkit\\venv\\Scripts\\python.exe run.py config.yaml')
    killed, asked = [], []

    monkeypatch.setattr(lt, '_pid_alive', lambda p: True)
    monkeypatch.setattr(
        'app.services.onetrainer_service.request_graceful_stop',
        lambda pid: asked.append(pid) or True)
    monkeypatch.setattr(lt.subprocess, 'run',
                        lambda *a, **k: killed.append(a[0]) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(lt, '_wait_for_training_process_exit', lambda pid: True)
    monkeypatch.setattr(lt, '_save_queue', lambda q: None)

    assert lt.stop_training() is True
    assert asked == [], 'an ai-toolkit run must never be asked to save itself'
    assert killed and killed[0][:2] == ['taskkill', '/F']
    assert state.get('training_in_progress') is False     # fence released
    assert state.get('training_pid') is None


def test_unreadable_cmdline_keeps_the_historical_behaviour(state, monkeypatch):
    """Cannot tell? Keep asking — never kill a possible OneTrainer run."""
    state.update({'training_pid': 4242, 'training_in_progress': True,
                  'training_trainer': 'onetrainer'})
    _running(monkeypatch, '')
    asked = []
    monkeypatch.setattr(
        'app.services.onetrainer_service.request_graceful_stop',
        lambda pid: asked.append(pid) or True)
    monkeypatch.setattr(
        'app.services.onetrainer_service.wait_for_exit', lambda pid: True)
    monkeypatch.setattr(lt, '_save_queue', lambda q: None)
    assert lt.stop_training() is True
    assert asked == [4242]


def test_is_onetrainer_process_reads_the_command_line(monkeypatch):
    monkeypatch.setattr(lt, '_training_process_cmdline',
                        lambda p: 'python E:/OneTrainer/src/train.py' if p == 1 else '')
    assert lt._is_onetrainer_process(1) is True
    assert lt._is_onetrainer_process(2) is None
    monkeypatch.setattr(lt, '_training_process_cmdline',
                        lambda p: 'F:/ai-toolkit/venv/python.exe run.py')
    assert lt._is_onetrainer_process(3) is False
