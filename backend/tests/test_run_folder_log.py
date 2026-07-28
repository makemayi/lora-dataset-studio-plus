"""A run that dies in seconds must leave its log where the button opens.

Reported by wannadecryptor (Discord #help): training died after ~3 s and there
was NO log anywhere in the run folder. The panel explicitly tells you to open
`training.log` via "📂 Run folder".

Ground truth in `lora_training`:
  - the launch writes the log to   <output>/<run_name>/training.log
  - "📂 Run folder" opened          <output>/<run_name>/lora_<trigger>/

`lora_<trigger>` is ai-toolkit's save_root — created by ai-toolkit itself when
it reaches the first save. A run that dies at boot never creates it, so the
button's `os.makedirs` MADE an empty folder and opened that: an empty window,
no log, on exactly the failure the message was written for.
"""
import os
import sys
import time

import pytest


def _stub_aitoolkit(tmp_path, app, run_py_body):
    """ai-toolkit install whose run.py is a REAL python script we control, run
    by the REAL interpreter — so the launch spawns a genuine process that dies
    on its own, stderr and all."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    root.mkdir(parents=True)
    (root / 'run.py').write_text(run_py_body, encoding='utf-8')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root), 'python': sys.executable}})
    return root


# run.py of a boot-time crash: writes a traceback to stderr, exits non-zero.
_DYING_RUN_PY = (
    "import sys\n"
    "print('a FutureWarning nobody cares about', file=sys.stderr)\n"
    "print('RuntimeError: no CUDA GPUs are available', file=sys.stderr)\n"
    "sys.exit(1)\n"
)


@pytest.fixture()
def dying_run(app, tmp_path, monkeypatch):
    """Launch a real training that dies immediately; yield (dataset, result)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    _stub_aitoolkit(tmp_path, app, _DYING_RUN_PY)
    monkeypatch.setattr(lt, '_watch_training', lambda *a, **k: None)
    # These tests are about the run folder and the log the launch writes, and they
    # need a REAL interpreter to spawn a real dying process. The interpreter guard
    # would refuse that interpreter for the one thing this fixture does not care
    # about — whether it can import torch — which is true of any CI runner and of
    # most machines. Stub the guard, not the launch: its own coverage lives in
    # test_capabilities.py and test_training_diagnostics.py, where the probe is
    # what is under test rather than a precondition of it.
    monkeypatch.setattr(lt, 'assert_interpreter_ready', lambda *a, **k: None)

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Dying', 'dyingtrig')
        for i in range(12):
            svc.db.session.add(lt.FaceDatasetImage(
                dataset_id=ds.id, status='keep', filename=f'x{i}.webp',
                caption='a caption here'))
        svc.db.session.commit()

        def fake_export(user_id, dataset_id, masked=True):
            folder = tmp_path / 'exported'
            folder.mkdir(exist_ok=True)
            return str(folder)

        monkeypatch.setattr(lt, 'export_dataset_to_aitoolkit', fake_export)
        result = lt.launch_training(LOCAL_USER, ds.id, masked=False)
        # Wait for the real process to die (well under a second in practice).
        for _ in range(50):
            if os.path.getsize(result['log_path']):
                break
            time.sleep(0.1)
        yield ds, result


def test_run_folder_button_opens_the_folder_holding_the_log(dying_run, app, monkeypatch):
    """wannadecryptor's scenario: a run dead in seconds, zero checkpoint, and the
    UI checkpoint selector carrying nothing. The folder "📂 Run folder" opens
    must CONTAIN training.log."""
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    ds, result = dying_run

    opened = []
    monkeypatch.setattr(lt.os, 'startfile', opened.append, raising=False)
    monkeypatch.setattr(lt.subprocess, 'Popen', lambda args, **kw: opened.append(args[1]))

    with app.app_context():
        # No selection at all — no checkpoint was ever produced, so this is what
        # the panel sends: the persisted base/variant, nothing chosen.
        path = lt.open_training_folder(LOCAL_USER, ds.id, target='run')

    log_path = os.path.realpath(result['log_path'])
    assert os.path.isfile(log_path), 'the launch must have written a log'
    assert os.path.dirname(log_path) == os.path.realpath(path), (
        f'the button opens {path!r} but the log lives in '
        f'{os.path.dirname(log_path)!r}')
    assert 'training.log' in os.listdir(path)
    assert opened, 'the folder must actually be revealed'


def test_run_folder_still_leads_to_the_checkpoints(dying_run, app, monkeypatch):
    """Opening the run's top folder must not cost access to the checkpoints:
    ai-toolkit's save_root sits INSIDE it."""
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    ds, _ = dying_run

    monkeypatch.setattr(lt.os, 'startfile', lambda p: None, raising=False)
    with app.app_context():
        path = lt.open_training_folder(LOCAL_USER, ds.id, target='run')
        save_root = lt._run_dir(LOCAL_USER, ds.id)
    assert os.path.dirname(os.path.normpath(save_root)) == os.path.normpath(path)


def test_progress_reads_the_log_the_launch_wrote(dying_run, app):
    """Writer and reader agree on the log path — the crash text the panel shows
    comes from the same file the button now reveals."""
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    ds, result = dying_run
    with app.app_context():
        prog = lt.training_progress(LOCAL_USER, ds.id)
        assert prog['log_exists'] is True
        assert lt._run_log_path(ds) == result['log_path']
    with open(result['log_path'], encoding='utf-8') as fh:
        assert 'no CUDA GPUs are available' in fh.read()
