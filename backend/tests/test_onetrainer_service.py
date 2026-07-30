from pathlib import Path

import pytest


@pytest.fixture
def onetrainer(monkeypatch, tmp_path):
    # Isolate config.json per-test the same way tests/conftest.py's own
    # `app` fixture does — without this, save_config() below writes to the
    # REAL repo config.json and state leaks across tests.
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    from app import config as cfg
    from app.services import onetrainer_service as ots
    monkeypatch.setattr(cfg, '_cache', None)
    yield ots, cfg


def test_onetrainer_defaults_are_blank(onetrainer):
    ots, cfg = onetrainer
    assert cfg.get('onetrainer.dir') == ''
    assert cfg.get('onetrainer.python') == ''


def test_onetrainer_path_dir_is_none_when_unconfigured(onetrainer):
    ots, cfg = onetrainer
    assert ots.onetrainer_path('dir') is None


def test_onetrainer_path_venv_python_prefers_explicit_override(onetrainer, tmp_path):
    ots, cfg = onetrainer
    fake_root = tmp_path / 'OneTrainer'
    fake_root.mkdir()
    explicit = tmp_path / 'my_python.exe'
    explicit.write_text('')
    cfg.save_config({'onetrainer': {'dir': str(fake_root), 'python': str(explicit)}})
    assert ots.onetrainer_path('venv_python') == explicit


def test_onetrainer_path_venv_python_derives_from_dir_when_no_override(onetrainer, tmp_path):
    ots, cfg = onetrainer
    fake_root = tmp_path / 'OneTrainer'
    (fake_root / 'venv' / 'Scripts').mkdir(parents=True)
    venv_py = fake_root / 'venv' / 'Scripts' / 'python.exe'
    venv_py.write_text('')
    cfg.save_config({'onetrainer': {'dir': str(fake_root), 'python': ''}})
    assert ots.onetrainer_path('venv_python') == venv_py


def test_is_installed_false_without_a_real_venv_python(onetrainer):
    ots, cfg = onetrainer
    assert ots.is_installed() is False


def test_trainer_column_exists_and_defaults_to_ai_toolkit(app):
    from app.models import TrainingRunRecord
    with app.app_context():
        cols = {c.name for c in TrainingRunRecord.__table__.columns}
        assert 'trainer' in cols


def test_build_job_config_overrides_only_what_this_app_owns(onetrainer, tmp_path):
    ots, cfg = onetrainer
    dataset_folder = str(tmp_path / 'dataset')
    training_folder = str(tmp_path / 'run')
    config = ots.build_job_config(
        trigger='lola', dataset_folder=dataset_folder,
        training_folder=training_folder, steps=2000, num_images=25, rank=32)
    # Ownership boundary: these come from OUR app's UI/dataset state.
    assert config['workspace_dir'] == training_folder
    assert config['cache_dir'] == str(Path(training_folder) / 'cache')
    assert config['output_model_destination'] == str(Path(training_folder) / 'lola.safetensors')
    assert config['lora_rank'] == 32
    assert config['epochs'] == 80          # ceil(2000 / 25)
    # Ownership boundary: everything else stays whatever OneTrainer's own
    # shipped preset says — this function must NOT invent values for them.
    assert 'model_type' not in config
    assert 'base_model_name' not in config
    assert 'training_method' not in config


def test_build_job_config_epochs_is_at_least_one(onetrainer, tmp_path):
    ots, cfg = onetrainer
    config = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=10, num_images=9999, rank=16)
    assert config['epochs'] == 1


def test_build_concepts_points_at_the_exported_folder(onetrainer, tmp_path):
    ots, cfg = onetrainer
    concepts = ots.build_concepts(trigger='lola', dataset_folder=str(tmp_path / 'ds'))
    assert concepts == [{
        'name': 'lola', 'path': str(tmp_path / 'ds'), 'enabled': True,
    }]


def test_shipped_preset_path_for_krea2(onetrainer):
    ots, cfg = onetrainer
    p = ots.KREA2_PRESET_RELATIVE_PATH
    assert p == 'training_presets/Krea 2/#krea2 LoRA 16GB.json'


def test_launch_writes_config_and_concepts_and_spawns_the_right_command(
        onetrainer, tmp_path, monkeypatch):
    ots, cfg = onetrainer
    root = tmp_path / 'OneTrainer'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    venv_py = root / 'venv' / 'Scripts' / 'python.exe'
    venv_py.write_text('')
    cfg.save_config({'onetrainer': {'dir': str(root)}})

    captured = {}
    class FakeProc:
        pid = 4242
    def fake_popen(cmd, cwd=None, env=None, **kw):
        captured['cmd'] = cmd
        captured['cwd'] = cwd
        return FakeProc()
    monkeypatch.setattr(ots.subprocess, 'Popen', fake_popen)

    training_folder = tmp_path / 'run'
    result = ots.launch(
        trigger='lola', dataset_folder=str(tmp_path / 'ds'),
        training_folder=str(training_folder), steps=1000, num_images=20, rank=16)

    assert result['pid'] == 4242
    assert (training_folder / 'concepts.json').is_file()
    assert (training_folder / 'config.json').is_file()
    assert captured['cwd'] == str(root)
    assert captured['cmd'][0] == str(venv_py)
    assert captured['cmd'][1] == 'scripts/train.py'
    # Hyphenated, matching train.py's REAL argparse flags (confirmed against
    # its own usage output — the underscored form fails argparse outright).
    assert '--preset-path' in captured['cmd']
    assert captured['cmd'][captured['cmd'].index('--preset-path') + 1] == \
        str(root / ots.KREA2_PRESET_RELATIVE_PATH)
    assert '--config-path' in captured['cmd']
    assert captured['cmd'][captured['cmd'].index('--config-path') + 1] == \
        str(training_folder / 'config.json')


def test_launch_raises_when_onetrainer_not_installed(onetrainer, tmp_path):
    ots, cfg = onetrainer
    with pytest.raises(RuntimeError, match='OneTrainer is not configured'):
        ots.launch(trigger='x', dataset_folder=str(tmp_path),
                   training_folder=str(tmp_path / 'run'), steps=100,
                   num_images=10, rank=16)


def test_checkpoint_ready_true_only_when_destination_file_exists(onetrainer, tmp_path):
    ots, cfg = onetrainer
    training_folder = tmp_path / 'run'
    training_folder.mkdir()
    dest = training_folder / 'lola.safetensors'
    assert ots.checkpoint_ready(str(dest)) is False
    dest.write_bytes(b'fake-weights')
    assert ots.checkpoint_ready(str(dest)) is True


def test_launch_training_registers_and_tracks_like_an_ai_toolkit_run(
        onetrainer, tmp_path, monkeypatch, app):
    from app.services import checkpoint_registry, face_dataset_service as svc
    from app.config import LOCAL_USER
    ots, cfg = onetrainer
    root = tmp_path / 'OneTrainer'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('')
    cfg.save_config({'onetrainer': {'dir': str(root)}})

    class FakeProc:
        pid = 555
        def poll(self):
            return None
    monkeypatch.setattr(ots.subprocess, 'Popen', lambda *a, **k: FakeProc())

    with app.app_context():
        import os as _os
        ds = svc.create_dataset(LOCAL_USER, 'Lola', 'lola')
        d = svc._dataset_dir(ds.id)
        _os.makedirs(d, exist_ok=True)
        with open(_os.path.join(d, 'ref.webp'), 'wb') as fh:
            fh.write(b'\x89PNG\r\n\x1a\n')
        ds.train_type = 'krea'
        ds.ref_filename = 'ref.webp'
        svc.db.session.commit()

        result = ots.launch_training(LOCAL_USER, ds.id, steps=100, check_captions=False)

        assert result['pid'] == 555
        from app.job_queue import queue_manager
        assert queue_manager._get_system_state('training_pid', None) == 555
        assert queue_manager._get_system_state('training_in_progress', False) is True
        rec = checkpoint_registry.latest_record(ds.id, 'krea')
        assert rec is not None
        assert rec.source == 'local'
        assert rec.trainer == 'onetrainer'
