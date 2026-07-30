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
