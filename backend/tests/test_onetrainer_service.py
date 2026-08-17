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
    assert config['lora_alpha'] == 32.0     # scale factor 1.0 — MUST track rank
    assert config['batch_size'] == 1        # matches the epochs approximation
    # A STRING, matching OneTrainer's own schema — its shipped Krea 2 preset
    # writes "512", not 512. This assertion used to demand the int, which was
    # never checked against the preset.
    assert config['resolution'] == '1024'
    assert config['epochs'] == 80          # ceil(2000 / 25)
    assert config['peft_type'] == 'LORA'   # default when the caller doesn't pass one
    # Ownership boundary: everything else stays whatever OneTrainer's own
    # shipped preset says — this function must NOT invent values for them.
    assert 'model_type' not in config
    assert 'base_model_name' not in config
    assert 'training_method' not in config


def test_build_job_config_accepts_an_explicit_peft_type(onetrainer, tmp_path):
    ots, cfg = onetrainer
    config = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=32, peft_type='OFT_2')
    assert config['peft_type'] == 'OFT_2'


def test_build_job_config_unrecognised_peft_type_degrades_to_lora(onetrainer, tmp_path):
    ots, cfg = onetrainer
    config = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=32, peft_type='NOT_A_REAL_METHOD')
    assert config['peft_type'] == 'LORA'


def test_build_job_config_lora_alpha_tracks_a_different_rank_too(onetrainer, tmp_path):
    """MEASURED regression: a run left at the shipped preset's lora_alpha=1
    with rank=32 trained a LoRA scaled to ~1/32 of its intended strength —
    indistinguishable from non-convergence. alpha must equal whatever rank
    THIS call was given, not a fixed number."""
    ots, cfg = onetrainer
    config = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=8)
    assert config['lora_rank'] == 8
    assert config['lora_alpha'] == 8.0


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

    # Pin the VRAM: which shipped preset is used now depends on the card, and a
    # test that reads the TEST MACHINE's GPU passes or fails by accident.
    monkeypatch.setattr('app.services.run_environment.local_vram_gb', lambda: 16)

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
        from PIL import Image
        from app.models import FaceDatasetImage
        ds = svc.create_dataset(LOCAL_USER, 'Lola', 'lola')
        d = svc._dataset_dir(ds.id)
        _os.makedirs(d, exist_ok=True)
        Image.new('RGB', (64, 64), (200, 100, 50)).save(_os.path.join(d, 'a.png'))
        ds.train_type = 'krea'
        row = FaceDatasetImage(dataset_id=ds.id, filename='a.png', status='keep',
                               caption='a photo')
        svc.db.session.add(row)
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

        import json as _json
        written = _json.loads(open(result['config_path'], encoding='utf-8').read())
        assert written['peft_type'] == 'LORA', 'default setting, no override saved'


def test_launch_training_forwards_the_configured_peft_type(
        onetrainer, tmp_path, monkeypatch, app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    ots, cfg = onetrainer
    root = tmp_path / 'OneTrainer'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('')
    cfg.save_config({'onetrainer': {'dir': str(root), 'peft_type': 'OFT_2'}})

    class FakeProc:
        pid = 556
        def poll(self):
            return None
    monkeypatch.setattr(ots.subprocess, 'Popen', lambda *a, **k: FakeProc())

    with app.app_context():
        import os as _os
        import json as _json
        from PIL import Image
        from app.models import FaceDatasetImage
        ds = svc.create_dataset(LOCAL_USER, 'Mimi', 'mimi')
        d = svc._dataset_dir(ds.id)
        _os.makedirs(d, exist_ok=True)
        Image.new('RGB', (64, 64), (200, 100, 50)).save(_os.path.join(d, 'a.png'))
        ds.train_type = 'krea'
        row = FaceDatasetImage(dataset_id=ds.id, filename='a.png', status='keep',
                               caption='a photo')
        svc.db.session.add(row)
        svc.db.session.commit()

        result = ots.launch_training(LOCAL_USER, ds.id, steps=100, check_captions=False)
        written = _json.loads(open(result['config_path'], encoding='utf-8').read())
        assert written['peft_type'] == 'OFT_2'


# --- The GPU guards this lane never had (2026-08-09) ---------------------------
# OneTrainer's launch checked only "is another training running". It would start
# on a card ComfyUI was rendering on, or one an idle ComfyUI was still holding —
# and the second case killed an ai-toolkit run the same day: an idle ComfyUI sat
# on 4.4 GB, the run got ~18 GB of a 24 GB card, and the step time went
# 8.4 s -> 78 s -> 104 s before the process died at step 3 with no error.
# OneTrainer trains the same 12B Krea model on the same card.

def _trainable_krea_dataset(svc, LOCAL_USER, name):
    import os as _os
    from PIL import Image
    from app.models import FaceDatasetImage
    ds = svc.create_dataset(LOCAL_USER, name, name.lower().replace(' ', '_'))
    d = svc._dataset_dir(ds.id)
    _os.makedirs(d, exist_ok=True)
    Image.new('RGB', (64, 64), (200, 100, 50)).save(_os.path.join(d, 'a.png'))
    ds.train_type = 'krea'
    svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, filename='a.png',
                                        status='keep', caption='a photo'))
    svc.db.session.commit()
    return ds


def _installed_onetrainer(cfg, tmp_path):
    root = tmp_path / 'OneTrainer'
    (root / 'venv' / 'Scripts').mkdir(parents=True, exist_ok=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('')
    cfg.save_config({'onetrainer': {'dir': str(root)}})


@pytest.mark.parametrize('verdict_name,should_pass', [
    ('FREED', True),
    ('COMFYUI_OFFLINE', True),
    ('UNKNOWN', False),
])
def test_onetrainer_refuses_a_card_comfyui_has_not_released(
        onetrainer, tmp_path, monkeypatch, app, verdict_name, should_pass):
    from app.config import LOCAL_USER
    from app.gpu_window import GpuBusyError
    from app.services import face_dataset_service as svc
    from app.utils.comfyui import ComfyVramFreeVerdict
    ots, cfg = onetrainer
    _installed_onetrainer(cfg, tmp_path)

    called = []
    monkeypatch.setattr(
        'app.utils.comfyui.free_comfyui_vram',
        lambda: (called.append(True), getattr(ComfyVramFreeVerdict, verdict_name))[1])

    class FakeProc:
        pid = 777
        def poll(self):
            return None
    monkeypatch.setattr(ots.subprocess, 'Popen', lambda *a, **k: FakeProc())

    with app.app_context():
        ds = _trainable_krea_dataset(svc, LOCAL_USER, f'OT {verdict_name}')
        if should_pass:
            result = ots.launch_training(LOCAL_USER, ds.id, steps=100,
                                         check_captions=False)
            assert result['pid'] == 777
        else:
            with pytest.raises(GpuBusyError) as e:
                ots.launch_training(LOCAL_USER, ds.id, steps=100,
                                    check_captions=False)
            assert 'did not confirm' in str(e.value)
    assert called, 'the guard must actually ask ComfyUI to release the card'


def test_onetrainer_refuses_while_comfyui_has_queued_work(
        onetrainer, tmp_path, monkeypatch, app):
    """The check the ai-toolkit lane always had and this one never did."""
    from app.config import LOCAL_USER
    from app.gpu_window import GpuBusyError
    from app.job_queue import queue_manager
    from app.services import face_dataset_service as svc
    ots, cfg = onetrainer
    _installed_onetrainer(cfg, tmp_path)
    monkeypatch.setattr(queue_manager, 'has_comfyui_work', lambda: True)
    spawned = []
    monkeypatch.setattr(ots.subprocess, 'Popen',
                        lambda *a, **k: spawned.append(True))

    with app.app_context():
        ds = _trainable_krea_dataset(svc, LOCAL_USER, 'OT busy')
        with pytest.raises(GpuBusyError) as e:
            ots.launch_training(LOCAL_USER, ds.id, steps=100, check_captions=False)
    assert 'queued or active work' in str(e.value)
    assert spawned == [], 'nothing may spawn while ComfyUI owns the card'


# --- Config parity with the ai-toolkit lane (2026-08-09) -----------------------
# Reported as "the in-app OneTrainer still needs polish — mainly the config".
# Diffed against what the app generates for ai-toolkit: this lane dropped the
# learning rate and the resolution on the floor and pinned the 16 GB preset.

def test_the_24gb_preset_is_chosen_when_the_card_can_take_it(onetrainer):
    """The two shipped presets differ in EXACTLY ONE field: the 16 GB one adds
    transformer.offload_fraction 0.3, i.e. 30% of the transformer swapped from
    system RAM every step. Pinning it on a 24 GB card is a pure slowdown."""
    ots, _cfg = onetrainer
    assert ots.krea2_preset_relative_path(24) == ots.KREA2_PRESET_24GB_RELATIVE_PATH
    assert ots.krea2_preset_relative_path(23.99) == ots.KREA2_PRESET_24GB_RELATIVE_PATH
    assert ots.krea2_preset_relative_path(20) == ots.KREA2_PRESET_24GB_RELATIVE_PATH


@pytest.mark.parametrize('vram', [16, 12, 19.9, None, 'unknown'])
def test_an_unknown_or_small_card_keeps_the_conservative_preset(onetrainer, vram, monkeypatch):
    """Guessing upward turns a slow run into an OOM, so anything that is not a
    confident 20 GB+ keeps the offloading preset."""
    ots, _cfg = onetrainer
    if vram is None:
        monkeypatch.setattr('app.services.run_environment.local_vram_gb', lambda: None)
        assert ots.krea2_preset_relative_path() == ots.KREA2_PRESET_RELATIVE_PATH
    else:
        assert ots.krea2_preset_relative_path(vram) == ots.KREA2_PRESET_RELATIVE_PATH


def test_the_learning_rate_is_written_only_when_the_app_resolved_one(onetrainer):
    """Left unset, a run silently used the preset's 0.0003 while the SAME
    dataset trained at 0.0001 on ai-toolkit — a 3x divergence with nothing on
    screen. Absent still means 'the preset decides', for callers with no view."""
    ots, _cfg = onetrainer
    common = dict(trigger='t', dataset_folder='d', training_folder='tf',
                  steps=100, num_images=10, rank=32)
    assert 'learning_rate' not in ots.build_job_config(**common)
    assert ots.build_job_config(**common, learning_rate=0.0001)['learning_rate'] == 0.0001


def test_resolution_is_a_string_and_overridable(onetrainer):
    ots, _cfg = onetrainer
    common = dict(trigger='t', dataset_folder='d', training_folder='tf',
                  steps=100, num_images=10, rank=32)
    assert ots.build_job_config(**common)['resolution'] == str(ots.KREA2_RESOLUTION)
    assert ots.build_job_config(**common, resolution=768)['resolution'] == '768'


def test_launch_training_forwards_the_datasets_lr_and_resolution(
        onetrainer, tmp_path, monkeypatch, app):
    """Both come from the SAME resolvers the ai-toolkit lane uses, so the two
    lanes cannot drift apart on the values the app owns."""
    import json as _json
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    ots, cfg = onetrainer
    _installed_onetrainer(cfg, tmp_path)
    monkeypatch.setattr(lt, '_lr_eff', lambda _ds: 0.00012)
    monkeypatch.setattr(lt, '_effective_resolution', lambda _ds: [768, 1024])

    class FakeProc:
        pid = 888
        def poll(self):
            return None
    monkeypatch.setattr(ots.subprocess, 'Popen', lambda *a, **k: FakeProc())

    with app.app_context():
        ds = _trainable_krea_dataset(svc, LOCAL_USER, 'OT cfg')
        result = ots.launch_training(LOCAL_USER, ds.id, steps=100, check_captions=False)
        written = _json.loads(open(result['config_path'], encoding='utf-8').read())
    assert written['learning_rate'] == 0.00012
    assert written['resolution'] == '1024', 'the largest of the resolution list'


# --- what the Advanced-options panel is worth on this lane -------------------
#
# The declaration these tests guard exists because the panel showed ~40 settings
# and three of them reached OneTrainer, with nothing on screen saying so. A
# hand-written list would drift from `build_job_config` the first time a field
# moved — which is the same bug wearing a different hat — so the list is held to
# the code here.

def test_every_job_config_key_is_declared(onetrainer, tmp_path):
    """A field added to the job config without a status entry fails HERE.

    `build_job_config`'s output is the ground truth for what this lane sends.
    Anything in it that maps to a panel setting must be declared, or the panel
    goes on greying out something it now honours.
    """
    ots, _cfg = onetrainer
    written = ots.build_job_config(
        trigger='ztrig', dataset_folder=str(tmp_path / 'ds'),
        training_folder=str(tmp_path / 'run'), steps=100, num_images=10,
        rank=32, learning_rate=0.0001, resolution=768)
    for setting, config_key in ots.JOB_CONFIG_SETTING_KEYS.items():
        assert config_key in written, (
            f'{setting} is declared as reaching the job config under '
            f'{config_key!r}, and it is not there')
        assert ots.setting_status(setting)[0] == ots.SETTING_APPLIES, (
            f'{setting} reaches the job config but is not declared as applying')


def test_the_pinned_rank_is_the_rank_the_launch_actually_passes(
        onetrainer, monkeypatch, app, tmp_path):
    """The 'rank is pinned' claim is checked against the CALL, not a comment.

    `launch_training` hardcodes the rank. If someone later wires the panel's
    rank through, this test fails and the declaration has to stop saying the
    control does nothing — which is exactly the direction the drift should
    travel.
    """
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    ots, cfg = onetrainer
    _installed_onetrainer(cfg, tmp_path)
    monkeypatch.setattr(lt, '_lr_eff', lambda _ds: 0.0001)
    monkeypatch.setattr(lt, '_effective_resolution', lambda _ds: [1024])

    # The real launch path, with only the process faked: a stubbed `launch`
    # would only prove what the stub was handed, and the claim is about what
    # ends up in the job OneTrainer reads.
    class FakeProc:
        pid = 889

        def poll(self):
            return None

    import json as _j
    monkeypatch.setattr(ots.subprocess, 'Popen', lambda *a, **k: FakeProc())
    with app.app_context():
        ds = _trainable_krea_dataset(svc, LOCAL_USER, 'OT pinned rank')
        result = ots.launch_training(LOCAL_USER, ds.id, steps=100,
                                     check_captions=False)
        written = _j.loads(open(result['config_path'], encoding='utf-8').read())

    assert written['lora_rank'] == ots.PINNED_RANK
    assert written['lora_alpha'] == float(ots.PINNED_RANK), 'alpha follows the rank'
    assert ots.setting_status('rank')[0] == ots.SETTING_PINNED
    assert ots.setting_status('alpha')[0] == ots.SETTING_PINNED


def test_a_pinned_or_preset_setting_carries_no_promise_of_applying(onetrainer):
    """Only three settings may claim to apply. The count is asserted on purpose:
    growing it is a decision, not something that should happen by accident."""
    ots, _cfg = onetrainer
    applying = sorted(k for k, v in ots.ONETRAINER_SETTING_STATUS.items()
                      if v[0] == ots.SETTING_APPLIES)
    assert applying == ['dual_captions', 'learning_rate', 'resolution']


def test_every_pinned_setting_says_why(onetrainer):
    """A greyed control with no reason teaches nothing — the user is left to
    guess whether it is broken or deliberate."""
    ots, _cfg = onetrainer
    for key, (state, why) in ots.ONETRAINER_SETTING_STATUS.items():
        if state == ots.SETTING_PINNED:
            assert why.strip(), f'{key} is pinned and does not say why'


def test_an_unknown_setting_is_reported_as_preset_owned(onetrainer):
    """Failing safe: a setting this module never heard of is certainly not one
    it sends, so it must not be shown as live."""
    ots, _cfg = onetrainer
    assert ots.setting_status('something_invented_later')[0] == ots.SETTING_PRESET


def test_the_settings_status_route_answers_without_onetrainer_installed(client):
    """The panel needs this to render its greyed state on a machine that has
    not installed OneTrainer yet — gating it would leave those users looking at
    controls that silently do nothing, which is the state being fixed."""
    r = client.get('/api/train/onetrainer/settings-status')
    assert r.status_code == 200
    body = r.get_json()['settings']
    assert body['learning_rate']['state'] == 'applies'
    assert body['rank']['state'] == 'pinned'
    assert body['rank']['why'], 'a greyed control with no reason teaches nothing'
    assert body['optimizer']['state'] == 'preset'
