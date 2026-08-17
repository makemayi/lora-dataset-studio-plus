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
    # No batch chosen = the preset's own survives, so this app writes nothing.
    assert 'batch_size' not in config
    # A STRING, matching OneTrainer's own schema — its shipped Krea 2 preset
    # writes "512", not 512. This assertion used to demand the int, which was
    # never checked against the preset.
    assert config['resolution'] == '1024'
    assert config['epochs'] == 160         # ceil(2000 * 2 / 25) — the preset's batch
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
    """The dataset's own values reach the config, so the two lanes cannot drift
    apart on what the app owns.

    The rate is now read from the STORED setting rather than through `_lr_eff`:
    that resolver falls back to a family default and never returns None, so
    calling it wrote a learning rate on every run and silently overrode the
    shipped preset's own. This test used to patch the resolver, which is why it
    could not see that.
    """
    import json as _json
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    ots, cfg = onetrainer
    _installed_onetrainer(cfg, tmp_path)
    monkeypatch.setattr(lt, '_effective_resolution', lambda _ds: [768, 1024])

    class FakeProc:
        pid = 888
        def poll(self):
            return None
    monkeypatch.setattr(ots.subprocess, 'Popen', lambda *a, **k: FakeProc())

    with app.app_context():
        ds = _trainable_krea_dataset(svc, LOCAL_USER, 'OT cfg')
        lt.update_train_settings(LOCAL_USER, ds.id, {'learning_rate': 0.00012})
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
    """The applying set is asserted by NAME, on purpose: growing it is a
    decision, not something that should happen by accident. It went from three
    to five when the panel started asking for epochs and batch in OneTrainer's
    own vocabulary instead of deriving them from a step count."""
    ots, _cfg = onetrainer
    applying = sorted(k for k, v in ots.ONETRAINER_SETTING_STATUS.items()
                      if v[0] == ots.SETTING_APPLIES)
    assert applying == ['batch_size', 'dual_captions', 'epochs',
                        'learning_rate', 'lr_scheduler', 'min_snr_gamma',
                        'resolution', 'te1_lr', 'te2_lr', 'warmup']


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


# --- epochs and batch, in OneTrainer's own vocabulary ------------------------

def test_the_caller_epochs_and_batch_win_over_the_step_derivation(onetrainer, tmp_path):
    """The panel asks for these in OneTrainer's vocabulary, so what it asks for
    is what runs — no approximation in between."""
    ots, _cfg = onetrainer
    c = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=4000, num_images=1200, rank=32, epochs=10, batch_size=3)
    assert c['epochs'] == 10
    assert c['batch_size'] == 3


def test_the_derivation_survives_for_callers_with_no_opinion_and_now_counts_batch(
        onetrainer, tmp_path):
    """A run launched from anywhere but that panel still works — and the old
    form silently assumed batch 1, so at batch 3 it bought a THIRD of the
    training it promised. One epoch is images/batch optimizer steps."""
    ots, _cfg = onetrainer
    at_default = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=2000, num_images=25, rank=32)
    # No batch chosen: the arithmetic assumes the PRESET's batch, because that
    # is what will actually run once this app stops writing a 1 over it.
    assert at_default['epochs'] == 160     # ceil(2000 * 2 / 25)
    assert 'batch_size' not in at_default

    at_three = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=2000, num_images=25, rank=32, batch_size=3)
    assert at_three['epochs'] == 240       # ceil(2000 * 3 / 25)
    assert at_three['batch_size'] == 3


def test_epochs_and_batch_are_refused_when_they_are_a_typo(app):
    """Bounded, not merely positive: a four-digit epoch count or a batch no
    consumer card can hold is a typo, and a typo that reaches a trainer costs
    hours before it says so."""
    import pytest as _pytest
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'ot-bounds', 'ztrig')
        for bad in ({'epochs': 0}, {'epochs': 5000}, {'epochs': 2.5},
                    {'batch_size': 0}, {'batch_size': 999}, {'batch_size': True}):
            with _pytest.raises(ValueError):
                lt.update_train_settings(LOCAL_USER, ds.id, bad)
        eff = lt.update_train_settings(LOCAL_USER, ds.id, {'epochs': 10, 'batch_size': 3})
        assert eff is not None
        stored = lt._train_settings(svc.get_dataset(LOCAL_USER, ds.id))
        assert stored['epochs'] == 10 and stored['batch_size'] == 3
        # 'auto' clears it, the same three-way contract every other key uses.
        lt.update_train_settings(LOCAL_USER, ds.id, {'epochs': 'auto'})
        assert 'epochs' not in lt._train_settings(svc.get_dataset(LOCAL_USER, ds.id))


# --- the text encoders ------------------------------------------------------

def test_a_text_encoder_rate_also_unfreezes_the_encoder(onetrainer, tmp_path):
    """The shipped Krea 2 preset ships `"text_encoder": {"train": false}`, so a
    learning rate on its own is a number attached to a component that never
    learns — stored, and inert."""
    ots, _cfg = onetrainer
    c = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=32, te1_lr=1e-5, te2_lr=5e-6)
    assert c['text_encoder'] == {'train': True, 'learning_rate': 1e-5}
    assert c['text_encoder_2'] == {'train': True, 'learning_rate': 5e-6}


def test_the_text_encoder_keys_are_spelled_the_way_onetrainer_reads_them(
        onetrainer, tmp_path):
    """OneTrainer's BaseConfig.from_dict PRINTS on an unknown key and carries on
    — a misspelling is not an error there, it is a line in a log and a run that
    quietly ignores the setting. So the names are pinned here, where a typo
    fails loudly."""
    ots, _cfg = onetrainer
    c = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=32, te1_lr=1e-5)
    assert 'text_encoder' in c, 'OneTrainer names the first one text_encoder'
    assert set(c['text_encoder']) == {'train', 'learning_rate'}, (
        'only these two — a partial nested dict merges, and naming anything '
        'else here would overwrite what the preset tuned')


def test_no_text_encoder_key_at_all_when_the_user_set_no_rate(onetrainer, tmp_path):
    """Silence is the ownership boundary: without a rate the preset's own
    text-encoder block must arrive untouched."""
    ots, _cfg = onetrainer
    c = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=32)
    assert 'text_encoder' not in c
    assert 'text_encoder_2' not in c


def test_the_learning_rate_is_left_to_the_preset_unless_the_user_chose_one(
        onetrainer, monkeypatch, app, tmp_path):
    """MEASURED: the shipped Krea 2 preset asks for 0.0003 and every run this
    app launched wrote 0.0001 over it, because the resolver it called falls back
    to a family default rather than returning None. Three times off, chosen by
    OneTrainer's maintainers for this model, replaced by a number picked for a
    different trainer, with nothing on screen saying so."""
    import json as _j
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    ots, cfg = onetrainer
    _installed_onetrainer(cfg, tmp_path)
    monkeypatch.setattr(lt, '_effective_resolution', lambda _ds: [1024])

    class FakeProc:
        pid = 991

        def poll(self):
            return None

    monkeypatch.setattr(ots.subprocess, 'Popen', lambda *a, **k: FakeProc())
    with app.app_context():
        ds = _trainable_krea_dataset(svc, LOCAL_USER, 'OT lr silence')
        r = ots.launch_training(LOCAL_USER, ds.id, steps=100, check_captions=False)
        written = _j.loads(open(r['config_path'], encoding='utf-8').read())
        assert 'learning_rate' not in written, (
            'no stored rate = the preset decides, which is what the ownership '
            'rule at the top of build_job_config always said')

        lt.update_train_settings(LOCAL_USER, ds.id, {'learning_rate': 3e-4})
        r2 = ots.launch_training(LOCAL_USER, ds.id, steps=100, check_captions=False)
        written2 = _j.loads(open(r2['config_path'], encoding='utf-8').read())
        assert written2['learning_rate'] == 3e-4


# --- the schedule, and the batch the preset asked for -----------------------

def test_the_shipped_preset_really_does_ask_for_that_batch(onetrainer):
    """KREA2_PRESET_BATCH_SIZE is a constant so the epoch arithmetic has a
    number even when the install is not reachable. Checked against the real
    file when it IS, so the constant cannot drift into a comfortable fiction."""
    import json as _j
    import os as _os
    root = _os.environ.get('LDS_ONETRAINER_DIR') or r'E:\OneTrainer'
    ots, _cfg = onetrainer
    for rel in (ots.KREA2_PRESET_RELATIVE_PATH, ots.KREA2_PRESET_24GB_RELATIVE_PATH):
        path = _os.path.join(root, rel)
        if not _os.path.isfile(path):
            import pytest as _p
            _p.skip(f'OneTrainer preset not installed here: {rel}')
        with open(path, encoding='utf-8') as fh:
            assert _j.load(fh).get('batch_size') == ots.KREA2_PRESET_BATCH_SIZE


def test_no_batch_chosen_leaves_the_presets_own_alone(onetrainer, tmp_path):
    """MEASURED: the shipped preset asks for batch 2 and this app wrote 1 over
    it on every run — silently halving the throughput the preset was tuned
    around, the same way it silently overrode the learning rate."""
    ots, _cfg = onetrainer
    c = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=32)
    assert 'batch_size' not in c
    chosen = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=32, batch_size=3)
    assert chosen['batch_size'] == 3


def test_the_scheduler_is_mapped_into_onetrainers_own_vocabulary(onetrainer, tmp_path):
    """The two vocabularies are NOT the same list. An unmapped string is not an
    error in OneTrainer — it is a printed line and a run that keeps its
    default — so the mapping is asserted rather than trusted."""
    ots, _cfg = onetrainer

    def sched(name, **kw):
        return ots.build_job_config(
            trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
            steps=100, num_images=10, rank=32, lr_scheduler=name, **kw)

    assert sched('cosine')['learning_rate_scheduler'] == 'COSINE'
    assert sched('cosine_with_restarts')['learning_rate_scheduler'] == 'COSINE_WITH_RESTARTS'
    assert sched('linear')['learning_rate_scheduler'] == 'LINEAR'
    # Every mapped name must be a real member of OneTrainer's enum.
    assert set(ots._SCHEDULER_TO_ONETRAINER.values()) <= {
        'CONSTANT', 'LINEAR', 'COSINE', 'COSINE_WITH_RESTARTS',
        'COSINE_WITH_HARD_RESTARTS', 'REX', 'ADAFACTOR', 'CUSTOM'}


def test_warmup_rides_only_with_the_schedule_that_means_warmup(onetrainer, tmp_path):
    """OneTrainer has no constant_with_warmup member: it says CONSTANT and
    carries the warmup in its own field. Attaching that field to every schedule
    would hand it a warmup nobody asked for."""
    ots, _cfg = onetrainer

    def sched(name, warmup):
        return ots.build_job_config(
            trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
            steps=100, num_images=10, rank=32, lr_scheduler=name, warmup_steps=warmup)

    withw = sched('constant_with_warmup', 200)
    assert withw['learning_rate_scheduler'] == 'CONSTANT'
    assert withw['learning_rate_warmup_steps'] == 200.0

    without = sched('cosine', 200)
    assert without['learning_rate_scheduler'] == 'COSINE'
    assert 'learning_rate_warmup_steps' not in without


def test_an_unknown_scheduler_name_writes_nothing(onetrainer, tmp_path):
    """Failing safe: a value this map has never heard of leaves OneTrainer's own
    default in place rather than handing it a string it will only print about."""
    ots, _cfg = onetrainer
    c = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=32, lr_scheduler='rex_but_misspelled')
    assert 'learning_rate_scheduler' not in c


def test_min_snr_gamma_is_written_in_onetrainers_MODERN_shape(onetrainer, tmp_path):
    """The field named `min_snr_gamma` would be silently ignored on this path.

    Over there it is a LEGACY name that __migration_2 rewrites into
    loss_weight_fn + loss_weight_strength, and migrations only run when
    `migrate=True` — which train.py sets to `preset_path is None`, and this app
    always passes a preset. Right name, right value, no effect: exactly the
    class of mistake that copying a field list from elsewhere produces.
    """
    ots, _cfg = onetrainer
    c = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=32, min_snr_gamma=5)
    assert c['loss_weight_fn'] == 'MIN_SNR_GAMMA'
    assert c['loss_weight_strength'] == 5.0
    assert 'min_snr_gamma' not in c, 'the legacy name never reaches this config'

    off = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=32)
    assert 'loss_weight_fn' not in off and 'loss_weight_strength' not in off
