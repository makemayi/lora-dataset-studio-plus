"""Min-SNR gamma — ONE setting, TWO shapes.

Both trainers support the idea and neither spells it the same way, so the value
is translated on each side rather than passed through. These pin both
translations, because "we wrote the field" is not the same claim as "the trainer
reads it".

  ai-toolkit   train.min_snr_gamma = <value>
               (TrainConfig.min_snr_gamma; jobs/process/BaseSDTrainProcess.py
               only applies it above ~1e-6, so a stored 0 must not be written)

  OneTrainer   loss_weight_fn = MIN_SNR_GAMMA
               loss_weight_strength = <value>
               (`min_snr_gamma` over there is a LEGACY name rewritten by
               __migration_2, and migrations run only when migrate=True —
               train.py ties that to `preset_path is None`, and this app always
               passes a preset, so the legacy name would be silently ignored)
"""
import pytest


def _dataset(app, tmp_path, name):
    from app import config as cfg
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    return svc.create_dataset(LOCAL_USER, name, 'zsnr', train_type='krea')


def test_aitoolkit_gets_a_bare_min_snr_gamma_in_its_train_block(app, tmp_path):
    from app.config import LOCAL_USER
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(app, tmp_path, 'snr-on')
        folder = tmp_path / 'ds'
        folder.mkdir()
        lt.update_train_settings(LOCAL_USER, ds.id, {'min_snr_gamma': 5})
        train = lt.build_job_config(ds, str(folder), steps=1000)['config']['process'][0]['train']
        assert train['min_snr_gamma'] == 5.0


def test_aitoolkit_writes_nothing_when_it_is_unset(app, tmp_path):
    """Absent, not zero: the trainer's own branch treats a 0 as off anyway, and
    writing one would claim a decision the user never made."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(app, tmp_path, 'snr-off')
        folder = tmp_path / 'ds'
        folder.mkdir()
        train = lt.build_job_config(ds, str(folder), steps=1000)['config']['process'][0]['train']
        assert 'min_snr_gamma' not in train


def test_a_stored_zero_is_the_same_as_off(app, tmp_path):
    from app.config import LOCAL_USER
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(app, tmp_path, 'snr-zero')
        folder = tmp_path / 'ds'
        folder.mkdir()
        lt.update_train_settings(LOCAL_USER, ds.id, {'min_snr_gamma': 0})
        train = lt.build_job_config(ds, str(folder), steps=1000)['config']['process'][0]['train']
        assert 'min_snr_gamma' not in train


def test_it_is_refused_outside_a_sane_range(app, tmp_path):
    from app.config import LOCAL_USER
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(app, tmp_path, 'snr-bounds')
        for bad in ({'min_snr_gamma': -1}, {'min_snr_gamma': 100},
                    {'min_snr_gamma': 'five'}):
            with pytest.raises(ValueError):
                lt.update_train_settings(LOCAL_USER, ds.id, bad)


def test_the_two_lanes_translate_the_same_number_differently(app, tmp_path):
    """The point of the setting being shared: one number, two shapes. If these
    ever agree on a field name, one of the translations has been lost."""
    from app.config import LOCAL_USER
    from app.services import lora_training as lt
    from app.services import onetrainer_service as ots
    with app.app_context():
        ds = _dataset(app, tmp_path, 'snr-both')
        folder = tmp_path / 'ds'
        folder.mkdir()
        lt.update_train_settings(LOCAL_USER, ds.id, {'min_snr_gamma': 5})
        train = lt.build_job_config(ds, str(folder), steps=1000)['config']['process'][0]['train']
        ot = ots.build_job_config(
            trigger='x', dataset_folder=str(folder), training_folder=str(folder),
            steps=1000, num_images=10, rank=32, min_snr_gamma=5)
    assert train['min_snr_gamma'] == 5.0
    assert ot['loss_weight_strength'] == 5.0
    assert 'min_snr_gamma' not in ot
    assert 'loss_weight_fn' not in train
