"""Advanced training settings that are stored GLOBALLY but mean something
FAMILY-specific.

`train_settings` is one JSON blob per dataset while every recipe in
`lora_training` is calibrated per family. Three settings were carried across a
family switch with no memory and no warning:

  * quantize / quantize_te / low_vram — the calibrated defaults are True on the
    12B DiTs (krea, flux, flux2klein, zimage) and False on the 2B ones (anima,
    sdxl). A `False` chosen on Anima produced, after switching to Krea 2, a job
    config with `quantize: false`, no `low_vram` and no `qtype`: an unquantised
    12B model on a machine the recipe assumes has 24 GB. Nothing warned.
  * timestep_type — 'sigmoid' picked under Z-Image overwrote the canonical
    'weighted' of FLUX.2 Klein / Anima. No error, no slowdown, a different LoRA.
  * resolution — global on purpose (see _FAMILY_SCOPED_SETTING_KEYS); the tests
    below pin the guarantees that make that safe.

The two get DIFFERENT treatments and that is the point: a memory for the setting
whose meaning is family-bound (timestep_type), a warning for the one that is a
statement about the card (the memory savers). See `memory_saving_risk`.
"""
import json

import pytest

from app.config import LOCAL_USER


@pytest.fixture()
def anima_ds(app):
    """A dataset on Anima (2B) — the family whose calibrated default is
    "no quantisation", i.e. where switching the savers off is the NORMAL,
    harmless thing to do. That is what makes it the dangerous origin."""
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'anime girl', 'trg',
                                kind='character', train_type='anima')
        return ds.id


def _settings(app, ds_id):
    from app.services import face_dataset_service as svc
    with app.app_context():
        return json.loads(svc.get_dataset(LOCAL_USER, ds_id).train_settings or '{}')


def _krea_model_block(app, ds_id, tmp_path):
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, ds_id)
        cfg = lt._build_job_config_krea(ds, str(tmp_path / 'imgs'), 1000,
                                        training_folder=str(tmp_path))
        return cfg['config']['process'][0]


# --- 1) the reported danger: an OOM recipe built in silence --------------------

def test_savers_disabled_elsewhere_still_reach_a_12b_config(app, anima_ds, tmp_path):
    """MEASURED repro. The flags DO travel — that part is by design (they are a
    statement about the card) — so this test pins the emitted config and the
    NEXT one pins the warning that must accompany it."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        lt.update_train_settings(LOCAL_USER, anima_ds,
                                 {'quantize': False, 'quantize_te': False,
                                  'low_vram': False})
        svc.set_train_type(LOCAL_USER, anima_ds, 'krea')

    proc = _krea_model_block(app, anima_ds, tmp_path)
    model = proc['model']
    assert model['quantize'] is False and model['quantize_te'] is False
    assert 'low_vram' not in model      # omitted == False in ai-toolkit's ModelConfig
    assert 'qtype' not in model         # no quantisation → the key means nothing


def test_the_12b_run_says_the_savers_are_off(app, anima_ds):
    """RED before the fix: `training_preflight` had no row at all about the
    memory levers, so an unquantised 12B run started with nothing said and died
    (or crawled) mid-run — hours of GPU, and real money on the cloud lane."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        lt.update_train_settings(LOCAL_USER, anima_ds,
                                 {'quantize': False, 'quantize_te': False,
                                  'low_vram': False})
        svc.set_train_type(LOCAL_USER, anima_ds, 'krea')
        rep = lt.training_preflight(LOCAL_USER, anima_ds, train_type='krea')
        row = next((c for c in rep['checks'] if c['id'] == 'memory_saving'), None)
        assert row is not None, 'no preflight row about the disabled memory savers'
        assert row['scope'] == 'dataset'      # travels with the job, cloud included
        assert 'Quantise base model' in row['detail']


def test_the_cloud_lane_states_the_pod_requirement(app, anima_ds):
    """The cloud lane is where this mistake bills real money, so the row is NOT
    a 'machine' row that the lane filter drops — and it names the requirement
    rather than reading a local card that will not run the job."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        lt.update_train_settings(LOCAL_USER, anima_ds, {'quantize': False})
        svc.set_train_type(LOCAL_USER, anima_ds, 'krea')
        rep = lt.training_preflight(LOCAL_USER, anima_ds, train_type='krea',
                                    lane='cloud')
        assert any(c['id'] == 'memory_saving' for c in rep['checks'])
        assert any('pod' in w.lower() and 'GB' in w for w in rep['warnings'])


def test_the_warning_is_provenance_blind(app):
    """Set directly on Krea 2, never carried from anywhere: the same danger, so
    the same sentence. This is the argument for a warning over a per-family
    memory — a memory only ever catches the carried-over half."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'k', 'trg', kind='character',
                                train_type='krea')
        lt.update_train_settings(LOCAL_USER, ds.id, {'low_vram': False})
        assert lt.memory_saving_risk(svc.get_dataset(LOCAL_USER, ds.id),
                                     'krea')['disabled'] == ['low_vram']


def test_switching_a_saver_ON_is_never_a_risk(app, anima_ds):
    """The other direction costs precision and speed, never a run — reporting it
    would be exactly the noise that teaches people to click through preflights."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        lt.update_train_settings(LOCAL_USER, anima_ds, {'quantize': True})
        ds = svc.get_dataset(LOCAL_USER, anima_ds)
        assert lt.memory_saving_risk(ds, 'anima') is None
        rep = lt.training_preflight(LOCAL_USER, anima_ds, train_type='anima')
        assert not any(c['id'] == 'memory_saving' for c in rep['checks'])


# --- 2) timestep_type: the setting that gets a MEMORY, both directions ---------

def test_timestep_type_does_not_leak_onto_the_next_family(app, anima_ds):
    """RED before the fix: 'sigmoid' chosen under Z-Image was still in
    train_settings on FLUX.2 Klein, whose canonical schedule is 'weighted' —
    silently a different LoRA, with nothing to observe afterwards."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        svc.set_train_type(LOCAL_USER, anima_ds, 'zimage')
        lt.update_train_settings(LOCAL_USER, anima_ds, {'timestep_type': 'sigmoid'})
        svc.set_train_type(LOCAL_USER, anima_ds, 'flux2klein')
        ds = svc.get_dataset(LOCAL_USER, anima_ds)
        assert 'timestep_type' not in _settings(app, anima_ds)
        assert lt._timestep_type_eff(ds, lt._DEFAULT_TIMESTEP['flux2klein']) == 'weighted'


def test_coming_back_restores_what_was_left_there(app, anima_ds):
    """The other direction of the same switch: nothing was destroyed."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        svc.set_train_type(LOCAL_USER, anima_ds, 'zimage')
        lt.update_train_settings(LOCAL_USER, anima_ds, {'timestep_type': 'shift'})
        svc.set_train_type(LOCAL_USER, anima_ds, 'flux2klein')
        lt.update_train_settings(LOCAL_USER, anima_ds, {'timestep_type': 'linear'})
        svc.set_train_type(LOCAL_USER, anima_ds, 'zimage')
        assert _settings(app, anima_ds)['timestep_type'] == 'shift'
        svc.set_train_type(LOCAL_USER, anima_ds, 'flux2klein')
        assert _settings(app, anima_ds)['timestep_type'] == 'linear'


def test_a_family_left_on_auto_comes_back_on_auto(app, anima_ds):
    """`{}` remembered (configured, everything on Auto) must not read as "never
    configured" and must not resurrect a value from a third family."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        svc.set_train_type(LOCAL_USER, anima_ds, 'zimage')      # nothing set here
        svc.set_train_type(LOCAL_USER, anima_ds, 'krea')
        lt.update_train_settings(LOCAL_USER, anima_ds, {'timestep_type': 'shift'})
        svc.set_train_type(LOCAL_USER, anima_ds, 'zimage')
        assert 'timestep_type' not in _settings(app, anima_ds)
        assert svc.remembered_family_settings(
            svc.get_dataset(LOCAL_USER, anima_ds), 'zimage') == {}


def test_the_memory_survives_a_corrupt_blob(app, anima_ds):
    """Same degrade-to-nothing discipline as _train_settings/family_base_memory."""
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, anima_ds)
        ds.train_family_settings = 'not json at all'
        svc.db.session.commit()
        assert svc.family_settings_memory(svc.get_dataset(LOCAL_USER, anima_ds)) == {}
        assert svc.set_train_type(LOCAL_USER, anima_ds, 'krea') is True


# --- 3) resolution stays global — the guarantees that make that safe -----------

def test_krea_never_emits_a_resolution_outside_the_allowed_set(app, anima_ds, tmp_path):
    """`update_train_settings` is the only writer and it refuses anything outside
    _RES_CHOICES, so no family switch can smuggle an unsupported resolution into
    a Krea 2 config. (Note: contrary to the inventory note, Krea does NOT pin
    1024 — it emits `_train_res(ds)`, default [768, 1024].)"""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        svc.set_train_type(LOCAL_USER, anima_ds, 'krea')
        with pytest.raises(ValueError):
            lt.update_train_settings(LOCAL_USER, anima_ds, {'resolution': '512'})
        with pytest.raises(ValueError):
            lt.update_train_settings(LOCAL_USER, anima_ds, {'resolution': 1024})
        lt.update_train_settings(LOCAL_USER, anima_ds, {'resolution': '768'})
    proc = _krea_model_block(app, anima_ds, tmp_path)
    assert proc['datasets'][0]['resolution'] == [768]


def test_a_768_run_is_not_told_to_drop_to_768(app, anima_ds, monkeypatch):
    """The 24 GB row used to end with "Drop the resolution to 768" whatever the
    resolution already was. Advising a change the user already made is how a
    preflight teaches people to click through it."""
    from app import capabilities
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: 12)
    with app.app_context():
        svc.set_train_type(LOCAL_USER, anima_ds, 'krea')
        lt.update_train_settings(LOCAL_USER, anima_ds, {'resolution': '768'})
        rep = lt.training_preflight(LOCAL_USER, anima_ds, train_type='krea')
        row = next(c for c in rep['checks'] if c['id'] == 'vram')
        assert 'already at 768' in row['detail']
        assert not any('Drop the resolution to 768' in w for w in rep['warnings'])

        lt.update_train_settings(LOCAL_USER, anima_ds, {'resolution': '768,1024'})
        rep = lt.training_preflight(LOCAL_USER, anima_ds, train_type='krea')
        assert any('Drop the resolution to 768' in w for w in rep['warnings'])


# --- 4) anti-regression: a single-family dataset changes NOTHING ---------------

def test_a_single_family_dataset_is_untouched(app):
    """No family switch → no stash, no restore, no new key, and a preflight with
    no memory row. Byte-identical to before this change."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'solo', 'trg', kind='character',
                                train_type='krea')
        lt.update_train_settings(LOCAL_USER, ds.id,
                                 {'timestep_type': 'shift', 'rank': 32})
        before = svc.get_dataset(LOCAL_USER, ds.id).train_settings
        svc.set_train_type(LOCAL_USER, ds.id, 'krea')          # same family = no-op
        after = svc.get_dataset(LOCAL_USER, ds.id)
        assert after.train_settings == before
        assert after.train_family_settings is None
        rep = lt.training_preflight(LOCAL_USER, ds.id, train_type='krea')
        assert not any(c['id'] == 'memory_saving' for c in rep['checks'])


def test_a_legacy_dataset_reads_back_identical(app):
    """The migration is additive and lazy: a dataset that predates the column has
    NULL there, keeps every setting it had, and emits the same config."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'legacy', 'trg', kind='character',
                                train_type='flux2klein')
        ds.train_settings = json.dumps({'timestep_type': 'sigmoid', 'rank': 16})
        ds.train_family_settings = None
        svc.db.session.commit()
        ds = svc.get_dataset(LOCAL_USER, ds.id)
        assert svc.family_settings_memory(ds) == {}
        # Untouched: no family switch happened, so the stored sigmoid still wins.
        assert lt._timestep_type_eff(ds, 'weighted') == 'sigmoid'


# --- 5) the duplicated family-label map ---------------------------------------

def test_family_label_is_defined_once_and_knows_zimage(app):
    """It was declared twice in lora_training; the second shadowed the first, so
    the first (which had no 'zimage') was dead code that read as a bug."""
    import inspect
    from app.services import lora_training as lt
    src = inspect.getsource(lt)
    assert src.count('\n_FAMILY_LABEL = {') == 1
    assert lt._FAMILY_LABEL['zimage'] == 'Z-Image'
