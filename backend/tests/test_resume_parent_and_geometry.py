"""A resume inherits its parent — and its LoRA geometry — from the CHECKPOINT it
loads, not from whichever record happens to be newest in the lane.

Reproduces the Estelle incident: a lane held two cloud runs (a rank-64 one, then a
rank-32 one) whose saves share one local run dir. Continuing the rank-64 file drew
the lineage edge to the rank-32 record — the graph then claimed a continuation that
is physically impossible (rank-32 weights cannot load into a rank-64 network), and
the launch stamped the dataset's LIVE rank onto the run instead of the checkpoint's.
"""
import json

import pytest


def _rec(dataset_id, family='krea', variant='base', steps=1000, version=1,
         source='cloud', cloud_run_id=None, settings=None):
    from app.models import TrainingRunRecord
    from app.extensions import db
    r = TrainingRunRecord(
        dataset_id=dataset_id, family=family, source=source, base_model='',
        variant=variant, steps=steps, version=version, fingerprint='fp',
        manifest='[]', cloud_run_id=cloud_run_id,
        settings=json.dumps(settings) if settings else None)
    db.session.add(r)
    db.session.commit()
    return r


# --- the pure helpers ---------------------------------------------------------

def test_network_geometry_reads_rank_and_alpha(app):
    from app.services import checkpoint_registry as reg
    with app.app_context():
        rec = _rec(1, settings={'rank': 32, 'alpha': 16, 'lr': 0.0001})
        assert reg.network_geometry(rec) == {'rank': 32, 'alpha': 16}


def test_network_geometry_omits_legacy_boolean_alpha(app):
    from app.services import checkpoint_registry as reg
    with app.app_context():
        rec = _rec(1, settings={'rank': 32, 'alpha': True})
        assert reg.network_geometry(rec) == {'rank': 32}


def test_network_geometry_omits_legacy_float_alpha(app):
    from app.services import checkpoint_registry as reg
    with app.app_context():
        rec = _rec(1, settings={'rank': 32, 'alpha': 16.0})
        assert reg.network_geometry(rec) == {'rank': 32}


def test_network_geometry_reads_known_adapter_topology(app):
    from app.services import checkpoint_registry as reg
    with app.app_context():
        rec = _rec(1, settings={
            'rank': 32, 'alpha': 16, 'network_type': 'lokr',
            'lokr_factor': 'auto', 'lokr_full_rank': False,
        })
        assert reg.network_geometry(rec) == {
            'rank': 32, 'alpha': 16, 'network_type': 'lokr',
            'lokr_factor': 'auto', 'lokr_full_rank': False,
        }


def test_network_geometry_keeps_legacy_lokr_full_rank_unknown(app):
    """An old LoKr snapshot must expose the missing fact, never invent False."""
    from app.services import checkpoint_registry as reg
    with app.app_context():
        rec = _rec(1, settings={
            'rank': 32, 'alpha': 16, 'network_type': 'lokr', 'lokr_factor': 16,
        })
        assert reg.network_geometry(rec) == {
            'rank': 32, 'alpha': 16, 'network_type': 'lokr', 'lokr_factor': 16,
        }


def test_network_geometry_unknown_stays_empty(app):
    """A legacy record recorded nothing — unknown geometry must never be enforced."""
    from app.services import checkpoint_registry as reg
    with app.app_context():
        assert reg.network_geometry(_rec(1)) == {}
        assert reg.network_geometry(None) == {}


def test_resume_source_checkpoint_prefers_numbered_over_final():
    from app.services import lora_training as lt
    cks = [{'step': 2500, 'filename': 'lora.safetensors', 'final': True, 'record_id': 9},
           {'step': 2500, 'filename': 'lora_000002500.safetensors', 'record_id': 7}]
    assert lt.resume_source_checkpoint(cks, 2500)['record_id'] == 7
    assert lt.resume_source_checkpoint(cks, 999) is None


def test_describe_geometry_conflict():
    from app.services import lora_training as lt
    assert lt.describe_geometry_conflict({'rank': 64, 'alpha': 32}, 64, 32) is None
    assert lt.describe_geometry_conflict({}, 64, 32) is None          # unknown → silent
    msg = lt.describe_geometry_conflict({'rank': 32, 'alpha': 32}, 64, 32)
    assert msg and 'rank 32' in msg and 'rank 64' in msg
    lora_to_lokr = lt.describe_geometry_conflict(
        {'rank': 64, 'alpha': 32, 'network_type': 'lora'}, 64, 32,
        network_type='lokr', lokr_factor=16, lokr_full_rank=False)
    assert lora_to_lokr and 'LoRA' in lora_to_lokr and 'LoKr' in lora_to_lokr
    factor = lt.describe_geometry_conflict(
        {'rank': 64, 'alpha': 32, 'network_type': 'lokr',
         'lokr_factor': 16, 'lokr_full_rank': False}, 64, 32,
        network_type='lokr', lokr_factor=32, lokr_full_rank=False)
    assert factor and 'factor 16' in factor and 'factor 32' in factor
    full_rank = lt.describe_geometry_conflict(
        {'rank': 64, 'alpha': 32, 'network_type': 'lokr',
         'lokr_factor': 16, 'lokr_full_rank': True}, 64, 32,
        network_type='lokr', lokr_factor=16, lokr_full_rank=False)
    assert full_rank and 'lokr_full_rank=True' in full_rank
    assert lt.describe_geometry_conflict(
        {'rank': 64, 'alpha': 32, 'network_type': 'lokr',
         'lokr_factor': 16, 'lokr_full_rank': False}, 64, 32,
        network_type='lokr', lokr_factor=16, lokr_full_rank=False) is None
    # A snapshot from before topology provenance deliberately stays permissive.
    assert lt.describe_geometry_conflict({'rank': 64, 'alpha': 32}, 64, 32,
                                         network_type='lokr', lokr_factor=16,
                                         lokr_full_rank=False) is None


# --- the local lane -----------------------------------------------------------

def _lane(app, monkeypatch, live_rank=64, parent_settings=None, live_settings=None):
    """A krea/base dataset whose run dir holds two runs' saves: an OLD rank-64 run
    (record A) and a NEWER rank-32 run (record B). The file at step 2500 belongs to
    A. Returns (ds, recA, recB, launched)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    ds = svc.create_dataset(LOCAL_USER, 'Lane', 'lanetrig')
    ds.train_type = 'krea'
    ds.train_variant = 'base'
    current = {'rank': live_rank, 'alpha': 32}
    current.update(live_settings or {})
    ds.train_settings = json.dumps(current)
    svc.db.session.commit()
    recA = _rec(ds.id, steps=2500, cloud_run_id=902,
                settings=parent_settings or {'rank': 64, 'alpha': 32})
    recB = _rec(ds.id, steps=3500, version=1, cloud_run_id=903,
                settings={'rank': 32, 'alpha': 32})
    # The run dir as the incident had it: the newest record's own saves are gone,
    # only the OLDER run's step-2500 file is still on disk.
    monkeypatch.setattr(lt, 'list_checkpoints', lambda *a, **k: [
        {'step': 2000, 'filename': 'lora_000002000.safetensors', 'record_id': recB.id},
        {'step': 2500, 'filename': 'lora_000002500.safetensors', 'record_id': recA.id},
    ])
    monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
    # Resuming below the latest archives the run dir aside — irrelevant here and
    # it needs a configured ai-toolkit, so neutralize it.
    monkeypatch.setattr(lt, '_seed_continuation_from', lambda *a, **k: None)
    launched = {}
    monkeypatch.setattr(lt, 'launch_training',
                        lambda *a, **k: launched.update(k) or {'started': True})
    return ds, recA, recB, launched


def test_local_continue_parents_on_the_record_that_made_the_file(app, monkeypatch):
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, recA, recB, launched = _lane(app, monkeypatch, live_rank=64)
        lt.continue_training(LOCAL_USER, ds.id, extra_steps=500)
        # recB is the NEWEST record of the lane and the old code picked it; the
        # step-2500 file was written by recA, so recA is the true parent.
        assert launched['parent_record_id'] == recA.id
        assert launched['resumed_from'] == 2500


def test_local_continue_refuses_a_rank_the_weights_cannot_load(app, monkeypatch):
    """Resuming the rank-32 save while the dataset now says rank 64 must fail
    loudly BEFORE anything launches — never train a fresh LoRA in disguise."""
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, recA, recB, launched = _lane(app, monkeypatch, live_rank=64)
        with pytest.raises(ValueError) as ei:
            lt.continue_training(LOCAL_USER, ds.id, extra_steps=500, from_step=2000)
        assert 'rank' in str(ei.value)
        assert not launched                      # nothing was launched


def test_local_continue_allows_the_matching_rank(app, monkeypatch):
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, recA, recB, launched = _lane(app, monkeypatch, live_rank=32)
        lt.continue_training(LOCAL_USER, ds.id, extra_steps=500, from_step=2000)
        assert launched['parent_record_id'] == recB.id


def test_local_continue_refuses_lora_to_lokr(app, monkeypatch):
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _, _, launched = _lane(
            app, monkeypatch, live_rank=64,
            parent_settings={'rank': 64, 'alpha': 32, 'network_type': 'lora'},
            live_settings={'network_type': 'lokr', 'lokr_factor': 16})
        with pytest.raises(ValueError, match='LoRA'):
            lt.continue_training(LOCAL_USER, ds.id, extra_steps=500)
        assert not launched


def test_local_continue_refuses_lokr_factor_mismatch(app, monkeypatch):
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _, _, launched = _lane(
            app, monkeypatch, live_rank=64,
            parent_settings={'rank': 64, 'alpha': 32, 'network_type': 'lokr',
                             'lokr_factor': 16, 'lokr_full_rank': False},
            live_settings={'network_type': 'lokr', 'lokr_factor': 32})
        with pytest.raises(ValueError, match='factor'):
            lt.continue_training(LOCAL_USER, ds.id, extra_steps=500)
        assert not launched


def test_local_continue_refuses_lokr_full_rank_mismatch(app, monkeypatch):
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _, _, launched = _lane(
            app, monkeypatch, live_rank=64,
            parent_settings={'rank': 64, 'alpha': 32, 'network_type': 'lokr',
                             'lokr_factor': 16, 'lokr_full_rank': True},
            live_settings={'network_type': 'lokr', 'lokr_factor': 16})
        with pytest.raises(ValueError, match='full_rank'):
            lt.continue_training(LOCAL_USER, ds.id, extra_steps=500)
        assert not launched


def test_local_continue_refuses_legacy_lokr_with_unknown_full_rank(app, monkeypatch):
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _, _, launched = _lane(
            app, monkeypatch, live_rank=64,
            parent_settings={'rank': 64, 'alpha': 32, 'network_type': 'lokr',
                             'lokr_factor': 16},
            live_settings={'network_type': 'lokr', 'lokr_factor': 16})
        with pytest.raises(ValueError, match='lokr_full_rank'):
            lt.continue_training(LOCAL_USER, ds.id, extra_steps=500)
        assert not launched


def test_local_continue_allows_matching_lokr_geometry(app, monkeypatch):
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    topology = {'rank': 64, 'alpha': 32, 'network_type': 'lokr',
                'lokr_factor': 16, 'lokr_full_rank': False}
    with app.app_context():
        ds, recA, _, launched = _lane(
            app, monkeypatch, live_rank=64, parent_settings=topology,
            live_settings={'network_type': 'lokr', 'lokr_factor': 16})
        lt.continue_training(LOCAL_USER, ds.id, extra_steps=500)
        assert launched['parent_record_id'] == recA.id


# --- the local→cloud lane (the one that actually burned 3000 steps) -----------

def test_local_to_cloud_continue_inherits_parent_and_geometry(app, monkeypatch):
    from app.services import cloud_training as ct
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, recA, recB, _ = _lane(app, monkeypatch, live_rank=32)
        monkeypatch.setattr(lt, 'checkpoint_file_path',
                            lambda *a, **k: '/tmp/lora_000002500.safetensors')
        launched = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda *a, **k: launched.update(k) or {'run_id': 1})
        ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, extra_steps=500)

        assert launched['parent_record_id'] == recA.id       # not the newest record
        assert launched['resumed_from'] == 2500
        # …and the run trains at the CHECKPOINT's rank, not the dataset's live 32.
        snap = json.loads(launched['train_settings_snapshot'])
        assert snap['rank'] == 64 and snap['alpha'] == 32


def test_local_to_cloud_inherits_known_lokr_topology(app, monkeypatch):
    from app.services import cloud_training as ct
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    topology = {'rank': 64, 'alpha': 32, 'network_type': 'lokr',
                'lokr_factor': 'auto', 'lokr_full_rank': True}
    with app.app_context():
        # The live preset changed back to LoRA but retained a stale factor.  The
        # cloud child must replay the source's complete known topology instead.
        ds, recA, _, _ = _lane(
            app, monkeypatch, live_rank=32, parent_settings=topology,
            live_settings={'network_type': 'lora', 'lokr_factor': 16})
        monkeypatch.setattr(lt, 'checkpoint_file_path',
                            lambda *a, **k: '/tmp/lora_000002500.safetensors')
        launched = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda *a, **k: launched.update(k) or {'run_id': 1})
        ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, extra_steps=500)

        assert launched['parent_record_id'] == recA.id
        snap = json.loads(launched['train_settings_snapshot'])
        assert {k: snap[k] for k in topology} == topology
        view = ct._run_config_dataset(ds, {
            'train_type': 'krea', 'variant': 'base', 'base_model': '',
            'train_settings_snapshot': launched['train_settings_snapshot'],
        })
        network = lt._network_block(view, 64, 'krea')
        assert network['lokr_full_rank'] is True
        assert 'lokr_factor' not in network       # source's explicit auto wins over live 16


def test_local_to_cloud_refuses_legacy_lokr_with_unknown_full_rank(app, monkeypatch):
    from app.services import cloud_training as ct
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _, _, _ = _lane(
            app, monkeypatch, live_rank=64,
            parent_settings={'rank': 64, 'alpha': 32, 'network_type': 'lokr',
                             'lokr_factor': 16},
            live_settings={'network_type': 'lokr', 'lokr_factor': 16})
        monkeypatch.setattr(lt, 'checkpoint_file_path',
                            lambda *a, **k: '/tmp/lora_000002500.safetensors')
        launched = []
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda *a, **k: launched.append(k))
        with pytest.raises(ValueError, match='lokr_full_rank'):
            ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, extra_steps=500)
        assert launched == []


@pytest.mark.parametrize('setting', ('alpha', 'grad_accum'))
@pytest.mark.parametrize('value', (True, False, 1.0))
def test_update_train_settings_rejects_non_integer_numeric_values(app, setting, value):
    """Neither bools nor equal-comparing floats are valid integer settings."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Strict numeric', 'strictnumeric')
        with pytest.raises(ValueError, match=setting):
            lt.update_train_settings(LOCAL_USER, ds.id, {setting: value})
        assert setting not in lt.snapshot_train_settings(LOCAL_USER, ds.id)


@pytest.mark.parametrize('invalid_value', (True, 1.0), ids=('boolean', 'float'))
def test_legacy_non_integer_numeric_settings_are_ignored_by_all_readers(
        app, tmp_path, invalid_value):
    """Old JSON values must fall back to integer defaults without rewriting it."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Legacy numeric', 'legacynumeric',
                                train_type='krea')
        ds.train_settings = json.dumps({
            'rank': 32, 'alpha': invalid_value, 'grad_accum': invalid_value,
        })
        svc.db.session.commit()
        folder = tmp_path / 'dataset'
        folder.mkdir()

        proc = lt.build_job_config(
            ds, str(folder), steps=1000,
            training_folder=str(tmp_path / 'training'),
        )['config']['process'][0]
        assert proc['network']['linear_alpha'] == 32
        assert type(proc['network']['linear_alpha']) is int
        assert proc['train']['gradient_accumulation'] == 1
        assert type(proc['train']['gradient_accumulation']) is int

        snap = lt.launch_settings_snapshot(ds, 'krea')
        assert snap['alpha'] == 32 and type(snap['alpha']) is int
        assert snap['grad_accum'] == 1 and type(snap['grad_accum']) is int
        effective = lt.effective_train_settings(ds, 'krea')
        assert effective['alpha'] == 32
        assert effective['alpha_setting'] is None
        assert effective['grad_accum'] is None
        assert lt.snapshot_train_settings(LOCAL_USER, ds.id) == {'rank': 32}
        assert json.loads(ds.train_settings) == {
            'rank': 32, 'alpha': invalid_value, 'grad_accum': invalid_value,
        }
