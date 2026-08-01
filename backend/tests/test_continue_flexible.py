"""Flexible continuation: custom step counts, resuming from an EARLIER checkpoint
without destroying the run's later saves, and the safe-subset settings overrides a
continue is allowed to change. Covers both the local (ai-toolkit seed) and the
override-validation path shared with cloud."""
import glob
import json
import os
from pathlib import Path
import threading

import pytest


class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid

    def wait(self):
        return None


def _configure_aitoolkit(tmp_path, monkeypatch, app):
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


def _stub_launch(monkeypatch, tmp_path, app):
    """Neutralize everything past the seed decision so a continue reaches
    launch_training and writes a real job config without spawning ai-toolkit."""
    from app.services import lora_training as lt
    _configure_aitoolkit(tmp_path, monkeypatch, app)
    monkeypatch.setattr(lt.subprocess, 'Popen', lambda a, **k: _FakeProc())
    monkeypatch.setattr(lt, '_watch_training', lambda *a, **k: None)
    monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
    (tmp_path / 'exported').mkdir(exist_ok=True)
    monkeypatch.setattr(lt, 'export_dataset_to_aitoolkit',
                        lambda u, d, masked=True: str(tmp_path / 'exported'))


def _stub_weights_seed(monkeypatch, lt):
    """Unit tests that mock checkpoint discovery do not own a real run folder."""
    monkeypatch.setattr(
        lt, '_seed_continuation_from',
        lambda *args, **kwargs: 'mock-superseded-run')


def _seed_run(lt, svc, LOCAL_USER, name, trig, steps):
    """A zimage/turbo dataset with a real ai-toolkit run dir holding one numbered
    save per step. Returns (dataset, run_dir, trigger)."""
    ds = svc.create_dataset(LOCAL_USER, name, trig)
    ds.train_type = 'zimage'
    ds.train_variant = 'turbo'
    svc.db.session.commit()
    trigger = lt._safe_trigger(ds)
    run_dir = lt._run_dir(LOCAL_USER, ds.id, None, 'zimage', 'turbo')
    os.makedirs(run_dir, exist_ok=True)
    for s in steps:
        p = os.path.join(run_dir, f'lora_{trigger}_{s:09d}.safetensors')
        with open(p, 'wb') as fh:
            fh.write(f'WEIGHTS-{s}'.encode())
    return ds, run_dir, trigger


def test_checkpoint_exposes_exact_state_gate_reason(app, tmp_path, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    from app.config import LOCAL_USER

    with app.app_context():
        _configure_aitoolkit(tmp_path, monkeypatch, app)
        ds, run_dir, _ = _seed_run(
            lt, svc, LOCAL_USER, 'State gate', 'state_gate', [500])
        status_dir = os.path.join(os.path.dirname(run_dir), '.lds-state')
        os.makedirs(status_dir, exist_ok=True)
        with open(
            os.path.join(status_dir, 'bridge-status.json'), 'w', encoding='utf-8'
        ) as status_file:
            json.dump(
                {
                    'status': 'unsupported',
                    'reasons': [
                        'main dataset #1 uses ai-toolkit buckets; '
                        'the v1 bridge does not capture bucket ordering and crop geometry'
                    ],
                },
                status_file,
            )

        checkpoint = lt.list_checkpoints(
            LOCAL_USER, ds.id, None, 'zimage', 'turbo')[0]

    assert checkpoint['resume_state']['status'] == 'missing'
    assert 'bucket ordering and crop geometry' in checkpoint['resume_state']['reason']


_EXACT_PUBLIC_WEIGHTS = b'EXACT-PUBLIC\nWEIGHTS\r\n\x00\xff'


def _create_exact_bundle(tmp_path, save_root, step):
    """Create a structurally complete opaque bundle for service integration."""
    from app.services import training_state_bundle as state
    from app.services.training_state_identity import EXACT_CAPABILITIES
    from app.training_bridge.lds_aitk_bridge_contract import ARTIFACT_FILENAMES

    source_root = tmp_path / f'exact-sources-{step}'
    source_root.mkdir()
    sources = {}
    for logical, filename in ARTIFACT_FILENAMES.items():
        source = source_root / filename
        source.write_bytes(
            # The LF is deliberate: on Windows, an os.open() destination that
            # omits O_BINARY expands it to CRLF and corrupts the checkpoint.
            _EXACT_PUBLIC_WEIGHTS if logical == 'public_checkpoint'
            else f'opaque-{logical}'.encode())
        sources[filename] = source
    metadata = state.BundleMetadata(
        completed_step=step,
        next_step=step + 1,
        optimizer_updates_completed=step,
        toolkit_revision='a' * 40,
        toolkit_runtime={
            'python': '3.12.4',
            'torch': '2.7.1+cu128',
            'cuda': '12.8',
            'cuda_devices': 1,
            'protocol': 1,
            'shape_revision': 'aitk-base-sd-train/v1',
        },
        config_hash='1' * 64,
        dataset_hash='2' * 64,
        base_model_hash='3' * 64,
        network_hash='4' * 64,
        capabilities=tuple(sorted(EXACT_CAPABILITIES)),
        state_level='exact',
    )
    return state.create_bundle(save_root, sources, metadata)


# --- Custom step count --------------------------------------------------------

def test_continue_custom_extra_steps_targets_last_plus_extra(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Custom steps', 'customsteps')
        launched = {}
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(
            lt, 'list_checkpoints',
            lambda *a, **k: [{'step': 250, 'filename': 'a.safetensors'},
                             {'step': 1000, 'filename': 'b.safetensors'}])
        monkeypatch.setattr(
            lt, 'launch_training',
            lambda *a, **k: launched.update(k) or {'started': True})
        _stub_weights_seed(monkeypatch, lt)

        res = lt.continue_training(LOCAL_USER, ds.id, extra_steps=333)

    assert res['resumed_from'] == 1000               # default = latest
    assert res['target_steps'] == 1333
    assert launched['steps'] == 1333                 # last + custom extra


def test_continue_extra_steps_floor_100(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Floor', 'floorextra')
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(lt, 'list_checkpoints',
                            lambda *a, **k: [{'step': 500, 'filename': 'a.safetensors'}])
        monkeypatch.setattr(lt, 'launch_training', lambda *a, **k: {'started': True})
        _stub_weights_seed(monkeypatch, lt)

        res = lt.continue_training(LOCAL_USER, ds.id, extra_steps=5)   # below the floor

    assert res['target_steps'] == 600                # 500 + max(100, 5)


# --- Resume from an EARLIER checkpoint, non-destructively ----------------------

def test_continue_from_lower_checkpoint_seeds_clean_and_keeps_originals(
        app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds, run_dir, trig = _seed_run(lt, svc, LOCAL_USER, 'Lower', 'lowertrig',
                                      [500, 1000])
        # baseline: both saves visible to the hub
        assert [c['step'] for c in lt.list_checkpoints(
            LOCAL_USER, ds.id, None, 'zimage', 'turbo')] == [500, 1000]

        res = lt.continue_training(LOCAL_USER, ds.id, extra_steps=200, from_step=500)

        assert res['resumed_from'] == 500
        assert res['target_steps'] == 700
        with open(res['config_path'], encoding='utf-8') as fh:
            cfg = json.load(fh)
        assert cfg['config']['process'][0]['train']['steps'] == 700

        # Fresh save_root holds ONLY the seeded 500 — ai-toolkit resumes from it,
        # never the over-cooked 1000.
        assert sorted(os.listdir(run_dir)) == [f'lora_{trig}_000000500.safetensors']
        with open(os.path.join(run_dir, f'lora_{trig}_000000500.safetensors'), 'rb') as fh:
            assert fh.read() == b'WEIGHTS-500'         # the exact earlier weights

        # The original run is set aside intact — BOTH saves recoverable, nothing deleted.
        training_folder = os.path.dirname(run_dir)
        superseded = glob.glob(training_folder + '_superseded_*')
        assert len(superseded) == 1
        assert sorted(os.listdir(os.path.join(superseded[0], f'lora_{trig}'))) == [
            f'lora_{trig}_000000500.safetensors',
            f'lora_{trig}_000001000.safetensors']


def test_continue_from_latest_weights_only_archives_and_seeds_clean(
        app, tmp_path, monkeypatch):
    """Weights-only is literal even at the latest step: mutable sidecar state is
    left in the recoverable archived source and only the chosen weights are seeded."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _stub_launch(monkeypatch, tmp_path, app)
    events = []
    original_seed = lt._seed_continuation_from

    def reconcile_before_lane_mutation():
        assert lt._launch_transaction_lock._is_owned()
        assert lt._queue_lock._is_owned()
        events.append('reconcile')
        return False

    def checked_seed(*args, **kwargs):
        assert events == ['reconcile']
        events.append('seed')
        return original_seed(*args, **kwargs)

    monkeypatch.setattr(
        lt, '_reconcile_exact_resume_journals',
        reconcile_before_lane_mutation)
    monkeypatch.setattr(lt, '_seed_continuation_from', checked_seed)
    with app.app_context():
        ds, run_dir, trig = _seed_run(lt, svc, LOCAL_USER, 'Latest', 'latesttrig',
                                      [500, 1000])
        res = lt.continue_training(LOCAL_USER, ds.id, extra_steps=200, from_step=1000)

        assert res['resumed_from'] == 1000 and res['target_steps'] == 1200
        assert res['resume_mode'] == 'weights_only'
        assert sorted(os.listdir(run_dir)) == [
            f'lora_{trig}_000001000.safetensors']
        superseded = glob.glob(os.path.dirname(run_dir) + '_superseded_*')
        assert len(superseded) == 1
        assert sorted(os.listdir(os.path.join(
            superseded[0], f'lora_{trig}'))) == [
                f'lora_{trig}_000000500.safetensors',
                f'lora_{trig}_000001000.safetensors']
        assert events == ['reconcile', 'seed', 'reconcile']


def test_continue_from_missing_step_is_rejected(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds, run_dir, trig = _seed_run(lt, svc, LOCAL_USER, 'Miss', 'misstrig',
                                      [500, 1000])
        with pytest.raises(ValueError, match='no checkpoint at step 750'):
            lt.continue_training(LOCAL_USER, ds.id, from_step=750)
        # nothing archived on the rejected request
        assert glob.glob(os.path.dirname(run_dir) + '_superseded_*') == []


def test_continue_full_state_verifies_archives_and_hands_bundle_to_launch(
        app, tmp_path, monkeypatch):
    from pathlib import Path

    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt

    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds, run_dir, trig = _seed_run(
            lt, svc, LOCAL_USER, 'Exact', 'exacttrig', [500, 1000])
        bundle = _create_exact_bundle(tmp_path, run_dir, 1000)
        launched = {}
        monkeypatch.setattr(
            lt, '_exact_resume_identity',
            lambda *args, **kwargs: ({}, None, []))
        monkeypatch.setattr(
            lt, 'launch_training',
            lambda *args, **kwargs:
            launched.update(kwargs) or {'started': True})

        result = lt.continue_training(
            LOCAL_USER,
            ds.id,
            extra_steps=200,
            from_step=1000,
            resume_mode='full_state',
            state_bundle_id=bundle.bundle_id,
        )

        assert result['resume_mode'] == 'full_state'
        assert result['state_bundle_id'] == bundle.bundle_id
        assert result['target_steps'] == 1200
        assert launched['_state_bridge_required'] is True
        journal = Path(launched['_state_resume_journal'])
        assert journal.is_file()
        assert lt._read_exact_resume_journal(journal)['phase'] == 'seeded'
        restore_bundle = Path(launched['_state_restore_bundle'])
        assert restore_bundle.name == bundle.bundle_id
        assert restore_bundle.is_dir()
        assert str(restore_bundle).startswith(result['archived_run'])
        # The new live lane contains only the public checkpoint required for
        # ai-toolkit construction; private state remains in the canonical bundle.
        assert sorted(os.listdir(run_dir)) == [
            f'lora_{trig}_000001000.safetensors']
        assert (Path(run_dir) / f'lora_{trig}_000001000.safetensors'
                ).read_bytes() == _EXACT_PUBLIC_WEIGHTS


def test_continue_full_state_refuses_corrupt_bundle_before_archiving(
        app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    from app.services import training_state_bundle as state

    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds, run_dir, _trig = _seed_run(
            lt, svc, LOCAL_USER, 'Corrupt exact', 'corruptexact', [1000])
        bundle = _create_exact_bundle(tmp_path, run_dir, 1000)
        bundle_path = state.resolve_bundle_path(
            run_dir, bundle.bundle_id, require_exists=True)
        public = next(
            record for record in bundle.artifacts
            if record.name == 'checkpoint.safetensors')
        bundle_path.joinpath(*public.path.split('/')).write_bytes(b'corrupt')
        launched = []
        monkeypatch.setattr(
            lt, '_exact_resume_identity',
            lambda *args, **kwargs: ({}, None, []))
        monkeypatch.setattr(
            lt, 'launch_training',
            lambda *args, **kwargs:
            launched.append(kwargs) or {'started': True})
        training_folder = os.path.dirname(run_dir)

        with pytest.raises(ValueError, match='missing or corrupt'):
            lt.continue_training(
                LOCAL_USER,
                ds.id,
                from_step=1000,
                resume_mode='full_state',
                state_bundle_id=bundle.bundle_id,
            )

        assert launched == []
        assert os.path.isdir(training_folder)
        assert glob.glob(training_folder + '_superseded_*') == []


def test_exact_seed_rehashes_public_checkpoint_after_bundle_verification(
        app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    from app.services import training_state_bundle as state

    _configure_aitoolkit(tmp_path, monkeypatch, app)
    with app.app_context():
        ds, run_dir, trig = _seed_run(
            lt, svc, LOCAL_USER, 'Seed rehash', 'seed_rehash', [1000])
        bundle = _create_exact_bundle(tmp_path, run_dir, 1000)
        inspected = state.verify_bundle(run_dir, bundle.bundle_id)
        bundle_path = state.resolve_bundle_path(
            run_dir, bundle.bundle_id, require_exists=True)
        public = next(
            record for record in inspected.artifacts
            if record.name == 'checkpoint.safetensors')
        source = bundle_path.joinpath(*public.path.split('/'))
        source.write_bytes(b'X' * public.size_bytes)

        with pytest.raises(ValueError, match='changed before seeding'):
            lt._archive_and_seed_exact_bundle(
                LOCAL_USER,
                ds.id,
                None,
                'zimage',
                'turbo',
                f'lora_{trig}_000001000.safetensors',
                bundle.bundle_id,
                inspected,
            )

        assert Path(run_dir).is_dir()
        assert (
            Path(run_dir) / f'lora_{trig}_000001000.safetensors'
        ).read_bytes() == b'WEIGHTS-1000'
        assert not list(lt._exact_resume_journal_root().glob('*.json'))


def test_exact_seed_refuses_public_checkpoint_replaced_by_symlink(
        app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    from app.services import training_state_bundle as state

    _configure_aitoolkit(tmp_path, monkeypatch, app)
    with app.app_context():
        ds, run_dir, trig = _seed_run(
            lt, svc, LOCAL_USER, 'Seed link', 'seed_link', [1000])
        bundle = _create_exact_bundle(tmp_path, run_dir, 1000)
        inspected = state.verify_bundle(run_dir, bundle.bundle_id)
        bundle_path = state.resolve_bundle_path(
            run_dir, bundle.bundle_id, require_exists=True)
        public = next(
            record for record in inspected.artifacts
            if record.name == 'checkpoint.safetensors')
        source = bundle_path.joinpath(*public.path.split('/'))
        outside = tmp_path / 'outside-public-checkpoint.safetensors'
        outside.write_bytes(b'Z' * public.size_bytes)
        source.unlink()
        try:
            source.symlink_to(outside)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

        with pytest.raises(ValueError, match='not a regular file'):
            lt._archive_and_seed_exact_bundle(
                LOCAL_USER,
                ds.id,
                None,
                'zimage',
                'turbo',
                f'lora_{trig}_000001000.safetensors',
                bundle.bundle_id,
                inspected,
            )

        assert outside.read_bytes() == b'Z' * public.size_bytes
        assert Path(run_dir).is_dir()
        assert not list(lt._exact_resume_journal_root().glob('*.json'))


def test_exact_resume_pre_spawn_journal_failure_restores_lane_and_clears_fence(
        app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt

    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds, run_dir, _trig = _seed_run(
            lt, svc, LOCAL_USER, 'Journal fail', 'journal_fail', [1000])
        bundle = _create_exact_bundle(tmp_path, run_dir, 1000)
        monkeypatch.setattr(
            lt, '_exact_resume_identity',
            lambda *args, **kwargs: ({}, None, []))
        original_update = lt._update_exact_resume_journal

        def fail_launching(path, **changes):
            if changes.get('phase') == 'launching':
                raise OSError('simulated journal fsync failure')
            return original_update(path, **changes)

        monkeypatch.setattr(
            lt, '_update_exact_resume_journal', fail_launching)

        def fake_launch(*_args, **kwargs):
            lt.queue_manager._set_system_state(
                'training_in_progress', True, ttl_seconds=None)
            lt.queue_manager._set_system_state(
                'training_run_token', 'pre-spawn-token', ttl_seconds=None)
            lt._mark_exact_resume_launching(
                kwargs['_state_resume_journal'], 'pre-spawn-token')
            pytest.fail('journal failure should abort before Popen')

        monkeypatch.setattr(lt, 'launch_training', fake_launch)

        with pytest.raises(OSError, match='journal fsync failure'):
            lt.continue_training(
                LOCAL_USER,
                ds.id,
                from_step=1000,
                resume_mode='full_state',
                state_bundle_id=bundle.bundle_id,
            )

        assert os.path.isdir(run_dir)
        assert os.path.isfile(
            os.path.join(
                run_dir, 'lora_journal_fail_000001000.safetensors'))
        assert not list(
            lt._exact_resume_journal_root().glob('*.json'))
        assert lt.queue_manager._get_system_state(
            'training_in_progress', False) is False


def test_exact_resume_transaction_serializes_competitor_before_rollback(
        app, tmp_path, monkeypatch):
    """A second exact launch cannot own the live lane before #1 rolls back."""
    from app.services import lora_training as lt

    entered = threading.Event()
    release = threading.Event()
    second_attempted = threading.Event()
    events = []
    outcomes = []
    archive_number = 0

    with app.app_context():
        lt._clear_training_identity(ttl_seconds=1)

    def fake_archive(*_args, **_kwargs):
        nonlocal archive_number
        assert lt._launch_transaction_lock._is_owned()
        assert lt._queue_lock._is_owned()
        archive_number += 1
        number = archive_number
        events.append(f'archive-{number}')
        if number == 1:
            entered.set()
            assert release.wait(timeout=5)
        return (
            str(tmp_path / f'archive-{number}'),
            str(tmp_path / f'bundle-{number}'),
            str(tmp_path / f'journal-{number}.json'),
        )

    def fake_launch(*_args, **kwargs):
        assert lt._launch_transaction_lock._is_owned()
        archived = Path(kwargs['_state_resume_archived']).name
        events.append(f'launch-{archived[-1]}')
        if archived.endswith('1'):
            raise RuntimeError('pre-spawn failure')
        return {'started': True}

    def fake_rollback(_live, archived):
        assert lt._launch_transaction_lock._is_owned()
        assert lt._queue_lock._is_owned()
        events.append(f'rollback-{Path(archived).name[-1]}')

    monkeypatch.setattr(
        lt, '_archive_and_seed_exact_bundle', fake_archive)
    monkeypatch.setattr(lt, 'launch_training', fake_launch)
    monkeypatch.setattr(
        lt, '_rollback_unlaunched_exact_resume', fake_rollback)
    monkeypatch.setattr(lt, '_delete_exact_resume_journal', lambda _path: None)

    def run(number):
        if number == 2:
            second_attempted.set()
        try:
            with app.app_context():
                result = lt._launch_exact_resume_transaction(
                    'local',
                    number,
                    None,
                    'zimage',
                    'turbo',
                    'checkpoint.safetensors',
                    'a' * 32,
                    object(),
                    tmp_path / 'live',
                    {},
                )
            outcomes.append((number, result[0]['started']))
        except RuntimeError as exc:
            outcomes.append((number, str(exc)))

    first = threading.Thread(target=run, args=(1,))
    second = threading.Thread(target=run, args=(2,))
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    assert second_attempted.wait(timeout=5)
    assert 'archive-2' not in events
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert events == [
        'archive-1', 'launch-1', 'rollback-1', 'archive-2', 'launch-2']
    assert sorted(outcomes) == [
        (1, 'pre-spawn failure'),
        (2, True),
    ]


def test_exact_resume_transaction_reconciles_before_archive_and_seed(
        app, tmp_path, monkeypatch):
    from app.services import lora_training as lt

    observed = []

    def unresolved():
        assert lt._launch_transaction_lock._is_owned()
        assert lt._queue_lock._is_owned()
        observed.append('reconciled')
        return True

    monkeypatch.setattr(
        lt, '_reconcile_exact_resume_journals', unresolved)
    monkeypatch.setattr(
        lt,
        '_archive_and_seed_exact_bundle',
        lambda *_a, **_k: pytest.fail(
            'archive/seed must not run while an older exact lane is held'),
    )

    with app.app_context(), pytest.raises(
            ValueError, match='requires operator recovery'):
        lt._launch_exact_resume_transaction(
            'local',
            1,
            None,
            'zimage',
            'turbo',
            'checkpoint.safetensors',
            'a' * 32,
            object(),
            tmp_path / 'live',
            {},
        )

    assert observed == ['reconciled']


def test_continue_full_state_rejects_unadvertised_bundle_before_identity(
        app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt

    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds, run_dir, trig = _seed_run(
            lt, svc, LOCAL_USER, 'Unadvertised', 'unadvertised', [1000])
        bundle = _create_exact_bundle(tmp_path, run_dir, 1000)
        monkeypatch.setattr(
            lt, 'list_checkpoints',
            lambda *args, **kwargs: [{
                'step': 1000,
                'filename': f'lora_{trig}_000001000.safetensors',
                'resume_state': {'bundle_id': None},
            }])
        identity_calls = []
        monkeypatch.setattr(
            lt, '_exact_resume_identity',
            lambda *args, **kwargs:
            identity_calls.append(True) or ({}, None, []))

        with pytest.raises(
                ValueError,
                match='selected state bundle does not belong'):
            lt.continue_training(
                LOCAL_USER,
                ds.id,
                from_step=1000,
                resume_mode='full_state',
                state_bundle_id=bundle.bundle_id,
            )

        assert identity_calls == []
        assert os.path.isdir(os.path.dirname(run_dir))
        assert glob.glob(os.path.dirname(run_dir) + '_superseded_*') == []


@pytest.mark.parametrize('key,value', [
    ('save_every', 500),
    ('sample_every', 500),
])
def test_continue_full_state_rejects_cadence_override_before_identity(
        app, tmp_path, monkeypatch, key, value):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt

    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds, run_dir, trig = _seed_run(
            lt, svc, LOCAL_USER, f'Cadence {key}', f'cadence_{key}', [1000])
        bundle = _create_exact_bundle(tmp_path, run_dir, 1000)
        monkeypatch.setattr(
            lt, 'list_checkpoints',
            lambda *args, **kwargs: [{
                'step': 1000,
                'filename': f'lora_{trig}_000001000.safetensors',
                'resume_state': {'bundle_id': bundle.bundle_id},
            }])
        identity_calls = []
        monkeypatch.setattr(
            lt, '_exact_resume_identity',
            lambda *args, **kwargs:
            identity_calls.append(True) or ({}, None, []))

        with pytest.raises(ValueError, match='save cadence or sample cadence'):
            lt.continue_training(
                LOCAL_USER,
                ds.id,
                from_step=1000,
                overrides={key: value},
                resume_mode='full_state',
                state_bundle_id=bundle.bundle_id,
            )

        assert identity_calls == []
        assert os.path.isdir(os.path.dirname(run_dir))
        assert glob.glob(os.path.dirname(run_dir) + '_superseded_*') == []


def test_exact_resume_watcher_restores_source_lane_on_bridge_restore_error(
        app, tmp_path, monkeypatch):
    from app.services import lora_training as lt

    live = tmp_path / 'run'
    archived = tmp_path / 'run_superseded'
    live.mkdir()
    archived.mkdir()
    (live / 'seeded.safetensors').write_bytes(b'seed')
    (archived / 'original.safetensors').write_bytes(b'original')
    status = live / '.lds-state' / 'bridge-status.json'
    status.parent.mkdir()
    status.write_text(
        json.dumps({
            'status': 'restore_error',
            'reasons': ['StateBridgeError: optimizer parameter shape changed'],
        }),
        encoding='utf-8',
    )
    log_path = live / 'training.log'
    log_path.write_text('bridge restore failed', encoding='utf-8')
    captured = {}

    class FailedProcess:
        returncode = 1

        @staticmethod
        def wait():
            return None

    monkeypatch.setattr(
        lt,
        '_crash_payload',
        lambda *_args: {
            'dataset_id': 9,
            'rc': 1,
            'log_tail': 'bridge restore failed',
            'excerpt': {'kind': 'error', 'text': 'bridge restore failed'},
        },
    )
    monkeypatch.setattr(lt, 'process_training_queue', lambda: None)
    monkeypatch.setattr(
        lt.queue_manager,
        '_set_system_state',
        lambda key, value, **_kwargs: captured.update({key: value}),
    )

    lt._watch_training(
        app,
        FailedProcess(),
        str(log_path),
        9,
        {
            'training_folder': str(live),
            'archived': str(archived),
            'status_path': str(status),
        },
    )

    assert (live / 'original.safetensors').read_bytes() == b'original'
    assert not archived.exists()
    failed = list(tmp_path.glob('run_failed_full_state_*'))
    assert len(failed) == 1
    assert (failed[0] / 'seeded.safetensors').read_bytes() == b'seed'
    error = captured['training_error']
    assert error['exact_resume']['rolled_back'] is True
    assert 'optimizer parameter shape changed' in error['exact_resume']['reason']
    assert 'restored the original run' in error['excerpt']['text']


def test_exact_resume_restart_recovers_dead_child_when_watcher_never_started(
        app, tmp_path, monkeypatch):
    """The durable journal replaces the in-memory watcher after Flask exits."""
    from app.services import lora_training as lt

    _configure_aitoolkit(tmp_path, monkeypatch, app)
    with app.app_context():
        output = Path(lt._output_dir())
        output.mkdir(parents=True, exist_ok=True)
        live = output / 'u1_restart'
        archived = output / 'u1_restart_superseded_test'
        live.mkdir()
        archived.mkdir()
        (live / 'seeded.safetensors').write_bytes(b'SEED')
        (archived / 'original.safetensors').write_bytes(b'ORIGINAL')
        journal, _ = lt._new_exact_resume_journal(91, live, archived)
        lt._update_exact_resume_journal(
            journal,
            phase='spawned',
            pid=424242,
            pid_create_time=1234.5,
            run_token='restart-token',
        )
        monkeypatch.setattr(
            lt, '_pid_alive_with_birth', lambda *_args: False)

        assert lt.recover_training_fence() is None

        assert (live / 'original.safetensors').read_bytes() == b'ORIGINAL'
        assert not archived.exists()
        assert not journal.exists()
        failed = list(output.glob('u1_restart_failed_full_state_*'))
        assert len(failed) == 1
        assert (failed[0] / 'seeded.safetensors').read_bytes() == b'SEED'


def test_exact_resume_restart_clears_journal_after_first_boundary_without_rollback(
        app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.training_bridge.lds_aitk_bridge_contract import atomic_json_nofollow

    _configure_aitoolkit(tmp_path, monkeypatch, app)
    with app.app_context():
        output = Path(lt._output_dir())
        output.mkdir(parents=True, exist_ok=True)
        live = output / 'u1_started'
        archived = output / 'u1_started_superseded_test'
        live.mkdir()
        archived.mkdir()
        (live / 'new-work').write_bytes(b'NEW')
        (archived / 'source').write_bytes(b'SOURCE')
        journal, value = lt._new_exact_resume_journal(92, live, archived)
        lt._update_exact_resume_journal(
            journal,
            phase='spawned',
            pid=777,
            pid_create_time=12.0,
            run_token='started-token',
        )
        atomic_json_nofollow(
            value['status_path'],
            {'status': 'training', 'training_started': True},
        )
        monkeypatch.setattr(
            lt, '_pid_alive_with_birth',
            lambda *_args: pytest.fail('started status must win before PID probe'))

        assert lt._reconcile_exact_resume_journals() is False

        assert not journal.exists()
        assert live.is_dir() and archived.is_dir()
        assert (live / 'new-work').read_bytes() == b'NEW'
        assert (archived / 'source').read_bytes() == b'SOURCE'


def test_exact_resume_launching_without_any_pid_holds_fail_closed(
        app, tmp_path, monkeypatch):
    """No PID cannot prove Popen failed; never rename under an unknown child."""
    from app.services import lora_training as lt

    _configure_aitoolkit(tmp_path, monkeypatch, app)
    with app.app_context():
        output = Path(lt._output_dir())
        output.mkdir(parents=True, exist_ok=True)
        live = output / 'u1_launch_gap'
        archived = output / 'u1_launch_gap_superseded_test'
        live.mkdir()
        archived.mkdir()
        (live / 'seed').write_bytes(b'SEED')
        (archived / 'source').write_bytes(b'SOURCE')
        journal, _ = lt._new_exact_resume_journal(93, live, archived)
        lt._update_exact_resume_journal(
            journal,
            phase='launching',
            run_token='gap-token',
        )

        assert lt._reconcile_exact_resume_journals() is True

        assert (live / 'seed').read_bytes() == b'SEED'
        assert (archived / 'source').read_bytes() == b'SOURCE'
        assert journal.exists()
        assert lt.queue_manager._get_system_state(
            'training_in_progress', False) is True


def test_exact_resume_journal_after_256_junk_entries_still_holds_fail_closed(
        app, tmp_path, monkeypatch):
    """Unrelated directory entries cannot hide a later active journal."""
    from app.services import lora_training as lt

    _configure_aitoolkit(tmp_path, monkeypatch, app)
    with app.app_context():
        root = lt._exact_resume_journal_root()
        root.mkdir(parents=True, exist_ok=True)
        for index in range(256):
            (root / f'junk-{index:03d}.tmp').write_text('junk')

        output = Path(lt._output_dir())
        output.mkdir(parents=True, exist_ok=True)
        live = output / 'u1_launch_after_junk'
        archived = output / 'u1_launch_after_junk_superseded_test'
        live.mkdir()
        archived.mkdir()
        (live / 'seed').write_bytes(b'SEED')
        (archived / 'source').write_bytes(b'SOURCE')
        journal, _ = lt._new_exact_resume_journal(94, live, archived)
        lt._update_exact_resume_journal(
            journal,
            phase='launching',
            run_token='after-junk-token',
        )

        assert lt._reconcile_exact_resume_journals() is True

        assert journal.exists()
        assert (live / 'seed').read_bytes() == b'SEED'
        assert (archived / 'source').read_bytes() == b'SOURCE'
        assert lt.queue_manager._get_system_state(
            'training_in_progress', False) is True


def test_exact_resume_path_guard_rejects_linked_parent(
        app, tmp_path, monkeypatch):
    from app.services import lora_training as lt

    _configure_aitoolkit(tmp_path, monkeypatch, app)
    with app.app_context():
        output = Path(lt._output_dir())
        output.mkdir(parents=True, exist_ok=True)
        outside = tmp_path / 'outside-run-root'
        outside.mkdir()
        linked = output / 'linked-parent'
        try:
            os.symlink(outside, linked, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip('directory symlink creation is unavailable')

        assert lt._journal_path_is_unsafe(linked / 'run') is True


# --- Safe-subset settings overrides -------------------------------------------

def test_continue_applies_safe_overrides(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Overrides', 'ovrtrig')
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(lt, 'list_checkpoints',
                            lambda *a, **k: [{'step': 1000, 'filename': 'b.safetensors'}])
        monkeypatch.setattr(lt, 'launch_training', lambda *a, **k: {'started': True})
        _stub_weights_seed(monkeypatch, lt)

        lt.continue_training(LOCAL_USER, ds.id, extra_steps=500,
                             overrides={'save_every': 250, 'sample_every': 500})

        eff = lt.effective_train_settings(svc.get_dataset(LOCAL_USER, ds.id))
    assert eff['save_every'] == 250
    assert eff['sample_every'] == 500


def test_continue_rejects_forbidden_override(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Forbidden', 'forbidtrig')
        launched = []
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(lt, 'list_checkpoints',
                            lambda *a, **k: [{'step': 1000, 'filename': 'b.safetensors'}])
        monkeypatch.setattr(lt, 'launch_training',
                            lambda *a, **k: launched.append(k) or {'started': True})

        # rank changes the LoRA weight shape → the checkpoint could not load.
        with pytest.raises(ValueError, match='cannot change when continuing.*rank'):
            lt.continue_training(LOCAL_USER, ds.id, overrides={'rank': 32})
        # a forbidden key fails BEFORE any launch and without persisting settings.
        assert launched == []
        assert svc.get_dataset(LOCAL_USER, ds.id).train_settings in (None, '{}')


def test_validate_resume_overrides_value_and_key_rules():
    from app.services import lora_training as lt
    assert lt.validate_resume_overrides(None) == {}
    assert lt.validate_resume_overrides({'save_every': 500}) == {'save_every': 500}
    # multi-line prompt string is normalized to a trimmed list
    assert lt.validate_resume_overrides(
        {'sample_prompts': 'a\n \nb'}) == {'sample_prompts': ['a', 'b']}
    with pytest.raises(ValueError, match='save_every must be one of'):
        lt.validate_resume_overrides({'save_every': 7})
    with pytest.raises(ValueError, match='cannot change when continuing.*optimizer'):
        lt.validate_resume_overrides({'optimizer': 'prodigy'})
    # timestep_type is the deliberate safe exception (two-phase texture recipe):
    # honored when it names a real weighting, refused on anything else.
    assert lt.validate_resume_overrides(
        {'timestep_type': 'shift'}) == {'timestep_type': 'shift'}
    with pytest.raises(ValueError, match='timestep_type must be one of'):
        lt.validate_resume_overrides({'timestep_type': 'lowest-noise'})


# --- Cloud continue: same knobs, seeding an arbitrary checkpoint onto a pod ----

@pytest.fixture()
def ct(app, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    from app.services import cloud_training
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
    monkeypatch.setattr(cloud_training, '_reconcile_before_launch', lambda a: None)
    return cloud_training


@pytest.fixture()
def seeded_dataset(app, client):
    return client.post('/api/dataset/create',
                       json={'name': 'Lola', 'trigger_word': 'lola'}).get_json()['id']


def _seed_done_run(ct, dataset_id, staging, steps=1000, **params):
    """A 'done' cloud run whose staging holds two harvested checkpoints (500, 1000)."""
    p = {'steps': steps, 'variant': 'turbo', 'train_type': 'zimage', 'masked': True}
    p.update(params)
    run = ct.CloudTrainingRun(
        dataset_id=dataset_id, status='done', job_name='lds1_x',
        vast_label='lds-1', staging_dir=str(staging), train_params=json.dumps(p))
    ct.db.session.add(run)
    ct.db.session.commit()
    (staging / 'lds1_x_000000500.safetensors').write_bytes(b'w500')
    (staging / 'lds1_x_000001000.safetensors').write_bytes(b'w1000')
    return run


def test_cloud_continue_from_lower_checkpoint_selects_it_and_keeps_staging(
        ct, app, seeded_dataset, monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging)
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(dataset_id=dataset_id, **kw), {'ok': True})[1])
        res = ct.continue_cloud_run('local', src.id, extra_steps=300, from_step=500)

    assert res['resumed_from'] == 500 and res['target_steps'] == 800
    assert captured['steps'] == 800
    assert captured['resume_step'] == 500
    assert captured['resume_ckpt_path'] == str(staging / 'lds1_x_000000500.safetensors')
    # the source run's staging is read-only here — nothing moved or deleted
    assert (staging / 'lds1_x_000000500.safetensors').exists()
    assert (staging / 'lds1_x_000001000.safetensors').exists()


def test_cloud_continue_default_is_latest_checkpoint(ct, app, seeded_dataset,
                                                     monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging)
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(**kw), {'ok': True})[1])
        res = ct.continue_cloud_run('local', src.id, extra_steps=500)
    assert res['resumed_from'] == 1000 and captured['resume_step'] == 1000


def test_cloud_continue_merges_safe_overrides_into_snapshot(ct, app, seeded_dataset,
                                                            monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging,
                             train_settings_snapshot=json.dumps({'rank': 32}))
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(**kw), {'ok': True})[1])
        ct.continue_cloud_run('local', src.id, extra_steps=500,
                              overrides={'sample_every': 250})
    merged = json.loads(captured['train_settings_snapshot'])
    assert merged['sample_every'] == 250          # override folded in
    assert merged['rank'] == 32                    # run's own settings preserved


def test_cloud_continue_rejects_forbidden_override(ct, app, seeded_dataset,
                                                   monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging)
        launched = []
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda *a, **k: launched.append(k))
        with pytest.raises(ValueError, match='cannot change when continuing.*alpha'):
            ct.continue_cloud_run('local', src.id, overrides={'alpha': 8})
        assert launched == []


def test_cloud_full_state_is_refused_before_any_launch(
        ct, app, seeded_dataset, monkeypatch, tmp_path):
    """Today's pod cannot activate the LDS bridge, so the API must never turn a
    full-state request into a silent weights-only launch."""
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging)
        launched = []
        monkeypatch.setattr(
            ct, 'launch_cloud_training',
            lambda *args, **kwargs: launched.append((args, kwargs)))

        with pytest.raises(ValueError, match='full-state resume is not supported'):
            ct.continue_cloud_run(
                'local',
                src.id,
                resume_mode='full_state',
                state_bundle_id='0123456789abcdef0123456789abcdef',
            )

    assert launched == []


def test_cloud_continue_from_missing_step_is_rejected(ct, app, seeded_dataset,
                                                      monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging)
        with pytest.raises(ValueError, match='no harvested checkpoint at step 700'):
            ct.continue_cloud_run('local', src.id, from_step=700)


# --- LR factor: scale the continuation's learning rate (½ polish / ⅒ finish) -----

def test_resolve_resume_lr_rules():
    from app.services import lora_training as lt
    assert lt.resolve_resume_lr({}, None) is None            # keep current
    assert lt.resolve_resume_lr({}, 1) is None
    # factors scale the family-fixed 1e-4 default of a non-adaptive run
    assert lt.resolve_resume_lr({}, 0.5) == pytest.approx(5e-5)
    assert lt.resolve_resume_lr({}, 0.1) == pytest.approx(1e-5)
    # an explicit stored learning_rate is the base a further continue scales
    assert lt.resolve_resume_lr({'learning_rate': 5e-5}, 0.5) == pytest.approx(2.5e-5)
    with pytest.raises(ValueError, match='lr_factor must be one of'):
        lt.resolve_resume_lr({}, 0.25)
    # Prodigy adapts the LR itself → the factor is refused, not silently swallowed
    with pytest.raises(ValueError, match='Prodigy'):
        lt.resolve_resume_lr({'optimizer': 'prodigy'}, 0.5)


def test_validate_resume_overrides_lr_factor():
    from app.services import lora_training as lt
    assert lt.validate_resume_overrides({'lr_factor': 0.5}) == {'lr_factor': 0.5}
    assert lt.validate_resume_overrides({'lr_factor': 0.1}) == {'lr_factor': 0.1}
    # keep-current (1) is a no-op: dropped so nothing redundant persists
    assert lt.validate_resume_overrides({'lr_factor': 1}) == {}
    with pytest.raises(ValueError, match='lr_factor must be one of'):
        lt.validate_resume_overrides({'lr_factor': 0.75})


def test_continue_lr_factor_half_reduces_effective_lr(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'LR half', 'lrhalf')
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(lt, 'list_checkpoints',
                            lambda *a, **k: [{'step': 1000, 'filename': 'b.safetensors'}])
        monkeypatch.setattr(lt, 'launch_training', lambda *a, **k: {'started': True})
        _stub_weights_seed(monkeypatch, lt)

        lt.continue_training(LOCAL_USER, ds.id, extra_steps=500,
                             overrides={'lr_factor': 0.5})

        eff = lt.effective_train_settings(svc.get_dataset(LOCAL_USER, ds.id))
    # 1e-4 run → 5e-5, exposed as the effective LR the Continue dialog reads back.
    assert eff['learning_rate'] == pytest.approx(5e-5)


def test_continue_keep_current_lr_leaves_default(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'LR keep', 'lrkeep')
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(lt, 'list_checkpoints',
                            lambda *a, **k: [{'step': 1000, 'filename': 'b.safetensors'}])
        monkeypatch.setattr(lt, 'launch_training', lambda *a, **k: {'started': True})
        _stub_weights_seed(monkeypatch, lt)

        lt.continue_training(LOCAL_USER, ds.id, extra_steps=500)   # no lr_factor

        after = svc.get_dataset(LOCAL_USER, ds.id)
        assert lt.effective_train_settings(after)['learning_rate'] == pytest.approx(1e-4)
        # no learning_rate persisted → the dataset stays byte-identical to a default
        assert 'learning_rate' not in (lt._train_settings(after))


def test_continue_lr_factor_emitted_in_job_config(app, tmp_path, monkeypatch):
    """The reduced LR reaches the ACTUAL ai-toolkit job config (and thus the run's
    provenance snapshot), not just the stored settings."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds, run_dir, trig = _seed_run(lt, svc, LOCAL_USER, 'LR cfg', 'lrcfg', [1000])
        res = lt.continue_training(LOCAL_USER, ds.id, extra_steps=200,
                                   overrides={'lr_factor': 0.1})
        with open(res['config_path'], encoding='utf-8') as fh:
            cfg = json.load(fh)
        assert cfg['config']['process'][0]['train']['lr'] == pytest.approx(1e-5)
        # provenance/⎘ Share-config snapshot records the effective LR too
        assert lt.launch_settings_snapshot(
            svc.get_dataset(LOCAL_USER, ds.id))['lr'] == pytest.approx(1e-5)


def test_continue_lr_factor_refused_on_prodigy_no_side_effect(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'LR prodigy', 'lrprodigy')
        ds.train_settings = json.dumps({'optimizer': 'prodigy'})
        svc.db.session.commit()
        launched = []
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(lt, 'list_checkpoints',
                            lambda *a, **k: [{'step': 1000, 'filename': 'b.safetensors'}])
        monkeypatch.setattr(lt, 'launch_training',
                            lambda *a, **k: launched.append(k) or {'started': True})

        with pytest.raises(ValueError, match='Prodigy'):
            lt.continue_training(LOCAL_USER, ds.id, overrides={'lr_factor': 0.5})
        # refused BEFORE launch, and no learning_rate leaked into the settings
        assert launched == []
        assert 'learning_rate' not in lt._train_settings(
            svc.get_dataset(LOCAL_USER, ds.id))


def test_cloud_continue_lr_factor_folds_learning_rate_into_snapshot(
        ct, app, seeded_dataset, monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging)
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(**kw), {'ok': True})[1])
        ct.continue_cloud_run('local', src.id, extra_steps=500,
                              overrides={'lr_factor': 0.5})
    merged = json.loads(captured['train_settings_snapshot'])
    # a default (1e-4) cloud run continues at 5e-5, carried in the per-run snapshot
    assert merged['learning_rate'] == pytest.approx(5e-5)


def test_cloud_continue_lr_factor_refused_on_prodigy(ct, app, seeded_dataset,
                                                     monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging,
                             train_settings_snapshot=json.dumps({'optimizer': 'prodigy'}))
        launched = []
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda *a, **k: launched.append(k))
        with pytest.raises(ValueError, match='Prodigy'):
            ct.continue_cloud_run('local', src.id, overrides={'lr_factor': 0.5})
        assert launched == []


# --- Lane choice: continue a LOCAL run's checkpoint IN THE CLOUD ---------------
# The mirror of continue_cloud_run. Same pod-side seam (resume_ckpt_path on a
# fresh pod); the file comes from the ai-toolkit run dir instead of a cloud run's
# staging. launch_cloud_training is stubbed — no pod is ever rented in tests.

def _local_run_for_cloud(app, tmp_path, monkeypatch, steps=(500, 1000)):
    """A real local run dir with saves — the source a cloud continuation seeds from."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, monkeypatch, app)
    return _seed_run(lt, svc, LOCAL_USER, 'CloudFromLocal', 'cloudfromlocal', steps)


def test_local_checkpoint_can_be_continued_in_the_cloud(ct, app, monkeypatch, tmp_path):
    from app.config import LOCAL_USER
    with app.app_context():
        ds, run_dir, trig = _local_run_for_cloud(app, tmp_path, monkeypatch)
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(dataset_id=dataset_id, **kw), {'ok': True})[1])
        res = ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, extra_steps=200,
                                             from_step=500)
    assert res['resumed_from'] == 500 and res['target_steps'] == 700
    assert captured['steps'] == 700 and captured['resume_step'] == 500
    # the LOCAL file is what gets seeded onto the fresh pod
    assert captured['resume_ckpt_path'] == os.path.join(
        run_dir, f'lora_{trig}_000000500.safetensors')
    # and unlike the local lane, nothing on disk is archived or re-seeded
    assert sorted(os.listdir(run_dir)) == [f'lora_{trig}_000000500.safetensors',
                                           f'lora_{trig}_000001000.safetensors']


def test_local_to_cloud_continue_defaults_to_the_latest_save(ct, app, monkeypatch,
                                                             tmp_path):
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _run_dir, _trig = _local_run_for_cloud(app, tmp_path, monkeypatch)
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(**kw), {'ok': True})[1])
        res = ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, extra_steps=500)
    assert res['resumed_from'] == 1000 and captured['resume_step'] == 1000


def test_local_to_cloud_continue_rejects_a_step_that_is_not_a_save(ct, app,
                                                                  monkeypatch, tmp_path):
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _run_dir, _trig = _local_run_for_cloud(app, tmp_path, monkeypatch)
        launched = []
        monkeypatch.setattr(ct, 'launch_cloud_training', lambda *a, **k: launched.append(k))
        with pytest.raises(ValueError, match='no local checkpoint at step 777'):
            ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, from_step=777)
        assert launched == []


def test_local_to_cloud_continue_without_any_local_save_is_refused(ct, app,
                                                                   monkeypatch):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Empty lane', 'emptylane')
        launched = []
        monkeypatch.setattr(ct, 'launch_cloud_training', lambda *a, **k: launched.append(k))
        with pytest.raises(ValueError, match='no local checkpoint to continue from'):
            ct.continue_local_run_in_cloud(LOCAL_USER, ds.id)
        assert launched == []


def test_local_to_cloud_continue_refuses_a_forbidden_override(ct, app, monkeypatch,
                                                              tmp_path):
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _run_dir, _trig = _local_run_for_cloud(app, tmp_path, monkeypatch)
        launched = []
        monkeypatch.setattr(ct, 'launch_cloud_training', lambda *a, **k: launched.append(k))
        with pytest.raises(ValueError, match='cannot change when continuing.*alpha'):
            ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, overrides={'alpha': 8})
        assert launched == []


def test_local_to_cloud_continue_keeps_overrides_out_of_the_dataset(ct, app,
                                                                    monkeypatch, tmp_path):
    """A cloud launch freezes its settings in a per-run snapshot — continuing a
    local run in the cloud must NOT persist the tweak on the dataset (that is a
    local-lane behaviour, and it would silently change the next local run)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _run_dir, _trig = _local_run_for_cloud(app, tmp_path, monkeypatch)
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(**kw), {'ok': True})[1])
        ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, extra_steps=500,
                                       overrides={'sample_every': 250,
                                                  'lr_factor': 0.5})
        persisted = lt._train_settings(svc.get_dataset(LOCAL_USER, ds.id))
    merged = json.loads(captured['train_settings_snapshot'])
    assert merged['sample_every'] == 250
    # a default (1e-4) run continues at 5e-5, carried in the run snapshot only
    assert merged['learning_rate'] == pytest.approx(5e-5)
    assert 'sample_every' not in persisted and 'learning_rate' not in persisted
