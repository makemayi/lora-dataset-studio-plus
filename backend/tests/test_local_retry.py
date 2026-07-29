"""↻ Retry of a FAILED LOCAL training run (Runs page).

Local runs carry no status column: their launch is recorded once in the
provenance registry (TrainingRunRecord, source='local'), and the only signal
that one crashed is the transient global `training_error` the watcher writes on
rc≠0 (cleared on the next launch, TTL-capped). Local training is single-flight,
so at most one local run is "failed" at a time.

`retry_local_run` mirrors the cloud ↻ Retry: a REAL launch_training replaying the
identity params stamped for that launch (family/variant/base/masked/steps) with
every normal guardrail — GPU-collision refusal, normal preflight, no bypass —
and the live dataset (slider settings included) as the source of truth.
"""
import json

import pytest

from app.config import LOCAL_USER


def _configure_aitoolkit(tmp_path, monkeypatch, app):
    """Fake ai-toolkit install: venv python + run.py present, dir configured."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid

    def wait(self):
        return None


def _mk_ds(app, n_keep=6, trigger='rt_trig', name='Retry', slider=False):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.services import lora_training as lt
    ds = svc.create_dataset(LOCAL_USER, name, trigger, train_type='zimage')
    for i in range(n_keep):
        svc.db.session.add(FaceDatasetImage(
            dataset_id=ds.id, filename=f'k{i}.webp', status='keep',
            framing='face', caption=f'a nice varied caption number {i}'))
    svc.db.session.commit()
    if slider:
        lt.update_slider_settings(LOCAL_USER, ds.id, {
            'enabled': True, 'positive': 'very muscular body',
            'negative': 'skinny frail body', 'target_class': 'person'})
    return ds


def _register(app, ds, **kw):
    from app.services import checkpoint_registry as cr
    defaults = dict(family='zimage', source='local', base_model='', variant='turbo',
                    masked=True, steps=1000)
    defaults.update(kw)
    return cr.register_launch(LOCAL_USER, ds.id, **defaults)


def _mark_failed(ds, rc=1, log_tail='boom\nRuntimeError: bad allocation'):
    from app.job_queue import queue_manager
    queue_manager._set_system_state('training_in_progress', False, ttl_seconds=60)
    queue_manager._set_system_state(
        'training_error', {'dataset_id': ds.id, 'rc': rc, 'log_tail': log_tail},
        ttl_seconds=60)


# --- failed-run detection --------------------------------------------------------

def test_failed_local_run_detection(app):
    from app.services import lora_training as lt
    from app.job_queue import queue_manager
    with app.app_context():
        ds = _mk_ds(app)
        rec = _register(app, ds)
        queue_manager._set_system_state('training_in_progress', False, ttl_seconds=60)
        assert lt.failed_local_run() is None          # no crash recorded yet
        _mark_failed(ds)
        fid, msg = lt.failed_local_run()
        assert fid == rec.id
        assert 'bad allocation' in msg and 'exit code 1' in msg
        # a live run hides the affordance (its own launch cleared the error state)
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=60)
        assert lt.failed_local_run() is None


# --- Runs hub wiring -------------------------------------------------------------

def test_all_runs_marks_failed_local_run_only(app):
    """The Runs hub tags the failed local run with status='error' + its record_id
    (the ↻ Retry target) + a crash message; an earlier completed local launch of
    the same dataset stays unmarked, so no Retry button shows on it."""
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _mk_ds(app, trigger='ar_trig', name='AllRuns')
        older = _register(app, ds, steps=1000)     # earlier launch (completed)
        newest = _register(app, ds, steps=1500)    # the launch that crashed
        _mark_failed(ds, rc=1, log_tail='RuntimeError: bad allocation')
        out = ct.all_runs(limit=10)
        local_rows = {r['record_id']: r for r in out['recent'] if r['source'] == 'local'}
        assert local_rows[newest.id]['status'] == 'error'
        assert 'bad allocation' in local_rows[newest.id]['error']
        assert local_rows[older.id].get('status') is None


# --- param replay ----------------------------------------------------------------

def test_retry_local_run_replays_stamped_params(app, monkeypatch):
    """The retry hands launch_training the EXACT identity params stamped for the
    failed launch — family/variant/base/masked/steps."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk_ds(app)
        rec = _register(app, ds, family='krea', base_model='merged/base',
                        variant='turbo', masked=False, steps=1500)
        _mark_failed(ds, rc=3221225477, log_tail='std::bad_alloc')
        captured = {}
        monkeypatch.setattr(
            lt, 'launch_training',
            lambda user_id, dataset_id, **kw: captured.update(
                user_id=user_id, dataset_id=dataset_id, **kw) or {'started': True})
        out = lt.retry_local_run(LOCAL_USER, rec.id)
        assert out == {'started': True}
        assert captured['dataset_id'] == ds.id
        assert captured['train_type'] == 'krea'
        assert captured['variant'] == 'turbo'
        assert captured['base_model'] == 'merged/base'
        assert captured['masked'] is False
        assert captured['steps'] == 1500


def test_retry_local_run_maps_official_base_to_none(app, monkeypatch):
    """A stamped empty base_model is the OFFICIAL base — replayed as None, never
    an empty string that a downstream would treat as a custom path."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk_ds(app)
        rec = _register(app, ds, base_model='')
        _mark_failed(ds)
        captured = {}
        monkeypatch.setattr(
            lt, 'launch_training',
            lambda user_id, dataset_id, **kw: captured.update(**kw) or {'started': True})
        lt.retry_local_run(LOCAL_USER, rec.id)
        assert captured['base_model'] is None


# --- guards ----------------------------------------------------------------------

def test_retry_local_run_refused_when_run_in_progress(app, monkeypatch):
    """A live run yields the exact GPU-collision message (before any preflight),
    and launch_training is never reached."""
    from app.services import lora_training as lt
    from app.job_queue import queue_manager
    with app.app_context():
        ds = _mk_ds(app)
        rec = _register(app, ds)
        _mark_failed(ds)
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=60)
        queue_manager._set_system_state('training_pid', 4242, ttl_seconds=60)
        monkeypatch.setattr(lt, '_pid_alive', lambda pid: True)
        monkeypatch.setattr(lt, 'launch_training',
                            lambda *a, **k: pytest.fail('must not launch during a live run'))
        with pytest.raises(ValueError, match='already in progress'):
            lt.retry_local_run(LOCAL_USER, rec.id)


def test_retry_local_run_refused_when_not_failed(app, monkeypatch):
    """A completed/never-failed run has no recorded failure — refused cleanly
    (this is the backend mirror of the button only showing on error rows)."""
    from app.services import lora_training as lt
    from app.job_queue import queue_manager
    with app.app_context():
        ds = _mk_ds(app)
        rec = _register(app, ds)
        queue_manager._set_system_state('training_in_progress', False, ttl_seconds=60)
        monkeypatch.setattr(lt, 'launch_training',
                            lambda *a, **k: pytest.fail('must not launch a non-failed run'))
        with pytest.raises(ValueError, match='no recorded failure'):
            lt.retry_local_run(LOCAL_USER, rec.id)


def test_retry_local_run_rejects_unknown_and_cloud_records(app):
    from app.services import lora_training as lt
    with app.app_context():
        with pytest.raises(ValueError, match='unknown training run'):
            lt.retry_local_run(LOCAL_USER, 999999)
        ds = _mk_ds(app)
        crec = _register(app, ds, source='cloud', cloud_run_id=1)
        with pytest.raises(ValueError, match='local run can be retried'):
            lt.retry_local_run(LOCAL_USER, crec.id)


# --- slider replay: end to end (config actually written) -------------------------

def test_retry_local_run_replays_slider_config_with_768(app, tmp_path, monkeypatch):
    """Retrying a failed SLIDER run re-emits the slider recipe faithfully — the
    concept_slider process — and, thanks to the new default, at 768 only (the
    resolution that keeps the slider loss's VRAM peak under 24 GB). This is the
    exact regression reported: the first slider run OOM'd in 768+1024."""
    from app.services import lora_training as lt
    _configure_aitoolkit(tmp_path, monkeypatch, app)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    monkeypatch.setattr(lt, '_aitoolkit_supports_concept_slider', lambda: True)
    monkeypatch.setattr(lt, '_watch_training', lambda *a, **k: None)
    monkeypatch.setattr(lt.subprocess, 'Popen', lambda args, **kw: _FakeProc())

    def fake_export(user_id, dataset_id, masked=True):
        folder = tmp_path / 'exp'
        folder.mkdir(exist_ok=True)
        return str(folder)

    monkeypatch.setattr(lt, 'export_dataset_to_aitoolkit', fake_export)
    with app.app_context():
        ds = _mk_ds(app, slider=True)
        rec = _register(app, ds, family='zimage', variant='turbo', steps=1000)
        _mark_failed(ds, log_tail='RuntimeError: bad allocation')
        out = lt.retry_local_run(LOCAL_USER, rec.id)
        with open(out['config_path'], encoding='utf-8') as fh:
            cfg = json.load(fh)
        proc = cfg['config']['process'][0]
        assert proc['type'] == 'concept_slider'
        assert proc['datasets'][0]['resolution'] == [768]


# --- route surface ---------------------------------------------------------------

def test_retry_route_posts_record_id(app, client, monkeypatch):
    from app import capabilities
    from app.services import lora_training as lt
    monkeypatch.setattr(capabilities, 'probe', lambda: {'aitoolkit': {'valid': True}})
    captured = {}
    monkeypatch.setattr(
        lt, 'retry_local_run',
        lambda user_id, record_id, **kw: captured.update(record_id=record_id, **kw)
        or {'started': True, 'pid': 7})
    r = client.post('/api/dataset/train/retry', json={'record_id': 55})
    assert r.status_code == 200
    d = r.get_json()
    assert d['ok'] is True and d['pid'] == 7
    assert captured['record_id'] == 55
    # No body key = no confirmation. Every flag is explicitly False, never absent:
    # "not sent" and "declined" must reach the service as the same thing.
    assert all(captured[k] is False for k in lt.CONFIRMATION_FLAGS)


def test_retry_route_gated_when_aitoolkit_unconfigured(app, client, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe', lambda: {'aitoolkit': {'valid': False}})
    r = client.post('/api/dataset/train/retry', json={'record_id': 1})
    assert r.status_code == 409


# --- confirmable pre-flight guards on the retry lane (GitHub #23, 1Tomber) --------
#
# Reported: a run started with one uncaptioned image (confirmed "train anyway")
# failed before training on a missing Hugging Face token. After adding the token,
# ↻ Retry did nothing at all — the route never read allow_uncaptioned, so the
# guard refused again, and the 400 died as an uncaught promise in the console.
#
# A retry is a LAUNCH: it re-exports the LIVE dataset, so it meets the live
# dataset's guards. The consent therefore comes from the retry itself, not from
# the record of the failed launch — nothing stores it there, and a consent given
# for "1 image has no caption" must not silently cover the twelve that lost their
# caption since. The two halves below are the whole contract: the guard still
# refuses an unconfirmed retry, and a confirmed one goes through.

def _uncaption_one(ds):
    """Take the caption off one kept image — 1Tomber's dataset, exactly."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    img = (FaceDatasetImage.query.filter_by(dataset_id=ds.id, status='keep')
           .order_by(FaceDatasetImage.id.asc()).first())
    img.caption = '   '                      # whitespace counts as no caption
    svc.db.session.commit()


def _launchable(lt, tmp_path, monkeypatch):
    """Everything a real launch needs EXCEPT the guards under test."""
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    monkeypatch.setattr(lt, '_watch_training', lambda *a, **k: None)
    monkeypatch.setattr(lt.subprocess, 'Popen', lambda args, **kw: _FakeProc())
    folder = tmp_path / 'exp'
    folder.mkdir(exist_ok=True)
    monkeypatch.setattr(lt, 'export_dataset_to_aitoolkit',
                        lambda user_id, dataset_id, masked=True: str(folder))


def _retry_ready(app, client, tmp_path, monkeypatch):
    from app import capabilities
    from app.services import lora_training as lt
    monkeypatch.setattr(capabilities, 'probe', lambda: {'aitoolkit': {'valid': True}})
    _configure_aitoolkit(tmp_path, monkeypatch, app)
    _launchable(lt, tmp_path, monkeypatch)


def test_retry_of_uncaptioned_run_is_refused_then_confirmable(app, client, tmp_path,
                                                              monkeypatch):
    """THE reported scenario. Unconfirmed → the same UNCAPTIONED: refusal Start
    gives (never masked); confirmed in the retry payload → the run starts."""
    _retry_ready(app, client, tmp_path, monkeypatch)
    with app.app_context():
        ds = _mk_ds(app, n_keep=20, trigger='u_trig', name='Uncaptioned')
        _uncaption_one(ds)
        rec = _register(app, ds)
        _mark_failed(ds)

        plain = client.post('/api/dataset/train/retry', json={'record_id': rec.id})
        assert plain.status_code == 400
        assert plain.get_json()['error'].startswith('UNCAPTIONED: ')
        assert '1 kept image(s)' in plain.get_json()['error']

        confirmed = client.post('/api/dataset/train/retry',
                                json={'record_id': rec.id, 'allow_uncaptioned': True})
        assert confirmed.status_code == 200, confirmed.get_json()
        assert confirmed.get_json()['started'] is True


def test_retry_still_refuses_a_run_that_was_never_confirmed(app, client, tmp_path,
                                                            monkeypatch):
    """Anti-masking. The fix must not be "drop the guard on retry": with no
    confirmation in the payload, EVERY guard still refuses — including on a run
    that is being retried for the tenth time."""
    _retry_ready(app, client, tmp_path, monkeypatch)
    with app.app_context():
        ds = _mk_ds(app, n_keep=20, trigger='n_trig', name='NeverConfirmed')
        _uncaption_one(ds)
        rec = _register(app, ds)
        _mark_failed(ds)
        for _ in range(3):
            r = client.post('/api/dataset/train/retry', json={'record_id': rec.id})
            assert r.status_code == 400
            assert r.get_json()['error'].startswith('UNCAPTIONED: ')
        # …and confirming something ELSE does not open the caption gate
        r = client.post('/api/dataset/train/retry',
                        json={'record_id': rec.id, 'allow_not_ready': True})
        assert r.status_code == 400
        assert r.get_json()['error'].startswith('UNCAPTIONED: ')


def test_retry_consent_is_never_inherited_from_the_failed_launch(app, monkeypatch):
    """The arbitration, pinned. `retry_local_run` reads its confirmations from
    its OWN call, never from the record: the record stores none, and the dataset
    it replays is the mutable live one. A stale consent relaunching in silence
    would be the reported defect in reverse."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk_ds(app, trigger='i_trig', name='Inherit')
        rec = _register(app, ds)
        _mark_failed(ds)
        captured = {}
        monkeypatch.setattr(
            lt, 'launch_training',
            lambda user_id, dataset_id, **kw: captured.update(**kw) or {'started': True})
        lt.retry_local_run(LOCAL_USER, rec.id)
        assert all(captured[k] is False for k in lt.CONFIRMATION_FLAGS)


def test_retry_forwards_every_confirmation_flag(app, monkeypatch):
    """The whole family, not just the one that was reported: Start can confirm
    five pre-flight refusals, so Retry forwards five."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk_ds(app, trigger='f_trig', name='Flags')
        rec = _register(app, ds)
        _mark_failed(ds)
        captured = {}
        monkeypatch.setattr(
            lt, 'launch_training',
            lambda user_id, dataset_id, **kw: captured.update(**kw) or {'started': True})
        lt.retry_local_run(LOCAL_USER, rec.id,
                           **{k: True for k in lt.CONFIRMATION_FLAGS})
        assert all(captured[k] is True for k in lt.CONFIRMATION_FLAGS)
        assert set(lt.CONFIRMATION_FLAGS) == {
            'allow_caption_mismatch', 'allow_uncaptioned', 'allow_caption_quality',
            'allow_unverified_weights', 'allow_not_ready'}


def test_retry_rejects_an_unknown_confirmation_flag(app):
    """A misspelled flag must not read as "the user did not confirm" — that is
    how a bypass silently stops bypassing."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk_ds(app, trigger='x_trig', name='Typo')
        rec = _register(app, ds)
        _mark_failed(ds)
        with pytest.raises(ValueError, match='unknown confirmation flag'):
            lt.retry_local_run(LOCAL_USER, rec.id, allow_uncaption=True)


@pytest.mark.parametrize('n_keep,marker,flag', [
    (3, 'NOT_READY: ', 'allow_not_ready'),
    (20, 'MISMATCH_CAPTION: ', 'allow_caption_mismatch'),
])
def test_retry_confirms_the_other_preflight_guards(app, client, tmp_path, monkeypatch,
                                                   n_keep, marker, flag):
    """Measured end to end, not assumed: the image floor and the caption-style
    mismatch dead-ended ↻ Retry exactly like the uncaptioned guard did."""
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    _retry_ready(app, client, tmp_path, monkeypatch)
    with app.app_context():
        ds = _mk_ds(app, n_keep=n_keep, trigger=f'g_{flag[6:12]}', name='Guard')
        if marker.startswith('MISMATCH'):
            for img in FaceDatasetImage.query.filter_by(dataset_id=ds.id,
                                                        status='keep').all():
                img.caption = '1girl, solo, long hair, smile, outdoors'
            svc.db.session.commit()
        rec = _register(app, ds)
        _mark_failed(ds)
        plain = client.post('/api/dataset/train/retry', json={'record_id': rec.id})
        assert plain.status_code == 400
        assert plain.get_json()['error'].startswith(marker)
        ok = client.post('/api/dataset/train/retry',
                         json={'record_id': rec.id, flag: True})
        assert ok.status_code == 200, ok.get_json()
