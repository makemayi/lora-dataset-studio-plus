"""Delivering a full model to THIS computer, and what it is allowed to cost.

The contract under test is an order, not a feature: harvest, prove, and only
then let the pod go. Everything after the proof — the Hugging Face backup, the
pod cleanup — is allowed to fail without costing the run, and everything before
it is allowed to fail without costing the pod.
"""
import json
import struct

import pytest

from app.services import dense_local_delivery as dld
from app.services.aitoolkit_remote import TransferCancelled

GB = 1000 ** 3


def _safetensors(payload=b'\x01\x02\x03\x04\x05\x06\x07\x08') -> bytes:
    """A real (tiny) safetensors container: 8-byte length, JSON header, data."""
    header = json.dumps({
        'w': {'dtype': 'BF16', 'shape': [2, 2],
              'data_offsets': [0, len(payload)]},
        '__metadata__': {'format': 'pt'},
    }).encode('utf-8')
    return struct.pack('<Q', len(header)) + header + payload


class _FakeRemote:
    """The pod's file seam: list what it has, stream it back."""

    def __init__(self, blobs, truncate=None, cancel_after=0):
        self.blobs = dict(blobs)
        self.truncate = truncate or set()
        self.cancel_after = cancel_after
        self.downloads = []
        self.settings = {'DATASETS_FOLDER': '/datasets',
                         'TRAINING_FOLDER': '/output'}

    def get_settings(self):
        return dict(self.settings)

    def list_files(self, job_id):
        return [{'path': path, 'size': len(data)}
                for path, data in sorted(self.blobs.items())]

    def download_public_file(self, remote_path, dest_path, timeout=None,
                             expected_size=None, attempts=3, on_progress=None,
                             resume=False, should_cancel=None):
        self.downloads.append({'path': remote_path, 'resume': resume,
                               'attempts': attempts,
                               'cancellable': should_cancel is not None})
        if should_cancel is not None and should_cancel():
            raise TransferCancelled(f'download of {remote_path} cancelled')
        data = self.blobs[remote_path]
        if remote_path in self.truncate:
            data = data[:len(data) // 2]
        with open(dest_path, 'wb') as fh:
            fh.write(data)
        if on_progress:
            on_progress(len(data), expected_size or len(data))


@pytest.fixture()
def ct(app, monkeypatch):
    from app.services import cloud_training, storage_locations
    monkeypatch.setattr(storage_locations, 'free_space',
                        lambda path: {'free_bytes': 4 * 1000 ** 4,
                                      'total_bytes': 8 * 1000 ** 4})
    return cloud_training


def _dense_run(ct, dataset_id, tmp_path, delivery='both', **params):
    staging = tmp_path / 'staging'
    staging.mkdir(exist_ok=True)
    run = ct.CloudTrainingRun(
        dataset_id=dataset_id, status='training', run_name='dense',
        job_name='Krea_lds1_dense', remote_job_id='job-1',
        vast_instance_id='pod-1', base_url='http://pod.example',
        staging_dir=str(staging),
        train_params=json.dumps({
            'training_mode': 'full_transformer', 'train_type': 'krea',
            'variant': 'base', 'dense_delivery': delivery, 'steps': 3000,
            **params}))
    ct.db.session.add(run)
    ct.db.session.commit()
    return run


@pytest.fixture()
def dataset_id(client):
    return client.post('/api/dataset/create', json={
        'name': 'Dense delivery', 'trigger_word': 'person',
    }).get_json()['id']


# --- the modes ------------------------------------------------------------------

def test_an_unstamped_run_keeps_its_hugging_face_only_meaning():
    """Every run that exists today was delivered to the Hub and nothing else.
    Reading a missing stamp as anything but 'hub' would rewrite their history."""
    assert dld.run_mode({}) == 'hub'
    assert dld.run_mode({'dense_delivery': None}) == 'hub'
    assert dld.run_mode({'dense_delivery': 'local'}) == 'local'
    assert dld.run_mode({'dense_delivery': 'both'}) == 'both'
    # A typo in config.json must not silently disable a delivery.
    assert dld.normalize_mode('nonsense') == dld.DEFAULT_MODE
    assert dld.normalize_mode('disk') == 'local'
    assert dld.delivers_local('both') and dld.delivers_hub('both')
    assert dld.delivers_local('local') and not dld.delivers_hub('local')


def test_the_default_delivery_brings_the_model_home_and_keeps_it_resumable(app):
    with app.app_context():
        assert dld.configured_mode() == 'both'


# --- the disk ------------------------------------------------------------------

def test_the_disk_refusal_says_how_short_it_is_and_can_be_overridden(
        app, monkeypatch):
    from app.services import storage_locations
    monkeypatch.setattr(storage_locations, 'free_space',
                        lambda path: {'free_bytes': 12 * GB,
                                      'total_bytes': 500 * GB})
    with app.app_context():
        forecast = dld.local_delivery_forecast(keeps=1)
        assert forecast['fits'] is False
        assert forecast['shortfall_bytes'] > 0
        message = dld.disk_refusal_message(forecast)
        assert message.startswith('LOCAL_DISK_FULL: ')
        assert 'free' in message and 'Settings' in message
        with pytest.raises(ValueError, match='LOCAL_DISK_FULL'):
            dld.assert_local_disk_headroom(_forecast=forecast)
        # The estimate is generous by construction, so the user keeps the last
        # word — exactly like the Hugging Face storage refusal.
        assert dld.assert_local_disk_headroom(
            _forecast=forecast, allow_override=True) is forecast


def test_an_unmeasurable_volume_never_blocks(app, monkeypatch):
    from app.services import storage_locations
    monkeypatch.setattr(storage_locations, 'free_space', lambda path: None)
    with app.app_context():
        forecast = dld.local_delivery_forecast()
        assert forecast['fits'] is None
        assert dld.assert_local_disk_headroom(_forecast=forecast) is forecast


def test_dropping_the_master_only_budgets_for_the_twin(app, monkeypatch):
    from app.services import storage_locations
    monkeypatch.setattr(storage_locations, 'free_space',
                        lambda path: {'free_bytes': 4 * 1000 ** 4,
                                      'total_bytes': 8 * 1000 ** 4})
    with app.app_context():
        keep = dld.local_delivery_forecast(keep_bf16=True)
        drop = dld.local_delivery_forecast(keep_bf16=False)
    assert drop['needed_bytes'] < keep['needed_bytes']
    assert drop['master_bytes'] == 0


# --- what to take off the pod ---------------------------------------------------

def test_the_newest_master_and_its_twin_are_what_come_home():
    files = [
        {'path': '/o/Krea_j_000000500.safetensors', 'size': 10},
        {'path': '/o/Krea_j_000002500.safetensors', 'size': 12},
        {'path': '/o/Krea_j_000002500_fp8.safetensors', 'size': 5},
    ]
    picked = dld.select_pod_artifacts(files)
    assert picked['master']['path'].endswith('_000002500.safetensors')
    assert picked['fp8']['path'].endswith('_fp8.safetensors')
    # keep_bf16 off: only the twin is worth the bytes...
    dropped = dld.select_pod_artifacts(files, keep_bf16=False)
    assert dropped['master'] is None and dropped['fp8']
    # ... unless the twin never appeared, in which case the master is the only
    # result there is and leaving it behind would empty-hand the run.
    only_master = dld.select_pod_artifacts(files[:2], keep_bf16=False)
    assert only_master['master']['path'].endswith('_000002500.safetensors')
    assert dld.step_of('Krea_j_000002500.safetensors') == 2500
    assert dld.step_of('Krea_j_000002500_fp8.safetensors') == 2500
    assert dld.step_of('Krea_j.safetensors', default=3000) == 3000


# --- proving what landed --------------------------------------------------------

def test_a_file_is_proven_by_its_size_and_its_header(tmp_path):
    good = tmp_path / 'good.safetensors'
    good.write_bytes(_safetensors())
    proof = dld.verify_local_file(good, expected_size=good.stat().st_size)
    assert proof['tensors'] == 1 and proof['size_bytes'] == good.stat().st_size

    with pytest.raises(dld.LocalDeliveryError, match='advertised'):
        dld.verify_local_file(good, expected_size=good.stat().st_size + 1)

    # Right length, wrong contents: a proxy error page renamed into place.
    garbage = tmp_path / 'garbage.safetensors'
    garbage.write_bytes(b'<html>502 Bad Gateway</html>')
    with pytest.raises(dld.LocalDeliveryError, match='not a readable'):
        dld.verify_local_file(garbage)

    empty = tmp_path / 'empty.safetensors'
    empty.write_bytes(b'')
    with pytest.raises(dld.LocalDeliveryError):
        dld.verify_local_file(empty)


# --- the run's ending -----------------------------------------------------------

def test_the_trainer_never_pushes_while_it_trains(ct):
    """The 403 that killed run #146 arrived on a mid-training push. A delivery
    that brings the model home does not push at all; the historical hub-only
    delivery is untouched."""
    def job():
        return {'config': {'name': 'x', 'process': [{
            'type': 'sd_trainer', 'save': {'dtype': 'bf16'},
            'datasets': [{'folder_path': 'C:/staging'}], 'train': {}}]}}

    pod = {'DATASETS_FOLDER': '/datasets', 'TRAINING_FOLDER': '/output'}
    for delivery in ('local', 'both'):
        out = ct._cloudify_job_config(
            job(), 'Krea_job', 'C:/staging', pod,
            run_params={'training_mode': 'full_transformer',
                        'dense_delivery': delivery,
                        'hf_repo_id': 'tester/Krea-dense'})
        assert 'push_to_hub' not in out['config']['process'][0]['save']
    hub = ct._cloudify_job_config(
        job(), 'Krea_job', 'C:/staging', pod,
        run_params={'training_mode': 'full_transformer',
                    'dense_delivery': 'hub',
                    'hf_repo_id': 'tester/Krea-dense'})
    assert hub['config']['process'][0]['save']['push_to_hub'] is True


def test_the_pod_dies_only_after_the_local_copy_is_proven(
        ct, app, dataset_id, tmp_path, monkeypatch):
    blobs = {'/o/Krea_lds1_dense_000003000.safetensors': _safetensors(b'x' * 64),
             '/o/Krea_lds1_dense_000003000_fp8.safetensors': _safetensors(b'y' * 32)}
    remote = _FakeRemote(blobs)
    destroyed = []
    pushed = []
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda iid: destroyed.append(iid) or True)
    from app.services import dense_pod_hub
    monkeypatch.setattr(
        dense_pod_hub, 'push_master',
        lambda *a, **k: pushed.append(k) or {
            'state': 'done', 'detail': 'uploaded', 'result': {'bytes': 1}})
    monkeypatch.setattr(ct, '_verify_full_transformer_artifact',
                        lambda run, _api=None: 'available')

    with app.app_context():
        run = _dense_run(ct, dataset_id, tmp_path,
                         hf_repo_id='tester/Krea-dense')
        assert ct._deliver_dense_locally(
            run, remote, should_cancel=lambda: False) is True
        params = json.loads(run.train_params)

        assert run.status == 'done'
        assert params['local_artifact_status'] == 'available'
        assert params['local_weight_filename'].endswith('_000003000.safetensors')
        assert params['local_fp8_filename'].endswith('_fp8.safetensors')
        assert params['hub_backup_status'] == 'done'
        # The files are in the DURABLE store, not in the disposable staging dir.
        store = ct.checkpoint_store_dir(run)
        assert set(ct.run_checkpoint_files(run)) == {
            params['local_weight_filename'], params['local_fp8_filename']}
        assert run.checkpoint_local_path.startswith(store)
        # The backup is the master alone: the twin is regenerated in seconds and
        # would eat the private quota twice as fast.
        assert len(pushed) == 1
        assert pushed[0]['src_path'].endswith('_000003000.safetensors')
        # ... and only THEN is the machine released.
        assert destroyed == ['pod-1']
        # Long transfers resume and can be interrupted.
        assert all(d['resume'] and d['cancellable'] for d in remote.downloads)


def test_a_truncated_download_keeps_the_pod(
        ct, app, dataset_id, tmp_path, monkeypatch):
    path = '/o/Krea_lds1_dense_000003000.safetensors'
    remote = _FakeRemote({path: _safetensors(b'x' * 64)}, truncate={path})
    destroyed = []
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda iid: destroyed.append(iid) or True)
    with app.app_context():
        run = _dense_run(ct, dataset_id, tmp_path, delivery='local')
        assert ct._deliver_dense_locally(run, remote) is False
        assert run.status == 'error_pod_kept'
        assert destroyed == []
        params = json.loads(run.train_params)
        # Nothing is ever published as available on a file we could not prove...
        assert params['local_artifact_status'] == 'failed'
        # ... and the card is told WHY, not just that it did not happen.
        assert 'brought home' in params['local_artifact_detail']
        assert 'fetched again' in (run.error or '')


def test_cancelling_a_transfer_keeps_the_pod_and_what_landed(
        ct, app, dataset_id, tmp_path, monkeypatch):
    path = '/o/Krea_lds1_dense_000003000.safetensors'
    remote = _FakeRemote({path: _safetensors(b'x' * 64)})
    destroyed = []
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda iid: destroyed.append(iid) or True)
    with app.app_context():
        run = _dense_run(ct, dataset_id, tmp_path, delivery='local')
        assert ct._deliver_dense_locally(
            run, remote, should_cancel=lambda: True) is False
        assert run.status == 'error_pod_kept'
        assert destroyed == []
        assert 'cancelled' in (run.error or '')
        params = json.loads(run.train_params)
        assert params['local_artifact_status'] == 'cancelled'
        assert 'continues from there' in params['local_artifact_detail']


def test_a_refused_hub_backup_still_leaves_the_run_done(
        ct, app, dataset_id, tmp_path, monkeypatch):
    """The whole point of the ordering: a full private quota costs the ability
    to continue this model later, and nothing else."""
    path = '/o/Krea_lds1_dense_000003000.safetensors'
    remote = _FakeRemote({path: _safetensors(b'x' * 64)})
    monkeypatch.setattr(ct.vast_client, 'destroy_instance', lambda iid: True)
    from app.services import dense_pod_hub
    monkeypatch.setattr(dense_pod_hub, 'push_master', lambda *a, **k: {
        'state': 'failed', 'result': None,
        'detail': 'No Hugging Face copy was made (403 storage limit).'})
    with app.app_context():
        run = _dense_run(ct, dataset_id, tmp_path,
                         hf_repo_id='tester/Krea-dense')
        assert ct._deliver_dense_locally(run, remote) is True
        params = json.loads(run.train_params)
        assert run.status == 'done'
        assert params['local_artifact_status'] == 'available'
        assert params['hub_backup_status'] == 'failed'
        assert params['artifact_status'] == 'missing'
        assert '403' in params['hub_backup_detail']


def test_fetching_again_needs_a_kept_pod_and_runs_in_the_background(
        ct, app, dataset_id, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        ct, '_deliver_dense_locally',
        lambda run, remote, **k: calls.append(run.id) or True)
    monkeypatch.setattr(ct, '_make_remote', lambda run: object())
    with app.app_context():
        run = _dense_run(ct, dataset_id, tmp_path, delivery='local')
        # A run that is still training has nothing to recover.
        with pytest.raises(ValueError, match='nothing to fetch'):
            ct.fetch_dense_locally(run.id)
        run.status = 'error_pod_kept'
        ct.db.session.commit()
        assert ct._can_fetch_dense_locally(run) is True
        result = ct.fetch_dense_locally(run.id)
        assert result['state'] == 'fetching'
        thread = ct._dense_fetch_threads.get(run.id)
        # The transfer makes the run ACTIVE again, so a Stop pressed during it
        # asks who is in charge — and a stop that finds nobody destroys the pod
        # mid-transfer. The fetch registers as this run's thread precisely so
        # that question has an answer.
        assert ct._monitor_threads.get(run.id) is thread
        if thread:
            thread.join(timeout=10)
        assert calls == [run.id]
        assert ct._monitor_threads.get(run.id) is None
        # A hub-only run is not offered a local fetch at all.
        hub_run = _dense_run(ct, dataset_id, tmp_path, delivery='hub')
        hub_run.status = 'error_pod_kept'
        ct.db.session.commit()
        assert ct._can_fetch_dense_locally(hub_run) is False
