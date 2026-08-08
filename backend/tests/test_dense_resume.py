"""Continuing a FULL MODEL from the Hub: the pod fetches its own checkpoint.

Run #146 proved the mechanism the hard way — relaunching its job by hand made
ai-toolkit auto-resume from the checkpoint sitting in the job's save_root, twice.
What was missing was a way to put that checkpoint on a FRESH pod, and the Hub
road was the first answer: the pod pulls the file itself, over a datacenter
link, in minutes. Everything else is the LoRA path, unchanged: one
continue_cloud_run, one launch, one set of guards.

This road used to be the ONLY one, because a 26 GB upload could not survive a
multipart body built in memory. The other road (pushing the local copy from
this computer, in resumable slices) lives in test_dense_resume_direct.py; what
is tested HERE is that offering it changed nothing about this one.
"""
import json

import pytest

from app.services import dense_pod_hub


@pytest.fixture()
def ct(app, monkeypatch):
    from app.services import cloud_training, storage_locations
    monkeypatch.setattr(storage_locations, 'free_space',
                        lambda path: {'free_bytes': 4 * 1000 ** 4,
                                      'total_bytes': 8 * 1000 ** 4})
    return cloud_training


@pytest.fixture()
def dataset_id(client):
    return client.post('/api/dataset/create', json={
        'name': 'Dense resume', 'trigger_word': 'person',
    }).get_json()['id']


def _delivered_run(ct, dataset_id, **params):
    run = ct.CloudTrainingRun(
        dataset_id=dataset_id, status='done', run_name='dense',
        job_name='Krea_lds7_dense',
        train_params=json.dumps({
            'training_mode': 'full_transformer', 'train_type': 'krea',
            'variant': 'base', 'dense_delivery': 'both', 'steps': 3000,
            'artifact_status': 'available',
            'hf_repo_id': 'tester/Krea-2-full-7-dense',
            'hf_weight_filename': 'Krea_lds7_dense_000003000.safetensors',
            **params}))
    ct.db.session.add(run)
    ct.db.session.commit()
    return run


# --- what a full model can be continued from ------------------------------------

def test_the_hub_copy_is_offered_when_it_is_the_only_one_left(ct, app, dataset_id):
    """A run whose local files are gone (purged, or delivered to the Hub only)
    still has the Hub road, and it is the one offered."""
    with app.app_context():
        run = _delivered_run(ct, dataset_id)
        [candidate] = ct._dense_resume_candidates(run)
        assert candidate['source'] == 'hub'
        assert candidate['step'] == 3000
        assert candidate['repo_id'] == 'tester/Krea-2-full-7-dense'
        payload = ct._run_payload(run)
        assert payload['resume_steps'] == [3000]
        assert payload['resume_checkpoints'][0]['source'] == 'hub'

        # An unverified delivery is not a source: seeding a checkpoint that may
        # be truncated would spend a fresh pod training from garbage.
        pending = _delivered_run(ct, dataset_id, artifact_status='verification_pending')
        assert ct._dense_resume_candidates(pending) == []

        # Neither is the quantized twin — fp8 weights cannot be trained on.
        twin = _delivered_run(
            ct, dataset_id,
            hf_weight_filename='Krea_lds7_dense_000003000_fp8.safetensors')
        assert ct._dense_resume_candidates(twin) == []


def test_a_kept_pod_can_still_be_continued(ct, app, dataset_id):
    """The state a dense run ends in when something went wrong is exactly the
    state in which continuing it matters most."""
    with app.app_context():
        run = _delivered_run(ct, dataset_id)
        run.status = 'error_pod_kept'
        ct.db.session.commit()
        assert ct._run_payload(run)['resume_steps'] == [3000]


def test_continue_launches_a_dense_run_that_resumes_from_the_hub(
        ct, app, dataset_id, monkeypatch):
    seen = {}

    def _fake_launch(user_id, ds_id, **kwargs):
        seen.update(kwargs)
        return {'run_id': 99, 'status': 'preparing'}

    monkeypatch.setattr(ct, 'launch_cloud_training', _fake_launch)
    with app.app_context():
        run = _delivered_run(ct, dataset_id)
        res = ct.continue_cloud_run('local', run.id, extra_steps=1000)

    assert res['resumed_from'] == 3000
    assert res['target_steps'] == 4000
    assert res['resume_source'] == 'hub'
    # The mode is replayed — without it the continuation would quietly become a
    # LoRA run on a dense recipe.
    assert seen['training_mode'] == 'full_transformer'
    assert seen['steps'] == 4000
    assert seen['resume_ckpt_path'] is None
    assert seen['resume_hf'] == {
        'repo_id': 'tester/Krea-2-full-7-dense',
        'filename': 'Krea_lds7_dense_000003000.safetensors'}
    assert seen['resume_step'] == 3000


def test_continuing_from_an_earlier_step_is_refused_when_it_does_not_exist(
        ct, app, dataset_id):
    with app.app_context():
        run = _delivered_run(ct, dataset_id)
        with pytest.raises(ValueError, match='no harvested checkpoint at step'):
            ct.continue_cloud_run('local', run.id, from_step=1500)
        run.status = 'training'
        ct.db.session.commit()
        with pytest.raises(ValueError, match='still running'):
            ct.continue_cloud_run('local', run.id)


def test_a_dense_continuation_stamps_its_hub_seed_and_asks_for_a_big_disk(
        ct, app, dataset_id, monkeypatch):
    """The launch is a normal cloud launch: same guards, and the pod profile
    that a 26 GB checkpoint plus a 26 GB base actually needs."""
    monkeypatch.setattr(ct, '_start_monitor', lambda *a, **k: None)
    monkeypatch.setattr(ct, '_reconcile_before_launch', lambda *a, **k: None)
    monkeypatch.setattr(ct.lt, 'assert_trainable', lambda *a, **k: None)
    monkeypatch.setattr(ct, '_assert_official_base_reachable', lambda *a, **k: None)
    monkeypatch.setattr(ct, '_validate_full_transformer_token',
                        lambda token, _api=None, **_kw: (object(), 'tester', False))
    monkeypatch.setattr(ct, '_assert_dense_storage_headroom',
                        lambda *a, **k: {'fits': True})
    monkeypatch.setattr(ct, '_create_full_transformer_repo', lambda run, token, **_kw: {
        'hf_repo_id': 'tester/Krea-2-full-9-dense',
        'hf_url': 'https://huggingface.co/tester/Krea-2-full-9-dense'})
    monkeypatch.setenv('VAST_API_KEY', 'vast-test')
    monkeypatch.setenv('HF_CLOUD_TOKEN', 'hf-cloud-secret-x')
    with app.app_context():
        result = ct.launch_cloud_training(
            'local', dataset_id, train_type='krea', variant='base',
            training_mode='full_transformer', steps=4000,
            resume_hf={'repo_id': 'tester/Krea-2-full-7-dense',
                       'filename': 'Krea_lds7_dense_000003000.safetensors'},
            resume_step=3000, resumed_from=3000)
        run = ct.db.session.get(ct.CloudTrainingRun, result['run_id'])
        params = json.loads(run.train_params)
    assert params['resume_source'] == 'hub'
    assert params['resume_hf_repo_id'] == 'tester/Krea-2-full-7-dense'
    assert params['resume_step'] == 3000
    # search_offers(min_disk_gb=...) is fed from this: a dense pod that cannot
    # hold base + checkpoint is a rental that fails after it is paid for.
    assert ct._disk_gb_for({'full_transformer': {}}, params) == 200


# --- the seeding itself ---------------------------------------------------------

def test_the_pod_fetches_its_own_resume_checkpoint(ct, app, dataset_id, monkeypatch):
    calls = {}

    def _fetch(remote, **kwargs):
        calls.update(kwargs)
        return {'ok': True, 'bytes': 26 * 1000 ** 3}

    monkeypatch.setattr(dense_pod_hub, 'fetch_checkpoint', _fetch)

    class _Remote:
        def seed_checkpoint(self, *a, **k):
            raise AssertionError('the Hub road must not touch this uplink at '
                                 'all — the pod fetches the file itself')

        def upload_file_slice(self, *a, **k):
            raise AssertionError('the Hub road must not touch this uplink at '
                                 'all — the pod fetches the file itself')

    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='provisioning', run_name='dense',
            job_name='Krea_lds9_dense', vast_instance_id='pod-9',
            train_params=json.dumps({
                'training_mode': 'full_transformer', 'train_type': 'krea',
                'dense_delivery': 'both', 'resume_step': 3000,
                'resume_hf_repo_id': 'tester/Krea-2-full-7-dense',
                'resume_hf_filename': 'Krea_lds7_dense_000003000.safetensors'}))
        ct.db.session.add(run)
        ct.db.session.commit()
        ct._seed_resume_checkpoint(run, _Remote(), {
            'DATASETS_FOLDER': '/datasets', 'TRAINING_FOLDER': '/output'})

    # THIS job's save_root, under THIS job's name: that is what ai-toolkit's
    # auto-resume globs, and the step it reads back.
    assert calls['dest_path'] == '/output/Krea_lds9_dense/Krea_lds9_dense_000003000.safetensors'
    assert calls['repo_id'] == 'tester/Krea-2-full-7-dense'
    assert calls['filename'] == 'Krea_lds7_dense_000003000.safetensors'
    assert calls['instance_id'] == 'pod-9'


def test_a_launch_with_no_seed_stays_a_no_op(ct, app, dataset_id):
    class _Remote:
        def seed_checkpoint(self, *a, **k):
            raise AssertionError('nothing to seed')

    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=dataset_id, status='provisioning', run_name='plain',
            job_name='lds10_plain', train_params=json.dumps({}))
        ct.db.session.add(run)
        ct.db.session.commit()
        assert ct._seed_resume_checkpoint(run, _Remote(), {
            'DATASETS_FOLDER': '/datasets', 'TRAINING_FOLDER': '/output'}) is None


# --- the commands the pod is given ---------------------------------------------

def test_the_pod_commands_never_interpolate_a_user_value():
    """Every value the user influences travels in argv, quoted; the program text
    is a constant. A repository id carries a dataset name, and a quoting mistake
    in a remote command is not a bug anyone finds twice."""
    push = dense_pod_hub.build_push_command(
        '/output/j/Krea_j.safetensors', "tester/Krea-it's-fine",
        'Krea_j.safetensors', '/datasets/_lds_hub/hf_token.txt')
    assert push.startswith('python -c ')
    assert dense_pod_hub.PUSH_PROGRAM in push.replace("'\\''", "'")
    assert "'tester/Krea-it'\\''s-fine'" in push
    assert "'" not in dense_pod_hub.PUSH_PROGRAM
    assert "'" not in dense_pod_hub.FETCH_PROGRAM

    fetch = dense_pod_hub.build_fetch_command(
        'tester/repo', 'w.safetensors', '/output/j/w.safetensors', '')
    assert fetch.endswith("'tester/repo' 'w.safetensors' '/output/j/w.safetensors' ''")

    # vast carries 16384 characters and a sibling lane was refused twice for
    # going over. These programs are inlined on purpose and stay tiny; the
    # ceiling is what makes "tiny" a fact rather than an intention.
    assert len(push) < dense_pod_hub.MAX_COMMAND_CHARS
    with pytest.raises(dense_pod_hub.PodHubError, match='ceiling'):
        dense_pod_hub.build_fetch_command(
            'tester/repo', 'w.safetensors', 'x' * dense_pod_hub.MAX_COMMAND_CHARS)


def test_a_result_line_is_read_out_of_the_pods_chatter():
    output = ('Collecting huggingface_hub\nDownloading...\n'
              'LDS_HUB_RESULT {"ok": true, "bytes": 26}\n')
    assert dense_pod_hub.parse_result(output) == {'ok': True, 'bytes': 26}
    assert dense_pod_hub.parse_result('nothing useful') is None
    assert dense_pod_hub.parse_result('LDS_HUB_RESULT {broken') is None


def test_a_failed_backup_is_reported_and_never_raised(tmp_path):
    class _Remote:
        def get_settings(self):
            return {'DATASETS_FOLDER': '/datasets', 'TRAINING_FOLDER': '/output'}

        def seed_checkpoint(self, *a, **k):
            return None

    class _Vast:
        def execute_command(self, instance_id, command):
            return 'https://vast/result'

        def fetch_command_result(self, url):
            return 'LDS_HUB_RESULT {"ok": false, "error": "403 storage limit"}'

    out = dense_pod_hub.push_master(
        _Remote(), instance_id='pod-1', src_path='/output/j/w.safetensors',
        repo_id='tester/repo', path_in_repo='w.safetensors',
        hf_token='hf-x', tmp_dir=str(tmp_path), vast=_Vast(),
        _sleep=lambda s: None, _now=lambda: 0.0)
    assert out['state'] == 'failed'
    assert '403 storage limit' in out['detail']
    assert 'cannot be resumed later' in out['detail']

    # ... and the fetch half RAISES instead, because a resume that cannot place
    # its checkpoint must never train from scratch on the user's money.
    with pytest.raises(dense_pod_hub.PodHubError, match='403'):
        dense_pod_hub.fetch_checkpoint(
            _Remote(), instance_id='pod-1', repo_id='tester/repo',
            filename='w.safetensors', dest_path='/output/j/w.safetensors',
            hf_token='hf-x', tmp_dir=str(tmp_path), vast=_Vast(),
            _sleep=lambda s: None, _now=lambda: 0.0)
