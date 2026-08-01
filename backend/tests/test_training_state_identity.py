import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import training_state_identity as identity


def _job():
    return {
        'job': 'extension',
        'config': {
            'name': 'lora_subject',
            'process': [{
                'training_folder': 'C:/mutable/run',
                'network': {'type': 'lora', 'linear': 16, 'linear_alpha': 16},
                'model': {'arch': 'flux', 'name_or_path': 'org/model'},
                'save': {'save_every': 250, 'max_step_saves_to_keep': 4},
                'sample': {'sample_every': 250, 'prompts': ['subject']},
                'datasets': [{
                    'folder_path': 'C:/mutable/data',
                    'mask_path': 'C:/mutable/data/_masks',
                    'resolution': 768,
                }],
                'train': {
                    'steps': 1000,
                    'optimizer': 'adamw8bit',
                    'lr': 1e-4,
                },
            }],
        },
    }


def _krea_job():
    job = _job()
    model = job['config']['process'][0]['model']
    model.clear()
    model.update({
        'arch': 'krea2',
        'name_or_path': 'krea/Krea-2-Raw',
        'assistant_lora_path': (
            'ostris/krea2_turbo_training_adapter/'
            'krea2_turbo_training_adapter_v1.safetensors'),
    })
    return job


def test_job_identity_ignores_target_paths_and_presentation_but_keeps_cadence():
    first, network, model = identity._normalized_job(_job())
    changed = copy.deepcopy(_job())
    process = changed['config']['process'][0]
    process['training_folder'] = 'D:/another/run'
    process['datasets'][0]['folder_path'] = 'D:/another/data'
    process['datasets'][0]['mask_path'] = 'D:/another/masks'
    process['train']['steps'] = 5000
    process['save']['max_step_saves_to_keep'] = 99
    process['sample']['prompts'] = ['different preview only']
    second, _, _ = identity._normalized_job(changed)

    assert first == second
    assert network == {'type': 'lora', 'linear': 16, 'linear_alpha': 16}
    assert model == {'arch': 'flux', 'name_or_path': 'org/model'}

    changed['config']['process'][0]['train']['optimizer'] = 'prodigy'
    third, _, _ = identity._normalized_job(changed)
    assert third != first

    changed = copy.deepcopy(_job())
    changed['config']['process'][0]['save']['save_every'] = 1000
    cadence, _, _ = identity._normalized_job(changed)
    assert cadence != first

    changed = copy.deepcopy(_job())
    changed['config']['process'][0]['sample']['sample_every'] = 1000
    cadence, _, _ = identity._normalized_job(changed)
    assert cadence != first


def test_runtime_fingerprint_requires_full_package_and_gpu_identity(
    tmp_path, monkeypatch
):
    python = tmp_path / 'python.exe'
    python.write_bytes(b'placeholder')
    complete = {
        'python': '3.12.9',
        'torch': '2.7.1+cu128',
        'cuda': '12.8',
        'cudnn': '90701',
        'cuda_devices': 1,
        'gpus': [{
            'index': 0,
            'name': 'NVIDIA RTX Test',
            'compute_capability': '8.9',
        }],
        'gpu_driver': '576.80',
        'packages': {
            'numpy': '2.2.6',
            'accelerate': '1.8.1',
            'diffusers': '0.34.0',
            'transformers': '4.53.0',
            'safetensors': '0.5.3',
            'bitsandbytes': '0.46.0',
            'Example_Package': '2.0',
            'example.package': '1.0',
        },
    }
    monkeypatch.setattr(
        identity.subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(complete) + '\n',
        ),
    )
    result = identity._runtime_fingerprint(python)
    assert result['packages']['diffusers'] == '0.34.0'
    assert result['packages']['example-package'] == '1.0|2.0'
    assert 'Example_Package' not in result['packages']
    assert result['gpus'][0]['compute_capability'] == '8.9'
    assert result['gpu_driver'] == '576.80'

    incomplete = copy.deepcopy(complete)
    incomplete['packages']['bitsandbytes'] = ''
    monkeypatch.setattr(
        identity.subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(incomplete) + '\n',
        ),
    )
    with pytest.raises(
        identity.TrainingStateIdentityError,
        match='runtime identity is incomplete',
    ):
        identity._runtime_fingerprint(python)


def test_krea_hosted_inputs_are_commit_pinned_and_commit_change_changes_identity(
        tmp_path, monkeypatch):
    """Base, assistant, implicit TE and implicit VAE all load local pinned bytes."""
    commits = {
        'krea/Krea-2-Raw': '1' * 40,
        'ostris/krea2_turbo_training_adapter': '2' * 40,
        'Qwen/Qwen3-VL-4B-Instruct': '3' * 40,
        'Qwen/Qwen-Image': '4' * 40,
    }

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def model_info(self, repo_id, *, timeout, files_metadata):
            return SimpleNamespace(sha=commits[repo_id])

    calls = []

    # Deliberately has no `local_dir_use_symlinks` keyword. This is compatible
    # with huggingface_hub releases where that deprecated argument was removed.
    def fake_snapshot_download(
            *, repo_id, revision, cache_dir, token,
            allow_patterns, etag_timeout):
        calls.append(('snapshot', repo_id, revision, allow_patterns))
        root = (
            Path(cache_dir)
            / ('models--' + repo_id.replace('/', '--'))
            / 'snapshots' / revision)
        root.mkdir(parents=True, exist_ok=True)
        names = allow_patterns or ['config.json']
        for name in names:
            artifact = root.joinpath(*name.split('/'))
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f'{repo_id}@{revision}:{name}'.encode())
        return str(root)

    def fake_hf_hub_download(
            *, repo_id, filename, revision, cache_dir, token, etag_timeout):
        calls.append(('file', repo_id, revision, [filename]))
        artifact = (
            Path(cache_dir)
            / ('models--' + repo_id.replace('/', '--'))
            / 'snapshots' / revision
        ).joinpath(*filename.split('/'))
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(
            f'{repo_id}@{revision}:{filename}'.encode())
        return str(artifact)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'HfApi', FakeApi)
    monkeypatch.setattr(
        huggingface_hub, 'snapshot_download', fake_snapshot_download)
    monkeypatch.setattr(
        huggingface_hub, 'hf_hub_download', fake_hf_hub_download)
    monkeypatch.setattr(
        identity, '_runtime_fingerprint',
        lambda _python: {
            'python': '3.12', 'torch': '2.7', 'cuda': '12.8',
            'cudnn': '9', 'cuda_devices': 1, 'gpus': [{}],
            'gpu_driver': '1', 'packages': {}, 'protocol': 2,
            'shape_revision': 'test',
        })

    first_job = _krea_job()
    pins = identity.pin_job_model_artifacts(
        first_job, cache_dir=tmp_path / 'hf', token='secret')
    model = first_job['config']['process'][0]['model']
    assert Path(model['name_or_path']).is_dir()
    assert model['model_kwargs']['checkpoint_filename'] == 'raw.safetensors'
    assert Path(model['assistant_lora_path']).is_file()
    assert Path(model['model_kwargs']['text_encoder_path']).is_dir()
    assert Path(model['model_kwargs']['vae_path']).is_dir()
    assert {pin['repo'] for pin in pins} == set(commits)
    assert len(calls) == 4

    prepared = {
        'manifest': [],
        'snapshot': {'captions': {}, 'images': {}, 'dataset': {}},
    }
    toolkit = {'supported': True, 'aitoolkit_revision': 'a' * 40}
    first_identity = identity.build_identity(
        job_config=first_job,
        prepared=prepared,
        toolkit_probe=toolkit,
        python_path=tmp_path / 'python.exe',
    )
    assert {
        record['field'] for record in first_identity['model_artifacts']
    } == {
        'model.name_or_path',
        'model.assistant_lora_path',
        'model.model_kwargs.text_encoder_path',
        'model.model_kwargs.vae_path',
    }
    assert all(
        len(record['sha256']) == 64
        and os.path.isabs(record['lexical_path'])
        and record['repo']
        and record['commit']
        for record in first_identity['model_artifacts']
    )

    # Re-resolving the same mutable repo names to a new commit must produce a
    # different compatibility identity even when a test serves identical bytes.
    commits['krea/Krea-2-Raw'] = '5' * 40
    second_job = _krea_job()
    identity.pin_job_model_artifacts(
        second_job, cache_dir=tmp_path / 'hf', token='secret')
    second_identity = identity.build_identity(
        job_config=second_job,
        prepared=prepared,
        toolkit_probe=toolkit,
        python_path=tmp_path / 'python.exe',
    )
    assert first_identity['config_hash'] == second_identity['config_hash']
    assert first_identity['base_hash'] != second_identity['base_hash']
    assert (
        identity.compatibility_spec(first_identity).base_model_hash
        != identity.compatibility_spec(second_identity).base_model_hash
    )


def test_pre_resolved_model_pins_apply_without_resolving_again(
        tmp_path, monkeypatch):
    local = tmp_path / 'snapshot'
    local.mkdir()
    (local / 'raw.safetensors').write_bytes(b'RAW')
    te = tmp_path / 'te'
    te.mkdir()
    (te / 'config.json').write_bytes(b'TE')
    vae = tmp_path / 'vae'
    vae.mkdir()
    (vae / 'config.json').write_bytes(b'VAE')
    adapter = tmp_path / 'adapter.safetensors'
    adapter.write_bytes(b'ADAPTER')
    pins = [
        {
            'field': 'model.name_or_path',
            'repo': 'krea/Krea-2-Raw',
            'commit': '1' * 40,
            'filename': 'raw.safetensors',
            'original': 'krea/Krea-2-Raw',
            'local_path': str(local),
            'artifact_path': str(local),
            'checkpoint_filename': 'raw.safetensors',
        },
        {
            'field': 'model.assistant_lora_path',
            'repo': 'ostris/krea2_turbo_training_adapter',
            'commit': '2' * 40,
            'filename': 'krea2_turbo_training_adapter_v1.safetensors',
            'original': (
                'ostris/krea2_turbo_training_adapter/'
                'krea2_turbo_training_adapter_v1.safetensors'),
            'local_path': str(adapter),
            'artifact_path': str(adapter),
        },
        {
            'field': 'model.model_kwargs.text_encoder_path',
            'repo': 'Qwen/Qwen3-VL-4B-Instruct',
            'commit': '3' * 40,
            'original': 'Qwen/Qwen3-VL-4B-Instruct',
            'local_path': str(te),
            'artifact_path': str(te),
        },
        {
            'field': 'model.model_kwargs.vae_path',
            'repo': 'Qwen/Qwen-Image',
            'commit': '4' * 40,
            'original': 'Qwen/Qwen-Image',
            'local_path': str(vae),
            'artifact_path': str(vae),
        },
    ]
    monkeypatch.setattr(
        identity, '_materialize_hf_snapshot',
        lambda *args, **kwargs: pytest.fail('resolver must not run'))
    monkeypatch.setattr(
        identity.run_environment, 'pinned_hf_artifact_signature',
        lambda path, *_args, **_kwargs:
        identity.run_environment.artifact_signature(path, strict=True))

    job = _krea_job()
    applied = identity.pin_job_model_artifacts(
        job, cache_dir=tmp_path / 'hf', pins=pins)

    assert applied == pins
    assert job['config']['process'][0]['model']['name_or_path'] == str(local)


def test_every_local_weight_bearing_model_field_gets_full_sha256(tmp_path):
    files = {}
    for name in ('base', 'extras', 'assistant', 'vae', 'te', 'implicit-te',
                 'implicit-vae'):
        artifact = tmp_path / f'{name}.bin'
        artifact.write_bytes((name * 17).encode())
        files[name] = str(artifact)
    job = _job()
    job['config']['process'][0]['model'] = {
        'arch': 'test',
        'name_or_path': files['base'],
        'extras_name_or_path': files['extras'],
        'assistant_lora_path': files['assistant'],
        'vae_path': files['vae'],
        'te_name_or_path': files['te'],
        'model_kwargs': {
            'text_encoder_path': files['implicit-te'],
            'vae_path': files['implicit-vae'],
        },
    }

    artifacts = identity._artifact_identities(job)

    assert set(artifacts) == {
        'model.name_or_path',
        'model.extras_name_or_path',
        'model.assistant_lora_path',
        'model.vae_path',
        'model.te_name_or_path',
        'model.model_kwargs.text_encoder_path',
        'model.model_kwargs.vae_path',
    }
    assert all(
        len(value['sha256']) == 64 and value['sampled'] is False
        for value in artifacts.values())


@pytest.mark.parametrize('arch', [
    'flux2_klein_4b',
    'flux2_klein_9b',
])
def test_exact_pinning_fails_closed_for_hidden_unpinnable_dependencies(
        tmp_path, arch):
    job = _job()
    job['config']['process'][0]['model']['arch'] = arch

    with pytest.raises(
            identity.TrainingStateIdentityError,
            match='exact-state model pinning is unavailable'):
        identity.pin_job_model_artifacts(
            job, cache_dir=tmp_path / 'hub')


def test_identity_publish_refuses_symlink_target_without_touching_victim(
        tmp_path):
    victim = tmp_path / 'victim.json'
    victim.write_text('do-not-touch', encoding='utf-8')
    target = tmp_path / 'context.json'
    try:
        os.symlink(victim, target)
    except (OSError, NotImplementedError):
        pytest.skip('symlink creation is unavailable on this platform')

    with pytest.raises(OSError, match='unsafe|link|reparse'):
        identity.write_identity(target, {'schema': 'test'})

    assert victim.read_text(encoding='utf-8') == 'do-not-touch'
    assert target.is_symlink()


def test_identity_publish_refuses_symlink_parent(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    linked = tmp_path / 'linked'
    try:
        os.symlink(real, linked, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('directory symlink creation is unavailable on this platform')

    with pytest.raises(OSError, match='link|reparse'):
        identity.write_identity(linked / 'context.json', {'schema': 'test'})

    assert not (real / 'context.json').exists()


def test_hf_snapshot_hash_allows_only_same_repo_blob_symlinks(tmp_path):
    commit = 'a' * 40
    model_root = tmp_path / 'hub' / 'models--org--model'
    blobs = model_root / 'blobs'
    snapshot = model_root / 'snapshots' / commit
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / 'abc'
    blob.write_bytes(b'IMMUTABLE-BYTES')
    linked = snapshot / 'model.safetensors'
    try:
        os.symlink(blob, linked)
    except (OSError, NotImplementedError):
        pytest.skip('symlink creation is unavailable on this platform')

    safe = identity.run_environment.pinned_hf_artifact_signature(
        snapshot, 'org/model', commit, strict=True)
    assert safe['sha256']
    assert safe['files'] == 1

    outside = tmp_path / 'outside.bin'
    outside.write_bytes(b'OUTSIDE')
    linked.unlink()
    os.symlink(outside, linked)
    with pytest.raises(OSError, match='escapes'):
        identity.run_environment.pinned_hf_artifact_signature(
            snapshot, 'org/model', commit, strict=True)


def test_child_rehash_rejects_hf_bytes_mutated_after_parent_identity(tmp_path):
    from app.training_bridge.lds_aitk_bridge_contract import (
        verify_identity_model_artifacts,
    )

    commit = 'b' * 40
    model_root = tmp_path / 'hub' / 'models--org--model'
    (model_root / 'blobs').mkdir(parents=True)
    snapshot = model_root / 'snapshots' / commit
    snapshot.mkdir(parents=True)
    weights = snapshot / 'model.safetensors'
    weights.write_bytes(b'ORIGINAL-HF-BYTES')
    job = _job()
    job['config']['process'][0]['model']['name_or_path'] = str(snapshot)
    job['_lds_model_pins'] = [{
        'field': 'model.name_or_path',
        'repo': 'org/model',
        'commit': commit,
    }]
    artifacts = identity._artifact_identities(job)
    contracts = identity._model_artifact_contracts(job, artifacts)

    verify_identity_model_artifacts({'model_artifacts': contracts})
    weights.write_bytes(b'M' * len(b'ORIGINAL-HF-BYTES'))

    with pytest.raises(ValueError, match='bytes changed before load'):
        verify_identity_model_artifacts({'model_artifacts': contracts})


def test_child_rehash_rejects_hf_snapshot_symlink_retarget(tmp_path):
    from app.training_bridge.lds_aitk_bridge_contract import (
        verify_identity_model_artifacts,
    )

    commit = 'c' * 40
    model_root = tmp_path / 'hub' / 'models--org--model'
    blobs = model_root / 'blobs'
    snapshot = model_root / 'snapshots' / commit
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    first = blobs / 'first'
    second = blobs / 'second'
    first.write_bytes(b'FIRST-BLOB')
    second.write_bytes(b'SECOND-BLOB')
    linked = snapshot / 'model.safetensors'
    try:
        linked.symlink_to(first)
    except (OSError, NotImplementedError):
        pytest.skip('symlink creation is unavailable on this platform')
    job = _job()
    job['config']['process'][0]['model']['name_or_path'] = str(snapshot)
    job['_lds_model_pins'] = [{
        'field': 'model.name_or_path',
        'repo': 'org/model',
        'commit': commit,
    }]
    artifacts = identity._artifact_identities(job)
    contracts = identity._model_artifact_contracts(job, artifacts)
    verify_identity_model_artifacts({'model_artifacts': contracts})

    linked.unlink()
    linked.symlink_to(second)

    with pytest.raises(ValueError, match='bytes changed before load'):
        verify_identity_model_artifacts({'model_artifacts': contracts})


def test_child_identity_loader_refuses_symlink_target(tmp_path):
    from app.training_bridge.lds_aitk_bridge_contract import (
        IDENTITY_SCHEMA,
        load_identity,
    )

    victim = tmp_path / 'identity-victim.json'
    victim.write_text(json.dumps({
        'schema': IDENTITY_SCHEMA,
        'config_hash': '1',
        'dataset_hash': '2',
        'base_hash': '3',
        'network_hash': '4',
        'toolkit_revision': '5',
        'runtime': {'python': 'test'},
    }), encoding='utf-8')
    target = tmp_path / 'context.json'
    try:
        target.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip('symlink creation is unavailable on this platform')

    with pytest.raises(ValueError, match='invalid identity file'):
        load_identity(target)
