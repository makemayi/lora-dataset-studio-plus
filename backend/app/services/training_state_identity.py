"""Stable compatibility identity for LDS exact training-state bundles.

The injected bridge only receives a small JSON document; it must not import the
Flask application.  This module builds that document from the same job config
and frozen dataset snapshot that LDS records for the launch.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import subprocess

from . import run_environment, training_state_bundle
from ..training_bridge.lds_aitk_bridge_contract import (
    atomic_json_nofollow,
    BRIDGE_PROTOCOL,
    IDENTITY_SCHEMA,
    SHAPE_REVISION,
)


_RUNTIME_PROBE = (
    "import json,platform,re,subprocess,torch\n"
    "from importlib import metadata\n"
    "_raw={}\n"
    "for _dist in metadata.distributions():\n"
    "    _name=_dist.metadata.get('Name')\n"
    "    if not _name:continue\n"
    "    _key=re.sub(r'[-_.]+','-',str(_name)).lower()\n"
    "    _raw.setdefault(_key,set()).add(str(_dist.version or ''))\n"
    "_versions={k:'|'.join(sorted(v)) for k,v in sorted(_raw.items())}\n"
    "_count=int(torch.cuda.device_count())\n"
    "_gpus=[{'index':i,'name':str(torch.cuda.get_device_name(i)),"
    "'compute_capability':'.'.join(str(x) for x in "
    "torch.cuda.get_device_capability(i))} for i in range(_count)]\n"
    "_driver='none'\n"
    "if _count:\n"
    "    try:\n"
    "        _proc=subprocess.run(['nvidia-smi','--query-gpu=driver_version',"
    "'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)\n"
    "        _vals=[x.strip() for x in _proc.stdout.splitlines() if x.strip()]\n"
    "        _driver='|'.join(_vals) if _proc.returncode==0 and "
    "len(_vals)==_count else ''\n"
    "    except Exception:_driver=''\n"
    "print(json.dumps({"
    "'python':platform.python_version(),"
    "'torch':str(torch.__version__),"
    "'cuda':str(torch.version.cuda or ''),"
    "'cudnn':str(torch.backends.cudnn.version() or ''),"
    "'cuda_devices':_count,"
    "'gpus':_gpus,"
    "'gpu_driver':_driver,"
    "'packages':_versions"
    "},sort_keys=True))\n"
)

EXACT_CAPABILITIES = frozenset({
    'dataloader-dataset-state',
    'dataloader-order-cursor',
    'deterministic-latent-cache',
    'ema',
    'optimizer',
    'preprocessing-cache-bytes',
    'raw-weights',
    'rng-cuda',
    'rng-numpy',
    'rng-python',
    'rng-torch',
    'scaler',
    'scheduler',
})


class TrainingStateIdentityError(RuntimeError):
    pass


_HF_REPO_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$')
_HF_COMMIT_RE = re.compile(r'^[0-9a-fA-F]{40,64}$')
_MODEL_PATH_FIELDS = (
    ('name_or_path',),
    ('extras_name_or_path',),
    ('assistant_lora_path',),
    ('vae_path',),
    ('te_name_or_path',),
    ('model_kwargs', 'text_encoder_path'),
    ('model_kwargs', 'vae_path'),
)
_KREA_TEXT_ENCODER = 'Qwen/Qwen3-VL-4B-Instruct'
_KREA_VAE = 'Qwen/Qwen-Image'
_UNPINNABLE_HIDDEN_DEPENDENCIES = {
    'flux2_klein_4b': (
        'FLUX.2 Klein loads hard-coded Qwen text-encoder and ai-toolkit VAE '
        'repositories that the job config cannot pin'),
    'flux2_klein_9b': (
        'FLUX.2 Klein loads hard-coded Qwen text-encoder and ai-toolkit VAE '
        'repositories that the job config cannot pin'),
}


def _model_process(job_config: dict) -> dict:
    try:
        process = job_config['config']['process'][0]
        model = process['model']
    except (KeyError, IndexError, TypeError) as exc:
        raise TrainingStateIdentityError(
            'job config has no supported model block') from exc
    if not isinstance(model, dict):
        raise TrainingStateIdentityError('job config model block is invalid')
    return model


def _nested_get(mapping: dict, path: tuple[str, ...]):
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _nested_set(mapping: dict, path: tuple[str, ...], value) -> None:
    current = mapping
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def _field_name(path: tuple[str, ...]) -> str:
    return 'model.' + '.'.join(path)


def _is_hf_repo(value) -> bool:
    return isinstance(value, str) and bool(_HF_REPO_RE.fullmatch(value.strip()))


def _hf_token_value(token):
    value = str(token or '').strip()
    return value or None


def _resolve_hf_commit(repo_id: str, *, token=None) -> str:
    """Bounded resolution of one mutable repo name to an immutable commit."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise TrainingStateIdentityError(
            'huggingface_hub is unavailable for immutable model pinning') from exc
    token = _hf_token_value(token)
    try:
        info = HfApi(token=token).model_info(
            repo_id, timeout=15, files_metadata=False)
    except Exception as exc:
        raise TrainingStateIdentityError(
            f'could not resolve an immutable revision for {repo_id}') from exc
    commit = str(getattr(info, 'sha', '') or '').strip().lower()
    if not _HF_COMMIT_RE.fullmatch(commit):
        raise TrainingStateIdentityError(
            f'Hugging Face returned no immutable commit for {repo_id}')
    return commit


def _materialize_hf_snapshot(
        repo_id: str,
        *,
        cache_dir,
        token=None,
        allow_patterns=None,
) -> tuple[str, str]:
    """Resolve a mutable repo id once, then download that exact commit locally."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise TrainingStateIdentityError(
            'huggingface_hub is unavailable for immutable model pinning') from exc
    token = _hf_token_value(token)
    commit = _resolve_hf_commit(repo_id, token=token)
    cache_root = Path(cache_dir)
    try:
        local = snapshot_download(
            repo_id=repo_id,
            revision=commit,
            cache_dir=str(cache_root),
            token=token,
            allow_patterns=allow_patterns,
            etag_timeout=15,
        )
    except Exception as exc:
        raise TrainingStateIdentityError(
            f'could not materialize immutable model snapshot '
            f'{repo_id}@{commit}') from exc
    local = os.path.abspath(os.fspath(local))
    run_environment.pinned_hf_artifact_signature(
        local, repo_id, commit, strict=True)
    return local, commit


def _materialize_hf_file(
        repo_id: str,
        filename: str,
        *,
        commit: str,
        cache_dir,
        token=None,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise TrainingStateIdentityError(
            'huggingface_hub is unavailable for immutable adapter pinning') from exc
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=commit,
            cache_dir=str(cache_dir),
            token=_hf_token_value(token),
            etag_timeout=15,
        )
    except Exception as exc:
        raise TrainingStateIdentityError(
            f'could not materialize immutable model file '
            f'{repo_id}@{commit}/{filename}') from exc
    local = os.path.abspath(os.fspath(local))
    run_environment.pinned_hf_artifact_signature(
        local, repo_id, commit, strict=True)
    return local


def _krea_checkpoint_filename(repo_id: str) -> str:
    return repo_id.rsplit('/', 1)[-1].rsplit('-', 1)[-1].lower() + '.safetensors'


def _sanitize_pin(pin: dict) -> dict:
    return {
        key: pin[key]
        for key in ('field', 'repo', 'commit', 'filename')
        if pin.get(key) is not None
    }


def _ensure_krea_implicit_dependencies(model: dict) -> None:
    if str(model.get('arch') or '').lower() != 'krea2':
        return
    kwargs = model.get('model_kwargs')
    if not isinstance(kwargs, dict):
        kwargs = {}
        model['model_kwargs'] = kwargs
    kwargs.setdefault('text_encoder_path', _KREA_TEXT_ENCODER)
    kwargs.setdefault('vae_path', _KREA_VAE)


def pin_job_model_artifacts(
        job_config: dict,
        *,
        cache_dir,
        token=None,
        pins: list[dict] | None = None,
) -> list[dict]:
    """Rewrite every hosted model input to immutable local bytes.

    With ``pins=None`` mutable repo ids are resolved and materialized. Passing a
    previously returned pin list applies exactly those already-resolved paths,
    which closes the resolve/verify/spawn race during full-state continuation.
    """
    model = _model_process(job_config)
    arch = str(model.get('arch') or '').strip().lower()
    hidden_reason = _UNPINNABLE_HIDDEN_DEPENDENCIES.get(arch)
    if hidden_reason:
        raise TrainingStateIdentityError(
            f'exact-state model pinning is unavailable: {hidden_reason}')
    _ensure_krea_implicit_dependencies(model)
    if pins is not None:
        applied = []
        by_field = {str(pin.get('field')): pin for pin in pins}
        for path in _MODEL_PATH_FIELDS:
            field = _field_name(path)
            pin = by_field.get(field)
            if pin is None:
                continue
            original = _nested_get(model, path)
            if original != pin.get('original'):
                raise TrainingStateIdentityError(
                    f'model pin no longer matches {field}')
            commit = str(pin.get('commit') or '')
            local_path = str(pin.get('local_path') or '')
            if not _HF_COMMIT_RE.fullmatch(commit) or not os.path.exists(local_path):
                raise TrainingStateIdentityError(
                    f'immutable model pin is unavailable for {field}')
            run_environment.pinned_hf_artifact_signature(
                pin.get('artifact_path') or local_path,
                pin.get('repo'), commit, strict=True)
            _nested_set(model, path, local_path)
            if pin.get('checkpoint_filename'):
                _nested_set(
                    model, ('model_kwargs', 'checkpoint_filename'),
                    pin['checkpoint_filename'])
            applied.append(dict(pin))
        if len(applied) != len(pins):
            raise TrainingStateIdentityError(
                'model pin set does not match the current job config')
        job_config['_lds_model_pins'] = [
            _sanitize_pin(pin) for pin in applied]
        return applied

    resolved = []
    for path in _MODEL_PATH_FIELDS:
        raw = _nested_get(model, path)
        if not isinstance(raw, str) or not raw.strip():
            continue
        value = raw.strip()
        field = _field_name(path)
        repo = None
        filename = None
        allow_patterns = None
        rewritten_to_file = False
        if path == ('assistant_lora_path',):
            parts = value.replace('\\', '/').split('/')
            if not os.path.exists(value) and len(parts) >= 3:
                candidate = '/'.join(parts[:2])
                if _is_hf_repo(candidate):
                    repo = candidate
                    filename = '/'.join(parts[2:])
                    allow_patterns = [filename]
                    rewritten_to_file = True
        elif _is_hf_repo(value) and not os.path.exists(value):
            repo = value
            if (
                path == ('name_or_path',)
                and str(model.get('arch') or '').lower() == 'krea2'
            ):
                filename = _krea_checkpoint_filename(repo)
                allow_patterns = [filename]
        if repo is None:
            # The child runs with ai-toolkit as its cwd. Preserve the exact
            # parent-hashed load target by making every already-local model
            # input lexical and absolute before the job JSON is written.
            if os.path.exists(value):
                _nested_set(model, path, os.path.abspath(os.fspath(value)))
            continue
        if rewritten_to_file:
            commit = _resolve_hf_commit(repo, token=token)
            local_path = _materialize_hf_file(
                repo,
                filename,
                commit=commit,
                cache_dir=cache_dir,
                token=token,
            )
            local_snapshot = os.path.dirname(local_path)
        else:
            local_snapshot, commit = _materialize_hf_snapshot(
                repo,
                cache_dir=cache_dir,
                token=token,
                allow_patterns=allow_patterns,
            )
            local_path = local_snapshot
        if not os.path.exists(local_path):
            raise TrainingStateIdentityError(
                f'pinned model artifact is missing for {field}')
        pin = {
            'field': field,
            'repo': repo,
            'commit': commit,
            'filename': filename,
            'original': value,
            'local_path': local_path,
            'artifact_path': local_path if rewritten_to_file else local_snapshot,
        }
        if (
            path == ('name_or_path',)
            and str(model.get('arch') or '').lower() == 'krea2'
        ):
            pin['checkpoint_filename'] = filename
            _nested_set(
                model, ('model_kwargs', 'checkpoint_filename'), filename)
        _nested_set(model, path, local_path)
        resolved.append(pin)
    job_config['_lds_model_pins'] = [
        _sanitize_pin(pin) for pin in resolved]
    return resolved


def _artifact_identities(job_config: dict) -> dict:
    model = _model_process(job_config)
    artifacts = {}
    pins = {
        str(pin.get('field')): pin
        for pin in (job_config.get('_lds_model_pins') or ())
        if isinstance(pin, dict)
    }
    for path in _MODEL_PATH_FIELDS:
        value = _nested_get(model, path)
        if not isinstance(value, str) or not value.strip():
            continue
        field = _field_name(path)
        if not os.path.isabs(value):
            raise TrainingStateIdentityError(
                f'{field} is not pinned to an absolute local artifact')
        if not os.path.exists(value):
            raise TrainingStateIdentityError(
                f'{field} is not pinned to a readable local artifact')
        try:
            pin = pins.get(field)
            if pin is not None:
                signature = run_environment.pinned_hf_artifact_signature(
                    value, pin.get('repo'), pin.get('commit'), strict=True)
            else:
                signature = run_environment.artifact_signature(
                    value, strict=True)
        except (OSError, ValueError) as exc:
            raise TrainingStateIdentityError(
                f'{field} could not be hashed byte-for-byte') from exc
        artifacts[field] = signature
    if not artifacts:
        raise TrainingStateIdentityError(
            'job config has no hashable local model artifacts')
    return artifacts


def _model_artifact_contracts(
        job_config: dict, artifacts: dict[str, dict]) -> list[dict]:
    """Describe the exact local bytes the child must re-hash before loaders run."""
    model = _model_process(job_config)
    pins = {
        str(pin.get('field')): pin
        for pin in (job_config.get('_lds_model_pins') or ())
        if isinstance(pin, dict)
    }
    contracts = []
    for path in _MODEL_PATH_FIELDS:
        field = _field_name(path)
        signature = artifacts.get(field)
        value = _nested_get(model, path)
        if signature is None or not isinstance(value, str) or not value.strip():
            continue
        contract = {
            'field': field,
            'lexical_path': os.path.abspath(os.fspath(value)),
            'kind': signature['kind'],
            'size': int(signature['size']),
            'sha256': signature['sha256'],
        }
        if signature['kind'] == 'directory':
            contract['files'] = int(signature['files'])
        pin = pins.get(field)
        if pin is not None:
            contract.update({
                'repo': str(pin.get('repo') or ''),
                'commit': str(pin.get('commit') or '').lower(),
            })
        contracts.append(contract)
    if len(contracts) != len(artifacts):
        raise TrainingStateIdentityError(
            'model artifact contracts do not match the hashed model inputs')
    return sorted(contracts, key=lambda item: item['field'])


def _normalized_package_versions(value) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError('package versions must be a JSON object')
    grouped: dict[str, set[str]] = {}
    for raw_name, raw_version in value.items():
        if not isinstance(raw_name, str) or not isinstance(raw_version, str):
            raise ValueError('package name/version must be strings')
        name = re.sub(r'[-_.]+', '-', raw_name).lower()
        if not name or not raw_version:
            raise ValueError('package name/version cannot be empty')
        grouped.setdefault(name, set()).update(raw_version.split('|'))
    return {
        name: '|'.join(sorted(versions))
        for name, versions in sorted(grouped.items())
    }


def _runtime_fingerprint(python_path) -> dict:
    """Probe the interpreter that actually trains.

    This deliberately runs for every identity. Keying a cache only on
    ``python.exe`` is unsafe: pip can replace Torch/CUDA packages without
    touching the interpreter binary, which would let an incompatible bundle
    retain a stale "exact" label.
    """
    python = Path(python_path).expanduser().resolve()
    try:
        python.stat()
    except OSError as exc:
        raise TrainingStateIdentityError(
            'ai-toolkit interpreter is unavailable for exact-state capture') from exc
    try:
        proc = subprocess.run(
            [str(python), '-c', _RUNTIME_PROBE],
            capture_output=True,
            text=True,
            timeout=90,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrainingStateIdentityError(
            'could not probe the ai-toolkit runtime for exact-state capture') from exc
    if proc.returncode != 0:
        raise TrainingStateIdentityError(
            'the ai-toolkit runtime could not report its Torch/CUDA identity')
    try:
        value = json.loads((proc.stdout or '').strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise TrainingStateIdentityError(
            'the ai-toolkit runtime returned an invalid identity') from exc
    packages = value.get('packages') if isinstance(value, dict) else None
    required_packages = {
        'numpy', 'accelerate', 'diffusers', 'transformers',
        'safetensors', 'bitsandbytes',
    }
    try:
        normalized_packages = _normalized_package_versions(packages)
    except ValueError:
        normalized_packages = {}
    if (
        not isinstance(value, dict)
        or not value.get('python')
        or not value.get('torch')
        or not isinstance(value.get('cuda_devices'), int)
        or not required_packages.issubset(normalized_packages)
        or not isinstance(value.get('gpus'), list)
        or len(value['gpus']) != value['cuda_devices']
        or not isinstance(value.get('gpu_driver'), str)
        or not value['gpu_driver']
    ):
        raise TrainingStateIdentityError(
            'the ai-toolkit runtime identity is incomplete')
    value['packages'] = normalized_packages
    runtime = {
        **value,
        'protocol': BRIDGE_PROTOCOL,
        'shape_revision': SHAPE_REVISION,
    }
    return runtime


def _normalized_job(job_config: dict) -> tuple[dict, dict, dict]:
    """Return compatibility config, network block and model block.

    Target length and presentation text may legitimately change on a
    continuation. Dataset/run paths are machine locations, not identities.
    Save/sample cadence remains hashed because ai-toolkit's alternating
    main/regularisation loader selection depends on whether the current step is
    a save or sample boundary. Everything that shapes optimizer/model/dataset
    semantics therefore remains hashed.
    """
    if not isinstance(job_config, dict):
        raise TrainingStateIdentityError('job config is not a JSON object')
    normalized = copy.deepcopy(job_config)
    # Hosted commit/digest facts belong to base_hash. Keeping them in the
    # generic config hash would misreport a model revision mismatch as a
    # training-setting mismatch.
    normalized.pop('_lds_model_pins', None)
    try:
        process = normalized['config']['process'][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise TrainingStateIdentityError(
            'job config has no supported training process') from exc
    if not isinstance(process, dict):
        raise TrainingStateIdentityError(
            'job config training process is invalid')
    network = copy.deepcopy(process.get('network'))
    if not isinstance(network, dict) or not isinstance(process.get('model'), dict):
        raise TrainingStateIdentityError(
            'job config has no model/network identity')

    process.pop('training_folder', None)
    train = process.get('train')
    if isinstance(train, dict):
        train.pop('steps', None)
    save = process.get('save')
    if isinstance(save, dict):
        save.pop('max_step_saves_to_keep', None)
    sample = process.get('sample')
    if isinstance(sample, dict):
        # Prompt wording affects only generated previews. Keep sample_every and
        # all other scheduling fields: changing them may change which loader is
        # consumed at an optimizer boundary.
        sample.pop('prompts', None)
    normalized_model = process['model']
    for path in _MODEL_PATH_FIELDS:
        value = _nested_get(normalized_model, path)
        if isinstance(value, str) and os.path.exists(value):
            _nested_set(normalized_model, path, f'<{_field_name(path)}>')
    for dataset in process.get('datasets') or ():
        if not isinstance(dataset, dict):
            continue
        # The launch adds an explicit `_lds_state.masked` fact.  The derived
        # mask folder and its metadata may not exist yet while a continuation is
        # being validated, so their filesystem presence is not an identity.
        dataset.pop('mask_path', None)
        dataset.pop('mask_min_value', None)
        for key in tuple(dataset):
            if key == 'folder_path' or key.endswith('_path'):
                dataset[key] = f'<{key}>'
    model = copy.deepcopy(normalized_model)
    return normalized, network, model


def build_identity(
    *,
    job_config: dict,
    prepared: dict,
    toolkit_probe: dict,
    python_path,
) -> dict:
    """Build the immutable bridge identity for one local launch."""
    if not isinstance(toolkit_probe, dict) or not toolkit_probe.get('supported'):
        raise TrainingStateIdentityError(
            'the installed ai-toolkit lifecycle is not supported by the state bridge')
    revision = toolkit_probe.get('aitoolkit_revision')
    if not isinstance(revision, str) or not revision:
        raise TrainingStateIdentityError(
            'ai-toolkit has no immutable revision for exact-state compatibility')
    if not isinstance(prepared, dict):
        raise TrainingStateIdentityError(
            'the dataset snapshot could not be frozen for exact-state capture')

    artifact_identities = _artifact_identities(job_config)
    artifact_contracts = _model_artifact_contracts(
        job_config, artifact_identities)
    normalized, network, model = _normalized_job(job_config)
    snapshot = prepared.get('snapshot')
    dataset_snapshot = {}
    if isinstance(snapshot, dict):
        dataset_snapshot = {
            key: snapshot[key]
            for key in ('captions', 'images', 'dataset')
            if key in snapshot
        }
    dataset_value = {
        'manifest': prepared.get('manifest') or [],
        'snapshot': dataset_snapshot,
    }
    base_file = None
    if isinstance(snapshot, dict):
        env = snapshot.get('env')
        if isinstance(env, dict):
            raw_base_file = env.get('base_file')
            if isinstance(raw_base_file, dict):
                base_file = {
                    key: value for key, value in raw_base_file.items()
                    if key != 'folder'
                }
    pins = job_config.get('_lds_model_pins')
    if not isinstance(pins, list):
        pins = []

    return {
        'schema': IDENTITY_SCHEMA,
        'config_hash': training_state_bundle.sha256_json(normalized),
        'dataset_hash': training_state_bundle.sha256_json(dataset_value),
        'base_hash': training_state_bundle.sha256_json({
            'model': model,
            'artifacts': artifact_identities,
            'hosted_pins': pins,
            'base_file': base_file,
        }),
        'network_hash': training_state_bundle.sha256_json(network),
        'toolkit_revision': revision,
        'runtime': _runtime_fingerprint(python_path),
        # The compatibility digest above names the expected bytes. This
        # path-bearing copy is intentionally outside the portable hash so the
        # child can re-open and re-hash those exact lexical load targets after
        # Popen, immediately before ai-toolkit imports its model loaders.
        'model_artifacts': artifact_contracts,
    }


def compatibility_spec(identity: dict) -> training_state_bundle.CompatibilitySpec:
    return training_state_bundle.CompatibilitySpec(
        config_hash=identity['config_hash'],
        dataset_hash=identity['dataset_hash'],
        base_model_hash=identity['base_hash'],
        network_hash=identity['network_hash'],
        toolkit_revision=identity['toolkit_revision'],
        toolkit_runtime=identity['runtime'],
        required_capabilities=EXACT_CAPABILITIES,
        minimum_state_level='exact',
    )


def write_identity(path, identity: dict) -> Path:
    """Atomically publish a bridge identity without exposing a partial JSON file."""
    target = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    atomic_json_nofollow(target, identity)
    return target
